"""
agent.py — LangGraph node functions for SecureOps Assistant.

Six nodes, two route functions. Each node is a pure-ish function:
takes AgentState + LLMRouter, returns a partial dict that LangGraph
merges into state. Nodes never mutate state directly — this keeps
them independently testable without a running graph.

NODE EXECUTION ORDER (see build_agent_graph for full wiring):
  classify_query → routes to one of:
    simple        → retrieve (rag-layer) → synthesize_answer
    multi_hop     → decompose_query → retrieve → synthesize_answer
    out_of_scope  → handle_refusal → END
    ambiguous     → handle_clarification → END
  synthesize_answer → validate_citations → routes to:
    passed → END
    retry  → synthesize_answer (max 1 retry, see MAX_NODE_RETRIES)

WHY VALIDATE_CITATIONS USES TOKEN OVERLAP, NOT A SECOND LLM CALL:
A second LLM call to judge groundedness would be slower, cost
another 15-RPM-budget call, and introduce its own hallucination
risk (the judge model could be wrong too). Token overlap is a
cheap, deterministic, explainable proxy — good enough to catch
the worst hallucinations without adding latency or LLM cost.

IMPORT PATTERN — WHY "import src.prompts as prompts" IS USED
INSTEAD OF IMPORTING EACH FUNCTION NAME DIRECTLY:
This file is unit-tested with mocker.patch("src.prompts.build_x").
patch() only replaces the attribute on the module object. A name
bound via a direct named import is resolved once at import time
and is invisible to later patching. Module-qualified calls
(prompts.build_x(...)) resolve the lookup at call time, so the
patched mock is correctly intercepted. Any new cross-module
function call added to this file that tests may want to spy on
must follow this same pattern — see test_agent.py for the two
existing spy tests this protects.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

import src.prompts as prompts
from src.schemas import AgentState, Citation
from src.contracts import (
    ChunkDict,
    MAX_NODE_RETRIES,
    get_chunk_doc,
    get_chunk_section,
)
from src.llm import LLMRouter, JSONParseError

logger = logging.getLogger(__name__)

REFUSAL_STRING = (
    "I don't have enough information in the corpus to answer "
    "this question."
)

CITATION_PATTERN = re.compile(r"\[Source:\s*([^|]+?)\s*\|\s*([^\]]+?)\]")


def classify_query(state: AgentState, llm: LLMRouter) -> Dict:
    """
    Classifies the query into one of 4 types to determine routing.

    Reads: state["query"]
    Returns: {"query_type": str}
    LLM calls: 1 (llm.generate_json)
    Prompt calls: prompts.build_classification_prompt (module-qualified
                  — spied on by test_classify_query_passes_query_to_prompt_builder)

    Defaults to "simple" on JSONParseError rather than propagating
    the exception — a misclassified query degrades gracefully to
    single-shot RAG rather than crashing the pipeline.

    Called by: entry point of build_agent_graph, routes via
    route_by_query_type to one of 4 paths.
    """
    prompt = prompts.build_classification_prompt(state["query"])
    try:
        result = llm.generate_json(prompt)
        return {"query_type": result.get("query_type", "simple")}
    except JSONParseError:
        logger.warning("classify_query: JSONParseError, defaulting to simple")
        return {"query_type": "simple"}


def decompose_query(state: AgentState, llm: LLMRouter) -> Dict:
    """
    Breaks a multi-hop query into up to 3 sub-queries, one per
    knowledge source (NIST, CISA, MITRE ATT&CK).

    Reads: state["query"]
    Returns: {"sub_queries": List[str]}
    LLM calls: 1 (llm.generate_json)
    Prompt calls: prompts.build_decomposition_prompt (module-qualified)

    Falls back to [state["query"]] (treat as a single sub-query)
    if the LLM returns an empty list or raises JSONParseError —
    degrades to single-shot retrieval rather than failing the
    multi_hop path entirely.

    Called by: route_by_query_type when query_type == "multi_hop".
    """
    prompt = prompts.build_decomposition_prompt(state["query"])
    try:
        result = llm.generate_json(prompt)
        sub_queries = result.get("sub_queries", [])
        if not sub_queries:
            return {"sub_queries": [state["query"]]}
        return {"sub_queries": sub_queries[:3]}
    except JSONParseError:
        logger.warning("decompose_query: JSONParseError, using original query")
        return {"sub_queries": [state["query"]]}


def handle_refusal(state: AgentState) -> Dict:
    """
    Returns the standard refusal response for out-of-scope queries.

    Reads: nothing from state — the refusal message is fixed.
    Returns: {"refusal": bool, "confidence": float, "answer": str}
    LLM calls: 0
    Prompt calls: none

    Called by: route_by_query_type when query_type == "out_of_scope".
    """
    return {
        "refusal": True,
        "confidence": 0.0,
        "answer": REFUSAL_STRING,
    }


def handle_clarification(state: AgentState, llm: LLMRouter) -> Dict:
    """
    Generates a single clarifying question for an ambiguous query.

    Reads: state["query"]
    Returns: {"needs_clarification": bool, "clarification_question": str}
    LLM calls: 1 (llm.generate)
    Prompt calls: none (clarification prompt is built inline — it
                  is not part of src.prompts since it is not shared
                  with or tested via the prompts module)

    Called by: route_by_query_type when query_type == "ambiguous".
    """
    clarification_prompt = (
        f"The following query is ambiguous: '{state['query']}'. "
        f"Generate a single clarifying question to ask the user."
    )
    question = llm.generate(clarification_prompt, "")
    return {
        "needs_clarification": True,
        "clarification_question": question,
    }


def synthesize_answer(state: AgentState, llm: LLMRouter) -> Dict:
    """
    Generates the grounded answer from retrieved context.

    Reads: state["query"], state["retrieved_chunks"]
    Returns: {"answer": str, "citations": List[Citation],
             "confidence": float}
    LLM calls: 1 (llm.generate)
    Prompt calls: prompts.build_system_prompt (module-qualified
                  — spied on by
                  test_synthesize_answer_calls_build_system_prompt_with_chunks)

    Called after: retrieve (rag-layer) on simple/multi_hop path
    Called by: validate_citations on retry (max 1 retry, see
               MAX_NODE_RETRIES in src.contracts)
    """
    chunks = state["retrieved_chunks"]
    system_prompt = prompts.build_system_prompt(chunks)
    answer_text = llm.generate(system_prompt, state["query"])

    citations = _parse_citations(answer_text)
    confidence = _max_chunk_score(chunks)

    return {
        "answer": answer_text,
        "citations": citations,
        "confidence": confidence,
    }


def _max_chunk_score(chunks: List[ChunkDict]) -> float:
    """
    Returns the highest relevance score among retrieved chunks.
    Returns 0.0 if chunks is empty (no retrieval evidence).
    """
    return max((c["score"] for c in chunks), default=0.0)


def _parse_citations(answer_text: str) -> List[Citation]:
    """
    Extracts all [Source: doc | section] citations from LLM output
    using CITATION_PATTERN. Returns empty list if no matches found
    (expected for refusal answers).

    Args:
        answer_text: Raw LLM output containing inline citations.

    Returns:
        List of Citation objects, snippet field left empty
        (snippet is populated later if needed by output-layer).
    """
    citations = []
    for match in CITATION_PATTERN.finditer(answer_text):
        doc, section = match.group(1).strip(), match.group(2).strip()
        citations.append(Citation(doc=doc, section=section, snippet=""))
    return citations


def validate_citations(state: AgentState) -> Dict:
    """
    Checks whether the answer is grounded in retrieved chunks via
    token overlap. Does NOT manage retry_count — that is owned by
    route_by_validation, which decides whether to retry and
    increment_retry_count, which increments retry_count only when
    actually routing back to synthesize_answer. This keeps the
    "did validation pass" decision (here) separate from the "how
    many retries have happened" lifecycle (there), avoiding the
    ordering bug where both functions independently touched
    retry_count.

    Reads: state["answer"], state["retrieved_chunks"]
    Returns: {"validation_passed": bool}
    LLM calls: 0
    Prompt calls: none

    Skips the overlap check entirely for refusal answers — the
    refusal string is never "ungrounded", there's nothing to ground.

    Called by: synthesize_answer's downstream edge. Routes via
    route_by_validation to either END or increment_retry_count.
    """
    if state["answer"].strip() == REFUSAL_STRING:
        return {"validation_passed": True}

    overlap = _token_overlap(state["answer"], state["retrieved_chunks"])
    return {"validation_passed": overlap >= 0.25}


def increment_retry_count(state: AgentState) -> Dict:
    """
    Increments retry_count by 1. Called only on the retry path,
    between validate_citations failing and synthesize_answer
    being called again. This is the single place retry_count
    changes — owning the increment here (not in validate_citations)
    means route_by_validation always reads the count from BEFORE
    the current attempt, avoiding the off-by-one where a node's
    own increment caused the router to immediately treat the
    first failure as already-at-limit.

    Reads: state["retry_count"]
    Returns: {"retry_count": int} incremented by 1
    LLM calls: 0
    Prompt calls: none

    Called by: route_by_validation's "retry" edge, before
    synthesize_answer runs again.
    """
    return {"retry_count": state["retry_count"] + 1}


def _token_overlap(answer: str, chunks: List[ChunkDict]) -> float:
    """
    Computes the fraction of answer tokens that also appear in
    the combined retrieved chunk text. Used by validate_citations
    to detect ungrounded/hallucinated claims.

    Returns 0.0 if answer has no tokens (empty string edge case).

    Args:
        answer: The generated answer text to check.
        chunks: The retrieved chunks the answer should be grounded in.

    Returns:
        Float between 0.0 and 1.0 — fraction of answer tokens
        found in the combined chunk text.
    """
    chunk_text_combined = " ".join(c["text"].lower() for c in chunks)
    chunk_tokens = set(chunk_text_combined.split())

    answer_tokens = set(answer.lower().split())
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & chunk_tokens) / len(answer_tokens)


def route_by_query_type(state: AgentState) -> str:
    """
    Dispatches to one of 4 paths based on classify_query's output.

    Reads: state["query_type"]
    Returns: one of "simple", "multi_hop", "out_of_scope", "ambiguous"
    LLM calls: 0
    Prompt calls: none

    Called by: LangGraph conditional edge after classify_query.
    """
    return state["query_type"]


def route_by_validation(state: AgentState) -> str:
    """
    Routes based on validation_passed and current retry_count.
    retry_count here reflects retries already completed BEFORE
    this validation check — it is incremented by
    increment_retry_count only on the retry path, after this
    function has made its decision. This ordering means the
    first failure sees retry_count=0 (not yet incremented) and
    correctly routes to retry.

    Reads: state["validation_passed"], state["retry_count"]
    Returns: "retry" if validation failed and under MAX_NODE_RETRIES,
             otherwise "passed"
    LLM calls: 0
    Prompt calls: none

    Called by: LangGraph conditional edge after validate_citations.
    """
    if state["validation_passed"]:
        return "passed"
    if state["retry_count"] >= MAX_NODE_RETRIES:
        return "passed"
    return "retry"


def _stub_retrieve(state: AgentState) -> Dict:
    """
    TEMPORARY STUB — replaced when rag-layer branch merges.

    In tests, the real retrieval_fn is injected via run_agent's
    retrieval_fn parameter and bound into the graph at build time.
    This stub only fires if no retrieval_fn was injected — it
    should never be reached in correctly-wired tests or production.

    Real implementation (rag-layer, src/retrieval.py) does:
      1. Embed query with BAAI/bge-small-en-v1.5
      2. ChromaDB cosine similarity search -> top 20
      3. BM25 keyword search -> top 20
      4. RRF fusion -> merged ranked list
      5. Cross-encoder reranker -> top 5
      6. Validate each chunk with validate_chunk() from src.contracts
      7. Return List[ChunkDict] ordered by score descending
    """
    return {"retrieved_chunks": []}


def _stub_verify(state: AgentState) -> Dict:
    """
    TEMPORARY STUB — replaced when rag-layer branch merges.
    Always reports sufficient. Real implementation checks
    RETRIEVAL_CONFIDENCE_THRESHOLD from src.contracts against
    the top chunk score.
    """
    return {"validation_passed": True}


def _stub_rewrite(state: AgentState) -> Dict:
    """
    TEMPORARY STUB — replaced when rag-layer branch merges.
    Passthrough — returns the same sub_queries unchanged.
    """
    return {}


def _retrieve_for_all_sub_queries(
    state: AgentState, retrieval_fn: Callable
) -> List[ChunkDict]:
    """
    Calls retrieval_fn once per sub-query and flattens results.

    CALL COUNT CONTRACT (tested explicitly in test_agent.py):
      simple path:    sub_queries == [original_query] (set by
                       run_agent's initial_state) -> 1 call
      multi_hop path: sub_queries == 2-3 items from decompose_query
                       -> 2-3 calls

    This function is the single place retrieval_fn gets called
    in the graph — both simple and multi_hop route through the
    same "retrieve" node, differentiated only by how many items
    are in state["sub_queries"] when this function runs.
    """
    sub_queries = state.get("sub_queries") or [state["query"]]
    all_chunks: List[ChunkDict] = []
    for sq in sub_queries:
        all_chunks.extend(retrieval_fn(sq))
    return all_chunks


CLASSIFY_ROUTING_MAP = {
    "simple": "retrieve",
    "multi_hop": "decompose",
    "out_of_scope": "refuse",
    "ambiguous": "clarify",
}

VERIFY_ROUTING_MAP = {
    "sufficient": "synthesize",
    "insufficient": "rewrite",
}

VALIDATE_ROUTING_MAP = {
    "passed": END,
    "retry": "increment_retry",
}


def _make_classify_node(llm_router: LLMRouter) -> Callable:
    def node(state: AgentState) -> Dict:
        return classify_query(state, llm_router)

    return node


def _make_decompose_node(llm_router: LLMRouter) -> Callable:
    def node(state: AgentState) -> Dict:
        return decompose_query(state, llm_router)

    return node


def _make_clarify_node(llm_router: LLMRouter) -> Callable:
    def node(state: AgentState) -> Dict:
        return handle_clarification(state, llm_router)

    return node


def _make_synthesize_node(llm_router: LLMRouter) -> Callable:
    def node(state: AgentState) -> Dict:
        return synthesize_answer(state, llm_router)

    return node


def _make_retrieve_node(retrieval_fn: Callable) -> Callable:
    def node(state: AgentState) -> Dict:
        return {"retrieved_chunks": _retrieve_for_all_sub_queries(state, retrieval_fn)}

    return node


def build_agent_graph(
    llm_router: LLMRouter, retrieval_fn: Callable = None
) -> CompiledStateGraph:
    """
    Builds and compiles the SecureOps Assistant LangGraph.

    TOPOLOGY:
      classify --[route_by_query_type]-->
        "simple"        -> retrieve -> verify -> synthesize -> validate
        "multi_hop"     -> decompose -> retrieve -> verify ->
                            synthesize -> validate
        "out_of_scope"  -> refuse -> END
        "ambiguous"     -> clarify -> END

      verify --[stub, always "sufficient" for now]-->
        "sufficient"    -> synthesize
        "insufficient"  -> rewrite -> retrieve (loop back)
        NOTE: "insufficient" path is currently unreachable —
        _stub_verify always returns validation_passed=True.
        This activates once rag-layer replaces _stub_verify
        with real RETRIEVAL_CONFIDENCE_THRESHOLD logic.

      validate --[route_by_validation]-->
        "passed" -> END
        "retry"  -> increment_retry_count -> synthesize (loop back,
                    max MAX_NODE_RETRIES times, see src.contracts)

    RAG-LAYER INTEGRATION POINT:
      If retrieval_fn is None, falls back to _stub_retrieve
      (always returns empty chunks) — used only by tests that
      don't exercise real retrieval behaviour. In production,
      rag-layer's retrieval_fn is injected here when dev merges.

    Args:
        llm_router: The LLMRouter instance all LLM-calling nodes
                    will use (classify, decompose, clarify, synthesize).
        retrieval_fn: Optional. Matches the RetrievalFn protocol
                     from src.contracts. Falls back to stub if None.

    Returns:
        A compiled LangGraph CompiledStateGraph with .invoke(state).
    """
    graph = StateGraph(AgentState)

    graph.add_node("classify", _make_classify_node(llm_router))
    graph.add_node("decompose", _make_decompose_node(llm_router))
    graph.add_node("refuse", handle_refusal)
    graph.add_node("clarify", _make_clarify_node(llm_router))

    if retrieval_fn:
        graph.add_node("retrieve", _make_retrieve_node(retrieval_fn))
    else:
        graph.add_node("retrieve", _stub_retrieve)

    graph.add_node("verify", _stub_verify)
    graph.add_node("rewrite", _stub_rewrite)
    graph.add_node("synthesize", _make_synthesize_node(llm_router))
    graph.add_node("validate", validate_citations)
    graph.add_node("increment_retry", increment_retry_count)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify", route_by_query_type, CLASSIFY_ROUTING_MAP
    )

    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "verify")

    graph.add_conditional_edges(
        "verify", lambda s: "sufficient", VERIFY_ROUTING_MAP
    )

    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("synthesize", "validate")

    graph.add_conditional_edges(
        "validate", route_by_validation, VALIDATE_ROUTING_MAP
    )
    graph.add_edge("increment_retry", "synthesize")

    graph.add_edge("refuse", END)
    graph.add_edge("clarify", END)

    return graph.compile()


def run_agent(query: str, llm_router: LLMRouter, retrieval_fn: Callable):
    """
    Runs the full SecureOps Assistant agent graph for a single query.

    Initialises AgentState with sub_queries=[query] so that the
    simple path calls retrieval_fn exactly once via
    _retrieve_for_all_sub_queries (see that function's docstring
    for the full call-count contract).

    FIELD MAPPING NOTE — clarification fallback:
    AgentState has needs_clarification/clarification_question
    fields, but SecureOpsAnswer has no equivalent field (by
    design — SecureOpsAnswer is the external-facing schema,
    AgentState is internal). When the ambiguous path is taken,
    this function falls back to clarification_question as the
    answer text, since SecureOpsAnswer.answer is the only field
    that reaches the caller. Gradio (src/gradio_demo.py, not
    yet built) will be responsible for displaying this
    distinctly from a normal answer if needed.

    Returns:
        SecureOpsAnswer with sources_used computed from citations.
    """
    from src.schemas import AGENTSTATE_DEFAULTS, SecureOpsAnswer

    initial_state = dict(AGENTSTATE_DEFAULTS)
    initial_state["query"] = query
    initial_state["sub_queries"] = [query]

    graph = build_agent_graph(llm_router, retrieval_fn)
    final_state = graph.invoke(initial_state)

    return SecureOpsAnswer(
        answer=final_state.get("answer")
        or final_state.get("clarification_question")
        or "",
        citations=final_state.get("citations", []),
        confidence=final_state.get("confidence", 0.0),
        refusal=final_state.get("refusal", False),
        query_type=final_state.get("query_type", "simple"),
        sources_used=[c.doc for c in final_state.get("citations", [])],
    )
