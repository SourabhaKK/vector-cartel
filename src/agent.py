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
from typing import Dict, List

import src.prompts as prompts
from src.schemas import AgentState, Citation
from src.contracts import ChunkDict, get_chunk_doc, get_chunk_section
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
    Checks whether the generated answer is grounded in retrieved
    context, using token overlap rather than a second LLM call
    (see module docstring for rationale).

    Reads: state["answer"], state["retrieved_chunks"],
           state["retry_count"]
    Returns: {"validation_passed": bool} and, on failure,
             {"retry_count": int} incremented by 1
    LLM calls: 0
    Prompt calls: none

    Skips the overlap check entirely for refusal answers (the
    refusal string is never "ungrounded" — there's nothing to
    ground) and once retry_count has hit MAX_NODE_RETRIES, to
    avoid an infinite retry loop.

    Called by: synthesize_answer's downstream edge. Routes via
    route_by_validation to either END or back to synthesize_answer.
    """
    if state["answer"].strip() == REFUSAL_STRING:
        return {"validation_passed": True}

    if state["retry_count"] >= 1:
        return {"validation_passed": True}

    overlap = _token_overlap(state["answer"], state["retrieved_chunks"])

    if overlap >= 0.25:
        return {"validation_passed": True}
    else:
        return {
            "validation_passed": False,
            "retry_count": state["retry_count"] + 1,
        }


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
    Decides whether to retry synthesize_answer or finish.

    Reads: state["validation_passed"], state["retry_count"]
    Returns: "retry" if validation failed and under MAX_NODE_RETRIES,
             otherwise "passed"
    LLM calls: 0
    Prompt calls: none

    Called by: LangGraph conditional edge after validate_citations.
    """
    if state["validation_passed"]:
        return "passed"
    if state["retry_count"] >= 1:
        return "passed"
    return "retry"
