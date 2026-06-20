"""
schemas.py — Core data models for SecureOps Assistant pipeline.

DESIGN DECISIONS DOCUMENTED HERE:

AgentState is a TypedDict, NOT a Pydantic BaseModel.
Reason: LangGraph StateGraph requires dict-compatible state.
Nodes return partial dicts that LangGraph merges into state.
A Pydantic BaseModel would break this merge behaviour.
At runtime AgentState is a plain Python dict — bracket access
only (state["query"]), never dot access (state.query).

Citation and SecureOpsAnswer are Pydantic BaseModel.
Reason: These are output objects, not LangGraph state.
They benefit from Pydantic validation (confidence bounds,
snippet length) and are returned to the Gradio interface.

ChunkDict is imported from src.contracts, not defined here.
Reason: ChunkDict is the contract between rag-layer and this
layer. It must be defined in one place only.
"""

from typing import List, Optional, TypedDict

from pydantic import BaseModel, Field, computed_field, field_validator

from src.contracts import ChunkDict, InputScanResult


class Citation(BaseModel):
    doc: str = Field(
        description="Document name matching ChunkDict metadata "
        "'doc' field. e.g. 'NIST SP 800-82 Rev 3', "
        "'CISA Advisory ICSA-24-001', 'MITRE ATT&CK ICS'"
    )
    section: str = Field(
        description="Section or identifier within the document. "
        "e.g. '5.2.3' for NIST, 'T0836' for MITRE, "
        "'PR.AC-1' for CSF"
    )
    page: Optional[int] = Field(
        default=None,
        description="Page number. None for CISA advisories and "
        "MITRE techniques which have no page numbers.",
    )
    snippet: str = Field(
        description="Short excerpt from the source chunk. "
        "Max 200 characters. Used in citations block "
        "of the Gradio interface output."
    )

    @field_validator("snippet")
    @classmethod
    def snippet_must_not_exceed_200_chars(cls, value: str) -> str:
        if len(value) > 200:
            raise ValueError("snippet must not exceed 200 characters")
        return value


class SecureOpsAnswer(BaseModel):
    answer: str = Field(
        description="Full answer text with inline citations in "
        "[Source: doc | section] format."
    )
    citations: List[Citation] = Field(
        description="List of Citation objects, one per source "
        "chunk referenced in the answer."
    )
    confidence: float = Field(
        description="Retrieval confidence score 0.0 to 1.0. "
        "Derived from top chunk cosine similarity score."
    )
    refusal: bool = Field(
        default=False,
        description="True when query is unanswerable from corpus "
        "or was blocked by the input scanner.",
    )
    query_type: str = Field(
        default="simple",
        description="Classification label from query classifier node. "
        "One of: simple, multi_hop, out_of_scope, ambiguous.",
    )

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_valid_probability(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        return value

    @computed_field(
        description="Deduplicated list of document names from citations. "
        "Computed from citations list — do not set manually."
    )
    @property
    def sources_used(self) -> List[str]:
        seen = []
        for citation in self.citations:
            if citation.doc not in seen:
                seen.append(citation.doc)
        return seen

    def __repr__(self) -> str:
        preview = self.answer[:80] + "..." if len(self.answer) > 80 else self.answer
        return (
            f"SecureOpsAnswer("
            f"answer=\"{preview}\", "
            f"citations={len(self.citations)}, "
            f"confidence={self.confidence:.2f}, "
            f"refusal={self.refusal})"
        )


class AgentState(TypedDict):
    # ── INPUT ──────────────────────────────────────────────────
    # Set by: run_agent() entry point before graph starts
    query: str

    # ── AGENT CLASSIFICATION (llm-and-agentic) ─────────────────
    # Set by: classify_query node
    # Values: "simple" | "multi_hop" | "out_of_scope" | "ambiguous"
    query_type: str

    # Set by: decompose_query node (multi_hop path only)
    # Empty list on simple path — retrieval uses original query
    sub_queries: List[str]

    # ── RETRIEVAL (rag-layer) ───────────────────────────────────
    # Set by: retrieval_fn injected from rag-layer
    # Each item must pass validate_chunk() from src.contracts
    retrieved_chunks: List[ChunkDict]

    # ── GENERATION (llm-and-agentic) ───────────────────────────
    # Set by: synthesize_answer node
    answer: str

    # Set by: synthesize_answer node
    # Parsed from inline [Source: doc | section] in LLM output
    citations: List[Citation]

    # Set by: synthesize_answer node (derived from top retrieved
    # chunk score) OR handle_refusal node (hardcoded to 0.0) —
    # these are mutually exclusive graph paths, never sequential
    confidence: float

    # ── OUTPUT FLAGS (output-layer + llm-and-agentic) ──────────
    # Set by: handle_refusal node (this branch). input_gate node
    # is output-layer's (Kaveen's) — not yet merged into dev, see
    # InputScannerProtocol in src.contracts for that contract.
    refusal: bool

    # Set by: handle_clarification node
    needs_clarification: bool

    # Set by: handle_clarification node
    clarification_question: str

    # ── VALIDATION (llm-and-agentic) ────────────────────────────
    # Set by: validate_citations node (src/agent.py — NOT
    # output-layer, despite this section header; the answer-
    # groundedness check lives in this branch) OR _stub_verify
    # node (always True, temporary — see src/agent.py module
    # docstring). _stub_verify's write is currently unread: the
    # "verify" node's outgoing edge uses a separate hardcoded
    # routing lambda, not this field, so two functions write
    # validation_passed but only validate_citations' write is
    # ever actually consulted by route_by_validation.
    validation_passed: bool

    # ── CONTROL FLOW ───────────────────────────────────────────
    # Set by: increment_retry_count node ONLY, on the "retry" edge
    # after validate_citations fails. validate_citations itself
    # does NOT touch this field (see validate_citations docstring —
    # this split was a deliberate fix for a bug where both functions
    # touching retry_count caused an off-by-one routing error).
    # Checked by route_by_validation to prevent infinite loops.
    # Must not exceed MAX_NODE_RETRIES from src.contracts.
    retry_count: int

    # Reserved for: any node on unexpected exception. NOT YET
    # IMPLEMENTED — no node in src/agent.py currently writes this
    # field. Exceptions from LLM calls (RateLimitError,
    # AllProvidersExhausted) currently propagate uncaught out of
    # run_agent and are only caught by chat_fn's top-level safety
    # net in src/gradio_demo.py, never recorded here. None during
    # normal operation.
    error: Optional[str]


AGENTSTATE_DEFAULTS: AgentState = {
    "query": "",
    "query_type": "simple",
    "sub_queries": [],
    "retrieved_chunks": [],
    "answer": "",
    "citations": [],
    "confidence": 0.0,
    "refusal": False,
    "needs_clarification": False,
    "clarification_question": "",
    "retry_count": 0,
    "error": None,
    "validation_passed": False,
}
