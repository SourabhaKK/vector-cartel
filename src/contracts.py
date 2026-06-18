"""
================================================================================
src/contracts.py — SHARED INTERFACE CONTRACTS
Vector Cartel · SecureOps Assistant · AAI Tech Talks Hackathon 2026
================================================================================

PURPOSE
-------
This file is the single source of truth for all data structures that pass
between the three pipeline layers. It lives on the dev branch and is imported
by all branches.

    rag-layer (Jay + Sana)   →   llm-and-agentic (SK)   →   output-layer (Kaveen)
    builds retrieval_fn           builds agent nodes          builds output schema
    MUST return List[ChunkDict]   reads List[ChunkDict]        reads SecureOpsAnswer

DO NOT MODIFY THIS FILE without telling the full team first.
Any change to a field name, type, or structure here will break
the other two layers at merge time.

HOW TO USE THIS FILE
--------------------
Every branch pulls dev at the start of each day:

    git checkout <your-branch>
    git merge dev

Then import what you need:

    # In rag-layer (retrieval.py):
    from src.contracts import ChunkDict, RetrievalFn, validate_chunk

    # In llm-and-agentic (agent.py, llm.py):
    from src.contracts import ChunkDict, RetrievalFn, get_chunk_doc, get_chunk_section

    # In output-layer (answer.py, security.py):
    from src.contracts import ChunkDict, SecureOpsAnswerContract

BRANCH OWNERSHIP
----------------
    src/contracts.py    →  dev branch (no single owner — team agreement required)
    src/ingestion.py    →  rag-layer  (Jay Sadhu)
    src/chunking.py     →  rag-layer  (Jay Sadhu)
    src/retrieval.py    →  rag-layer  (Sana Shikalgar)
    src/agent.py        →  llm-and-agentic (SK)
    src/llm.py          →  llm-and-agentic (SK)
    src/prompts.py      →  llm-and-agentic (SK)
    src/schemas.py      →  llm-and-agentic (SK)
    src/security.py     →  output-layer (Kaveen Prabodhya)
    src/answer.py       →  output-layer (Kaveen Prabodhya)
    src/evaluation.py   →  output-layer (Kaveen Prabodhya)

INTEGRATION POINTS (where layers connect — highest merge risk)
--------------------------------------------------------------
Point 1: rag-layer → llm-and-agentic
    Function:   retrieval_fn(query: str) -> List[ChunkDict]
    Risk:       key name disagreements in metadata dict
    Mitigation: use get_chunk_doc() and get_chunk_section() accessors
                defined in this file, never read metadata keys directly

Point 2: llm-and-agentic → output-layer
    Object:     SecureOpsAnswer (defined in src/schemas.py)
    Risk:       field name or type mismatch
    Mitigation: Kaveen imports SecureOpsAnswer directly from src/schemas.py
                do not redefine it in output-layer

Point 3: output-layer → llm-and-agentic (security gate)
    Function:   InputScanner.scan(query: str) -> Tuple[bool, Optional[str]]
    Returns:    (is_clean: bool, reason: Optional[str])
    Risk:       return type disagreement
    Mitigation: defined as InputScanResult in this file
================================================================================
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1 — CHUNK DICTIONARY CONTRACT
# ==============================================================================
# A ChunkDict is the unit of data that travels from the rag-layer retrieval
# pipeline into the llm-and-agentic agent nodes.
#
# Every chunk returned by retrieval_fn MUST have this structure.
# The agent nodes will crash with a KeyError if any required key is missing.
#
# WHO PRODUCES THIS:  src/retrieval.py  (Sana Shikalgar — rag-layer)
# WHO CONSUMES THIS:  src/agent.py      (SK — llm-and-agentic)
#
# IMPORTANT: Do not add extra keys to metadata without updating get_chunk_doc()
# and get_chunk_section() below and telling the full team.
# ==============================================================================

# ChunkDict is implemented as a plain Dict for LangGraph compatibility.
# The TypedDict below is for documentation and type-checking only.
# At runtime, chunks are plain Python dicts.
#
# Required metadata keys:
#   doc      — human-readable document name
#              e.g. "NIST SP 800-82 Rev 3", "CISA Advisory 2024-001"
#   section  — section or identifier within the document
#              e.g. "5.2.3", "T0836", "PR.AC-1"
#   page     — page number (integer) or None if not applicable
#              CISA advisories and MITRE techniques may not have page numbers
#
# Optional metadata keys (add as needed, do not remove required ones):
#   vendor   — CISA advisories only, e.g. "Siemens"
#   date     — CISA advisories only, e.g. "2024-03-15"
#   cvss     — CISA advisories only, e.g. 8.9
#   technique_id  — MITRE ATT&CK only, e.g. "T0836"
#   tactic        — MITRE ATT&CK only, e.g. "Impair Process Control"

# Type alias — use this in all type annotations across all branches
ChunkDict = Dict[str, Any]

# Example of a valid ChunkDict — copy this structure in your tests:
#
#   {
#       "text": "Section 5.2.3 covers remote access controls for OT networks...",
#       "metadata": {
#           "doc": "NIST SP 800-82 Rev 3",
#           "section": "5.2.3",
#           "page": 87,
#       },
#       "score": 0.91,
#   }


def validate_chunk(chunk: Dict[str, Any]) -> bool:
    """
    Validates that a chunk dict has all required keys and correct types.

    Call this in rag-layer before returning chunks from retrieval_fn,
    and in llm-and-agentic before passing chunks to agent nodes.

    Args:
        chunk: A dict returned by the retrieval pipeline.

    Returns:
        True if the chunk is valid, False otherwise.
        Logs a warning describing the first validation failure found.

    Usage:
        # In rag-layer (retrieval.py) — validate before returning:
        chunks = [c for c in raw_chunks if validate_chunk(c)]

        # In llm-and-agentic (agent.py) — validate before consuming:
        if not all(validate_chunk(c) for c in state["retrieved_chunks"]):
            logger.warning("Invalid chunks received from retrieval layer")
    """
    # Check top-level keys exist
    if "text" not in chunk:
        logger.warning("validate_chunk FAILED: missing key 'text'")
        return False

    if "metadata" not in chunk:
        logger.warning("validate_chunk FAILED: missing key 'metadata'")
        return False

    if "score" not in chunk:
        logger.warning("validate_chunk FAILED: missing key 'score'")
        return False

    # Check text is a non-empty string
    if not isinstance(chunk["text"], str) or not chunk["text"].strip():
        logger.warning("validate_chunk FAILED: 'text' must be a non-empty string")
        return False

    # Check metadata is a dict with required keys
    metadata = chunk["metadata"]
    if not isinstance(metadata, dict):
        logger.warning("validate_chunk FAILED: 'metadata' must be a dict")
        return False

    if "doc" not in metadata:
        logger.warning("validate_chunk FAILED: metadata missing required key 'doc'")
        return False

    if "section" not in metadata:
        logger.warning(
            "validate_chunk FAILED: metadata missing required key 'section'"
        )
        return False

    # Check score is a float or int between 0 and 1
    score = chunk["score"]
    if not isinstance(score, (int, float)):
        logger.warning("validate_chunk FAILED: 'score' must be a number")
        return False

    if not (0.0 <= float(score) <= 1.0):
        logger.warning(
            f"validate_chunk FAILED: 'score' must be between 0.0 and 1.0, got {score}"
        )
        return False

    return True


# ==============================================================================
# SECTION 2 — SAFE METADATA ACCESSORS
# ==============================================================================
# Use these functions to read chunk metadata instead of accessing dict keys
# directly. They handle key name variations gracefully so a small naming
# inconsistency between branches does not crash the pipeline.
#
# RULE: Never write chunk["metadata"]["doc"] anywhere in the codebase.
#       Always write get_chunk_doc(chunk) instead.
#
# WHY: Jay might name the field "document_name", Sana might name it "doc".
#      These accessors try both and fall back to a safe default.
# ==============================================================================

def get_chunk_doc(chunk: ChunkDict) -> str:
    """
    Safely reads the document name from a chunk's metadata.

    Tries the following keys in order: "doc", "document_name", "source"
    Returns "Unknown Document" if none are found.

    Args:
        chunk: A ChunkDict returned by the retrieval pipeline.

    Returns:
        The document name as a string.

    Usage (in llm-and-agentic agent.py):
        doc_name = get_chunk_doc(chunk)
        citation = Citation(doc=doc_name, section=get_chunk_section(chunk), ...)
    """
    metadata = chunk.get("metadata", {})
    return (
        metadata.get("doc")
        or metadata.get("document_name")
        or metadata.get("source")
        or "Unknown Document"
    )


def get_chunk_section(chunk: ChunkDict) -> str:
    """
    Safely reads the section identifier from a chunk's metadata.

    Tries the following keys in order: "section", "section_id",
    "section_title", "technique_id", "category"
    Returns "Unknown Section" if none are found.

    Args:
        chunk: A ChunkDict returned by the retrieval pipeline.

    Returns:
        The section identifier as a string.

    Usage (in llm-and-agentic agent.py):
        section = get_chunk_section(chunk)
    """
    metadata = chunk.get("metadata", {})
    return (
        metadata.get("section")
        or metadata.get("section_id")
        or metadata.get("section_title")
        or metadata.get("technique_id")
        or metadata.get("category")
        or "Unknown Section"
    )


def get_chunk_page(chunk: ChunkDict) -> Optional[int]:
    """
    Safely reads the page number from a chunk's metadata.

    Returns None if page is not present (CISA advisories and MITRE
    techniques do not have page numbers — this is expected and valid).

    Args:
        chunk: A ChunkDict returned by the retrieval pipeline.

    Returns:
        Page number as int, or None.
    """
    metadata = chunk.get("metadata", {})
    page = metadata.get("page") or metadata.get("page_number")
    if page is None:
        return None
    try:
        return int(page)
    except (ValueError, TypeError):
        return None


def get_chunk_score(chunk: ChunkDict) -> float:
    """
    Safely reads the relevance score from a chunk.

    Returns 0.0 if score is missing or invalid.

    Args:
        chunk: A ChunkDict returned by the retrieval pipeline.

    Returns:
        Score as float between 0.0 and 1.0.
    """
    score = chunk.get("score", 0.0)
    try:
        return float(score)
    except (ValueError, TypeError):
        return 0.0


# ==============================================================================
# SECTION 3 — RETRIEVAL FUNCTION PROTOCOL
# ==============================================================================
# This defines the exact function signature that rag-layer must implement
# and that llm-and-agentic uses as a LangGraph node.
#
# WHO IMPLEMENTS THIS:  src/retrieval.py  (Sana Shikalgar — rag-layer)
# WHO CALLS THIS:       src/agent.py      (SK — llm-and-agentic)
#
# THE CONTRACT:
#   Input:  a single query string (already cleaned by input scanner)
#   Output: a list of ChunkDicts, ordered by relevance score descending
#           return empty list [] if no relevant chunks found
#           never return None, never raise exceptions to the caller
#
# RETRIEVAL PIPELINE INSIDE retrieval_fn (rag-layer responsibility):
#   1. Embed query with BAAI/bge-small-en-v1.5
#   2. ChromaDB cosine similarity search → top 20
#   3. BM25 keyword search → top 20
#   4. RRF fusion → merged ranked list
#   5. Cross-encoder reranker → top 5
#   6. Validate each chunk with validate_chunk()
#   7. Return List[ChunkDict] ordered by score descending
# ==============================================================================

@runtime_checkable
class RetrievalFn(Protocol):
    """
    Protocol defining the interface for the retrieval function.

    rag-layer must build a function matching this signature.
    llm-and-agentic uses this as the type annotation for the
    retrieval node in the LangGraph StateGraph.

    The function must:
    - Accept a single query string
    - Return a List[ChunkDict] ordered by score descending
    - Return [] (empty list) when nothing relevant is found
    - Never raise exceptions — catch internally and return []
    - Each returned chunk must pass validate_chunk()

    Example implementation skeleton (for rag-layer):

        def build_retrieval_fn(
            chroma_collection,
            bm25_index,
            embedder,
            reranker,
        ) -> RetrievalFn:

            def retrieval_fn(query: str) -> List[ChunkDict]:
                try:
                    dense_results = chroma_collection.query(...)
                    bm25_results = bm25_index.get_scores(...)
                    merged = rrf_fusion(dense_results, bm25_results)
                    reranked = reranker.rerank(query, merged)[:5]
                    return [c for c in reranked if validate_chunk(c)]
                except Exception as e:
                    logger.error(f"retrieval_fn failed: {e}")
                    return []

            return retrieval_fn

    Example stub (used in llm-and-agentic until rag-layer merges):

        _stub_retrieval_fn: RetrievalFn = lambda query: []
    """

    def __call__(self, query: str) -> List[ChunkDict]:
        ...


# ==============================================================================
# SECTION 4 — INPUT SCANNER RESULT CONTRACT
# ==============================================================================
# The InputScanner lives in output-layer (Kaveen's src/security.py).
# It is wired into the agent's input gate in llm-and-agentic (src/agent.py).
# This section defines the return type they must agree on.
#
# WHO IMPLEMENTS THIS:  src/security.py  (Kaveen Prabodhya — output-layer)
# WHO CALLS THIS:       src/agent.py     (SK — llm-and-agentic)
#
# THE CONTRACT:
#   scan(query: str) -> InputScanResult
#   InputScanResult = Tuple[bool, Optional[str]]
#     [0] bool          — True = clean query, False = blocked
#     [1] Optional[str] — reason for blocking, None if clean
#
# EXAMPLE RETURN VALUES:
#   (True,  None)                          — clean, proceed
#   (False, "injection keyword detected")  — blocked
#   (False, "query exceeds 1000 chars")    — blocked
# ==============================================================================

# Type alias for the return type of InputScanner.scan()
# Use this in both security.py and agent.py type annotations.
InputScanResult = Tuple[bool, Optional[str]]

# Example values for use in tests across all branches:
CLEAN_SCAN_RESULT: InputScanResult = (True, None)
BLOCKED_SCAN_RESULT: InputScanResult = (False, "injection keyword detected")


@runtime_checkable
class InputScannerProtocol(Protocol):
    """
    Protocol defining the interface for the InputScanner class.

    output-layer must implement a class matching this protocol.
    llm-and-agentic uses this as the type annotation for the
    security gate at the agent entry point.

    The scan method must:
    - Accept a single query string
    - Return InputScanResult = Tuple[bool, Optional[str]]
    - Never raise exceptions — catch internally and return blocked result
    - Be callable with just the query string — no other required args

    Example usage in llm-and-agentic (agent.py input gate node):

        def input_gate_node(
            state: AgentState,
            scanner: InputScannerProtocol,
        ) -> dict:
            is_clean, reason = scanner.scan(state["query"])
            if not is_clean:
                return {
                    "refusal": True,
                    "answer": f"Query blocked: {reason}",
                }
            return {}  # no state change, proceed to classifier
    """

    def scan(self, query: str) -> InputScanResult:
        ...


# ==============================================================================
# SECTION 5 — SECUREOPSANSWER CONTRACT REFERENCE
# ==============================================================================
# SecureOpsAnswer is the final output object of the full pipeline.
# It is DEFINED in src/schemas.py (llm-and-agentic branch, SK's file).
# It is CONSUMED in src/answer.py and src/evaluation.py (output-layer, Kaveen).
#
# Kaveen: import SecureOpsAnswer directly from src.schemas — do NOT redefine it.
#
#   from src.schemas import SecureOpsAnswer, Citation
#
# This section documents the expected fields for reference only.
# The authoritative definition is always src/schemas.py.
#
# FIELD REFERENCE (do not copy — import from src.schemas):
#   answer:       str          — full answer text with inline citations
#   citations:    List[Citation]  — list of Citation objects
#   confidence:   float        — 0.0 to 1.0, derived from retrieval scores
#   refusal:      bool         — True if query was unanswerable or blocked
#   query_type:   str          — "simple", "multi_hop", "out_of_scope", "ambiguous"
#   sources_used: List[str]    — distinct document names from citations
#
# Citation fields:
#   doc:     str           — document name matching ChunkDict metadata "doc"
#   section: str           — section identifier matching ChunkDict metadata "section"
#   page:    Optional[int] — page number, may be None
#   snippet: str           — max 200 chars of the source chunk text
# ==============================================================================

# This import makes SecureOpsAnswer available from contracts.py as a convenience.
# If src/schemas.py does not exist yet (early dev), this will fail gracefully.
try:
    from src.schemas import SecureOpsAnswer, Citation  # noqa: F401
    _SCHEMAS_AVAILABLE = True
except ImportError:
    _SCHEMAS_AVAILABLE = False
    logger.debug(
        "src/schemas.py not yet available — SecureOpsAnswer not importable "
        "from contracts. This is expected before llm-and-agentic is merged."
    )


# ==============================================================================
# SECTION 6 — RETRIEVAL CONFIDENCE THRESHOLD
# ==============================================================================
# This threshold is used in TWO places and must be the same value in both.
# Import this constant — do not hardcode 0.35 in your own files.
#
# Used by:
#   llm-and-agentic (agent.py) — retrieval verifier node decides
#                                 "sufficient" vs "insufficient"
#   rag-layer (retrieval.py)   — optional: filter out very low scoring chunks
#                                 before returning
# ==============================================================================

# Minimum cosine similarity score for a retrieved chunk to be considered
# relevant. Chunks with max score below this trigger the query rewriter.
RETRIEVAL_CONFIDENCE_THRESHOLD: float = 0.35

# Minimum token overlap fraction for a sentence in the generated answer
# to be considered grounded in the retrieved context.
# Used by the output validator in output-layer (src/security.py).
OUTPUT_GROUNDEDNESS_THRESHOLD: float = 0.25

# Maximum number of retries allowed per agent node before forcing a pass.
# Prevents infinite loops in the LangGraph retry edges.
MAX_NODE_RETRIES: int = 1


# ==============================================================================
# SECTION 7 — QUICK REFERENCE FOR AI CODING ASSISTANTS
# ==============================================================================
# If you are using Claude, Copilot, or another AI tool to help build your
# layer, paste this section into your prompt as context:
#
# "I am building part of a RAG pipeline called SecureOps Assistant.
#  The shared data contract is defined in src/contracts.py.
#
#  A retrieved chunk looks like this:
#  {
#      'text': str,
#      'metadata': {'doc': str, 'section': str, 'page': Optional[int]},
#      'score': float  # 0.0 to 1.0
#  }
#
#  Always use get_chunk_doc(chunk) and get_chunk_section(chunk) from
#  src/contracts.py instead of reading metadata keys directly.
#
#  The retrieval function signature is:
#      retrieval_fn(query: str) -> List[ChunkDict]
#
#  The input scanner return type is:
#      scanner.scan(query: str) -> Tuple[bool, Optional[str]]
#      True = clean, False = blocked
#
#  The final answer object is SecureOpsAnswer from src/schemas.py."
# ==============================================================================
