"""
retrieval.py — SecureOps Assistant · retrieval layer
================================================================================
Vector Cartel · AAI Tech Talks Hackathon 2026 · WMG, University of Warwick

PURPOSE
-------
Implement the contract's ``retrieval_fn(query) -> List[ChunkDict]`` — the single
function the llm-and-agentic layer calls. It runs the mandated hybrid recipe over
the indexes built by ``index.py``:

    query
      → bge-base embed (query instruction)        ┐
      → ChromaDB cosine          top-20  (dense)  │
      → BM25 keyword             top-20  (sparse) ┘
      → RRF fusion (by chunk_id) → fused candidates
      → cross-encoder rerank     → top-5
      → fill ``score`` ∈ [0,1], validate, return ordered desc

WHY THIS SHAPE
--------------
* Dense finds *semantic* matches; BM25 finds *exact* identifiers (CVE/CWE/codes).
  RRF fuses two rankings without needing comparable score scales — it uses rank
  position only, so a cosine distance and a BM25 score combine cleanly.
* The cross-encoder reads (query, chunk) *together* and is far more accurate than
  either first-stage retriever; it is the final arbiter over a small candidate
  pool. Its logit is squashed with a sigmoid into the [0,1] the contract requires.

CONTRACT GUARANTEES (see contracts.py)
--------------------------------------
* returns ``List[ChunkDict]`` ordered by ``score`` descending
* returns ``[]`` when nothing relevant is found — never None, never raises
* every returned chunk passes ``validate_chunk``
* ``RETRIEVAL_CONFIDENCE_THRESHOLD`` is imported, never hardcoded; by default the
  top-5 are returned regardless (the agent's verifier compares the max score to
  the threshold to decide "sufficient" vs "rewrite"). Pass ``min_score`` to hard
  filter instead.

USAGE
-----
    from src.retrieval import retrieve        # convenience (lazy singletons)
    chunks = retrieve("privilege escalation in Siemens RUGGEDCOM")

    # or, explicit (preferred for the agent — build once, inject):
    from src.retrieval import build_retrieval_fn
    retrieval_fn = build_retrieval_fn()
    chunks = retrieval_fn(query)

    # CLI (run as a module from the repo root, now that this lives under src/):
    python -m src.retrieval "how do I mitigate T0836?" -v
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import math
from typing import Any, Callable, Dict, List, Optional

from src.index import Indexes, embed_query, load_indexes, tokenize_for_bm25

logger = logging.getLogger("secureops.retrieval")

__all__ = ["build_retrieval_fn", "retrieve", "rrf_fuse", "RERANK_MODEL"]

# Optional contract import (validate + shared threshold).
try:
    from src.contracts import RETRIEVAL_CONFIDENCE_THRESHOLD, validate_chunk
except Exception:  # noqa: BLE001 - allow standalone runs
    RETRIEVAL_CONFIDENCE_THRESHOLD = 0.35

    def validate_chunk(c: Dict[str, Any]) -> bool:  # minimal fallback
        return (
            isinstance(c.get("text"), str)
            and bool(c["text"].strip())
            and isinstance(c.get("metadata"), dict)
            and "doc" in c["metadata"]
            and "section" in c["metadata"]
            and isinstance(c.get("score"), (int, float))
            and 0.0 <= float(c["score"]) <= 1.0
        )


# ==============================================================================
# SECTION 0 — Configuration
# ==============================================================================

# Cross-encoder reranker — swappable constant. bge-reranker-base pairs with the
# bge-base embedder and is strong for Tier 3; switch to
# "cross-encoder/ms-marco-MiniLM-L-6-v2" for a much lighter/faster fallback.
RERANK_MODEL = "BAAI/bge-reranker-base"

DENSE_K = 20          # ChromaDB cosine candidates
SPARSE_K = 20         # BM25 candidates
RRF_K = 60            # RRF damping constant (standard)
FUSE_TOP = 30         # fused candidates handed to the reranker
FINAL_K = 5           # chunks returned to the agent


# ==============================================================================
# SECTION 1 — First-stage retrievers
# ==============================================================================

def _dense_search(idx: Indexes, query: str, k: int) -> List[str]:
    """Return up to ``k`` chunk_ids by cosine similarity, best first."""
    res = idx.collection.query(query_embeddings=[embed_query(query)], n_results=k)
    ids = res.get("ids") or [[]]
    return list(ids[0])


def _sparse_search(idx: Indexes, query: str, k: int) -> List[str]:
    """Return up to ``k`` chunk_ids by BM25 score, best first."""
    import numpy as np

    tokens = tokenize_for_bm25(query)
    if not tokens:
        return []
    scores = idx.bm25.get_scores(tokens)
    top = np.argsort(scores)[::-1][:k]
    # keep only positive-score hits (BM25 returns 0 for no lexical overlap)
    return [idx.chunk_ids[i] for i in top if scores[i] > 0]


# ==============================================================================
# SECTION 2 — Reciprocal Rank Fusion
# ==============================================================================

def rrf_fuse(ranked_lists: List[List[str]], k: int = RRF_K) -> List[str]:
    """Reciprocal Rank Fusion over several ranked id-lists.

    Each list contributes ``1 / (k + rank)`` (rank is 0-based) to an id's score.
    Rank-based, so it merges incomparable score scales (cosine vs BM25) cleanly.
    Returns ids ordered by fused score descending.
    """
    fused: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=fused.get, reverse=True)


# ==============================================================================
# SECTION 3 — Cross-encoder reranker
# ==============================================================================

_RERANKER = None


def get_reranker():
    """Lazy-load the CrossEncoder reranker (cached, GPU if available)."""
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("loading reranker %s on %s", RERANK_MODEL, device)
        _RERANKER = CrossEncoder(RERANK_MODEL, device=device)
    return _RERANKER


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _rerank(query: str, chunks: List[Dict[str, Any]], reranker) -> List[Dict[str, Any]]:
    """Score (query, chunk.text) pairs, set chunk['score'] = sigmoid(logit),
    return chunks sorted by score descending."""
    if not chunks:
        return []
    pairs = [(query, c["text"]) for c in chunks]
    raw = reranker.predict(pairs)
    for c, s in zip(chunks, raw):
        c["score"] = max(0.0, min(1.0, _sigmoid(float(s))))
    return sorted(chunks, key=lambda c: c["score"], reverse=True)


# ==============================================================================
# SECTION 4 — Retrieval function builder (contract entry point)
# ==============================================================================

def build_retrieval_fn(
    indexes: Optional[Indexes] = None,
    reranker=None,
    *,
    dense_k: int = DENSE_K,
    sparse_k: int = SPARSE_K,
    fuse_top: int = FUSE_TOP,
    final_k: int = FINAL_K,
    min_score: Optional[float] = None,
) -> Callable[[str], List[Dict[str, Any]]]:
    """Build a ``retrieval_fn(query) -> List[ChunkDict]`` matching the contract.

    Loads the indexes + reranker once (or accepts injected ones, e.g. for tests).
    ``min_score`` (default None) optionally hard-filters chunks below a score; by
    default all ``final_k`` are returned and the agent decides on confidence via
    ``RETRIEVAL_CONFIDENCE_THRESHOLD``.
    """
    idx = indexes if indexes is not None else load_indexes()
    rer = reranker if reranker is not None else get_reranker()

    def retrieval_fn(query: str) -> List[Dict[str, Any]]:
        try:
            query = (query or "").strip()
            if not query:
                return []

            dense = _dense_search(idx, query, dense_k)
            sparse = _sparse_search(idx, query, sparse_k)
            fused_ids = rrf_fuse([dense, sparse])[:fuse_top]
            if not fused_ids:
                return []

            # reconstruct full ChunkDicts (fresh copies so we don't mutate the
            # cached store when we write the score)
            candidates = [dict(idx.id_to_chunk[cid]) for cid in fused_ids if cid in idx.id_to_chunk]
            reranked = _rerank(query, candidates, rer)

            top = reranked[:final_k]
            if min_score is not None:
                top = [c for c in top if c["score"] >= min_score]

            valid = [c for c in top if validate_chunk(c)]
            if len(valid) != len(top):
                logger.warning("dropped %d chunks failing validate_chunk", len(top) - len(valid))
            return valid
        except Exception:  # noqa: BLE001 - contract: never raise to the caller
            logger.exception("retrieval_fn failed for query %r", query)
            return []

    return retrieval_fn


# ==============================================================================
# SECTION 5 — Convenience singleton (lazy)
# ==============================================================================

_RETRIEVAL_FN: Optional[Callable[[str], List[Dict[str, Any]]]] = None


def retrieve(query: str) -> List[Dict[str, Any]]:
    """Convenience wrapper that lazily builds and reuses a default retrieval_fn."""
    global _RETRIEVAL_FN
    if _RETRIEVAL_FN is None:
        _RETRIEVAL_FN = build_retrieval_fn()
    return _RETRIEVAL_FN(query)


# ==============================================================================
# SECTION 6 — CLI
# ==============================================================================

def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Query the SecureOps hybrid retriever.")
    parser.add_argument("query", nargs="+", help="the query text")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    query = " ".join(args.query)
    fn = build_retrieval_fn(min_score=args.min_score)
    results = fn(query)

    print(f"\nQuery: {query!r}")
    print(f"Confidence threshold: {RETRIEVAL_CONFIDENCE_THRESHOLD}  "
          f"(top score {results[0]['score']:.3f})" if results else "  (no results)")
    print(f"Returned {len(results)} chunks:\n")
    for i, c in enumerate(results, 1):
        m = c["metadata"]
        print(f"{i}. score={c['score']:.3f}  [{m.get('source_type')}]  "
              f"{m.get('doc')} · {m.get('section')}  ({m.get('chunk_id')})")
        print(f"     {c['text'][:160].replace(chr(10), ' ')} ...\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
