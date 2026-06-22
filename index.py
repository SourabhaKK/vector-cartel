"""
index.py — SecureOps Assistant · index-build layer
================================================================================
Vector Cartel · AAI Tech Talks Hackathon 2026 · WMG, University of Warwick

PURPOSE
-------
Turn the ``chunks.jsonl`` handoff produced by ``chunking.py`` into the two
searchable indexes the hybrid retriever needs, both keyed by ``chunk_id`` so
their results can be fused (RRF) downstream:

    chunks.jsonl
        ├──► DENSE  : bge embeddings → ChromaDB collection (cosine)   semantic
        └──► SPARSE : BM25 over tokenised text                        exact terms

    chunks.jsonl  ──►  index.py  ──►  index_store/{chroma, bm25.pkl}

WHY HYBRID (dense + BM25), NOT A SINGLE MODEL
---------------------------------------------
The corpus is dense with exact identifiers — ``CVE-2026-27668``, ``GV.RR-01``,
``CWE-266``, version strings. Dense embeddings are weak on arbitrary identifiers;
classical BM25 matches them exactly. So the two indexes are complementary: the
embedder carries *semantics* ("privilege escalation", "network segmentation"),
BM25 carries *lexical / identifier* matching. retrieval.py fuses them with RRF.

EMBEDDING MODEL
---------------
``EMBED_MODEL`` is a single swappable constant (default ``bge-base-en-v1.5``).
Falling back to ``bge-small-en-v1.5`` is a one-line change and needs no other
edits — same family, same 512-token cap our chunking is sized to. No
security-specific embedder is needed because BM25 owns identifier matching.

The query (not the documents) is prefixed with the BGE retrieval instruction,
and all vectors are L2-normalised for cosine — both required for BGE to score
correctly. ``embed_query`` / ``embed_documents`` / ``tokenize_for_bm25`` are
exported so retrieval.py uses the *identical* embedding + tokenisation at query
time (mismatched query handling silently destroys recall).

USAGE
-----
    # build (one-off, re-run when chunks.jsonl changes):
    python index.py --chunks chunks.jsonl --selftest -v

    # from retrieval.py:
    from index import load_indexes, embed_query, tokenize_for_bm25
    idx = load_indexes()
    dense = idx.collection.query(query_embeddings=[embed_query(q)], n_results=20)
    bm25_scores = idx.bm25.get_scores(tokenize_for_bm25(q))
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("secureops.index")

__all__ = [
    "build_index",
    "load_indexes",
    "load_chunks",
    "embed_documents",
    "embed_query",
    "tokenize_for_bm25",
    "Indexes",
    "EMBED_MODEL",
]


# ==============================================================================
# SECTION 0 — Configuration
# ==============================================================================

# Single swappable constant. Fall back to "BAAI/bge-small-en-v1.5" if compute is
# tight — same family, same 512-token cap, no other code changes needed.
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# BGE retrieval instruction — prepended to the QUERY only, never to documents.
# (If you switch to an E5 model, use "query: " / "passage: " on both instead.)
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

INDEX_DIR = Path("index_store")
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_FILE = INDEX_DIR / "bm25.pkl"
COLLECTION_NAME = "secureops"

EMBED_BATCH = 64

# Reserved metadata keys Chroma stores natively / we never want as filters.
_DROP_META_KEYS: tuple = ()


# ==============================================================================
# SECTION 1 — Chunk loading
# ==============================================================================

def load_chunks(path: str | Path = "chunks.jsonl") -> List[Dict[str, Any]]:
    """Load the ChunkDicts from the JSONL handoff file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run chunking.py first to produce it."
        )
    chunks: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping malformed JSON on line %d", ln)
    logger.info("loaded %d chunks from %s", len(chunks), path)
    return chunks


# ==============================================================================
# SECTION 2 — Dense embedding (bge, normalised, query-instruction aware)
# ==============================================================================

_EMBEDDER = None


def get_embedder():
    """Lazy-load the SentenceTransformer embedder (cached process-wide)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer  # heavy import

        import torch  # noqa: WPS433 - local import keeps module import cheap

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("loading embedder %s on %s", EMBED_MODEL, device)
        _EMBEDDER = SentenceTransformer(EMBED_MODEL, device=device)
    return _EMBEDDER


def embed_documents(texts: List[str], show_progress: bool = False):
    """Embed passages (no instruction prefix), L2-normalised for cosine."""
    model = get_embedder()
    return model.encode(
        texts,
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )


def embed_query(text: str):
    """Embed a single query WITH the BGE retrieval instruction, normalised.

    retrieval.py must use this (not embed_documents) for the query, or BGE
    scores degrade. Returns a 1-D float list suitable for Chroma's
    ``query_embeddings=[embed_query(q)]``.
    """
    model = get_embedder()
    vec = model.encode(
        QUERY_INSTRUCTION + text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.tolist()


# ==============================================================================
# SECTION 3 — Sparse tokeniser (shared by build AND query time)
# ==============================================================================

# Keep security identifiers whole: "cve-2026-27668", "gv.rr-01", "5.4.1" stay as
# single tokens (joined by - . /), so an exact-ID query matches exactly. Plain
# words tokenise normally. The SAME function must run on documents and queries.
_BM25_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")


def tokenize_for_bm25(text: str) -> List[str]:
    """Lower-case, identifier-preserving tokeniser for BM25."""
    return _BM25_TOKEN_RE.findall((text or "").lower())


# ==============================================================================
# SECTION 4 — Build
# ==============================================================================

def _chroma_safe_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma requires non-empty scalar metadata. Chunking already produces
    scalars; this is a defensive final coercion (drop None, stringify oddities)."""
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if k in _DROP_META_KEYS or v is None:
            continue
        out[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return out


def build_index(
    chunks_path: str | Path = "chunks.jsonl",
    index_dir: str | Path = INDEX_DIR,
    rebuild: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Build the dense (Chroma) and sparse (BM25) indexes from chunks.jsonl.

    Returns a small stats dict. Re-running with ``rebuild=True`` (default) wipes
    the previous store so the indexes always match the current chunks.jsonl.
    """
    import chromadb

    index_dir = Path(index_dir)
    chroma_dir = index_dir / "chroma"
    bm25_file = index_dir / "bm25.pkl"

    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError("no chunks to index")

    chunk_ids = [c["metadata"]["chunk_id"] for c in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("duplicate chunk_id values — IDs must be unique")
    texts = [c["text"] for c in chunks]

    if rebuild and index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # --- DENSE: ChromaDB (cosine) -------------------------------------------
    logger.info("embedding %d chunks with %s ...", len(texts), EMBED_MODEL)
    embeddings = embed_documents(texts, show_progress=show_progress)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - absent on first build
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    metadatas = [_chroma_safe_meta(c["metadata"]) for c in chunks]
    for i in range(0, len(chunks), 512):  # Chroma add cap is generous; batch anyway
        sl = slice(i, i + 512)
        collection.add(
            ids=chunk_ids[sl],
            embeddings=[e.tolist() for e in embeddings[sl]],
            documents=texts[sl],
            metadatas=metadatas[sl],
        )
    logger.info("chroma collection '%s' built: %d vectors", COLLECTION_NAME, collection.count())

    # --- SPARSE: BM25 --------------------------------------------------------
    from rank_bm25 import BM25Okapi

    logger.info("tokenising + building BM25 over %d chunks ...", len(texts))
    corpus_tokens = [tokenize_for_bm25(t) for t in texts]
    bm25 = BM25Okapi(corpus_tokens)
    with open(bm25_file, "wb") as fh:
        pickle.dump(
            {
                "bm25": bm25,
                "chunk_ids": chunk_ids,
                "chunks": chunks,          # full ChunkDicts, aligned to chunk_ids
                "embed_model": EMBED_MODEL,
                "tokenizer": "idpreserve-v1",
            },
            fh,
        )
    logger.info("bm25 index written -> %s", bm25_file)

    dims = int(embeddings.shape[1]) if hasattr(embeddings, "shape") else None
    return {
        "chunks": len(chunks),
        "embed_model": EMBED_MODEL,
        "embed_dim": dims,
        "chroma_count": collection.count(),
        "index_dir": str(index_dir.resolve()),
    }


# ==============================================================================
# SECTION 5 — Load (consumed by retrieval.py)
# ==============================================================================

@dataclass
class Indexes:
    """Everything retrieval.py needs, loaded once.

    ``collection``    : Chroma collection (dense, cosine)
    ``bm25``          : fitted BM25Okapi model
    ``chunk_ids``     : ordering aligned to the BM25 corpus
    ``chunks``        : full ChunkDicts aligned to ``chunk_ids``
    ``id_to_chunk``   : chunk_id -> ChunkDict, for reconstructing results
    ``embed_model``   : model name the index was built with (sanity check)
    """

    collection: Any
    bm25: Any
    chunk_ids: List[str]
    chunks: List[Dict[str, Any]]
    id_to_chunk: Dict[str, Dict[str, Any]]
    embed_model: str


def load_indexes(index_dir: str | Path = INDEX_DIR) -> Indexes:
    """Load the dense + sparse indexes built by :func:`build_index`."""
    import chromadb

    index_dir = Path(index_dir)
    chroma_dir = index_dir / "chroma"
    bm25_file = index_dir / "bm25.pkl"
    if not chroma_dir.exists() or not bm25_file.exists():
        raise FileNotFoundError(
            f"index not found under {index_dir} — run `python index.py` first."
        )

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(COLLECTION_NAME)

    with open(bm25_file, "rb") as fh:
        store = pickle.load(fh)

    if store.get("embed_model") != EMBED_MODEL:
        logger.warning(
            "index built with %s but EMBED_MODEL is now %s — rebuild to align "
            "query and document embeddings.",
            store.get("embed_model"),
            EMBED_MODEL,
        )

    chunks = store["chunks"]
    return Indexes(
        collection=collection,
        bm25=store["bm25"],
        chunk_ids=store["chunk_ids"],
        chunks=chunks,
        id_to_chunk={c["metadata"]["chunk_id"]: c for c in chunks},
        embed_model=store.get("embed_model", "unknown"),
    )


# ==============================================================================
# SECTION 6 — CLI / self-test
# ==============================================================================

def _selftest(idx: Indexes, query: str = "Siemens privilege escalation vulnerability") -> None:
    """Sanity check: run the query through BOTH indexes and show top hits.
    This is NOT the retriever (no RRF/rerank) — just proof the wiring works."""
    import numpy as np

    print(f"\n--- self-test query: {query!r} ---")

    dense = idx.collection.query(query_embeddings=[embed_query(query)], n_results=3)
    print("\n[dense / ChromaDB cosine] top 3:")
    for cid, dist in zip(dense["ids"][0], dense["distances"][0]):
        sec = idx.id_to_chunk[cid]["metadata"].get("section", "?")
        print(f"   cos_sim={1 - dist:6.3f}  {cid}  · {sec}")

    scores = idx.bm25.get_scores(tokenize_for_bm25(query))
    top = np.argsort(scores)[::-1][:3]
    print("\n[sparse / BM25] top 3:")
    for i in top:
        cid = idx.chunk_ids[i]
        sec = idx.id_to_chunk[cid]["metadata"].get("section", "?")
        print(f"   bm25={scores[i]:6.3f}  {cid}  · {sec}")


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the SecureOps hybrid index.")
    parser.add_argument("--chunks", default="chunks.jsonl")
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    parser.add_argument("--selftest", action="store_true", help="run a sample query after building")
    parser.add_argument("--query", default="Siemens privilege escalation vulnerability")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    stats = build_index(args.chunks, args.index_dir)
    print("\nIndex built:")
    for k, v in stats.items():
        print(f"  {k:13s}: {v}")

    if args.selftest:
        _selftest(load_indexes(args.index_dir), args.query)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
