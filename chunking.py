"""
chunking.py — SecureOps Assistant · chunking layer
================================================================================
Vector Cartel · AAI Tech Talks Hackathon 2026 · WMG, University of Warwick

PURPOSE
-------
Turn the uniform ``Document`` units produced by ``ingestion.py`` into the
``ChunkDict`` units defined in ``contracts.py`` — the unit the retrieval layer
embeds, indexes (ChromaDB + BM25) and ranks.

    List[Document]  ──►  chunking.py  ──►  List[ChunkDict]

STRATEGY (hybrid — a different method per source; see README §6)
----------------------------------------------------------------
    CISA advisories : structure-aware (split on ##) + record-based
                      (one chunk per ### CVE block: desc + products + metrics)
    MITRE ATT&CK    : parent-child (Description = parent, each Mitigation = child)
    NIST SP 800-82  : recursive splitting within each page, ~512-token budget
    NIST CSF 2.0    : recursive splitting, section = subcategory code / heading

On top of every chunk:
    * a deterministic CONTEXTUAL HEADER is prepended before embedding so a small
      chunk stays self-describing (Anthropic "Contextual Retrieval", template
      variant — reproducible, zero cost, no hallucination risk)
    * RICH SCALAR METADATA (doc/section/page + vendor/date/cvss/technique_id/
      tactic) is attached for the contract's hybrid + metadata-filter retrieval

HARD CONSTRAINT
---------------
The contract suggests the embedder ``bge-small-en-v1.5`` (512-token cap). Chunks
longer than that are silently TRUNCATED at embed time. The truncation budget is
measured in *tokens* — not characters or words — so this module sizes every
chunk with the embedder's own tokenizer (see ``count_tokens``). Token counting
is the only proxy that is not fooled by token-dense security text (a single
``CVSS:3.1/AV:N/...`` vector is one "word" but ~26 tokens), so it is the safe
unit for guaranteeing no chunk is silently cut. A graceful fallback chain
(bge tokenizer → tiktoken → heuristic) keeps chunking from hard-depending on any
single package.

OUTPUT (contract ChunkDict)
---------------------------
    {"text": "<contextual header>\\n<body>",
     "metadata": {"doc": str, "section": str, "page": Optional[int], ...},
     "score": 0.0}          # score is assigned at RETRIEVAL time, not here

ChromaDB metadata must be scalar (str/int/float/bool) and non-null, so list
fields (cves, sectors) are flattened to comma-strings and None values dropped.

USAGE
-----
    from chunking import chunk_corpus
    from ingestion import load_corpus
    chunks = chunk_corpus(load_corpus("new_corpus"))

    # or from the command line:
    python chunking.py --corpus new_corpus --sample --out chunks.jsonl
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestion import Document, load_corpus

logger = logging.getLogger("secureops.chunking")

__all__ = [
    "chunk_corpus", "chunk_document", "add_llm_context", "CHUNK_CONFIG",
    "count_tokens", "tokenizer_name",
]

# Optional: validate against the shared contract if it is importable.
try:
    from contracts import validate_chunk  # type: ignore

    _HAVE_CONTRACT = True
except Exception:  # noqa: BLE001 - contracts may pull optional deps
    _HAVE_CONTRACT = False


# ==============================================================================
# SECTION 0 — Configuration
# ==============================================================================

CHUNK_CONFIG = {
    # Budgets are measured in TOKENS (the unit the embedder truncates on), via
    # the embedder's own tokenizer — not characters or words. bge-small's hard
    # cap is 512 tokens; we target 480 for the WHOLE embedded text (header +
    # body) so the 2 special tokens ([CLS]/[SEP]) added at embed time still fit
    # with margin.
    "max_tokens": 480,
    "overlap_tokens": 50,
    # chunks shorter than this (in tokens) are dropped as noise (empty headings).
    "min_tokens": 12,
}

ChunkDict = Dict[str, Any]


# ------------------------------------------------------------------------------
# Token measurement
# ------------------------------------------------------------------------------
# Chunk sizes are measured in *tokens* because the embedder truncates on a token
# budget. Security text (CVSS vectors, CVE/CWE IDs, version strings) packs many
# tokens into few words/chars, so token counting is the only proxy that reliably
# avoids silent truncation. The counter is lazy-initialised and cached, and
# degrades gracefully so chunking never hard-depends on a heavy/offline package.

_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_TOKEN_COUNTER = None
_TOKENIZER_NAME = "uninitialised"


def _build_token_counter():
    """Return ``(count_fn, name)`` — prefer the embedder's exact tokenizer."""
    # 1) the real bge-small tokenizer (exact match to the embedder's budget)
    try:
        from transformers import AutoTokenizer  # type: ignore

        tk = AutoTokenizer.from_pretrained(_EMBED_MODEL)
        return (
            lambda s: len(tk.encode(s, add_special_tokens=False)) if s else 0,
            f"transformers:{_EMBED_MODEL}",
        )
    except Exception as exc:  # noqa: BLE001 - offline / not installed
        logger.info("bge-small tokenizer unavailable (%s); trying tiktoken", exc)
    # 2) tiktoken — a different vocab, but a solid subword proxy
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return (lambda s: len(enc.encode(s)) if s else 0, "tiktoken:cl100k_base")
    except Exception:  # noqa: BLE001
        logger.info("tiktoken unavailable; using heuristic token estimate")
    # 3) heuristic — conservative chars/token for dense security text
    return (lambda s: max(1, ceil(len(s) / 3.3)) if s else 0, "heuristic:chars/3.3")


def count_tokens(text: str) -> int:
    """Token count under the active tokenizer (lazy-initialised, cached)."""
    global _TOKEN_COUNTER, _TOKENIZER_NAME
    if _TOKEN_COUNTER is None:
        _TOKEN_COUNTER, _TOKENIZER_NAME = _build_token_counter()
        logger.info("token counter: %s", _TOKENIZER_NAME)
    return _TOKEN_COUNTER(text)


def tokenizer_name() -> str:
    """Name of the active tokenizer (forces lazy init)."""
    count_tokens("")
    return _TOKENIZER_NAME


# ==============================================================================
# SECTION 1 — Generic text helpers
# ==============================================================================

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_CVE_RE = re.compile(r"CVE-\d{4}-\d+")
_SENTENCE_SPLIT_RE = re.compile(r"(\n+|(?<=[.!?])\s+)")


def _slug(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", (text or "").strip()).strip("-").lower()
    return s[:maxlen] or "x"


def _atomic_units(text: str) -> List[str]:
    """Split text into the smallest sensible units (sentences / lines),
    keeping their trailing whitespace so re-joining is lossless."""
    parts = _SENTENCE_SPLIT_RE.split(text)
    units: List[str] = []
    for p in parts:
        if not p:
            continue
        if p.strip() == "" and units:        # a separator → glue to previous unit
            units[-1] += p
        else:
            units.append(p)
    return units


def _effective_max(header: str) -> int:
    """Body budget (in TOKENS) once the contextual header (+ its blank line) is
    prepended, so the *embedded* text stays under the bge-small token cap."""
    return max(CHUNK_CONFIG["max_tokens"] - count_tokens(header) - 1, 120)


def _hard_split_oversized(unit: str, max_tokens: int) -> List[str]:
    """Last-resort split of a single atomic unit that alone exceeds the token
    budget (e.g. a giant unbroken table row). Cuts on characters, shrinking the
    cut point until each piece fits the token budget."""
    pieces: List[str] = []
    s = unit
    while count_tokens(s) > max_tokens:
        # estimate a char cut point from the current token density, then shrink
        approx = max(1, int(len(s) * max_tokens / max(count_tokens(s), 1)))
        piece = s[:approx]
        while count_tokens(piece) > max_tokens and len(piece) > 1:
            piece = piece[: int(len(piece) * 0.9)]
        pieces.append(piece)
        s = s[len(piece):]
    if s.strip():
        pieces.append(s)
    return pieces


def recursive_split(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """Greedy sentence-merge splitter with TOKEN-based budgeting and overlap.

    Merges atomic units (sentences/lines) until adding the next would exceed
    ``max_tokens``; the following chunk is then seeded with the trailing units of
    the previous one (~``overlap_tokens`` tokens) for context continuity. A
    single atomic unit larger than the budget is hard-split as a last resort.
    Used as the fallback for any section/page that does not already fit.
    """
    text = text.strip()
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]

    chunks: List[str] = []
    cur: List[str] = []
    cur_tok = 0

    def flush() -> None:
        if cur and "".join(cur).strip():
            chunks.append("".join(cur).strip())

    for unit in _atomic_units(text):
        ut = count_tokens(unit)

        if ut > max_tokens:                        # unit too big to ever fit
            flush()
            cur, cur_tok = [], 0
            pieces = _hard_split_oversized(unit, max_tokens)
            chunks.extend(p.strip() for p in pieces[:-1] if p.strip())
            if pieces:                             # keep remainder open to pack onto
                cur = [pieces[-1]]
                cur_tok = count_tokens(pieces[-1])
            continue

        if cur and cur_tok + ut > max_tokens:      # would overflow → close chunk
            flush()
            # seed next chunk with trailing units (~overlap_tokens) for continuity
            overlap_units: List[str] = []
            otok = 0
            for u in reversed(cur):
                if overlap_units and otok + count_tokens(u) > max_tokens // 2:
                    break                          # guard: never carry > half a chunk
                overlap_units.insert(0, u)
                otok += count_tokens(u)
                if otok >= overlap_tokens:
                    break
            cur = overlap_units
            cur_tok = sum(count_tokens(u) for u in cur)
            if cur_tok + ut > max_tokens:          # overlap+unit still overflow →
                cur, cur_tok = [], 0               # sacrifice overlap at this seam

        cur.append(unit)
        cur_tok += ut

    flush()
    return [c for c in chunks if c.strip()]


def _split_headings(body: str) -> List[Dict[str, Any]]:
    """Parse a Markdown body into a flat list of heading sections.

    Each section: ``{"level": int, "title": str|None, "text": str}``.
    The text before the first heading is returned as a level-0 section.
    """
    sections: List[Dict[str, Any]] = []
    cur: Dict[str, Any] = {"level": 0, "title": None, "lines": []}
    for line in body.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            sections.append(cur)
            cur = {"level": len(m.group(1)), "title": m.group(2).strip(), "lines": []}
        else:
            cur["lines"].append(line)
    sections.append(cur)
    for s in sections:
        s["text"] = "\n".join(s["lines"]).strip()
        del s["lines"]
    return sections


def _scalar_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a metadata dict to ChromaDB-safe scalars: drop None, join lists."""
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            joined = ", ".join(str(x) for x in v if x is not None)
            if joined:
                out[k] = joined
            continue
        if isinstance(v, (bool, int, float, str)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _make_chunk(
    *,
    header: str,
    body: str,
    doc: str,
    section: str,
    source_type: str,
    chunk_id: str,
    extra_meta: Dict[str, Any],
    page: Optional[int] = None,
    parent_id: Optional[str] = None,
) -> Optional[ChunkDict]:
    """Assemble one contract-shaped ChunkDict (or None if the body is too small)."""
    body = body.strip()
    if count_tokens(body) < CHUNK_CONFIG["min_tokens"]:
        return None
    text = f"{header}\n\n{body}" if header else body
    meta: Dict[str, Any] = {
        "doc": doc,
        "section": section or "unknown",
        "source_type": source_type,
        "chunk_id": chunk_id,
    }
    if page is not None:
        meta["page"] = page
    if parent_id:
        meta["parent_id"] = parent_id
    meta.update(extra_meta)
    return {"text": text, "metadata": _scalar_meta(meta), "score": 0.0}


# ==============================================================================
# SECTION 2 — Contextual header builders (deterministic / template)
# ==============================================================================

def _advisory_header(doc: str, meta: Dict[str, Any], cve: Optional[str]) -> str:
    parts = [doc]
    if meta.get("title"):
        parts.append(str(meta["title"]))
    if meta.get("vendor"):
        parts.append(str(meta["vendor"]))
    if cve:
        parts.append(cve)
    if meta.get("cvss") is not None:
        parts.append(f"CVSS {meta['cvss']}")
    return "[" + " · ".join(parts) + "]"


def _attck_header(meta: Dict[str, Any]) -> str:
    parts = [f"MITRE ATT&CK {meta.get('technique_id', '')}".strip()]
    if meta.get("name"):
        parts.append(str(meta["name"]))
    if meta.get("tactic"):
        parts.append(str(meta["tactic"]))
    return "[" + " · ".join(p for p in parts if p) + "]"


def _nist_header(doc: str, section: str, page: Optional[int]) -> str:
    parts = [doc]
    if section and not section.startswith("p."):
        parts.append(f"§{section}")
    if page is not None:
        parts.append(f"p.{page}")
    return "[" + " · ".join(parts) + "]"


# ==============================================================================
# SECTION 3 — CISA advisory chunker (structure-aware + record-based)
# ==============================================================================

def _chunk_advisory(doc: Document) -> List[ChunkDict]:
    m = doc.metadata
    alert_code = m.get("alert_code", doc.doc)
    base = {
        "alert_code": alert_code,
        "title": m.get("title"),
        "vendor": m.get("vendor"),
        "date": m.get("date"),
        "cvss": m.get("cvss"),
        "cves": m.get("cves"),
        "sectors": m.get("sectors"),
    }

    sections = _split_headings(doc.text)

    # Group into "primary" blocks: every ## section, plus each ### CVE promoted
    # to its own block. Deeper headings (#### Affected Products / Metrics) attach
    # to the current block.
    groups: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for s in sections:
        title, level, text = s["title"], s["level"], s["text"]
        if level == 0 and not text:
            continue
        is_cve = level == 3 and title and _CVE_RE.search(title)
        is_primary = level in (1, 2) or is_cve
        if level == 1:
            continue  # title / "Release Date / Alert Code" preamble — redundant
        if is_primary or current is None:
            current = {"title": title or "Overview", "is_cve": bool(is_cve), "parts": []}
            groups.append(current)
            if text:
                current["parts"].append(text)
        else:  # attached deeper subsection — keep its label inline
            block = (f"**{title}**\n{text}" if title else text).strip()
            if block:
                current["parts"].append(block)

    chunks: List[ChunkDict] = []
    for gi, g in enumerate(groups):
        body = "\n\n".join(p for p in g["parts"] if p).strip()
        if not body:
            continue
        cve = None
        if g["is_cve"]:
            mobj = _CVE_RE.search(g["title"])
            cve = mobj.group(0) if mobj else None
            section = cve or g["title"]
        else:
            section = g["title"]
        header = _advisory_header(doc.doc, m, cve)
        extra = dict(base)
        if cve:
            extra["cve"] = cve
        for ci, piece in enumerate(recursive_split(body, _effective_max(header), CHUNK_CONFIG["overlap_tokens"])):
            cid = f"{alert_code}::{_slug(section)}::{gi}-{ci}"
            chunk = _make_chunk(
                header=header, body=piece, doc=doc.doc, section=section,
                source_type="advisory", chunk_id=cid, extra_meta=extra,
            )
            if chunk:
                chunks.append(chunk)
    return chunks


# ==============================================================================
# SECTION 4 — MITRE ATT&CK chunker (parent-child)
# ==============================================================================

def _chunk_attck(doc: Document) -> List[ChunkDict]:
    m = doc.metadata
    tid = m.get("technique_id", doc.doc)
    base = {
        "technique_id": tid,
        "name": m.get("name"),
        "tactic": m.get("tactic"),
        "is_subtechnique": m.get("is_subtechnique"),
        "parent_technique": m.get("parent_technique"),
    }
    header = _attck_header(m)
    sections = _split_headings(doc.text)

    chunks: List[ChunkDict] = []
    parent_id = f"{tid}::desc"

    # Parent: the Description section (fall back to all pre-Mitigations text).
    desc_text = ""
    in_mitigations = False
    mitigations: List[Dict[str, str]] = []
    for s in sections:
        title = (s["title"] or "").strip()
        if s["level"] == 2 and title.lower().startswith("description"):
            desc_text = s["text"]
        elif s["level"] == 2 and title.lower().startswith("mitigation"):
            in_mitigations = True
        elif s["level"] == 3 and in_mitigations:
            mitigations.append({"title": title, "text": s["text"]})

    if not desc_text:  # robustness: no explicit Description heading
        desc_text = sections[0]["text"] if sections else doc.text

    for ci, piece in enumerate(recursive_split(desc_text, _effective_max(header), CHUNK_CONFIG["overlap_tokens"])):
        cid = parent_id if ci == 0 else f"{parent_id}-{ci}"
        chunk = _make_chunk(
            header=header, body=piece, doc=doc.doc, section=tid,
            source_type="attck", chunk_id=cid, extra_meta=base,
        )
        if chunk:
            chunks.append(chunk)

    # Children: one chunk per mitigation, linked to the parent.
    for mit in mitigations:
        body = (f"**{mit['title']}**\n{mit['text']}" if mit["title"] else mit["text"]).strip()
        section = f"{tid} / {mit['title']}" if mit["title"] else tid
        for ci, piece in enumerate(recursive_split(body, _effective_max(header), CHUNK_CONFIG["overlap_tokens"])):
            cid = f"{tid}::mit::{_slug(mit['title'])}-{ci}"
            chunk = _make_chunk(
                header=header, body=piece, doc=doc.doc, section=section,
                source_type="attck", chunk_id=cid, extra_meta=base, parent_id=parent_id,
            )
            if chunk:
                chunks.append(chunk)
    return chunks


# ==============================================================================
# SECTION 5 — NIST PDF chunker (recursive, heading-aware section tracking)
# ==============================================================================

_NIST_NUM_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,4})\.?\s+[A-Z]")
_CSF_CODE_RE = re.compile(r"\b([A-Z]{2}\.[A-Z]{2}-\d{2})\b")
_CSF_FUNC_RE = re.compile(r"\b(GOVERN|IDENTIFY|PROTECT|DETECT|RESPOND|RECOVER)\b")


def _detect_nist_section(text: str) -> Optional[str]:
    """Return the most relevant section identifier visible on a page, or None."""
    for line in text.splitlines()[:12]:
        m = _NIST_NUM_HEADING_RE.match(line)
        if m:
            return m.group(1)
    code = _CSF_CODE_RE.search(text)
    if code:
        return code.group(1)
    func = _CSF_FUNC_RE.search(text)
    if func:
        return func.group(1).title()
    return None


def _chunk_nist(docs: List[Document]) -> List[ChunkDict]:
    """Chunk NIST PDF pages, grouped by document and processed in page order so
    the current section can carry across pages that have no heading of their own."""
    chunks: List[ChunkDict] = []
    by_doc: Dict[str, List[Document]] = {}
    for d in docs:
        by_doc.setdefault(d.doc, []).append(d)

    for doc_name, pages in by_doc.items():
        pages.sort(key=lambda d: (d.page if d.page is not None else 1_000_000))
        current_section: Optional[str] = None
        # ``pi`` is the page's position in document order — it disambiguates the
        # chunk_id when two physical pages share a page number (front-matter
        # pages whose printed number was unrecovered fall back to the PDF index
        # and can collide with real body pages).
        for pi, page_doc in enumerate(pages):
            detected = _detect_nist_section(page_doc.text)
            if detected:
                current_section = detected
            section = current_section or f"p.{page_doc.page}"
            header = _nist_header(doc_name, section, page_doc.page)
            pieces = recursive_split(page_doc.text, _effective_max(header), CHUNK_CONFIG["overlap_tokens"])
            for ci, piece in enumerate(pieces):
                cid = f"{_slug(doc_name)}::p{page_doc.page}::{pi}-{ci}"
                chunk = _make_chunk(
                    header=header, body=piece, doc=doc_name, section=section,
                    source_type="nist_pdf", chunk_id=cid,
                    extra_meta={"printed_page": page_doc.metadata.get("printed_page")},
                    page=page_doc.page,
                )
                if chunk:
                    chunks.append(chunk)
    return chunks


# ==============================================================================
# SECTION 6 — Dispatch / orchestrator
# ==============================================================================

def chunk_document(doc: Document) -> List[ChunkDict]:
    """Chunk a single non-PDF Document (advisory or ATT&CK)."""
    if doc.source_type == "advisory":
        return _chunk_advisory(doc)
    if doc.source_type == "attck":
        return _chunk_attck(doc)
    raise ValueError(f"use chunk_corpus for source_type={doc.source_type!r}")


def chunk_corpus(documents: List[Document]) -> List[ChunkDict]:
    """Chunk every Document into contract-shaped ChunkDicts.

    Validates each chunk against ``contracts.validate_chunk`` when available;
    invalid chunks are dropped with a warning.
    """
    nist_docs = [d for d in documents if d.source_type == "nist_pdf"]
    chunks: List[ChunkDict] = []

    for d in documents:
        try:
            if d.source_type == "advisory":
                chunks.extend(_chunk_advisory(d))
            elif d.source_type == "attck":
                chunks.extend(_chunk_attck(d))
        except Exception:  # noqa: BLE001 - one bad doc must not kill the run
            logger.exception("failed to chunk %s", d.doc)

    try:
        chunks.extend(_chunk_nist(nist_docs))
    except Exception:  # noqa: BLE001
        logger.exception("failed to chunk NIST PDFs")

    if _HAVE_CONTRACT:
        valid = [c for c in chunks if validate_chunk(c)]
        dropped = len(chunks) - len(valid)
        if dropped:
            logger.warning("dropped %d chunks failing validate_chunk", dropped)
        chunks = valid

    logger.info("produced %d chunks from %d documents", len(chunks), len(documents))
    return chunks


# ==============================================================================
# SECTION 7 — Optional LLM contextual-enrichment hook (NIST only)
# ==============================================================================

def add_llm_context(
    chunks: List[ChunkDict],
    generate_fn,
    only_source_types: tuple = ("nist_pdf",),
) -> List[ChunkDict]:
    """OPTIONAL Tier-3 enhancement — turn on only if evaluation shows weak NIST
    retrieval (see README §6 and the "template vs LLM" decision).

    Prepends an LLM-generated 1-2 sentence context line to each targeted chunk,
    situating it within its document (Anthropic Contextual Retrieval). By default
    this only touches ``nist_pdf`` chunks, because advisory/ATT&CK chunks are
    already self-describing via their deterministic template headers — so we pay
    LLM cost/latency only where it measurably helps.

    Parameters
    ----------
    chunks:
        Output of :func:`chunk_corpus`.
    generate_fn:
        Callable ``(prompt: str) -> str`` wrapping the local model (e.g. Gemma)
        or any LLM. Kept injectable so this module has no hard LLM dependency.
    only_source_types:
        Which source types to enrich (default: NIST PDFs only).

    Returns
    -------
    The same list with enriched ``text`` on targeted chunks. Metadata is never
    LLM-derived — structured fields stay deterministic to avoid hallucinated
    security data.
    """
    prompt_tmpl = (
        "Document: {doc} (section {section}).\n"
        "Write ONE short sentence situating the following passage within the "
        "document, to improve search retrieval. Use only the passage; do not "
        "invent facts.\n\nPassage:\n{body}\n\nContext sentence:"
    )
    for c in chunks:
        if c["metadata"].get("source_type") not in only_source_types:
            continue
        try:
            ctx = generate_fn(
                prompt_tmpl.format(
                    doc=c["metadata"].get("doc", ""),
                    section=c["metadata"].get("section", ""),
                    body=c["text"][:1500],
                )
            ).strip()
            if ctx:
                c["text"] = f"{ctx}\n{c['text']}"
        except Exception:  # noqa: BLE001 - enrichment is best-effort
            logger.warning("LLM context failed for chunk %s", c["metadata"].get("chunk_id"))
    return chunks


# ==============================================================================
# SECTION 8 — CLI / inspection entry point
# ==============================================================================

def _summarize(chunks: List[ChunkDict]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        st = c["metadata"].get("source_type", "?")
        s = summary.setdefault(st, {"count": 0, "tok": 0, "max": 0})
        n = count_tokens(c["text"])
        s["count"] += 1
        s["tok"] += n
        s["max"] = max(s["max"], n)
    for s in summary.values():
        s["avg"] = round(s["tok"] / s["count"]) if s["count"] else 0
    return summary


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk the SecureOps corpus.")
    parser.add_argument("--corpus", default="new_corpus")
    parser.add_argument("--out", help="write chunks to this JSONL file")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    docs = load_corpus(args.corpus)
    chunks = chunk_corpus(docs)
    summary = _summarize(chunks)

    print(f"\nCorpus: {Path(args.corpus).resolve()}")
    print(f"Tokenizer: {tokenizer_name()}")
    print(f"Documents: {len(docs)}  ->  Chunks: {len(chunks)}\n")
    print(f"{'source_type':<12} {'chunks':>7} {'avg_tok':>9} {'max_tok':>9}")
    print("-" * 40)
    over = 0
    for src in ("nist_pdf", "advisory", "attck"):
        s = summary.get(src)
        if s:
            print(f"{src:<12} {s['count']:>7} {s['avg']:>9} {s['max']:>9}")
            if s["max"] > CHUNK_CONFIG["max_tokens"]:
                over += 1
    print(f"\nmax_tokens budget: {CHUNK_CONFIG['max_tokens']}  "
          f"({'OK — all chunks within budget' if over == 0 else 'WARNING: some chunks exceed budget'})")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(chunks)} chunks -> {Path(args.out).resolve()}")

    if args.sample:
        seen: set[str] = set()
        print("\n--- sample chunk per source ---")
        for c in chunks:
            st = c["metadata"].get("source_type")
            if st in seen:
                continue
            seen.add(st)
            print(f"\n[{st}] chunk_id={c['metadata'].get('chunk_id')}")
            print(f"  metadata: {c['metadata']}")
            print(f"  text[:280]: {c['text'][:280].replace(chr(10), ' ')} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
