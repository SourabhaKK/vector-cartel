"""
ingestion.py — SecureOps Assistant · corpus ingestion layer
================================================================================
Vector Cartel · AAI Tech Talks Hackathon 2026 · WMG, University of Warwick

PURPOSE
-------
Turn the cleaned "silver" corpus on disk (``new_corpus/``) into a uniform,
in-memory list of :class:`Document` units that the chunking stage can process
without caring about source-specific file formats.

This is the *adapter* layer of the RAG pipeline:

    new_corpus/ (files)  ──►  ingestion.py  ──►  List[Document]  ──►  chunking.py

It loads and parses three source types, each with its own quirks:

    * CISA ICS advisories  (Markdown + YAML frontmatter)
    * MITRE ATT&CK for ICS (Markdown + YAML frontmatter)
    * NIST PDFs            (SP 800-82 Rev. 3, CSF 2.0)

WHAT IT DOES                         | WHAT IT DOES NOT DO
-------------------------------------|------------------------------------------
parse frontmatter -> metadata        | chunk text into ChunkDicts
extract PDF text page-by-page        | embed / build vector or BM25 indexes
strip PDF running headers/footers    | rank or retrieve
recover the *printed* page number    | call any LLM
normalize everything to Document     | mutate files on disk

The ``metadata`` dict on every Document is the seed for the downstream
``ChunkDict`` metadata defined in ``contracts.py`` (doc / section / page +
optional vendor / date / cvss / technique_id / tactic). Sections are assigned
later, at chunk time, because they depend on the chunking strategy.

USAGE
-----
    from ingestion import load_corpus
    docs = load_corpus("new_corpus")          # List[Document]

    # or from the command line, to inspect the corpus:
    python ingestion.py --corpus new_corpus --sample
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("secureops.ingestion")

__all__ = [
    "Document",
    "load_corpus",
    "load_advisories",
    "load_attck",
    "load_nist_pdfs",
    "parse_frontmatter",
]

# Optional dependency: PyYAML gives us the most robust frontmatter parsing.
# If it is not installed we fall back to a small parser tuned to this corpus.
try:
    import yaml  # type: ignore

    _HAVE_YAML = True
except ImportError:  # pragma: no cover - environment dependent
    _HAVE_YAML = False


# ==============================================================================
# SECTION 1 — The uniform document unit
# ==============================================================================

@dataclass(slots=True)
class Document:
    """A single logical document unit handed from ingestion to chunking.

    Attributes
    ----------
    text:
        Full cleaned text of this unit. For Markdown sources this is the whole
        body; for PDFs it is one page of cleaned text.
    source_type:
        One of ``"advisory"``, ``"attck"``, ``"nist_pdf"``.
    doc:
        Human-readable document name -> becomes ``ChunkDict.metadata["doc"]``,
        e.g. ``"NIST SP 800-82 Rev. 3"`` or ``"CISA Advisory ICSA-26-111-02"``.
    page:
        Printed page number for PDFs, or ``None`` for Markdown sources.
    metadata:
        Source-specific fields (vendor, cvss_score, cves, technique_id, ...).
        Carried forward verbatim and selected/flattened by the chunker.
    source_path:
        Absolute path of the originating file, for traceability and debugging.
    """

    text: str
    source_type: str
    doc: str
    page: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Document.text must be a str")
        if self.source_type not in {"advisory", "attck", "nist_pdf"}:
            raise ValueError(f"unknown source_type: {self.source_type!r}")

    @property
    def char_len(self) -> int:
        return len(self.text)


# ==============================================================================
# SECTION 2 — Frontmatter parsing (shared by the two Markdown sources)
# ==============================================================================

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _coerce_scalar(value: str) -> Any:
    """Convert a raw frontmatter scalar string into a typed Python value."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    low = value.lower()
    if low in {"", "null", "none", "~"}:
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    # int / float
    try:
        if re.fullmatch(r"-?\d+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def _parse_frontmatter_fallback(block: str) -> Dict[str, Any]:
    """Minimal YAML-frontmatter parser tuned to this corpus.

    Handles ``key: value`` scalars and block lists of the form::

        key:
          - item1
          - item2

    Used only when PyYAML is unavailable.
    """
    data: Dict[str, Any] = {}
    current_list_key: Optional[str] = None
    for raw_line in block.splitlines():
        if not raw_line.strip():
            continue
        list_item = re.match(r"\s+-\s+(.*)$", raw_line)
        if list_item and current_list_key is not None:
            data[current_list_key].append(_coerce_scalar(list_item.group(1)))
            continue
        kv = re.match(r"([\w.\-]+):\s*(.*)$", raw_line)
        if not kv:
            continue
        key, val = kv.group(1), kv.group(2)
        if val.strip() == "":
            data[key] = []
            current_list_key = key
        else:
            data[key] = _coerce_scalar(val)
            current_list_key = None
    return data


def parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    """Split a Markdown file into ``(frontmatter_dict, body)``.

    Returns an empty dict and the original text if no frontmatter is present.
    Never raises on malformed frontmatter — it logs and degrades gracefully.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    block, body = match.group(1), text[match.end():]
    if _HAVE_YAML:
        try:
            parsed = yaml.safe_load(block)
            if isinstance(parsed, dict):
                return parsed, body
            logger.warning("frontmatter did not parse to a dict; using fallback")
        except yaml.YAMLError as exc:  # pragma: no cover - depends on data
            logger.warning("PyYAML failed (%s); using fallback parser", exc)
    return _parse_frontmatter_fallback(block), body


# ==============================================================================
# SECTION 3 — CISA advisory loader
# ==============================================================================

def _load_advisory_file(path: Path) -> Optional[Document]:
    text = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    if not body.strip():
        logger.warning("advisory has empty body, skipping: %s", path.name)
        return None

    alert_code = str(meta.get("alert_code") or path.stem)
    title = meta.get("title") or alert_code

    metadata: Dict[str, Any] = {
        "alert_code": alert_code,
        "title": title,
        "vendor": meta.get("vendor") or None,
        "date": meta.get("release_date") or None,
        "cvss": meta.get("cvss_score"),
        "cvss_version": meta.get("cvss_version") or None,
        "cves": meta.get("cves") or [],
        "cwe": meta.get("cwe") or [],
        "sectors": meta.get("sectors") or [],
        "countries": meta.get("countries") or None,
        "url": meta.get("url") or None,
    }
    return Document(
        text=body.strip(),
        source_type="advisory",
        doc=f"CISA Advisory {alert_code}",
        page=None,
        metadata=metadata,
        source_path=str(path.resolve()),
    )


def load_advisories(corpus_dir: Path) -> List[Document]:
    """Load every ``*.md`` under ``<corpus_dir>/advisories``."""
    adv_dir = corpus_dir / "advisories"
    docs: List[Document] = []
    if not adv_dir.is_dir():
        logger.warning("no advisories directory at %s", adv_dir)
        return docs
    for path in sorted(adv_dir.glob("*.md")):
        try:
            doc = _load_advisory_file(path)
            if doc is not None:
                docs.append(doc)
        except Exception:  # noqa: BLE001 - one bad file must not kill the run
            logger.exception("failed to load advisory %s", path.name)
    logger.info("loaded %d advisories", len(docs))
    return docs


# ==============================================================================
# SECTION 4 — MITRE ATT&CK loader
# ==============================================================================

def _load_attck_file(path: Path) -> Optional[Document]:
    text = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    if not body.strip():
        logger.warning("attck technique has empty body, skipping: %s", path.name)
        return None

    technique_id = str(meta.get("technique_id") or path.stem.replace("-", "."))
    name = meta.get("name") or technique_id
    tactics = meta.get("tactics") or []

    metadata: Dict[str, Any] = {
        "technique_id": technique_id,
        "name": name,
        "tactics": tactics,
        "tactic": ", ".join(tactics) if tactics else None,  # contract scalar field
        "tactic_ids": meta.get("tactic_ids") or [],
        "platforms": meta.get("platforms") or [],
        "is_subtechnique": bool(meta.get("is_subtechnique", False)),
        "parent_technique": meta.get("parent_technique") or None,
        "url": meta.get("url") or None,
    }
    return Document(
        text=body.strip(),
        source_type="attck",
        doc=f"MITRE ATT&CK {technique_id}",
        page=None,
        metadata=metadata,
        source_path=str(path.resolve()),
    )


def load_attck(corpus_dir: Path) -> List[Document]:
    """Load every ``*.md`` under ``<corpus_dir>/attck``."""
    attck_dir = corpus_dir / "attck"
    docs: List[Document] = []
    if not attck_dir.is_dir():
        logger.warning("no attck directory at %s", attck_dir)
        return docs
    for path in sorted(attck_dir.glob("*.md")):
        try:
            doc = _load_attck_file(path)
            if doc is not None:
                docs.append(doc)
        except Exception:  # noqa: BLE001
            logger.exception("failed to load attck technique %s", path.name)
    logger.info("loaded %d ATT&CK techniques", len(docs))
    return docs


# ==============================================================================
# SECTION 5 — NIST PDF loader
# ==============================================================================
# Each registered PDF carries its display name and the running-header patterns
# that appear at the top of every page. We strip those headers and the footer
# page number; the footer number is the *printed* page, which we keep as the
# citation page (avoids the printed-vs-PDF-index offset problem entirely).

_PDF_REGISTRY: Dict[str, Dict[str, Any]] = {
    "nist_sp800_82r3.pdf": {
        "doc": "NIST SP 800-82 Rev. 3",
        "headers": [
            re.compile(r"^NIST\s+SP\s+800-82r3", re.IGNORECASE),
            re.compile(r"Guide to Operational Technology", re.IGNORECASE),
            re.compile(r"^September\s+2023$", re.IGNORECASE),
        ],
    },
    "nist_csf_2_0.pdf": {
        "doc": "NIST CSF 2.0",
        "headers": [
            re.compile(r"^NIST\s+CSWP\s+29", re.IGNORECASE),
            re.compile(r"The NIST Cybersecurity Framework", re.IGNORECASE),
            re.compile(r"^February\s+26,\s+2024$", re.IGNORECASE),
        ],
    },
}

_PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")


def _clean_pdf_page(raw: str, header_patterns: List[re.Pattern]) -> tuple[str, Optional[int]]:
    """Strip running header lines + footer page number from one PDF page.

    Returns ``(cleaned_text, printed_page_number_or_None)``.
    """
    lines = raw.splitlines()

    # Drop running-header lines, but only in the first few lines of the page,
    # so an identical phrase appearing in the body is never removed.
    kept: List[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if idx < 5 and stripped and any(p.search(stripped) for p in header_patterns):
            continue
        kept.append(line)

    # The printed page number is a line of pure digits sitting at the very top
    # or very bottom of the page. pypdf's text order does not always match the
    # visual order, so the footer number can surface as the first OR last line —
    # check both ends and strip it (and keep it as the citation page).
    page_number: Optional[int] = None
    nonempty = [i for i, line in enumerate(kept) if line.strip()]
    for cand in ((nonempty[-1], nonempty[0]) if nonempty else ()):
        if _PAGE_NUMBER_RE.fullmatch(kept[cand].strip()):
            page_number = int(kept[cand].strip())
            kept[cand] = ""  # blanked; trailing/leading blanks removed on strip()
            break

    text = "\n".join(kept)
    text = re.sub(r"[ \t]+\n", "\n", text)       # trailing whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, page_number


def _load_pdf_file(path: Path, min_chars: int) -> List[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pypdf is required to load NIST PDFs — `pip install pypdf`"
        ) from exc

    reg = _PDF_REGISTRY.get(path.name)
    doc_name = reg["doc"] if reg else path.stem
    headers = reg["headers"] if reg else []

    reader = PdfReader(str(path))
    docs: List[Document] = []
    for index, pdf_page in enumerate(reader.pages, start=1):
        try:
            raw = pdf_page.extract_text() or ""
        except Exception:  # noqa: BLE001 - some pages fail to extract
            logger.warning("text extraction failed on %s page %d", path.name, index)
            continue
        text, printed_page = _clean_pdf_page(raw, headers)
        if len(text) < min_chars:
            continue  # near-empty / full-page figure — nothing useful to embed
        docs.append(
            Document(
                text=text,
                source_type="nist_pdf",
                doc=doc_name,
                page=printed_page if printed_page is not None else index,
                metadata={
                    "doc_file": path.name,
                    "pdf_index": index,
                    "printed_page": printed_page,
                },
                source_path=str(path.resolve()),
            )
        )
    logger.info("loaded %d text pages from %s", len(docs), path.name)
    return docs


def load_nist_pdfs(corpus_dir: Path, min_chars: int = 50) -> List[Document]:
    """Load all ``*.pdf`` directly under ``<corpus_dir>``, one Document per page.

    Pages whose cleaned text is shorter than ``min_chars`` (figure-only pages,
    blank pages) are skipped.
    """
    docs: List[Document] = []
    for path in sorted(corpus_dir.glob("*.pdf")):
        try:
            docs.extend(_load_pdf_file(path, min_chars))
        except Exception:  # noqa: BLE001
            logger.exception("failed to load PDF %s", path.name)
    return docs


# ==============================================================================
# SECTION 6 — Orchestrator
# ==============================================================================

def load_corpus(corpus_dir: str | Path = "new_corpus", min_pdf_chars: int = 50) -> List[Document]:
    """Load the entire corpus into a single list of :class:`Document`.

    Parameters
    ----------
    corpus_dir:
        Path to the cleaned corpus directory (default ``"new_corpus"``).
    min_pdf_chars:
        Minimum cleaned-text length for a PDF page to be kept.

    Raises
    ------
    FileNotFoundError:
        If ``corpus_dir`` does not exist.
    """
    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"corpus directory not found: {root.resolve()} "
            f"(run the data-download notebook through Step 4 first)"
        )

    docs: List[Document] = []
    docs.extend(load_nist_pdfs(root, min_chars=min_pdf_chars))
    docs.extend(load_advisories(root))
    docs.extend(load_attck(root))

    if not docs:
        logger.warning("no documents loaded from %s", root.resolve())
    else:
        logger.info("corpus loaded: %d total document units", len(docs))
    return docs


# ==============================================================================
# SECTION 7 — CLI / inspection entry point
# ==============================================================================

def _summarize(docs: List[Document]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for d in docs:
        s = summary.setdefault(d.source_type, {"count": 0, "chars": 0})
        s["count"] += 1
        s["chars"] += d.char_len
    for s in summary.values():
        s["avg_chars"] = round(s["chars"] / s["count"]) if s["count"] else 0
    return summary


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest the SecureOps corpus.")
    parser.add_argument("--corpus", default="new_corpus", help="corpus directory")
    parser.add_argument("--min-pdf-chars", type=int, default=50)
    parser.add_argument("--sample", action="store_true", help="print one sample per source")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    docs = load_corpus(args.corpus, min_pdf_chars=args.min_pdf_chars)
    summary = _summarize(docs)

    print(f"\nCorpus: {Path(args.corpus).resolve()}")
    print(f"Total document units: {len(docs)}\n")
    print(f"{'source_type':<12} {'count':>6} {'avg_chars':>10} {'total_chars':>12}")
    print("-" * 44)
    for src in ("nist_pdf", "advisory", "attck"):
        s = summary.get(src)
        if s:
            print(f"{src:<12} {s['count']:>6} {s['avg_chars']:>10} {s['chars']:>12}")

    if args.sample:
        seen: set[str] = set()
        print("\n--- sample document per source ---")
        for d in docs:
            if d.source_type in seen:
                continue
            seen.add(d.source_type)
            print(f"\n[{d.source_type}] doc={d.doc!r} page={d.page}")
            print(f"  metadata keys: {sorted(d.metadata)}")
            preview = d.text[:240].replace("\n", " ")
            print(f"  text[:240]: {preview} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
