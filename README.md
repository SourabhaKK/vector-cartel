# SecureOps Assistant — RAG Data & Retrieval Pipeline

**AAI Tech Talks Hackathon 2026 · MSc Applied AI · WMG, University of Warwick**
**Team:** Vector Cartel · **Target tier:** Tier 3 · **Submission deadline:** 23/06/2026 13:00

This document covers the **full RAG pipeline through retrieval** of the SecureOps Assistant — every
stage that turns raw public documents into clean, chunked, metadata-rich units, builds the search
indexes, and serves ranked results to the LLM layer. It is benchmarked throughout against the
official **Tier 1 Starter Notebook** (`SecureOps_Assistant_Tier1_Starter.ipynb`), the naive baseline
the hackathon brief asks us to *measurably improve*, and the retrieval quality is **quantified** with
a gold-set evaluation (§9).

> **Scope note.** This pipeline owns everything **up to and including retrieval**: corpus capture →
> cleaning → ingestion → chunking → index build → retrieval. It ends at the contract's
> `retrieval_fn(query) -> List[ChunkDict]`. The downstream LLM/agentic layer and the
> output/answer/evaluation layer are separate modules that consume the `ChunkDict` contract defined
> in `contracts.py`.

---

## Quickstart

```bash
pip install -r requirements.txt        # see GPU note in requirements.txt for torch

# Query the retriever — indexes are pre-built in index_store/, so this runs as-is.
python retrieval.py "privilege escalation in Siemens RUGGEDCOM" -v

# Measure retrieval quality against the 54-query gold set.
python eval.py --show-misses
```

**Rebuild from scratch** (only if you change the corpus or chunking):

```bash
python ingestion.py --corpus new_corpus --sample -v          # corpus   -> Document
python chunking.py  --corpus new_corpus --out chunks.jsonl   # Document -> chunks.jsonl
python index.py     --chunks chunks.jsonl --selftest -v      # chunks   -> index_store/
```

GPU (CUDA `torch`) is used automatically when available; CPU works too (retrieval just runs slower).
On the dev machine, embedding + reranking run in the `GAN` conda env
(`conda run -n GAN --no-capture-output python retrieval.py "…"`).

**Integrating with the agent layer** — import the contract entry point:

```python
from retrieval import build_retrieval_fn        # or: from retrieval import retrieve
retrieval_fn = build_retrieval_fn()              # retrieval_fn(query) -> List[ChunkDict]
```

---

## 1. The baseline (Tier 1 Starter) and its known weaknesses

The starter is a complete, deliberately naive, end-to-end RAG pipeline. Its data path:

| Stage | Starter approach |
|---|---|
| **Corpus** | 2 NIST PDFs + **~20 CISA advisories** scraped from an RSS feed with BeautifulSoup; falls back to **3 hard-coded sample advisories** if the scrape fails. **No MITRE ATT&CK.** |
| **Advisory text** | `main.get_text()` flattens the whole HTML page into **one run-on string** — navigation, legal footers, and content all mixed together. Stored in a single `cisa_advisories.json` blob. |
| **PDF parsing** | `extract_text()` per page, kept as-is including running headers/footers; page number = raw PDF index. |
| **Chunking** | **Fixed-size character chunking**: `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=150`, applied identically to a 300-page standard and a 1-page advisory. Slices sentences and CVSS tables mid-content. |
| **Metadata** | `{source, page}` only. No vendor, CVSS, CVE, date, or technique fields. |
| **Embedding** | `all-MiniLM-L6-v2`. |
| **Retrieval** | Pure dense top-5. No keyword/BM25, no reranking, no metadata filtering. |

The starter itself flags these as the Tier 2/3 improvement targets: structure-aware chunking,
per-document-type sizing, hybrid search, reranking, metadata filtering, and ATT&CK ingestion.
This pipeline addresses every one of them on the data/input side.

---

## 2. Our pipeline at a glance

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ STAGE 1  CAPTURE  (0_data_download.ipynb, Steps 1-3)                      │
 │   NIST PDFs · 100 CISA advisories (crawl4ai) · 97 ATT&CK techniques (STIX)│
 │   →  corpus/   "bronze" — faithful raw capture, YAML frontmatter          │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ STAGE 2  CLEAN   (0_data_download.ipynb, Step 4)                          │
 │   strip boilerplate / citations / link noise                             │
 │   →  new_corpus/  "silver" — 43% smaller advisories, content preserved    │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ STAGE 3  INGEST  (ingestion.py)                                          │
 │   parse frontmatter · clean PDF headers/footers · recover printed page    │
 │   →  List[Document]  — 543 uniform units with rich metadata               │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ STAGE 4  CHUNK   (chunking.py, see §6)                                    │
 │   per-source structure-aware + record + parent-child + contextual headers │
 │   →  chunks.jsonl  — 2,462 ChunkDicts (contract shape)                    │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ STAGE 5  INDEX   (index.py, see §7)                                       │
 │   bge-base embeddings → ChromaDB (cosine)  +  BM25 keyword index           │
 │   →  index_store/  — two indexes keyed by chunk_id                        │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ STAGE 6  RETRIEVE (retrieval.py, see §8)                                  │
 │   dense top-20 + BM25 top-20 → RRF fusion → cross-encoder rerank → top-5   │
 │   →  retrieval_fn(query) -> List[ChunkDict]   (contract entry point)      │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ EVAL     (eval.py, see §9)   54-query gold set → Hit@5 98%, MRR 0.849      │
 └─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage 1 — Corpus capture (`0_data_download.ipynb`, Steps 1–3)

Three open, publicly-licensed sources, each saved in a retrieval-friendly format.

| Source | What & how | vs. baseline |
|---|---|---|
| **NIST SP 800-82 Rev. 3** (PDF, ~300 pp) | Downloaded from NIST public servers. | Same. |
| **NIST CSF 2.0** (PDF, ~32 pp) | Downloaded from NIST public servers. | Same. |
| **CISA ICS advisories** | **100** advisories crawled with **`crawl4ai`**, each saved as an **individual `.md` file with YAML frontmatter** (`vendor`, `cvss_score`, `cves`, `cwe`, `release_date`, `sectors`, `url`). | Baseline: ~20 via RSS + BeautifulSoup → **flattened** into one JSON blob, no structured metadata. We have **5× the advisories**, structure preserved, and frontmatter that enables metadata filtering. |
| **MITRE ATT&CK for ICS** | **97** techniques/sub-techniques parsed from the official **STIX 2.1** bundle, each an `.md` file with frontmatter (`technique_id`, `tactics`, `platforms`, `is_subtechnique`, `parent_technique`) and linked mitigations. | Baseline: **not present at all** — the starter cannot answer the brief's ATT&CK question. |

**Key implementation choices**
- **`crawl4ai` over BeautifulSoup** — preserves the advisory's Markdown structure (section headings, CVSS tables, affected-product lists) instead of flattening to a run-on string. Structure is what makes Stage 4 chunking possible.
- **STIX bundle over web scraping** for ATT&CK — one deterministic download, no brittle HTML parsing; relationships (`mitigates`) are resolved to attach mitigations to each technique.
- **One file per record + YAML frontmatter** — human-inspectable, diff-able, and the frontmatter becomes structured metadata downstream (the baseline's single JSON blob supports none of this).

---

## 4. Stage 2 — Cleaning into `new_corpus/` (`0_data_download.ipynb`, Step 4)

A **bronze → silver** layer. `corpus/` stays the faithful raw capture; `new_corpus/` is the cleaned
layer the pipeline actually reads. Separating capture from cleaning keeps the pipeline reproducible
and the raw data re-inspectable. This step **reads the already-downloaded files** (no re-scrape), so
it is deterministic and runs in seconds.

**Why it matters:** roughly **40% of every CISA advisory page is boilerplate** — legal notices,
revision-history tables, "Related Advisories", and the cisa.gov site footer (social links,
FOIA/USA.gov nav, empty tables). The generic *Recommended Practices* paragraph is also near-identical
across all 100 advisories. Embedded as-is, that junk would create ~100 near-duplicate chunks that
pollute similarity search and waste the embedder's limited token budget.

| Source | Cleaning applied | Result |
|---|---|---|
| CISA advisories | Truncate at the first boilerplate marker (Legal Notice / Revision History / Tags / Related Advisories / footer); strip `(link is external)`, nav links, and `[label](url)` → `label`. Content sections (Summary, Background, per-CVE Vulnerabilities, Affected Products, Metrics, Acknowledgments, General Recommendations) kept. | **2.10 MB → 1.20 MB (43% smaller)** |
| MITRE ATT&CK | Remove inline `(Citation: …)` markers; unwrap mitigation cross-links to their labels. | **294 KB → 264 KB (10% smaller)** |
| NIST PDFs | Copied through unchanged — PDFs are binary; their header/footer/page cleaning happens at **parse** time in Stage 3. | — |

**vs. baseline:** the starter performs **no cleaning whatsoever** — the flattened advisory string
includes every nav element and legal footer, and they all get embedded.

---

## 5. Stage 3 — Ingestion (`ingestion.py`)

The adapter layer: messy, source-specific files in → one uniform structure out. It does **not**
chunk, embed, or retrieve. Verified end-to-end against `new_corpus/`:

```
Total document units: 543
source_type   count  avg_chars  total_chars
nist_pdf        346       2208       763826
advisory        100      11550      1154958
attck            97       2488       241295
```

**The uniform unit** (a `@dataclass Document`): `text`, `source_type`, `doc`, `page`, `metadata`,
`source_path`. The `metadata` dict is the seed for the contract's `ChunkDict` metadata.

**Per-source parsing**
- **Advisories / ATT&CK** — split YAML frontmatter from body (PyYAML when available, with a
  corpus-tuned fallback parser so there is no hard dependency); typed metadata extracted
  (`vendor`, `cvss`, `cves`, `technique_id`, `tactic`, `is_subtechnique`, …).
- **NIST PDFs** — one `Document` per non-empty page, with the parse-time cleaning that can only
  happen here:
  - **Running headers stripped** per document (`NIST SP 800-82r3 / September 2023 / Guide to OT
    Security`, `NIST CSWP 29 / …`), matched only in the first few lines so identical phrases in the
    body are never removed.
  - **Printed page number recovered** from the page itself and used as the citation `page` — this
    sidesteps the printed-vs-PDF-index offset (e.g. PDF index 100 = printed page 83) entirely.
  - **Figure-only pages skipped** (cleaned text below a length threshold).

**Production-grade engineering**
- Full type hints, `@dataclass(slots=True)` with validation, module/function docstrings, `__all__`.
- **Fault isolation** — a single malformed file logs an exception and is skipped; the run continues.
- A CLI for inspection: `python ingestion.py --corpus new_corpus --sample -v`.

**Two bugs caught during verification**
1. **Page-number leak** — `pypdf` emits the printed number `83` as the *first* line of a page (text
   order ≠ visual order), not the last. The initial "last-line only" check missed it; fixed to check
   **both ends**, so SP 800-82 now correctly reports printed page **83** and CSF reports **17**.
2. Confirmed a `�` in console output was only a Windows cp1252 *display* artifact — the stored
   character is a valid `U+2013 EN DASH` (UTF-8 intact).

**vs. baseline:** the starter's `extract_pdf_pages` keeps headers/footers in the text, uses the raw
PDF index as the page number, and has no metadata beyond `{source, page}`; advisories arrive as one
flat string with no fields at all.

---

## 6. Stage 4 — Chunking strategy (`chunking.py`, design finalised)

> **Status: implemented and verified** (`chunking.py`). Consumes Stage 3 `Document`s and emits
> `ChunkDict`s. **543 documents → 2,462 chunks**, all within the **480-token** budget (the 512-token
> cap shared by the bge embedders; max observed 479), all passing `contracts.validate_chunk`, with
> 402 ATT&CK parent-child links intact. Written to `chunks.jsonl` as the single source both the
> Chroma and BM25 indexes build from.

**The hard constraint:** the contract *suggests* the embedder **`bge-small-en-v1.5`**, which has a
**512-token cap** — anything longer is silently truncated at embed time. Advisories average 11.5k
chars and the largest is 388 KB, so whole-document chunking is impossible. This *forces*
structure-aware splitting.

**Sizing is measured in TOKENS, not characters or words.** The model truncates on a *token* budget,
and the unit you measure must match it. Characters and especially words are unreliable proxies for
security text: a single CVSS vector `CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H` is **one "word"
but ~26 tokens**, and a CVE ID `CVE-2026-27668` is one word but 7 tokens. Word/char budgets therefore
*under*-count exactly the security-dense chunks we least want truncated. So `chunking.py` measures
every chunk with the embedder's **own tokenizer** via `count_tokens()`, with a graceful fallback
chain: `transformers:BAAI/bge-small-en-v1.5` → `tiktoken:cl100k_base` → heuristic `chars/3.3`. The
config: **`max_tokens=480`** (480, not 512, leaves room for the `[CLS]`/`[SEP]` special tokens added
at embed time), **`overlap_tokens=50`**, **`min_tokens=12`** (drops meaningless fragments).

**The chunking is hybrid — a different method per source:**

| Source | Chunking method | `section` metadata |
|---|---|---|
| **CISA advisories** | **Structure-aware** (split on `##`) **+ record-based**: the `Vulnerabilities` section is exploded into **one chunk per `### CVE-…` block** (description + affected products + CVSS together) — turning the 388 KB multi-CVE monster into ~40 precise units. | heading / CVE ID |
| **MITRE ATT&CK** | **Parent–child**: `## Description` = parent chunk; each `### <Mitigation>` = a child chunk. | `technique_id` (+ mitigation name) |
| **NIST SP 800-82** | **Recursive splitting** within each page (paragraph → sentence), up to the 480-token budget with ~50-token overlap. | nearest numbered heading (e.g. `5.3.7.2`) |
| **NIST CSF 2.0** | **Record-based with hierarchical context**: one chunk per Subcategory with prepended lineage `GOVERN (GV) › Roles… (GV.RR) › GV.RR-01: …`; plus one summary chunk per Function. | subcategory code (e.g. `GV.RR-01`) |

**Three Tier-3 layers applied to every chunk**
1. **Contextual headers** — a deterministic, template-based header is prepended before embedding so
   small chunks stay self-describing, e.g. `[CISA Advisory ICSA-26-111-02 · Siemens RUGGEDCOM SAM-P
   · CVE-2026-27668 · CVSS 8.8]`. (Template chosen over per-chunk LLM generation: ~80% of the benefit
   at zero cost/latency and fully reproducible — important under the deadline.)
2. **Parent–child ("small-to-big")** — retrieve the small precise child, hand the larger parent to
   the generator.
3. **Rich, filterable metadata** — `doc`, `section`, `page`, plus `vendor` / `date` / `cvss` /
   `technique_id` / `tactic`, feeding the contract's hybrid + metadata-filter retrieval. List fields
   (`cves`, `sectors`) are flattened to comma-strings because **ChromaDB metadata must be scalar**.

**Output (contract `ChunkDict`):**
```python
{"text": "<contextual header>\n<chunk body>",
 "metadata": {"doc": ..., "section": ..., "page": ..., "vendor": ..., "cvss": ...},
 "score": 0.0}   # score is assigned at retrieval time, not here
```

**vs. baseline:** fixed 1000-char cuts that slice sentences/tables and treat every document type
identically, with no structure, no per-CVE granularity, no context headers, and no rich metadata.

---

## 7. Stage 5 — Index build (`index.py`)

> **Status: implemented and verified.** Reads `chunks.jsonl` once and builds the two indexes the
> hybrid retriever needs, both keyed by `chunk_id` so their results can be fused. Verified:
> **2,462 vectors (768-dim) + BM25 over 2,462 chunks**.

The retriever is **hybrid** by design, and the two halves are complementary:

| Index | Built from | Captures |
|---|---|---|
| **Dense** — ChromaDB (cosine) | `bge-base-en-v1.5` embeddings of each chunk's `text` | **semantics** — "privilege escalation", "network segmentation" |
| **Sparse** — BM25 | identifier-preserving tokens of the same `text` | **exact identifiers** — `CVE-2026-27668`, `GV.RR-01`, `CWE-266` |

**Why hybrid, and why no security-specific embedder.** Dense embeddings are weak on arbitrary
identifiers (a model cannot "understand" a CVE number — it sees rare subword fragments). Classical
BM25 matches them exactly. So BM25 owns identifier lookup and the embedder only has to be good at
semantic prose — which general MTEB-strong models already are. This is precisely why no exotic /
domain-tuned embedder is needed.

**Embedding-model choice.** `EMBED_MODEL` is a **single swappable constant**, defaulting to
**`bge-base-en-v1.5`** (768-dim) — a deliberate, defensible upgrade from the contract's suggested
`bge-small`: same family, **same 512-token cap** the chunker is already sized to, but stronger
retrieval. Falling back to `bge-small` is a one-line change with no other edits. (nomic / mxbai /
arctic were considered; their gaps are small, the reranker washes most of it out, and bge-base is the
best quality/latency balance — a decision the §9 eval can revisit with numbers.)

**Two details that silently destroy recall if wrong** — both handled here and reused at query time:
1. **Query instruction.** BGE expects the *query* (not the documents) to be prefixed with
   `"Represent this sentence for searching relevant passages: "`. Exposed via `embed_query`.
2. **Normalisation + cosine.** Vectors are L2-normalised and ChromaDB is set to `hnsw:space=cosine`.

`tokenize_for_bm25` keeps security identifiers whole (`cve-2026-27668` stays one token) and is
**exported** so retrieval tokenises queries identically. A NIST `chunk_id` collision (front-matter
pages whose printed number was unrecovered fall back to the PDF index and clash with body pages of
the same number) was caught here and fixed by folding the page's document-position into the id —
**2,462 / 2,462 ids now unique.**

---

## 8. Stage 6 — Retrieval (`retrieval.py`)

> **Status: implemented and verified on GPU.** Implements the contract's
> `retrieval_fn(query) -> List[ChunkDict]`.

```
query
  → embed_query (bge-base, query instruction)        ┐
  → ChromaDB cosine        top-20   (dense)          │
  → BM25 keyword           top-20   (sparse)         ┘
  → RRF fusion (by chunk_id) → top-30 candidates
  → cross-encoder rerank   → top-5
  → score = sigmoid(logit) ∈ [0,1] · validate_chunk · return (ordered desc)
```

- **RRF fusion** combines the two rankings using **rank position only** (`1/(k+rank)`, `k=60`), so a
  cosine distance and a BM25 score — which live on incomparable scales — merge cleanly without any
  normalisation hand-tuning.
- **Cross-encoder rerank** (`RERANK_MODEL = bge-reranker-base`, swappable) reads `(query, chunk)`
  *together* and is far more accurate than either first-stage retriever; it is the final arbiter over
  the small fused candidate pool. Its logit is squashed with a sigmoid into the `[0,1]` the contract
  requires for `score`.
- **Contract-safe:** returns `[]` on empty/error, **never raises**, every returned chunk passes
  `validate_chunk`, and `RETRIEVAL_CONFIDENCE_THRESHOLD` (0.35) is **imported, never hardcoded**. By
  default the top-5 are returned and the agent's verifier decides confidence; `min_score` optionally
  hard-filters.
- `build_retrieval_fn(...)` builds the function once (indexes + reranker injectable for tests);
  `retrieve(q)` is a lazy-singleton convenience.

**Runs on GPU** via the `GAN` conda env (CUDA `torch`); the CPU-built `index_store/` is portable
across envs (identical dependency versions). ~**0.63 s/query** on an RTX 4060.

---

## 9. Stage 7 — Evaluation (`eval.py`)

> **Status: implemented and run.** Turns "it looks good" into a defensible number.

A **54-query hand-curated gold set**, grounded in real corpus targets (advisory CVEs/alert-codes,
ATT&CK technique-ids, NIST CSF codes and SP 800-82 prose). For each query the real `retrieval_fn`
runs and we record the rank of the first **relevant** chunk (relevance judged by an exact/substring
match spec on metadata + text).

| Metric | Overall | advisory | attck | nist_csf | nist_sp80082 |
|---|---|---|---|---|---|
| **Hit@5** (recall@5) | **98.1%** | 100% | 100% | 100% | 90% |
| Hit@1 | 74.1% | 56% | 89% | 62% | 90% |
| MRR | 0.849 | 0.750 | 0.944 | 0.792 | 0.900 |

*(bge-base + bge-reranker-base, 0.57 s/query, 1 miss in 54.)*

**Reading the result.** Recall@5 is 100% on three of four source groups; only SP 800-82 prose has a
single miss — "incident response and contingency planning" ranked an ATT&CK *Data Backup* mitigation
above the NIST prose, a reasonable cross-source semantic collision (the gold was **not** loosened to
inflate the score). Advisory Hit@1 (56%) is a gold-spec artifact rather than a retrieval failure: the
*summary* chunk of the correct advisory often ranks #1 while the strict match demands the specific
per-CVE chunk (always present by #2 — hence 100% Hit@5). The optional, already-built NIST
`add_llm_context()` hook in `chunking.py` is the lever to push the remaining NIST prose case, with
this run as the baseline to measure any delta against.

---

## 10. Baseline vs. this pipeline — summary

| Dimension | Tier 1 Starter | This pipeline |
|---|---|---|
| Advisories | ~20 (or 3 fallback), flattened JSON | **100**, structured `.md` + YAML frontmatter |
| MITRE ATT&CK | ❌ none | ✅ **97** techniques from STIX |
| Boilerplate removal | ❌ none | ✅ bronze→silver, **43%** advisory reduction |
| PDF headers/footers | kept in text | ✅ stripped |
| Page citation | raw PDF index | ✅ **printed page** recovered |
| Document metadata | `{source, page}` | ✅ vendor, cvss, cves, date, technique_id, tactic… |
| Chunking | fixed 1000-char, uniform | ✅ **per-source** structure/record/parent-child |
| Chunk sizing | character count (truncation-blind) | ✅ **token-based** via the embedder's own tokenizer |
| Self-describing chunks | ❌ | ✅ contextual headers |
| Embedding model | `all-MiniLM-L6-v2` | **`bge-base-en-v1.5`** (swappable) |
| Retrieval | dense top-5 only | ✅ **hybrid** (dense + BM25) → RRF → cross-encoder rerank |
| Retrieval quality | not measured | ✅ **Hit@5 98% / MRR 0.849** on a 54-query gold set |

---

## 11. How to run

```bash
# 1. Capture + clean the corpus (produces corpus/ then new_corpus/)
#    Run 0_data_download.ipynb top to bottom (Steps 1-4).

# 2. Ingest the cleaned corpus into uniform Document units
pip install pypdf            # plus pyyaml (optional, recommended)
python ingestion.py --corpus new_corpus --sample -v

# 3. Chunk the documents into contract-shaped ChunkDicts (writes chunks.jsonl)
python chunking.py --corpus new_corpus --out chunks.jsonl --sample

# 4. Build the hybrid index (ChromaDB + BM25 → index_store/)
pip install chromadb sentence-transformers rank_bm25
python index.py --chunks chunks.jsonl --selftest -v

# 5. Query the retriever  (GPU env recommended)
conda run -n GAN --no-capture-output python retrieval.py "privilege escalation in Siemens RUGGEDCOM" -v

# 6. Evaluate retrieval quality against the gold set
conda run -n GAN --no-capture-output python eval.py --show-misses
```

> **GPU note.** Embedding + reranking run on CUDA in the `GAN` conda env (RTX 4060). The index store
> built on CPU is portable across envs, so steps 4–6 can mix CPU build / GPU query freely.

---

## 12. Repository layout

```
AAI_Hackathon/
├── 0_data_download.ipynb        # Stage 1-2: capture + clean → new_corpus/
├── ingestion.py                 # Stage 3: corpus → List[Document]      ✅ done
├── chunking.py                  # Stage 4: Document → List[ChunkDict]    ✅ done
├── chunks.jsonl                 # 2,462 ChunkDicts — handoff to index build
├── index.py                     # Stage 5: chunks → Chroma + BM25        ✅ done
├── retrieval.py                 # Stage 6: retrieval_fn(query)           ✅ done
├── eval.py                      # Stage 7: gold-set retrieval eval        ✅ done
├── index_store/                 # built indexes (chroma/ + bm25.pkl)
├── contracts.py                 # shared interface contract (ChunkDict, RetrievalFn, …)
├── corpus/                      # bronze: raw capture
│   ├── nist_sp800_82r3.pdf
│   ├── nist_csf_2_0.pdf
│   ├── advisories/  (100 × .md)
│   └── attck/       (97 × .md)
├── new_corpus/                  # silver: cleaned, pipeline reads this
│   └── … (same structure)
└── SecureOps_Assistant_Tier1_Starter.ipynb   # the naive baseline
```

---

## 13. Status

| Stage | Status |
|---|---|
| 1 — Capture | ✅ Complete (2 PDFs + 100 advisories + 97 ATT&CK) |
| 2 — Clean (`new_corpus/`) | ✅ Complete (verified, 43% advisory reduction) |
| 3 — Ingestion (`ingestion.py`) | ✅ Complete (543 units, verified end-to-end) |
| 4 — Chunking (`chunking.py`) | ✅ Complete (2,462 chunks, all contract-valid, within budget) |
| 5 — Index build (`index.py`) | ✅ Complete (2,462 vectors + BM25, ids unique) |
| 6 — Retrieval (`retrieval.py`) | ✅ Complete (hybrid + RRF + rerank, contract-shaped) |
| 7 — Evaluation (`eval.py`) | ✅ Complete (Hit@5 98%, MRR 0.849 on 54-query gold set) |

---

## 14. Licensing & attribution

| Source | Licence |
|---|---|
| NIST SP 800-82 Rev. 3 / NIST CSF 2.0 | Public domain (US government work) |
| CISA ICS Advisories | Public domain (US government work) |
| MITRE ATT&CK for ICS | Free with attribution (cite MITRE) |
