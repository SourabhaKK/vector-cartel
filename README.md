markdown<div align="center">

# 🔒 Vector Cartel — SecureOps Assistant

**RAG-based Question Answering for Industrial Cybersecurity**

*AAI Tech Talks Hackathon 2026 · WMG, University of Warwick*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-FF6B35?style=flat)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-6B4FBB?style=flat)
![Gemini](https://img.shields.io/badge/Gemini_1.5_Flash-LLM-4285F4?style=flat&logo=google&logoColor=white)
![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-2EA44F?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

</div>

---

## What Is This?

SecureOps Assistant helps junior security analysts at manufacturing companies navigate industrial cybersecurity guidance — NIST standards, CISA advisories, and MITRE ATT&CK for ICS — without reading hundreds of pages manually.

**The problem:** A hallucinated security recommendation in an OT environment is worse than no recommendation at all. A wrong firewall rule can take down a production line.

**Our solution:** A Retrieval-Augmented Generation (RAG) system that grounds every answer in a source document, cites every claim, and says *"I don't have enough information"* rather than guessing.

**What makes us different:**
- Structure-aware chunking per document type — not naive 512-token sliding windows
- Hybrid BM25 + dense retrieval with cross-encoder reranking — finds exact CVE IDs *and* semantically related content
- LangGraph agent that classifies, decomposes, and routes queries — handles multi-hop cross-document questions (CISA → MITRE → NIST in one answer)
- Prompt injection defence via spotlighting + chunk scanning
- RAGAS evaluation with baseline comparison — evidence, not vibes

---

## Team

| Member | GitHub | Branch | Responsibility |
|--------|--------|--------|----------------|
| Sourabha Kallapur | [@SK](https://github.com/SK) | `llm-and-agentic` | LLM integration and Agentic Layer |
| Jay Sadhu | [@Jay-Sadhu](https://github.com/Jay-Sadhu) | `rag-layer` | RAG Layer |
| Sana Shikalgar | [@Sana-Shikalgar](https://github.com/Sana-Shikalgar) | `rag-layer` | RAG Layer |
| Kaveen Prabodhya | [@kaveenprabodhya](https://github.com/kaveenprabodhya) | `output-layer` | Output Layer |

---

## Architecture Overview
OFFLINE (run once)

──────────────────────────────────────────────────────

NIST SP 800-82 ──┐

NIST CSF 2.0   ──┤─► Parser ─► Structure-Aware Chunker ─► bge-small Embedder

CISA Advisories──┤                                              │         │

MITRE ATT&CK  ──┘                                        ChromaDB     BM25 Index

(Google Drive persistent)
ONLINE (per query)

──────────────────────────────────────────────────────

User Query

│

┌───▼───────────────┐

│  Input Scanner     │  ── blocks direct prompt injection

└───┬───────────────┘

│

┌───▼───────────────────────────────────────────┐

│  LangGraph Agent                              │

│  ┌─────────────┐   simple  ──► Single RAG     │

│  │  Classifier │   multi_hop ► Decomposer     │

│  │             │   ambiguous ► Clarify         │

│  └─────────────┘   out_scope ► Refuse          │

└───────────────────────────────────────────────┘

│

┌───────────▼───────────────────────────────┐

│  Dual Retrieval (parallel)                │

│  Dense (ChromaDB) + BM25 ─► RRF ─► Rerank│

└───────────────────────────────────────────┘

│

┌───────────▼──────────────┐

│  Chunk Scanner            │  ── blocks poisoned corpus docs

└───────────────────────────┘

│

┌───────────▼──────────────────────────────────────┐

│  Gemini 1.5 Flash                                │

│  System prompt: citation rules + spotlighting    │

│  Fallback: Mistral-7B via HuggingFace            │

└──────────────────────────────────────────────────┘

│

┌───────────▼──────────────┐

│  Output Validator         │  ── groundedness check (token overlap)

└───────────────────────────┘

│

┌───────────▼─────────────────────────────────────┐

│  Structured Answer                               │

│  { answer, citations[], confidence, refusal }    │

└──────────────────────────────────────────────────┘

---

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| PDF parsing | `pdfplumber` | Preserves NIST table structure |
| STIX parsing | `stix2` | Official MITRE ATT&CK parser |
| HTML parsing | `requests` + `BeautifulSoup` | Structured CISA advisory field extraction |
| Embedding model | `BAAI/bge-small-en-v1.5` | Outperforms MiniLM on BEIR benchmarks (Zhang et al., 2023) |
| Vector store | `ChromaDB` (persistent) | Survives Colab session resets via Google Drive mount |
| Sparse index | `rank_bm25` | Exact match for CVE IDs, technique IDs, vendor names |
| Fusion | Custom RRF (k=60) | Proven to outperform weighted score merging (Cormack et al., 2009) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Joint query-chunk scoring, ~80ms on T4 |
| Agent | `LangGraph` | Stateful conditional routing + retry loops |
| LLM (primary) | `Gemini 1.5 Flash` | 1M context, fast, free tier |
| LLM (fallback) | `Mistral-7B-Instruct` via HuggingFace | Automatic fallback on Gemini 429 error |
| Evaluation | `RAGAS` | Faithfulness, Answer Relevance, Context Precision, Context Recall |
| Interface | `Gradio` | ChatInterface demo for submission video |

---

## Quick Start

### 1. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vector-cartel/vector-cartel/blob/main/notebooks/secureops_pipeline.ipynb)

### 2. Mount Google Drive

The first cell mounts Google Drive. This is **mandatory** — ChromaDB persists vectors to Drive. Without it, all vectors are lost on session reset.

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Add Your Gemini API Key

In Colab: **Secrets** (🔑 icon, left sidebar) → **Add new secret**
- Name: `GEMINI_API_KEY`
- Value: your key from [aistudio.google.com](https://aistudio.google.com)

Never hardcode API keys in the notebook.

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download the Corpus

```bash
bash corpus/download_corpus.sh
```

This fetches NIST SP 800-82 Rev 3, NIST CSF 2.0, CISA ICS advisories (50–100), and MITRE ATT&CK ICS STIX bundle.

### 6. Run All Cells in Order

The notebook is structured in order:
1. Corpus ingestion + chunking
2. Embedding + indexing
3. Retrieval pipeline
4. LangGraph agent + LLM
5. Security layer
6. Gradio demo interface

---

## Corpus Sources

| Source | URL | Format | What It Covers |
|--------|-----|--------|----------------|
| NIST SP 800-82 Rev. 3 | [csrc.nist.gov](https://csrc.nist.gov/pubs/sp/800/82/r3/final) | PDF ~300pp | Core OT security reference |
| NIST CSF 2.0 | [nist.gov/cyberframework](https://www.nist.gov/cyberframework) | PDF ~60pp | Security framework |
| CISA ICS Advisories | [cisa.gov/advisories](https://www.cisa.gov/news-events/cybersecurity-advisories) | HTML structured | Real-world vulnerability bulletins |
| MITRE ATT&CK for ICS | [attack.mitre.org/matrices/ics](https://attack.mitre.org/matrices/ics) | STIX2 JSON | Adversary techniques and tactics |

All sources are public domain (US government works) or openly licensed with attribution (MITRE).

---

## Evaluation Results

| Metric | Baseline (starter notebook) | Vector Cartel pipeline | Delta |
|--------|----------------------------|------------------------|-------|
| Faithfulness | TBD | TBD | TBD |
| Answer Relevance | TBD | TBD | TBD |
| Context Precision | TBD | TBD | TBD |
| Context Recall | TBD | TBD | TBD |

*Results measured on a 20-question gold QA test set. 5 simple factual · 5 multi-hop cross-document · 5 honesty tests · 5 adversarial injection attempts.*

---

## Branch Structure
main                  ← stable, always runnable — judges use this

dev                   ← integration branch, all members merge here first

├── rag-layer         ← Jay Sadhu + Sana Shikalgar

├── llm-and-agentic   ← SK

└── output-layer      ← Kaveen Prabodhya

**Rules:**
- Never push directly to `main`
- Always merge to `dev` first and verify end-to-end before merging `dev` → `main`
- Each member owns their branch — coordinate on shared interfaces (metadata schema, answer schema) via `dev`

---

## Commit Message Guide
<type>(<scope>): <short description>
[optional body — what and why, not how]

**Types:**

| Type | When to use |
|------|-------------|
| `feat` | New functionality added |
| `fix` | Bug fixed |
| `refactor` | Code restructured, no behaviour change |
| `eval` | Evaluation results or test set updated |
| `docs` | README, comments, or documentation |
| `chore` | Requirements, configs, setup |
| `security` | Security layer changes |

**Good examples:**
feat(chunking): add section-boundary chunker for NIST SP 800-82
Splits document at numbered section headers (\n\d+.\d+).

Each chunk carries section_id, section_title, parent_section, page metadata.

Validated: no chunk exceeds 512 tokens, no mid-sentence cuts.

feat(agent): add query classifier node with 4-way routing
Gemini Flash structured output returns simple/multi_hop/out_of_scope/ambiguous.

Refusal node triggers before any retrieval on out_of_scope queries.

Cost: ~100 tokens per classification call.

fix(retrieval): fix RRF score accumulation for duplicate chunk IDs
Chunks appearing in both dense and BM25 results were being overwritten

instead of having scores summed. Fixed with dict .get(id, 0) + score.

eval(ragas): add baseline scores on 20 QA gold test set
Faithfulness: 0.61, Answer Relevance: 0.58, Context Precision: 0.54.

Baseline = starter notebook with naive chunking and dense-only retrieval.

security(scanner): implement chunk scanner for poisoned document detection
Scans top-5 retrieved chunks for injection patterns before LLM call.

Patterns: ignore, disregard, override, system:, [INST], forget previous.

Tested against poisoned_advisory.html — correctly quarantines flagged chunk.

**Bad examples — never do this:**
fixed stuff

update

wip

final version

Commit after every meaningful unit of work. Small commits are easier to debug than one massive end-of-day dump.

---

## File Structure
vector-cartel/

│

├── README.md

├── requirements.txt

├── .gitignore

│

├── notebooks/

│   └── secureops_pipeline.ipynb       # main Colab notebook

│

├── src/

│   ├── ingestion.py                   # Jay — PDF + HTML + STIX parsing

│   ├── chunking.py                    # Jay — structure-aware chunkers per source

│   ├── retrieval.py                   # Sana — BM25 + dense + RRF + reranker

│   ├── agent.py                       # SK — LangGraph StateGraph

│   ├── llm.py                         # SK — Gemini integration + backoff + fallback

│   ├── prompts.py                     # SK — system prompt engine + spotlighting

│   ├── security.py                    # Kaveen — input/chunk scanner + output validator

│   ├── answer.py                      # Kaveen — Pydantic answer schema

│   └── evaluation.py                  # Kaveen — RAGAS harness + baseline comparison

│

├── corpus/

│   └── download_corpus.sh

│

├── data/

│   ├── qa_test_set.csv

│   └── poisoned_advisory.html

│

├── evaluation/

│   └── ragas_results.json

│

├── docs/

│   ├── architecture.png

│   └── AI_usage_statement.md

│

└── pitch/

└── vector_cartel_pitch.pdf

---

## AI Usage Statement

| Tool | Used For |
|------|----------|
| Claude (Anthropic) | Architecture design, system prompt engineering, documentation, debugging guidance |
| GitHub Copilot | Boilerplate code generation, autocomplete |
| Gemini (Google) | Primary LLM for the system itself |

All architectural decisions, evaluation design, red-teaming strategy, and system understanding are the team's own. Every team member can explain every component of the system.

---

## Submission Checklist

- [ ] Public GitHub repository accessible to judges
- [ ] `notebooks/secureops_pipeline.ipynb` runs end-to-end in Google Colab
- [ ] `corpus/download_corpus.sh` successfully downloads all sources
- [ ] `requirements.txt` installs all dependencies without errors
- [ ] Demo video (60–90 seconds) linked below
- [ ] Pitch slides linked below
- [ ] AI usage statement complete
- [ ] RAGAS results filled in above

**Demo video:** [link TBD]  
**Pitch slides:** [link TBD]

---

## Deadline

**Submission:** 23 June 2026, 13:00  
**Pitch event:** 23 June 2026, 14:00 — IMC004

---

<div align="center">

*Built by Vector Cartel · AAI Tech Talks Hackathon 2026 · WMG, University of Warwick*

</div>
