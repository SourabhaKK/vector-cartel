# Tier 2 Evaluation Evidence — SecureOps Assistant

Vector Cartel · AAI Tech Talks Hackathon 2026 · WMG, University of Warwick

Per the brief's Tier 2 direction: *"No measurement of quality → 15–25 item test set; retrieval hit rate + groundedness (RAGAS / LLM-as-judge)."*

## Methodology

Two complementary checks, deliberately split so the larger one needs zero LLM API calls:

1. **Retrieval hit rate + MRR** — `eval/eval_set.py` (19 doc/section items + 1 honesty item) against `eval/run_retrieval_eval.py`. Every `expected_doc`/`expected_section` was verified against the real, indexed corpus (`chunks.jsonl`) *before* writing the question, not guessed — a miss reflects a genuine retrieval gap, not a bad ground truth. Calls only `build_retrieval_fn()` (local embedder + reranker), so it runs regardless of Gemini/HuggingFace quota.
2. **Groundedness spot-check** — per the starter kit README's own suggested method ("pick 5 answers, open the cited blocks, verify the claim is actually there"), using live, real-API transcripts captured during this project (not mocked).

## Results — Retrieval (run `python -m eval.run_retrieval_eval`)

| Metric | Result |
|---|---|
| Hit rate (expected doc in top-5) | **19 / 19 = 100%** |
| Mean Reciprocal Rank | **0.965** |

| ID | Question | Doc | Rank |
|---|---|---|---|
| nist82-01 | NIST remote access to OT networks | NIST SP 800-82 Rev. 3 §6.2.10 | 1 |
| nist82-02 | IT vs OT security priorities | NIST SP 800-82 Rev. 3 §2.3 | 1 |
| nist82-03 | Encryption for remote access sessions | NIST SP 800-82 Rev. 3 §1.4.1 | 1 |
| nist82-04 | Configuration control definition | NIST SP 800-82 Rev. 3 §128 | 1 |
| nist82-05 | Least privilege for OT remote access | NIST SP 800-82 Rev. 3 | 1 |
| csf-01 | Govern function / organizational context | NIST CSF 2.0 GV.OC-01 | 1 |
| csf-02 | Access permissions / least privilege | NIST CSF 2.0 PR.AA-05 | 1 |
| csf-03 | Cyber threat intelligence sources | NIST CSF 2.0 ID.RA-02 | 1 |
| csf-04 | CSF 2.0 functions | NIST CSF 2.0 | 1 |
| cisa-01 | ICSA-26-134-14 / CVE-2025-22871 (Siemens) | CISA Advisory ICSA-26-134-14 | 1 |
| cisa-02 | CVE-2026-8805 (Mitsubishi Electric) | CISA Advisory ICSA-26-169-05 | 1 |
| cisa-03 | CVE-2025-3465 (ABB) | CISA Advisory ICSA-26-139-01 | 1 |
| cisa-04 | ICSA-26-111-07 (Siemens, multi-CVE) | CISA Advisory ICSA-26-111-07 | 1 |
| cisa-05 | CVE-2024-41975 (ABB) | CISA Advisory ICSA-26-132-04 | 1 |
| cisa-06 | Siemens advisories summary | CISA Advisory (any) | 1 |
| attck-01 | ATT&CK technique: manipulation of control logic | MITRE ATT&CK T0831 | **3** |
| attck-02 | ATT&CK technique: manipulate I/O image | MITRE ATT&CK T0835 | 1 |
| attck-03 | ATT&CK technique: modify parameter | MITRE ATT&CK T0836 | 1 |
| attck-04 | Human user authentication mitigation | MITRE ATT&CK (any) | 1 |

**`attck-01` is the brief's exact ATT&CK example question** (Section 2). Before the duplicate-text dedup fix (see below), this query retrieved **zero** relevant ATT&CK techniques — every one of the top-5 slots was consumed by the same "Human User Authentication" mitigation paragraph duplicated across 20 different technique IDs, crowding out T0831 ("Manipulation of Control") entirely. After the fix, T0831 surfaces at rank 3.

### Fix: `_dedupe_candidates_by_text` (`src/retrieval.py`)

Root cause: MITRE ATT&CK mitigation text is frequently copy-pasted verbatim across dozens of technique IDs. Each occurrence is wrapped in a different per-technique header (`[MITRE ATT&CK T0800 — ...]` vs `[MITRE ATT&CK T0816 — ...]`), so the chunks are not identical strings even though the dedup-worthy content (everything after the header) is — comparing full text silently fails to catch this. Left undeduped, these near-identical vectors flood the fused candidate pool by sheer repetition. Fix: dedupe on the body after the header, keeping only the first (best fused-rank) occurrence. Covered by 4 unit tests in `tests/test_retrieval.py`.

## Results — Groundedness spot-check (live transcripts, real API)

4 of 5 brief Section 2 example questions tested against the live system with a real Gemini key; citations checked against the actual cited chunk text:

| Question | Outcome | Citation check |
|---|---|---|
| NIST remote access to OT networks | Answered, 13 citations | All claims map to real NIST SP 800-82 §6.2.10 / §128 / §1.4.1 text |
| Summarise Siemens advisories | Answered, 13 citations across 3 real advisories | CVEs/CVSS scores/affected products match the actual advisory text |
| IT vs OT security priorities | Answered, 2 citations | Matches NIST SP 800-82 §2.3 p.37 verbatim ordering (C-I-A vs Safety-A-I-C) |
| "What is our company's firewall configuration?" | **Correctly refused** | Honesty test required by the brief/README — passed |

The 5th example question (ATT&CK techniques) is covered by the retrieval-layer fix above rather than re-spent live-API groundedness testing, since Gemini's daily quota was exhausted by this session's testing.

## Known limitation (documented, not hidden)

Query classification (`classify_query`) was observed to classify the same input inconsistently across runs (e.g. the honesty-test question as `out_of_scope` in some runs, `simple` in others) despite `temperature=0.0` already being set. Root-caused to residual sampling variance in Gemini's serving infrastructure — temperature=0 reduces but does not guarantee bit-identical outputs. Mitigations applied: a fixed `seed=0` was added to tighten determinism further, and (independently) `run_agent`'s `refusal` flag is synced directly to the answer text rather than relying solely on the classifier's routing decision, so misclassification no longer produces an unsafe (non-refusing) response even when it occurs.
