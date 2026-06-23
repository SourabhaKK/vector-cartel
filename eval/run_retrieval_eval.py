"""
run_retrieval_eval.py -- retrieval hit-rate + MRR over EVAL_SET.

Deliberately calls only ``build_retrieval_fn`` (embedder + reranker,
both local models) -- zero Gemini/HuggingFace API calls, so it can run
any time regardless of API quota. Run from the repo root:

    python -m eval.run_retrieval_eval
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_set import EVAL_SET
from src.retrieval import build_retrieval_fn


def _hit_rank(results, expected_doc: str) -> int | None:
    """1-based rank of the first chunk whose doc contains expected_doc, else None."""
    for i, c in enumerate(results, start=1):
        if expected_doc in (c["metadata"].get("doc") or ""):
            return i
    return None


def main() -> int:
    retrieval_fn = build_retrieval_fn()

    items = [e for e in EVAL_SET if e["kind"] in ("doc", "section")]
    hits = 0
    reciprocal_ranks = []
    print(f"Retrieval evaluation -- {len(items)} items (excludes the honesty/refusal item)\n")

    for item in items:
        results = retrieval_fn(item["question"])
        rank = _hit_rank(results, item["expected_doc"])
        hit = rank is not None
        hits += int(hit)
        reciprocal_ranks.append(1.0 / rank if hit else 0.0)

        section_note = ""
        if hit and item["kind"] == "section":
            matched = results[rank - 1]["metadata"].get("section", "")
            section_ok = item["expected_section"] in matched
            section_note = f"  (section expected={item['expected_section']!r} got={matched!r} {'OK' if section_ok else 'MISMATCH'})"

        status = f"rank {rank}" if hit else "MISS"
        print(f"[{item['id']:10s}] {status:8s}  {item['question'][:70]}{section_note}")

    hit_rate = hits / len(items)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print(f"\n--- Summary ---")
    print(f"Hit rate (expected doc in top-5): {hits}/{len(items)} = {hit_rate:.1%}")
    print(f"Mean Reciprocal Rank: {mrr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
