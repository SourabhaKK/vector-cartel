"""
eval.py — SecureOps Assistant · retrieval evaluation harness
================================================================================
Vector Cartel · AAI Tech Talks Hackathon 2026 · WMG, University of Warwick

PURPOSE
-------
Measure retrieval quality with a small, hand-curated GOLD set so "it looks good"
becomes a defensible number. For each gold query we run the real retrieval_fn and
check whether a *relevant* chunk appears in the returned top-k, and at what rank.

METRICS
-------
    Hit@1   — fraction of queries whose #1 chunk is relevant
    Hit@5   — fraction with a relevant chunk anywhere in top-5 (recall@5)
    MRR     — mean reciprocal rank of the first relevant chunk (0 if none in top-5)

Reported overall and per source_type, with the misses listed for inspection.

GOLD DESIGN
-----------
Each item: {"query", "source", "match"}. ``match`` is a relevance spec; a chunk
is relevant when ALL provided keys are satisfied:
    technique_id / cve / alert_code : exact equality on metadata
    section / section_prefix        : exact / prefix match on metadata.section
    doc_contains                    : substring of metadata.doc
    text_contains_any               : ANY of these substrings in chunk text (ci)
Targets were sampled from the live corpus, so every gold answer exists.

USAGE
-----
    conda run -n GAN --no-capture-output python -m src.eval -v
    python -m src.eval --show-misses    # also print the top hit for each miss
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Dict, List, Optional

from src.retrieval import build_retrieval_fn

try:
    from src.contracts import RETRIEVAL_CONFIDENCE_THRESHOLD
except Exception:  # noqa: BLE001
    RETRIEVAL_CONFIDENCE_THRESHOLD = 0.35

logger = logging.getLogger("secureops.eval")


# ==============================================================================
# GOLD SET — hand-curated, grounded in the live corpus
# ==============================================================================

GOLD: List[Dict[str, Any]] = [
    # --- CISA advisories: semantic by product + exact-ID lookups (18) ---
    {"query": "privilege escalation in Siemens RUGGEDCOM CROSSBOW access manager",
     "source": "advisory", "match": {"cve": "CVE-2026-27668"}},
    {"query": "CVE-2026-24032 Siemens SINEC NMS vulnerability",
     "source": "advisory", "match": {"cve": "CVE-2026-24032"}},
    {"query": "security flaw in the Hardy Barth Salia EV charge controller",
     "source": "advisory", "match": {"alert_code": "ICSA-26-111-05"}},
    {"query": "Zero Motorcycles firmware vulnerability",
     "source": "advisory", "match": {"cve": "CVE-2026-1354"}},
    {"query": "Siemens SCALANCE wireless flaw CVE-2020-24588",
     "source": "advisory", "match": {"cve": "CVE-2020-24588"}},
    {"query": "vulnerability in Siemens Analytics Toolkit",
     "source": "advisory", "match": {"alert_code": "ICSA-26-111-04"}},
    {"query": "Siemens RUGGEDCOM CROSSBOW Station Access Controller issue",
     "source": "advisory", "match": {"alert_code": "ICSA-26-111-08"}},
    {"query": "Silex Technology SD-330AC and AMC Manager vulnerability CVE-2026-32955",
     "source": "advisory", "match": {"cve": "CVE-2026-32955"}},
    {"query": "Siemens Industrial Edge Management security advisory",
     "source": "advisory", "match": {"cve": "CVE-2026-33892"}},
    {"query": "SenseLive X3050 device vulnerability",
     "source": "advisory", "match": {"cve": "CVE-2026-40630"}},
    {"query": "Yadea T5 electric bicycle security issue",
     "source": "advisory", "match": {"cve": "CVE-2025-70994"}},
    {"query": "Carlson Software VASCO-B GNSS receiver flaw",
     "source": "advisory", "match": {"cve": "CVE-2026-3893"}},
    {"query": "Milesight cameras vulnerability CVE-2026-28747",
     "source": "advisory", "match": {"cve": "CVE-2026-28747"}},
    {"query": "Intrado 911 Emergency Gateway EGW vulnerability",
     "source": "advisory", "match": {"cve": "CVE-2026-6074"}},
    {"query": "NSA GRASSMARLIN network mapping tool vulnerability",
     "source": "advisory", "match": {"cve": "CVE-2026-6807"}},
    {"query": "ABB System 800xA and Symphony Plus IEC 61850 vulnerability",
     "source": "advisory", "match": {"cve": "CVE-2025-3756"}},
    {"query": "ABB PCM600 vulnerability CVE-2018-1002208",
     "source": "advisory", "match": {"cve": "CVE-2018-1002208"}},
    {"query": "Siemens SINEC NMS second advisory CVE-2026-25654",
     "source": "advisory", "match": {"cve": "CVE-2026-25654"}},

    # --- MITRE ATT&CK for ICS: techniques + a mitigation-targeted query (18) ---
    {"query": "how do adversaries activate firmware update mode to inhibit response",
     "source": "attck", "match": {"technique_id": "T0800"}},
    {"query": "brute force I/O technique to manipulate physical process",
     "source": "attck", "match": {"technique_id": "T0806"}},
    {"query": "data destruction to inhibit response functions in ICS",
     "source": "attck", "match": {"technique_id": "T0809"}},
    {"query": "command-line interface used for execution by adversaries",
     "source": "attck", "match": {"technique_id": "T0807"}},
    {"query": "automated collection of data from control systems",
     "source": "attck", "match": {"technique_id": "T0802"}},
    {"query": "monitoring process state for collection",
     "source": "attck", "match": {"technique_id": "T0801"}},
    {"query": "modify a parameter to impair process control on a controller",
     "source": "attck", "match": {"technique_id": "T0836"}},
    {"query": "denial of service to inhibit response in industrial systems",
     "source": "attck", "match": {"technique_id": "T0814"}},
    {"query": "forcing a device restart or shutdown",
     "source": "attck", "match": {"technique_id": "T0816"}},
    {"query": "drive-by compromise as an initial access vector",
     "source": "attck", "match": {"technique_id": "T0817"}},
    {"query": "exploit a public-facing application to gain access",
     "source": "attck", "match": {"technique_id": "T0819"}},
    {"query": "external remote services used for initial access",
     "source": "attck", "match": {"technique_id": "T0822"}},
    {"query": "adversary-in-the-middle to collect or manipulate traffic",
     "source": "attck", "match": {"technique_id": "T0830"}},
    {"query": "manipulate the I/O image to inhibit response",
     "source": "attck", "match": {"technique_id": "T0835"}},
    {"query": "modify alarm settings to hide malicious activity",
     "source": "attck", "match": {"technique_id": "T0838"}},
    {"query": "network connection enumeration for discovery",
     "source": "attck", "match": {"technique_id": "T0840"}},
    {"query": "modify controller tasking to change program execution",
     "source": "attck", "match": {"technique_id": "T0821"}},
    {"query": "use network segmentation to mitigate firmware update mode abuse",
     "source": "attck", "match": {"technique_id": "T0800"}},

    # --- NIST CSF 2.0 (coded subcategories) (8) ---
    {"query": "establishing organizational cybersecurity roles and responsibilities",
     "source": "nist_csf", "match": {"doc_contains": "CSF", "text_contains_any": ["GV.RR-01", "roles", "responsibilities"]}},
    {"query": "understanding the organizational context and mission for governance",
     "source": "nist_csf", "match": {"doc_contains": "CSF", "text_contains_any": ["GV.OC-01", "organizational context", "mission"]}},
    {"query": "cybersecurity supply chain risk management practices",
     "source": "nist_csf", "match": {"doc_contains": "CSF", "text_contains_any": ["GV.SC", "supply chain"]}},
    {"query": "identifying and recording asset vulnerabilities in risk assessment",
     "source": "nist_csf", "match": {"doc_contains": "CSF", "text_contains_any": ["ID.RA-02", "vulnerabilit"]}},
    {"query": "managing identities and authenticating users for access",
     "source": "nist_csf", "match": {"doc_contains": "CSF", "text_contains_any": ["PR.AA-05", "access permission", "authenticat"]}},
    {"query": "protecting the resilience of technology infrastructure",
     "source": "nist_csf", "match": {"doc_contains": "CSF", "text_contains_any": ["PR.IR-02", "resilience", "infrastructure"]}},
    {"query": "executing the recovery plan after a cybersecurity incident",
     "source": "nist_csf", "match": {"doc_contains": "CSF", "text_contains_any": ["RC.RP-02", "recovery"]}},
    {"query": "incident management and coordination during response",
     "source": "nist_csf", "match": {"doc_contains": "CSF", "text_contains_any": ["RS.MA-01", "incident"]}},

    # --- NIST SP 800-82 (prose topics) (10) ---
    {"query": "remote access controls and secure remote maintenance for OT",
     "source": "nist_sp80082", "match": {"doc_contains": "NIST SP 800-82", "text_contains_any": ["remote access", "remote maintenance"]}},
    {"query": "network segmentation and segregation for industrial control systems",
     "source": "nist_sp80082", "match": {"doc_contains": "NIST SP 800-82", "text_contains_any": ["segment", "segregat", "zone"]}},
    {"query": "defense in depth strategy for ICS security",
     "source": "nist_sp80082", "match": {"doc_contains": "NIST SP 800-82", "text_contains_any": ["defense-in-depth", "defense in depth", "layer"]}},
    {"query": "incident response and contingency planning for control systems",
     "source": "nist_sp80082", "match": {"doc_contains": "NIST SP 800-82", "text_contains_any": ["incident response", "contingency"]}},
    {"query": "firewall configuration and boundary protection between IT and OT",
     "source": "nist_sp80082", "match": {"doc_contains": "NIST SP 800-82", "text_contains_any": ["firewall", "boundary"]}},
    {"query": "least privilege and access control for OT accounts",
     "source": "nist_sp80082", "match": {"doc_contains": "NIST SP 800-82", "text_contains_any": ["least privilege", "access control"]}},
    {"query": "physical and environmental security controls for ICS",
     "source": "nist_sp80082", "match": {"doc_contains": "NIST SP 800-82", "text_contains_any": ["physical", "environmental"]}},
    {"query": "security awareness and training program for OT staff",
     "source": "nist_sp80082", "match": {"doc_contains": "NIST SP 800-82", "text_contains_any": ["awareness", "training"]}},
    {"query": "patch management and vulnerability remediation in OT environments",
     "source": "nist_sp80082", "match": {"doc_contains": "NIST SP 800-82", "text_contains_any": ["patch", "vulnerability management", "remediation"]}},
    {"query": "continuous monitoring and logging of control system activity",
     "source": "nist_sp80082", "match": {"doc_contains": "NIST SP 800-82", "text_contains_any": ["monitoring", "logging", "audit"]}},
]


# ==============================================================================
# Relevance judging
# ==============================================================================

def is_relevant(chunk: Dict[str, Any], match: Dict[str, Any]) -> bool:
    """True iff the chunk satisfies ALL keys in the match spec."""
    m = chunk.get("metadata", {})
    text = (chunk.get("text") or "").lower()
    for key, val in match.items():
        if key in ("technique_id", "cve", "alert_code", "section"):
            if str(m.get(key, "")) != str(val):
                return False
        elif key == "section_prefix":
            if not str(m.get("section", "")).startswith(str(val)):
                return False
        elif key == "doc_contains":
            if str(val).lower() not in str(m.get("doc", "")).lower():
                return False
        elif key == "text_contains_any":
            if not any(str(s).lower() in text for s in val):
                return False
        else:
            logger.warning("unknown match key %r", key)
            return False
    return True


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate(gold: List[Dict[str, Any]] = GOLD, k: int = 5, show_misses: bool = False) -> Dict[str, Any]:
    fn = build_retrieval_fn()

    per_source: Dict[str, Dict[str, float]] = {}
    rows: List[Dict[str, Any]] = []
    total_latency = 0.0

    for item in gold:
        t0 = time.perf_counter()
        results = fn(item["query"])
        total_latency += time.perf_counter() - t0

        rank = None
        for i, c in enumerate(results[:k], 1):
            if is_relevant(c, item["match"]):
                rank = i
                break

        top_score = results[0]["score"] if results else 0.0
        rows.append({
            "query": item["query"], "source": item["source"], "rank": rank,
            "top_score": top_score, "results": results,
        })

        s = per_source.setdefault(item["source"], {"n": 0, "hit1": 0, "hitk": 0, "rr": 0.0})
        s["n"] += 1
        if rank == 1:
            s["hit1"] += 1
        if rank is not None:
            s["hitk"] += 1
            s["rr"] += 1.0 / rank

    n = len(gold)
    hit1 = sum(1 for r in rows if r["rank"] == 1)
    hitk = sum(1 for r in rows if r["rank"] is not None)
    mrr = sum(1.0 / r["rank"] for r in rows if r["rank"]) / n if n else 0.0

    # ---- report ----
    print(f"\n{'='*64}\nRetrieval evaluation — {n} gold queries, k={k}\n{'='*64}")
    print(f"{'metric':<10}{'overall':>10}")
    print(f"{'Hit@1':<10}{hit1/n:>10.1%}")
    print(f"{'Hit@'+str(k):<10}{hitk/n:>10.1%}")
    print(f"{'MRR':<10}{mrr:>10.3f}")
    print(f"{'avg lat':<10}{total_latency/n:>9.2f}s")

    print(f"\n{'source':<12}{'n':>4}{'Hit@1':>8}{'Hit@'+str(k):>8}{'MRR':>8}")
    print("-" * 40)
    for src, s in per_source.items():
        print(f"{src:<12}{int(s['n']):>4}{s['hit1']/s['n']:>8.0%}"
              f"{s['hitk']/s['n']:>8.0%}{s['rr']/s['n']:>8.3f}")

    misses = [r for r in rows if r["rank"] is None]
    print(f"\nMisses (no relevant chunk in top-{k}): {len(misses)}")
    for r in misses:
        print(f"  ✗ [{r['source']}] {r['query']}")
        if show_misses and r["results"]:
            top = r["results"][0]
            print(f"      got #1: {top['metadata'].get('chunk_id')} "
                  f"(score {top['score']:.3f}, {top['metadata'].get('section')})")

    return {
        "n": n, "hit@1": hit1 / n, f"hit@{k}": hitk / n, "mrr": mrr,
        "avg_latency_s": total_latency / n, "per_source": per_source,
    }


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate SecureOps retrieval.")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--show-misses", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    evaluate(k=args.k, show_misses=args.show_misses)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
