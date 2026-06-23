"""
build_architecture_diagram.py -- generates the SecureOps Assistant
architecture diagram for pitch slides as a single PNG, styled like a
clean Figma/UML system diagram: swimlane panels per layer, white cards
with a colored accent bar + drop shadow, right-angle elbow connectors.

Run: python eval/build_architecture_diagram.py
Output: eval/architecture_diagram.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

FIG_W, FIG_H = 17, 12.5
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("white")

ACCENT = {
    "rag1": "#2563eb",
    "rag2": "#0891b2",
    "agentic": "#7c3aed",
    "security": "#ea580c",
    "llm": "#dc2626",
    "eval": "#16a34a",
    "io": "#1f2937",
}
PANEL_TINT = {
    "rag1": "#eff6ff",
    "rag2": "#ecfeff",
    "agentic": "#f5f3ff",
    "llm": "#fef2f2",
    "eval": "#f0fdf4",
}


def panel(x, y, w, h, title, key):
    """A swimlane background panel with a tab-style header label."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=PANEL_TINT[key], edgecolor=ACCENT[key], linewidth=1.3,
        linestyle=(0, (5, 3)), alpha=1.0, zorder=0,
    ))
    tab_w = 0.22 + 0.115 * len(title)
    ax.add_patch(FancyBboxPatch(
        (x + 0.18, y + h - 0.05), tab_w, 0.38, boxstyle="round,pad=0,rounding_size=0.06",
        facecolor=ACCENT[key], edgecolor="none", zorder=1,
    ))
    ax.text(x + 0.18 + tab_w / 2, y + h - 0.05 + 0.19, title, ha="center", va="center",
            fontsize=10.5, color="white", weight="bold", zorder=2, family="sans-serif")


def card(x, y, w, h, title, subtitle, key, fontsize=9.2, sub_fontsize=7.4):
    """A white card: drop shadow + colored left accent bar + title/subtitle."""
    shadow = FancyBboxPatch(
        (x + 0.045, y - 0.045), w, h, boxstyle="round,pad=0,rounding_size=0.07",
        facecolor="#000000", edgecolor="none", alpha=0.10, zorder=2,
    )
    ax.add_patch(shadow)
    body = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07",
        facecolor="white", edgecolor="#d1d5db", linewidth=0.9, zorder=3,
    )
    ax.add_patch(body)
    accent_w = 0.09
    accent = FancyBboxPatch(
        (x, y), accent_w, h, boxstyle="round,pad=0,rounding_size=0.035",
        facecolor=ACCENT[key], edgecolor="none", zorder=4,
    )
    ax.add_patch(accent)
    ty = y + h / 2 + (0.10 if subtitle else 0)
    ax.text(x + accent_w + 0.14, ty, title, ha="left", va="center",
            fontsize=fontsize, color="#111827", weight="bold", zorder=5, family="sans-serif")
    if subtitle:
        ax.text(x + accent_w + 0.14, y + h / 2 - 0.15, subtitle, ha="left", va="center",
                fontsize=sub_fontsize, color="#6b7280", zorder=5, family="sans-serif",
                linespacing=1.3)
    return (x + w / 2, y, x + w / 2, y + h, x, x + w)  # cx, bottom, cx, top, left, right


def routed(points, color="#4b5563", lw=1.5, dashed=False):
    """Draws a multi-segment orthogonal path; only the final segment gets an
    arrowhead. Avoids matplotlib's 'angle' connectionstyle, which raises on
    degenerate (parallel-ray) geometry for some box layouts."""
    ls = (0, (4, 2)) if dashed else "solid"
    pts = list(points)
    for i in range(len(pts) - 2):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, linestyle=ls,
                 zorder=1, solid_capstyle="round", solid_joinstyle="round")
    x1, y1 = pts[-2]
    x2, y2 = pts[-1]
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color=color, lw=lw,
                         mutation_scale=11, linestyle=ls, shrinkA=0, shrinkB=1.5, zorder=1)
    ax.add_patch(a)


def elbow(p1, p2, color="#4b5563", lw=1.5, dashed=False, via="h"):
    """Single-bend orthogonal connector.
    via='h': horizontal then vertical (bend at (x2,y1)) -- arrives vertically.
    via='v': vertical then horizontal (bend at (x1,y2)) -- arrives horizontally.
    """
    x1, y1 = p1
    x2, y2 = p2
    if abs(x1 - x2) < 0.04 or abs(y1 - y2) < 0.04:
        routed([p1, p2], color, lw, dashed)
        return
    bend = (x2, y1) if via == "h" else (x1, y2)
    routed([p1, bend, p2], color, lw, dashed)


def straight(p1, p2, color="#4b5563", lw=1.5, dashed=False):
    routed([p1, p2], color, lw, dashed)


def caption(x, y, text, fontsize=7.3, color="#6b7280"):
    ax.text(x, y, text, fontsize=fontsize, color=color, family="sans-serif", style="italic")


# ==============================================================================
# Title block
# ==============================================================================
ax.text(0.3, FIG_H - 0.45, "SecureOps Assistant", fontsize=20, weight="bold",
        color="#111827", family="sans-serif")
ax.text(0.3, FIG_H - 0.85, "System Architecture  ·  AAI Tech Talks Hackathon 2026  ·  Vector Cartel",
        fontsize=10.5, color="#6b7280", family="sans-serif")

sources = [
    ("NIST SP 800-82 Rev.3", "~300pg OT standard"),
    ("NIST CSF 2.0", "framework"),
    ("CISA Advisories", "100 real, 0 fallback"),
    ("MITRE ATT&CK for ICS", "499 chunks"),
]

# ---- Panel 1: RAG Layer 1 ----
p1_x, p1_y, p1_w, p1_h = 0.3, 8.75, 10.7, 2.3
panel(p1_x, p1_y, p1_w, p1_h, "RAG LAYER 1 — Corpus & Indexing", "rag1")

src_y, src_w, src_h = 10.25, 1.95, 0.55
src_cards = []
for i, (t, s) in enumerate(sources):
    cx = 0.7 + i * 2.45
    c = card(cx, src_y, src_w, src_h, t, s, "rag1", fontsize=8.3, sub_fontsize=6.8)
    src_cards.append(c)

chunk_c = card(2.95, 9.55, 5.0, 0.5, "Structure-Aware Chunking", "per source type + contextual header + metadata", "rag1", fontsize=8.6, sub_fontsize=6.8)
for c in src_cards:
    elbow((c[0], c[1]), (chunk_c[0], chunk_c[3]), color=ACCENT["rag1"], via="h")

dense_c = card(0.7, 8.9, 3.5, 0.5, "Dense Index", "BGE embeddings → ChromaDB (cosine)", "rag1", fontsize=8.4, sub_fontsize=6.8)
sparse_c = card(5.95, 8.9, 3.5, 0.5, "Sparse Index", "BM25, identifier-preserving tokenizer", "rag1", fontsize=8.4, sub_fontsize=6.8)
elbow((chunk_c[0], chunk_c[1]), (dense_c[0], dense_c[3]), color=ACCENT["rag1"], via="h")
elbow((chunk_c[0], chunk_c[1]), (sparse_c[0], sparse_c[3]), color=ACCENT["rag1"], via="h")

# ---- Panel 2: RAG Layer 2 ----
p2_x, p2_y, p2_w, p2_h = 0.3, 7.05, 10.7, 1.45
panel(p2_x, p2_y, p2_w, p2_h, "RAG LAYER 2 — Hybrid Retrieval", "rag2")

fuse_c = card(0.7, 7.4, 2.85, 0.5, "RRF Fusion", "dense + sparse rankings", "rag2", fontsize=8.4, sub_fontsize=6.8)
ddf_c = card(3.85, 7.4, 2.85, 0.5, "Dedupe + Filter", "drops boilerplate dupes, vendor/date", "rag2", fontsize=8.2, sub_fontsize=6.6)
rerank_c = card(7.0, 7.4, 2.85, 0.5, "Cross-Encoder Rerank", "bge-reranker-base → top-5", "rag2", fontsize=8.4, sub_fontsize=6.8)

elbow((dense_c[0], dense_c[1]), (fuse_c[0] - 0.3, fuse_c[3]), color=ACCENT["rag2"], via="h")
elbow((sparse_c[0], sparse_c[1]), (fuse_c[0] + 0.3, fuse_c[3]), color=ACCENT["rag2"], via="h")
straight((fuse_c[5], (fuse_c[1] + fuse_c[3]) / 2), (ddf_c[4], (ddf_c[1] + ddf_c[3]) / 2), color=ACCENT["rag2"])
straight((ddf_c[5], (ddf_c[1] + ddf_c[3]) / 2), (rerank_c[4], (rerank_c[1] + rerank_c[3]) / 2), color=ACCENT["rag2"])

# ---- Panel 3: Agentic Layer (with Security Gate) ----
p3_x, p3_y, p3_w, p3_h = 0.3, 3.55, 10.7, 3.25
panel(p3_x, p3_y, p3_w, p3_h, "AGENTIC LAYER — LangGraph Routing", "agentic")

gate_c = card(3.55, 5.95, 4.2, 0.5, "Input Gate — InputScanner", "blocks prompt injection before any LLM call", "security", fontsize=8.4, sub_fontsize=6.6)
elbow((rerank_c[0], rerank_c[1]), (gate_c[0], gate_c[3]), color=ACCENT["rag2"], via="h")

classify_c = card(3.55, 5.25, 4.2, 0.5, "Classify Query", "simple · multi_hop · out_of_scope · ambiguous", "agentic", fontsize=8.4, sub_fontsize=6.6)
straight((gate_c[0], gate_c[1]), (classify_c[0], classify_c[3]), color=ACCENT["security"])

branch_y, branch_w, branch_h = 4.45, 2.4, 0.5
retrieve_c = card(0.55, branch_y, branch_w, branch_h, "Retrieve", "simple path", "agentic", fontsize=8.2, sub_fontsize=6.6)
decomp_c = card(3.15, branch_y, branch_w, branch_h, "Decompose → Retrieve", "multi-hop", "agentic", fontsize=7.7, sub_fontsize=6.6)
refuse_c = card(5.75, branch_y, branch_w, branch_h, "Refuse", "out_of_scope", "agentic", fontsize=8.2, sub_fontsize=6.6)
clarify_c = card(8.35, branch_y, branch_w, branch_h, "Clarify", "ambiguous", "agentic", fontsize=8.2, sub_fontsize=6.6)
for c in (retrieve_c, decomp_c, refuse_c, clarify_c):
    elbow((classify_c[0], classify_c[1]), (c[0], c[3]), color=ACCENT["agentic"], via="h")

synth_c = card(0.55, 3.75, 4.95, 0.5, "Verify → Synthesize Answer", "calls LLM Layer with retrieved context", "agentic", fontsize=8.2, sub_fontsize=6.6)
elbow((retrieve_c[0], retrieve_c[1]), (synth_c[4] + 1.2, synth_c[3]), color=ACCENT["agentic"], via="h")
elbow((decomp_c[0], decomp_c[1]), (synth_c[5] - 1.2, synth_c[3]), color=ACCENT["agentic"], via="h")

validate_c = card(5.85, 3.75, 4.95, 0.5, "Validate Citations", "groundedness check · retry ×1 on failure", "agentic", fontsize=8.2, sub_fontsize=6.6)
straight((synth_c[5], (synth_c[1] + synth_c[3]) / 2), (validate_c[4], (validate_c[1] + validate_c[3]) / 2), color=ACCENT["agentic"])

retry_start = (validate_c[0] - 1.0, validate_c[3])
retry_end = (synth_c[0] + 1.0, synth_c[3])
routed([retry_start, (retry_start[0], 4.78), (retry_end[0], 4.78), retry_end], color="#9ca3af", lw=1.3, dashed=True)
caption(6.0, 4.85, "retry on low groundedness", fontsize=6.6)

# ---- Panel 4: LLM Layer ----
p4_x, p4_y, p4_w, p4_h = 0.3, 1.85, 10.7, 1.55
panel(p4_x, p4_y, p4_w, p4_h, "LLM LAYER — LLMRouter", "llm")

router_c = card(3.95, 2.5, 3.0, 0.5, "LLMRouter", "primary → fallback chain", "llm", fontsize=8.6, sub_fontsize=6.8)
elbow((synth_c[0] + 1.0, synth_c[1]), (router_c[0], router_c[3]), color=ACCENT["agentic"], via="h")

gemini_c = card(1.4, 1.95, 2.6, 0.45, "Gemini 2.5 Flash", "primary · temp=0, seed=0, thinking_budget=0", "llm", fontsize=8.0, sub_fontsize=6.3)
hf_c = card(6.3, 1.95, 2.6, 0.45, "HuggingFace (Qwen2.5)", "fallback · temp=0 · router.huggingface.co", "llm", fontsize=8.0, sub_fontsize=6.3)
elbow((router_c[0] - 0.4, router_c[1]), (gemini_c[0], gemini_c[3]), color=ACCENT["llm"], via="h")
elbow((router_c[0] + 0.4, router_c[1]), (hf_c[0], hf_c[3]), color=ACCENT["llm"], via="h")
caption(6.95, 1.78, "on MaxRetriesExceeded", fontsize=6.5)

# ---- Output + exception routes ----
output_c = card(4.05, 0.6, 3.0, 0.65, "SecureOpsAnswer", "answer + citations + confidence + refusal", "io", fontsize=8.4, sub_fontsize=6.6)
straight((validate_c[0] - 1.0, p3_y), (output_c[0], output_c[3]), color=ACCENT["io"])

refuse_x = output_c[2] + 0.3
refuse_mid_y = output_c[3] + 0.35
routed([(refuse_c[0], branch_y), (refuse_c[0], refuse_mid_y), (refuse_x, refuse_mid_y), (refuse_x, output_c[3])],
       color=ACCENT["agentic"], dashed=True)

clarify_x = output_c[2] + 0.85
clarify_mid_y = output_c[3] + 0.55
routed([(clarify_c[0], branch_y), (clarify_c[0], clarify_mid_y), (clarify_x, clarify_mid_y), (clarify_x, output_c[3])],
       color=ACCENT["agentic"], dashed=True)

bypass_start = (gate_c[2] + 0.1, gate_c[1] + 0.25)
bypass_end = (output_c[5], output_c[1] + 0.3)
routed([bypass_start, (10.5, gate_c[1] + 0.25), (10.5, output_c[1] + 0.3), bypass_end],
       color=ACCENT["security"], dashed=True)
caption(10.55, 1.6, "blocked queries\nshort-circuit here", fontsize=6.3, color=ACCENT["security"])

# ==============================================================================
# Panel 5: Evaluation Layer (side panel)
# ==============================================================================
p5_x, p5_y, p5_w, p5_h = 11.4, 5.55, 5.0, 5.5
panel(p5_x, p5_y, p5_w, p5_h, "EVALUATION LAYER", "eval")

eval_set_c = card(11.8, 9.8, 4.2, 0.55, "Eval Set — 20 items", "ground-truthed against real corpus", "eval", fontsize=8.4, sub_fontsize=6.8)
metrics_c = card(11.8, 8.95, 4.2, 0.55, "Hit Rate 100%  ·  MRR 0.965", "across NIST, CSF, CISA, ATT&CK", "eval", fontsize=9.2, sub_fontsize=6.8)
ground_c = card(11.8, 8.1, 4.2, 0.55, "Groundedness Spot-Check", "live transcripts, citation audit", "eval", fontsize=8.4, sub_fontsize=6.8)
beforeafter_c = card(11.8, 7.0, 4.2, 0.75, "Before / After Evidence", "dedup fix: ATT&CK Q4 miss → rank 3\ninjection: silent refusal → explicit block", "eval", fontsize=8.2, sub_fontsize=6.6)

elbow((eval_set_c[0], eval_set_c[1]), (metrics_c[0], metrics_c[3]), color=ACCENT["eval"], via="h")
elbow((metrics_c[0], metrics_c[1]), (ground_c[0], ground_c[3]), color=ACCENT["eval"], via="h")
elbow((ground_c[0], ground_c[1]), (beforeafter_c[0], beforeafter_c[3]), color=ACCENT["eval"], via="h")

eval_link_start = (eval_set_c[4], (eval_set_c[1] + eval_set_c[3]) / 2)
eval_link_end = (rerank_c[5], (rerank_c[1] + rerank_c[3]) / 2)
routed([eval_link_start, (rerank_c[5] + 0.45, eval_link_start[1]), (rerank_c[5] + 0.45, eval_link_end[1]), eval_link_end],
       color=ACCENT["eval"], dashed=True)
caption(10.6, 8.65, "measures", fontsize=6.6, color=ACCENT["eval"])

# ==============================================================================
# Legend
# ==============================================================================
legend_items = [
    ("RAG Layer 1 — Corpus & Indexing", ACCENT["rag1"]),
    ("RAG Layer 2 — Hybrid Retrieval", ACCENT["rag2"]),
    ("Agentic Layer — LangGraph routing", ACCENT["agentic"]),
    ("Security Gate — InputScanner", ACCENT["security"]),
    ("LLM Layer — Gemini + HF fallback", ACCENT["llm"]),
    ("Evaluation Layer", ACCENT["eval"]),
]
handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=11, label=t)
           for t, c in legend_items]
leg = ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, -0.04),
                 fontsize=8.2, framealpha=0.97, ncol=2, edgecolor="#d1d5db")

plt.tight_layout()
plt.savefig("eval/architecture_diagram.png", dpi=220, bbox_inches="tight", facecolor="white")
print("Saved eval/architecture_diagram.png")
