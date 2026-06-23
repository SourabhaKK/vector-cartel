"""
build_architecture_diagram_simple.py -- high-level, 5-box version of the
SecureOps Assistant architecture diagram, for the opening "architecture"
beat of the pitch (90s slot). One card per layer, single vertical flow,
no internal node detail -- the detailed version (build_architecture_diagram.py)
is the Q&A/appendix backup.

Run: python eval/build_architecture_diagram_simple.py
Output: eval/architecture_diagram_simple.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_W, FIG_H = 14.2, 12.5
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("white")

ACCENT = {
    "rag1": "#2563eb",
    "rag2": "#0891b2",
    "agentic": "#7c3aed",
    "llm": "#dc2626",
    "eval": "#16a34a",
    "io": "#1f2937",
}


def card(x, y, w, h, title, subtitle, key, fontsize=14, sub_fontsize=10.5, wrap_chars=58):
    """White card with colored left accent bar + drop shadow.

    subtitle is auto-wrapped to wrap_chars per line (tuned to the card's
    width/fontsize) so long descriptions can't bleed past the right edge --
    manual '\\n' in the source text is preserved as a forced line break.
    """
    import textwrap

    wrapped = "\n".join(
        "\n".join(textwrap.wrap(line, width=wrap_chars)) for line in subtitle.split("\n")
    )

    ax.add_patch(FancyBboxPatch(
        (x + 0.06, y - 0.06), w, h, boxstyle="round,pad=0,rounding_size=0.1",
        facecolor="#000000", edgecolor="none", alpha=0.10, zorder=2,
    ))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1",
        facecolor="white", edgecolor="#d1d5db", linewidth=1.0, zorder=3,
    ))
    accent_w = 0.14
    ax.add_patch(FancyBboxPatch(
        (x, y), accent_w, h, boxstyle="round,pad=0,rounding_size=0.05",
        facecolor=ACCENT[key], edgecolor="none", zorder=4,
    ))
    text_x = x + accent_w + 0.28
    ax.text(text_x, y + h - 0.3, title, ha="left", va="top",
            fontsize=fontsize, color="#111827", weight="bold", zorder=5, family="sans-serif")
    ax.text(text_x, y + h - 0.62, wrapped, ha="left", va="top",
            fontsize=sub_fontsize, color="#6b7280", zorder=5, family="sans-serif", linespacing=1.45)
    return (x + w / 2, y, x + w / 2, y + h, x, x + w)  # cx, bottom, cx, top, left, right


def dark_card(x, y, w, h, title, subtitle):
    """Solid dark card, used for the query-in / answer-out endpoints."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1",
        facecolor="#1f2937", edgecolor="none", zorder=4.5,
    ))
    ax.text(x + w / 2, y + h / 2 + 0.1, title, ha="center", va="center",
            fontsize=12.5, color="white", weight="bold", zorder=5, family="sans-serif")
    ax.text(x + w / 2, y + h / 2 - 0.18, subtitle, ha="center", va="center",
            fontsize=9.5, color="#d1d5db", zorder=5, family="sans-serif")
    return (x + w / 2, y, x + w / 2, y + h, x, x + w)


def vline(p1, p2, color="#374151", lw=2.2, dashed=False):
    ls = (0, (5, 3)) if dashed else "solid"
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", color=color, lw=lw,
                         connectionstyle="arc3,rad=0.0", mutation_scale=16,
                         linestyle=ls, shrinkA=2, shrinkB=2, zorder=1)
    ax.add_patch(a)


def badge(cx, top_y, color, number):
    ax.add_patch(plt.Circle((cx, top_y), 0.22, facecolor=color, edgecolor="none", zorder=6))
    ax.text(cx, top_y, number, ha="center", va="center", fontsize=11, color="white",
            weight="bold", zorder=7, family="sans-serif")


# ==============================================================================
# Title
# ==============================================================================
ax.text(0.4, FIG_H - 0.5, "SecureOps Assistant", fontsize=23, weight="bold",
        color="#111827", family="sans-serif")
ax.text(0.4, FIG_H - 0.95, "High-Level Architecture · AAI Tech Talks Hackathon 2026 · Vector Cartel",
        fontsize=11.5, color="#6b7280", family="sans-serif")

# ==============================================================================
# Query in
# ==============================================================================
query_x, query_w = 2.4, 6.2
query_c = dark_card(query_x, 10.35, query_w, 0.7, "Security Analyst Query",
                     "natural-language question, e.g. via Gradio")

# ==============================================================================
# 4 main layer cards, single vertical flow
# ==============================================================================
layer_x, layer_w, layer_h, gap = 1.5, 8.0, 1.55, 0.45
specs = [
    ("rag1", "1 · RAG Layer 1", "Corpus & Indexing — NIST, CISA (100 advisories), ATT&CK → structure-aware chunking → dense (ChromaDB) + sparse (BM25) indexes"),
    ("rag2", "2 · RAG Layer 2", "Hybrid Retrieval — RRF fusion → dedupe/metadata filter → cross-encoder rerank"),
    ("agentic", "3 · Agentic Layer", "Security Gate (InputScanner) → LangGraph routing — classify, decompose, retrieve, verify, synthesize, validate with retry"),
    ("llm", "4 · LLM Layer", "LLMRouter — Gemini 2.5 Flash (primary, deterministic) → HuggingFace fallback on quota/rate-limit exhaustion"),
]

y = 10.35 - gap
cards = []
y_positions = []
for i, (key, title, subtitle) in enumerate(specs):
    y -= layer_h + gap
    y_positions.append(y)
    c = card(layer_x, y, layer_w, layer_h, title, subtitle, key, fontsize=15, sub_fontsize=10, wrap_chars=62)
    cards.append(c)
    badge(layer_x - 0.45, y + layer_h - 0.25, ACCENT[key], str(i + 1))

vline((query_c[0], query_c[1]), (cards[0][0], cards[0][3]))
for i in range(len(cards) - 1):
    vline((cards[i][0], cards[i][1]), (cards[i + 1][0], cards[i + 1][3]))

# ==============================================================================
# Output
# ==============================================================================
out_y = y_positions[-1] - 1.0
output_c = dark_card(query_x, out_y, query_w, 0.7, "SecureOpsAnswer",
                      "grounded answer + citations + confidence + honest refusal")
vline((cards[-1][0], cards[-1][1]), (output_c[0], output_c[3]))

# ==============================================================================
# Evaluation Layer -- side card with dashed "measures" link into RAG Layer 2
# ==============================================================================
eval_card_x, eval_card_w, eval_card_h = layer_x + layer_w + 1.4, 2.6, 1.7
eval_card_y = (y_positions[1] + y_positions[2]) / 2 - 0.2
eval_c = card(eval_card_x, eval_card_y, eval_card_w, eval_card_h, "5 · Evaluation Layer",
              "Hit rate 100% · MRR 0.965\nGroundedness spot-check\nBefore/after fix evidence",
              "eval", fontsize=12, sub_fontsize=9)
badge(eval_card_x - 0.45, eval_card_y + eval_card_h - 0.25, ACCENT["eval"], "5")

rag2_c = cards[1]
vline((rag2_c[5], (rag2_c[1] + rag2_c[3]) / 2), (eval_c[4], (eval_c[1] + eval_c[3]) / 2),
      color=ACCENT["eval"], dashed=True, lw=1.8)
ax.text((rag2_c[5] + eval_c[4]) / 2, (rag2_c[1] + rag2_c[3]) / 2 + 0.18, "measures",
        ha="center", fontsize=9, color=ACCENT["eval"], style="italic", family="sans-serif")

plt.tight_layout()
plt.savefig("eval/architecture_diagram_simple.png", dpi=220, bbox_inches="tight", facecolor="white")
print("Saved eval/architecture_diagram_simple.png")
