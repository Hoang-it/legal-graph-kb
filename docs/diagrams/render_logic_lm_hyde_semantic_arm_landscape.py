"""Landscape (16:9, slide-ready) architecture of the Logic-LM x HyDE-semantic arm.

Same generic module roles + flow as render_logic_lm_hyde_semantic_arm.py, but
laid out as a left->right serpentine (row 1 ->, reasoning <-, output ->) so it
fits a 16:9 slide. One hypothesis with two uses, the single treatment-only
injection point, the repair loop, the integrity guardrail and the supporting
stores all mirror the runtime code.

Run:  python docs/diagrams/render_logic_lm_hyde_semantic_arm_landscape.py
Out:  docs/diagrams/logic_lm_hyde_semantic_arm_landscape.png  (+ .svg)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent

C = {
    "input": "#64748b", "det": "#2563eb", "llm": "#d97706", "retr": "#0891b2",
    "sym": "#7c3aed", "solve": "#059669", "store": "#475569", "out": "#0d9488",
}
EDGE = "#475569"
INK = "#0f172a"
MUTE = "#475569"
RED = "#dc2626"

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

B: dict[str, tuple[float, float, float, float]] = {}


def box(name, cx, cy, w, h, title, color, sub="", badge=None, tag=None,
        title_fs=11.5, sub_fs=8.2):
    B[name] = (cx, cy, w, h)
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.09",
        linewidth=0, facecolor=color, alpha=0.97, zorder=3))
    if sub:
        ax.text(cx, cy + h / 2 - 0.23, title, ha="center", va="top",
                color="white", fontsize=title_fs, fontweight="bold", zorder=4)
        ax.text(cx, cy + h / 2 - 0.23 - 0.34, sub, ha="center", va="top",
                color="white", fontsize=sub_fs, zorder=4, linespacing=1.2)
    else:
        ax.text(cx, cy, title, ha="center", va="center", color="white",
                fontsize=title_fs, fontweight="bold", zorder=4)
    if badge is not None:
        ax.text(cx - w / 2, cy + h / 2, str(badge), ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=color, zorder=5,
                bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec=color, lw=1.6))
    if tag is not None:
        ax.text(cx - w / 2 + 0.05, cy + h / 2 + 0.07, tag, ha="left", va="bottom",
                fontsize=8.0, style="italic", color=MUTE, zorder=5)


def stat(name, text, dy=0.14, color=MUTE):
    cx, cy, w, h = B[name]
    ax.text(cx, cy - h / 2 - dy, text, ha="center", va="top",
            fontsize=7.6, color=color, style="italic", zorder=4)


def T(n):
    cx, cy, w, h = B[n]; return (cx, cy + h / 2)
def Bt(n):
    cx, cy, w, h = B[n]; return (cx, cy - h / 2)
def Lf(n):
    cx, cy, w, h = B[n]; return (cx - w / 2, cy)
def Rt(n):
    cx, cy, w, h = B[n]; return (cx + w / 2, cy)


def arrow(p1, p2, label=None, color=EDGE, ls="-", lw=2.0, rad=0.0,
          mut=14, lfs=7.6, lcol=None, ldx=0.0, ldy=0.0, la="center", z=2):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=mut, lw=lw, color=color,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=z))
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + ldx, (p1[1] + p2[1]) / 2 + ldy
        ax.text(mx, my, label, ha=la, va="center", fontsize=lfs,
                color=lcol or INK, zorder=6, linespacing=1.15,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.9))


# ---- title + subtitle -----------------------------------------------------
ax.text(8.0, 8.62, "Logic-LM × HyDE-semantic  —  per-question arm architecture",
        ha="center", fontsize=17, fontweight="bold", color=INK)
ax.text(8.0, 8.20, "generic module roles  ·  one hypothesis → two uses (① embedding · ② rule-gen)  ·  "
        "treatment vs. control differ at a single injection point",
        ha="center", fontsize=9.6, color=MUTE)

# ---- legend ---------------------------------------------------------------
legend = [
    ("Input", C["input"]), ("Deterministic", C["det"]), ("LLM call", C["llm"]),
    ("Retrieval", C["retr"]), ("Symbolic build", C["sym"]), ("Solver", C["solve"]),
    ("Store / model", C["store"]), ("Output", C["out"]),
]
lx = 1.0
for lab, col in legend:
    ax.add_patch(FancyBboxPatch((lx, 7.66), 0.28, 0.24, boxstyle="round,pad=0.02",
                                fc=col, ec="none", zorder=4))
    ax.text(lx + 0.37, 7.78, lab, ha="left", va="center", fontsize=8.2, color=MUTE)
    lx += 0.5 + 0.082 * len(lab) + 0.42

# ---- supporting stores (top band) ----------------------------------------
box("ONT", 5.0, 7.05, 3.0, 0.5, "Domain Ontology / Concept Store", C["store"], title_fs=8.8)
box("CACHE", 8.7, 7.05, 3.0, 0.5, "Hypothesis Cache (disk · shared)", C["store"], title_fs=8.8)
box("EMB", 11.4, 7.05, 1.85, 0.5, "Embedding Model", C["store"], title_fs=8.8)
box("KG", 13.45, 7.05, 1.95, 0.5, "Knowledge Graph", C["store"], title_fs=8.8)

# ---- ROW 1 (left -> right) : understand / hypothesis / search -------------
box("Q", 1.6, 6.05, 2.2, 0.78, "User Question", C["input"], title_fs=11)
box("CONCEPT", 5.0, 6.05, 3.0, 1.02, "1 · Concept Frame Builder", C["det"],
    sub="question → concept frame (no LLM)", badge="1", tag="understand", title_fs=10.6)
box("HYDE", 8.7, 6.05, 3.0, 1.02, "2 · Hypothesis Generator", C["llm"],
    sub="→ hypothetical legal passage", badge="2", tag="hypothesis", title_fs=10.6)
box("RETR", 12.4, 6.05, 3.0, 1.02, "3 · Dense Vector Retrieval", C["retr"],
    sub="embed → vector search → clauses", badge="3", tag="search", title_fs=10.6)

# ---- ROW 2 (right -> left) : symbolic reasoning cluster -------------------
ax.add_patch(FancyBboxPatch((2.55, 3.42), 11.85, 1.46,
             boxstyle="round,pad=0.04,rounding_size=0.09",
             fc="#eef2f7", ec="#94a3b8", lw=1.4, ls="--", zorder=1))
ax.text(2.72, 4.78, "4 · Symbolic Reasoning", ha="left", va="top",
        fontsize=9.6, fontweight="bold", color="#334155", zorder=2)
ax.text(14.25, 4.78, "[ reason ]", ha="right", va="top", fontsize=8.2,
        style="italic", color="#64748b", zorder=2)

box("ASM", 12.4, 4.08, 3.2, 1.12, "Rule-Gen Assembler  (LLM)", C["llm"],
    sub="Treatment: + hypothesis · HyDE prompt\nControl:  no hypothesis · default prompt",
    title_fs=10.2, sub_fs=7.8)
box("PROG", 8.4, 4.08, 2.9, 1.0, "Logic Program Generator", C["sym"],
    sub="facts · rules · query · citations", title_fs=10.2, sub_fs=7.8)
box("SOLVE", 4.45, 4.08, 2.8, 1.0, "Symbolic Solver", C["solve"],
    sub="execute → solutions + trace", title_fs=10.6, sub_fs=7.8)

# ---- ROW 3 (left -> right) : conclude / output ---------------------------
box("RENDER", 4.45, 2.15, 2.95, 1.0, "5 · Answer Renderer (LLM)", C["llm"],
    sub="→ IRAC + plain answer (no hypothesis)", badge="5", tag="conclude",
    title_fs=10.0, sub_fs=7.8)
box("CITE", 8.4, 2.15, 2.95, 0.92, "Citation Parser & Verifier", C["det"],
    sub="from answer → fallback from program", title_fs=10.0, sub_fs=7.8)
box("REC", 12.3, 2.15, 3.0, 0.92, "Answer Record", C["out"],
    sub="answer · plain · hypothesis · citations · prolog · tokens",
    title_fs=10.2, sub_fs=7.3)
box("MET", 8.0, 0.62, 14.8, 0.8, "Metric Engine   (family: qa)", C["out"],
    sub="citation Recall / Precision / F1 · display rate · BERTScore · ROUGE / BLEU "
        "· Prolog reliability · latency", title_fs=10.6, sub_fs=7.8)

# prompt-store captions (central prompt store, inline to avoid clutter)
stat("HYDE", "prompt: hypothesis generation")
stat("RENDER", "prompt: IRAC + plain renderer")
stat("CITE", "verified vs Knowledge Graph")

# ---- main-flow arrows -----------------------------------------------------
arrow(Rt("Q"), Lf("CONCEPT"))
arrow(Rt("CONCEPT"), Lf("HYDE"))
arrow(Rt("HYDE"), Lf("RETR"), "①  embedding", ldy=0.0, lfs=7.2, lcol=C["retr"])
arrow(Bt("RETR"), T("ASM"), "retrieved clauses", ldx=1.5, ldy=0.0, lfs=7.2)
arrow(Lf("ASM"), Rt("PROG"))
arrow(Lf("PROG"), Rt("SOLVE"))
arrow(Bt("SOLVE"), T("RENDER"), "trace + solutions\n(if success)", ldx=1.5, lfs=7.0)
arrow(Rt("RENDER"), Lf("CITE"))
arrow(Rt("CITE"), Lf("REC"))
arrow(Bt("REC"), T("MET"))

# ---- hypothesis injection : single treatment-only edge (red dashed) -------
arrow((Bt("HYDE")[0] + 0.4, Bt("HYDE")[1]), (T("ASM")[0] - 0.95, T("ASM")[1]),
      color=RED, ls=(0, (5, 3)), lw=2.4, rad=-0.26, mut=16)
ax.text(10.25, 5.18, "②  rule-gen guidance\n(treatment only)", ha="center", va="center",
        fontsize=8.1, fontweight="bold", color=RED, zorder=6, linespacing=1.2,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=RED, lw=1.1, alpha=0.95))

# ---- repair loop : solver back to assembler (dashed, below the cluster) ----
arrow((Bt("SOLVE")[0] + 0.5, Bt("SOLVE")[1] + 0.02),
      (Bt("ASM")[0] - 0.2, Bt("ASM")[1] + 0.02),
      color="#7c3aed", ls=(0, (4, 3)), lw=1.7, rad=-0.30, mut=12)
ax.text(8.4, 3.06, "repair ≤ 2×  (re-inject hypothesis)", ha="center", va="center",
        fontsize=7.4, color="#6d28d9", zorder=6,
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.92))

# ---- resource feeds (dashed) ---------------------------------------------
def feed(src, dst, double=False):
    p1 = Bt(src); p2 = T(dst)
    style = "<|-|>" if double else "-|>"
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                                 lw=1.4, color="#94a3b8", linestyle=(0, (3, 3)), zorder=2))

feed("ONT", "CONCEPT")
feed("CACHE", "HYDE", double=True)
feed("EMB", "RETR")
feed("KG", "RETR")

# ---- integrity guardrail (left, ticks into the reasoning cluster) ---------
ax.add_patch(FancyBboxPatch((0.18, 3.10), 2.30, 1.80,
             boxstyle="round,pad=0.05,rounding_size=0.09",
             fc="#fef3c7", ec="#b45309", lw=1.4, ls="--", zorder=1))
ax.text(1.33, 4.78, "Integrity guardrail", ha="center", va="top",
        fontsize=9.0, fontweight="bold", color="#92400e", zorder=2)
ax.text(1.33, 4.40,
        "hypothesis is GUIDANCE\nonly: never facts,\nthresholds, or article /\nclause numbers / citations\n— only from the clauses.",
        ha="center", va="top", fontsize=7.1, color="#92400e", zorder=2, linespacing=1.3)
ax.plot([2.48, 3.00], [4.0, 4.0], ls=":", lw=1.4, color="#b45309", zorder=2)

# ---- comparison arms (left-bottom) ---------------------------------------
ax.add_patch(FancyBboxPatch((0.18, 1.12), 2.72, 1.80,
             boxstyle="round,pad=0.05,rounding_size=0.09",
             fc="#f8fafc", ec="#94a3b8", lw=1.3, zorder=1))
ax.text(1.54, 2.80, "Comparison arms", ha="center", va="top",
        fontsize=9.0, fontweight="bold", color="#334155", zorder=2)
comp = [
    ("Treatment: this pipeline", C["llm"]),
    ("Control: no hypothesis", C["sym"]),
    ("Ref: direct QA (no solver)", C["retr"]),
    ("Ref: graph RAG · LLM-only", C["input"]),
]
yy = 2.44
for text, col in comp:
    ax.text(0.32, yy, "●", ha="left", va="top", fontsize=7.6, color=col, zorder=2)
    ax.text(0.58, yy, text, ha="left", va="top", fontsize=6.9, color="#334155", zorder=2)
    yy -= 0.36

fig.savefig(OUT / "logic_lm_hyde_semantic_arm_landscape.png", dpi=190,
            bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "logic_lm_hyde_semantic_arm_landscape.svg",
            bbox_inches="tight", facecolor="white")
print("wrote", OUT / "logic_lm_hyde_semantic_arm_landscape.png")
print("wrote", OUT / "logic_lm_hyde_semantic_arm_landscape.svg")
