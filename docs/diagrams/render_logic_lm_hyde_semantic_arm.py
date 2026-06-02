"""Render the per-question architecture of the Logic-LM x HyDE-semantic arm.

Module boxes use GENERIC role names (Concept Frame Builder, Hypothesis
Generator, Dense Vector Retrieval, Rule-Gen Prompt Assembler, Logic Program
Generator, Symbolic Solver, Answer Renderer, Citation Parser) rather than
project-specific class/file names, so the figure reads in any context. The
flow, the five reasoning phases, the single hypothesis-injection point
(treatment-only), the repair loop, the integrity guardrail, and the supporting
stores all mirror the runtime code.

Run:  python docs/diagrams/render_logic_lm_hyde_semantic_arm.py
Out:  docs/diagrams/logic_lm_hyde_semantic_arm.png  (+ .svg)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent

# Consistent palette (shared hex vocabulary with the other repo diagrams).
C = {
    "input": "#64748b",  # slate  — user input
    "det":   "#2563eb",  # blue   — deterministic step (no LLM)
    "llm":   "#d97706",  # amber  — LLM call
    "retr":  "#0891b2",  # cyan   — retrieval (encode + vector search)
    "sym":   "#7c3aed",  # purple — symbolic program build
    "solve": "#059669",  # green  — symbolic solver
    "store": "#475569",  # gray   — supporting store / model
    "out":   "#0d9488",  # teal   — output / metrics
}
EDGE = "#475569"
INK = "#0f172a"
MUTE = "#475569"
RED = "#dc2626"

fig, ax = plt.subplots(figsize=(15, 20))
ax.set_xlim(0, 15)
ax.set_ylim(0, 20)
ax.axis("off")

B: dict[str, tuple[float, float, float, float]] = {}  # name -> (cx, cy, w, h)


def box(name, cx, cy, w, h, title, color, sub="", badge=None, tag=None,
        title_fs=12.5, sub_fs=8.6):
    B[name] = (cx, cy, w, h)
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=0, facecolor=color, alpha=0.97, zorder=3))
    if sub:
        ax.text(cx, cy + h / 2 - 0.27, title, ha="center", va="top",
                color="white", fontsize=title_fs, fontweight="bold", zorder=4)
        ax.text(cx, cy + h / 2 - 0.27 - 0.40, sub, ha="center", va="top",
                color="white", fontsize=sub_fs, zorder=4, linespacing=1.25)
    else:
        ax.text(cx, cy, title, ha="center", va="center", color="white",
                fontsize=title_fs, fontweight="bold", zorder=4)
    if badge is not None:
        ax.text(cx - w / 2, cy + h / 2, str(badge), ha="center", va="center",
                fontsize=10, fontweight="bold", color=color, zorder=5,
                bbox=dict(boxstyle="circle,pad=0.20", fc="white", ec=color, lw=1.7))
    if tag is not None:
        ax.text(cx + w / 2 - 0.10, cy + h / 2 + 0.02, tag, ha="right", va="bottom",
                fontsize=8.2, style="italic", color=MUTE, zorder=5)


def stat(name, text, dy=0.16, color=MUTE):
    cx, cy, w, h = B[name]
    ax.text(cx, cy - h / 2 - dy, text, ha="center", va="top",
            fontsize=8.1, color=color, style="italic", zorder=4)


def T(n):
    cx, cy, w, h = B[n]; return (cx, cy + h / 2)
def Bt(n):
    cx, cy, w, h = B[n]; return (cx, cy - h / 2)
def L(n):
    cx, cy, w, h = B[n]; return (cx - w / 2, cy)
def R(n):
    cx, cy, w, h = B[n]; return (cx + w / 2, cy)


def arrow(p1, p2, label=None, color=EDGE, ls="-", lw=2.1, rad=0.0,
          mut=15, lfs=8.0, lcol=None, ldx=0.0, ldy=0.16, la="center", z=2):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=mut, lw=lw, color=color,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=z))
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + ldx, (p1[1] + p2[1]) / 2 + ldy
        ax.text(mx, my, label, ha=la, va="center", fontsize=lfs,
                color=lcol or INK, zorder=6, linespacing=1.2,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.9))


# --------------------------------------------------------------------------
# Title + subtitle
# --------------------------------------------------------------------------
ax.text(7.5, 19.55, "Logic-LM × HyDE-semantic  —  per-question arm architecture",
        ha="center", fontsize=18.5, fontweight="bold", color=INK)
ax.text(7.5, 19.08, "generic module roles  ·  one hypothesis with two uses  ·  "
        "treatment vs. control differs at a single injection point",
        ha="center", fontsize=10.5, color=MUTE)

# --------------------------------------------------------------------------
# Legend
# --------------------------------------------------------------------------
legend = [
    ("Input", C["input"]), ("Deterministic (no LLM)", C["det"]),
    ("LLM call", C["llm"]), ("Retrieval", C["retr"]),
    ("Symbolic build", C["sym"]), ("Solver", C["solve"]),
    ("Store / model", C["store"]), ("Output", C["out"]),
]
lx = 0.6
for lab, col in legend:
    ax.add_patch(FancyBboxPatch((lx, 18.42), 0.30, 0.26, boxstyle="round,pad=0.02",
                                fc=col, ec="none", zorder=4))
    ax.text(lx + 0.40, 18.55, lab, ha="left", va="center", fontsize=8.4, color=MUTE)
    lx += 0.55 + 0.085 * len(lab) + 0.42

# --------------------------------------------------------------------------
# Main per-question pipeline (top -> bottom), generic module names
# --------------------------------------------------------------------------
CXM = 5.8
WM = 5.4
box("Q", CXM, 18.5, 3.6, 0.72, "User Question", C["input"])
box("CONCEPT", CXM, 17.15, WM, 1.05, "1 · Concept Frame Builder", C["det"],
    sub="question → domain concept frame\n(deterministic · no LLM · no network)",
    badge="1", tag="understand")
box("HYDE", CXM, 15.45, WM, 1.18, "2 · Hypothesis Generator", C["llm"],
    sub="concept frame → hypothetical legal passage\ngeneral wording · no article / clause numbers",
    badge="2", tag="hypothesis")
box("RETR", CXM, 13.55, WM, 1.22, "3 · Dense Vector Retrieval", C["retr"],
    sub="embed hypothesis (mean-pool) → vector search\n→ ranked legal clauses (grounding + citations)",
    badge="3", tag="search")

# Reasoning cluster container
ax.add_patch(FancyBboxPatch((2.78, 6.55), 6.06, 5.98,
             boxstyle="round,pad=0.04,rounding_size=0.10",
             fc="#eef2f7", ec="#94a3b8", lw=1.5, ls="--", zorder=1))
ax.text(2.95, 12.40, "4 · Symbolic Reasoning", ha="left", va="top",
        fontsize=10.5, fontweight="bold", color="#334155", zorder=2)
ax.text(8.66, 12.40, "[ reason ]", ha="right", va="top", fontsize=8.6,
        style="italic", color="#64748b", zorder=2)

box("CLIENT", CXM, 11.25, WM, 1.30, "Rule-Gen Prompt Assembler  (LLM)", C["llm"],
    sub="Treatment:  + hypothesis  ·  HyDE rule-gen prompt\nControl:   no hypothesis  ·  default rule-gen prompt",
    title_fs=11.5)
box("PROG", CXM, 9.55, WM, 1.12, "Logic Program Generator + Validator", C["sym"],
    sub="emit program: facts · rules · query · citations",
    title_fs=11.5)
box("SOLVE", CXM, 8.00, WM, 1.10, "Symbolic Solver", C["solve"],
    sub="execute program → solutions + reasoning trace",
    title_fs=12)

box("RENDER", CXM, 5.60, WM, 1.16, "5 · Answer Renderer  (LLM)", C["llm"],
    sub="reasoning trace → IRAC + plain-language answer\nhypothesis is NOT used in this step",
    badge="5", tag="conclude")
box("CITE", CXM, 4.00, WM, 1.00, "Citation Parser & Verifier", C["det"],
    sub="citations from answer → fallback from program facts",
    title_fs=11.5)
box("REC", CXM, 2.52, 4.9, 0.95, "Answer Record", C["out"],
    sub="answer · plain_answer · hypothesis · citations · prolog status · tokens",
    title_fs=11.5, sub_fs=8.0)
box("MET", 6.0, 1.05, 11.2, 0.92, "Metric Engine   (family: qa)", C["out"],
    sub="citation Recall / Precision / F1 · display rate · BERTScore · ROUGE / BLEU "
        "· Prolog reliability · latency", title_fs=11.5, sub_fs=8.0)

# prompt-store captions (central prompt store, shown inline to avoid clutter)
stat("HYDE", "system prompt: hypothesis generation")
stat("CLIENT", "system prompt: rule-gen (default / HyDE variant)")
stat("RENDER", "system prompt: IRAC + plain renderer")
stat("CITE", "verified against the Knowledge Graph")

# --------------------------------------------------------------------------
# Supporting stores / models (right column)
# --------------------------------------------------------------------------
CXR = 11.9
WR = 4.8
box("ONT", CXR, 17.15, WR, 0.92, "Domain Ontology / Concept Store", C["store"], title_fs=10.5)
box("CACHE", CXR, 15.45, WR, 0.92, "Hypothesis Cache  (disk)", C["store"], title_fs=10.5,
    sub="shared by treatment + control", sub_fs=8.0)
box("EMB", CXR, 13.95, WR, 0.82, "Embedding Model", C["store"], title_fs=10.5)
box("KG", CXR, 12.85, WR, 0.82, "Knowledge Graph  (vector index)", C["store"], title_fs=10.5)

# Comparison arms (context of the experiment) — bottom-right
ax.add_patch(FancyBboxPatch((9.30, 3.55), 5.25, 3.95,
             boxstyle="round,pad=0.05,rounding_size=0.10",
             fc="#f8fafc", ec="#94a3b8", lw=1.4, zorder=1))
ax.text(11.92, 7.30, "Comparison arms (same experiment)", ha="center", va="top",
        fontsize=10, fontweight="bold", color="#334155", zorder=2)
comp = [
    ("● Treatment", "this pipeline — hypothesis → rule-gen", C["llm"]),
    ("● Control", "same retrieval + cache, NO hypothesis,\n   default rule-gen prompt", C["sym"]),
    ("● Reference", "direct QA on the same retrieval\n   (no solver layer)", C["retr"]),
    ("● Reference", "graph RAG  ·  LLM-only", C["input"]),
]
yy = 6.78
for head, body, col in comp:
    ax.text(9.55, yy, head, ha="left", va="top", fontsize=8.8, fontweight="bold", color=col, zorder=2)
    ax.text(9.95, yy - 0.30, body, ha="left", va="top", fontsize=8.0, color="#475569",
            zorder=2, linespacing=1.2)
    yy -= 0.30 + 0.30 * (body.count("\n") + 1) + 0.12

# --------------------------------------------------------------------------
# Integrity guardrail (left-lower) — the key design constraint
# --------------------------------------------------------------------------
ax.add_patch(FancyBboxPatch((0.20, 6.62), 2.45, 2.55,
             boxstyle="round,pad=0.05,rounding_size=0.10",
             fc="#fef3c7", ec="#b45309", lw=1.5, ls="--", zorder=1))
ax.text(1.42, 9.00, "Integrity guardrail", ha="center", va="top",
        fontsize=9.6, fontweight="bold", color="#92400e", zorder=2)
ax.text(1.42, 8.55,
        "hypothesis is GUIDANCE\nONLY — never a source of\nfacts, thresholds, article /\nclause numbers, or citations.\n"
        "Those come only from the\nretrieved clauses & question.",
        ha="center", va="top", fontsize=7.7, color="#92400e", zorder=2, linespacing=1.3)
# tick from guardrail to the rule-gen assembler
ax.plot([1.42, 2.55], [9.17, 10.78], ls=":", lw=1.3, color="#b45309", zorder=1)

# --------------------------------------------------------------------------
# Main flow arrows (solid)
# --------------------------------------------------------------------------
arrow(Bt("Q"), T("CONCEPT"))
arrow(Bt("CONCEPT"), T("HYDE"), "concept frame + cache key", ldx=2.05, lfs=7.6)
arrow(Bt("HYDE"), T("RETR"), "hypothesis → embedding (both arms)", ldx=2.35, lfs=7.6)
arrow(Bt("RETR"), T("CLIENT"), "retrieved clauses", ldx=1.75, lfs=7.6)
arrow(Bt("CLIENT"), T("PROG"), "generated program (JSON)", ldx=1.95, lfs=7.5)
arrow(Bt("PROG"), T("SOLVE"), "validated program + query", ldx=2.0, lfs=7.5)
arrow(Bt("SOLVE"), T("RENDER"), "trace + solutions  (if success)", ldx=2.15, lfs=7.5)
arrow(Bt("RENDER"), T("CITE"), "answer text", ldx=1.55, lfs=7.6)
arrow(Bt("CITE"), T("REC"))
arrow(Bt("REC"), T("MET"))

# --------------------------------------------------------------------------
# Hypothesis injection — the single treatment-only edge (left, red dashed)
# --------------------------------------------------------------------------
arrow((L("HYDE")[0], L("HYDE")[1] - 0.18), (L("CLIENT")[0], L("CLIENT")[1] + 0.18),
      color=RED, ls=(0, (5, 3)), lw=2.2, rad=0.42, mut=15)
ax.text(1.30, 13.35, "hypothesis\n→ rule-gen\n(treatment only)", ha="center", va="center",
        fontsize=8.4, fontweight="bold", color=RED, zorder=6, linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=RED, lw=1.2, alpha=0.95))

# --------------------------------------------------------------------------
# Repair loop (right of cluster, dashed) — solver back to assembler
# --------------------------------------------------------------------------
arrow((R("SOLVE")[0], R("SOLVE")[1] + 0.10), (R("CLIENT")[0], R("CLIENT")[1] - 0.20),
      color="#7c3aed", ls=(0, (4, 3)), lw=1.8, rad=-0.55, mut=13)
ax.text(9.18, 9.55, "repair ≤ 2×\n(re-inject\nhypothesis)", ha="left", va="center",
        fontsize=7.8, color="#6d28d9", zorder=6, linespacing=1.2,
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.9))

# --------------------------------------------------------------------------
# Resource feeds (right, short dashed)
# --------------------------------------------------------------------------
def feed(src, dst, label=None, ldy=0.18, double=False):
    p1 = L(src); p2 = (R(dst)[0], R(dst)[1])
    style = "<|-|>" if double else "-|>"
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12,
                                 lw=1.5, color="#94a3b8", linestyle=(0, (3, 3)), zorder=2))
    if label:
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + ldy, label, ha="center",
                va="center", fontsize=7.3, color=MUTE, zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

feed("ONT", "CONCEPT", "concepts + entities")
feed("CACHE", "HYDE", "cache", double=True)
feed("EMB", "RETR", "encode")
feed("KG", "RETR", "vector index")

fig.savefig(OUT / "logic_lm_hyde_semantic_arm.png", dpi=180, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "logic_lm_hyde_semantic_arm.svg", bbox_inches="tight", facecolor="white")
print("wrote", OUT / "logic_lm_hyde_semantic_arm.png")
print("wrote", OUT / "logic_lm_hyde_semantic_arm.svg")
