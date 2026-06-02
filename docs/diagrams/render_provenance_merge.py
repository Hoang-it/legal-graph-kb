"""Visualize the provenance-merge step of build_ontology_kg.py (Algorithm 1, step 6/7).

Input  : 3 provenance sources of one semantic node (clause ids, possibly overlapping)
Merge  : set-union (dedup) -> lift Clause -> Article -> Law
Output : node carrying mentioned_in_clauses / article_ids / laws

Run:  python docs/diagrams/render_provenance_merge.py
Out:  docs/diagrams/provenance_merge.png  (+ .svg)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
INK, MUTE, EDGE, DUP = "#0f172a", "#475569", "#64748b", "#dc2626"

# (fill, edge) per role
S1 = ("#dbeafe", "#2563eb")   # source 1
S2 = ("#cffafe", "#0891b2")   # source 2
S3 = ("#fef3c7", "#d97706")   # source 3
UN = ("#d1fae5", "#059669")   # union
LF = ("#ede9fe", "#7c3aed")   # lift
OP = ("#ccfbf1", "#0d9488")   # output

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")


def panel(cx, cy, w, h, header, lines, fill, edge):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.04,rounding_size=0.06",
                 facecolor=fill, edgecolor=edge, lw=1.8, zorder=3))
    ax.text(cx - w / 2 + 0.2, cy + h / 2 - 0.22, header, ha="left", va="top",
            fontsize=9.5, fontweight="bold", color=edge, zorder=4)
    yy = cy + h / 2 - 0.62
    for txt, dup in lines:
        ax.text(cx - w / 2 + 0.38, yy, txt, ha="left", va="top", fontsize=8.6,
                family="monospace", color=(DUP if dup else INK), zorder=4)
        yy -= 0.34


def op(cx, cy, w, h, label, fill, edge):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.05,rounding_size=0.1",
                 facecolor=edge, edgecolor="none", zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", color="white",
            fontsize=11.5, fontweight="bold", zorder=4)


def arrow(p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                 lw=2.1, color=EDGE, zorder=2))


# ---- title -----------------------------------------------------------------
ax.text(8, 8.62, "Provenance Merge — Input / Output", ha="center",
        fontsize=18, fontweight="bold", color=INK)
ax.text(8, 8.18, "per semantic node:  union 3 provenance sources  ->  lift  Clause -> Article -> Law",
        ha="center", fontsize=10.5, color=MUTE)

# ---- column headers --------------------------------------------------------
ax.text(3.0, 7.62, "INPUT", ha="center", fontsize=12, fontweight="bold", color=MUTE)
ax.text(9.3, 7.62, "MERGE", ha="center", fontsize=12, fontweight="bold", color=MUTE)
ax.text(13.4, 7.62, "OUTPUT", ha="center", fontsize=12, fontweight="bold", color=MUTE)
ax.text(3.0, 7.2, 'node:  subject:nguoi_lao_dong', ha="center", fontsize=9,
        style="italic", color=MUTE)

# ---- INPUT: 3 source panels ------------------------------------------------
panel(3.0, 6.05, 5.0, 1.35, "(1) EXTRACTED_FROM / DEFINES",
      [("L41_2024.A2.K1", True), ("L58_2014.A2.K1", False)], *S1)
panel(3.0, 4.30, 5.0, 1.35, "(2) source_clause of incident edges",
      [("L41_2024.A2.K1", True), ("L41_2024.A4.K7", True)], *S2)
panel(3.0, 2.35, 5.0, 1.72, "(3) existing  mentioned_in  property",
      [("L41_2024.A4.K7", True), ("L41_2024.A2.K1.a", False),
       ("L45_2019.A3.K1", False)], *S3)

# ---- MERGE: union + lift ----------------------------------------------------
op(9.3, 5.5, 3.0, 1.0, "UNION\n(set dedup)", *UN)
ax.text(9.3, 4.78, "7 raw refs  ->  5 unique clauses", ha="center", va="top",
        fontsize=8.6, color=MUTE, style="italic")
op(9.3, 3.1, 3.0, 1.0, "LIFT\nclause -> article -> law", *LF)
ax.text(9.3, 2.38, "5 clauses -> 4 articles -> 3 laws", ha="center", va="top",
        fontsize=8.6, color=MUTE, style="italic")

# arrows: sources -> union
arrow((5.55, 6.05), (7.85, 5.7))
arrow((5.55, 4.30), (7.85, 5.5))
arrow((5.55, 2.35), (7.85, 5.25))
# union -> lift
arrow((9.3, 5.0), (9.3, 3.62))
# union -> output (clauses) ; lift -> output (articles/laws)
arrow((10.8, 5.5), (11.15, 5.7))
arrow((10.8, 3.1), (11.15, 3.4))

# ---- OUTPUT node box -------------------------------------------------------
ox, oy, ow, oh = 13.4, 4.45, 4.7, 5.05
ax.add_patch(FancyBboxPatch((ox - ow / 2, oy - oh / 2), ow, oh,
             boxstyle="round,pad=0.05,rounding_size=0.08",
             facecolor=OP[0], edgecolor=OP[1], lw=2.0, zorder=3))
xL = ox - ow / 2 + 0.3
ax.text(ox, oy + oh / 2 - 0.28, "OUTPUT node — self-describing provenance",
        ha="center", va="top", fontsize=9.6, fontweight="bold", color=OP[1], zorder=4)


def field(y, title):
    ax.text(xL, y, title, ha="left", va="top", fontsize=9, fontweight="bold",
            color=INK, zorder=4)


def mono(y, txt):
    ax.text(xL + 0.25, y, txt, ha="left", va="top", fontsize=8.5,
            family="monospace", color=INK, zorder=4)


field(6.25, "mentioned_in_clauses  (5)")
for i, c in enumerate(["L41_2024.A2.K1", "L41_2024.A2.K1.a", "L41_2024.A4.K7",
                       "L45_2019.A3.K1", "L58_2014.A2.K1"]):
    mono(5.92 - i * 0.31, c)
field(4.20, "article_ids  (4)")
mono(3.88, "L41_2024.A2   L41_2024.A4")
mono(3.57, "L45_2019.A3   L58_2014.A2")
field(3.05, "laws  (3)")
mono(2.73, "L41_2024   L45_2019   L58_2014")

# ---- footnote (dedup legend) ----------------------------------------------
ax.text(3.0, 1.05, "red ids appear in >1 source  ->  set-union keeps each once",
        ha="center", fontsize=8.6, color=DUP, style="italic")

fig.savefig(OUT / "provenance_merge.png", dpi=180, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "provenance_merge.svg", bbox_inches="tight", facecolor="white")
print("wrote", OUT / "provenance_merge.png")
print("wrote", OUT / "provenance_merge.svg")
