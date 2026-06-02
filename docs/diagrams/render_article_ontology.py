"""Visualize the ontology subgraph of ONE article from ontology_kg_full.json.

Article: Điều 94, Luật BHXH 41/2024 — "Đối tượng và điều kiện hưởng trợ cấp thai sản".
Subgraph = every semantic edge whose source_clause falls in Điều 94, plus endpoints.
Data is real (names + provenance clauses taken verbatim from the export).

Run:  python docs/diagrams/render_article_ontology.py
Out:  docs/diagrams/article_ontology_A94.png  (+ .svg)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
INK, MUTE, EDGE = "#0f172a", "#475569", "#475569"

COL = {
    "subject": "#2563eb",    # đối tượng
    "benefit": "#059669",    # chế độ
    "rule": "#7c3aed",       # LegalRule (Logic-LM)
    "cond": "#d97706",       # LegalCondition (Logic-LM)
}

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")


def node(cx, cy, w, h, lines, color, fs=11):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.05,rounding_size=0.1",
                 facecolor=color, edgecolor="none", alpha=0.96, zorder=4))
    ax.text(cx, cy, "\n".join(lines), ha="center", va="center", color="white",
            fontsize=fs, fontweight="bold", zorder=5)


def arrow(p1, p2, label=None, color=EDGE):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15,
                 lw=2.0, color=color, zorder=3))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.16, label, ha="center", va="bottom", fontsize=8.2,
                color=INK, zorder=5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))


# ---- title -----------------------------------------------------------------
ax.text(8, 8.64, "Ontology subgraph — Điều 94, Luật BHXH 41/2024/QH15",
        ha="center", fontsize=17, fontweight="bold", color=INK)
ax.text(8, 8.2, "Đối tượng & điều kiện hưởng trợ cấp thai sản   ·   "
        "11 node / 10 cạnh trích từ ontology_kg_full.json (theo source_clause ∈ Điều 94)",
        ha="center", fontsize=10, color=MUTE)

# ---- legend ----------------------------------------------------------------
leg = [("Subject (đối tượng)", COL["subject"], 0.8),
       ("Benefit (chế độ)", COL["benefit"], 4.3),
       ("LegalRule (Logic-LM)", COL["rule"], 7.6),
       ("LegalCondition (Logic-LM)", COL["cond"], 11.5)]
for label, c, x in leg:
    ax.add_patch(FancyBboxPatch((x, 7.5), 0.3, 0.28, boxstyle="round,pad=0.02",
                                fc=c, ec="none", zorder=4))
    ax.text(x + 0.42, 7.64, label, ha="left", va="center", fontsize=9, color=MUTE)

# ====== TOP cluster — ENTITLED_TO ==========================================
ax.text(3.9, 7.0, "AI ĐƯỢC HƯỞNG   —   ENTITLED_TO", ha="center",
        fontsize=11, fontweight="bold", color=COL["subject"])

# subjects (left) + benefit hub (center)
node(2.3, 6.35, 2.7, 0.95, ["Lao động nam"], COL["subject"])
node(2.3, 5.05, 2.7, 0.95, ["Cha"], COL["subject"])
node(2.3, 3.75, 2.7, 0.95, ["Mẹ"], COL["subject"])
node(8.6, 5.05, 3.2, 1.45, ["Chế độ thai sản"], COL["benefit"], fs=13)

bx = 8.6 - 1.6  # benefit left edge
arrow((3.65, 6.35), (bx, 5.45), "Đ94.K1.b")
arrow((3.65, 5.05), (bx, 5.05), "Đ94.K2 · K5")
arrow((3.65, 3.75), (bx, 4.65), "Đ94.K2 · K6")

# ---- divider ---------------------------------------------------------------
ax.plot([0.6, 15.4], [3.05, 3.05], ls=(0, (6, 5)), lw=1.1, color="#cbd5e1", zorder=1)

# ====== BOTTOM cluster — REQUIRES (Logic-LM enrichment) =====================
# dashed container to mark the inference layer
ax.add_patch(FancyBboxPatch((1.2, 0.35), 12.6, 2.2,
             boxstyle="round,pad=0.04,rounding_size=0.06",
             fc="#faf5ff", ec="#7c3aed", lw=1.3, ls="--", zorder=1))
ax.text(5.2, 2.74, "ĐIỀU KIỆN   —   REQUIRES   (lớp suy luận Logic-LM: predicate canonical)",
        ha="center", fontsize=11, fontweight="bold", color=COL["rule"])

node(3.4, 1.95, 3.6, 0.98, ["eligible_maternity_benefit", "(Điều 94 · K1)"], COL["rule"], fs=9)
node(3.4, 0.82, 3.6, 0.98, ["eligible_maternity_benefit", "(Điều 94 · K2)"], COL["rule"], fs=9)
node(9.7, 1.95, 4.0, 0.98, ["social_insurance_", "contribution_period"], COL["cond"], fs=9)
node(9.7, 0.82, 4.0, 0.98, ["mother_deceased_", "after_birth"], COL["cond"], fs=9)

arrow((5.2, 1.95), (7.7, 1.95), "Đ94.K1")
arrow((5.2, 0.82), (7.7, 0.82), "Đ94.K2")

# ---- footnote --------------------------------------------------------------
ax.text(15.4, 0.15, "tên đối tượng/chế độ = tiếng Việt (lớp B3);  predicate = tiếng Anh (lớp Logic-LM)",
        ha="right", va="bottom", fontsize=8, color=MUTE, style="italic")

fig.savefig(OUT / "article_ontology_A94.png", dpi=180, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "article_ontology_A94.svg", bbox_inches="tight", facecolor="white")
print("wrote", OUT / "article_ontology_A94.png")
print("wrote", OUT / "article_ontology_A94.svg")
