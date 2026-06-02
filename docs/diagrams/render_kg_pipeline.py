"""Render the Knowledge Graph construction pipeline as a slide-ready figure.

Module boxes use GENERIC role names (Document Parser, Rule-based Extractor,
LLM Semantic Extractor, Merge & Normalize, Embedding Encoder, Graph Loader)
rather than project-specific file names, so the figure reads in any context.
Stats are the real numbers from data/graph/processed/extraction_summary.md
and embeddings.parquet.

Run:  python docs/diagrams/render_kg_pipeline.py
Out:  docs/diagrams/kg_build_pipeline.png  (+ .svg)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent

C = {
    "input": "#64748b",  # slate  — source
    "det": "#2563eb",    # blue   — deterministic / regex
    "llm": "#d97706",    # amber  — LLM
    "merge": "#059669",  # green  — merge & validate
    "enc": "#7c3aed",    # purple — neural encoder
    "store": "#0d9488",  # teal   — storage + index
}
EDGE = "#475569"
INK = "#0f172a"
MUTE = "#334155"

fig, ax = plt.subplots(figsize=(17, 9))
ax.set_xlim(0, 17)
ax.set_ylim(0, 9)
ax.axis("off")


def box(cx, cy, w, h, lines, color, badge=None):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=0, facecolor=color, alpha=0.96, zorder=3,
        )
    )
    ax.text(cx, cy, "\n".join(lines), ha="center", va="center",
            color="white", fontsize=12.5, fontweight="bold", zorder=4)
    if badge:
        ax.text(cx - w / 2, cy + h / 2, badge, ha="center", va="center",
                fontsize=9, fontweight="bold", color=color, zorder=5,
                bbox=dict(boxstyle="circle,pad=0.22", fc="white", ec=color, lw=1.6))


def stat(cx, cy_box, h, text):
    ax.text(cx, cy_box - h / 2 - 0.18, text, ha="center", va="top",
            fontsize=8.4, color=MUTE, style="italic", zorder=4)


def arrow(p1, p2, label=None, rad=0.0, dy=0.16):
    ax.add_patch(
        FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16, lw=2.1,
                        color=EDGE, connectionstyle=f"arc3,rad={rad}", zorder=2)
    )
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + dy, label, ha="center", va="bottom", fontsize=8,
                color=INK, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


# ---- title + subtitle -----------------------------------------------------
ax.text(8.5, 8.62, "Knowledge Graph Construction Pipeline", ha="center",
        fontsize=18, fontweight="bold", color=INK)
ax.text(8.5, 8.16, "Vietnamese legal corpus  ·  generic module roles  ·  "
        "blue stages are deterministic (no LLM)", ha="center", fontsize=10.5, color=MUTE)

# ---- legend ---------------------------------------------------------------
legend = [
    ("Source", C["input"], 0.9),
    ("Deterministic", C["det"], 2.55),
    ("LLM", C["llm"], 4.9),
    ("Encoder", C["enc"], 6.25),
    ("Merge + validate", C["merge"], 8.1),
    ("Storage + index", C["store"], 10.8),
]
for label, col, x in legend:
    ax.add_patch(FancyBboxPatch((x, 7.42), 0.3, 0.28, boxstyle="round,pad=0.02",
                                fc=col, ec="none", zorder=4))
    ax.text(x + 0.42, 7.56, label, ha="left", va="center", fontsize=9, color=MUTE)

# ---- boxes ----------------------------------------------------------------
# (cx, cy, w, h)
P_IN = (1.5, 4.4, 2.1, 1.25)
P_B1 = (4.3, 4.4, 2.3, 1.25)
P_B2 = (7.4, 5.9, 2.5, 1.15)
P_B3 = (7.4, 2.95, 2.5, 1.15)
P_B4 = (10.6, 4.4, 2.4, 1.25)
P_B5 = (13.6, 5.9, 2.4, 1.1)
P_B6 = (13.6, 2.95, 2.4, 1.1)
P_VS = (16.1, 5.9, 1.7, 1.0)
P_DB = (16.1, 2.95, 1.7, 1.0)

box(*P_IN[:2], P_IN[2], P_IN[3], ["Legal", "Documents", "(.docx)"], C["input"])
box(*P_B1[:2], P_B1[2], P_B1[3], ["Document", "Parser"], C["det"], "B1")
box(*P_B2[:2], P_B2[2], P_B2[3], ["Rule-based", "Extractor"], C["det"], "B2")
box(*P_B3[:2], P_B3[2], P_B3[3], ["LLM Semantic", "Extractor"], C["llm"], "B3")
box(*P_B4[:2], P_B4[2], P_B4[3], ["Merge &", "Normalize"], C["merge"], "B4")
box(*P_B5[:2], P_B5[2], P_B5[3], ["Embedding", "Encoder"], C["enc"], "B5")
box(*P_B6[:2], P_B6[2], P_B6[3], ["Graph", "Loader"], C["store"], "B6")
box(*P_VS[:2], P_VS[2], P_VS[3], ["Vector", "Store"], C["store"])
box(*P_DB[:2], P_DB[2], P_DB[3], ["Graph", "Database"], C["store"])

# ---- stats under boxes ----------------------------------------------------
stat(*P_IN[:2], P_IN[3], "5 laws  ·  507 articles")
stat(*P_B1[:2], P_B1[3], "regex state machine, 0 LLM\n-> Structured Tree (3,112 nodes)")
stat(*P_B2[:2], P_B2[3], "cross-refs · citations · definitions\n-> 817 internal + 65 external refs")
stat(*P_B3[:2], P_B3[3], "entities + semantic relations\n-> 303 semantic nodes")
stat(*P_B4[:2], P_B4[3], "dedup + fail-fast validate\n-> 3,415 nodes / 5,628 edges")
stat(*P_B5[:2], P_B5[3], "BGE-M3, 1024-d, normalized")
stat(*P_B6[:2], P_B6[3], "UNWIND / MERGE (idempotent)")
stat(*P_VS[:2], P_VS[3], "3,014 vectors")
stat(*P_DB[:2], P_DB[3], "vector + fulltext\n+ constraints")

# ---- arrows ---------------------------------------------------------------
arrow((P_IN[0] + P_IN[2] / 2, 4.4), (P_B1[0] - P_B1[2] / 2, 4.4))
# B1 -> split
arrow((P_B1[0] + P_B1[2] / 2, 4.7), (P_B2[0] - P_B2[2] / 2, 5.6))
arrow((P_B1[0] + P_B1[2] / 2, 4.1), (P_B3[0] - P_B3[2] / 2, 3.25))
# extractors -> merge
arrow((P_B2[0] + P_B2[2] / 2, 5.6), (P_B4[0] - P_B4[2] / 2, 4.75), "reference layer")
arrow((P_B3[0] + P_B3[2] / 2, 3.25), (P_B4[0] - P_B4[2] / 2, 4.05), "semantic layer")
# merge -> split
arrow((P_B4[0] + P_B4[2] / 2, 4.75), (P_B5[0] - P_B5[2] / 2, 5.6), "unified KG")
arrow((P_B4[0] + P_B4[2] / 2, 4.05), (P_B6[0] - P_B6[2] / 2, 3.25))
# encoder/loader -> stores
arrow((P_B5[0] + P_B5[2] / 2, 5.9), (P_VS[0] - P_VS[2] / 2, 5.9))
arrow((P_B6[0] + P_B6[2] / 2, 2.95), (P_DB[0] - P_DB[2] / 2, 2.95))

# ---- provenance guardrail band -------------------------------------------
gb_x0, gb_x1, gb_y, gb_h = 6.0, 16.95, 1.05, 0.95
ax.add_patch(
    FancyBboxPatch((gb_x0, gb_y - gb_h / 2), gb_x1 - gb_x0, gb_h,
                   boxstyle="round,pad=0.05,rounding_size=0.12",
                   fc="#fef3c7", ec="#b45309", lw=1.5, ls="--", zorder=1)
)
ax.text((gb_x0 + gb_x1) / 2, gb_y + 0.18,
        "Provenance & Anti-fabrication Guardrail", ha="center", va="center",
        fontsize=10.5, fontweight="bold", color="#92400e", zorder=2)
ax.text((gb_x0 + gb_x1) / 2, gb_y - 0.2,
        "(1) schema validation   ->   (2) DB constraints: unique id + required text / "
        "mentioned_in / source_clause   ->   (3) byte-for-byte substring check",
        ha="center", va="center", fontsize=8.6, color="#92400e", zorder=2)
# dashed up-ticks from guardrail to the stages it protects
for sx, sy in [(P_B3[0], P_B3[1] - P_B3[3] / 2), (P_B4[0], P_B4[1] - P_B4[3] / 2),
               (P_B6[0], P_B6[1] - P_B6[3] / 2)]:
    ax.plot([sx, sx], [gb_y + gb_h / 2, sy - 0.55], ls=":", lw=1.2, color="#b45309", zorder=1)

fig.savefig(OUT / "kg_build_pipeline.png", dpi=180, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "kg_build_pipeline.svg", bbox_inches="tight", facecolor="white")
print("wrote", OUT / "kg_build_pipeline.png")
print("wrote", OUT / "kg_build_pipeline.svg")
