"""Experiment 06 metrics — citation recall / precision / F1 at K, retrieval-only.

Reads per-arm records produced by ``scripts/exp06_run.py``:

    experiments/06_retrieval_dense_vs_full/results/<arm>/A<stt>.json

For each ``(arm, K)`` over ``K ∈ {5, 10, 12, 20, 30, all}``, computes:

- per-question recall = |retrieved@K ∩ gold| / |gold|
- per-question precision = |retrieved@K ∩ gold| / |retrieved@K|  (|retrieved|=0 → 0)
- per-question F1 = 2·P·R / (P+R)  (P+R=0 → 0)
- macro-averaged across questions with non-empty gold

Reports macro overall and stratified by gold-corpus category
(in_corpus / mixed / ooc / unparseable) — same categorisation as
``scripts/audit_retrieval.py``.

Also re-derives ``gold_citations_normalized.json`` for this experiment via
``eval_core.gold.validate_gold_citations`` so the experiment is
self-contained and the gold is auditable.

Writes:

- ``metrics/gold_citations_normalized.json``
- ``metrics/academic_metrics.json``
- ``metrics/academic_metrics.csv``
- ``report/academic_report.md``

Usage::

    python scripts/exp06_metrics.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eval_core.gold import validate_gold_citations
from src.legal_metadata import load_law_metadata

EXP_DIR = _REPO / "experiments" / "06_retrieval_dense_vs_full"
RESULTS_DIR = EXP_DIR / "results"
METRICS_DIR = EXP_DIR / "metrics"
REPORT_DIR = EXP_DIR / "report"
QUESTIONS_PATH = _REPO / "data" / "eval" / "questions_200.json"
REGISTRY_PATH = _REPO / "data" / "legal_sources.yaml"

ARMS = ("dense", "full_rerank")
KS: tuple[int | None, ...] = (5, 10, 12, 20, 30, None)  # None = all retrieved

_RE_CODE = re.compile(r"\d+/\d{4}/(?:QH\d+|N[ĐD]-CP|NQ-CP|TT-[A-Z]+|CP|TTg)")


# ---------------------------------------------------------------------------
# Gold categorisation — same heuristic used in audit_retrieval.py
# ---------------------------------------------------------------------------


def categorize(raw, in_corpus_codes: set[str]) -> str:
    if not raw:
        return "unparseable"
    if isinstance(raw, list):
        raw = "\n".join(str(x) for x in raw)
    hits = _RE_CODE.findall(raw)
    if not hits:
        return "unparseable"
    in_kg = sum(1 for h in hits if h in in_corpus_codes)
    if in_kg == len(hits):
        return "in_corpus"
    if in_kg == 0:
        return "ooc"
    return "mixed"


# ---------------------------------------------------------------------------
# Per-question scoring
# ---------------------------------------------------------------------------


def _retrieved_at_k(retrieved: list[str], k: int | None) -> list[str]:
    return retrieved if k is None else retrieved[:k]


def recall(retrieved_k: list[str], gold: set[str]) -> float | None:
    if not gold:
        return None
    pool = set(retrieved_k)
    return len(gold & pool) / len(gold)


def precision(retrieved_k: list[str], gold: set[str]) -> float | None:
    if not retrieved_k:
        return 0.0 if gold else None
    pool = set(retrieved_k)
    return len(gold & pool) / len(pool)


def f1_score(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    if (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


def r_precision(retrieved: list[str], gold: set[str]) -> float | None:
    """Precision at K=|gold|. When |gold|<<K_max this is the IR-standard
    way to read precision/recall as a single number — at K=|gold| the
    two are identical, so the value is exactly the fraction of gold the
    retriever placed in the top-|gold| positions."""
    if not gold:
        return None
    k = len(gold)
    pool = set(retrieved[:k])
    return len(gold & pool) / k


def reciprocal_rank(retrieved: list[str], gold: set[str]) -> float | None:
    """1 / rank of the first retrieved article that is in gold (1-indexed).
    0 if no gold article is retrieved at all."""
    if not gold:
        return None
    for i, aid in enumerate(retrieved, start=1):
        if aid in gold:
            return 1.0 / i
    return 0.0


import math


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float | None:
    """Binary-relevance NDCG@K. Gain = 1 if retrieved[i] ∈ gold else 0.
    Discount = 1/log2(i+1). Normalised by IDCG = sum_{i=1..min(|gold|,k)} 1/log2(i+1)."""
    if not gold:
        return None
    pool = retrieved[:k]
    dcg = 0.0
    for i, aid in enumerate(pool, start=1):
        if aid in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _macro(values) -> float | None:
    vs = [v for v in values if v is not None]
    return round(mean(vs), 4) if vs else None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def load_gold() -> dict[int, list[str]]:
    """Re-derive gold for this experiment via the canonical validator."""
    ok, summary = validate_gold_citations(
        questions_path=QUESTIONS_PATH,
        registry_path=REGISTRY_PATH,
        out_dir=METRICS_DIR,
    )
    if not ok:
        print(
            f"FAIL: gold validation found {summary['n_errors']} errors; "
            f"see {summary['errors_path']}",
            file=sys.stderr,
        )
        sys.exit(1)
    data = json.loads(Path(summary["normalized_path"]).read_text(encoding="utf-8"))
    return {int(k): v.get("gold_articles") or [] for k, v in data["records"].items()}


def load_arm_records(arm: str) -> dict[int, dict]:
    arm_dir = RESULTS_DIR / arm
    if not arm_dir.is_dir():
        raise FileNotFoundError(arm_dir)
    out: dict[int, dict] = {}
    for p in sorted(arm_dir.glob("A*.json")):
        if p.name.endswith(".error.json"):
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        out[int(rec["stt"])] = rec
    return out


def score_arm(
    arm: str,
    records: dict[int, dict],
    gold_map: dict[int, list[str]],
    questions: dict[int, dict],
    in_corpus_codes: set[str],
) -> list[dict]:
    rows: list[dict] = []
    for stt, rec in records.items():
        retrieved = list(
            (rec.get("retrieval_only") or {}).get("final_article_ids") or []
        )
        gold = set(gold_map.get(stt) or [])
        row: dict = {
            "stt": stt,
            "arm": arm,
            "n_gold": len(gold),
            "n_retrieved_all": len(retrieved),
            "retrieved": retrieved,
            "gold": sorted(gold),
            "category": categorize(
                questions[stt].get("gold_citations_raw"), in_corpus_codes
            ),
            "elapsed_s": (rec.get("retrieval_only") or {}).get("elapsed_s"),
        }
        for k in KS:
            r_k = _retrieved_at_k(retrieved, k)
            r = recall(r_k, gold)
            p = precision(r_k, gold)
            f1 = f1_score(p, r)
            kk = "all" if k is None else str(k)
            row[f"recall@{kk}"] = r
            row[f"precision@{kk}"] = p
            row[f"f1@{kk}"] = f1
            row[f"n_retrieved@{kk}"] = len(r_k)
        # Rank-aware metrics — not capped by K-vs-|gold| asymmetry.
        row["r_precision"] = r_precision(retrieved, gold)
        row["mrr"] = reciprocal_rank(retrieved, gold)
        row["ndcg@10"] = ndcg_at_k(retrieved, gold, 10)
        row["ndcg@all"] = ndcg_at_k(retrieved, gold, len(retrieved) or 1)
        rows.append(row)
    return rows


def aggregate_macro(rows: list[dict]) -> dict:
    summary: dict = {"n": len(rows)}
    for k in KS:
        kk = "all" if k is None else str(k)
        summary[f"recall@{kk}"] = _macro([r[f"recall@{kk}"] for r in rows])
        summary[f"precision@{kk}"] = _macro([r[f"precision@{kk}"] for r in rows])
        summary[f"f1@{kk}"] = _macro([r[f"f1@{kk}"] for r in rows])
        summary[f"avg_n_retrieved@{kk}"] = round(
            mean([r[f"n_retrieved@{kk}"] for r in rows]), 2
        ) if rows else None
    summary["r_precision"] = _macro([r["r_precision"] for r in rows])
    summary["mrr"] = _macro([r["mrr"] for r in rows])
    summary["ndcg@10"] = _macro([r["ndcg@10"] for r in rows])
    summary["ndcg@all"] = _macro([r["ndcg@all"] for r in rows])
    latencies = [r["elapsed_s"] for r in rows if r["elapsed_s"] is not None]
    summary["avg_elapsed_s"] = round(mean(latencies), 3) if latencies else None
    return summary


def aggregate_stratified(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cat in ("in_corpus", "mixed", "ooc", "unparseable"):
        sub = [r for r in rows if r["category"] == cat]
        out[cat] = aggregate_macro(sub) if sub else {"n": 0}
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_csv(per_arm_rows: dict[str, list[dict]]) -> Path:
    out = METRICS_DIR / "academic_metrics.csv"
    field_ks = [("all" if k is None else str(k)) for k in KS]
    fieldnames = ["arm", "stt", "category", "n_gold"]
    for kk in field_ks:
        fieldnames += [
            f"n_retrieved@{kk}",
            f"recall@{kk}",
            f"precision@{kk}",
            f"f1@{kk}",
        ]
    fieldnames += ["r_precision", "mrr", "ndcg@10", "ndcg@all", "elapsed_s"]

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for arm, rows in per_arm_rows.items():
            for r in rows:
                row_out = {"arm": arm, "stt": r["stt"], "category": r["category"], "n_gold": r["n_gold"]}
                for kk in field_ks:
                    row_out[f"n_retrieved@{kk}"] = r[f"n_retrieved@{kk}"]
                    row_out[f"recall@{kk}"] = r[f"recall@{kk}"]
                    row_out[f"precision@{kk}"] = r[f"precision@{kk}"]
                    row_out[f"f1@{kk}"] = r[f"f1@{kk}"]
                row_out["r_precision"] = r["r_precision"]
                row_out["mrr"] = r["mrr"]
                row_out["ndcg@10"] = r["ndcg@10"]
                row_out["ndcg@all"] = r["ndcg@all"]
                row_out["elapsed_s"] = r["elapsed_s"]
                writer.writerow(row_out)
    return out


def write_json(per_arm_summary: dict, per_arm_strat: dict) -> Path:
    out = METRICS_DIR / "academic_metrics.json"
    payload = {
        "experiment": "06_retrieval_dense_vs_full",
        "arms": list(ARMS),
        "Ks": [("all" if k is None else k) for k in KS],
        "overall_macro": per_arm_summary,
        "stratified": per_arm_strat,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_report(per_arm_summary: dict, per_arm_strat: dict) -> Path:
    out = REPORT_DIR / "academic_report.md"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    field_ks = [("all" if k is None else str(k)) for k in KS]

    lines: list[str] = []
    lines.append("# Experiment 06 — Retrieval-only A/B (dense vs full_rerank)")
    lines.append("")
    lines.append("Dataset: 200 BHXH questions. Metric granularity: article.")
    lines.append("")
    lines.append("## Overall macro (all 200 questions with non-empty gold)")
    lines.append("")

    # Recall table
    def _table_metric(name: str) -> list[str]:
        rows = []
        rows.append("| arm | n | " + " | ".join(f"@{kk}" for kk in field_ks) + " |")
        rows.append("|---|---:|" + "|".join(["---:"] * len(field_ks)) + "|")
        for arm in ARMS:
            s = per_arm_summary[arm]
            vals = " | ".join(_fmt(s[f"{name}@{kk}"]) for kk in field_ks)
            rows.append(f"| {arm} | {s['n']} | {vals} |")
        return rows

    lines.append("### Citation recall")
    lines += _table_metric("recall")
    lines.append("")
    lines.append("### Citation precision")
    lines += _table_metric("precision")
    lines.append("")
    lines.append("### Citation F1")
    lines += _table_metric("f1")
    lines.append("")
    lines.append("### Average retrieved-set size at K")
    lines += _table_metric("avg_n_retrieved")
    lines.append("")

    # Rank-aware metrics — not capped by |gold|/K asymmetry.
    lines.append("### Rank-aware metrics (recommended when |gold| << K)")
    lines.append("")
    lines.append("| arm | n | R-Precision | MRR | NDCG@10 | NDCG@all |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        s = per_arm_summary[arm]
        lines.append(
            f"| {arm} | {s['n']} | "
            f"{_fmt(s['r_precision'])} | "
            f"{_fmt(s['mrr'])} | "
            f"{_fmt(s['ndcg@10'])} | "
            f"{_fmt(s['ndcg@all'])} |"
        )
    lines.append("")
    lines.append("- **R-Precision** = precision at K=|gold| per question. "
                 "Since K=|gold| here, R-Precision = recall = F1 — a single fair number when |gold| is small.")
    lines.append("- **MRR** = mean reciprocal rank of the *first* gold article retrieved. Captures \"how fast does the right answer appear at the top?\"")
    lines.append("- **NDCG@10** = binary-relevance NDCG truncated at 10, normalised by ideal DCG (= 1 if all |gold| are in top-10 in order).")
    lines.append("- **NDCG@all** = NDCG over the full retrieved list (caps at retrieved size).")
    lines.append("")

    # Latency
    lines.append("### Latency")
    lines.append("")
    lines.append("| arm | avg elapsed (s) |")
    lines.append("|---|---:|")
    for arm in ARMS:
        lines.append(f"| {arm} | {_fmt(per_arm_summary[arm]['avg_elapsed_s'])} |")
    lines.append("")

    # Stratified
    lines.append("## Stratified by gold corpus type")
    lines.append("")
    for cat in ("in_corpus", "mixed", "ooc", "unparseable"):
        lines.append(f"### {cat}")
        lines.append("")
        n_present = max(per_arm_strat[arm][cat].get("n", 0) for arm in ARMS)
        if not n_present:
            lines.append("_(no questions in this stratum)_")
            lines.append("")
            continue
        # K-based recall/precision/F1 table
        lines.append("| arm | n | " + " | ".join(f"R@{kk}" for kk in field_ks) +
                     " | " + " | ".join(f"P@{kk}" for kk in field_ks) +
                     " | " + " | ".join(f"F1@{kk}" for kk in field_ks) + " |")
        ncols = 1 + 3 * len(field_ks)
        lines.append("|---|---:|" + "|".join(["---:"] * ncols) + "|")
        for arm in ARMS:
            s = per_arm_strat[arm][cat]
            if not s.get("n"):
                continue
            r_vals = " | ".join(_fmt(s[f"recall@{kk}"]) for kk in field_ks)
            p_vals = " | ".join(_fmt(s[f"precision@{kk}"]) for kk in field_ks)
            f_vals = " | ".join(_fmt(s[f"f1@{kk}"]) for kk in field_ks)
            lines.append(f"| {arm} | {s['n']} | {r_vals} | {p_vals} | {f_vals} |")
        lines.append("")
        # Rank-aware subtable for this stratum
        lines.append(f"_Rank-aware ({cat})_")
        lines.append("")
        lines.append("| arm | n | R-Precision | MRR | NDCG@10 | NDCG@all |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for arm in ARMS:
            s = per_arm_strat[arm][cat]
            if not s.get("n"):
                continue
            lines.append(
                f"| {arm} | {s['n']} | "
                f"{_fmt(s.get('r_precision'))} | "
                f"{_fmt(s.get('mrr'))} | "
                f"{_fmt(s.get('ndcg@10'))} | "
                f"{_fmt(s.get('ndcg@all'))} |"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Recall denominator = |gold|; questions with empty gold are skipped.")
    lines.append("- Precision denominator = |retrieved@K|; if a question's retrieved set is empty, precision = 0.")
    lines.append("- F1 = harmonic mean of per-question P/R; macro across questions.")
    lines.append("- Arm `dense` retrieves up to `dense_k=50` BGE-M3 LoRA hits (article-deduped).")
    lines.append("- Arm `full_rerank` retrieves up to 12 final articles (rerank2_top_k=12).")
    lines.append("  Reported recall@K for K > 12 caps at the natural pool size.")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Validating gold citations ...")
    gold_map = load_gold()
    print(f"      OK — {len(gold_map)} questions normalized")

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    q_by_stt = {q["stt"]: q for q in questions}
    in_corpus_codes = {m.full_id for m in load_law_metadata().values()}

    print("[2/4] Loading per-arm records + scoring ...")
    per_arm_rows: dict[str, list[dict]] = {}
    per_arm_summary: dict[str, dict] = {}
    per_arm_strat: dict[str, dict] = {}
    for arm in ARMS:
        recs = load_arm_records(arm)
        if not recs:
            print(f"      WARN: arm {arm!r} has 0 records — skipping", file=sys.stderr)
            per_arm_rows[arm] = []
            per_arm_summary[arm] = aggregate_macro([])
            per_arm_strat[arm] = aggregate_stratified([])
            continue
        rows = score_arm(arm, recs, gold_map, q_by_stt, in_corpus_codes)
        per_arm_rows[arm] = rows
        per_arm_summary[arm] = aggregate_macro(rows)
        per_arm_strat[arm] = aggregate_stratified(rows)
        print(f"      {arm:<12} scored {len(rows)} questions")

    print("[3/4] Writing artifacts ...")
    csv_path = write_csv(per_arm_rows)
    json_path = write_json(per_arm_summary, per_arm_strat)
    print(f"      {csv_path}")
    print(f"      {json_path}")

    print("[4/4] Writing report ...")
    report_path = write_report(per_arm_summary, per_arm_strat)
    print(f"      {report_path}")

    # Console summary
    print()
    print("=== Overall macro ===")
    print(f"  {'arm':<14}{'n':>5}{'R@5':>8}{'R@10':>8}{'R@12':>8}{'P@12':>8}{'F1@12':>8}")
    for arm in ARMS:
        s = per_arm_summary[arm]
        print(
            f"  {arm:<14}{s['n']:>5}"
            f"{_fmt(s['recall@5']):>8}{_fmt(s['recall@10']):>8}{_fmt(s['recall@12']):>8}"
            f"{_fmt(s['precision@12']):>8}{_fmt(s['f1@12']):>8}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
