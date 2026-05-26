"""Sinh report markdown + CSV cho multimodel experiment.

Output:
    reports/multimodel_report.md  — comparison table model × arm,
                                    per-model Prolog reliability,
                                    per-model pairwise (graphrag vs no_retrieval)
    data/eval/multimodel/metrics.csv — wide CSV (1 row / câu / combo)
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

from experiments.generate_report import (
    METRIC_SPECS,
    PROLOG_SPECS,
    DIRECTION_BETTER,
    _safe_mean,
    _safe_std,
    _fmt,
)


def _value_from_chain(rec: dict, chain: list[str]):
    """Extract nested value from 1 record. Returns None nếu missing/wrong type."""
    v = rec
    for k in chain:
        if not isinstance(v, dict):
            return None
        v = v.get(k)
    return v


def _format_value(v, metric_name: str = "") -> str:
    """Format số với precision tuỳ metric."""
    if v is None:
        return "N/A"
    if metric_name == "cost_usd":
        return f"{v:.6f}"
    if metric_name == "latency_s":
        return f"{v:.2f}"
    return f"{v:.4f}"

METRICS_PATH = Path("data/eval/multimodel/metrics.json")
REPORT_OUT = Path("reports/multimodel_report.md")
CSV_OUT = Path("data/eval/multimodel/metrics.csv")

# Arm display order
ARM_ORDER = ["elite_no_retrieval", "elite_graphrag", "elite_ontology"]
# Model display order (cost/capability ascending)
MODEL_ORDER = ["gpt-4o-mini", "gpt-4.1", "gpt-4o", "gpt-5-mini", "gpt-5"]


def _bool_rate(records: list[dict], chain: list[str]) -> float | None:
    vals = []
    for r in records:
        v = _value_from_chain(r, chain)
        if v is not None:
            vals.append(1.0 if v else 0.0)
    return sum(vals) / len(vals) if vals else None


def _scalar_mean(records: list[dict], chain: list[str]) -> float | None:
    return _safe_mean(_value_from_chain(r, chain) for r in records)


def _scalar_std(records: list[dict], chain: list[str]) -> float | None:
    return _safe_std(_value_from_chain(r, chain) for r in records)


def _group_by_arm_model(metrics: dict) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    for combo, recs in metrics.items():
        if not recs:
            continue
        arm = recs[0].get("arm", "")
        model = recs[0].get("model", "")
        out[(arm, model)] = recs
    return out


def main() -> int:
    if not METRICS_PATH.exists():
        print(f"FAIL: {METRICS_PATH} not found — chạy compute_multimodel_metrics trước.")
        return 1

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    by_am = _group_by_arm_model(metrics)
    if not by_am:
        print("FAIL: metrics.json không có record nào")
        return 1

    arms = sorted({a for a, _ in by_am}, key=lambda a: ARM_ORDER.index(a)
                  if a in ARM_ORDER else len(ARM_ORDER))
    models = sorted({m for _, m in by_am}, key=lambda m: MODEL_ORDER.index(m)
                    if m in MODEL_ORDER else len(MODEL_ORDER))

    n_samples = {(a, m): len(recs) for (a, m), recs in by_am.items()}

    # ----------------------------------------------------------------------
    # MD report
    # ----------------------------------------------------------------------
    lines: list[str] = []
    lines.append("# Multi-model Experiment Report — elite_no_retrieval vs elite_graphrag")
    lines.append("")
    lines.append("**Dataset**: 200 câu BHXH (FB group). Each combo runs on the same questions.")
    lines.append("")
    lines.append(f"**Models compared**: {', '.join(models)}")
    lines.append(f"**Arms compared**: {', '.join(arms)}")
    lines.append(f"**Total combos**: {len(by_am)}")
    lines.append("")
    lines.append("**Sample sizes**:")
    lines.append("")
    lines.append("| Arm \\ Model | " + " | ".join(models) + " |")
    lines.append("|" + "|".join(["---"] * (len(models) + 1)) + "|")
    for arm in arms:
        row = [arm]
        for model in models:
            row.append(str(n_samples.get((arm, model), 0)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("**Inference models**: as per columns. **Judge model**: gpt-4o-mini.")
    lines.append("")

    # -------------------------------------------------------------
    # Per-model × per-arm aggregate (each combo = mean ± std)
    # -------------------------------------------------------------
    lines.append("## Aggregate results (mean values per combo)")
    lines.append("")

    for metric_name, chain in METRIC_SPECS:
        direction = DIRECTION_BETTER.get(metric_name, "")
        lines.append(f"### `{metric_name}` ({direction} better)")
        lines.append("")
        lines.append("| Arm | " + " | ".join(models) + " |")
        lines.append("|" + "|".join(["---"] * (len(models) + 1)) + "|")
        for arm in arms:
            row = [arm]
            for model in models:
                recs = by_am.get((arm, model), [])
                if not recs:
                    row.append("—")
                    continue
                m = _scalar_mean(recs, chain)
                s = _scalar_std(recs, chain)
                if m is None:
                    row.append("N/A")
                else:
                    row.append(f"{_format_value(m, metric_name)} ± {_format_value(s, metric_name) if s else '0'}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # -------------------------------------------------------------
    # Prolog reliability (Logic-LM 4 metrics) — all combos
    # -------------------------------------------------------------
    lines.append("## Prolog reliability (Logic-LM metrics, Pan et al. EMNLP'23)")
    lines.append("")
    lines.append("> Đo độ tin cậy của symbolic solver loop (per-combo, áp dụng cho tất cả elite arms).")
    lines.append("")

    for metric_name, chain, kind in PROLOG_SPECS:
        direction = DIRECTION_BETTER.get(metric_name, "")
        lines.append(f"### `{metric_name}` ({direction} better)")
        lines.append("")
        lines.append("| Arm | " + " | ".join(models) + " |")
        lines.append("|" + "|".join(["---"] * (len(models) + 1)) + "|")
        for arm in arms:
            row = [arm]
            for model in models:
                recs = by_am.get((arm, model), [])
                if not recs:
                    row.append("—")
                    continue
                val = (_bool_rate(recs, chain) if kind == "bool"
                       else _scalar_mean(recs, chain))
                row.append(f"{val:.4f}" if val is not None else "N/A")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # -------------------------------------------------------------
    # Prolog status distribution per combo
    # -------------------------------------------------------------
    lines.append("### Prolog status distribution")
    lines.append("")
    all_statuses: set[str] = set()
    status_by_combo: dict[tuple[str, str], Counter] = {}
    for (arm, model), recs in by_am.items():
        c = Counter()
        for r in recs:
            v = _value_from_chain(r, ["prolog_rollback", "prolog_status"])
            if v:
                c[v] += 1
                all_statuses.add(v)
        status_by_combo[(arm, model)] = c
    statuses = sorted(all_statuses)

    lines.append("| Status | " + " | ".join(
        [f"{a} × {m}" for a in arms for m in models if (a, m) in by_am]) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * sum(
        1 for a in arms for m in models if (a, m) in by_am)) + "|")
    for st in statuses:
        row = [f"`{st}`"]
        for arm in arms:
            for model in models:
                if (arm, model) not in by_am:
                    continue
                row.append(str(status_by_combo[(arm, model)].get(st, 0)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # -------------------------------------------------------------
    # Pairwise per model: elite_graphrag vs elite_no_retrieval
    # -------------------------------------------------------------
    lines.append("## Pairwise judge: elite_graphrag vs elite_no_retrieval (per model)")
    lines.append("")
    lines.append("> Within mỗi model, judge so sánh 2 arm trên cùng câu hỏi (position swap).")
    lines.append("")

    for model in models:
        gr_recs = by_am.get(("elite_graphrag", model), [])
        pws = [r["pairwise_vs_no_retrieval"] for r in gr_recs
               if "pairwise_vs_no_retrieval" in r]
        if not pws:
            continue
        lines.append(f"### `{model}` (n={len(pws)})")
        lines.append("")
        consensus = Counter(p["consensus"] for p in pws)
        lines.append("| Consensus | Count | % |")
        lines.append("|---|---:|---:|")
        for label, cnt in consensus.most_common():
            lines.append(f"| {label} | {cnt} | {cnt / len(pws) * 100:.1f}% |")
        lines.append("")

        # Position swap detail
        votes_ab = Counter(p.get("vote_ab") for p in pws)
        votes_ba = Counter(p.get("vote_ba") for p in pws)
        all_votes = sorted(set(list(votes_ab) + list(votes_ba)))
        lines.append("**Position swap (A=elite_no_retrieval, B=elite_graphrag):**")
        lines.append("")
        lines.append("| Vote | A=no_retr B=graphrag | A=graphrag B=no_retr |")
        lines.append("|---|---:|---:|")
        for v in all_votes:
            lines.append(f"| {v} | {votes_ab.get(v, 0)} | {votes_ba.get(v, 0)} |")
        lines.append("")

    # -------------------------------------------------------------
    # Discussion (auto-generated)
    # -------------------------------------------------------------
    lines.append("## Discussion (auto-generated)")
    lines.append("")
    lines.append("### Best (arm, model) per metric")
    lines.append("")
    lines.append("| Metric | Best combo | Value |")
    lines.append("|---|---|---|")
    all_metric_specs = METRIC_SPECS + [(name, chain) for name, chain, _ in PROLOG_SPECS]
    for spec in all_metric_specs:
        if len(spec) == 3:
            metric_name, chain, kind = spec
            agg = _bool_rate if kind == "bool" else _scalar_mean
        else:
            metric_name, chain = spec
            agg = _scalar_mean
        direction = DIRECTION_BETTER.get(metric_name, "higher")
        best_combo = None
        best_val = None
        for (arm, model), recs in by_am.items():
            v = agg(recs, chain)
            if v is None:
                continue
            if best_val is None or (
                (direction == "higher" and v > best_val) or
                (direction == "lower" and v < best_val)
            ):
                best_val = v
                best_combo = (arm, model)
        if best_combo:
            arm, model = best_combo
            lines.append(f"| {metric_name} | **{arm} × {model}** | "
                         f"{_format_value(best_val, metric_name)} |")
    lines.append("")

    # Per-model arm winner (no-retrieval vs graphrag, for key metrics)
    lines.append("### Within each model: does retrieval (graphrag) help?")
    lines.append("")
    lines.append("| Model | Faithfulness Δ | Halluc. Δ | Prolog success Δ | Cost Δ | Latency Δ |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model in models:
        nr = by_am.get(("elite_no_retrieval", model), [])
        gr = by_am.get(("elite_graphrag", model), [])
        if not nr or not gr:
            continue
        def _delta(chain, agg=_scalar_mean):
            a = agg(gr, chain)
            b = agg(nr, chain)
            if a is None or b is None:
                return None
            return a - b
        d_faith = _delta(["faithfulness", "faithfulness"])
        d_halu = _delta(["hallucination", "hallucination_rate"])
        d_prolog = _delta(["prolog_rollback", "prolog_success"], _bool_rate)
        d_cost = _delta(["cost", "cost_usd"])
        d_lat = _delta(["latency", "latency_s"])
        def _fmt(v, decimals=4):
            if v is None:
                return "N/A"
            sign = "+" if v >= 0 else ""
            return f"{sign}{v:.{decimals}f}"
        lines.append(f"| {model} | {_fmt(d_faith)} | {_fmt(d_halu)} | "
                     f"{_fmt(d_prolog)} | {_fmt(d_cost, 6)} | {_fmt(d_lat, 2)}s |")
    lines.append("")
    lines.append("(Δ = elite_graphrag − elite_no_retrieval. Positive = retrieval increases.)")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append("1. **Self-enhancement bias** (Zheng 2023): Judge=gpt-4o-mini. Khi compare with itself "
                 "(gpt-4o variants), absolute scores có thể inflated. Cross-model relative compare vẫn fair.")
    lines.append("2. **Reasoning models** (gpt-5*, o-series): có thể auto-fallback temperature→1.0 nếu API "
                 "reject, document trong `experiments/elite_pipelines.py:_chat_with_fallback`.")
    lines.append("3. **Latency** không direct compare: gpt-5* dùng reasoning tokens (vô hình từ user) →"
                 " latency cao + completion_tokens cao là expected, không phải bug.")
    lines.append("4. **Citation extraction** từ IRAC text: dùng bracketed + inline regex; có thể miss "
                 "vài natural-language reference.")
    lines.append("5. **Pairwise** dùng position swap để giảm bias; chỉ tin strong-consensus rows.")
    lines.append("6. **SWI-Prolog timeout=15s**: câu phức tạp có thể bị giết silently → "
                 "`prolog_success=False`, `prolog_status='unable_to_conclude'`.")
    lines.append("")

    # ----------------------------------------------------------------------
    # Write outputs
    # ----------------------------------------------------------------------
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report: {REPORT_OUT}")

    # CSV (1 row / câu / combo)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "arm", "model", "combo", "stt",
        "citation_validity", "citation_recall", "citation_precision",
        "faithfulness", "answer_relevance", "hallucination_rate",
        "bertscore_f1", "cost_usd", "latency_s",
        "prolog_success", "n_repair_rounds", "prolog_status",
    ]
    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for combo, recs in metrics.items():
            for r in recs:
                row = {
                    "arm": r.get("arm"),
                    "model": r.get("model"),
                    "combo": r.get("combo"),
                    "stt": r.get("stt"),
                }
                row["citation_validity"] = _value_from_chain(r, ["citation_validity", "validity_rate"])
                row["citation_recall"] = _value_from_chain(r, ["citation_recall", "recall"])
                row["citation_precision"] = _value_from_chain(r, ["citation_precision", "precision"])
                row["faithfulness"] = _value_from_chain(r, ["faithfulness", "faithfulness"])
                row["answer_relevance"] = _value_from_chain(r, ["answer_relevance", "answer_relevance"])
                row["hallucination_rate"] = _value_from_chain(r, ["hallucination", "hallucination_rate"])
                row["bertscore_f1"] = _value_from_chain(r, ["bertscore", "bertscore_f1"])
                row["cost_usd"] = _value_from_chain(r, ["cost", "cost_usd"])
                row["latency_s"] = _value_from_chain(r, ["latency", "latency_s"])
                row["prolog_success"] = _value_from_chain(r, ["prolog_rollback", "prolog_success"])
                row["n_repair_rounds"] = _value_from_chain(r, ["prolog_rollback", "n_repair_rounds"])
                row["prolog_status"] = _value_from_chain(r, ["prolog_rollback", "prolog_status"])
                w.writerow(row)
    print(f"Saved CSV   : {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
