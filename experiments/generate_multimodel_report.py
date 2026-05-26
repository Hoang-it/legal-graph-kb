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
    ELITE_BIASED_METRICS,
    _safe_mean,
    _safe_std,
    _fmt,
)
# R2 elite arms: cả 2 đều là elite (NR + GR)
R2_ELITE_BASE = {"elite_no_retrieval", "elite_graphrag"}


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
    # Per-model × per-arm aggregate (each combo = mean ± std + n_valid)
    # -------------------------------------------------------------
    MIN_N_VALID = 30
    lines.append("## Aggregate results (macro mean ± std, n_valid/total)")
    lines.append("")
    lines.append(f"> Cells với `n_valid < {MIN_N_VALID}` → 'insufficient'. "
                 f"Khi n_valid << n_total: metric chỉ đo trên subset (selection bias).")

    # API error counts per combo
    api_err_per_combo = {(a, m): sum(1 for r in recs if r.get("api_error"))
                          for (a, m), recs in by_am.items()}
    if any(v > 0 for v in api_err_per_combo.values()):
        affected = [f"{a}×{m}={n}" for (a, m), n in api_err_per_combo.items() if n > 0]
        lines.append(f"> **API errors** (excluded from aggregates): {', '.join(affected)}.")
    lines.append("> **BERTScore + answer_relevance dropped cho cả 2 arms (NR + GR)** vì cả 2 đều output IRAC structured text — không có baseline prose để fair compare. (Metrics này giữ trong R1 nơi có graphrag/llm_only baseline.)")
    lines.append("")

    def _clean_recs(arm, model):
        return [r for r in by_am.get((arm, model), []) if not r.get("api_error")]

    def _n_valid(recs, chain):
        return sum(1 for r in recs if _value_from_chain(r, chain) is not None)

    for metric_name, chain in METRIC_SPECS:
        direction = DIRECTION_BETTER.get(metric_name, "")
        lines.append(f"### `{metric_name}` ({direction} better)")
        lines.append("")
        lines.append("| Arm | " + " | ".join(models) + " |")
        lines.append("|" + "|".join(["---"] * (len(models) + 1)) + "|")
        for arm in arms:
            row = [arm]
            for model in models:
                # Drop biased metrics for elite (cả NR và GR đều là elite IRAC).
                # Override: if first metric record có _used_plain_answer=True →
                # compute_metrics đã dùng plain_answer → fair, fall through to render.
                if metric_name in ELITE_BIASED_METRICS and arm in R2_ELITE_BASE:
                    recs0 = by_am.get((arm, model), [])
                    used_plain = False
                    if recs0:
                        used_plain = bool(_value_from_chain(recs0[0], chain[:-1] + ["_used_plain_answer"]))
                    if not used_plain:
                        row.append("dropped (IRAC bias)")
                        continue
                recs = _clean_recs(arm, model)
                if not recs:
                    row.append("—")
                    continue
                n_t = len(recs)
                n_v = _n_valid(recs, chain)
                if n_v < MIN_N_VALID:
                    row.append(f"insufficient (n={n_v}/{n_t})")
                    continue
                m = _scalar_mean(recs, chain)
                s = _scalar_std(recs, chain)
                if m is None:
                    row.append(f"N/A (n=0/{n_t})")
                else:
                    base = f"{_format_value(m, metric_name)} ± {_format_value(s, metric_name) if s else '0'}"
                    row.append(f"{base} (n={n_v}/{n_t})")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # -------------------------------------------------------------
    # Micro-average citation metrics
    # -------------------------------------------------------------
    lines.append("### Citation metrics — micro-average (corpus-level Σ correct / Σ extracted)")
    lines.append("")
    lines.append("> Less sensitive to selection bias than macro mean above.")
    lines.append("")
    micro_specs = [
        ("citation_validity", ["citation_validity", "n_valid"], ["citation_validity", "n_citations"]),
        ("citation_recall", ["citation_recall", "n_with_cite"], ["citation_recall", "n_sentences"]),
        ("citation_precision", ["citation_precision", "n_supported"], ["citation_precision", "n_citations"]),
        ("faithfulness", ["faithfulness", "n_supported"], ["faithfulness", "n_claims"]),
    ]
    for metric_name, num_chain, denom_chain in micro_specs:
        lines.append(f"#### `{metric_name}` micro")
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
                num = sum(int(_value_from_chain(r, num_chain) or 0) for r in recs)
                denom = sum(int(_value_from_chain(r, denom_chain) or 0) for r in recs)
                if denom == 0:
                    row.append("N/A")
                else:
                    row.append(f"{num/denom:.4f} (Σ={num}/{denom})")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # -------------------------------------------------------------
    # Prolog reliability (Logic-LM 4 metrics) — all combos
    # -------------------------------------------------------------
    lines.append("## Prolog reliability (Logic-LM metrics, Pan et al. EMNLP'23)")
    lines.append("")
    lines.append("> Đo độ tin cậy của symbolic solver loop (per-combo, áp dụng cho tất cả elite arms).")
    lines.append("")

    lines.append("> **API-error records excluded** từ mọi tỉ lệ dưới (tránh ô nhiễm).")
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
                recs = _clean_recs(arm, model)
                if not recs:
                    row.append("—")
                    continue
                val = (_bool_rate(recs, chain) if kind == "bool"
                       else _scalar_mean(recs, chain))
                if val is None:
                    row.append("N/A")
                else:
                    row.append(f"{val:.4f} (n={len(recs)})")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    # API error rate row
    if any(v > 0 for v in api_err_per_combo.values()):
        lines.append("### API error rate (records excluded from above)")
        lines.append("")
        lines.append("| Arm | " + " | ".join(models) + " |")
        lines.append("|" + "|".join(["---"] * (len(models) + 1)) + "|")
        for arm in arms:
            row = [arm]
            for model in models:
                n_err = api_err_per_combo.get((arm, model), 0)
                row.append(f"{n_err}/200 = {n_err/200*100:.1f}%")
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
        # AUDIT FIX: filter API-error records — pairwise data trên failure
        # messages (e.g. "[Pipeline không trả về kết luận]") không có ý nghĩa.
        gr_clean = [r for r in gr_recs if not r.get("api_error")]
        pws = [r["pairwise_vs_no_retrieval"] for r in gr_clean
               if "pairwise_vs_no_retrieval" in r]
        if not pws:
            continue
        n_total = len(pws)
        n_excluded_api = len(gr_recs) - len(gr_clean)
        if n_excluded_api > 0:
            lines.append(f"> _{n_excluded_api} GR records excluded vì API errors (pairwise vô nghĩa)._\n")
        consensus = Counter(p["consensus"] for p in pws)
        n_split = consensus.get("split", 0)
        n_consistent = n_total - n_split
        gr_key = f"elite_graphrag__{model.replace('.', '_')}"
        nr_key = f"elite_no_retrieval__{model.replace('.', '_')}"
        n_gr = consensus.get(gr_key, 0)
        n_nr = consensus.get(nr_key, 0)
        n_tie = consensus.get("tie", 0)
        lines.append(f"### `{model}` (n={n_total})")
        lines.append("")
        lines.append("**Consensus breakdown (full):**")
        lines.append("")
        lines.append("| Consensus | Count | % of n=200 |")
        lines.append("|---|---:|---:|")
        for label, cnt in consensus.most_common():
            lines.append(f"| {label} | {cnt} | {cnt / n_total * 100:.1f}% |")
        lines.append("")
        # AUDIT FIX: consistent-verdict subset reporting
        lines.append(f"**On consistent-verdict subset** (n_consistent = "
                     f"{n_consistent} = {n_total} − {n_split} split):")
        lines.append("")
        if n_consistent > 0:
            lines.append(f"- **elite_graphrag**: {n_gr}/{n_consistent} = "
                         f"{n_gr/n_consistent*100:.1f}% wins")
            lines.append(f"- **elite_no_retrieval**: {n_nr}/{n_consistent} = "
                         f"{n_nr/n_consistent*100:.1f}% wins")
            if n_tie:
                lines.append(f"- ties: {n_tie}/{n_consistent} = "
                             f"{n_tie/n_consistent*100:.1f}%")
        else:
            lines.append("- No consistent verdicts")
        lines.append("")
        # Position swap detail
        votes_ab = Counter(p.get("vote_ab") for p in pws)
        votes_ba = Counter(p.get("vote_ba") for p in pws)
        all_votes = sorted(set(list(votes_ab) + list(votes_ba)))
        lines.append("**Position-swap detail (raw votes per direction):**")
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
    lines.append("> Combos với `n_valid < 30` bị loại. API-error records excluded. "
                 "Biased metrics (BERTScore, answer_relevance) skipped vì R2 cả 2 arms đều là IRAC.")
    lines.append("")
    lines.append("| Metric | Best combo | Value | n_valid |")
    lines.append("|---|---|---|---:|")
    all_metric_specs = METRIC_SPECS + [(name, chain) for name, chain, _ in PROLOG_SPECS]
    MIN_N_VALID = 30
    for spec in all_metric_specs:
        if len(spec) == 3:
            metric_name, chain, kind = spec
            agg = _bool_rate if kind == "bool" else _scalar_mean
        else:
            metric_name, chain = spec
            agg = _scalar_mean
        # Skip biased metrics for elite (R2 arms đều là elite)
        if metric_name in ELITE_BIASED_METRICS:
            lines.append(f"| {metric_name} | _dropped (IRAC bias)_ | — | — |")
            continue
        direction = DIRECTION_BETTER.get(metric_name, "higher")
        best_combo = None
        best_val = None
        best_nvalid = 0
        for (arm, model), recs in by_am.items():
            # AUDIT FIX: clean records (exclude API errors)
            clean = [r for r in recs if not r.get("api_error")]
            v = agg(clean, chain)
            n_v = sum(1 for r in clean if _value_from_chain(r, chain) is not None)
            if v is None or n_v < MIN_N_VALID:
                continue
            if best_val is None or (
                (direction == "higher" and v > best_val) or
                (direction == "lower" and v < best_val)
            ):
                best_val = v
                best_combo = (arm, model)
                best_nvalid = n_v
        if best_combo:
            arm, model = best_combo
            lines.append(f"| {metric_name} | **{arm} × {model}** | "
                         f"{_format_value(best_val, metric_name)} | {best_nvalid} |")
        else:
            lines.append(f"| {metric_name} | — (all insufficient) | — | — |")
    lines.append("")

    # Pairwise winner per model (USING FIXED PAIRWISE DATA)
    lines.append("### Pairwise winner per model (elite_graphrag vs elite_no_retrieval)")
    lines.append("")
    lines.append("> Strong-consensus per model. Numbers từ corrected `_vote` (2026-05-26 fix).")
    lines.append("")
    lines.append("| Model | NR wins | GR wins | Split | Tie | Verdict |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for model in models:
        gr_recs = by_am.get(("elite_graphrag", model), [])
        if not gr_recs:
            continue
        # AUDIT FIX: filter API errors
        gr_clean = [r for r in gr_recs if not r.get("api_error")]
        pws = [r["pairwise_vs_no_retrieval"] for r in gr_clean
               if "pairwise_vs_no_retrieval" in r]
        if not pws:
            continue
        n_tot = len(pws)
        n_excluded = len(gr_recs) - len(gr_clean)
        cons = Counter(p["consensus"] for p in pws)
        nr_key = f"elite_no_retrieval__{model.replace('.', '_')}"
        gr_key = f"elite_graphrag__{model.replace('.', '_')}"
        w_nr = cons.get(nr_key, 0)
        w_gr = cons.get(gr_key, 0)
        n_split = cons.get("split", 0)
        n_tie = cons.get("tie", 0)
        n_consistent = n_tot - n_split
        # Verdict báo cáo trên consistent-verdict subset
        if n_consistent > 0:
            gr_pct_consistent = w_gr / n_consistent * 100
            nr_pct_consistent = w_nr / n_consistent * 100
            if w_gr > w_nr * 1.5:
                verdict = f"**GR beats NR** ({gr_pct_consistent:.1f}% of {n_consistent} consistent)"
            elif w_nr > w_gr * 1.5:
                verdict = f"**NR beats GR** ({nr_pct_consistent:.1f}% of {n_consistent} consistent)"
            else:
                verdict = f"Tied (GR={gr_pct_consistent:.1f}%, NR={nr_pct_consistent:.1f}% of {n_consistent})"
        else:
            verdict = "No consistent verdicts"
        excl_note = f" (excluded {n_excluded} API err)" if n_excluded else ""
        lines.append(f"| {model}{excl_note} | {w_nr} ({w_nr/n_tot*100:.1f}%) | "
                     f"{w_gr} ({w_gr/n_tot*100:.1f}%) | "
                     f"{n_split} ({n_split/n_tot*100:.1f}%) | {n_tie} | {verdict} |")
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
    lines.append("4. **Citation source asymmetry** (audit 2026-05-26): "
                 "`citation_validity` + `faithfulness` + `hallucination` dùng `record['citation_ids']` "
                 "(bao gồm fallback parser của Prolog `legal_source(...)` facts). "
                 "`citation_recall` + `citation_precision` chỉ dùng `parse_citations(answer_text)`. "
                 "Khi IRAC text không có brackets, gap có thể xuất hiện. Xem `reports/report_v2.md` "
                 "cho `citation_text_coverage` per combo.")
    lines.append("5. **Selection bias**: cells `n_valid < 30` → 'insufficient'. "
                 "Cells `n_valid << n_total` (đặc biệt `elite_no_retrieval × gpt-5-mini` với n=23) "
                 "không comparable trực tiếp với cells có n_valid cao. Macro + micro tables show riêng.")
    lines.append("6. **gpt-5-mini abstention** (audit 2026-05-26): "
                 "108/198 success records có `based_on(source_X)` in Prolog trace nhưng `citation_ids=[]` vì "
                 "gpt-5-mini declare `legal_source(...,article: none,...)` (honest abstention without retrieval). "
                 "Fallback parser regex `article_(\\d+)` không match → citation_ids empty. "
                 "Không phải bug pipeline — đây là behavioral difference của reasoning models. "
                 "gpt-4.1/gpt-4o fabricate article numbers cụ thể (validity ~95% confirms most fabricated numbers happen valid).")
    lines.append("7. **Pairwise** (FIXED 2026-05-26): `_vote(w, a_first=False)` trước đây invert vote_ba → consensus inflated as 'split'. "
                 "Đã fix — bảng dưới phản ánh đúng. Position bias trong reality moderate, không 'mạnh' như báo cáo cũ.")
    lines.append("8. **SWI-Prolog timeout=15s**: câu phức tạp có thể bị giết silently → "
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
