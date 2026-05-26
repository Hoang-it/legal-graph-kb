"""audit_reaggregate.py — Post-process metrics to fix selection bias + show n_valid.

Reads existing aggregated `metrics.json` files + raw per-sample logs.
**Does NOT run any inference or judge calls** — chỉ re-compute aggregates.

Tackles 3 issues found in audit:
1. **Selection bias**: macro mean filters None → cells with low n_valid look
   comparable but aren't. → Show n_valid/n_total in every cell.
2. **Macro hides skew**: per-record mean of per-record rates ≠ corpus-level rate.
   → Add micro-average (Σ numerator / Σ denominator) for citation metrics.
3. **Input asymmetry citation_ids vs parse_citations(text)**: elite IRAC text
   doesn't match regex; fallback parser extracts from Prolog facts.
   → Add `citation_text_coverage` column = % records where the two sources agree.

Cells with `n_valid < 30` (MIN_N_VALID) → "insufficient (n=X/200)" (not enough
samples to trust). Threshold defensible by sample-size guidance for proportion
estimation (n>=30 for normal approx).

Output:
    reports/report_v2.md
    reports/comparison_validity_table.csv

Inputs auto-discovered:
    R1: data/eval/metrics.json + data/eval/results/{arm}/A*.json
    R2: data/eval/multimodel/metrics.json + data/eval/multimodel/results/{combo}/A*.json
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean, stdev

# ---------------------------------------------------------------------------
# Citation regex (mirror compute_metrics.parse_citations behavior)
# ---------------------------------------------------------------------------
_CIT_BRACKET = re.compile(
    r"\[Điều\s+(\d+)(?:\s+khoản\s+(\d+))?(?:\s+điểm\s+([a-zđ]))?\]"
)
_CIT_INLINE = re.compile(
    r"Điều\s+(\d+)(?:[,\s]+[Kk]ho[ảa]n\s+(\d+))?(?:[,\s]+[ĐđDd]i[ểe]m\s+([a-zđ]))?"
)

MIN_N_VALID = 30
ELITE_BASE_ARMS = {"elite_no_retrieval", "elite_ontology", "elite_graphrag"}

SOURCES = [
    {
        "name": "R1 (5-arm, gpt-4o-mini)",
        "metrics_json": Path("data/eval/metrics.json"),
        "raw_root": Path("data/eval/results"),
    },
    {
        "name": "R2 (multi-model)",
        "metrics_json": Path("data/eval/multimodel/metrics.json"),
        "raw_root": Path("data/eval/multimodel/results"),
    },
]

REPORT_OUT = Path("reports/report_v2.md")
CSV_OUT = Path("reports/comparison_validity_table.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text_citation_ids(text: str) -> set[str]:
    """Replicate parse_citations() ID extraction from compute_metrics."""
    if not text:
        return set()
    out: set[str] = set()
    spans: list[tuple[int, int]] = []
    for m in _CIT_BRACKET.finditer(text):
        art, cl, pt = m.group(1), m.group(2), m.group(3)
        cid = f"L41_2024.A{art}"
        if cl:
            cid += f".K{cl}"
            if pt:
                cid += f".{pt}"
        out.add(cid)
        spans.append((m.start(), m.end()))
    for m in _CIT_INLINE.finditer(text):
        if any(not (m.end() <= s or m.start() >= e) for s, e in spans):
            continue
        art, cl, pt = m.group(1), m.group(2), m.group(3)
        cid = f"L41_2024.A{art}"
        if cl:
            cid += f".K{cl}"
            if pt:
                cid += f".{pt}"
        out.add(cid)
    return out


def load_raw_records(raw_dir: Path) -> dict[int, dict]:
    """stt → raw record."""
    out: dict[int, dict] = {}
    if not raw_dir.exists():
        return out
    for fp in raw_dir.glob("A*.json"):
        if fp.name.endswith(".error.json"):
            continue
        try:
            r = json.loads(fp.read_text(encoding="utf-8"))
            out[r["stt"]] = r
        except Exception:
            pass
    return out


def safe_mean(vs):
    nn = [v for v in vs if v is not None]
    return mean(nn) if nn else None


def safe_std(vs):
    nn = [v for v in vs if v is not None]
    return stdev(nn) if len(nn) > 1 else None


def extract_chain(record: dict, chain: list[str]):
    v = record
    for k in chain:
        if not isinstance(v, dict):
            return None
        v = v.get(k)
    return v


def is_elite(arm: str) -> bool:
    if arm in ELITE_BASE_ARMS:
        return True
    return any(arm.startswith(base + "__") for base in ELITE_BASE_ARMS)


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def fmt_macro(mean_v, std_v, n_valid: int, n_total: int) -> str:
    if n_valid == 0:
        return f"N/A (n=0/{n_total})"
    if n_valid < MIN_N_VALID:
        return f"insufficient (n={n_valid}/{n_total})"
    if mean_v is None:
        return f"N/A (n={n_valid}/{n_total})"
    if std_v is not None:
        return f"{mean_v:.4f} ± {std_v:.4f} (n={n_valid}/{n_total})"
    return f"{mean_v:.4f} (n={n_valid}/{n_total})"


def fmt_micro(num: int, denom: int) -> str:
    if denom == 0:
        return f"N/A (Σ_denom=0)"
    rate = num / denom
    return f"{rate:.4f} (Σ={num}/{denom})"


def fmt_macro_no_n(mean_v, std_v, n_valid: int, n_total: int) -> str:
    """For metrics tính trên all records (latency, cost) — vẫn show n."""
    if n_valid < MIN_N_VALID:
        return f"insufficient (n={n_valid}/{n_total})"
    if mean_v is None:
        return f"N/A"
    if std_v is not None:
        return f"{mean_v:.4f} ± {std_v:.4f} (n={n_valid})"
    return f"{mean_v:.4f} (n={n_valid})"


# ---------------------------------------------------------------------------
# Metric spec
# ---------------------------------------------------------------------------
# (display_name, value_chain, micro_num_chain, micro_denom_chain)
# None for micro = report only macro
METRIC_SPEC = [
    ("citation_validity",
        ["citation_validity", "validity_rate"],
        ["citation_validity", "n_valid"],
        ["citation_validity", "n_citations"]),
    ("citation_recall",
        ["citation_recall", "recall"],
        ["citation_recall", "n_with_cite"],
        ["citation_recall", "n_sentences"]),
    ("citation_precision",
        ["citation_precision", "precision"],
        ["citation_precision", "n_supported"],
        ["citation_precision", "n_citations"]),
    ("faithfulness",
        ["faithfulness", "faithfulness"],
        ["faithfulness", "n_supported"],
        ["faithfulness", "n_claims"]),
    ("hallucination_rate",
        ["hallucination", "hallucination_rate"],
        None, None),  # special: aggregate_halu_micro
    ("answer_relevance",
        ["answer_relevance", "answer_relevance"],
        None, None),
    ("bertscore_f1",
        ["bertscore", "bertscore_f1"],
        None, None),
    ("cost_usd",
        ["cost", "cost_usd"],
        None, None),
    ("latency_s",
        ["latency", "latency_s"],
        None, None),
]


def aggregate_metric(records: list[dict], chain: list[str],
                     num_chain: list[str] | None,
                     denom_chain: list[str] | None) -> dict:
    """Returns dict with macro + micro stats."""
    vals = [extract_chain(r, chain) for r in records]
    n_total = len(records)
    n_valid = sum(1 for v in vals if v is not None)
    out = {
        "n_valid": n_valid,
        "n_total": n_total,
        "macro_mean": safe_mean(vals),
        "macro_std": safe_std(vals),
        "micro_num": None,
        "micro_denom": None,
        "micro_rate": None,
    }
    if num_chain and denom_chain:
        num_tot = sum(int(extract_chain(r, num_chain) or 0) for r in records)
        denom_tot = sum(int(extract_chain(r, denom_chain) or 0) for r in records)
        out["micro_num"] = num_tot
        out["micro_denom"] = denom_tot
        out["micro_rate"] = (num_tot / denom_tot) if denom_tot else None
    return out


def aggregate_halu_micro(records: list[dict]) -> tuple[int, int]:
    """Corpus-level hallucination: Σ (misstate+unsupported+invented) /
    Σ max(1, n_claims + n_invented).
    Mirrors per-record formula in compute_metrics:m_hallucination."""
    num_tot = denom_tot = 0
    for r in records:
        h = r.get("hallucination") or {}
        nm = int(h.get("n_misstate") or 0)
        nu = int(h.get("n_unsupported") or 0)
        ni = int(h.get("n_invented_citations") or 0)
        nc = int(h.get("n_claims") or 0)
        # Skip records where hallucination wasn't computable
        if h.get("hallucination_rate") is None:
            continue
        num_tot += nm + nu + ni
        denom_tot += max(1, nc + ni)
    return num_tot, denom_tot


def aggregate_prolog(records: list[dict]) -> dict:
    """Rates over records with prolog_success != None."""
    pr_valid = [r for r in records
                if (r.get("prolog_rollback") or {}).get("prolog_success") is not None]
    n_total = len(records)
    n_valid = len(pr_valid)
    if n_valid == 0:
        return {}
    n_success = sum(1 for r in pr_valid
                    if r["prolog_rollback"]["prolog_success"])
    n_first = sum(1 for r in pr_valid
                  if r["prolog_rollback"].get("first_try_success"))
    n_repair = sum(1 for r in pr_valid
                   if r["prolog_rollback"].get("repair_invoked"))
    rounds = [r["prolog_rollback"].get("n_repair_rounds") for r in pr_valid]
    rounds = [v for v in rounds if v is not None]
    return {
        "n_valid": n_valid,
        "n_total": n_total,
        "prolog_success_rate":   {"num": n_success, "denom": n_valid,
                                   "rate": n_success / n_valid},
        "first_try_success_rate": {"num": n_first, "denom": n_valid,
                                    "rate": n_first / n_valid},
        "repair_invoked_rate":   {"num": n_repair, "denom": n_valid,
                                   "rate": n_repair / n_valid},
        "avg_repair_rounds":     {"sum": sum(rounds),
                                   "n": len(rounds),
                                   "mean": sum(rounds) / len(rounds) if rounds else None},
    }


def compute_text_coverage(metric_records: list[dict],
                          raw_map: dict[int, dict]) -> dict:
    """Expose gap giữa citation_ids (pipeline output) và parse_citations(text)."""
    n_total = len(metric_records)
    n_ids_only = n_text_only = n_both = n_neither = n_overlap = 0
    for m in metric_records:
        raw = raw_map.get(m["stt"])
        if not raw:
            continue
        id_set = set(raw.get("citation_ids") or [])
        text_set = extract_text_citation_ids(raw.get("answer") or "")
        if id_set and text_set:
            n_both += 1
            if id_set & text_set:
                n_overlap += 1
        elif id_set:
            n_ids_only += 1
        elif text_set:
            n_text_only += 1
        else:
            n_neither += 1
    return {
        "n_total": n_total,
        "n_both": n_both,            # both sources nonempty
        "n_overlap": n_overlap,       # intersection non-empty
        "n_ids_only": n_ids_only,     # IDs from pipeline (fallback) but text has none
        "n_text_only": n_text_only,   # text has citations but pipeline IDs empty
        "n_neither": n_neither,
        "coverage_rate": (n_overlap / n_total) if n_total else None,
        "agreement_when_both": (n_overlap / n_both) if n_both else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Audit Re-aggregation Report v2")
    lines.append("")
    lines.append("Post-processed from raw per-sample metrics. **No inference re-run, no judge re-call.**")
    lines.append("")
    lines.append(f"- Cell format: `mean ± std (n=valid/total)`")
    lines.append(f"- Cells với **n_valid < {MIN_N_VALID}** → `insufficient (n=X/200)` (sample size không đủ tin cậy)")
    lines.append(f"- Citation metrics có **macro** (per-record mean) + **micro** (corpus-level Σ correct / Σ extracted)")
    lines.append(f"- `citation_text_coverage` = % records mà `citation_ids` (pipeline) overlap với `parse_citations(text)` (regex)")
    lines.append(f"  → expose gap khi elite IRAC text dùng format non-standard mà pipeline fallback bù lại")
    lines.append("")

    csv_rows: list[dict] = []

    for src in SOURCES:
        if not src["metrics_json"].exists():
            print(f"SKIP {src['name']}: {src['metrics_json']} not found", file=sys.stderr)
            continue
        with src["metrics_json"].open(encoding="utf-8") as f:
            data = json.load(f)

        lines.append(f"\n## Source: {src['name']}")
        lines.append("")
        lines.append(f"- metrics.json: `{src['metrics_json']}`")
        lines.append(f"- raw records: `{src['raw_root']}/{{arm}}/A*.json`")
        lines.append("")

        for arm in sorted(data.keys()):
            recs = data[arm]
            n_total = len(recs)
            raw_dir = src["raw_root"] / arm
            raw_map = load_raw_records(raw_dir)

            lines.append(f"### `{arm}` (n_total={n_total}, raw_loaded={len(raw_map)})")
            lines.append("")
            lines.append("| Metric | Macro mean ± std (n_valid/total) | Micro Σ correct / Σ extracted |")
            lines.append("|---|---|---|")

            csv_row = {"source": src["name"], "arm": arm, "n_total": n_total}

            for metric, chain, num_chain, denom_chain in METRIC_SPEC:
                agg = aggregate_metric(recs, chain, num_chain, denom_chain)
                # Macro cell
                if metric in ("cost_usd", "latency_s"):
                    # These should always be n_valid=n_total (no None)
                    macro_str = fmt_macro_no_n(agg["macro_mean"], agg["macro_std"],
                                               agg["n_valid"], agg["n_total"])
                else:
                    macro_str = fmt_macro(agg["macro_mean"], agg["macro_std"],
                                          agg["n_valid"], agg["n_total"])

                # Micro cell
                if metric == "hallucination_rate":
                    num_h, den_h = aggregate_halu_micro(recs)
                    micro_str = fmt_micro(num_h, den_h)
                elif agg["micro_num"] is not None:
                    micro_str = fmt_micro(agg["micro_num"], agg["micro_denom"])
                else:
                    micro_str = "—"

                lines.append(f"| {metric} | {macro_str} | {micro_str} |")

                # CSV
                csv_row[f"{metric}_macro_mean"] = agg["macro_mean"]
                csv_row[f"{metric}_macro_std"] = agg["macro_std"]
                csv_row[f"{metric}_n_valid"] = agg["n_valid"]
                if metric == "hallucination_rate":
                    num_h, den_h = aggregate_halu_micro(recs)
                    csv_row[f"{metric}_micro_num"] = num_h
                    csv_row[f"{metric}_micro_denom"] = den_h
                    csv_row[f"{metric}_micro"] = num_h / den_h if den_h else None
                elif agg["micro_num"] is not None:
                    csv_row[f"{metric}_micro_num"] = agg["micro_num"]
                    csv_row[f"{metric}_micro_denom"] = agg["micro_denom"]
                    csv_row[f"{metric}_micro"] = agg["micro_rate"]

            # Prolog metrics (elite only)
            if is_elite(arm):
                pr = aggregate_prolog(recs)
                if pr:
                    n_v = pr["n_valid"]
                    n_t = pr["n_total"]
                    for name in ("prolog_success_rate", "first_try_success_rate",
                                 "repair_invoked_rate"):
                        d = pr[name]
                        if n_v < MIN_N_VALID:
                            s = f"insufficient (n={n_v}/{n_t})"
                        else:
                            s = f"{d['rate']:.4f} (Σ={d['num']}/{d['denom']}, n_total={n_t})"
                        lines.append(f"| {name} | {s} | — |")
                        csv_row[name] = d["rate"]
                    avg = pr["avg_repair_rounds"]
                    if avg["n"] >= MIN_N_VALID:
                        s = f"{avg['mean']:.4f} (n={avg['n']}/{n_t})"
                    else:
                        s = f"insufficient (n={avg['n']}/{n_t})"
                    lines.append(f"| avg_repair_rounds | {s} | — |")
                    csv_row["avg_repair_rounds"] = avg["mean"]

            # citation_text_coverage
            cov = compute_text_coverage(recs, raw_map)
            if cov["n_total"]:
                cov_main = f"{cov['n_overlap']}/{cov['n_total']} = **{cov['coverage_rate']:.4f}**"
                cov_detail = (f"both={cov['n_both']}, "
                              f"ids_only={cov['n_ids_only']}, "
                              f"text_only={cov['n_text_only']}, "
                              f"neither={cov['n_neither']}")
                agreement = (f"agreement_when_both={cov['agreement_when_both']:.4f}"
                             if cov["agreement_when_both"] is not None
                             else "agreement_when_both=N/A")
                lines.append(f"| **citation_text_coverage** | {cov_main}<br>{cov_detail} | {agreement} |")
                csv_row["citation_text_coverage"] = cov["coverage_rate"]
                csv_row["citation_text_n_both"] = cov["n_both"]
                csv_row["citation_text_n_ids_only"] = cov["n_ids_only"]
                csv_row["citation_text_n_text_only"] = cov["n_text_only"]
                csv_row["citation_text_n_neither"] = cov["n_neither"]
                csv_row["citation_text_agreement_when_both"] = cov["agreement_when_both"]

            csv_rows.append(csv_row)
            lines.append("")

    # Cross-source comparison table for citation_validity (the most polluted metric)
    lines.append("\n## Cross-source citation_validity sanity check")
    lines.append("")
    lines.append("> Đây là metric bị selection bias nặng nhất ở R1. So sánh macro vs micro để thấy gap.")
    lines.append("")
    lines.append("| Source | Arm | Macro mean | Micro rate | n_valid (records with ≥1 citation) | n_total |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in csv_rows:
        macro = row.get("citation_validity_macro_mean")
        micro = row.get("citation_validity_micro")
        n_v = row.get("citation_validity_n_valid", 0)
        n_t = row.get("n_total", 0)
        macro_s = f"{macro:.4f}" if macro is not None else "N/A"
        micro_s = f"{micro:.4f}" if micro is not None else "N/A"
        lines.append(f"| {row['source']} | `{row['arm']}` | {macro_s} | {micro_s} | "
                     f"{n_v} | {n_t} |")
    lines.append("")

    # Citation text coverage cross-source
    lines.append("\n## citation_text_coverage cross-source")
    lines.append("")
    lines.append("> % records mà citation_ids (pipeline output) overlap với "
                 "parse_citations(answer_text). Thấp = pipeline output không trùng "
                 "với text-level regex extraction → asymmetric metric inputs.")
    lines.append("")
    lines.append("| Source | Arm | overlap/total | both_nonempty | ids_only | text_only | neither |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in csv_rows:
        cov = row.get("citation_text_coverage")
        if cov is None:
            continue
        cov_s = f"{cov:.4f}" if cov is not None else "N/A"
        lines.append(f"| {row['source']} | `{row['arm']}` | {cov_s} | "
                     f"{row.get('citation_text_n_both', 0)} | "
                     f"{row.get('citation_text_n_ids_only', 0)} | "
                     f"{row.get('citation_text_n_text_only', 0)} | "
                     f"{row.get('citation_text_n_neither', 0)} |")
    lines.append("")

    # Interpretation notes
    lines.append("\n## Interpretation notes")
    lines.append("")
    lines.append("1. **Macro vs Micro divergence**: nếu micro << macro → vài records "
                 "có ít citations nhưng valid kéo macro lên cao. Micro phản ánh "
                 "corpus-level reality.")
    lines.append("2. **n_valid << n_total**: arm hiếm khi sinh citation. So sánh "
                 "macro mean với arm có n_valid cao là so sánh apples-to-oranges.")
    lines.append("3. **citation_text_coverage thấp + n_ids_only cao**: pipeline "
                 "fallback parser (Prolog `legal_source(...)` facts) đang đóng góp "
                 "nhiều citations, nhưng text không có những citations đó → "
                 "text-based metrics (recall, precision) sẽ undercounted.")
    lines.append("4. **insufficient cells**: số sample < 30, không nên kết luận. "
                 "Để tính cell có ý nghĩa thống kê, cần re-run với prompt sửa hoặc "
                 "metric mở rộng.")
    lines.append("")

    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {REPORT_OUT} ({REPORT_OUT.stat().st_size / 1024:.1f} KB)")

    # CSV
    if csv_rows:
        all_keys: list[str] = []
        seen = set()
        # Preserve insertion order
        for row in csv_rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for row in csv_rows:
                w.writerow(row)
        print(f"Saved: {CSV_OUT} ({CSV_OUT.stat().st_size / 1024:.1f} KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
