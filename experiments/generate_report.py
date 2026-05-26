"""Sinh report markdown + CSV từ metrics.json (N arms support).

Output:
    reports/experiment_report.md  — 5-arm comparison table + Prolog reliability
                                    section + breakdown by law version + discussion
    data/eval/metrics.csv         — wide CSV (1 row / câu / arm)
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

METRICS_PATH = Path("data/eval/metrics.json")
QUESTIONS_PATH = Path("data/eval/questions_200.json")
REPORT_OUT = Path("reports/experiment_report.md")
CSV_OUT = Path("data/eval/metrics.csv")

PAIRWISE_BASELINE = "graphrag"
ELITE_ARMS = {"elite_no_retrieval", "elite_ontology", "elite_graphrag"}

# Existing 6 base metrics
METRIC_SPECS = [
    ("citation_validity",  ["citation_validity", "validity_rate"]),
    ("citation_recall",    ["citation_recall", "recall"]),
    ("citation_precision", ["citation_precision", "precision"]),
    ("faithfulness",       ["faithfulness", "faithfulness"]),
    ("answer_relevance",   ["answer_relevance", "answer_relevance"]),
    ("hallucination_rate", ["hallucination", "hallucination_rate"]),
    ("bertscore_f1",       ["bertscore", "bertscore_f1"]),
    ("cost_usd",           ["cost", "cost_usd"]),
    ("latency_s",          ["latency", "latency_s"]),
]
DIRECTION_BETTER = {
    "citation_validity":     "higher",
    "citation_recall":       "higher",
    "citation_precision":    "higher",
    "faithfulness":          "higher",
    "answer_relevance":      "higher",
    "hallucination_rate":    "lower",
    "bertscore_f1":          "higher",
    "cost_usd":              "lower",
    "latency_s":             "lower",
    # Prolog rollback (Logic-LM)
    "prolog_success_rate":   "higher",
    "first_try_success_rate": "higher",
    "repair_invoked_rate":   "lower",
    "avg_repair_rounds":     "lower",
}

# 4 Prolog rollback metrics — chỉ tính cho elite_* arms
PROLOG_SPECS = [
    ("prolog_success_rate",     ["prolog_rollback", "prolog_success"],   "bool"),
    ("first_try_success_rate",  ["prolog_rollback", "first_try_success"], "bool"),
    ("repair_invoked_rate",     ["prolog_rollback", "repair_invoked"],   "bool"),
    ("avg_repair_rounds",       ["prolog_rollback", "n_repair_rounds"],  "num"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(values):
    vs = [v for v in values if v is not None]
    return mean(vs) if vs else None


def _safe_median(values):
    vs = [v for v in values if v is not None]
    return median(vs) if vs else None


def _safe_std(values):
    vs = [v for v in values if v is not None]
    return stdev(vs) if len(vs) > 1 else None


def _extract(metrics, key_chain):
    vals = []
    for r in metrics:
        v = r
        for k in key_chain:
            if not isinstance(v, dict):
                v = None
                break
            v = v.get(k)
        vals.append(v)
    return vals


def _extract_bool_as_float(metrics, key_chain):
    """Convert True/False/None → 1.0/0.0/None để tính rate."""
    vals = []
    for r in metrics:
        v = r
        for k in key_chain:
            if not isinstance(v, dict):
                v = None
                break
            v = v.get(k)
        if v is None:
            vals.append(None)
        else:
            vals.append(1.0 if v else 0.0)
    return vals


def _tag_law_version(gold_citations_raw: str | None) -> str:
    if not gold_citations_raw:
        return "unknown"
    text = gold_citations_raw.lower()
    if "2024" in text or "41/2024" in text:
        return "new_2024"
    if "2014" in text or "58/2014" in text:
        return "old_2014"
    return "unknown"


def _fmt(v, fmt="{:.4f}"):
    return fmt.format(v) if v is not None else "N/A"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not METRICS_PATH.exists():
        print(f"FAIL: thiếu {METRICS_PATH}. Chạy compute_metrics trước.")
        return 1

    with METRICS_PATH.open(encoding="utf-8") as f:
        all_metrics = json.load(f)

    arms = list(all_metrics.keys())
    if not arms:
        print("FAIL: metrics.json không có arm nào", file=sys.stderr)
        return 1

    n_per_arm = {arm: len(recs) for arm, recs in all_metrics.items()}
    n_first = next(iter(n_per_arm.values()))

    # Map stt → law_version
    law_tag = {}
    if QUESTIONS_PATH.exists():
        with QUESTIONS_PATH.open(encoding="utf-8") as f:
            qs = json.load(f)
        for q in qs:
            law_tag[q["stt"]] = _tag_law_version(q.get("gold_citations_raw"))

    # ---- Build aggregate ----
    agg = {arm: {} for arm in arms}
    for arm in arms:
        for label, kc in METRIC_SPECS:
            vals = _extract(all_metrics[arm], kc)
            agg[arm][label] = {
                "mean": _safe_mean(vals),
                "median": _safe_median(vals),
                "std": _safe_std(vals),
                "n_valid": sum(1 for v in vals if v is not None),
            }
        # Prolog metrics (chỉ elite_*)
        if arm in ELITE_ARMS:
            for label, kc, kind in PROLOG_SPECS:
                vals = (
                    _extract_bool_as_float(all_metrics[arm], kc)
                    if kind == "bool"
                    else _extract(all_metrics[arm], kc)
                )
                agg[arm][label] = {
                    "mean": _safe_mean(vals),
                    "std": _safe_std(vals),
                    "n_valid": sum(1 for v in vals if v is not None),
                }

    # Pairwise: tally consensus per non-baseline arm
    pairwise_by_arm = {}
    for arm in arms:
        if arm == PAIRWISE_BASELINE:
            continue
        consensus = Counter()
        vote_ab = Counter()
        vote_ba = Counter()
        for r in all_metrics[arm]:
            pw = r.get("pairwise_vs_baseline")
            if pw:
                consensus[pw["consensus"]] += 1
                vote_ab[pw["vote_ab"]] += 1
                vote_ba[pw["vote_ba"]] += 1
        pairwise_by_arm[arm] = {
            "consensus": consensus,
            "vote_ab": vote_ab,
            "vote_ba": vote_ba,
            "total": sum(consensus.values()),
        }

    # Breakdown theo law version
    by_law = {
        arm: {tag: [] for tag in ("new_2024", "old_2014", "unknown")}
        for arm in arms
    }
    for arm in arms:
        for r in all_metrics[arm]:
            tag = law_tag.get(r["stt"], "unknown")
            by_law[arm][tag].append(r)

    # ===================================================================
    # Write markdown report
    # ===================================================================
    lines: list[str] = []

    arm_labels = ", ".join(arms)
    lines.append("# Experiment Report — 5-arm comparison (GraphRAG vs LLM-only vs Elite × 3)\n")
    lines.append(f"**Dataset**: 200 câu BHXH (FB group). Arms compared: `{arm_labels}`. "
                 f"Số sample / arm: " + ", ".join(f"{a}={n}" for a, n in n_per_arm.items()) + "\n")
    lines.append("**Models**: generator + judge đều `gpt-4o-mini` (self-bias risk — chỉ affect *absolute* scores, *relative* fair).\n")
    lines.append("**Arms**:\n")
    lines.append("- `graphrag`: vector search Neo4j + LLM generate answer text\n")
    lines.append("- `llm_only`: chỉ LLM, no retrieval\n")
    lines.append("- `elite_no_retrieval`: LLM → Prolog (no context, prompt relaxed) → SWI-Prolog → IRAC\n")
    lines.append("- `elite_ontology`: LLM → Prolog (ontology retrieval) → SWI-Prolog → IRAC\n")
    lines.append("- `elite_graphrag`: LLM → Prolog (GraphRAG retrieval) → SWI-Prolog → IRAC\n\n")

    # Paper refs
    lines.append("## Metrics & paper refs (peer-reviewed, không arXiv)\n")
    lines.append("| Metric | Paper | Venue |")
    lines.append("|---|---|---|")
    lines.append("| Faithfulness, Answer Relevance | Es et al. *RAGAs: Automated Evaluation of Retrieval Augmented Generation* | [EACL 2024 Demo](https://aclanthology.org/2024.eacl-demo.16/) |")
    lines.append("| Citation Precision/Recall | Liu, Zhang & Liang. *Evaluating Verifiability in Generative Search Engines* | [EMNLP Findings 2023](https://aclanthology.org/2023.findings-emnlp.467/) |")
    lines.append("| Hallucination Rate (legal) | Magesh et al. *Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools* | [JELS 2025, Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/jels.12413) (Stanford RegLab/HAI) |")
    lines.append("| LLM-as-Judge (pairwise) | Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* | [NeurIPS 2023 D&B](https://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html) |")
    lines.append("| BERTScore | Zhang et al. *BERTScore: Evaluating Text Generation with BERT* | [ICLR 2020 (OpenReview)](https://openreview.net/forum?id=SkeHuCVFDr) |")
    lines.append("| **Prolog rollback rate** (Logic-LM family) | Pan et al. *Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning* | [EMNLP Findings 2023](https://aclanthology.org/2023.findings-emnlp.248/) |")
    lines.append("\n")

    # ---- 5-arm aggregate table ----
    lines.append("## Aggregate results (mean ± std)\n")
    header = "| Metric | " + " | ".join(arms) + " | Direction |"
    sep = "|---|" + "|".join("---" for _ in arms) + "|---|"
    lines.append(header)
    lines.append(sep)
    for label, _ in METRIC_SPECS:
        cells = []
        for arm in arms:
            a = agg[arm].get(label, {})
            m, s = a.get("mean"), a.get("std")
            if m is not None and s is not None:
                cells.append(f"{m:.4f} ± {s:.4f}")
            elif m is not None:
                cells.append(f"{m:.4f}")
            else:
                cells.append("N/A")
        lines.append(f"| **{label}** | " + " | ".join(cells) + f" | {DIRECTION_BETTER[label]} better |")
    lines.append("")

    # ---- Prolog reliability section ----
    elite_present = [a for a in arms if a in ELITE_ARMS]
    if elite_present:
        lines.append("## Prolog reliability (Logic-LM metrics — chỉ áp dụng cho elite arms)\n")
        lines.append("> Đo độ tin cậy của symbolic solver loop. Pan et al. EMNLP'23 báo cáo các metric tương tự để compare LLM-as-reasoner vs LLM+symbolic.\n")
        header = "| Metric | " + " | ".join(elite_present) + " | Direction |"
        sep = "|---|" + "|".join("---" for _ in elite_present) + "|---|"
        lines.append(header)
        lines.append(sep)
        for label, _, _ in PROLOG_SPECS:
            cells = []
            for arm in elite_present:
                a = agg[arm].get(label, {})
                m = a.get("mean")
                if m is not None:
                    cells.append(f"{m:.4f}")
                else:
                    cells.append("N/A")
            lines.append(f"| **{label}** | " + " | ".join(cells) + f" | {DIRECTION_BETTER[label]} better |")
        lines.append("")
        # Breakdown của prolog_status
        lines.append("### Prolog status distribution\n")
        lines.append("| Status | " + " | ".join(elite_present) + " |")
        lines.append("|---|" + "|".join("---:" for _ in elite_present) + "|")
        all_statuses = set()
        status_counts = {arm: Counter() for arm in elite_present}
        for arm in elite_present:
            for r in all_metrics[arm]:
                s = r.get("prolog_rollback", {}).get("prolog_status") or "—"
                status_counts[arm][s] += 1
                all_statuses.add(s)
        for status in sorted(all_statuses):
            cells = [str(status_counts[arm].get(status, 0)) for arm in elite_present]
            lines.append(f"| `{status}` | " + " | ".join(cells) + " |")
        lines.append("")

    # ---- Pairwise judge (vs baseline) ----
    lines.append(f"## Pairwise judge vs `{PAIRWISE_BASELINE}` (LLM-as-Judge, position swap)\n")
    for arm, data in pairwise_by_arm.items():
        if not data["total"]:
            continue
        total = data["total"]
        lines.append(f"### `{arm}` vs `{PAIRWISE_BASELINE}` (n={total})\n")
        lines.append("| Consensus | Count | % |")
        lines.append("|---|---:|---:|")
        for k, v in data["consensus"].most_common():
            lines.append(f"| {k} | {v} | {v/total*100:.1f}% |")
        lines.append("")
        lines.append(f"**Position swap detail:**\n")
        lines.append(f"| Vote | A={PAIRWISE_BASELINE} B={arm} | A={arm} B={PAIRWISE_BASELINE} |")
        lines.append("|---|---:|---:|")
        voters = set(data["vote_ab"].keys()) | set(data["vote_ba"].keys())
        for v in sorted(voters):
            lines.append(f"| {v} | {data['vote_ab'].get(v, 0)} | {data['vote_ba'].get(v, 0)} |")
        lines.append("")

    # ---- Breakdown theo luật version ----
    lines.append("## Breakdown theo luật version (từ gold_citations_raw)\n")
    for tag in ("new_2024", "old_2014", "unknown"):
        n_questions = len(by_law[arms[0]][tag])
        if n_questions == 0:
            continue
        lines.append(f"### `{tag}` ({n_questions} câu)\n")
        header = "| Metric | " + " | ".join(arms) + " |"
        sep = "|---|" + "|".join("---" for _ in arms) + "|"
        lines.append(header)
        lines.append(sep)
        for label, kc in METRIC_SPECS:
            cells = []
            for arm in arms:
                vals = _extract(by_law[arm][tag], kc)
                m = _safe_mean(vals)
                cells.append(_fmt(m))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    # ---- Discussion (auto-generated insights) ----
    lines.append("## Discussion (auto-generated)\n")

    # Best arm per metric
    lines.append("### Winner per metric\n")
    lines.append("| Metric | Winner | Value |")
    lines.append("|---|---|---|")
    for label, _ in METRIC_SPECS:
        better = DIRECTION_BETTER[label]
        scores = [(arm, agg[arm].get(label, {}).get("mean")) for arm in arms]
        scores = [(a, m) for a, m in scores if m is not None]
        if not scores:
            continue
        winner_arm, winner_val = (
            max(scores, key=lambda x: x[1]) if better == "higher"
            else min(scores, key=lambda x: x[1])
        )
        lines.append(f"| {label} | **{winner_arm}** | {winner_val:.4f} |")
    if elite_present:
        for label, _, _ in PROLOG_SPECS:
            better = DIRECTION_BETTER[label]
            scores = [(arm, agg[arm].get(label, {}).get("mean")) for arm in elite_present]
            scores = [(a, m) for a, m in scores if m is not None]
            if not scores:
                continue
            winner_arm, winner_val = (
                max(scores, key=lambda x: x[1]) if better == "higher"
                else min(scores, key=lambda x: x[1])
            )
            lines.append(f"| {label} | **{winner_arm}** | {winner_val:.4f} |")
    lines.append("")

    # Elite no-retrieval observation
    if "elite_no_retrieval" in arms:
        a = agg["elite_no_retrieval"].get("prolog_success_rate", {}).get("mean")
        if a is not None:
            lines.append(
                f"**Elite no-retrieval ablation**: prolog_success_rate = {a:.0%}. "
                f"Càng thấp càng chứng minh elite CẦN retrieval. "
                f"Câu nào success nhờ LLM tự sinh được valid Prolog từ training data.\n"
            )

    # Elite ontology vs elite graphrag
    if "elite_ontology" in arms and "elite_graphrag" in arms:
        o_succ = agg["elite_ontology"].get("prolog_success_rate", {}).get("mean")
        g_succ = agg["elite_graphrag"].get("prolog_success_rate", {}).get("mean")
        if o_succ is not None and g_succ is not None:
            better = "elite_graphrag" if g_succ > o_succ else "elite_ontology"
            lines.append(
                f"**Ontology vs GraphRAG retrieval for symbolic reasoning**: "
                f"elite_ontology success={o_succ:.0%}, elite_graphrag success={g_succ:.0%}. "
                f"`{better}` retrieval cho ra Prolog program hợp lệ thường xuyên hơn.\n"
            )

    # ---- Caveats ----
    lines.append("\n## Caveats / Limitations\n")
    lines.append("1. **Self-enhancement bias** (Zheng 2023): judge = generator = `gpt-4o-mini` → bias đều cả 5 arm. Relative compare OK, absolute có thể inflated.")
    lines.append("2. **Elite no-retrieval prompt được relax** cho phép LLM tự cite — citation_validity của arm này dùng để cảnh báo (không equivalent với D/E).")
    lines.append("3. **Citation extraction** từ IRAC text (elite arms) dùng cả bracketed `[Điều X khoản Y]` và inline `Điều X, khoản Y` patterns; có thể miss vài citation natural language.")
    lines.append("4. **Pairwise judge** vs `graphrag` baseline có position bias mạnh (đã thấy trong eval 2-arm trước). Chỉ tin strong-consensus rows.")
    lines.append("5. **Prolog rollback** đo trên max=2 repair rounds (default elite). Cap thấp → chưa thấy điểm hội tụ thật của LLM-with-feedback.")
    lines.append("6. **SWI-Prolog timeout=15s** — câu phức tạp có thể bị giết silently → count vào prolog_success=False (status có thể là 'unable_to_conclude').\n")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report: {REPORT_OUT}")

    # ===================================================================
    # CSV (per-question wide format)
    # ===================================================================
    fieldnames = ["stt", "arm", "law_version"]
    for label, _ in METRIC_SPECS:
        fieldnames.append(label)
    for label, _, _ in PROLOG_SPECS:
        fieldnames.append(label)
    fieldnames.append("pairwise_consensus")
    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for arm in arms:
            for r in all_metrics[arm]:
                row = {
                    "stt": r["stt"],
                    "arm": arm,
                    "law_version": law_tag.get(r["stt"], "unknown"),
                }
                for label, kc in METRIC_SPECS:
                    vals = _extract([r], kc)
                    row[label] = vals[0] if vals[0] is not None else ""
                if arm in ELITE_ARMS:
                    for label, kc, kind in PROLOG_SPECS:
                        v = _extract([r], kc)[0]
                        if v is None:
                            row[label] = ""
                        else:
                            row[label] = (1 if v else 0) if kind == "bool" else v
                else:
                    for label, _, _ in PROLOG_SPECS:
                        row[label] = ""
                row["pairwise_consensus"] = (
                    r.get("pairwise_vs_baseline", {}).get("consensus", "")
                )
                w.writerow(row)
    print(f"Saved CSV   : {CSV_OUT}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
