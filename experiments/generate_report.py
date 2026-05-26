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

# AUDIT v2 (2026-05-26): split hallucination, drop BERTScore + answer_relevance
# cho elite arms (structured IRAC text vs free prose không fair compare).
METRIC_SPECS = [
    ("citation_validity",            ["citation_validity", "validity_rate"]),
    ("citation_recall",              ["citation_recall", "recall"]),
    ("citation_precision",           ["citation_precision", "precision"]),
    ("faithfulness",                 ["faithfulness", "faithfulness"]),
    ("content_hallucination_rate",   ["hallucination", "content_hallucination_rate"]),
    ("invented_citation_rate",       ["hallucination", "invented_citation_rate"]),
    ("answer_relevance",             ["answer_relevance", "answer_relevance"]),
    ("bertscore_f1",                 ["bertscore", "bertscore_f1"]),
    ("cost_usd",                     ["cost", "cost_usd"]),
    ("latency_s",                    ["latency", "latency_s"]),
]
# Metrics có structural bias với IRAC format (drop hoặc flag cho elite arms)
ELITE_BIASED_METRICS = {"answer_relevance", "bertscore_f1"}

DIRECTION_BETTER = {
    "citation_validity":     "higher",
    "citation_recall":       "higher",
    "citation_precision":    "higher",
    "faithfulness":          "higher",
    "answer_relevance":      "higher",
    "hallucination_rate":            "lower",
    "content_hallucination_rate":    "lower",
    "invented_citation_rate":        "lower",
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


def _value_chain(rec, chain):
    """Extract single value via nested keys from one record. None if missing."""
    v = rec
    for k in chain:
        if not isinstance(v, dict):
            return None
        v = v.get(k)
    return v


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
    MIN_N_VALID = 30  # threshold: below this → "insufficient (n=X/Y)"
    n_per_arm_total = {arm: len(all_metrics[arm]) for arm in arms}

    # API error count per arm (from audit_apply_fixes_v2 tagging)
    api_err_per_arm = {arm: sum(1 for r in all_metrics[arm] if r.get("api_error"))
                       for arm in arms}

    # Re-aggregate excluding API-error records (cells affected: ALL)
    def _arm_clean_recs(arm):
        return [r for r in all_metrics[arm] if not r.get("api_error")]

    lines.append("## Aggregate results (macro mean ± std, n_valid/total)\n")
    lines.append(f"> Cells với `n_valid < {MIN_N_VALID}` → 'insufficient' (sample size không đủ tin cậy).")
    lines.append(f"> Khi n_valid khác n_total → metric chỉ đo trên subset of records có valid value (selection bias warning).")
    if any(api_err_per_arm.values()):
        msg = ", ".join(f"{a}={n}" for a, n in api_err_per_arm.items() if n > 0)
        lines.append(f"> **API errors** (records với prompt+completion tokens = 0): {msg}. Đã exclude khỏi mọi aggregate dưới.")
    lines.append("")
    header = "| Metric | " + " | ".join(arms) + " | Direction |"
    sep = "|---|" + "|".join("---" for _ in arms) + "|---|"
    lines.append(header)
    lines.append(sep)
    for label, chain in METRIC_SPECS:
        cells = []
        for arm in arms:
            # Drop biased metrics for elite arms (BERTScore/AR — structural format diff).
            # Khi `plain_answer` field tồn tại trong raw records (từ new IRAC+plain
            # prompt), compute_metrics tự dùng plain_answer → cell sẽ là fair value;
            # nếu raw records không có plain_answer (pre-2026-05-27 data), drop.
            if label in ELITE_BIASED_METRICS and arm in ELITE_ARMS:
                # Sniff: check 1st metric record's `_used_plain_answer` flag
                used_plain = (all_metrics[arm][0].get(label, {}).get("_used_plain_answer")
                              if all_metrics[arm] else False)
                if not used_plain:
                    cells.append("dropped (IRAC bias)")
                    continue
                # else: fall through to normal cell render (computed on plain_answer)
            clean = _arm_clean_recs(arm)
            vals = _extract(clean, chain)
            m = _safe_mean(vals)
            s = _safe_std(vals)
            n_valid = sum(1 for v in vals if v is not None)
            n_total = len(clean)
            if n_valid < MIN_N_VALID:
                cells.append(f"insufficient (n={n_valid}/{n_total})")
            elif m is None:
                cells.append(f"N/A (n=0/{n_total})")
            elif s is not None:
                cells.append(f"{m:.4f} ± {s:.4f} (n={n_valid}/{n_total})")
            else:
                cells.append(f"{m:.4f} (n={n_valid}/{n_total})")
        lines.append(f"| **{label}** | " + " | ".join(cells) + f" | {DIRECTION_BETTER[label]} better |")
    lines.append("")
    lines.append("> **'dropped (IRAC bias)'** = elite arms output structured IRAC text "
                 "(Issue/Rule/Application/Conclusion headers), không thể fair compare "
                 "với free prose của graphrag/llm_only bằng BERTScore (lexical overlap) "
                 "hay answer_relevance (self-similarity với generated questions).")
    lines.append("")

    # ---- Micro-average citation metrics (corpus-level Σ correct / Σ extracted) ----
    lines.append("### Citation metrics — macro vs micro\n")
    lines.append("> **Macro** = mean of per-record rates (current table).\n"
                 "> **Micro** = corpus-level Σ correct / Σ extracted (less sensitive to records with few citations).\n")
    micro_specs = [
        ("citation_validity", ["citation_validity", "n_valid"], ["citation_validity", "n_citations"]),
        ("citation_recall", ["citation_recall", "n_with_cite"], ["citation_recall", "n_sentences"]),
        ("citation_precision", ["citation_precision", "n_supported"], ["citation_precision", "n_citations"]),
        ("faithfulness", ["faithfulness", "n_supported"], ["faithfulness", "n_claims"]),
    ]
    lines.append("| Metric | " + " | ".join(arms) + " |")
    lines.append("|---|" + "|".join("---" for _ in arms) + "|")
    for label, num_chain, denom_chain in micro_specs:
        cells = []
        for arm in arms:
            num = sum(int((_value_chain(r, num_chain) or 0)) for r in all_metrics[arm])
            denom = sum(int((_value_chain(r, denom_chain) or 0)) for r in all_metrics[arm])
            if denom == 0:
                cells.append("N/A")
            else:
                cells.append(f"{num/denom:.4f} (Σ={num}/{denom})")
        lines.append(f"| **{label}** | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Prolog reliability section ----
    elite_present = [a for a in arms if a in ELITE_ARMS]
    if elite_present:
        lines.append("## Prolog reliability (Logic-LM metrics — chỉ áp dụng cho elite arms)\n")
        lines.append("> Đo độ tin cậy của symbolic solver loop. Pan et al. EMNLP'23 báo cáo các metric tương tự để compare LLM-as-reasoner vs LLM+symbolic.")
        lines.append("> **API-error records excluded** từ mọi tỉ lệ dưới (tránh ô nhiễm với infrastructure failures).")
        lines.append("")
        header = "| Metric | " + " | ".join(elite_present) + " | Direction |"
        sep = "|---|" + "|".join("---" for _ in elite_present) + "|---|"
        lines.append(header)
        lines.append(sep)
        for label, chain, kind in PROLOG_SPECS:
            cells = []
            for arm in elite_present:
                clean = _arm_clean_recs(arm)
                if kind == "bool":
                    vals = _extract_bool_as_float(clean, chain)
                else:
                    vals = _extract(clean, chain)
                m = _safe_mean(vals)
                n_v = sum(1 for v in vals if v is not None)
                cells.append(f"{m:.4f} (n={n_v})" if m is not None else "N/A")
            lines.append(f"| **{label}** | " + " | ".join(cells) + f" | {DIRECTION_BETTER[label]} better |")
        lines.append("")
        # Show API error counts as separate row
        if any(api_err_per_arm.get(a, 0) > 0 for a in elite_present):
            lines.append("### API error rate (excluded from above)\n")
            lines.append("| Arm | API errors | % of 200 |")
            lines.append("|---|---:|---:|")
            for arm in elite_present:
                n = api_err_per_arm.get(arm, 0)
                lines.append(f"| {arm} | {n} | {n/200*100:.1f}% |")
            lines.append("")

        # Breakdown của prolog_status — show on RAW records (including API errors)
        lines.append("### Prolog status distribution (raw, including API errors)\n")
        lines.append("> Note: `unable_to_conclude` count cho elite arms có thể bao gồm API errors. Số real Prolog failures = (unable_to_conclude − api_errors).")
        lines.append("")
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
        cons = data["consensus"]
        n_split = cons.get("split", 0)
        n_consistent = total - n_split
        n_arm_wins = cons.get(arm, 0)
        n_base_wins = cons.get(PAIRWISE_BASELINE, 0)
        n_tie = cons.get("tie", 0)
        lines.append(f"### `{arm}` vs `{PAIRWISE_BASELINE}` (n={total})\n")
        lines.append("**Consensus breakdown:**\n")
        lines.append("| Consensus | Count | % of all (n=200) |")
        lines.append("|---|---:|---:|")
        for k, v in cons.most_common():
            lines.append(f"| {k} | {v} | {v/total*100:.1f}% |")
        lines.append("")
        # AUDIT FIX: report on consistent-verdict subset
        lines.append(f"**On consistent-verdict subset** (n_consistent = {n_consistent} = "
                     f"{total} − {n_split} split):")
        lines.append("")
        if n_consistent > 0:
            lines.append(f"- **{arm}**: {n_arm_wins}/{n_consistent} = "
                         f"{n_arm_wins/n_consistent*100:.1f}% wins")
            lines.append(f"- **{PAIRWISE_BASELINE}**: {n_base_wins}/{n_consistent} = "
                         f"{n_base_wins/n_consistent*100:.1f}% wins")
            if n_tie:
                lines.append(f"- ties: {n_tie}/{n_consistent} = "
                             f"{n_tie/n_consistent*100:.1f}%")
        else:
            lines.append("- No consistent verdicts (all split or no data)")
        lines.append("")
        lines.append(f"**Position-swap detail (raw votes per direction):**\n")
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

    # Best arm per metric — skip arms with n_valid < MIN_N_VALID để tránh
    # misleading winners.
    MIN_N_VALID = 30
    lines.append("### Winner per metric\n")
    lines.append("> Arms với `n_valid < 30` cho metric đó bị loại khỏi competition (insufficient sample).\n")
    lines.append("| Metric | Winner | Value | n_valid |")
    lines.append("|---|---|---|---:|")
    for label, _ in METRIC_SPECS:
        better = DIRECTION_BETTER[label]
        scores = []
        for arm in arms:
            a = agg[arm].get(label, {})
            m = a.get("mean")
            n_v = a.get("n_valid", 0)
            if m is None or n_v < MIN_N_VALID:
                continue
            scores.append((arm, m, n_v))
        if not scores:
            lines.append(f"| {label} | — (all insufficient) | — | — |")
            continue
        winner = max(scores, key=lambda x: x[1]) if better == "higher" else min(scores, key=lambda x: x[1])
        lines.append(f"| {label} | **{winner[0]}** | {winner[1]:.4f} | {winner[2]} |")
    if elite_present:
        for label, _, _ in PROLOG_SPECS:
            better = DIRECTION_BETTER[label]
            scores = []
            for arm in elite_present:
                a = agg[arm].get(label, {})
                m = a.get("mean")
                n_v = a.get("n_valid", 0)
                if m is None or n_v < MIN_N_VALID:
                    continue
                scores.append((arm, m, n_v))
            if not scores:
                continue
            winner = max(scores, key=lambda x: x[1]) if better == "higher" else min(scores, key=lambda x: x[1])
            lines.append(f"| {label} | **{winner[0]}** | {winner[1]:.4f} | {winner[2]} |")
    lines.append("")

    # Pairwise winner per arm (USING FIXED PAIRWISE DATA)
    lines.append("### Pairwise winner per arm (vs `graphrag` baseline)\n")
    lines.append("> Strong-consensus wins (both directions agree). Numbers từ corrected `_vote` logic (2026-05-26 fix).\n")
    lines.append("| Arm | Wins vs graphrag | graphrag wins | Split | Tie | Verdict |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for arm in arms:
        if arm == PAIRWISE_BASELINE:
            continue
        pwdata = pairwise_by_arm.get(arm)
        if not pwdata or not pwdata["total"]:
            continue
        cons = pwdata["consensus"]
        n_tot = pwdata["total"]
        w_arm = cons.get(arm, 0)
        w_base = cons.get(PAIRWISE_BASELINE, 0)
        n_split = cons.get("split", 0)
        n_tie = cons.get("tie", 0)
        # Verdict
        if w_arm > w_base * 1.5:
            verdict = f"**{arm} beats graphrag**"
        elif w_base > w_arm * 1.5:
            verdict = f"**graphrag beats {arm}**"
        else:
            verdict = "Tied / mixed"
        lines.append(f"| {arm} | {w_arm} ({w_arm/n_tot*100:.1f}%) | "
                     f"{w_base} ({w_base/n_tot*100:.1f}%) | "
                     f"{n_split} ({n_split/n_tot*100:.1f}%) | {n_tie} | {verdict} |")
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
    lines.append("3. **Citation source asymmetry** (audit 2026-05-26): `citation_validity` + `faithfulness` + `hallucination` dùng `record['citation_ids']` (bao gồm fallback parser của Prolog `legal_source(...)` facts). `citation_recall` + `citation_precision` chỉ dùng `parse_citations(answer_text)` — regex trên text. Khi IRAC text không có `[Điều X]` brackets (elite arms), text-based metrics có thể undercount vs ID-based metrics. Xem `reports/report_v2.md` cho `citation_text_coverage` per arm.")
    lines.append("4. **Selection bias**: cells với `n_valid < 30` được mark 'insufficient'. Cells với `n_valid << n_total` (e.g., elite_no_retrieval citation_validity n=52) là conditional mean trên subset, KHÔNG comparable trực tiếp với arm có `n_valid` cao. Macro + micro tables được show để giảm bias.")
    lines.append("5. **Pairwise judge** (FIXED 2026-05-26): trước đây `_vote(w, a_first=False)` invert vote_ba → tất cả consensus đều thành 'split'. Đã fix — bảng dưới phản ánh đúng. Position bias trong realityー moderate, không 'mạnh' như báo cáo cũ.")
    lines.append("6. **Prolog rollback** đo trên max=2 repair rounds (default elite). Cap thấp → chưa thấy điểm hội tụ thật của LLM-with-feedback.")
    lines.append("7. **SWI-Prolog timeout=15s** — câu phức tạp có thể bị giết silently → count vào prolog_success=False (status có thể là 'unable_to_conclude').\n")

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
