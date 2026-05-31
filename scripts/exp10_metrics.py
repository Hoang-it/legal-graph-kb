"""Experiment 10 metrics — HyDE GRACE-R 3-arm retrieval metrics.

Clone of ``scripts/exp09_metrics.py`` with two changes:

1. ``EXP_DIR`` → ``experiments/10_hyde_gracer``.
2. No pilot-subset auto-detection. Scores every record on disk —
   smoke runs land at n=5, full runs land at n=200. exp 10's
   ``--pilot-5`` smoke test is too small to share strata with exp
   08's pilot_50; auto-filtering would produce 0-question
   intersections.

Reads per-arm records produced by ``scripts/exp10_run.py``:

    experiments/10_hyde_gracer/results/<arm>/A<stt>.json

for arms ``dense``, ``dense_hyde``, ``dense_hyde2``.

Computes the same metrics as exp 09 (recall/precision/F1/NDCG at
K ∈ {12, 20, 30, 50, 70, 100, all}; R-Precision; MRR; latency).

Outputs:
- ``metrics/gold_citations_normalized.json``
- ``metrics/academic_metrics.json``
- ``metrics/academic_metrics.csv``
- ``report/academic_report.md``

The report's "success criteria" section compares against exp 09's
baseline numbers (frozen): dense=0.3832, dense_hyde=0.4736,
dense_hyde2=0.4210 R@12 in_corpus n=151. See exp 09 README.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

from eval_core.gold import validate_gold_citations
from src.legal_metadata import load_law_metadata

EXP_DIR = _REPO / "experiments" / "10_hyde_gracer"
RESULTS_DIR = EXP_DIR / "results"
METRICS_DIR = EXP_DIR / "metrics"
REPORT_DIR = EXP_DIR / "report"
QUESTIONS_PATH = _REPO / "data" / "eval" / "questions_200.json"
REGISTRY_PATH = _REPO / "data" / "legal_sources.yaml"

# Exp 09 frozen baselines (in_corpus stratum, n=151 full 200).
# Source: experiments/09_hyde2_grounded/README.md Result summary table.
EXP09_BASELINE_IN_CORPUS = {
    "dense": {
        "recall@12": 0.3832,
        "recall@100": 0.6592,
        "ndcg@12": 0.2186,
        "r_precision": 0.0635,
        "mrr": 0.2122,
    },
    "dense_hyde": {
        "recall@12": 0.4736,
        "recall@100": 0.7016,
        "ndcg@12": 0.2944,
        "r_precision": 0.1326,
        "mrr": 0.2843,
    },
    "dense_hyde2": {
        "recall@12": 0.4210,
        "recall@100": 0.5989,
        "ndcg@12": 0.2437,
        "r_precision": 0.1019,
        "mrr": 0.2192,
    },
}


ARMS = ("dense", "dense_hyde", "dense_hyde2")
KS: tuple[int | None, ...] = (12, 20, 30, 50, 70, 100, None)

_RE_CODE = re.compile(r"\d+/\d{4}/(?:QH\d+|N[ĐD]-CP|NQ-CP|TT-[A-Z]+|CP|TTg)")


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


def _retrieved_at_k(retrieved: list[str], k: int | None) -> list[str]:
    return retrieved if k is None else retrieved[:k]


def recall(retrieved_k: list[str], gold: set[str]) -> float | None:
    if not gold:
        return None
    return len(gold & set(retrieved_k)) / len(gold)


def precision(retrieved_k: list[str], gold: set[str]) -> float | None:
    if not retrieved_k:
        return 0.0 if gold else None
    return len(gold & set(retrieved_k)) / len(retrieved_k)


def f1_score(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    if (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


def r_precision(retrieved: list[str], gold: set[str]) -> float | None:
    if not gold:
        return None
    k = len(gold)
    return len(gold & set(retrieved[:k])) / k


def reciprocal_rank(retrieved: list[str], gold: set[str]) -> float | None:
    if not gold:
        return None
    for i, aid in enumerate(retrieved, start=1):
        if aid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float | None:
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


def load_gold() -> dict[int, list[str]]:
    ok, summary = validate_gold_citations(
        questions_path=QUESTIONS_PATH,
        registry_path=REGISTRY_PATH,
        out_dir=METRICS_DIR,
    )
    if not ok:
        print(f"FAIL: gold validation failed; see {summary['errors_path']}",
              file=sys.stderr)
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
        stt = int(rec["stt"])
        out[stt] = rec
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
        retrieved = list((rec.get("retrieval_only") or {}).get("final_article_ids") or [])
        gold = set(gold_map.get(stt) or [])
        row: dict = {
            "stt": stt,
            "arm": arm,
            "n_gold": len(gold),
            "n_retrieved_all": len(retrieved),
            "category": categorize(questions[stt].get("gold_citations_raw"), in_corpus_codes),
            "elapsed_s": (rec.get("retrieval_only") or {}).get("elapsed_s"),
        }
        for k in KS:
            r_k = _retrieved_at_k(retrieved, k)
            r = recall(r_k, gold)
            p = precision(r_k, gold)
            f1 = f1_score(p, r)
            n = ndcg_at_k(retrieved, gold, k if k is not None else max(len(retrieved), 1))
            kk = "all" if k is None else str(k)
            row[f"recall@{kk}"] = r
            row[f"precision@{kk}"] = p
            row[f"f1@{kk}"] = f1
            row[f"ndcg@{kk}"] = n
            row[f"n_retrieved@{kk}"] = len(r_k)
        row["r_precision"] = r_precision(retrieved, gold)
        row["mrr"] = reciprocal_rank(retrieved, gold)
        rows.append(row)
    return rows


def aggregate_macro(rows: list[dict]) -> dict:
    summary: dict = {"n": len(rows)}
    for k in KS:
        kk = "all" if k is None else str(k)
        summary[f"recall@{kk}"] = _macro([r[f"recall@{kk}"] for r in rows])
        summary[f"precision@{kk}"] = _macro([r[f"precision@{kk}"] for r in rows])
        summary[f"f1@{kk}"] = _macro([r[f"f1@{kk}"] for r in rows])
        summary[f"ndcg@{kk}"] = _macro([r[f"ndcg@{kk}"] for r in rows])
        summary[f"avg_n_retrieved@{kk}"] = round(
            mean([r[f"n_retrieved@{kk}"] for r in rows]), 2
        ) if rows else None
    summary["r_precision"] = _macro([r["r_precision"] for r in rows])
    summary["mrr"] = _macro([r["mrr"] for r in rows])
    latencies = [r["elapsed_s"] for r in rows if r["elapsed_s"] is not None]
    summary["avg_elapsed_s"] = round(mean(latencies), 3) if latencies else None
    return summary


def aggregate_stratified(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cat in ("in_corpus", "mixed", "ooc", "unparseable"):
        sub = [r for r in rows if r["category"] == cat]
        out[cat] = aggregate_macro(sub) if sub else {"n": 0}
    return out


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
            f"ndcg@{kk}",
        ]
    fieldnames += ["r_precision", "mrr", "elapsed_s"]

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for arm, rows in per_arm_rows.items():
            for r in rows:
                row_out = {"arm": arm, "stt": r["stt"], "category": r["category"],
                           "n_gold": r["n_gold"]}
                for kk in field_ks:
                    row_out[f"n_retrieved@{kk}"] = r[f"n_retrieved@{kk}"]
                    row_out[f"recall@{kk}"] = r[f"recall@{kk}"]
                    row_out[f"precision@{kk}"] = r[f"precision@{kk}"]
                    row_out[f"f1@{kk}"] = r[f"f1@{kk}"]
                    row_out[f"ndcg@{kk}"] = r[f"ndcg@{kk}"]
                row_out["r_precision"] = r["r_precision"]
                row_out["mrr"] = r["mrr"]
                row_out["elapsed_s"] = r["elapsed_s"]
                writer.writerow(row_out)
    return out


def write_json(per_arm_summary: dict, per_arm_strat: dict) -> Path:
    out = METRICS_DIR / "academic_metrics.json"
    payload = {
        "experiment": "10_hyde_gracer",
        "arms": list(ARMS),
        "Ks": [("all" if k is None else k) for k in KS],
        "overall_macro": per_arm_summary,
        "stratified": per_arm_strat,
        "exp09_baseline_in_corpus": EXP09_BASELINE_IN_CORPUS,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_report(per_arm_summary: dict, per_arm_strat: dict, n_scored: int) -> Path:
    out = REPORT_DIR / "academic_report.md"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    field_ks = [("all" if k is None else str(k)) for k in KS]

    lines: list[str] = []
    lines.append("# Experiment 10 — HyDE prompts under GRACE-R, retrieval-only 3-arm metrics")
    lines.append("")
    lines.append(f"Dataset: {n_scored} BHXH questions. Metric granularity: article.")
    lines.append("Arms: `dense`, `dense_hyde` (HyDE1 + GRACE-R prompt), "
                 "`dense_hyde2` (HyDE2 + GRACE-R grounded prompt).")
    lines.append("Generator: OpenAI `gpt-4o-mini` (n=1, max_tokens=700, temperature=0.0).")
    lines.append("HyDE2 seed_k = 5. Prompts override at "
                 "`experiments/10_hyde_gracer/prompts_override/`.")
    lines.append("")
    lines.append(f"## Overall macro (n={n_scored} with non-empty gold)")
    lines.append("")

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
    lines.append("### NDCG @K (binary relevance)")
    lines += _table_metric("ndcg")
    lines.append("")

    lines.append("### Rank-aware (K-independent)")
    lines.append("")
    lines.append("| arm | n | R-Precision | MRR |")
    lines.append("|---|---:|---:|---:|")
    for arm in ARMS:
        s = per_arm_summary[arm]
        lines.append(f"| {arm} | {s['n']} | {_fmt(s['r_precision'])} | {_fmt(s['mrr'])} |")
    lines.append("")

    lines.append("### Latency")
    lines.append("")
    lines.append("| arm | avg elapsed (s) |")
    lines.append("|---|---:|")
    for arm in ARMS:
        lines.append(f"| {arm} | {_fmt(per_arm_summary[arm]['avg_elapsed_s'])} |")
    lines.append("")

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
        for metric_name in ("recall", "precision", "f1", "ndcg"):
            lines.append(f"_{metric_name.capitalize()}@K — {cat}_")
            lines.append("")
            lines.append("| arm | n | " + " | ".join(f"@{kk}" for kk in field_ks) + " |")
            lines.append("|---|---:|" + "|".join(["---:"] * len(field_ks)) + "|")
            for arm in ARMS:
                s = per_arm_strat[arm][cat]
                if not s.get("n"):
                    continue
                vals = " | ".join(_fmt(s[f"{metric_name}@{kk}"]) for kk in field_ks)
                lines.append(f"| {arm} | {s['n']} | {vals} |")
            lines.append("")
        lines.append(f"_Rank-aware — {cat}_")
        lines.append("")
        lines.append("| arm | n | R-Precision | MRR |")
        lines.append("|---|---:|---:|---:|")
        for arm in ARMS:
            s = per_arm_strat[arm][cat]
            if not s.get("n"):
                continue
            lines.append(
                f"| {arm} | {s['n']} | {_fmt(s.get('r_precision'))} | {_fmt(s.get('mrr'))} |"
            )
        lines.append("")

    # ------------------------------------------------------------------
    # GRACE-R vs exp 09 baseline — head-to-head on in_corpus
    # ------------------------------------------------------------------
    lines.append("## GRACE-R vs exp 09 baseline (in_corpus)")
    lines.append("")
    lines.append("Baseline numbers frozen from `experiments/09_hyde2_grounded/README.md` "
                 "(full 200, n=151 in_corpus).")
    lines.append("")
    ic = {arm: per_arm_strat[arm]["in_corpus"] for arm in ARMS}

    def _delta_abs(a, b):
        if a is None or b is None:
            return None
        return round(a - b, 4)

    def _delta_rel(a, b):
        if a is None or b is None or b == 0:
            return None
        return round((a - b) / b * 100, 2)

    lines.append("### R@12 in_corpus")
    lines.append("")
    lines.append("| arm | exp 09 baseline | exp 10 (GRACE-R) | Δ abs | Δ rel % |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm in ARMS:
        base = EXP09_BASELINE_IN_CORPUS[arm]["recall@12"]
        gracer = ic[arm].get("recall@12")
        lines.append(
            f"| {arm} | {_fmt(base)} | {_fmt(gracer)} | "
            f"{_fmt(_delta_abs(gracer, base))} | "
            f"{_fmt(_delta_rel(gracer, base))} |"
        )
    lines.append("")

    lines.append("### NDCG@12 in_corpus")
    lines.append("")
    lines.append("| arm | exp 09 baseline | exp 10 (GRACE-R) | Δ abs | Δ rel % |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm in ARMS:
        base = EXP09_BASELINE_IN_CORPUS[arm]["ndcg@12"]
        gracer = ic[arm].get("ndcg@12")
        lines.append(
            f"| {arm} | {_fmt(base)} | {_fmt(gracer)} | "
            f"{_fmt(_delta_abs(gracer, base))} | "
            f"{_fmt(_delta_rel(gracer, base))} |"
        )
    lines.append("")

    lines.append("### R-Precision in_corpus")
    lines.append("")
    lines.append("| arm | exp 09 baseline | exp 10 (GRACE-R) | Δ abs | Δ rel % |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm in ARMS:
        base = EXP09_BASELINE_IN_CORPUS[arm]["r_precision"]
        gracer = ic[arm].get("r_precision")
        lines.append(
            f"| {arm} | {_fmt(base)} | {_fmt(gracer)} | "
            f"{_fmt(_delta_abs(gracer, base))} | "
            f"{_fmt(_delta_rel(gracer, base))} |"
        )
    lines.append("")

    # Pre-committed predictions from experiment README
    lines.append("### Pre-committed predictions (from experiment README)")
    lines.append("")
    lines.append("| # | Prediction | Threshold | Result |")
    lines.append("|---|---|---|:-:|")

    h1 = ic["dense_hyde"].get("recall@12")
    h1_base = EXP09_BASELINE_IN_CORPUS["dense_hyde"]["recall@12"]
    h2 = ic["dense_hyde2"].get("recall@12")
    h2_base = EXP09_BASELINE_IN_CORPUS["dense_hyde2"]["recall@12"]
    if h1 is not None and h2 is not None:
        p1 = "PASS" if h1 >= h1_base else "FAIL"
        p2 = "PASS" if h1 >= h1_base + 0.01 else "FAIL"
        p3 = "PASS" if h2 >= h2_base + 0.02 else "FAIL"
        gap_old = max(h1_base - h2_base, 1e-9)
        gap_new = h1_base - h2
        p4_ratio = gap_new / gap_old
        p4 = "PASS" if p4_ratio < 0.5 else "FAIL"
        p5 = "PASS" if h2 >= h1 else "FAIL"
    else:
        p1 = p2 = p3 = p4 = p5 = "—"
        p4_ratio = None

    lines.append(f"| P1 | GRACE-R HyDE1 ≥ exp09 HyDE1 R@12 | abs Δ ≥ 0 | {p1} |")
    lines.append(f"| P2 | GRACE-R HyDE1 > exp09 HyDE1 R@12 | abs Δ ≥ +0.01 | {p2} |")
    lines.append(f"| P3 | GRACE-R HyDE2 > exp09 HyDE2 R@12 | abs Δ ≥ +0.02 | {p3} |")
    if p4_ratio is not None:
        lines.append(f"| P4 | GRACE-R HyDE2 closes >50% of HyDE2-vs-HyDE1 gap | ratio={p4_ratio:.3f} < 0.5 | {p4} |")
    else:
        lines.append(f"| P4 | GRACE-R HyDE2 closes >50% of HyDE2-vs-HyDE1 gap | ratio < 0.5 | — |")
    lines.append(f"| P5 | GRACE-R HyDE2 ≥ GRACE-R HyDE1 R@12 | abs Δ ≥ 0 | {p5} |")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- All three arms share encoder + index with exp 09; only the dense "
                 "query embedding differs (raw question / HyDE1-GRACE-R doc / "
                 "HyDE2-GRACE-R doc).")
    lines.append("- HyDE1 + HyDE2 cache dirs are shared with exp 08/09 "
                 "(`artifacts/hyde/`, `artifacts/hyde2/`) but the new prompt_sha gives "
                 "disjoint cache keys → exp 10 always starts cold for HyDE1+HyDE2.")
    lines.append("- The `dense` arm has no LLM call; its records should match exp 09's "
                 "`dense` records bit-for-bit (same retriever, same seed_k=0 for raw).")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--full", action="store_true",
                   help="No-op flag retained for parity with exp 08/09 metrics CLI.")
    args = p.parse_args()
    _ = args.full  # accepted but ignored — exp10 always scores all on-disk records.

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
        try:
            recs = load_arm_records(arm)
        except FileNotFoundError:
            print(f"      WARN: arm {arm!r} has 0 records (dir missing) — skipping",
                  file=sys.stderr)
            per_arm_rows[arm] = []
            per_arm_summary[arm] = aggregate_macro([])
            per_arm_strat[arm] = aggregate_stratified([])
            continue
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
        print(f"      {arm:<14} scored {len(rows)} questions")

    print("[3/4] Writing artifacts ...")
    csv_path = write_csv(per_arm_rows)
    json_path = write_json(per_arm_summary, per_arm_strat)
    print(f"      {csv_path}")
    print(f"      {json_path}")

    print("[4/4] Writing report ...")
    n_scored = max((s.get("n") or 0) for s in per_arm_summary.values()) if per_arm_summary else 0
    report_path = write_report(per_arm_summary, per_arm_strat, n_scored)
    print(f"      {report_path}")

    print()
    print("=== Overall macro (exp 10 GRACE-R) ===")
    print(f"  {'arm':<14}{'n':>5}{'R@12':>9}{'R@100':>9}{'P@12':>9}{'NDCG@12':>9}{'R-Prec':>9}{'MRR':>9}")
    for arm in ARMS:
        s = per_arm_summary[arm]
        print(
            f"  {arm:<14}{s['n']:>5}"
            f"{_fmt(s['recall@12']):>9}{_fmt(s['recall@100']):>9}"
            f"{_fmt(s['precision@12']):>9}{_fmt(s['ndcg@12']):>9}"
            f"{_fmt(s['r_precision']):>9}{_fmt(s['mrr']):>9}"
        )

    print()
    print("=== In-corpus head-to-head vs exp 09 baseline ===")
    print(f"  {'arm':<14}{'metric':<12}{'exp09':>9}{'exp10':>9}{'Δ abs':>9}")
    for arm in ARMS:
        for metric in ("recall@12", "ndcg@12", "r_precision"):
            base = EXP09_BASELINE_IN_CORPUS[arm].get(metric)
            ours = per_arm_strat[arm]["in_corpus"].get(metric)
            d = (round(ours - base, 4) if (ours is not None and base is not None) else None)
            print(f"  {arm:<14}{metric:<12}{_fmt(base):>9}{_fmt(ours):>9}{_fmt(d):>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
