"""Experiment 09 metrics — HyDE2 retrieval-only 3-arm metrics.

Reads per-arm records produced by ``scripts/exp09_run.py``:

    experiments/09_hyde2_grounded/results/<arm>/A<stt>.json

for arms ``dense``, ``dense_hyde``, ``dense_hyde2``.

Computes, for K ∈ {12, 20, 30, 50, 70, 100, all}:

- recall@K, precision@K, F1@K
- NDCG@K (binary relevance, log2 discount)

Plus K-independent rank-aware metrics:

- R-Precision (= precision at K = |gold| per question)
- MRR (Mean Reciprocal Rank of first gold article)
- NDCG@all (over the full retrieved list)

Re-derives ``gold_citations_normalized.json`` via
``eval_core.gold.validate_gold_citations`` so the experiment is
self-contained (matches exp 07/08 shape exactly).

Outputs:
- ``metrics/gold_citations_normalized.json``
- ``metrics/academic_metrics.json``
- ``metrics/academic_metrics.csv``
- ``report/academic_report.md``

Pilot subset (exp 08's ``pilot_50_stt.json``, seed=0) is auto-detected
when present so pilot runs aren't aggregated with stale full-200
records. Pass ``--full`` to override and score every record on disk.
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

EXP_DIR = _REPO / "experiments" / "09_hyde2_grounded"
EXP08_DIR = _REPO / "experiments" / "08_hyde_retrieval"
RESULTS_DIR = EXP_DIR / "results"
METRICS_DIR = EXP_DIR / "metrics"
REPORT_DIR = EXP_DIR / "report"
QUESTIONS_PATH = _REPO / "data" / "eval" / "questions_200.json"
REGISTRY_PATH = _REPO / "data" / "legal_sources.yaml"
# Pilot list is reused from exp 08 for identical strata.
PILOT_50_PATH = EXP08_DIR / "pilot_50_stt.json"


def _pilot_subset(force_full: bool = False) -> set[int] | None:
    """Return exp 08's pilot stt set, or None for full dataset.

    Pass ``force_full=True`` (via ``--full``) to score every record on
    disk regardless of the pilot list.
    """
    if force_full:
        return None
    if not PILOT_50_PATH.exists():
        return None
    payload = json.loads(PILOT_50_PATH.read_text(encoding="utf-8"))
    return set(int(s) for s in payload.get("stt_list") or [])


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


def load_arm_records(arm: str, stt_subset: set[int] | None = None) -> dict[int, dict]:
    arm_dir = RESULTS_DIR / arm
    if not arm_dir.is_dir():
        raise FileNotFoundError(arm_dir)
    out: dict[int, dict] = {}
    for p in sorted(arm_dir.glob("A*.json")):
        if p.name.endswith(".error.json"):
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        stt = int(rec["stt"])
        if stt_subset is not None and stt not in stt_subset:
            continue
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
        "experiment": "09_hyde2_grounded",
        "arms": list(ARMS),
        "Ks": [("all" if k is None else k) for k in KS],
        "overall_macro": per_arm_summary,
        "stratified": per_arm_strat,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_report(per_arm_summary: dict, per_arm_strat: dict, n_scored: int) -> Path:
    out = REPORT_DIR / "academic_report.md"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    field_ks = [("all" if k is None else str(k)) for k in KS]

    lines: list[str] = []
    lines.append("# Experiment 09 — HyDE2 retrieval-only 3-arm metrics (K ∈ {12..100, all})")
    lines.append("")
    lines.append(f"Dataset: {n_scored} BHXH questions. Metric granularity: article.")
    lines.append("Arms: `dense`, `dense_hyde` (HyDE1), `dense_hyde2` (grounded).")
    lines.append("Generator: OpenAI `gpt-4o-mini` (n=1, max_tokens=700, temperature=0.0).")
    lines.append("HyDE2 seed_k = 5 (pass-1 dense top-5 → grounded LLM → pass-2 dense top-100).")
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
    lines.append("### Average retrieved-set size at K")
    lines += _table_metric("avg_n_retrieved")
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
    lines.append("Note: `dense_hyde` / `dense_hyde2` latency excludes LLM call time "
                 "when the doc was already cached (prewarm or prior run). HyDE2 also "
                 "includes a pass-1 dense seed retrieval (~50ms) on top of HyDE1's "
                 "single dense call.")
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
    # Success criteria — HyDE2 vs HyDE1 + sanity vs dense (plan §6)
    # ------------------------------------------------------------------
    lines.append("## HyDE2 success criteria (plan §6)")
    lines.append("")
    lines.append("Comparison **vs `dense_hyde`** (HyDE1) on the in_corpus stratum.")
    lines.append("HyDE2 wins iff sanity checks S1+S2 hold AND ≥1 of criteria 1/2/3 passes.")
    lines.append("")
    ic = {arm: per_arm_strat[arm]["in_corpus"] for arm in ARMS}

    def _delta_abs(a, b):
        if a is None or b is None:
            return None
        return round(a - b, 4)

    def _delta_rel(a, b):
        if a is None or b is None or b == 0:
            return None
        return round((a - b) / b, 4)

    crit1 = _delta_abs(ic["dense_hyde2"].get("recall@12"), ic["dense_hyde"].get("recall@12"))
    crit2 = _delta_rel(ic["dense_hyde2"].get("ndcg@12"), ic["dense_hyde"].get("ndcg@12"))
    crit3 = _delta_rel(ic["dense_hyde2"].get("r_precision"), ic["dense_hyde"].get("r_precision"))
    s1 = _delta_abs(ic["dense_hyde2"].get("recall@12"), ic["dense_hyde"].get("recall@12"))
    s2 = _delta_abs(ic["dense_hyde2"].get("recall@12"), ic["dense"].get("recall@12"))

    lines.append("| # | metric | dense_hyde | dense_hyde2 | Δ | threshold | passes? |")
    lines.append("|---|---|---:|---:|---:|---:|:-:|")
    lines.append(
        f"| 1 | R@12 in_corpus (abs Δ) | "
        f"{_fmt(ic['dense_hyde'].get('recall@12'))} | "
        f"{_fmt(ic['dense_hyde2'].get('recall@12'))} | "
        f"{_fmt(crit1)} | +0.030 | "
        f"{'PASS' if (crit1 is not None and crit1 >= 0.030) else '—'} |"
    )
    lines.append(
        f"| 2 | NDCG@12 in_corpus (rel Δ) | "
        f"{_fmt(ic['dense_hyde'].get('ndcg@12'))} | "
        f"{_fmt(ic['dense_hyde2'].get('ndcg@12'))} | "
        f"{_fmt(crit2)} | +0.050 | "
        f"{'PASS' if (crit2 is not None and crit2 >= 0.050) else '—'} |"
    )
    lines.append(
        f"| 3 | R-Precision in_corpus (rel Δ) | "
        f"{_fmt(ic['dense_hyde'].get('r_precision'))} | "
        f"{_fmt(ic['dense_hyde2'].get('r_precision'))} | "
        f"{_fmt(crit3)} | +0.150 | "
        f"{'PASS' if (crit3 is not None and crit3 >= 0.150) else '—'} |"
    )
    lines.append("")
    lines.append("**Sanity checks**:")
    lines.append("")
    lines.append("| # | check | value | passes? |")
    lines.append("|---|---|---:|:-:|")
    lines.append(
        f"| S1 | dense_hyde2 R@12 − dense_hyde R@12 ≥ 0 (no regression) | "
        f"{_fmt(s1)} | {'PASS' if (s1 is not None and s1 >= 0) else 'FAIL'} |"
    )
    lines.append(
        f"| S2 | dense_hyde2 R@12 − dense R@12 ≥ 0.030 (still beats baseline) | "
        f"{_fmt(s2)} | {'PASS' if (s2 is not None and s2 >= 0.030) else 'FAIL'} |"
    )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- HyDE1 records are produced fresh in this experiment (skip-when-on-disk).")
    lines.append("  HyDE1 LLM calls hit the shared HyDE cache at "
                 "`artifacts/hyde/openai__gpt-4o-mini/` populated by exp 08, so the "
                 "incremental LLM cost for HyDE1 here is $0 if exp 08 already ran.")
    lines.append("- HyDE2 has its own cache at "
                 "`artifacts/hyde2/openai__gpt-4o-mini/` keyed by seed_clause_ids; the "
                 "cache invalidates automatically when the LoRA model or the dense "
                 "index changes (because seed sets shift).")
    lines.append("- All three arms share encoder + index; only the dense query "
                 "embedding differs (raw question / HyDE1 doc / HyDE2 doc).")
    lines.append("- This is a retrieval-only diagnostic at article granularity. "
                 "The thesis-defining E2E metric (academic_v2 strict tuple) lives "
                 "in item B of `docs/plans/exp08_followups_and_strict_metric.md`.")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--full", action="store_true",
                   help="Score every record on disk, ignoring exp 08's "
                        "pilot_50_stt.json (use after the full 200 run).")
    args = p.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Validating gold citations ...")
    gold_map = load_gold()
    print(f"      OK — {len(gold_map)} questions normalized")

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    q_by_stt = {q["stt"]: q for q in questions}
    in_corpus_codes = {m.full_id for m in load_law_metadata().values()}

    stt_subset = _pilot_subset(force_full=args.full)
    if stt_subset is not None:
        print(f"      Pilot subset detected ({PILOT_50_PATH.name}, reused from exp 08): "
              f"scoring n={len(stt_subset)} questions")
    elif args.full and PILOT_50_PATH.exists():
        print(f"      --full set — ignoring {PILOT_50_PATH.name}; "
              f"scoring every record on disk.")

    print("[2/4] Loading per-arm records + scoring ...")
    per_arm_rows: dict[str, list[dict]] = {}
    per_arm_summary: dict[str, dict] = {}
    per_arm_strat: dict[str, dict] = {}
    for arm in ARMS:
        try:
            recs = load_arm_records(arm, stt_subset=stt_subset)
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
    print("=== Overall macro ===")
    print(f"  {'arm':<14}{'n':>5}{'R@12':>9}{'R@100':>9}{'P@12':>9}{'NDCG@12':>9}{'R-Prec':>9}{'MRR':>9}")
    for arm in ARMS:
        s = per_arm_summary[arm]
        print(
            f"  {arm:<14}{s['n']:>5}"
            f"{_fmt(s['recall@12']):>9}{_fmt(s['recall@100']):>9}"
            f"{_fmt(s['precision@12']):>9}{_fmt(s['ndcg@12']):>9}"
            f"{_fmt(s['r_precision']):>9}{_fmt(s['mrr']):>9}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
