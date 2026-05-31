"""Experiment 11 metrics — CypherWalkRetriever retrieval-only audit.

Reads per-arm records from ``scripts/exp11_run.py``:

    experiments/11_graphrag_cypher/results/<arm>/A<stt>.json

for arms ``dense_vanilla``, ``dense_then_expand``, ``cypher_walk``.

Computes, ARTICLE-LEVEL, for K ∈ {5, 12, 20, all} (plan §5.3):

- recall@K, precision@K, F1@K, NDCG@K
- R-Precision, MRR (K-independent)

Stratified by L41 presence in gold (plan §5.3): ``l41_only`` /
``mixed_l41_other`` / ``no_l41``.

Plus the new provenance diagnostics for ``cypher_walk`` (plan §5.3):
``cypher_used`` rate, mean ``n_cypher_new`` (overall + conditional on
``cypher_used``), ``fallback_used`` rate, mean ``cypher_attempts`` length.

And a pre-commitment check (plan §5.5) so the result can't be rationalised.

CLAUSE-LEVEL recall is NOT computed: the 200-question gold dataset carries
**0** khoản-level citations (``gold_items == gold_articles`` for all 200 —
verified against ``eval_core.gold``). Reporting a clause-level number would
require fabricating clause gold, so we report the article-level numbers and
state the limitation explicitly. See the report's "Honest limitations".

Outputs:
- metrics/gold_citations_normalized.json
- metrics/academic_metrics.json
- metrics/academic_metrics.csv
- report/retrieval_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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

from eval_core.gold import validate_gold_citations  # noqa: E402

EXP_DIR = _REPO / "experiments" / "11_graphrag_cypher"
RESULTS_DIR = EXP_DIR / "results"
METRICS_DIR = EXP_DIR / "metrics"
REPORT_DIR = EXP_DIR / "report"
QUESTIONS_PATH = _REPO / "data" / "eval" / "questions_200.json"
REGISTRY_PATH = _REPO / "data" / "legal_sources.yaml"
PILOT_50_PATH = EXP_DIR / "pilot_50_stt.json"

ARMS = ("dense_vanilla", "dense_then_expand", "cypher_walk")
KS: tuple[int | None, ...] = (5, 12, 20, None)
STRATA = ("l41_only", "mixed_l41_other", "no_l41", "empty_gold")


# ---------------------------------------------------------------------------
# Subset + strata
# ---------------------------------------------------------------------------


def _pilot_subset(force_full: bool = False) -> set[int] | None:
    if force_full or not PILOT_50_PATH.exists():
        return None
    payload = json.loads(PILOT_50_PATH.read_text(encoding="utf-8"))
    return set(int(s) for s in payload.get("stt_list") or [])


def l41_stratum(gold_articles: list[str]) -> str:
    if not gold_articles:
        return "empty_gold"
    laws = {a.split(".")[0] for a in gold_articles}
    if laws == {"L41_2024"}:
        return "l41_only"
    if "L41_2024" in laws:
        return "mixed_l41_other"
    return "no_l41"


# ---------------------------------------------------------------------------
# Metric primitives (identical definitions to exp08_metrics.py)
# ---------------------------------------------------------------------------


def _at_k(retrieved: list[str], k: int | None) -> list[str]:
    return retrieved if k is None else retrieved[:k]


def recall(rk: list[str], gold: set[str]) -> float | None:
    return None if not gold else len(gold & set(rk)) / len(gold)


def precision(rk: list[str], gold: set[str]) -> float | None:
    if not rk:
        return 0.0 if gold else None
    return len(gold & set(rk)) / len(rk)


def f1_score(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


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
    dcg = sum(1.0 / math.log2(i + 1)
              for i, aid in enumerate(retrieved[:k], start=1) if aid in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return 0.0 if idcg == 0 else dcg / idcg


def _macro(values) -> float | None:
    vs = [v for v in values if v is not None]
    return round(mean(vs), 4) if vs else None


# ---------------------------------------------------------------------------
# Load gold + records
# ---------------------------------------------------------------------------


def load_gold() -> tuple[dict[int, list[str]], dict]:
    ok, summary = validate_gold_citations(
        questions_path=QUESTIONS_PATH, registry_path=REGISTRY_PATH, out_dir=METRICS_DIR
    )
    if not ok:
        print(f"FAIL: gold validation; see {summary['errors_path']}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(Path(summary["normalized_path"]).read_text(encoding="utf-8"))
    gold_articles = {int(k): v.get("gold_articles") or [] for k, v in data["records"].items()}
    # Honesty probe: how many gold cites carry clause (.K) detail?
    n_clause_gold = sum(
        1 for v in data["records"].values()
        for it in (v.get("gold_items") or []) if ".K" in it
    )
    clause_note = {
        "granularity": data.get("granularity"),
        "n_gold_citations_with_clause_level": n_clause_gold,
        "clause_level_recall_measurable": n_clause_gold > 0,
    }
    return gold_articles, clause_note


def load_arm_records(arm: str, stt_subset: set[int] | None) -> dict[int, dict]:
    arm_dir = RESULTS_DIR / arm
    out: dict[int, dict] = {}
    if not arm_dir.is_dir():
        return out
    for p in sorted(arm_dir.glob("A*.json")):
        if p.name.endswith(".error.json"):
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        stt = int(rec["stt"])
        if stt_subset is not None and stt not in stt_subset:
            continue
        out[stt] = rec
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_arm(arm: str, records: dict[int, dict], gold_map: dict[int, list[str]]) -> list[dict]:
    rows: list[dict] = []
    for stt, rec in records.items():
        ro = rec.get("retrieval_only") or {}
        retrieved = list(ro.get("final_article_ids") or [])
        gold = set(gold_map.get(stt) or [])
        row: dict = {
            "stt": stt, "arm": arm, "n_gold": len(gold),
            "n_retrieved_all": len(retrieved),
            "stratum": l41_stratum(gold_map.get(stt) or []),
            "elapsed_s": ro.get("elapsed_s"),
            "provenance": ro.get("provenance") or {},
        }
        for k in KS:
            rk = _at_k(retrieved, k)
            r = recall(rk, gold)
            pr = precision(rk, gold)
            kk = "all" if k is None else str(k)
            row[f"recall@{kk}"] = r
            row[f"precision@{kk}"] = pr
            row[f"f1@{kk}"] = f1_score(pr, r)
            row[f"ndcg@{kk}"] = ndcg_at_k(retrieved, gold, k if k is not None else max(len(retrieved), 1))
            row[f"n_retrieved@{kk}"] = len(rk)
        row["r_precision"] = r_precision(retrieved, gold)
        row["mrr"] = reciprocal_rank(retrieved, gold)
        rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> dict:
    s: dict = {"n": len(rows)}
    for k in KS:
        kk = "all" if k is None else str(k)
        s[f"recall@{kk}"] = _macro([r[f"recall@{kk}"] for r in rows])
        s[f"precision@{kk}"] = _macro([r[f"precision@{kk}"] for r in rows])
        s[f"f1@{kk}"] = _macro([r[f"f1@{kk}"] for r in rows])
        s[f"ndcg@{kk}"] = _macro([r[f"ndcg@{kk}"] for r in rows])
        s[f"avg_n_retrieved@{kk}"] = (
            round(mean([r[f"n_retrieved@{kk}"] for r in rows]), 2) if rows else None
        )
    s["r_precision"] = _macro([r["r_precision"] for r in rows])
    s["mrr"] = _macro([r["mrr"] for r in rows])
    lat = [r["elapsed_s"] for r in rows if r["elapsed_s"] is not None]
    s["avg_elapsed_s"] = round(mean(lat), 3) if lat else None
    return s


def aggregate_stratified(rows: list[dict]) -> dict[str, dict]:
    return {
        cat: (aggregate([r for r in rows if r["stratum"] == cat])
              if any(r["stratum"] == cat for r in rows) else {"n": 0})
        for cat in STRATA
    }


def cypher_provenance(rows: list[dict]) -> dict:
    """Provenance diagnostics for cypher_walk (plan §5.3)."""
    n = len(rows)
    if not n:
        return {"n": 0}
    used = [r["provenance"].get("cypher_used") for r in rows]
    fbk = [r["provenance"].get("fallback_used") for r in rows]
    n_new = [r["provenance"].get("n_cypher_new", 0) for r in rows]
    n_fb = [r["provenance"].get("n_fallback_added", 0) for r in rows]
    n_att = [r["provenance"].get("n_cypher_attempts", 0) for r in rows]
    n_used = sum(1 for u in used if u)
    new_when_used = [v for u, v in zip(used, n_new) if u]
    return {
        "n": n,
        "cypher_used_rate": round(n_used / n, 4),
        "cypher_used_count": n_used,
        "fallback_used_rate": round(sum(1 for f in fbk if f) / n, 4),
        "mean_n_cypher_new_all": round(mean(n_new), 4),
        "mean_n_cypher_new_when_used": round(mean(new_when_used), 4) if new_when_used else None,
        "mean_n_fallback_added": round(mean(n_fb), 4),
        "mean_cypher_attempts": round(mean(n_att), 4),
    }


# ---------------------------------------------------------------------------
# Output
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
    fieldnames = ["arm", "stt", "stratum", "n_gold"]
    for kk in field_ks:
        fieldnames += [f"n_retrieved@{kk}", f"recall@{kk}", f"precision@{kk}", f"f1@{kk}", f"ndcg@{kk}"]
    fieldnames += ["r_precision", "mrr", "elapsed_s"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for arm, rows in per_arm_rows.items():
            for r in rows:
                row_out = {"arm": arm, "stt": r["stt"], "stratum": r["stratum"], "n_gold": r["n_gold"]}
                for kk in field_ks:
                    for m in ("n_retrieved", "recall", "precision", "f1", "ndcg"):
                        row_out[f"{m}@{kk}"] = r[f"{m}@{kk}"]
                row_out["r_precision"] = r["r_precision"]
                row_out["mrr"] = r["mrr"]
                row_out["elapsed_s"] = r["elapsed_s"]
                w.writerow(row_out)
    return out


def write_json(per_arm_summary, per_arm_strat, per_arm_prov, clause_note, n_scored) -> Path:
    out = METRICS_DIR / "academic_metrics.json"
    payload = {
        "experiment": "11_graphrag_cypher",
        "scope": "retrieval-only (article-level). No answer generation, no BERTScore, no citation parsing.",
        "arms": list(ARMS),
        "Ks": [("all" if k is None else k) for k in KS],
        "n_scored": n_scored,
        "clause_level_note": clause_note,
        "overall_macro": per_arm_summary,
        "stratified": per_arm_strat,
        "cypher_walk_provenance": per_arm_prov,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _delta(a, b):
    return None if (a is None or b is None) else round(a - b, 4)


def write_report(per_arm_summary, per_arm_strat, per_arm_prov, clause_note, n_scored) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "retrieval_report.md"
    field_ks = [("all" if k is None else str(k)) for k in KS]
    L: list[str] = []
    L.append("# Experiment 11 (REDO) — `CypherWalkRetriever` retrieval-only audit")
    L.append("")
    L.append(f"Dataset: {n_scored} BHXH questions. Metric granularity: **article**.")
    L.append("Arms: `dense_vanilla`, `dense_then_expand`, `cypher_walk`.")
    L.append("Retrieval-only: no answer generation, no BERTScore, no citation parsing.")
    L.append("Plan: [`docs/plans/exp11_cypher_walk_retriever.md`](../../../docs/plans/exp11_cypher_walk_retriever.md).")
    L.append("")

    L.append("## Honest limitations (read first)")
    L.append("")
    L.append(f"- **Clause-level recall is NOT reported.** The gold dataset carries "
             f"`{clause_note['n_gold_citations_with_clause_level']}` clause-level "
             f"(khoản) citations — gold is article-level only "
             f"(`gold_items == gold_articles` for all questions, granularity="
             f"`{clause_note['granularity']}`). A clause-level number would require "
             f"fabricating clause gold, so only article-level metrics appear below.")
    L.append("- All three arms reuse the **vanilla** `clause_vec` dense channel "
             "(not the v5 tuned index/reranker), so numbers are NOT comparable to "
             "exp 06/07/08 absolute values — only to each other within exp 11.")
    L.append("")

    def _metric_table(name: str) -> list[str]:
        rows = ["| arm | n | " + " | ".join(f"@{kk}" for kk in field_ks) + " |",
                "|---|---:|" + "|".join(["---:"] * len(field_ks)) + "|"]
        for arm in ARMS:
            s = per_arm_summary[arm]
            rows.append(f"| {arm} | {s['n']} | "
                        + " | ".join(_fmt(s.get(f'{name}@{kk}')) for kk in field_ks) + " |")
        return rows

    L.append(f"## Overall macro (n={n_scored})")
    L.append("")
    L.append("### Citation recall@K (article-level)")
    L += _metric_table("recall")
    L.append("")
    L.append("### Citation precision@K")
    L += _metric_table("precision")
    L.append("")
    L.append("### Citation F1@K")
    L += _metric_table("f1")
    L.append("")
    L.append("### NDCG@K (binary relevance)")
    L += _metric_table("ndcg")
    L.append("")
    L.append("### Rank-aware (K-independent) + latency")
    L.append("")
    L.append("| arm | n | R-Precision | MRR | avg elapsed (s) |")
    L.append("|---|---:|---:|---:|---:|")
    for arm in ARMS:
        s = per_arm_summary[arm]
        L.append(f"| {arm} | {s['n']} | {_fmt(s['r_precision'])} | {_fmt(s['mrr'])} | {_fmt(s['avg_elapsed_s'])} |")
    L.append("")

    L.append("## CypherWalk provenance — is the graph actually walked? (plan §5.3)")
    L.append("")
    prov = per_arm_prov.get("overall") or {"n": 0}
    if prov.get("n"):
        L.append("| quantity | value |")
        L.append("|---|---:|")
        L.append(f"| n (cypher_walk records) | {prov['n']} |")
        L.append(f"| **cypher_used rate** (≥1 NEW clause beyond seed) | **{_fmt(prov['cypher_used_rate'])}** ({prov['cypher_used_count']}/{prov['n']}) |")
        L.append(f"| mean n_cypher_new (all) | {_fmt(prov['mean_n_cypher_new_all'])} |")
        L.append(f"| **mean n_cypher_new (when cypher_used)** | **{_fmt(prov['mean_n_cypher_new_when_used'])}** |")
        L.append(f"| fallback_used rate | {_fmt(prov['fallback_used_rate'])} |")
        L.append(f"| mean n_fallback_added | {_fmt(prov['mean_n_fallback_added'])} |")
        L.append(f"| mean cypher_attempts | {_fmt(prov['mean_cypher_attempts'])} |")
    else:
        L.append("_(no cypher_walk records scored)_")
    L.append("")

    # Pre-commitment check (plan §5.5)
    L.append("## Pre-commitment check (plan §5.5) — stated before the run")
    L.append("")
    dv = per_arm_summary["dense_vanilla"]
    cw = per_arm_summary["cypher_walk"]
    no_l41_dv = per_arm_strat["dense_vanilla"].get("no_l41", {})
    no_l41_cw = per_arm_strat["cypher_walk"].get("no_l41", {})
    r12_lift = _delta(cw.get("recall@12"), dv.get("recall@12"))
    r12_lift_nol41 = _delta(no_l41_cw.get("recall@12"), no_l41_dv.get("recall@12"))
    used_rate = prov.get("cypher_used_rate")
    new_when_used = prov.get("mean_n_cypher_new_when_used")
    L.append("| prediction | threshold | observed | within prediction? |")
    L.append("|---|---|---:|:-:|")
    L.append(f"| cypher_used rate | ≥ 0.30 | {_fmt(used_rate)} | "
             f"{'✓' if (used_rate is not None and used_rate >= 0.30) else '✗ AUDIT'} |")
    L.append(f"| mean n_cypher_new (when used) | 1.5–3.0 | {_fmt(new_when_used)} | "
             f"{'✓' if (new_when_used is not None and 1.5 <= new_when_used <= 3.0) else '✗ AUDIT'} |")
    L.append(f"| recall@12 lift vs dense_vanilla (all strata) | +0.00 to +0.05 | {_fmt(r12_lift)} | "
             f"{'✓' if (r12_lift is not None and 0.0 <= r12_lift <= 0.05) else '✗ AUDIT'} |")
    L.append(f"| recall@12 lift on no_l41 stratum | near 0 (≤ +0.05) | {_fmt(r12_lift_nol41)} | "
             f"{'✓' if (r12_lift_nol41 is not None and r12_lift_nol41 <= 0.05) else '✗ AUDIT'} |")
    L.append("")
    L.append("> If recall@12 lift > +0.05 across all strata, the plan says treat as "
             "**suspicious and audit before celebrating** — do not rationalise.")
    L.append("")

    # Stratified recall@K
    L.append("## Stratified by L41 presence in gold")
    L.append("")
    for cat in STRATA:
        present = max((per_arm_strat[arm][cat].get("n", 0) for arm in ARMS), default=0)
        if not present:
            continue
        L.append(f"### {cat}")
        L.append("")
        L.append("_recall@K_")
        L.append("")
        L.append("| arm | n | " + " | ".join(f"@{kk}" for kk in field_ks) + " |")
        L.append("|---|---:|" + "|".join(["---:"] * len(field_ks)) + "|")
        for arm in ARMS:
            s = per_arm_strat[arm][cat]
            if not s.get("n"):
                continue
            L.append(f"| {arm} | {s['n']} | "
                     + " | ".join(_fmt(s.get(f'recall@{kk}')) for kk in field_ks) + " |")
        L.append("")
    L.append("## Notes")
    L.append("")
    L.append("- Recall denominator = |gold articles|; questions with empty gold skipped.")
    L.append("- Precision denominator = |retrieved@K|; empty retrieved set → precision = 0.")
    L.append("- `dense_then_expand` final set = vector-hit articles ∪ REFERENCES/CITES_EXTERNAL ref-target articles.")
    L.append("- For the per-stage cypher_walk funnel, run `python -m scripts.exp11_funnel`.")
    L.append("")
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--full", action="store_true",
                   help="Score every record on disk, ignoring pilot_50_stt.json.")
    args = p.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/4] Validating gold citations ...")
    gold_map, clause_note = load_gold()
    print(f"      OK — {len(gold_map)} questions; clause-level gold cites = "
          f"{clause_note['n_gold_citations_with_clause_level']} "
          f"(clause-level recall measurable: {clause_note['clause_level_recall_measurable']})")

    stt_subset = _pilot_subset(force_full=args.full)
    if stt_subset is not None:
        print(f"      Pilot subset detected ({PILOT_50_PATH.name}): scoring n={len(stt_subset)}")

    print("[2/4] Loading per-arm records + scoring ...")
    per_arm_rows: dict[str, list[dict]] = {}
    per_arm_summary: dict[str, dict] = {}
    per_arm_strat: dict[str, dict] = {}
    for arm in ARMS:
        recs = load_arm_records(arm, stt_subset)
        if not recs:
            print(f"      WARN: arm {arm!r} has 0 records — skipping", file=sys.stderr)
            per_arm_rows[arm] = []
            per_arm_summary[arm] = aggregate([])
            per_arm_strat[arm] = aggregate_stratified([])
            continue
        rows = score_arm(arm, recs, gold_map)
        per_arm_rows[arm] = rows
        per_arm_summary[arm] = aggregate(rows)
        per_arm_strat[arm] = aggregate_stratified(rows)
        print(f"      {arm:<18} scored {len(rows)} questions")

    cw_rows = per_arm_rows.get("cypher_walk") or []
    per_arm_prov = {
        "overall": cypher_provenance(cw_rows),
        "stratified": {cat: cypher_provenance([r for r in cw_rows if r["stratum"] == cat])
                       for cat in STRATA},
    }

    print("[3/4] Writing artifacts ...")
    n_scored = max((s.get("n") or 0) for s in per_arm_summary.values()) if per_arm_summary else 0
    csv_path = write_csv(per_arm_rows)
    json_path = write_json(per_arm_summary, per_arm_strat, per_arm_prov, clause_note, n_scored)
    print(f"      {csv_path}\n      {json_path}")

    print("[4/4] Writing report ...")
    report_path = write_report(per_arm_summary, per_arm_strat, per_arm_prov, clause_note, n_scored)
    print(f"      {report_path}")

    print()
    print("=== Overall macro (article-level) ===")
    print(f"  {'arm':<20}{'n':>5}{'R@5':>9}{'R@12':>9}{'R@20':>9}{'NDCG@12':>9}{'R-Prec':>9}{'MRR':>9}")
    for arm in ARMS:
        s = per_arm_summary[arm]
        print(f"  {arm:<20}{s['n']:>5}{_fmt(s['recall@5']):>9}{_fmt(s['recall@12']):>9}"
              f"{_fmt(s['recall@20']):>9}{_fmt(s['ndcg@12']):>9}{_fmt(s['r_precision']):>9}{_fmt(s['mrr']):>9}")
    cw_prov = per_arm_prov["overall"]
    if cw_prov.get("n"):
        print(f"\n=== cypher_walk provenance ===")
        print(f"  cypher_used_rate={_fmt(cw_prov['cypher_used_rate'])}  "
              f"mean_n_cypher_new(when_used)={_fmt(cw_prov['mean_n_cypher_new_when_used'])}  "
              f"fallback_used_rate={_fmt(cw_prov['fallback_used_rate'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
