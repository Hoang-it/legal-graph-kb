"""Experiment 12 metrics — HyDE × CypherWalk 2×2 factorial (retrieval-only).

Reads experiments/12_hyde_cypher_walk/results/<arm>/A<stt>.json for arms
dense_vanilla, dense_hyde, cypher_walk, cypher_walk_hyde.

Article-level recall/precision/F1/NDCG@{5,12,20,all} + R-Prec + MRR, L41
strata, cypher provenance — reusing the exp 11 metric engine verbatim — plus:

- the 2×2 interaction table (HyDE effect on seed / on walk; walk effect on
  each seed; combo vs best single arm),
- a per-stage gold-hit funnel for BOTH cypher arms (does the HyDE seed change
  whether the walk surfaces gold?),
- a pre-commitment check (stated in the README before the run).

CLAUSE-LEVEL recall is NOT reported (0/200 gold cites carry khoản — same as
exp 11). Outputs: metrics/{academic_metrics.json,csv}, report/retrieval_report.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

from eval_core.gold import validate_gold_citations  # noqa: E402
# Reuse the exp 11 metric engine — identical definitions, no drift.
from scripts.exp11_metrics import (  # noqa: E402
    KS,
    STRATA,
    _fmt,
    aggregate,
    aggregate_stratified,
    cypher_provenance,
    l41_stratum,
    recall,
    score_arm,
)

EXP_DIR = _REPO / "experiments" / "12_hyde_cypher_walk"
RESULTS_DIR = EXP_DIR / "results"
METRICS_DIR = EXP_DIR / "metrics"
REPORT_DIR = EXP_DIR / "report"
QUESTIONS_PATH = _REPO / "data" / "eval" / "questions_200.json"
REGISTRY_PATH = _REPO / "data" / "legal_sources.yaml"
PILOT_50_PATH = EXP_DIR / "pilot_50_stt.json"

ARMS = ("dense_vanilla", "dense_hyde", "cypher_walk", "cypher_walk_hyde")
CYPHER_ARMS = ("cypher_walk", "cypher_walk_hyde")
FIELD_KS = [("all" if k is None else str(k)) for k in KS]


def _pilot_subset(force_full: bool = False) -> set[int] | None:
    if force_full or not PILOT_50_PATH.exists():
        return None
    payload = json.loads(PILOT_50_PATH.read_text(encoding="utf-8"))
    return set(int(s) for s in payload.get("stt_list") or [])


def load_gold():
    ok, summary = validate_gold_citations(
        questions_path=QUESTIONS_PATH, registry_path=REGISTRY_PATH, out_dir=METRICS_DIR
    )
    if not ok:
        print(f"FAIL: gold validation; see {summary['errors_path']}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(Path(summary["normalized_path"]).read_text(encoding="utf-8"))
    gold_articles = {int(k): v.get("gold_articles") or [] for k, v in data["records"].items()}
    n_clause_gold = sum(1 for v in data["records"].values()
                        for it in (v.get("gold_items") or []) if ".K" in it)
    return gold_articles, {"granularity": data.get("granularity"),
                           "n_gold_citations_with_clause_level": n_clause_gold}


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


def _dedupe(seq):
    return list(dict.fromkeys(x for x in seq if x))


def funnel_rows(records: dict[int, dict], gold_map: dict[int, list[str]]) -> dict:
    """Per-stage gold-hit aggregate for a cypher arm: seed → +cypher_new → final."""
    seed_R = cyp_R = final_R = 0.0
    seed_g = cyp_g = final_g = 0
    n = 0
    for stt, rec in records.items():
        gold = set(gold_map.get(stt) or [])
        if not gold:
            continue
        n += 1
        ro = rec.get("retrieval_only") or {}
        seed = list(ro.get("seed_article_ids") or [])
        cyp = _dedupe(seed + list(ro.get("cypher_new_article_ids") or []))
        final = list(ro.get("final_article_ids") or [])
        seed_R += recall(seed, gold) or 0.0
        cyp_R += recall(cyp, gold) or 0.0
        final_R += recall(final, gold) or 0.0
        seed_g += len(set(seed) & gold)
        cyp_g += len(set(cyp) & gold)
        final_g += len(set(final) & gold)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "seed": {"recall@all": round(seed_R / n, 4), "gold_hits": seed_g},
        "+cypher_new": {"recall@all": round(cyp_R / n, 4), "gold_hits": cyp_g},
        "final": {"recall@all": round(final_R / n, 4), "gold_hits": final_g},
    }


def _delta(a, b):
    return None if (a is None or b is None) else round(a - b, 4)


def write_csv(per_arm_rows: dict[str, list[dict]]) -> Path:
    out = METRICS_DIR / "academic_metrics.csv"
    fieldnames = ["arm", "stt", "stratum", "n_gold"]
    for kk in FIELD_KS:
        fieldnames += [f"n_retrieved@{kk}", f"recall@{kk}", f"precision@{kk}", f"f1@{kk}", f"ndcg@{kk}"]
    fieldnames += ["r_precision", "mrr", "elapsed_s"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for arm, rows in per_arm_rows.items():
            for r in rows:
                row = {"arm": arm, "stt": r["stt"], "stratum": r["stratum"], "n_gold": r["n_gold"]}
                for kk in FIELD_KS:
                    for m in ("n_retrieved", "recall", "precision", "f1", "ndcg"):
                        row[f"{m}@{kk}"] = r[f"{m}@{kk}"]
                row["r_precision"] = r["r_precision"]
                row["mrr"] = r["mrr"]
                row["elapsed_s"] = r["elapsed_s"]
                w.writerow(row)
    return out


def write_json(summary, strat, prov, funnels, clause_note, n) -> Path:
    out = METRICS_DIR / "academic_metrics.json"
    out.write_text(json.dumps({
        "experiment": "12_hyde_cypher_walk",
        "scope": "retrieval-only (article-level), 2x2 HyDE x CypherWalk on vanilla clause_vec stack",
        "arms": list(ARMS), "Ks": [("all" if k is None else k) for k in KS], "n_scored": n,
        "clause_level_note": clause_note, "overall_macro": summary, "stratified": strat,
        "cypher_provenance": prov, "funnels": funnels,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_report(summary, strat, prov, funnels, clause_note, n) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "retrieval_report.md"
    L: list[str] = []
    L.append("# Experiment 12 — HyDE × CypherWalk 2×2 (retrieval-only, article-level)")
    L.append("")
    L.append(f"Dataset: {n} BHXH questions (same pilot subset as exp 11).")
    L.append("All arms on the **vanilla `clause_vec`** stack (not v5 tuned) so the HyDE")
    L.append("effect is isolated; numbers are NOT comparable to exp 08 absolute values.")
    L.append("Plan motivation: does a HyDE seed change the Cypher walk's effect on retrieval?")
    L.append("")
    L.append(f"- **Clause-level recall NOT reported**: {clause_note['n_gold_citations_with_clause_level']} "
             f"clause-level gold cites (article-level only).")
    L.append("")

    def _tbl(name):
        rows = ["| arm | n | " + " | ".join(f"@{kk}" for kk in FIELD_KS) + " |",
                "|---|---:|" + "|".join(["---:"] * len(FIELD_KS)) + "|"]
        for arm in ARMS:
            s = summary[arm]
            rows.append(f"| {arm} | {s['n']} | "
                        + " | ".join(_fmt(s.get(f'{name}@{kk}')) for kk in FIELD_KS) + " |")
        return rows

    L.append(f"## Overall macro (n={n})")
    L.append("")
    L.append("### Citation recall@K (article-level)")
    L += _tbl("recall")
    L.append("")
    L.append("### NDCG@K")
    L += _tbl("ndcg")
    L.append("")
    L.append("### Rank-aware + latency")
    L.append("")
    L.append("| arm | n | R-Prec | MRR | avg elapsed (s) |")
    L.append("|---|---:|---:|---:|---:|")
    for arm in ARMS:
        s = summary[arm]
        L.append(f"| {arm} | {s['n']} | {_fmt(s['r_precision'])} | {_fmt(s['mrr'])} | {_fmt(s['avg_elapsed_s'])} |")
    L.append("")

    # 2x2 interaction
    dv, dh = summary["dense_vanilla"], summary["dense_hyde"]
    cw, cwh = summary["cypher_walk"], summary["cypher_walk_hyde"]
    r = lambda a: a.get("recall@12")  # noqa: E731
    best_single = max((x for x in (r(dv), r(dh), r(cw)) if x is not None), default=None)
    L.append("## 2×2 interaction — recall@12 (the headline)")
    L.append("")
    L.append("|  | no walk | + cypher walk | walk effect (Δ) |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| **raw seed** | {_fmt(r(dv))} | {_fmt(r(cw))} | {_fmt(_delta(r(cw), r(dv)))} |")
    L.append(f"| **HyDE seed** | {_fmt(r(dh))} | {_fmt(r(cwh))} | {_fmt(_delta(r(cwh), r(dh)))} |")
    L.append(f"| **HyDE effect (Δ)** | {_fmt(_delta(r(dh), r(dv)))} | {_fmt(_delta(r(cwh), r(cw)))} | |")
    L.append("")
    L.append(f"- Best single arm (excl. combo) recall@12 = **{_fmt(best_single)}**; "
             f"combo `cypher_walk_hyde` = **{_fmt(r(cwh))}** "
             f"(Δ vs best = {_fmt(_delta(r(cwh), best_single))}).")
    L.append("")

    # Pre-commitment
    L.append("## Pre-commitment check (stated in README before the run)")
    L.append("")
    d_walk_hyde = _delta(r(cwh), r(dh))
    d_combo_best = _delta(r(cwh), best_single)
    L.append("| prediction | threshold | observed | verdict |")
    L.append("|---|---|---:|:-:|")
    L.append(f"| walk hurts even on HyDE seed (cypher_walk_hyde − dense_hyde) | ≤ +0.02 | "
             f"{_fmt(d_walk_hyde)} | {'✓' if (d_walk_hyde is not None and d_walk_hyde <= 0.02) else '✗ AUDIT'} |")
    L.append(f"| combo does not beat best single arm | ≤ +0.05 | {_fmt(d_combo_best)} | "
             f"{'✓' if (d_combo_best is not None and d_combo_best <= 0.05) else '✗ AUDIT'} |")
    for arm in CYPHER_ARMS:
        fn = funnels.get(arm, {})
        if fn.get("n"):
            seedg, cypg = fn["seed"]["gold_hits"], fn["+cypher_new"]["gold_hits"]
            L.append(f"| {arm}: +cypher_new adds ~0 gold (Σ gold seed→+cypher) | ≈ equal | "
                     f"{seedg}→{cypg} | {'✓' if cypg - seedg <= 1 else '✗ AUDIT'} |")
    L.append("")

    # Funnels
    L.append("## Per-stage gold-hit funnel — both cypher arms")
    L.append("")
    for arm in CYPHER_ARMS:
        fn = funnels.get(arm, {})
        if not fn.get("n"):
            continue
        L.append(f"### {arm} (n={fn['n']})")
        L.append("")
        L.append("| stage | recall@all | gold-hits (Σ) |")
        L.append("|---|---:|---:|")
        for stage in ("seed", "+cypher_new", "final"):
            s = fn[stage]
            L.append(f"| {stage} | {_fmt(s['recall@all'])} | {s['gold_hits']} |")
        L.append("")

    # provenance
    L.append("## CypherWalk provenance")
    L.append("")
    L.append("| arm | cypher_used | mean n_cypher_new (when used) | fallback_used |")
    L.append("|---|---:|---:|---:|")
    for arm in CYPHER_ARMS:
        pv = prov.get(arm, {})
        if pv.get("n"):
            L.append(f"| {arm} | {_fmt(pv['cypher_used_rate'])} | "
                     f"{_fmt(pv['mean_n_cypher_new_when_used'])} | {_fmt(pv['fallback_used_rate'])} |")
    L.append("")

    # strata recall@12
    L.append("## Stratified recall@12 by L41 presence")
    L.append("")
    L.append("| arm | l41_only | mixed_l41_other | no_l41 |")
    L.append("|---|---:|---:|---:|")
    for arm in ARMS:
        cells = []
        for cat in ("l41_only", "mixed_l41_other", "no_l41"):
            s = strat[arm].get(cat, {})
            cells.append(_fmt(s.get("recall@12")) if s.get("n") else "—")
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L.append("")
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] Gold ...")
    gold_map, clause_note = load_gold()
    print(f"      clause-level gold cites = {clause_note['n_gold_citations_with_clause_level']}")
    stt_subset = _pilot_subset(force_full=args.full)
    if stt_subset is not None:
        print(f"      pilot subset: n={len(stt_subset)}")

    print("[2/3] Scoring ...")
    per_arm_rows, summary, strat, prov, funnels = {}, {}, {}, {}, {}
    for arm in ARMS:
        recs = load_arm_records(arm, stt_subset)
        if not recs:
            print(f"      WARN: arm {arm!r} has 0 records", file=sys.stderr)
            per_arm_rows[arm] = []
            summary[arm] = aggregate([])
            strat[arm] = aggregate_stratified([])
            continue
        rows = score_arm(arm, recs, gold_map)
        per_arm_rows[arm] = rows
        summary[arm] = aggregate(rows)
        strat[arm] = aggregate_stratified(rows)
        if arm in CYPHER_ARMS:
            prov[arm] = cypher_provenance(rows)
            funnels[arm] = funnel_rows(recs, gold_map)
        print(f"      {arm:<18} scored {len(rows)}")

    print("[3/3] Writing ...")
    n = max((s.get("n") or 0) for s in summary.values()) if summary else 0
    write_csv(per_arm_rows)
    write_json(summary, strat, prov, funnels, clause_note, n)
    report = write_report(summary, strat, prov, funnels, clause_note, n)
    print(f"      {report}")

    print()
    print("=== recall@12 (article-level) ===")
    for arm in ARMS:
        print(f"  {arm:<20} R@12={_fmt(summary[arm].get('recall@12'))}  "
              f"R@5={_fmt(summary[arm].get('recall@5'))}  MRR={_fmt(summary[arm].get('mrr'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
