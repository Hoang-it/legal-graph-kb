"""Experiment 14 metrics — FAIR-PROMPT re-test, in-corpus headline.

Reads ``experiments/14_hyde_fair_prompts/results/<arm>/A<stt>.json`` for arms
``dense``, ``dense_hyde`` (HyDE1 bar), ``dense_hyde2`` (grounded, PARITY
prompt), ``dense_hyde_semantic`` (semantic, PARITY prompt) and computes
article-level recall@{1,5,10,12,20,all}, precision@{1,5,10,12,20}, R-Precision,
MRR, NDCG@12, stratified by gold corpus type (``in_corpus`` is the headline).
The challenger arms use parity prompts that hold HyDE1's vocabulary scaffold
constant, so only the grounding block differs between them.

    python -m scripts.exp14_metrics            # pilot-50 (auto)
    python -m scripts.exp14_metrics --full
"""
from __future__ import annotations

import argparse
import csv
import json
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
from src.legal_metadata import load_law_metadata  # noqa: E402
from scripts.exp09_metrics import (  # noqa: E402  (pure metric primitives)
    _macro,
    _retrieved_at_k,
    categorize,
    ndcg_at_k,
    precision,
    r_precision,
    recall,
    reciprocal_rank,
)

EXP_DIR = _REPO / "experiments" / "14_hyde_fair_prompts"
RESULTS_DIR = EXP_DIR / "results"
METRICS_DIR = EXP_DIR / "metrics"
REPORT_DIR = EXP_DIR / "report"
QUESTIONS_PATH = _REPO / "data" / "eval" / "questions_200.json"
REGISTRY_PATH = _REPO / "data" / "legal_sources.yaml"
PILOT_50_PATH = _REPO / "experiments" / "08_hyde_retrieval" / "pilot_50_stt.json"

ARMS = ("dense", "dense_hyde", "dense_hyde2", "dense_hyde_semantic")
# (arm, human label) for the per-challenger report rows.
CHALLENGERS = (("dense_hyde2", "HyDE2-fair (grounded)"),
               ("dense_hyde_semantic", "semantic-fair (concept frame)"))
KS: tuple[int | None, ...] = (1, 5, 10, 12, 20, None)
STRATA = ("in_corpus", "mixed", "ooc", "unparseable")


def _pilot_subset(force_full: bool) -> set[int] | None:
    if force_full or not PILOT_50_PATH.exists():
        return None
    payload = json.loads(PILOT_50_PATH.read_text(encoding="utf-8"))
    return set(int(s) for s in payload.get("stt_list") or [])


def load_gold() -> dict[int, list[str]]:
    ok, summary = validate_gold_citations(
        questions_path=QUESTIONS_PATH, registry_path=REGISTRY_PATH, out_dir=METRICS_DIR
    )
    if not ok:
        print(f"FAIL: gold validation; see {summary['errors_path']}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(Path(summary["normalized_path"]).read_text(encoding="utf-8"))
    return {int(k): v.get("gold_articles") or [] for k, v in data["records"].items()}


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


def score_arm(arm, records, gold_map, q_by_stt, in_corpus_codes) -> list[dict]:
    rows: list[dict] = []
    for stt, rec in records.items():
        retrieved = list((rec.get("retrieval_only") or {}).get("final_article_ids") or [])
        gold = set(gold_map.get(stt) or [])
        row: dict = {
            "stt": stt, "arm": arm, "n_gold": len(gold),
            "category": categorize(q_by_stt[stt].get("gold_citations_raw"), in_corpus_codes),
            "elapsed_s": (rec.get("retrieval_only") or {}).get("elapsed_s"),
            "semantic_context": rec.get("semantic_context"),
        }
        for k in KS:
            rk = _retrieved_at_k(retrieved, k)
            kk = "all" if k is None else str(k)
            row[f"recall@{kk}"] = recall(rk, gold)
            row[f"precision@{kk}"] = precision(rk, gold)
            row[f"ndcg@{kk}"] = ndcg_at_k(retrieved, gold, k if k is not None else max(len(retrieved), 1))
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
        s[f"ndcg@{kk}"] = _macro([r[f"ndcg@{kk}"] for r in rows])
    s["r_precision"] = _macro([r["r_precision"] for r in rows])
    s["mrr"] = _macro([r["mrr"] for r in rows])
    lat = [r["elapsed_s"] for r in rows if r["elapsed_s"] is not None]
    s["avg_elapsed_s"] = round(mean(lat), 3) if lat else None
    return s


def aggregate_stratified(rows: list[dict]) -> dict[str, dict]:
    return {cat: (aggregate([r for r in rows if r["category"] == cat])
                  if any(r["category"] == cat for r in rows) else {"n": 0})
            for cat in STRATA}


def provenance(rows: list[dict]) -> dict:
    sc = [r["semantic_context"] for r in rows if r.get("semantic_context")]
    n = len(rows)
    if not n:
        return {"n": 0}
    matched = [c for c in sc if c.get("concept_match")]
    return {
        "n": n,
        "concept_match_rate": round(len(matched) / n, 4),
        "fallback_rate": round(1 - len(matched) / n, 4),
        "mean_n_concepts": round(mean([c.get("n_concepts", 0) for c in sc]), 3) if sc else None,
        "mean_n_kg_entities": round(mean([c.get("n_kg_entities", 0) for c in sc]), 3) if sc else None,
    }


def _fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _delta(a, b):
    return None if (a is None or b is None) else round(a - b, 4)


def write_outputs(summ, strat, prov, n_scored) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    field_ks = [("all" if k is None else str(k)) for k in KS]

    (METRICS_DIR / "academic_metrics.json").write_text(json.dumps({
        "experiment": "14_hyde_fair_prompts",
        "scope": "retrieval-only, article-level, tuned stack (clause_vec_tuned). "
                 "Parity prompts for grounded + semantic arms. Headline = in_corpus.",
        "arms": list(ARMS), "Ks": [("all" if k is None else k) for k in KS],
        "n_scored": n_scored, "overall_macro": summ, "stratified": strat,
        "semantic_provenance": prov,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    with (METRICS_DIR / "academic_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        cols = ["arm", "stratum", "n"] + [f"recall@{kk}" for kk in field_ks] + \
               [f"precision@{kk}" for kk in field_ks] + ["r_precision", "mrr", "ndcg@12"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for arm in ARMS:
            for stratum, s in [("overall", summ[arm])] + [(c, strat[arm][c]) for c in STRATA]:
                if not s.get("n"):
                    continue
                row = {"arm": arm, "stratum": stratum, "n": s["n"], "ndcg@12": s.get("ndcg@12"),
                       "r_precision": s.get("r_precision"), "mrr": s.get("mrr")}
                for kk in field_ks:
                    row[f"recall@{kk}"] = s.get(f"recall@{kk}")
                    row[f"precision@{kk}"] = s.get(f"precision@{kk}")
                w.writerow(row)

    L: list[str] = []
    L.append("# Experiment 14 — Fair-prompt re-test (grounded + semantic HyDE), retrieval-only")
    L.append("")
    L.append(f"Dataset: {n_scored} BHXH questions. Tuned stack (BGE-M3 LoRA + `clause_vec_tuned`). "
             "Headline stratum = **in_corpus**.")
    L.append("Parity prompts (grounded + semantic) share HyDE1's vocab scaffold; only the "
             "grounding block differs. HyDE1 = canonical (frozen) prompt = the bar.")
    L.append("")

    def _tbl(metric, stratum=None):
        out = ["| arm | n | " + " | ".join(f"@{kk}" for kk in field_ks) + " |",
               "|---|---:|" + "|".join(["---:"] * len(field_ks)) + "|"]
        for arm in ARMS:
            s = summ[arm] if stratum is None else strat[arm][stratum]
            out.append(f"| {arm} | {s.get('n', 0)} | "
                       + " | ".join(_fmt(s.get(f"{metric}@{kk}")) for kk in field_ks) + " |")
        return out

    L.append("## In-corpus (headline)")
    L.append("")
    L.append("### recall@K")
    L += _tbl("recall", "in_corpus")
    L.append("")
    L.append("### precision@K  (precision@1 is the headline; @2+ is cardinality-capped)")
    L += _tbl("precision", "in_corpus")
    L.append("")
    L.append("| arm | n | R-Precision | MRR | NDCG@12 |")
    L.append("|---|---:|---:|---:|---:|")
    for arm in ARMS:
        s = strat[arm]["in_corpus"]
        L.append(f"| {arm} | {s.get('n',0)} | {_fmt(s.get('r_precision'))} | "
                 f"{_fmt(s.get('mrr'))} | {_fmt(s.get('ndcg@12'))} |")
    L.append("")

    pv = prov.get("in_corpus") or prov.get("overall") or {}
    L.append("## Semantic-frame provenance (dense_hyde_semantic) — frame builder UNCHANGED vs exp 13")
    L.append("")
    if pv.get("n"):
        L.append(f"- concept_match_rate (in_corpus): **{_fmt(pv.get('concept_match_rate'))}** "
                 f"(fallback {_fmt(pv.get('fallback_rate'))})")
        L.append(f"- mean concepts/q: {_fmt(pv.get('mean_n_concepts'))}; "
                 f"mean KG entities/q: {_fmt(pv.get('mean_n_kg_entities'))}")
    else:
        L.append("_(no semantic records)_")
    L.append("")

    L.append("## Overall macro (all strata)")
    L.append("")
    L.append("### recall@K")
    L += _tbl("recall")
    L.append("")
    L.append("## Stratified recall@12")
    L.append("")
    L.append("| arm | " + " | ".join(STRATA) + " |")
    L.append("|---|" + "|".join(["---:"] * len(STRATA)) + "|")
    for arm in ARMS:
        L.append(f"| {arm} | " + " | ".join(_fmt(strat[arm][c].get("recall@12")) for c in STRATA) + " |")
    L.append("")
    (REPORT_DIR / "retrieval_report.md").write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/3] Validating gold ...")
    gold_map = load_gold()
    q_by_stt = {q["stt"]: q for q in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))}
    in_corpus_codes = {m.full_id for m in load_law_metadata().values()}
    stt_subset = _pilot_subset(args.full)
    if stt_subset is not None:
        print(f"      pilot subset n={len(stt_subset)}")

    print("[2/3] Scoring arms ...")
    summ: dict = {}
    strat: dict = {}
    prov: dict = {}
    for arm in ARMS:
        recs = load_arm_records(arm, stt_subset)
        rows = score_arm(arm, recs, gold_map, q_by_stt, in_corpus_codes) if recs else []
        summ[arm] = aggregate(rows)
        strat[arm] = aggregate_stratified(rows)
        if arm == "dense_hyde_semantic":
            prov["overall"] = provenance(rows)
            prov["in_corpus"] = provenance([r for r in rows if r["category"] == "in_corpus"])
        print(f"      {arm:<22} scored {len(rows)}")

    print("[3/3] Writing metrics + report ...")
    n_scored = max((s.get("n") or 0) for s in summ.values()) if summ else 0
    write_outputs(summ, strat, prov, n_scored)

    print("\n=== in_corpus headline ===")
    print(f"  {'arm':<22}{'n':>4}{'R@5':>9}{'R@12':>9}{'P@1':>9}{'R-Prec':>9}{'MRR':>9}")
    for arm in ARMS:
        s = strat[arm]["in_corpus"]
        print(f"  {arm:<22}{s.get('n',0):>4}{_fmt(s.get('recall@5')):>9}{_fmt(s.get('recall@12')):>9}"
              f"{_fmt(s.get('precision@1')):>9}{_fmt(s.get('r_precision')):>9}{_fmt(s.get('mrr')):>9}")
    print(f"\n  report: {REPORT_DIR / 'retrieval_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
