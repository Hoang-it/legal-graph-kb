"""Per-stage retrieval funnel for the `cypher_walk` arm of experiment 11.

Mirrors ``scripts/exp08_funnel.py`` shape but answers the one question exp 11
exists to answer: **does the Cypher walk move gold articles into the pool
that the seed (vector) retrieval missed, or does it degenerate to seed +
fallback?**

Stages (article-level, plan §4 stage order):

1. ``seed (vector)``       — articles of the vector seed clauses
2. ``+ cypher_new``        — seed ∪ articles surfaced by the Cypher walk
3. ``+ fallback``          — seed ∪ articles added by the vanilla expand
                             fallback (only populated when fallback fired)
4. ``final (fused top-K)`` — the RRF-fused top-K article set actually returned

For each stage we macro-average, across questions with non-empty gold:

- recall@12 and recall@all
- the gold-articles-in-stage count (summed), so the cypher_new / fallback /
  final transitions show how many gold articles each step actually adds.

Reads ``experiments/11_graphrag_cypher/results/cypher_walk/A<stt>.json``
(the per-stage article projections written by ``exp11_run.py``). Output:
``experiments/11_graphrag_cypher/report/funnel_cypher_walk_K12.md`` +
console summary, stratified by L41 presence.
"""
from __future__ import annotations

import argparse
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
RESULTS_DIR = EXP_DIR / "results" / "cypher_walk"
REPORT_DIR = EXP_DIR / "report"
METRICS_DIR = EXP_DIR / "metrics"
QUESTIONS_PATH = _REPO / "data" / "eval" / "questions_200.json"
REGISTRY_PATH = _REPO / "data" / "legal_sources.yaml"
PILOT_50_PATH = EXP_DIR / "pilot_50_stt.json"

K = 12
STRATA = ("l41_only", "mixed_l41_other", "no_l41")


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


def _dedupe(seq) -> list[str]:
    return list(dict.fromkeys(x for x in seq if x))


def recall_at(pool: list[str], gold: set[str], k: int | None) -> float | None:
    if not gold:
        return None
    sub = pool if k is None else pool[:k]
    return len(set(sub) & gold) / len(gold)


def _macro(vs):
    xs = [v for v in vs if v is not None]
    return round(mean(xs), 4) if xs else None


def load_gold() -> dict[int, list[str]]:
    ok, summary = validate_gold_citations(
        questions_path=QUESTIONS_PATH, registry_path=REGISTRY_PATH, out_dir=METRICS_DIR
    )
    if not ok:
        print(f"FAIL: gold validation; see {summary['errors_path']}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(Path(summary["normalized_path"]).read_text(encoding="utf-8"))
    return {int(k): v.get("gold_articles") or [] for k, v in data["records"].items()}


def load_records(stt_subset: set[int] | None) -> list[dict]:
    if not RESULTS_DIR.is_dir():
        print(f"FAIL: results dir not found: {RESULTS_DIR}", file=sys.stderr)
        sys.exit(1)
    out: list[dict] = []
    for p in sorted(RESULTS_DIR.glob("A*.json")):
        if p.name.endswith(".error.json"):
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        if stt_subset is not None and int(rec["stt"]) not in stt_subset:
            continue
        out.append(rec)
    return out


def stage_pools(rec: dict) -> dict[str, list[str]]:
    ro = rec.get("retrieval_only") or {}
    seed = list(ro.get("seed_article_ids") or [])
    cyp = list(ro.get("cypher_new_article_ids") or [])
    fb = list(ro.get("fallback_article_ids") or [])
    final = list(ro.get("final_article_ids") or [])
    return {
        "seed (vector)": seed,
        "+ cypher_new": _dedupe(seed + cyp),
        "+ fallback": _dedupe(seed + fb),
        "final (fused top-K)": final,
    }


STAGE_ORDER = ["seed (vector)", "+ cypher_new", "+ fallback", "final (fused top-K)"]


def compute_row(rec: dict, gold_map: dict[int, list[str]]) -> dict:
    stt = int(rec["stt"])
    gold = set(gold_map.get(stt) or [])
    pools = stage_pools(rec)
    row: dict = {"stt": stt, "n_gold": len(gold)}
    for label in STAGE_ORDER:
        pool = pools[label]
        row[f"{label}|n"] = len(pool)
        row[f"{label}|recall@{K}"] = recall_at(pool, gold, K)
        row[f"{label}|recall@all"] = recall_at(pool, gold, None)
        row[f"{label}|gold_hits"] = len(set(pool) & gold)
    return row


def aggregate(rows: list[dict]) -> dict:
    out: dict = {"n": len(rows), "stage_order": STAGE_ORDER}
    for label in STAGE_ORDER:
        out[label] = {
            "avg_n": round(mean([r[f"{label}|n"] for r in rows]), 2) if rows else None,
            f"recall@{K}": _macro([r[f"{label}|recall@{K}"] for r in rows]),
            "recall@all": _macro([r[f"{label}|recall@all"] for r in rows]),
            "gold_hits_total": sum(r[f"{label}|gold_hits"] for r in rows),
        }
    return out


def _fmt(v):
    if v is None:
        return "—"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def write_markdown(overall: dict, strat: dict[str, dict]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "funnel_cypher_walk_K12.md"
    L: list[str] = []
    L.append(f"# Funnel — `cypher_walk` arm at K={K} (exp 11, n={overall['n']})")
    L.append("")
    L.append("Per-stage **article-level** retrieval recall + gold-hit counts for the")
    L.append("Cypher-walk retriever. Computed by")
    L.append("[`scripts/exp11_funnel.py`](../../../scripts/exp11_funnel.py) from the")
    L.append("per-stage projections in `results/cypher_walk/`. Macro-averaged across")
    L.append("questions with non-empty gold.")
    L.append("")
    L.append("Key question: does `+ cypher_new` raise recall above `seed (vector)`? If")
    L.append("`recall@all` for `+ cypher_new` equals `seed (vector)`, the walk surfaced")
    L.append("no gold the seed missed (the previous attempt's failure mode).")
    L.append("")

    def _section(title: str, agg: dict):
        if not agg.get("n"):
            return
        L.append(f"## {title} (n={agg['n']})")
        L.append("")
        L.append(f"| stage | avg \\|pool\\| | recall@{K} | recall@all | gold-hits (Σ) |")
        L.append("|---|---:|---:|---:|---:|")
        for label in agg["stage_order"]:
            s = agg[label]
            L.append(f"| {label} | {s['avg_n']} | {_fmt(s[f'recall@{K}'])} | "
                     f"{_fmt(s['recall@all'])} | {s['gold_hits_total']} |")
        L.append("")

    _section(f"Overall (all {overall['n']})", overall)
    for cat in STRATA:
        _section(f"{cat} stratum", strat.get(cat, {"n": 0}))

    L.append("## Notes")
    L.append("")
    L.append("- `+ cypher_new` = seed ∪ articles the Cypher walk surfaced beyond seed.")
    L.append("- `+ fallback` = seed ∪ articles the vanilla expand fallback added "
             "(populated only on questions where the Cypher walk found 0 new clauses).")
    L.append("- `final (fused top-K)` = the RRF-fused top-K actually returned — can be")
    L.append("  below `+ cypher_new`/`+ fallback` recall@all because it is K-capped.")
    L.append("- Gold-hits (Σ) = sum over questions of |stage pool ∩ gold| — the absolute")
    L.append("  count of gold articles present at each stage.")
    L.append("")
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--full", action="store_true",
                   help="Process every record on disk, ignoring pilot_50_stt.json.")
    args = p.parse_args()

    gold_map = load_gold()
    stt_subset = _pilot_subset(force_full=args.full)
    if stt_subset is not None:
        print(f"Pilot subset detected ({PILOT_50_PATH.name}): funnelling n={len(stt_subset)}")

    records = load_records(stt_subset)
    rows = [compute_row(r, gold_map) for r in records]

    cats: dict[str, list[dict]] = {c: [] for c in STRATA}
    for rec, row in zip(records, rows):
        cat = l41_stratum(gold_map.get(int(rec["stt"]), []))
        if cat in cats:
            cats[cat].append(row)

    overall = aggregate(rows)
    strat = {c: aggregate(cats[c]) for c in STRATA}
    out = write_markdown(overall, strat)

    print()
    print(f"=== cypher_walk funnel, K={K} (n={overall['n']}) ===")
    print(f"  {'stage':<22}{'avg_n':>8}{'R@'+str(K):>9}{'R@all':>9}{'gold_hits':>11}")
    for label in overall["stage_order"]:
        s = overall[label]
        print(f"  {label:<22}{s['avg_n']:>8}{_fmt(s[f'recall@{K}']):>9}"
              f"{_fmt(s['recall@all']):>9}{s['gold_hits_total']:>11}")
    print(f"\nWrote: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
