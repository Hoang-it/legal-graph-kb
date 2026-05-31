"""Per-stage retrieval funnel for the `full_rerank_hyde` arm of experiment 08.

Mirrors ``scripts/exp06_funnel.py`` shape but operates on
``experiments/08_hyde_retrieval/results/full_rerank_hyde/`` records.

Useful for diagnosing whether the HyDE-augmented dense channel changes the
per-stage gold profile of the full v5 pipeline — i.e. does HyDE shift gold
into the dense pool early enough that rerank1 / rerank2 can keep it, or
does it merely re-shuffle near-misses without raising the ceiling?

Stages reported (in pipeline order):
1. ``dense``          — BGE-M3 LoRA dense top-100 clauses, article-deduped,
                        using HyDE-embedded query for the `_hyde` arm
3. ``sparse``         — Lucene BM25 top-100 clauses (raw question — HyDE
                        does NOT touch sparse, plan §D3)
4. ``dense ∪ sparse`` — union of the two pre-temporal pools
5. ``post_temporal``  — pool after dropping laws not in force at event_date
6. ``fused (RRF)``    — top_after_fusion=150 from Reciprocal Rank Fusion
7. ``rerank1``        — rerank1_top_k=50 (seed set)
8. ``expanded``       — seeds + REFERS_TO neighbours
9. ``final (rerank2)``— rerank2_top_k=100 (final pool)

For each stage we compute, macro-averaged across all questions with
non-empty gold:

- recall@K with K=12
- recall over the entire stage pool (no K cap)
- For RANKED stages: NDCG@12 and MRR (rank of first gold article)

Outputs a markdown report to
``experiments/08_hyde_retrieval/report/funnel_full_rerank_hyde_K12.md``
plus a console summary, stratified by gold-corpus category.
"""
from __future__ import annotations

import argparse
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

EXP_DIR = _REPO / "experiments" / "08_hyde_retrieval"
RESULTS_DIR = EXP_DIR / "results" / "full_rerank_hyde"
REPORT_DIR = EXP_DIR / "report"
METRICS_DIR = EXP_DIR / "metrics"
QUESTIONS_PATH = _REPO / "data" / "eval" / "questions_200.json"
REGISTRY_PATH = _REPO / "data" / "legal_sources.yaml"
PILOT_50_PATH = EXP_DIR / "pilot_50_stt.json"


def _pilot_subset(force_full: bool = False) -> set[int] | None:
    """Return the stt set the pilot covers, or None for the full dataset.

    Pass ``force_full=True`` (via ``--full``) to ignore the pilot file and
    process every record on disk.
    """
    if force_full:
        return None
    if not PILOT_50_PATH.exists():
        return None
    payload = json.loads(PILOT_50_PATH.read_text(encoding="utf-8"))
    return set(int(s) for s in payload.get("stt_list") or [])

K = 12

STAGES_RANKED = [
    ("dense", "dense_article_ids"),
    ("sparse", "sparse_article_ids"),
    ("post_temporal", "post_temporal_article_ids"),
    ("fused (RRF)", "fused_article_ids"),
    ("rerank1", "rerank1_article_ids"),
    ("expanded", "expanded_article_ids"),
    ("final (rerank2)", "final_article_ids"),
]

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


def recall_at(pool: list[str], gold: set[str], k: int | None) -> float | None:
    if not gold:
        return None
    sub = pool if k is None else pool[:k]
    return len(set(sub) & gold) / len(gold)


def ndcg_at(pool: list[str], gold: set[str], k: int) -> float | None:
    if not gold:
        return None
    dcg = 0.0
    for i, aid in enumerate(pool[:k], start=1):
        if aid in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return 0.0 if ideal == 0 else dcg / ideal


def mrr(pool: list[str], gold: set[str]) -> float | None:
    if not gold:
        return None
    for i, aid in enumerate(pool, start=1):
        if aid in gold:
            return 1.0 / i
    return 0.0


def _macro(vs):
    xs = [v for v in vs if v is not None]
    return round(mean(xs), 4) if xs else None


def load_gold() -> dict[int, list[str]]:
    ok, summary = validate_gold_citations(
        questions_path=QUESTIONS_PATH,
        registry_path=REGISTRY_PATH,
        out_dir=METRICS_DIR,
    )
    if not ok:
        print(f"FAIL: gold validation, see {summary['errors_path']}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(Path(summary["normalized_path"]).read_text(encoding="utf-8"))
    return {int(k): v.get("gold_articles") or [] for k, v in data["records"].items()}


def load_records(stt_subset: set[int] | None = None) -> list[dict]:
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


def build_union_dense_sparse(retrieval: dict) -> list[str]:
    """Pre-temporal-filter union pool, first-occurrence preserved."""
    seen: dict[str, None] = {}
    for aid in (retrieval.get("dense_article_ids") or []):
        seen.setdefault(aid, None)
    for aid in (retrieval.get("sparse_article_ids") or []):
        seen.setdefault(aid, None)
    return list(seen.keys())


def _dedupe_first(seq):
    seen: dict[str, None] = {}
    for aid in seq or []:
        if aid:
            seen.setdefault(aid, None)
    return list(seen.keys())


def compute_per_record(rec: dict, gold_map: dict[int, list[str]]) -> dict:
    stt = int(rec["stt"])
    gold = set(gold_map.get(stt) or [])
    retrieval = rec.get("retrieval_only") or {}
    pools: dict[str, list[str]] = {}
    for label, key in STAGES_RANKED:
        raw = list(retrieval.get(key) or [])
        # expanded carries clause-level duplicates for the seed prefix.
        pools[label] = _dedupe_first(raw) if key == "expanded_article_ids" else raw
    pools["dense ∪ sparse"] = build_union_dense_sparse(retrieval)

    row: dict = {"stt": stt, "n_gold": len(gold)}
    for label, pool in pools.items():
        row[f"{label}|n"] = len(pool)
        row[f"{label}|recall@{K}"] = recall_at(pool, gold, K)
        row[f"{label}|recall@all"] = recall_at(pool, gold, None)
    for label in (
        "dense", "sparse", "fused (RRF)", "rerank1",
        "expanded", "final (rerank2)",
    ):
        pool = pools[label]
        row[f"{label}|ndcg@{K}"] = ndcg_at(pool, gold, K)
        row[f"{label}|mrr"] = mrr(pool, gold)
    return row


def aggregate(rows: list[dict]) -> dict:
    out: dict = {"n": len(rows)}
    labels = [lbl for lbl, _ in STAGES_RANKED] + ["dense ∪ sparse"]
    order = [
        "dense", "sparse", "dense ∪ sparse",
        "post_temporal", "fused (RRF)", "rerank1",
        "expanded", "final (rerank2)",
    ]
    out["stage_order"] = order
    for label in order:
        if label not in labels:
            continue
        out[label] = {
            f"recall@{K}": _macro([r[f"{label}|recall@{K}"] for r in rows]),
            "recall@all": _macro([r[f"{label}|recall@all"] for r in rows]),
            "avg_n": round(mean([r[f"{label}|n"] for r in rows]), 2) if rows else None,
        }
        if rows and f"{label}|ndcg@{K}" in rows[0]:
            out[label][f"ndcg@{K}"] = _macro([r[f"{label}|ndcg@{K}"] for r in rows])
            out[label]["mrr"] = _macro([r[f"{label}|mrr"] for r in rows])
    return out


def stage_drops(rows: list[dict]) -> list[dict]:
    transitions = [
        ("dense ∪ sparse", "post_temporal", "temporal filter"),
        ("post_temporal", "fused (RRF)", "RRF top-150 cap"),
        ("fused (RRF)", "rerank1", "rerank1 (50-seed cap)"),
        ("rerank1", "expanded", "graph expansion (additive)"),
        ("expanded", "final (rerank2)", "rerank2 (100-final cap)"),
    ]
    out = []
    for a, b, why in transitions:
        gold_in_a = 0
        gold_in_b = 0
        for r in rows:
            ra = r[f"{a}|recall@all"]
            rb = r[f"{b}|recall@all"]
            ng = r["n_gold"]
            if ra is None:
                continue
            gold_in_a += round(ra * ng)
            gold_in_b += round(rb * ng)
        out.append({
            "from": a,
            "to": b,
            "reason": why,
            "gold_in_from_stage_total": gold_in_a,
            "gold_in_to_stage_total": gold_in_b,
            "delta": gold_in_b - gold_in_a,
        })
    return out


def write_markdown(overall: dict, in_corpus: dict, mixed: dict, ooc: dict,
                    unparseable: dict, drops: dict[str, list[dict]]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "funnel_full_rerank_hyde_K12.md"

    def _fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines: list[str] = []
    lines.append(f"# Funnel — `full_rerank_hyde` arm at K={K} (exp 08, n={overall['n']})")
    lines.append("")
    lines.append("Per-stage retrieval recall + rank-aware metrics for the HyDE-")
    lines.append("augmented full pipeline. Computed by")
    lines.append("[`scripts/exp08_funnel.py`](../../../scripts/exp08_funnel.py) from")
    lines.append("the records in `results/full_rerank_hyde/`. Macro-averaged across")
    lines.append("questions with non-empty gold.")
    lines.append("")
    lines.append("Note: only the DENSE channel uses the HyDE doc embedding. Sparse")
    lines.append("channel keeps the raw question (plan §D3), so any lift in the")
    lines.append("post-RRF stages reflects the dense-side improvement propagating")
    lines.append("through the rest of the pipeline.")
    lines.append("")

    def _section(title: str, agg: dict, drop_rows: list[dict] | None):
        lines.append(f"## {title} (n={agg['n']})")
        lines.append("")
        lines.append(f"| stage | avg \\|pool\\| | recall@{K} | recall@all | NDCG@{K} | MRR |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for label in agg["stage_order"]:
            if label not in agg:
                continue
            s = agg[label]
            ndcg = s.get(f"ndcg@{K}", None)
            m = s.get("mrr", None)
            lines.append(
                f"| {label} | {s['avg_n']} | "
                f"{_fmt(s[f'recall@{K}'])} | {_fmt(s['recall@all'])} | "
                f"{_fmt(ndcg)} | {_fmt(m)} |"
            )
        lines.append("")
        if drop_rows:
            lines.append("**Stage-to-stage gold count (sum over all questions):**")
            lines.append("")
            lines.append("| from | to | total gold in `from` | total gold in `to` | Δ | cause |")
            lines.append("|---|---|---:|---:|---:|---|")
            for d in drop_rows:
                sign = "+" if d["delta"] >= 0 else ""
                lines.append(
                    f"| {d['from']} | {d['to']} | "
                    f"{d['gold_in_from_stage_total']} | "
                    f"{d['gold_in_to_stage_total']} | "
                    f"{sign}{d['delta']} | {d['reason']} |"
                )
            lines.append("")

    _section(f"Overall (all {overall['n']})", overall, drops["overall"])
    _section("in_corpus stratum", in_corpus, drops["in_corpus"])
    _section("mixed stratum", mixed, drops["mixed"])
    _section("ooc stratum", ooc, drops["ooc"])
    _section("unparseable stratum", unparseable, drops["unparseable"])

    lines.append("## Notes")
    lines.append("")
    lines.append("- `dense ∪ sparse` is the pre-temporal-filter pool (SET, not ranked) — NDCG/MRR not reported.")
    lines.append("- `post_temporal` ordering = dense pool (first) then sparse minus dense — synthetic order, rank-aware metrics partly mechanical.")
    lines.append("- `expanded` = rerank1 seeds (rerank1-score order) followed by REFERS_TO neighbours (graph order) — same caveat.")
    lines.append("- `rerank1` and `final (rerank2)` are TRUE rankings — NDCG/MRR reflect the cross-encoder's decisions.")
    lines.append("- To compare HyDE vs no-HyDE side-by-side, also run `python scripts/exp06_funnel.py` (or `exp08`'s own no-HyDE control via `--arm full_rerank` if you add that mode in future).")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--full", action="store_true",
                   help="Process every record on disk, ignoring pilot_50_stt.json.")
    args = p.parse_args()

    gold_map = load_gold()
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    q_by_stt = {q["stt"]: q for q in questions}
    in_corpus_codes = {m.full_id for m in load_law_metadata().values()}

    stt_subset = _pilot_subset(force_full=args.full)
    if stt_subset is not None:
        print(f"Pilot subset detected ({PILOT_50_PATH.name}): "
              f"funnelling n={len(stt_subset)} questions")
    elif args.full and PILOT_50_PATH.exists():
        print(f"--full set — ignoring {PILOT_50_PATH.name}; processing every record on disk.")

    records = load_records(stt_subset=stt_subset)
    rows = [compute_per_record(r, gold_map) for r in records]

    cats: dict[str, list[dict]] = {"in_corpus": [], "mixed": [], "ooc": [], "unparseable": []}
    for rec, row in zip(records, rows):
        cat = categorize(q_by_stt[int(rec["stt"])].get("gold_citations_raw"), in_corpus_codes)
        cats[cat].append(row)

    overall = aggregate(rows)
    in_corpus = aggregate(cats["in_corpus"])
    mixed = aggregate(cats["mixed"])
    ooc = aggregate(cats["ooc"])
    unparseable = aggregate(cats["unparseable"])

    drops = {
        "overall": stage_drops(rows),
        "in_corpus": stage_drops(cats["in_corpus"]),
        "mixed": stage_drops(cats["mixed"]),
        "ooc": stage_drops(cats["ooc"]),
        "unparseable": stage_drops(cats["unparseable"]),
    }

    out = write_markdown(overall, in_corpus, mixed, ooc, unparseable, drops)

    print()
    print(f"=== full_rerank_hyde funnel, K={K} (n={overall['n']} overall, n={in_corpus['n']} in_corpus) ===")
    print(f"  {'stage':<22} {'avg_n':>7} {'R@'+str(K)+'_ALL':>10} {'R@'+str(K)+'_IC':>10} "
          f"{'R_all_ALL':>10} {'R_all_IC':>10} {'NDCG@'+str(K)+'_IC':>12} {'MRR_IC':>8}")
    for label in overall["stage_order"]:
        if label not in overall:
            continue
        o = overall[label]
        i = in_corpus[label]
        def fmt(v): return f"{v:.4f}" if isinstance(v, float) else (v if v is not None else "—")
        print(
            f"  {label:<22} {o['avg_n']:>7} {fmt(o[f'recall@{K}']):>10} {fmt(i[f'recall@{K}']):>10} "
            f"{fmt(o['recall@all']):>10} {fmt(i['recall@all']):>10} "
            f"{fmt(i.get(f'ndcg@{K}')):>12} {fmt(i.get('mrr')):>8}"
        )

    print()
    print(f"=== Stage-to-stage gold delta (in_corpus n={in_corpus['n']}) ===")
    for d in drops["in_corpus"]:
        sign = "+" if d["delta"] >= 0 else ""
        print(f"  {d['from']:<22} -> {d['to']:<22} "
              f"Δ={sign}{d['delta']:>4} gold-hits  ({d['reason']})")

    print()
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
