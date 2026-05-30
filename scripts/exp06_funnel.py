"""Per-stage retrieval funnel for the `full_rerank` arm of experiment 06.

Reads every record under
``experiments/06_retrieval_dense_vs_full/results/full_rerank/`` and reports,
at K=12, what fraction of gold articles each pipeline stage contains.

Stages reported (in pipeline order):
1. ``dense``          — BGE-M3 LoRA dense top-30 clauses, article-deduped
2. ``sparse``         — Lucene BM25 top-30 clauses, article-deduped
3. ``dense ∪ sparse`` — union of the two pre-temporal pools
4. ``post_temporal``  — pool after dropping laws not in force at event_date
5. ``fused (RRF)``    — top_after_fusion=50 from Reciprocal Rank Fusion
6. ``rerank1``        — rerank1_top_k=15 (seed set)
7. ``expanded``       — seeds + REFERS_TO neighbours
8. ``final (rerank2)``— rerank2_top_k=12 (LLM context)

For each stage we compute, macro-averaged across all questions with
non-empty gold:

- recall@K with K=12 (top-12 of the stage's article list)
- recall over the entire stage pool (no K cap)
- |stage ∩ gold| / |gold| ranked at any position
- For RANKED stages (dense, sparse, fused, rerank1, expanded, final):
  NDCG@12 and MRR (rank of first gold article)

Reads the gold normalisation produced by exp 06 (or regenerates it via
``eval_core.gold``).

Outputs a markdown table to stdout plus
``experiments/06_retrieval_dense_vs_full/report/funnel_full_rerank_K12.md``.

Stratified by gold-corpus category to isolate where the rerank stack
matters.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from statistics import mean

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Windows console defaults to cp1252 which can't print Vietnamese / set-union ∪.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

from eval_core.gold import validate_gold_citations
from src.legal_metadata import load_law_metadata

EXP_DIR = _REPO / "experiments" / "06_retrieval_dense_vs_full"
RESULTS_DIR = EXP_DIR / "results" / "full_rerank"
REPORT_DIR = EXP_DIR / "report"
METRICS_DIR = EXP_DIR / "metrics"
QUESTIONS_PATH = _REPO / "data" / "eval" / "questions_200.json"
REGISTRY_PATH = _REPO / "data" / "legal_sources.yaml"

K = 12

# Stages — (label, accessor). For "dense_union_sparse" we build at runtime
# because the record doesn't store it directly.
STAGES_RANKED = [
    ("dense", "dense_article_ids"),
    ("sparse", "sparse_article_ids"),
    ("post_temporal", "post_temporal_article_ids"),
    ("fused (RRF)", "fused_article_ids"),
    ("rerank1 (top-15)", "rerank1_article_ids"),
    ("expanded", "expanded_article_ids"),
    ("final (rerank2, top-12)", "final_article_ids"),
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


def load_records() -> list[dict]:
    out: list[dict] = []
    for p in sorted(RESULTS_DIR.glob("A*.json")):
        if p.name.endswith(".error.json"):
            continue
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def build_union_dense_sparse(retrieval: dict) -> list[str]:
    """Pre-temporal union pool: dense_article_ids then sparse_article_ids
    (article-deduped, first-occurrence preserved). Reflects what the
    pipeline saw before temporal filtering."""
    seen: dict[str, None] = {}
    for aid in (retrieval.get("dense_article_ids") or []):
        seen.setdefault(aid, None)
    for aid in (retrieval.get("sparse_article_ids") or []):
        seen.setdefault(aid, None)
    return list(seen.keys())


def _dedupe_first(seq):
    """Article-first-occurrence dedupe. The pipeline stores
    ``expanded_article_ids`` with duplicates (seeds are clause-level — multiple
    clauses can share the same article_id). NDCG on raw expanded would
    double-count gold, so dedupe before any rank-aware metric."""
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
        # expanded_article_ids in raw records carries clause-level duplicates
        # for the seed prefix. Dedupe to article-first-occurrence so every
        # metric (recall, NDCG, MRR, precision) is at article granularity.
        pools[label] = _dedupe_first(raw) if key == "expanded_article_ids" else raw
    pools["dense ∪ sparse"] = build_union_dense_sparse(retrieval)

    row: dict = {"stt": stt, "n_gold": len(gold)}
    for label, pool in pools.items():
        row[f"{label}|n"] = len(pool)
        row[f"{label}|recall@{K}"] = recall_at(pool, gold, K)
        row[f"{label}|recall@all"] = recall_at(pool, gold, None)
    # Rank-aware only on ranked stages (skip post_temporal & dense∪sparse which are pre-RRF union, NOT a ranking)
    for label in (
        "dense", "sparse", "fused (RRF)", "rerank1 (top-15)",
        "expanded", "final (rerank2, top-12)",
    ):
        pool = pools[label]
        row[f"{label}|ndcg@{K}"] = ndcg_at(pool, gold, K)
        row[f"{label}|mrr"] = mrr(pool, gold)
    return row


def aggregate(rows: list[dict]) -> dict:
    out: dict = {"n": len(rows)}
    labels = [lbl for lbl, _ in STAGES_RANKED] + ["dense ∪ sparse"]
    # Re-order to match pipeline narrative
    order = [
        "dense", "sparse", "dense ∪ sparse",
        "post_temporal", "fused (RRF)", "rerank1 (top-15)",
        "expanded", "final (rerank2, top-12)",
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
        if f"{label}|ndcg@{K}" in (rows[0] if rows else {}):
            out[label][f"ndcg@{K}"] = _macro([r[f"{label}|ndcg@{K}"] for r in rows])
            out[label]["mrr"] = _macro([r[f"{label}|mrr"] for r in rows])
    return out


def stage_drops(rows: list[dict]) -> list[dict]:
    """For each consecutive stage pair in the pipeline, count questions
    where gold was *present in earlier stage* but *missing in later stage*."""
    transitions = [
        ("dense ∪ sparse", "post_temporal", "temporal filter"),
        ("post_temporal", "fused (RRF)", "RRF top-50 cap"),
        ("fused (RRF)", "rerank1 (top-15)", "rerank1 (15-seed cap)"),
        ("rerank1 (top-15)", "expanded", "graph expansion (additive)"),
        ("expanded", "final (rerank2, top-12)", "rerank2 (12-final cap)"),
    ]
    out = []
    for a, b, why in transitions:
        gold_in_a = 0
        gold_in_b = 0
        gold_kept = 0
        gold_added = 0
        for r in rows:
            stt = r["stt"]
            ga = set()
            gb = set()
            # We don't carry the gold here. Re-derive from row keys: we stored recall@all which is |intersect|/|gold|.
            # Inverse: |intersect| = recall@all * n_gold. Round to int.
            ra = r[f"{a}|recall@all"]
            rb = r[f"{b}|recall@all"]
            ng = r["n_gold"]
            if ra is None:
                continue
            ia = round(ra * ng)
            ib = round(rb * ng)
            gold_in_a += ia
            gold_in_b += ib
            gold_kept += min(ia, ib)
            gold_added += max(0, ib - ia)
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
    out = REPORT_DIR / "funnel_full_rerank_K12.md"

    def _fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines: list[str] = []
    lines.append(f"# Funnel — `full_rerank` arm at K={K} (exp 06, n=200)")
    lines.append("")
    lines.append("Per-stage retrieval recall + rank-aware metrics. Computed by")
    lines.append("[`scripts/exp06_funnel.py`](../../../scripts/exp06_funnel.py) from")
    lines.append("the 200 records in `results/full_rerank/`. Macro-averaged across")
    lines.append("questions with non-empty gold.")
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

    _section("Overall (all 200)", overall, drops["overall"])
    _section("in_corpus stratum", in_corpus, drops["in_corpus"])
    _section("mixed stratum", mixed, drops["mixed"])
    _section("ooc stratum", ooc, drops["ooc"])
    _section("unparseable stratum", unparseable, drops["unparseable"])

    lines.append("## Notes")
    lines.append("")
    lines.append("- `dense ∪ sparse` is the pre-temporal-filter pool used by the audit. It is a SET (rank-aware metrics not meaningful), so NDCG/MRR are not reported for it.")
    lines.append("- `post_temporal` ordering = dense pool (first), then sparse pool minus dense — i.e. the order the retriever stored. Rank-aware metrics on it reflect that synthetic ordering, not a true ranking.")
    lines.append("- `expanded` = rerank1 seeds (in rerank1 score order) followed by REFERS_TO neighbours (in graph traversal order). Same caveat as post_temporal — the rank is partly mechanical.")
    lines.append("- `rerank1` and `final (rerank2)` are TRUE rankings — NDCG/MRR there reflect the cross-encoder's decisions.")
    lines.append("- Gold counts in the funnel use `round(recall@all × |gold|)`. Sum across questions, so a single gold article can be counted multiple times if multiple questions share it.")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    gold_map = load_gold()
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    q_by_stt = {q["stt"]: q for q in questions}
    in_corpus_codes = {m.full_id for m in load_law_metadata().values()}

    records = load_records()
    rows = [compute_per_record(r, gold_map) for r in records]

    # Stratify
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

    # Console summary (overall + in_corpus side by side)
    print()
    print(f"=== full_rerank funnel, K={K} (n={overall['n']} overall, n={in_corpus['n']} in_corpus) ===")
    print(f"  {'stage':<26} {'avg_n':>7} {'R@'+str(K)+'_ALL':>10} {'R@'+str(K)+'_IC':>10} "
          f"{'R_all_ALL':>10} {'R_all_IC':>10} {'NDCG@'+str(K)+'_IC':>12} {'MRR_IC':>8}")
    for label in overall["stage_order"]:
        if label not in overall:
            continue
        o = overall[label]
        i = in_corpus[label]
        rk = o[f"recall@{K}"]
        ric = i[f"recall@{K}"]
        rallo = o["recall@all"]
        ralli = i["recall@all"]
        n = o.get(f"ndcg@{K}")
        nic = i.get(f"ndcg@{K}")
        mic = i.get("mrr")
        def fmt(v): return f"{v:.4f}" if isinstance(v, float) else (v if v is not None else "—")
        print(
            f"  {label:<26} {o['avg_n']:>7} {fmt(rk):>10} {fmt(ric):>10} "
            f"{fmt(rallo):>10} {fmt(ralli):>10} {fmt(nic):>12} {fmt(mic):>8}"
        )

    print()
    print(f"=== Stage-to-stage gold delta (in_corpus n={in_corpus['n']}) ===")
    for d in drops["in_corpus"]:
        sign = "+" if d["delta"] >= 0 else ""
        print(f"  {d['from']:<24} → {d['to']:<26} "
              f"Δ={sign}{d['delta']:>4} gold-hits  ({d['reason']})")

    print()
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
