"""Compute stage-by-stage retrieval recall@K from retrieve_only records.

Reads records produced by ``scripts/run_retrieval_only.py`` (one folder per
config under ``experiments/05_v5_retrieval_audit/<config>/A<stt>.json``) and
emits the gold drop-off table that drives the Week 1 decision tree.

Stages considered (article-level, deduped, ordered):
- dense_article_ids (BGE-M3 dense path only, pre-temporal)
- sparse_article_ids (Lucene BM25 path only, pre-temporal)
- post_temporal_article_ids (union, post temporal filter, pre RRF)
- fused_article_ids (post RRF, top ``top_after_fusion``)
- rerank1_article_ids (post first cross-encoder rerank)
- expanded_article_ids (rerank1 seeds + REFERS_TO neighbours)
- final_article_ids (post second cross-encoder rerank, top ``rerank2_top_k``)

For each stage, recall is computed against ``gold_articles`` parsed strict
from ``gold_citations_normalized.json`` (Sprint 2 version, includes the
QD366 fix).

Output: stdout table + per-config JSON saved to the same audit folder.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.legal_metadata import load_law_metadata

AUDIT_ROOT = Path("experiments/05_v5_retrieval_audit")

STAGES = (
    "dense_article_ids",
    "sparse_article_ids",
    "post_temporal_article_ids",
    "fused_article_ids",
    "rerank1_article_ids",
    "expanded_article_ids",
    "final_article_ids",
)
STAGE_LABELS = {
    "dense_article_ids": "dense",
    "sparse_article_ids": "sparse",
    "post_temporal_article_ids": "post_temporal",
    "fused_article_ids": "RRF",
    "rerank1_article_ids": "rerank1",
    "expanded_article_ids": "+expand",
    "final_article_ids": "final",
}

# K values to report per stage (capped to the stage's natural pool size).
KS = (5, 10, 12, 30, 50, 100)

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


def _load_gold() -> dict[int, list[str]]:
    """Prefer Sprint 2 normalized gold (includes QD366 entry)."""
    p = Path("experiments/04_v5_sprint2_m2/metrics/gold_citations_normalized.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    return {int(k): v.get("gold_articles") or [] for k, v in data.get("records", {}).items()}


def recall_at_k(retrieved: list[str], gold: set[str], k: int | None) -> float | None:
    if not gold:
        return None
    pool = set(retrieved if k is None else retrieved[:k])
    return len(gold & pool) / len(gold)


def _macro(values):
    vs = [v for v in values if v is not None]
    return round(mean(vs), 4) if vs else None


def analyze_config(config: str, gold_map: dict[int, list[str]], questions: dict[int, dict],
                   in_corpus_codes: set[str]) -> dict:
    cfg_dir = AUDIT_ROOT / config
    if not cfg_dir.is_dir():
        raise FileNotFoundError(cfg_dir)

    rows = []
    for rec_path in sorted(cfg_dir.glob("A*.json")):
        if rec_path.suffix == ".error.json":
            continue
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        stt = int(rec["stt"])
        gold = set(gold_map.get(stt) or [])
        ans = rec.get("retrieval_only") or {}
        row = {
            "stt": stt,
            "gold": sorted(gold),
            "category": categorize(
                questions[stt].get("gold_citations_raw"), in_corpus_codes
            ),
        }
        for stage in STAGES:
            articles = list(ans.get(stage) or [])
            row[stage] = articles
            for k in KS:
                row[f"{STAGE_LABELS[stage]}@{k}"] = recall_at_k(articles, gold, k)
            row[f"{STAGE_LABELS[stage]}@all"] = recall_at_k(articles, gold, None)
        rows.append(row)
    return {"config": config, "rows": rows}


def print_stage_table(config_results: list[dict]) -> None:
    """Aggregate per-stage macro recall, overall and per stratum."""

    def fmt(v):
        return "-" if v is None else f"{v:.3f}"

    for cr in config_results:
        rows = cr["rows"]
        print()
        print(f"=== {cr['config']} (n={len(rows)}) ===")
        # Overall
        print(f"  {'stage':<14}", *(f"{f'@{k}':>8}" for k in KS), f"{'@all':>8}")
        for stage in STAGES:
            label = STAGE_LABELS[stage]
            vals = [fmt(_macro([r[f"{label}@{k}"] for r in rows])) for k in KS]
            v_all = fmt(_macro([r[f"{label}@all"] for r in rows]))
            print(f"  {label:<14}", *(f"{v:>8}" for v in vals), f"{v_all:>8}")

        # Stratified — focus on in_corpus since OOC is identically 0 across stages
        print(f"  --- in_corpus subset ---")
        ic = [r for r in rows if r["category"] == "in_corpus"]
        if ic:
            print(
                f"  {'stage':<14}",
                *(f"{f'@{k}':>8}" for k in KS),
                f"{'@all':>8}  (n={len(ic)})",
            )
            for stage in STAGES:
                label = STAGE_LABELS[stage]
                vals = [fmt(_macro([r[f"{label}@{k}"] for r in ic])) for k in KS]
                v_all = fmt(_macro([r[f"{label}@all"] for r in ic]))
                print(f"  {label:<14}", *(f"{v:>8}" for v in vals), f"{v_all:>8}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "configs",
        nargs="+",
        help="Config folder names under experiments/05_v5_retrieval_audit/",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Also write summary JSON to AUDIT_ROOT/stage_analysis_<config>.json",
    )
    args = p.parse_args()

    questions = json.loads(Path("data/eval/questions_200.json").read_text(encoding="utf-8"))
    q_by_stt = {q["stt"]: q for q in questions}
    in_corpus_codes = {m.full_id for m in load_law_metadata().values()}
    gold_map = _load_gold()

    results = [analyze_config(c, gold_map, q_by_stt, in_corpus_codes) for c in args.configs]
    print_stage_table(results)

    if args.save:
        for cr in results:
            out = AUDIT_ROOT / f"stage_analysis_{cr['config']}.json"
            out.write_text(json.dumps(cr, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
