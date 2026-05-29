"""Retrieval recall audit — đo recall@K của vector search hiện tại.

Chạy BGE-M3 vector_search ở K=50 cho 200 câu hỏi, sau đó tính recall@K
tại K = [1, 5, 8, 16, 30, 50].

Không gọi LLM — chỉ cần Neo4j + BGE-M3.

Stratify:
  - all:         tất cả 200 câu
  - incorpus:    câu có ≥1 gold article từ L41_2024 (corpus đang index)
  - ooc_only:    câu không có gold nào từ L41_2024

Usage:
    python scripts/retrieval_audit.py [--out results/retrieval_audit.json]
                                      [--n 200]
                                      [--top-k 50]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

GOLD_NORMALIZED = _REPO_ROOT / "experiments/01_initial_eval/metrics/gold_citations_normalized.json"
QUESTIONS_PATH  = _REPO_ROOT / "data/eval/questions_200.json"
KS = [1, 5, 8, 16, 30, 50]


def load_gold(gold_path: Path) -> dict[str, list[str]]:
    """stt (str) → list of gold article_ids."""
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    return {stt: v["gold_articles"] for stt, v in data["records"].items()}


def incorpus_gold(gold_articles: list[str]) -> list[str]:
    """Only L41_2024.* articles from gold list."""
    return [a for a in gold_articles if a.startswith("L41_2024")]


def run_audit(n: int, top_k: int, out_path: Path) -> None:
    from runtime.rag_query import RagPipeline

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))[:n]
    gold_map = load_gold(GOLD_NORMALIZED)

    pipeline = RagPipeline()
    _ = pipeline.embed_model  # pre-load BGE-M3

    print(f"\nRunning retrieval audit — {len(questions)} questions, top_k={top_k}")
    print("-" * 60)

    records: list[dict] = []
    t_start = time.time()

    for i, q in enumerate(questions, 1):
        stt = str(q["stt"])
        question = q["question"]
        gold_all  = gold_map.get(stt, [])
        gold_in   = incorpus_gold(gold_all)

        hits = pipeline.vector_search(question, top_k=top_k)
        retrieved_articles = list(dict.fromkeys(h.article_id for h in hits))

        hits_at: dict[int, list[str]] = {}
        for k in KS:
            hits_at[k] = list(dict.fromkeys(h.article_id for h in hits[:k]))

        def recall_at(k: int, gold: list[str]) -> float | None:
            if not gold:
                return None
            hit = set(hits_at[k]) & set(gold)
            return round(len(hit) / len(gold), 4)

        rec = {
            "stt": stt,
            "question": question,
            "gold_all":  gold_all,
            "gold_incorpus": gold_in,
            "n_gold_all": len(gold_all),
            "n_gold_incorpus": len(gold_in),
            "retrieved_top50": retrieved_articles,
        }
        for k in KS:
            rec[f"recall_all@{k}"]       = recall_at(k, gold_all)
            rec[f"recall_incorpus@{k}"]  = recall_at(k, gold_in)

        records.append(rec)

        if i % 20 == 0 or i == len(questions):
            elapsed = time.time() - t_start
            r8_all = _mean([r[f"recall_all@8"] for r in records if r[f"recall_all@8"] is not None])
            r8_in  = _mean([r[f"recall_incorpus@8"] for r in records if r[f"recall_incorpus@8"] is not None])
            print(f"  [{i:>3}/{len(questions)}] {elapsed:.0f}s | recall_all@8={r8_all:.3f} recall_incorpus@8={r8_in:.3f}")

    pipeline.close()

    # Aggregate
    summary = _aggregate(records)
    output = {"summary": summary, "records": records}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_summary(summary)
    print(f"\nSaved → {out_path}")


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _aggregate(records: list[dict]) -> dict:
    strata = {
        "all": records,
        "incorpus": [r for r in records if r["n_gold_incorpus"] > 0],
        "ooc_only":  [r for r in records if r["n_gold_incorpus"] == 0],
    }
    summary: dict = {}
    for label, recs in strata.items():
        if not recs:
            continue
        entry: dict = {"n": len(recs)}
        for k in KS:
            key_all = f"recall_all@{k}"
            key_in  = f"recall_incorpus@{k}"
            vals_all = [r[key_all] for r in recs if r[key_all] is not None]
            vals_in  = [r[key_in]  for r in recs if r[key_in]  is not None]
            entry[f"recall_all@{k}"]       = _mean(vals_all)
            entry[f"recall_incorpus@{k}"]  = _mean(vals_in) if vals_in else None
            # hit_rate: % of questions where at least 1 gold article was retrieved
            entry[f"hit_rate_all@{k}"]      = _mean([1.0 if (r[key_all] or 0) > 0 else 0.0 for r in recs])
            entry[f"hit_rate_incorpus@{k}"] = _mean([1.0 if (r[key_in] or 0) > 0 else 0.0 for r in recs]) if vals_in else None
        summary[label] = entry
    return summary


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 70)
    print("RETRIEVAL RECALL AUDIT RESULTS")
    print("=" * 70)
    for label, entry in summary.items():
        print(f"\n[{label.upper()}]  n={entry['n']}")
        print(f"  {'K':>4} | {'recall_all':>12} | {'recall_incorp':>13} | {'hit_rate_all':>12} | {'hit_rate_in':>11}")
        print(f"  {'-'*4}-+-{'-'*12}-+-{'-'*13}-+-{'-'*12}-+-{'-'*11}")
        for k in KS:
            r_all  = entry.get(f"recall_all@{k}")
            r_in   = entry.get(f"recall_incorpus@{k}")
            hr_all = entry.get(f"hit_rate_all@{k}")
            hr_in  = entry.get(f"hit_rate_incorpus@{k}")
            def fmt(v):
                return f"{v:.4f}" if v is not None else "  N/A "
            print(f"  {k:>4} | {fmt(r_all):>12} | {fmt(r_in):>13} | {fmt(hr_all):>12} | {fmt(hr_in):>11}")
    print()
    # Highlight key number
    best = summary.get("incorpus", summary.get("all", {}))
    r8   = best.get("recall_all@8")
    r50  = best.get("recall_all@50")
    if r8 is not None:
        print(f"Key numbers (incorpus stratum, recall_all = gold includes all laws):")
        print(f"  Current system recall@8  = {r8:.1%}")
        print(f"  Headroom at recall@50    = {r50:.1%}  (upper bound if top-50 is used)")
        print(f"  v5 target recall (e2e)   = 70%+ (requires retrieval >> 70%)")
    print()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=_REPO_ROOT / "results/retrieval_audit.json")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    for req in (GOLD_NORMALIZED, QUESTIONS_PATH):
        if not req.exists():
            print(f"FAIL: missing {req}", file=sys.stderr)
            return 1

    run_audit(n=args.n, top_k=args.top_k, out_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
