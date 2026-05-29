"""Phân tích kết quả retrieval_audit.json với corpus definition đúng.

Corpus = {L41_2024, L58_2014, L45_2019} (3 luật được index trong Neo4j).

Usage:
    python scripts/retrieval_audit_analyze.py
    python scripts/retrieval_audit_analyze.py --input results/retrieval_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

INDEXED_LAWS = {"L41_2024", "L58_2014", "L45_2019"}
KS = [1, 5, 8, 16, 30, 50]


def law_prefix(article_id: str) -> str:
    return article_id.split(".")[0] if "." in article_id else article_id


def is_indexed(article_id: str) -> bool:
    return law_prefix(article_id) in INDEXED_LAWS


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def recall_at(gold: list[str], retrieved_top50: list[str], k: int) -> float | None:
    if not gold:
        return None
    hit = set(gold) & set(retrieved_top50[:k])
    return round(len(hit) / len(gold), 4)


def enrich(records: list[dict]) -> list[dict]:
    """Thêm gold_corpus + recall_corpus@K vào mỗi record."""
    for r in records:
        gold_all    = r["gold_all"]
        gold_corpus = [a for a in gold_all if is_indexed(a)]
        retrieved   = r.get("retrieved_top50", [])
        r["gold_corpus"]   = gold_corpus
        r["n_gold_corpus"] = len(gold_corpus)
        for k in KS:
            r[f"recall_corpus@{k}"] = recall_at(gold_corpus, retrieved, k)
            r[f"recall_all@{k}"]    = recall_at(gold_all,    retrieved, k)
    return records


def print_table(label: str, subset: list[dict]) -> None:
    n = len(subset)
    has_corpus = [r for r in subset if r["n_gold_corpus"] > 0]
    print(f"\n[{label}]  n={n}")
    print(f"  {'K':>3} | {'recall_corpus':>13} | {'recall_all':>10} | {'hit_rate_corpus':>15}")
    print(f"  {'-'*3}-+-{'-'*13}-+-{'-'*10}-+-{'-'*15}")
    for k in KS:
        rc = mean([r[f"recall_corpus@{k}"] for r in has_corpus
                   if r[f"recall_corpus@{k}"] is not None])
        ra = mean([r[f"recall_all@{k}"]    for r in subset
                   if r[f"recall_all@{k}"]    is not None])
        hr = mean([1.0 if (r[f"recall_corpus@{k}"] or 0) > 0 else 0.0
                   for r in has_corpus]) if has_corpus else None
        def fmt(v):
            return f"{v:.4f}" if v is not None else "  N/A "
        print(f"  {k:>3} | {fmt(rc):>13} | {fmt(ra):>10} | {fmt(hr):>15}")


def print_law_distribution(records: list[dict]) -> None:
    from collections import Counter
    gold_cnt = Counter()
    retr_cnt = Counter()
    for r in records:
        for a in r.get("gold_all", []):
            gold_cnt[law_prefix(a)] += 1
        for a in r.get("retrieved_top50", []):
            retr_cnt[law_prefix(a)] += 1

    print("\n--- Laws in gold ---")
    for law, cnt in gold_cnt.most_common():
        tag = " ← INDEXED" if law in INDEXED_LAWS else ""
        print(f"  {law:<25} {cnt:>4}{tag}")

    print("\n--- Laws in retrieved_top50 ---")
    for law, cnt in retr_cnt.most_common():
        print(f"  {law:<25} {cnt:>5}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/retrieval_audit.json",
    )
    p.add_argument("--laws", action="store_true", help="In phân bố luật trong gold và retrieval")
    args = p.parse_args()

    if not args.input.exists():
        print(f"FAIL: không tìm thấy {args.input}", file=sys.stderr)
        print("Chạy scripts/retrieval_audit.py trước.", file=sys.stderr)
        return 1

    data    = json.loads(args.input.read_text(encoding="utf-8"))
    records = enrich(data["records"])

    # Phân loại
    pure_corpus = [r for r in records if r["n_gold_corpus"] > 0 and r["n_gold_corpus"] == r["n_gold_all"]]
    mixed       = [r for r in records if r["n_gold_corpus"] > 0 and r["n_gold_corpus"] <  r["n_gold_all"]]
    true_ooc    = [r for r in records if r["n_gold_corpus"] == 0]
    has_corpus  = [r for r in records if r["n_gold_corpus"] > 0]

    print("=" * 65)
    print("RETRIEVAL RECALL — corpus = {L41_2024, L58_2014, L45_2019}")
    print("=" * 65)
    print(f"\nPhân loại 200 câu hỏi:")
    print(f"  Pure corpus  (gold 100% indexed): {len(pure_corpus)}")
    print(f"  Mixed        (indexed + other):   {len(mixed)}")
    print(f"  True OOC     (0 indexed gold):    {len(true_ooc)}")

    print_table(f"ALL QUESTIONS (n=200)", records)
    print_table(f"HAS CORPUS GOLD (n={len(has_corpus)})", has_corpus)
    print_table(f"PURE CORPUS (n={len(pure_corpus)})", pure_corpus)
    print_table(f"MIXED (n={len(mixed)})", mixed)
    print_table(f"TRUE OOC (n={len(true_ooc)})", true_ooc)

    # v5 target analysis
    n_total = len(records)
    n_ooc   = len(true_ooc)
    n_corp  = len(has_corpus)
    target  = 0.70
    needed_corpus_recall = (target * n_total) / n_corp if n_corp else None
    r8_corp  = mean([r["recall_corpus@8"]  for r in has_corpus if r["recall_corpus@8"]  is not None])
    r50_corp = mean([r["recall_corpus@50"] for r in has_corpus if r["recall_corpus@50"] is not None])
    print(f"\n{'='*65}")
    print("v5 TARGET ANALYSIS")
    print(f"{'='*65}")
    print(f"  E2E target              : {target:.0%} citation_recall (200 câu)")
    print(f"  True OOC (luôn recall=0): {n_ooc} câu ({n_ooc/n_total:.0%})")
    print(f"  Corpus questions        : {n_corp} câu")
    if needed_corpus_recall:
        print(f"  Corpus recall cần đạt   : {needed_corpus_recall:.1%}  (để đạt {target:.0%} overall)")
    print(f"  Hiện tại recall@8       : {r8_corp:.1%}  (gap = {needed_corpus_recall - r8_corp:.1%})")
    print(f"  Dense-only ceiling @50  : {r50_corp:.1%}  (gap = {needed_corpus_recall - r50_corp:.1%})")
    print()

    if args.laws:
        print_law_distribution(records)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
