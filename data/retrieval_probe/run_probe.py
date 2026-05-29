"""Layer 2 — retrieval quality probe.

Mục đích: Đo xem vector search trong Neo4j có trả về đúng clause không,
dùng 200 câu hỏi có gold citation làm ground truth.

Không mock. Không bịa. Chạy thật với Neo4j live.

Metrics cố ý đo những VẤN ĐỀ của data, không để đẹp số:
  - graph_coverage_rate: % câu hỏi mà gold citation nằm trong graph của mình.
    Dự kiến thấp vì nhiều câu dẫn Nghị định / Thông tư chưa được load.
  - recall@k (conditioned): với câu hỏi CÓ gold trong graph, top-k có lấy đúng không?
    Nếu thấp → embedding hoặc cấu trúc graph có vấn đề.
  - mrr_in_graph: hạng trung bình của hit đúng đầu tiên.
  - zero_recall_at_20_in_graph: số câu gold trong graph nhưng top-20 vẫn miss hoàn toàn.
    Đây là list debug quan trọng nhất để cải thiện graph.
  - score_gap: cosine score của hit ĐÚNG vs hit SAI. Nếu gần nhau → embedding không phân biệt được.

Chạy từ repo root:
    python data/retrieval_probe/run_probe.py
    python data/retrieval_probe/run_probe.py --top-k 20

Output:
    data/retrieval_probe/results/results_<ts>.json
    data/retrieval_probe/results/report_<ts>.md
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Repo root importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

# Defensive: empty OPENAI_BASE_URL làm SDK fail
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

from src.citations import DEFAULT_REGISTRY_PATH, load_registry, parse_gold_citations_raw
from runtime.rag_query import RagPipeline

# Các source đã được load vào Neo4j graph của mình
GRAPH_SOURCES: frozenset[str] = frozenset({"L41_2024", "L58_2014", "L45_2019"})

QUESTIONS_PATH = _REPO_ROOT / "data/eval/questions_200.json"
OUT_DIR = _REPO_ROOT / "data/retrieval_probe/results"

# Top-k mặc định cao để đo recall đầy đủ
DEFAULT_TOP_K = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def _safe_stdev(values: list[float]) -> float | None:
    return round(statistics.stdev(values), 4) if len(values) > 1 else None


def _source_of(article_id: str) -> str:
    """'L41_2024.A64' → 'L41_2024'"""
    return article_id.split(".")[0]


def _in_graph(article_id: str) -> bool:
    return _source_of(article_id) in GRAPH_SOURCES


# ---------------------------------------------------------------------------
# Core probe
# ---------------------------------------------------------------------------


def run_probe(top_k: int, questions_path: Path) -> dict:
    registry = load_registry(DEFAULT_REGISTRY_PATH)
    questions: list[dict] = json.loads(questions_path.read_text(encoding="utf-8"))
    n_total = len(questions)

    print(f"[probe] {n_total} questions, top_k={top_k}", file=sys.stderr)
    print("[probe] Khởi động pipeline (embed model + Neo4j)...", file=sys.stderr)

    pipeline = RagPipeline()
    try:
        _ = pipeline.embed_model  # warm-up — tải model ngay, không lazy trong loop
    except Exception as exc:
        pipeline.close()
        raise RuntimeError(f"Không khởi động được RagPipeline: {exc}") from exc

    records: list[dict] = []
    skipped_parse: list[dict] = []

    for idx, q in enumerate(questions, 1):
        stt = q["stt"]
        question_text = q["question"]
        raw = (q.get("gold_citations_raw") or "").strip()

        # --- Parse gold citations ---
        parse_result = parse_gold_citations_raw(raw, registry)
        if not parse_result.refs:
            # Không parse được gì → skip, ghi lại lý do
            skipped_parse.append({
                "stt": stt,
                "question": question_text[:100],
                "gold_citations_raw": raw[:120],
                "errors": [
                    {"type": e.error_type, "segment": e.text[:80], "detail": e.detail[:120]}
                    for e in parse_result.errors
                ],
            })
            continue

        gold_all: list[str] = sorted({ref.article_id for ref in parse_result.refs})
        gold_in_graph: list[str] = [a for a in gold_all if _in_graph(a)]
        gold_out_graph: list[str] = [a for a in gold_all if not _in_graph(a)]
        gold_set = frozenset(gold_in_graph)

        # --- Vector search ---
        t0 = time.perf_counter()
        try:
            hits = pipeline.vector_search(question_text, top_k=top_k)
        except Exception as exc:
            print(f"  [q{stt}] vector_search lỗi: {exc}", file=sys.stderr)
            continue
        elapsed_s = round(time.perf_counter() - t0, 3)

        # Map hit → article (SearchHit đã có .article_id từ Cypher query)
        retrieved: list[dict] = [
            {
                "rank": rank,
                "clause_id": h.clause_id,
                "article_id": h.article_id,
                "score": round(h.score, 5),
                "is_relevant": h.article_id in gold_set,
            }
            for rank, h in enumerate(hits, 1)
        ]

        # --- Recall@k ---
        recall_at: dict[int, int] = {}
        for k in (1, 3, 5, 10, 15, 20):
            if k > top_k:
                continue
            top_articles = {r["article_id"] for r in retrieved[:k]}
            recall_at[k] = 1 if (top_articles & gold_set) else 0

        # --- Rank của hit đúng đầu tiên ---
        first_hit: dict | None = next((r for r in retrieved if r["is_relevant"]), None)
        first_relevant_rank = first_hit["rank"] if first_hit else None
        first_relevant_article = first_hit["article_id"] if first_hit else None
        first_relevant_score = first_hit["score"] if first_hit else None

        # --- Phân phối score: relevant vs irrelevant ---
        relevant_scores = [r["score"] for r in retrieved if r["is_relevant"]]
        irrelevant_scores = [r["score"] for r in retrieved if not r["is_relevant"]]

        records.append(
            {
                "stt": stt,
                "question": question_text,
                "gold_all": gold_all,
                "gold_in_graph": gold_in_graph,
                "gold_out_graph": gold_out_graph,
                "has_in_graph_gold": bool(gold_in_graph),
                "n_gold_total": len(gold_all),
                "n_gold_in_graph": len(gold_in_graph),
                "n_gold_out_graph": len(gold_out_graph),
                "recall_at": {str(k): v for k, v in recall_at.items()},
                "first_relevant_rank": first_relevant_rank,
                "first_relevant_article": first_relevant_article,
                "first_relevant_score": first_relevant_score,
                "relevant_scores": relevant_scores,
                "irrelevant_scores": irrelevant_scores,
                "top_5_retrieved": [
                    {
                        "rank": r["rank"],
                        "article_id": r["article_id"],
                        "clause_id": r["clause_id"],
                        "score": r["score"],
                        "is_relevant": r["is_relevant"],
                    }
                    for r in retrieved[:5]
                ],
                "retrieval_elapsed_s": elapsed_s,
                "parse_warnings": [
                    {"type": e.error_type, "segment": e.text[:80]}
                    for e in parse_result.errors
                ],
            }
        )

        if idx % 25 == 0 or idx == n_total:
            print(f"  [{idx}/{n_total}] processed", file=sys.stderr)

    pipeline.close()

    # ---------------------------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------------------------
    in_graph_recs = [r for r in records if r["has_in_graph_gold"]]
    out_graph_recs = [r for r in records if not r["has_in_graph_gold"]]

    def recall_mean(recs: list[dict], k: int) -> float | None:
        if not recs:
            return None
        vals = [r["recall_at"].get(str(k), 0) for r in recs]
        return round(sum(vals) / len(vals), 4)

    def mrr(recs: list[dict]) -> float | None:
        if not recs:
            return None
        rr_vals = [
            (1.0 / r["first_relevant_rank"]) if r["first_relevant_rank"] else 0.0
            for r in recs
        ]
        return round(statistics.mean(rr_vals), 4)

    ks = [k for k in (1, 3, 5, 10, 15, 20) if k <= top_k]

    # Score stats — chỉ tính cho câu có in-graph gold (để biết embedding có phân biệt không)
    all_rel_scores: list[float] = []
    all_irrel_scores: list[float] = []
    for r in in_graph_recs:
        all_rel_scores.extend(r["relevant_scores"])
        all_irrel_scores.extend(r["irrelevant_scores"])

    rel_mean = _safe_mean(all_rel_scores)
    irrel_mean = _safe_mean(all_irrel_scores)
    score_gap = round(rel_mean - irrel_mean, 4) if (rel_mean and irrel_mean) else None

    # Gold source distribution (để xem nguồn nào thiếu trong graph)
    source_gold_counts: dict[str, int] = {}
    for r in records:
        for a in r["gold_all"]:
            src = _source_of(a)
            source_gold_counts[src] = source_gold_counts.get(src, 0) + 1
    source_gold_counts = dict(sorted(source_gold_counts.items(), key=lambda x: -x[1]))

    # Câu fail hoàn toàn: có gold trong graph nhưng recall@20 = 0
    zero_recall_recs = [
        r for r in in_graph_recs if r["recall_at"].get(str(min(20, top_k)), 0) == 0
    ]

    aggregate = {
        "n_questions_total": n_total,
        "n_questions_parsed": len(records),
        "n_questions_skipped_parse": len(skipped_parse),
        "n_in_graph_questions": len(in_graph_recs),
        "n_out_graph_questions": len(out_graph_recs),
        "graph_coverage_rate": round(len(in_graph_recs) / len(records), 4) if records else None,
        # Recall tính theo ALL parsed questions → bức tranh thực tế đầy đủ
        "recall_all_parsed": {str(k): recall_mean(records, k) for k in ks},
        # Recall tính riêng cho câu CÓ gold trong graph → đo chất lượng retrieval thuần túy
        "recall_in_graph_conditioned": {str(k): recall_mean(in_graph_recs, k) for k in ks},
        # MRR
        "mrr_all_parsed": mrr(records),
        "mrr_in_graph": mrr(in_graph_recs),
        # Failure cases
        "n_zero_recall_at_topk_in_graph": len(zero_recall_recs),
        "zero_recall_rate_in_graph": (
            round(len(zero_recall_recs) / len(in_graph_recs), 4) if in_graph_recs else None
        ),
        # Score gap — nếu gần 0 → embedding không phân biệt được relevant/irrelevant
        "score_gap_relevant_minus_irrelevant": score_gap,
        "relevant_score_stats": {
            "mean": rel_mean,
            "stdev": _safe_stdev(all_rel_scores),
            "min": round(min(all_rel_scores), 4) if all_rel_scores else None,
            "max": round(max(all_rel_scores), 4) if all_rel_scores else None,
            "n": len(all_rel_scores),
        },
        "irrelevant_score_stats": {
            "mean": irrel_mean,
            "stdev": _safe_stdev(all_irrel_scores),
            "min": round(min(all_irrel_scores), 4) if all_irrel_scores else None,
            "max": round(max(all_irrel_scores), 4) if all_irrel_scores else None,
            "n": len(all_irrel_scores),
        },
        # Phân bổ gold theo nguồn → thấy ngay nguồn nào cần load vào graph
        "gold_source_distribution": source_gold_counts,
        "graph_sources_loaded": sorted(GRAPH_SOURCES),
    }

    return {
        "probe_version": "retrieval_probe_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "top_k": top_k,
            "questions_path": str(questions_path),
            "graph_sources": sorted(GRAPH_SOURCES),
        },
        "aggregate": aggregate,
        "zero_recall_questions": [
            {
                "stt": r["stt"],
                "question": r["question"][:150],
                "gold_in_graph": r["gold_in_graph"],
                "top_5_retrieved": r["top_5_retrieved"],
            }
            for r in zero_recall_recs
        ],
        "records": records,
        "skipped_parse": skipped_parse,
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_report(result: dict, path: Path) -> None:
    agg = result["aggregate"]
    cfg = result["config"]
    top_k = cfg["top_k"]

    def pct(v: float | None) -> str:
        return f"{v * 100:.1f}%" if v is not None else "N/A"

    def bar(v: float | None, width: int = 30) -> str:
        if v is None:
            return ""
        filled = round(v * width)
        return "█" * filled + "░" * (width - filled)

    lines: list[str] = []
    a = lines.append

    a("# Retrieval Probe — Báo cáo chất lượng graph cho RAG")
    a(f"Thời gian: {result['timestamp']}")
    a(f"Cấu hình: top_k={top_k}, graph_sources={cfg['graph_sources']}")
    a("")

    a("---")
    a("## 1. Tổng quan dataset")
    a(f"| Chỉ số | Giá trị |")
    a(f"|--------|---------|")
    a(f"| Tổng số câu hỏi | {agg['n_questions_total']} |")
    a(f"| Parse gold citations thành công | {agg['n_questions_parsed']} |")
    a(f"| Bị skip (không parse được gold) | {agg['n_questions_skipped_parse']} |")
    a(f"| Câu có ≥1 gold article **trong graph** | {agg['n_in_graph_questions']} ({pct(agg['graph_coverage_rate'])}) |")
    a(f"| Câu gold hoàn toàn **ngoài graph** | {agg['n_out_graph_questions']} ({pct(1 - (agg['graph_coverage_rate'] or 0))}) |")
    a("")
    a("> **Ý nghĩa:** `graph_coverage_rate` là trần lý thuyết của citation recall từ retrieval.")
    a("> Nếu 60% câu hỏi dẫn nguồn ngoài graph → end-to-end recall không thể vượt 0.60 dù retrieval hoàn hảo.")
    a("")

    a("---")
    a("## 2. Phân bổ gold citations theo nguồn luật")
    a("")
    a("```")
    a(f"{'Nguồn':<30} {'# citations':>12}  Tình trạng")
    a("-" * 60)
    for src, cnt in agg["gold_source_distribution"].items():
        status = "  [TRONG GRAPH]" if src in agg["graph_sources_loaded"] else "  [NGOÀI GRAPH — chưa load]"
        a(f"{src:<30} {cnt:>12}{status}")
    a("```")
    a("")
    a("> Các nguồn 'NGOÀI GRAPH' là nguyên nhân trực tiếp khiến recall thấp ở cấp độ dataset.")
    a("")

    a("---")
    a("## 3. Recall@k — Khả năng tìm đúng article trong top-k")
    a("")
    a("### 3a. Tính trên TẤT CẢ câu hỏi đã parse (bức tranh thực tế)")
    a("*(Denominator = toàn bộ câu parse được, kể cả câu gold ngoài graph)*")
    a("")
    a("```")
    for k, v in sorted(agg["recall_all_parsed"].items(), key=lambda x: int(x[0])):
        a(f"Recall@{k:<3}: {v:.4f}  {bar(v)}")
    a("```")
    a("")

    a("### 3b. Conditioned — Chỉ tính câu có gold TRONG graph (đo retrieval thuần túy)")
    a("*(Denominator = chỉ câu mà graph của mình có thể trả lời)*")
    a("")
    a("```")
    for k, v in sorted(agg["recall_in_graph_conditioned"].items(), key=lambda x: int(x[0])):
        a(f"Recall@{k:<3}: {v:.4f}  {bar(v)}")
    a("```")
    a("")
    a("> Nếu Recall@20 (conditioned) < 0.5 → embedding hoặc cấu trúc Clause trong graph chưa ổn.")
    a("> Nếu 0.5–0.7 → retrieval trung bình, còn room cải thiện chunking / embedding.")
    a("> Nếu > 0.8 → retrieval OK, bottleneck chính là graph coverage (load thêm nguồn).")
    a("")

    a("---")
    a("## 4. MRR — Hạng trung bình của hit đúng đầu tiên")
    a("")
    a(f"| Subset | MRR |")
    a(f"|--------|-----|")
    a(f"| Tất cả parsed | {agg['mrr_all_parsed']} |")
    a(f"| In-graph only | {agg['mrr_in_graph']} |")
    a("")
    a("> MRR = 1/rank. MRR=1.0 nghĩa là luôn hit rank 1. MRR=0.2 nghĩa là hit trung bình ở rank 5.")
    a("")

    a("---")
    a("## 5. Failure analysis — Câu có gold trong graph nhưng retrieval hoàn toàn thất bại")
    a("")
    zero_n = agg["n_zero_recall_at_topk_in_graph"]
    zero_rate = agg["zero_recall_rate_in_graph"]
    a(f"**{zero_n} / {agg['n_in_graph_questions']} câu in-graph ({pct(zero_rate)}) có Recall@{top_k} = 0**")
    a("")
    a("> Đây là danh sách debug quan trọng nhất: article tồn tại trong graph nhưng embedding")
    a("> không kéo về được. Các câu này tiết lộ vấn đề của graph (text quality, embedding, chunking).")
    a("")
    zero_qs = result.get("zero_recall_questions", [])
    if zero_qs:
        for r in zero_qs:
            a(f"**Q{r['stt']}:** {r['question'][:120]}")
            a(f"  - Gold in graph: `{', '.join(r['gold_in_graph'])}`")
            top5_arts = [h['article_id'] for h in r['top_5_retrieved']]
            top5_scores = [h['score'] for h in r['top_5_retrieved']]
            a(f"  - Top-5 retrieved: {list(zip(top5_arts, top5_scores))}")
            a("")
    else:
        a("*(Không có câu nào thất bại hoàn toàn — hoặc không có câu nào có in-graph gold)*")
    a("")

    a("---")
    a("## 6. Score gap — Embedding có phân biệt relevant / irrelevant không?")
    a("")
    rs = agg["relevant_score_stats"]
    irs = agg["irrelevant_score_stats"]
    gap = agg["score_gap_relevant_minus_irrelevant"]
    a(f"| | Mean | Stdev | Min | Max | N hits |")
    a(f"|--|------|-------|-----|-----|--------|")
    a(f"| Relevant hits | {rs['mean']} | {rs['stdev']} | {rs['min']} | {rs['max']} | {rs['n']} |")
    a(f"| Irrelevant hits | {irs['mean']} | {irs['stdev']} | {irs['min']} | {irs['max']} | {irs['n']} |")
    a(f"| **Score gap** | **{gap}** | | | | |")
    a("")
    a("> Score gap < 0.02 → embedding gần như không phân biệt được relevant/irrelevant.")
    a("> Score gap > 0.05 → embedding có discriminative power, vấn đề nằm ở chỗ khác.")
    a("")

    a("---")
    a("## 7. Chẩn đoán và hướng cải thiện")
    a("")
    cov = agg["graph_coverage_rate"] or 0
    recall20_cond = agg["recall_in_graph_conditioned"].get("20") or agg["recall_in_graph_conditioned"].get(str(top_k))
    a(f"**Graph coverage: {pct(cov)}** — {pct(1 - cov)} câu hỏi có gold hoàn toàn ngoài graph.")
    a("")
    if recall20_cond is not None:
        if recall20_cond < 0.4:
            a(f"**Retrieval (in-graph): {pct(recall20_cond)} tại k={top_k} — CRITICAL**")
            a("Embedding không kéo về đúng clause dù article tồn tại trong graph.")
            a("→ Kiểm tra: (1) chunk text có bị truncate không, (2) embedding model phù hợp không,")
            a("  (3) vector index đã build chưa, (4) query và clause text có semantic mismatch không.")
        elif recall20_cond < 0.65:
            a(f"**Retrieval (in-graph): {pct(recall20_cond)} tại k={top_k} — THẤP**")
            a("Retrieval trả sai nhiều clause dù graph có data. Cần cải thiện:")
            a("→ (1) Chunking: Clause quá ngắn / thiếu context từ Article cha.")
            a("→ (2) Embedding: text Clause cần prefix 'Luật BHXH 2024: ' để align với query style.")
            a("→ (3) Expansion: sau vector search cần multi-hop qua REFERENCES để kéo thêm article.")
        elif recall20_cond < 0.80:
            a(f"**Retrieval (in-graph): {pct(recall20_cond)} tại k={top_k} — TRUNG BÌNH**")
            a("Retrieval chấp nhận được nhưng còn room lớn. Tập trung vào:")
            a("→ (1) Re-ranking sau vector search để đẩy relevant clause lên top-5.")
            a("→ (2) Load thêm nguồn luật vào graph để tăng coverage.")
        else:
            a(f"**Retrieval (in-graph): {pct(recall20_cond)} tại k={top_k} — TỐT**")
            a("Retrieval OK. Bottleneck chính là graph coverage — cần load thêm Nghị định / Thông tư.")
    a("")
    a("**Ưu tiên tiếp theo theo thứ tự tác động:**")
    a("1. Load các nguồn NGOÀI GRAPH có nhiều gold citations nhất (xem bảng §2)")
    a("2. Fix failure cases §5 (zero-recall dù in-graph) — sửa text/embedding của clause")
    a("3. Tăng top-k hoặc thêm multi-hop expansion để cải thiện recall@5 → recall@10")
    a("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description="Retrieval probe — đo chất lượng vector search cho RAG."
    )
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Số clause top-k trả về")
    p.add_argument("--questions", type=Path, default=QUESTIONS_PATH, help="Path questions JSON")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Thư mục lưu kết quả")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = args.out_dir / f"results_{ts}.json"
    report_path = args.out_dir / f"report_{ts}.md"

    print(f"=== Retrieval Probe v1 (top_k={args.top_k}) ===", file=sys.stderr)
    t_start = time.perf_counter()

    try:
        result = run_probe(top_k=args.top_k, questions_path=args.questions)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    results_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(result, report_path)

    elapsed = round(time.perf_counter() - t_start, 1)

    # In summary ra stdout (không dùng stderr — user cần đọc được)
    agg = result["aggregate"]
    print(f"\n=== SUMMARY (elapsed {elapsed}s) ===")
    print(f"Graph coverage:       {agg['graph_coverage_rate']:.4f}  ({agg['n_in_graph_questions']}/{agg['n_questions_parsed']} câu có gold trong graph)")
    print(f"")
    print(f"Recall@k (ALL parsed):")
    for k, v in sorted(agg["recall_all_parsed"].items(), key=lambda x: int(x[0])):
        print(f"  @{k:<3}: {v:.4f}")
    print(f"")
    print(f"Recall@k (in-graph conditioned):")
    for k, v in sorted(agg["recall_in_graph_conditioned"].items(), key=lambda x: int(x[0])):
        print(f"  @{k:<3}: {v:.4f}")
    print(f"")
    print(f"MRR (in-graph):       {agg['mrr_in_graph']}")
    print(f"Score gap:            {agg['score_gap_relevant_minus_irrelevant']}")
    print(f"Zero-recall failures: {agg['n_zero_recall_at_topk_in_graph']} / {agg['n_in_graph_questions']}")
    print(f"")
    print(f"Results: {results_path}")
    print(f"Report:  {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
