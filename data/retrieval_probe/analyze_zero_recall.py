"""Phân tích 71 câu zero-recall từ retrieval probe.

Với mỗi câu gold-in-graph nhưng top-20 miss hoàn toàn, script này:
1. Encode query thật (BGE-M3 + "query: " prefix)
2. Lấy embedding của gold clause(s) từ Neo4j
3. Tính query-gold cosine similarity trực tiếp
4. So với top-1 retrieved clause score → đo "khoảng cách" thật

Failure mode classification:
  A (embedding_far): query-gold cosine < 0.55
     → embedding không hiểu semantic của query/clause
  B (ranking_issue): query-gold 0.55–0.70, rank > 20
     → embedding OK nhưng nhiều clause khác score cao hơn
  C (corpus_crowding): query-gold > 0.70, rank > 20
     → gold bị "đẩy" bởi quá nhiều clause gần nhau trong corpus
  D (gold_clause_missing): gold article không có Clause trong graph
     → lỗi indexing / schema mismatch (không nên xảy ra)

Chạy từ repo root:
    python data/retrieval_probe/analyze_zero_recall.py [--results path/to/results.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

import numpy as np
from runtime.rag_query import RagPipeline, EMBED_MODEL, EMBED_DEVICE

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")
DB = os.getenv("NEO4J_DATABASE", "neo4j")

RESULTS_DIR = _REPO_ROOT / "data/retrieval_probe/results"


def latest_results() -> Path:
    files = sorted(RESULTS_DIR.glob("results_*.json"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy results_*.json trong {RESULTS_DIR}")
    return files[-1]


def fetch_clause_embeddings(driver, article_ids: list[str]) -> dict[str, tuple[str, list[float]]]:
    """Lấy tất cả Clause thuộc các article_ids, trả về {clause_id: (clause_text, embedding)}."""
    result = {}
    with driver.session(database=DB) as s:
        rows = s.run(
            """
            MATCH (a:Article)-[:HAS_CLAUSE]->(c:Clause)
            WHERE a.id IN $aids AND c.embedding IS NOT NULL
            RETURN c.id AS cid, c.text AS text, c.embedding AS emb
            """,
            aids=article_ids,
        ).data()
    for r in rows:
        result[r["cid"]] = (r["text"], r["emb"])
    return result


def classify_failure(query_gold_cosine: float | None, top1_score: float | None) -> str:
    if query_gold_cosine is None:
        return "D_gold_clause_missing"
    if query_gold_cosine < 0.55:
        return "A_embedding_far"
    if query_gold_cosine < 0.70:
        return "B_ranking_issue"
    return "C_corpus_crowding"


def run_analysis(results_path: Path, out_path: Path) -> None:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    zero_qs = data.get("zero_recall_questions", [])
    all_records = {r["stt"]: r for r in data.get("records", [])}

    if not zero_qs:
        print("Không có câu zero-recall trong file này.")
        return

    print(f"Phân tích {len(zero_qs)} câu zero-recall từ {results_path.name}", file=sys.stderr)

    pipeline = RagPipeline()
    try:
        _ = pipeline.embed_model
    except Exception as exc:
        pipeline.close()
        raise RuntimeError(f"Không load được pipeline: {exc}")

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(URI, auth=(USER, PWD))
    driver.verify_connectivity()

    analysis_records = []
    fail_modes: dict[str, int] = {"A_embedding_far": 0, "B_ranking_issue": 0,
                                   "C_corpus_crowding": 0, "D_gold_clause_missing": 0}

    for q in zero_qs:
        stt = q["stt"]
        question = q["question"]
        gold_articles = q["gold_in_graph"]
        top5 = q.get("top_5_retrieved", [])

        # Encode query (với "query: " prefix như production)
        q_emb = np.array(
            pipeline.embed_model.encode(
                ["query: " + question], normalize_embeddings=True, show_progress_bar=False
            )[0]
        )

        # Lấy clause embeddings cho gold articles
        clause_data = fetch_clause_embeddings(driver, gold_articles)

        # Tính query-gold cosine cho từng gold clause
        gold_sims: list[tuple[str, float, str]] = []
        for cid, (ctext, cemb) in clause_data.items():
            sim = float(np.dot(q_emb, np.array(cemb, dtype=np.float32)))
            gold_sims.append((cid, sim, ctext[:80]))

        gold_sims.sort(key=lambda x: -x[1])
        best_gold_sim = gold_sims[0][1] if gold_sims else None
        best_gold_clause = gold_sims[0][0] if gold_sims else None
        best_gold_text = gold_sims[0][2] if gold_sims else None

        top1_score = top5[0]["score"] if top5 else None
        top1_article = top5[0]["article_id"] if top5 else None

        mode = classify_failure(best_gold_sim, top1_score)
        fail_modes[mode] = fail_modes.get(mode, 0) + 1

        # Cần tìm rank thật của gold clause (search deeper nếu cần)
        # Để đơn giản: estimate từ score gap
        score_delta = (best_gold_sim - top1_score) if (best_gold_sim and top1_score) else None

        rec = {
            "stt": stt,
            "question": question[:120],
            "gold_articles": gold_articles,
            "best_gold_clause": best_gold_clause,
            "best_gold_clause_text": best_gold_text,
            "query_gold_cosine": round(best_gold_sim, 4) if best_gold_sim else None,
            "top1_retrieved_article": top1_article,
            "top1_retrieved_score": round(top1_score, 4) if top1_score else None,
            "score_delta_gold_minus_top1": round(score_delta, 4) if score_delta else None,
            "n_gold_clauses_found": len(clause_data),
            "failure_mode": mode,
            "all_gold_clause_sims": [
                {"clause_id": c, "cosine": round(s, 4), "text_preview": t}
                for c, s, t in gold_sims[:5]
            ],
        }
        analysis_records.append(rec)

    driver.close()
    pipeline.close()

    # Aggregate
    total = len(analysis_records)
    mode_desc = {
        "A_embedding_far":
            "Embedding xa (query-gold < 0.55): model không hiểu semantic",
        "B_ranking_issue":
            "Ranking issue (query-gold 0.55–0.70): nhiều clause khác score cao hơn",
        "C_corpus_crowding":
            "Corpus crowding (query-gold > 0.70): gold bị đẩy bởi clause gần nhau",
        "D_gold_clause_missing":
            "Gold clause thiếu embedding trong Neo4j",
    }

    gold_cosines = [r["query_gold_cosine"] for r in analysis_records if r["query_gold_cosine"]]
    top1_scores = [r["top1_retrieved_score"] for r in analysis_records if r["top1_retrieved_score"]]
    deltas = [r["score_delta_gold_minus_top1"] for r in analysis_records if r["score_delta_gold_minus_top1"] is not None]

    summary = {
        "results_source": str(results_path),
        "n_zero_recall": total,
        "failure_mode_counts": fail_modes,
        "query_gold_cosine_stats": {
            "mean": round(sum(gold_cosines) / len(gold_cosines), 4) if gold_cosines else None,
            "min": round(min(gold_cosines), 4) if gold_cosines else None,
            "max": round(max(gold_cosines), 4) if gold_cosines else None,
            "median": round(sorted(gold_cosines)[len(gold_cosines) // 2], 4) if gold_cosines else None,
            "pct_above_055": round(sum(1 for x in gold_cosines if x > 0.55) / len(gold_cosines), 3) if gold_cosines else None,
            "pct_above_070": round(sum(1 for x in gold_cosines if x > 0.70) / len(gold_cosines), 3) if gold_cosines else None,
        },
        "score_delta_stats": {
            "mean": round(sum(deltas) / len(deltas), 4) if deltas else None,
            "pct_negative": round(sum(1 for d in deltas if d < 0) / len(deltas), 3) if deltas else None,
        },
        "records": analysis_records,
    }

    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print report
    print(f"\n=== Phân tích {total} câu zero-recall ===\n")
    print("Failure mode breakdown:")
    for mode, cnt in sorted(fail_modes.items()):
        pct = cnt / total * 100
        desc = mode_desc.get(mode, "")
        print(f"  {mode:<25} {cnt:>3} ({pct:.0f}%)  — {desc}")

    print(f"\nQuery-gold cosine similarity:")
    s = summary["query_gold_cosine_stats"]
    print(f"  mean={s['mean']}  median={s['median']}  min={s['min']}  max={s['max']}")
    print(f"  % > 0.55 (có semantic signal): {s['pct_above_055']*100:.0f}%")
    print(f"  % > 0.70 (gần, ranking issue): {s['pct_above_070']*100:.0f}%")

    print(f"\nScore delta (gold − top1):")
    sd = summary["score_delta_stats"]
    print(f"  mean delta = {sd['mean']}  (âm = gold score thấp hơn top1)")
    print(f"  % âm: {sd['pct_negative']*100:.0f}% (gold luôn bị top1 vượt qua)")

    print(f"\nWorst cases (embedding_far, query-gold < 0.50):")
    worst = [r for r in analysis_records
             if r["query_gold_cosine"] and r["query_gold_cosine"] < 0.50]
    for r in sorted(worst, key=lambda x: x["query_gold_cosine"])[:5]:
        print(f"  Q{r['stt']} cosine={r['query_gold_cosine']}  gold={r['best_gold_clause']}")
        print(f"    Q: {r['question'][:90]}")
        print(f"    Gold: {r['best_gold_clause_text']}")

    print(f"\nOutput: {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path đến results_*.json (mặc định: file mới nhất)",
    )
    args = p.parse_args()

    results_path = args.results or latest_results()
    out_path = RESULTS_DIR / f"zero_recall_analysis_{results_path.stem.replace('results_', '')}.json"

    if not results_path.exists():
        print(f"FAIL: không tìm thấy {results_path}", file=sys.stderr)
        return 1

    try:
        run_analysis(results_path, out_path)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
