"""Chạy inference 2 arms (GraphRAG vs LLM-only) trên N câu hỏi.

Output mỗi arm 1 file JSON / câu hỏi:
    data/eval/results/graphrag/A{stt}.json
    data/eval/results/llm_only/A{stt}.json

Idempotent — skip nếu file đã tồn tại (dùng --force để chạy lại).
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

QUESTIONS_PATH = Path("data/eval/questions_200.json")
OUT_ROOT = Path("data/eval/results")
GRAPHRAG_DIR = OUT_ROOT / "graphrag"
LLM_ONLY_DIR = OUT_ROOT / "llm_only"


def load_questions(n: int | None = None) -> list[dict]:
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        qs = json.load(f)
    if n:
        qs = qs[:n]
    return qs


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_graphrag(questions: list[dict], force: bool, verbose: bool):
    from src.rag_query import RagPipeline

    pipeline = RagPipeline()
    # Pre-load embed model
    _ = pipeline.embed_model

    n_done, n_skipped, n_failed = 0, 0, 0
    t_total = time.time()
    try:
        for i, q in enumerate(questions, 1):
            stt = q["stt"]
            out_path = GRAPHRAG_DIR / f"A{stt}.json"
            if out_path.exists() and not force:
                n_skipped += 1
                continue

            try:
                result = pipeline.ask(q["question"], top_k=8, verbose=False)
                verified = pipeline.verify_citations(result.citation_ids)
                record = {
                    "arm": "graphrag",
                    "stt": stt,
                    "question": q["question"],
                    "answer": result.answer,
                    "citations": result.citations,
                    "citation_ids": result.citation_ids,
                    "citation_verified": verified,
                    "n_vector_hits": len(result.hits),
                    "vector_hits": [
                        {"clause_id": h.clause_id, "score": h.score, "text_preview": h.text[:200]}
                        for h in result.hits[:5]
                    ],
                    "n_semantic_edges": result.n_semantic_edges,
                    "n_refs": result.n_refs,
                    "elapsed_s": result.elapsed_s,
                    # Token usage not exposed by RagPipeline → re-call? Skip for now,
                    # cost ước tính từ phía dưới khi compute_metrics.
                    "gold_answer": q.get("gold_answer"),
                    "gold_citations_raw": q.get("gold_citations_raw"),
                }
                _save(out_path, record)
                n_done += 1
                if verbose or i % 10 == 0:
                    print(
                        f"  [graphrag {i:>3}/{len(questions)}] stt={stt} "
                        f"({result.elapsed_s:.1f}s, {len(result.citation_ids)} cits)",
                        flush=True,
                    )
            except Exception as e:
                n_failed += 1
                print(f"  ✗ [graphrag {stt}] {type(e).__name__}: {e}", file=sys.stderr)
                _save(
                    out_path.with_suffix(".error.json"),
                    {
                        "arm": "graphrag",
                        "stt": stt,
                        "error": f"{type(e).__name__}: {e}",
                    },
                )
    finally:
        pipeline.close()
    print(
        f"\nGraphRAG done: {n_done} new, {n_skipped} skipped, {n_failed} failed "
        f"({time.time() - t_total:.1f}s)"
    )


def run_llm_only(questions: list[dict], force: bool, verbose: bool):
    from experiments.llm_only import LlmOnlyPipeline

    pipeline = LlmOnlyPipeline()

    n_done, n_skipped, n_failed = 0, 0, 0
    t_total = time.time()
    try:
        for i, q in enumerate(questions, 1):
            stt = q["stt"]
            out_path = LLM_ONLY_DIR / f"A{stt}.json"
            if out_path.exists() and not force:
                n_skipped += 1
                continue
            try:
                result = pipeline.ask(q["question"])
                record = {
                    "arm": "llm_only",
                    "stt": stt,
                    "question": q["question"],
                    "answer": result.answer,
                    "citations": result.citations,
                    "citation_ids": result.citation_ids,
                    "elapsed_s": result.elapsed_s,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "gold_answer": q.get("gold_answer"),
                    "gold_citations_raw": q.get("gold_citations_raw"),
                }
                _save(out_path, record)
                n_done += 1
                if verbose or i % 10 == 0:
                    print(
                        f"  [llm_only {i:>3}/{len(questions)}] stt={stt} "
                        f"({result.elapsed_s:.1f}s, {len(result.citation_ids)} cits)",
                        flush=True,
                    )
            except Exception as e:
                n_failed += 1
                print(f"  ✗ [llm_only {stt}] {type(e).__name__}: {e}", file=sys.stderr)
                _save(
                    out_path.with_suffix(".error.json"),
                    {
                        "arm": "llm_only",
                        "stt": stt,
                        "error": f"{type(e).__name__}: {e}",
                    },
                )
    finally:
        pipeline.close()
    print(
        f"\nLlmOnly done: {n_done} new, {n_skipped} skipped, {n_failed} failed "
        f"({time.time() - t_total:.1f}s)"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200, help="Số câu đầu tiên (default 200).")
    p.add_argument("--arm", choices=["graphrag", "llm_only", "both"], default="both")
    p.add_argument("--force", action="store_true", help="Chạy lại dù file đã có.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    questions = load_questions(args.n)
    print(f"Loaded {len(questions)} questions từ {QUESTIONS_PATH}")

    if args.arm in ("graphrag", "both"):
        print("\n=== ARM A: GraphRAG ===")
        run_graphrag(questions, args.force, args.verbose)

    if args.arm in ("llm_only", "both"):
        print("\n=== ARM B: LLM-only ===")
        run_llm_only(questions, args.force, args.verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
