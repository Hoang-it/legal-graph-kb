"""Chạy inference 5 arms trên N câu hỏi.

Output mỗi arm 1 file JSON / câu hỏi:
    data/eval/results/{arm}/A{stt}.json
với arm ∈ {graphrag, llm_only, elite_no_retrieval, elite_ontology, elite_graphrag}

Idempotent — skip nếu file đã tồn tại (dùng --force để chạy lại).

CLI:
    python -m experiments.run_inference --arms graphrag,llm_only --n 200
    python -m experiments.run_inference --arms all --n 10
    python -m experiments.run_inference --arms elite_ontology,elite_graphrag --n 200
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

ALL_ARMS = (
    "graphrag",
    "llm_only",
    "elite_no_retrieval",
    "elite_ontology",
    "elite_graphrag",
    "elite_graphrag_logic",
)


def load_questions(n: int | None = None) -> list[dict]:
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        qs = json.load(f)
    if n:
        qs = qs[:n]
    return qs


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_arms(s: str) -> list[str]:
    if s == "all":
        return list(ALL_ARMS)
    arms = [a.strip() for a in s.split(",") if a.strip()]
    invalid = [a for a in arms if a not in ALL_ARMS]
    if invalid:
        raise SystemExit(
            f"Unknown arm(s): {invalid}. Valid: {list(ALL_ARMS)} or 'all'"
        )
    return arms


# ---------------------------------------------------------------------------
# Arm runners
# ---------------------------------------------------------------------------

def run_graphrag(questions: list[dict], force: bool, verbose: bool) -> None:
    from src.rag_query import RagPipeline

    arm = "graphrag"
    out_dir = OUT_ROOT / arm
    pipeline = RagPipeline()
    _ = pipeline.embed_model  # pre-load

    n_done, n_skipped, n_failed = 0, 0, 0
    t_total = time.time()
    try:
        for i, q in enumerate(questions, 1):
            stt = q["stt"]
            out_path = out_dir / f"A{stt}.json"
            if out_path.exists() and not force:
                n_skipped += 1
                continue
            try:
                result = pipeline.ask(q["question"], top_k=8, verbose=False)
                verified = pipeline.verify_citations(result.citation_ids)
                record = {
                    "arm": arm,
                    "stt": stt,
                    "question": q["question"],
                    "answer": result.answer,
                    "citations": result.citations,
                    "citation_ids": result.citation_ids,
                    "citation_verified": verified,
                    "n_vector_hits": len(result.hits),
                    "vector_hits": [
                        {"clause_id": h.clause_id, "score": h.score,
                         "text_preview": h.text[:200]}
                        for h in result.hits[:5]
                    ],
                    "n_semantic_edges": result.n_semantic_edges,
                    "n_refs": result.n_refs,
                    "elapsed_s": result.elapsed_s,
                    "gold_answer": q.get("gold_answer"),
                    "gold_citations_raw": q.get("gold_citations_raw"),
                }
                _save(out_path, record)
                n_done += 1
                if verbose or i % 10 == 0:
                    print(
                        f"  [{arm:<22} {i:>3}/{len(questions)}] stt={stt} "
                        f"({result.elapsed_s:.1f}s, {len(result.citation_ids)} cits)",
                        flush=True,
                    )
            except Exception as e:
                n_failed += 1
                print(f"  ✗ [{arm} {stt}] {type(e).__name__}: {e}", file=sys.stderr)
                _save(out_path.with_suffix(".error.json"),
                      {"arm": arm, "stt": stt, "error": f"{type(e).__name__}: {e}"})
    finally:
        pipeline.close()
    print(f"\n{arm} done: {n_done} new, {n_skipped} skipped, {n_failed} failed "
          f"({time.time() - t_total:.1f}s)")


def run_llm_only(questions: list[dict], force: bool, verbose: bool) -> None:
    from experiments.llm_only import LlmOnlyPipeline

    arm = "llm_only"
    out_dir = OUT_ROOT / arm
    pipeline = LlmOnlyPipeline()

    n_done, n_skipped, n_failed = 0, 0, 0
    t_total = time.time()
    try:
        for i, q in enumerate(questions, 1):
            stt = q["stt"]
            out_path = out_dir / f"A{stt}.json"
            if out_path.exists() and not force:
                n_skipped += 1
                continue
            try:
                result = pipeline.ask(q["question"])
                record = {
                    "arm": arm,
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
                        f"  [{arm:<22} {i:>3}/{len(questions)}] stt={stt} "
                        f"({result.elapsed_s:.1f}s, {len(result.citation_ids)} cits)",
                        flush=True,
                    )
            except Exception as e:
                n_failed += 1
                print(f"  ✗ [{arm} {stt}] {type(e).__name__}: {e}", file=sys.stderr)
                _save(out_path.with_suffix(".error.json"),
                      {"arm": arm, "stt": stt, "error": f"{type(e).__name__}: {e}"})
    finally:
        pipeline.close()
    print(f"\n{arm} done: {n_done} new, {n_skipped} skipped, {n_failed} failed "
          f"({time.time() - t_total:.1f}s)")


def _run_elite(
    arm: str,
    pipeline,
    questions: list[dict],
    force: bool,
    verbose: bool,
) -> None:
    """Generic runner cho 3 elite arms (cùng EliteAnswer dataclass)."""
    out_dir = OUT_ROOT / arm
    n_done, n_skipped, n_failed = 0, 0, 0
    t_total = time.time()
    try:
        for i, q in enumerate(questions, 1):
            stt = q["stt"]
            out_path = out_dir / f"A{stt}.json"
            if out_path.exists() and not force:
                n_skipped += 1
                continue
            try:
                ans = pipeline.ask(q["question"])
                record = {
                    "arm": arm,
                    "stt": stt,
                    "question": q["question"],
                    "answer": ans.answer,
                    "plain_answer": ans.plain_answer,  # NEW: prose form từ IRAC render
                    "citations": ans.citations,
                    "citation_ids": ans.citation_ids,
                    "citation_indices": ans.citation_indices,
                    # ---- Elite-specific metadata ----
                    "prolog_success": ans.prolog_success,
                    "prolog_status": ans.prolog_status,
                    "n_repair_rounds": ans.n_repair_rounds,
                    "prolog_trace": ans.prolog_trace,
                    "prolog_program": getattr(ans, "prolog_program", ""),
                    "prolog_error": getattr(ans, "prolog_error", ""),
                    "irac_sections": ans.irac_sections,
                    # ---- Standard fields ----
                    "elapsed_s": ans.elapsed_s,
                    "prompt_tokens": ans.prompt_tokens,
                    "completion_tokens": ans.completion_tokens,
                    "error": ans.error,
                    "gold_answer": q.get("gold_answer"),
                    "gold_citations_raw": q.get("gold_citations_raw"),
                }
                _save(out_path, record)
                n_done += 1
                if verbose or i % 5 == 0:
                    status = "✓" if ans.prolog_success else "✗"
                    print(
                        f"  [{arm:<22} {i:>3}/{len(questions)}] stt={stt} "
                        f"({ans.elapsed_s:.1f}s, repair={ans.n_repair_rounds}, "
                        f"{status} prolog={ans.prolog_status})",
                        flush=True,
                    )
            except Exception as e:
                n_failed += 1
                print(f"  ✗ [{arm} {stt}] {type(e).__name__}: {e}", file=sys.stderr)
                _save(out_path.with_suffix(".error.json"),
                      {"arm": arm, "stt": stt, "error": f"{type(e).__name__}: {e}"})
    finally:
        try:
            pipeline.close()
        except Exception:
            pass
    print(f"\n{arm} done: {n_done} new, {n_skipped} skipped, {n_failed} failed "
          f"({time.time() - t_total:.1f}s)")


def run_elite_no_retrieval(questions, force, verbose):
    from experiments.elite_pipelines import EliteNoRetrievalPipeline
    p = EliteNoRetrievalPipeline()
    _run_elite("elite_no_retrieval", p, questions, force, verbose)


def run_elite_ontology(questions, force, verbose):
    from experiments.elite_pipelines import EliteOntologyPipeline
    p = EliteOntologyPipeline()
    _run_elite("elite_ontology", p, questions, force, verbose)


def run_elite_graphrag(questions, force, verbose):
    from experiments.elite_pipelines import EliteGraphRAGPipeline
    p = EliteGraphRAGPipeline()  # tự tạo + warm up RagPipeline bên trong
    _run_elite("elite_graphrag", p, questions, force, verbose)


def run_elite_graphrag_logic(questions, force, verbose):
    from experiments.elite_pipelines import EliteGraphRAGLogicPipeline
    p = EliteGraphRAGLogicPipeline()
    _run_elite("elite_graphrag_logic", p, questions, force, verbose)


ARM_RUNNERS = {
    "graphrag": run_graphrag,
    "llm_only": run_llm_only,
    "elite_no_retrieval": run_elite_no_retrieval,
    "elite_ontology": run_elite_ontology,
    "elite_graphrag": run_elite_graphrag,
    "elite_graphrag_logic": run_elite_graphrag_logic,
}


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run inference cho N câu hỏi × các arm chỉ định."
    )
    p.add_argument("--n", type=int, default=200,
                   help="Số câu đầu tiên (default 200).")
    p.add_argument("--arms", type=str, default="all",
                   help=f"Comma-separated arms hoặc 'all'. Available: {', '.join(ALL_ARMS)}")
    p.add_argument("--force", action="store_true",
                   help="Chạy lại dù file đã có.")
    p.add_argument("--verbose", action="store_true")
    # Backward-compat: --arm (cũ) single-arm hoặc 'both'
    p.add_argument("--arm", type=str, default=None,
                   help="(Deprecated) Single arm hoặc 'both' (legacy graphrag+llm_only).")
    args = p.parse_args()

    # Resolve arms list
    if args.arm == "both":
        arms = ["graphrag", "llm_only"]
    elif args.arm:
        arms = [args.arm]
    else:
        arms = _parse_arms(args.arms)

    questions = load_questions(args.n)
    print(f"Loaded {len(questions)} questions từ {QUESTIONS_PATH}")
    print(f"Arms to run: {arms}")

    for arm in arms:
        runner = ARM_RUNNERS.get(arm)
        if runner is None:
            print(f"✗ Unknown arm: {arm}", file=sys.stderr)
            continue
        print(f"\n=== ARM: {arm} ===")
        runner(questions, args.force, args.verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
