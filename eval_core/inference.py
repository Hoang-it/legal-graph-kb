"""Chạy inference nhiều arms trên N câu hỏi.

Output mỗi arm 1 file JSON / câu hỏi:
    <results_root>/<arm>/A<stt>.json

Idempotent — skip nếu file đã tồn tại (dùng --force để chạy lại).

CLI:
    python -m eval_core.inference <experiment_path> [--arms ...] [--n ...] [--force]
    python -m eval_core run <experiment_path>                        # via eval_core.cli
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import suppress
from pathlib import Path

from dotenv import load_dotenv

from eval_core.arms import ALL_ARMS, parse_run_arms

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)


def load_questions(questions_path: Path, n: int | None = None) -> list[dict]:
    with Path(questions_path).open(encoding="utf-8") as f:
        qs = json.load(f)
    if n:
        qs = qs[:n]
    return qs


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_arms(s: str) -> list[str]:
    try:
        return parse_run_arms(s)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


# ---------------------------------------------------------------------------
# Arm runners — each writes to <results_root>/<arm>/
# ---------------------------------------------------------------------------


def run_graphrag(
    questions: list[dict],
    results_root: Path,
    force: bool,
    verbose: bool,
) -> None:
    from runtime.rag_query import RagPipeline

    arm = "graphrag"
    out_dir = results_root / arm
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


def run_llm_only(
    questions: list[dict],
    results_root: Path,
    force: bool,
    verbose: bool,
) -> None:
    from runtime.llm_only import LlmOnlyPipeline

    arm = "llm_only"
    out_dir = results_root / arm
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


def _run_logic_lm(
    arm: str,
    pipeline,
    questions: list[dict],
    results_root: Path,
    force: bool,
    verbose: bool,
) -> None:
    """Generic runner cho 3 logic-lm arms (cùng LogicLMAnswer dataclass)."""
    out_dir = results_root / arm
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
                    "hypothesis": getattr(ans, "hypothesis", ""),  # HyDE-semantic hypothesis (rỗng nếu arm không dùng)
                    "citations": ans.citations,
                    "citation_ids": ans.citation_ids,
                    "citation_indices": ans.citation_indices,
                    # ---- Logic-LM-specific metadata ----
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
        with suppress(Exception):
            pipeline.close()
    print(f"\n{arm} done: {n_done} new, {n_skipped} skipped, {n_failed} failed "
          f"({time.time() - t_total:.1f}s)")


def run_logic_lm_no_retrieval(questions, results_root, force, verbose):
    from runtime.logic_lm_pipelines import LogicLMNoRetrievalPipeline
    p = LogicLMNoRetrievalPipeline()
    _run_logic_lm("logic_lm_no_retrieval", p, questions, results_root, force, verbose)


def run_logic_lm_ontology(questions, results_root, force, verbose):
    from runtime.logic_lm_pipelines import LogicLMOntologyPipeline
    p = LogicLMOntologyPipeline()
    _run_logic_lm("logic_lm_ontology", p, questions, results_root, force, verbose)


def run_logic_lm_graphrag(questions, results_root, force, verbose):
    from runtime.logic_lm_pipelines import LogicLMGraphRAGPipeline
    p = LogicLMGraphRAGPipeline()  # tự tạo + warm up RagPipeline bên trong
    _run_logic_lm("logic_lm_graphrag", p, questions, results_root, force, verbose)


def run_logic_lm_hyde_semantic(questions, results_root, force, verbose):
    from runtime.logic_lm_pipelines import LogicLMHydeSemanticPipeline
    p = LogicLMHydeSemanticPipeline()  # dense_hyde_semantic retrieval + hypothesis vào rule-gen
    _run_logic_lm("logic_lm_hyde_semantic", p, questions, results_root, force, verbose)


def run_logic_lm_hyde_semantic_nohyp(questions, results_root, force, verbose):
    from runtime.logic_lm_pipelines import LogicLMHydeSemanticNoHypPipeline
    p = LogicLMHydeSemanticNoHypPipeline()  # control: cùng retrieval, không hypothesis
    _run_logic_lm("logic_lm_hyde_semantic_nohyp", p, questions, results_root, force, verbose)


def run_qa_hyde_semantic(
    questions: list[dict],
    results_root: Path,
    force: bool,
    verbose: bool,
) -> None:
    """dense_hyde_semantic retrieval → direct generation (no logic-LM)."""
    from runtime.qa_hyde_semantic import QAHydeSemanticPipeline

    arm = "qa_hyde_semantic"
    out_dir = results_root / arm
    pipeline = QAHydeSemanticPipeline()  # warms BGE-M3 in ctor

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
                verified = pipeline.verify_citations(result.citation_ids)
                record = {
                    "arm": arm,
                    "stt": stt,
                    "question": q["question"],
                    "answer": result.answer,
                    "citations": result.citations,
                    "citation_ids": result.citation_ids,
                    "citation_verified": verified,
                    "n_final_hits": result.n_final,
                    "hits": result.hits[:5],
                    "elapsed_s": result.elapsed_s,
                    "elapsed_breakdown": result.elapsed_breakdown,
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


def run_graphrag_v5(
    questions: list[dict],
    results_root: Path,
    force: bool,
    verbose: bool,
) -> None:
    """Plan v5 Sprint 1 vanilla pipeline (BGE-M3 dense + BM25 + RRF + CE rerank + REFERS_TO expand)."""
    from src.retrieval import V5RetrievalPipeline

    arm = "graphrag_v5"
    out_dir = results_root / arm
    pipeline = V5RetrievalPipeline()
    # warm up local models so per-question timings are clean
    _ = pipeline.embed_model
    _ = pipeline.reranker.model

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
                verified = pipeline.verify_citations(result.citation_ids)
                record = {
                    "arm": arm,
                    "stt": stt,
                    "question": q["question"],
                    "answer": result.answer,
                    "citations": result.citations,
                    "citation_ids": result.citation_ids,
                    "citation_verified": verified,
                    "n_final_hits": result.n_final,
                    "n_seeds": result.n_seeds,
                    "n_neighbors_added": result.n_neighbors_added,
                    "retrieval_audit": result.retrieval_audit,
                    "hits": result.hits,
                    "elapsed_s": result.elapsed_s,
                    "elapsed_breakdown": result.elapsed_breakdown,
                    "gold_answer": q.get("gold_answer"),
                    "gold_citations_raw": q.get("gold_citations_raw"),
                }
                _save(out_path, record)
                n_done += 1
                if verbose or i % 5 == 0:
                    print(
                        f"  [{arm:<14} {i:>3}/{len(questions)}] stt={stt} "
                        f"({result.elapsed_s:.1f}s, seeds={result.n_seeds}, "
                        f"+neigh={result.n_neighbors_added}, cits={len(result.citation_ids)})",
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


def run_graphrag_v5_m2(
    questions: list[dict],
    results_root: Path,
    force: bool,
    verbose: bool,
) -> None:
    """v5 Sprint 2 M2 arm: fine-tuned BGE-M3 + bge-reranker-base + tuned index.

    Identical logic to ``run_graphrag_v5`` except the arm name (so records go to
    ``results/graphrag_v5_m2/``) and the pipeline is constructed with the M2
    swap points (adapter, tuned dense index, base reranker). Each swap point
    falls back to its env var when constructor args are None, so the same
    pipeline class supports both vanilla and tuned modes.
    """
    from src.retrieval import V5RetrievalPipeline

    arm = "graphrag_v5_m2"
    out_dir = results_root / arm
    pipeline = V5RetrievalPipeline(
        adapter_path="models/bge-m3-bhxh-lora",
        dense_index="clause_vec_tuned",
        reranker_model="BAAI/bge-reranker-base",
    )
    _ = pipeline.embed_model
    _ = pipeline.reranker.model

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
                verified = pipeline.verify_citations(result.citation_ids)
                record = {
                    "arm": arm,
                    "stt": stt,
                    "question": q["question"],
                    "answer": result.answer,
                    "citations": result.citations,
                    "citation_ids": result.citation_ids,
                    "citation_verified": verified,
                    "n_final_hits": result.n_final,
                    "n_seeds": result.n_seeds,
                    "n_neighbors_added": result.n_neighbors_added,
                    "retrieval_audit": result.retrieval_audit,
                    "hits": result.hits,
                    "elapsed_s": result.elapsed_s,
                    "elapsed_breakdown": result.elapsed_breakdown,
                    "gold_answer": q.get("gold_answer"),
                    "gold_citations_raw": q.get("gold_citations_raw"),
                }
                _save(out_path, record)
                n_done += 1
                if verbose or i % 5 == 0:
                    print(
                        f"  [{arm:<16} {i:>3}/{len(questions)}] stt={stt} "
                        f"({result.elapsed_s:.1f}s, seeds={result.n_seeds}, "
                        f"+neigh={result.n_neighbors_added}, cits={len(result.citation_ids)})",
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


def run_graphrag_cypher(
    questions: list[dict],
    results_root: Path,
    force: bool,
    verbose: bool,
) -> None:
    """Hybrid vector-seed → LLM-Cypher → fallback GraphRAG arm.

    Each record captures full Cypher provenance: the rounds it took, the
    validation/execution errors at each round, whether a non-empty Cypher
    result was reached (``cypher_used``) and whether the answer ultimately
    fell back to vanilla vector+expand context (``fallback_used``).
    """
    from runtime.graphrag_cypher import GraphRagCypherPipeline

    arm = "graphrag_cypher"
    out_dir = results_root / arm
    pipeline = GraphRagCypherPipeline()
    _ = pipeline.embed_model  # pre-load

    n_done, n_skipped, n_failed = 0, 0, 0
    n_cypher_success, n_fallback = 0, 0
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
                verified = pipeline.rag.verify_citations(ans.citation_ids)
                record = {
                    "arm": arm,
                    "stt": stt,
                    "question": q["question"],
                    "answer": ans.answer,
                    "citations": ans.citations,
                    "citation_ids": ans.citation_ids,
                    "citation_verified": verified,
                    "n_vector_hits": len(ans.vector_hits),
                    "vector_hits": [
                        {"clause_id": h.clause_id, "score": h.score,
                         "text_preview": h.text[:200]}
                        for h in ans.vector_hits[:5]
                    ],
                    "cypher_used": ans.cypher_used,
                    "fallback_used": ans.fallback_used,
                    "cypher_rows_count": len(ans.cypher_rows),
                    "cypher_clause_ids_added": ans.cypher_clause_ids_added,
                    "cypher_attempts": [
                        {
                            "round": a.round,
                            "cypher": a.cypher,
                            "rationale": a.rationale,
                            "valid": a.valid,
                            "validation_error": a.validation_error,
                            "executed": a.executed,
                            "execution_error": a.execution_error,
                            "n_rows": a.n_rows,
                            "rows_preview": a.rows_preview,
                        }
                        for a in ans.cypher_attempts
                    ],
                    "cypher_rows": ans.cypher_rows,
                    "elapsed_s": ans.elapsed_s,
                    "elapsed_breakdown": ans.elapsed_breakdown,
                    "prompt_tokens": ans.prompt_tokens,
                    "completion_tokens": ans.completion_tokens,
                    "gold_answer": q.get("gold_answer"),
                    "gold_citations_raw": q.get("gold_citations_raw"),
                }
                _save(out_path, record)
                n_done += 1
                if ans.cypher_used:
                    n_cypher_success += 1
                if ans.fallback_used:
                    n_fallback += 1
                if verbose or i % 5 == 0:
                    tag = "GRAPH" if ans.cypher_used else "fallback"
                    print(
                        f"  [{arm:<16} {i:>3}/{len(questions)}] stt={stt} "
                        f"({ans.elapsed_s:.1f}s, {tag}, "
                        f"rounds={len(ans.cypher_attempts)}, "
                        f"+cls={len(ans.cypher_clause_ids_added)}, "
                        f"cits={len(ans.citation_ids)})",
                        flush=True,
                    )
            except Exception as e:
                n_failed += 1
                print(f"  ✗ [{arm} {stt}] {type(e).__name__}: {e}", file=sys.stderr)
                _save(out_path.with_suffix(".error.json"),
                      {"arm": arm, "stt": stt, "error": f"{type(e).__name__}: {e}"})
    finally:
        pipeline.close()
    print(
        f"\n{arm} done: {n_done} new, {n_skipped} skipped, {n_failed} failed "
        f"(cypher_used={n_cypher_success}, fallback={n_fallback}, "
        f"{time.time() - t_total:.1f}s)"
    )


ARM_RUNNERS = {
    "graphrag": run_graphrag,
    "llm_only": run_llm_only,
    "logic_lm_no_retrieval": run_logic_lm_no_retrieval,
    "logic_lm_ontology": run_logic_lm_ontology,
    "logic_lm_graphrag": run_logic_lm_graphrag,
    "logic_lm_hyde_semantic": run_logic_lm_hyde_semantic,
    "logic_lm_hyde_semantic_nohyp": run_logic_lm_hyde_semantic_nohyp,
    "qa_hyde_semantic": run_qa_hyde_semantic,
    "graphrag_v5": run_graphrag_v5,
    "graphrag_v5_m2": run_graphrag_v5_m2,
    "graphrag_cypher": run_graphrag_cypher,
}


# ---------------------------------------------------------------------------
# Experiment-aware entry point
# ---------------------------------------------------------------------------


def run_experiment(
    experiment,
    arms: list[str] | None = None,
    force: bool = False,
    verbose: bool = False,
) -> None:
    """Run inference for an :class:`eval_core.experiment.Experiment`.

    - ``arms``: optional override of which arms to run. If ``None``, runs
      every arm with ``mode=run`` in the experiment config.
    - Inherits arms (``mode=inherit``) are skipped — their records come
      from the parent and don't need to be re-generated.
    - Sets ``LEGAL_KG_PROMPTS_DIR`` to ``experiment.prompts_override_dir``
      for the duration of the run when one is configured.
    """
    from eval_core.experiment import Experiment

    if not isinstance(experiment, Experiment):
        raise TypeError(f"expected Experiment, got {type(experiment).__name__}")

    experiment.validate()
    runnable_arms = [
        name for name, spec in experiment.arms.items() if spec.mode == "run"
    ]
    if arms is None:
        arms = runnable_arms
    else:
        unknown = [a for a in arms if a not in runnable_arms]
        if unknown:
            raise ValueError(
                f"arms {unknown} are not declared with mode=run in {experiment.name!r}"
            )

    questions = load_questions(experiment.dataset.questions, experiment.dataset.n)
    print(f"Loaded {len(questions)} questions from {experiment.dataset.questions}")
    print(f"Arms to run: {arms}")
    print(f"Results dir: {experiment.results_dir}")

    override_dir = experiment.prompts_override_dir
    previous_override = os.environ.get("LEGAL_KG_PROMPTS_DIR")
    if override_dir is not None:
        os.environ["LEGAL_KG_PROMPTS_DIR"] = str(override_dir)
        print(f"Prompt override dir: {override_dir}")

    try:
        results_root = experiment.results_dir
        results_root.mkdir(parents=True, exist_ok=True)
        for arm in arms:
            runner = ARM_RUNNERS.get(arm)
            if runner is None:
                print(f"✗ Unknown arm: {arm}", file=sys.stderr)
                continue
            print(f"\n=== ARM: {arm} ===")
            runner(questions, results_root, force, verbose)
    finally:
        if override_dir is not None:
            if previous_override is None:
                os.environ.pop("LEGAL_KG_PROMPTS_DIR", None)
            else:
                os.environ["LEGAL_KG_PROMPTS_DIR"] = previous_override


def main() -> int:
    from eval_core.experiment import Experiment

    p = argparse.ArgumentParser(
        description="Run inference for an experiment folder."
    )
    p.add_argument("experiment", type=Path, help="Path to experiment folder.")
    p.add_argument(
        "--arms",
        type=str,
        default=None,
        help=(
            "Comma-separated arms to run (subset of arms with mode=run). "
            "Default = every mode=run arm in the experiment config. "
            f"Available: {', '.join(ALL_ARMS)}"
        ),
    )
    p.add_argument("--force", action="store_true",
                   help="Re-run even if record file already exists.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    try:
        experiment = Experiment.from_path(args.experiment)
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    arms = None
    if args.arms:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    try:
        run_experiment(experiment, arms=arms, force=args.force, verbose=args.verbose)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
