"""Multi-model inference: chạy logic-lm arms × N OpenAI models trên dataset.

Output mỗi (arm, model) 1 thư mục:
    <results_root>/multimodel/{arm}__{model_safe}/A{stt}.json

Idempotent — skip nếu file đã tồn tại (--force để chạy lại).

Real API + real Prolog: dùng cùng OpenAILLMClient/SWI-Prolog như inference
arms thường, chỉ swap model parameter. Khi reasoning model reject
``temperature`` / ``response_format``, logic_lm_pipelines tự fallback
(xem ``_TokenTrackingLLMClient._chat_with_fallback``).

CLI:
    python -m eval_core.multimodel <experiment_path> \\
        [--models gpt-4.1,gpt-4o,...] \\
        [--arms logic_lm_no_retrieval,logic_lm_graphrag] \\
        [--n 1] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from eval_core import paths

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

ALL_ARMS = ("logic_lm_no_retrieval", "logic_lm_ontology", "logic_lm_graphrag")


def model_safe(model: str) -> str:
    """Chuyển model name về form an toàn cho filesystem.

    ``gpt-4.1`` -> ``gpt-4_1``   ``gpt-4o`` -> ``gpt-4o``   ``gpt-5-mini`` -> ``gpt-5-mini``
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model)


def combo_dir(results_root: Path, arm: str, model: str) -> Path:
    return results_root / paths.MULTIMODEL_SUBDIR / f"{arm}__{model_safe(model)}"


def load_questions(questions_path: Path, n: int | None = None) -> list[dict]:
    with Path(questions_path).open(encoding="utf-8") as f:
        qs = json.load(f)
    if n:
        qs = qs[:n]
    return qs


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_pipeline(arm: str, model: str, shared_rag=None):
    """Khởi tạo pipeline cho 1 (arm, model). shared_rag chỉ dùng cho logic_lm_graphrag."""
    from runtime.logic_lm_pipelines import (
        LogicLMGraphRAGPipeline,
        LogicLMNoRetrievalPipeline,
        LogicLMOntologyPipeline,
    )

    if arm == "logic_lm_no_retrieval":
        return LogicLMNoRetrievalPipeline(model=model)
    if arm == "logic_lm_ontology":
        return LogicLMOntologyPipeline(model=model)
    if arm == "logic_lm_graphrag":
        return LogicLMGraphRAGPipeline(model=model, rag_pipeline=shared_rag)
    raise ValueError(f"Unknown arm: {arm}")


def _run_combo(
    arm: str,
    model: str,
    questions: list[dict],
    results_root: Path,
    force: bool,
    verbose: bool,
    shared_rag=None,
) -> dict:
    """Chạy 1 (arm, model) trên tất cả questions. Trả về summary dict."""
    out_dir = combo_dir(results_root, arm, model)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"{arm} | {model}"

    pipeline = _make_pipeline(arm, model, shared_rag=shared_rag)

    n_done = n_skipped = n_failed = 0
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
                    "model": model,
                    "stt": stt,
                    "question": q["question"],
                    "answer": ans.answer,
                    "plain_answer": ans.plain_answer,  # NEW: prose form
                    "citations": ans.citations,
                    "citation_ids": ans.citation_ids,
                    "citation_indices": ans.citation_indices,
                    "prolog_success": ans.prolog_success,
                    "prolog_status": ans.prolog_status,
                    "n_repair_rounds": ans.n_repair_rounds,
                    "prolog_trace": ans.prolog_trace,
                    "irac_sections": ans.irac_sections,
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
                    status_ico = "OK" if ans.prolog_success else "X"
                    print(
                        f"  [{label:<35} {i:>3}/{len(questions)}] stt={stt} "
                        f"({ans.elapsed_s:.1f}s, repair={ans.n_repair_rounds}, "
                        f"{status_ico} prolog={ans.prolog_status})",
                        flush=True,
                    )
            except Exception as e:
                n_failed += 1
                print(f"  ! [{label} stt={stt}] {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                _save(out_path.with_suffix(".error.json"),
                      {"arm": arm, "model": model, "stt": stt,
                       "error": f"{type(e).__name__}: {e}"})
    finally:
        try:
            pipeline.close()
        except Exception:
            pass

    elapsed = time.time() - t_total
    print(
        f"== {label}: {n_done} new, {n_skipped} skipped, {n_failed} failed "
        f"({elapsed:.1f}s) ==",
        flush=True,
    )
    return {"arm": arm, "model": model, "n_done": n_done,
            "n_skipped": n_skipped, "n_failed": n_failed,
            "elapsed_s": round(elapsed, 1)}


# ---------------------------------------------------------------------------
# Experiment-aware entry point
# ---------------------------------------------------------------------------


def run_experiment_multimodel(
    experiment,
    arms: list[str] | None = None,
    models: list[str] | None = None,
    force: bool = False,
    verbose: bool = False,
) -> list[dict]:
    """Run the multimodel matrix for an :class:`eval_core.experiment.Experiment`.

    Reads ``experiment.multimodel`` from config (arms × models). Arguments
    override those defaults when provided.
    """
    from eval_core.experiment import Experiment

    if not isinstance(experiment, Experiment):
        raise TypeError(f"expected Experiment, got {type(experiment).__name__}")

    mm = experiment.multimodel
    arms_to_run = list(arms or (mm.arms if mm else ()))
    models_to_run = list(models or (mm.models if mm else ()))
    if not arms_to_run or not models_to_run:
        raise ValueError(
            f"{experiment.name!r}: multimodel requires arms + models. "
            f"Got arms={arms_to_run}, models={models_to_run}"
        )
    invalid = [a for a in arms_to_run if a not in ALL_ARMS]
    if invalid:
        raise ValueError(f"Unknown multimodel arm(s): {invalid}. Valid: {list(ALL_ARMS)}")

    questions = load_questions(experiment.dataset.questions, experiment.dataset.n)
    results_root = experiment.results_dir
    print(f"Loaded {len(questions)} questions from {experiment.dataset.questions}")
    print(f"Models : {models_to_run}")
    print(f"Arms   : {arms_to_run}")
    print(f"Output : {results_root / paths.MULTIMODEL_SUBDIR}")
    print(f"Total combos: {len(models_to_run)} × {len(arms_to_run)} = {len(models_to_run) * len(arms_to_run)}")

    override_dir = experiment.prompts_override_dir
    previous_override = os.environ.get("LEGAL_KG_PROMPTS_DIR")
    if override_dir is not None:
        os.environ["LEGAL_KG_PROMPTS_DIR"] = str(override_dir)
        print(f"Prompt override dir: {override_dir}")

    # Share RagPipeline across all logic_lm_graphrag combos to amortize warm-up
    shared_rag = None
    if "logic_lm_graphrag" in arms_to_run:
        from runtime.rag_query import RagPipeline
        shared_rag = RagPipeline()
        _ = shared_rag.embed_model  # warm up
        print("RagPipeline warmed (shared across logic_lm_graphrag combos)\n", flush=True)

    summaries: list[dict] = []
    try:
        for arm in arms_to_run:
            for model in models_to_run:
                print(f"\n=== COMBO: arm={arm}  model={model} ===")
                try:
                    s = _run_combo(
                        arm, model, questions, results_root, force, verbose,
                        shared_rag=shared_rag if arm == "logic_lm_graphrag" else None,
                    )
                    summaries.append(s)
                except Exception as e:
                    print(f"!! COMBO {arm}/{model} crashed: {type(e).__name__}: {e}",
                          file=sys.stderr, flush=True)
                    summaries.append({"arm": arm, "model": model,
                                      "fatal_error": f"{type(e).__name__}: {e}"})
    finally:
        if shared_rag is not None:
            try:
                shared_rag.close()
            except Exception:
                pass
        if override_dir is not None:
            if previous_override is None:
                os.environ.pop("LEGAL_KG_PROMPTS_DIR", None)
            else:
                os.environ["LEGAL_KG_PROMPTS_DIR"] = previous_override

    print("\n\n=== SUMMARY ===")
    for s in summaries:
        print(f"  {s}")
    return summaries


def main() -> int:
    from eval_core.experiment import Experiment

    p = argparse.ArgumentParser(
        description="Run the multimodel matrix for an experiment folder."
    )
    p.add_argument("experiment", type=Path, help="Path to experiment folder.")
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated. Overrides experiment.multimodel.models.",
    )
    p.add_argument(
        "--arms",
        type=str,
        default=None,
        help="Comma-separated. Overrides experiment.multimodel.arms.",
    )
    p.add_argument("--force", action="store_true", help="Re-run even if file exists.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    try:
        experiment = Experiment.from_path(args.experiment)
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    arms = [a.strip() for a in args.arms.split(",") if a.strip()] if args.arms else None
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None

    try:
        run_experiment_multimodel(
            experiment, arms=arms, models=models, force=args.force, verbose=args.verbose,
        )
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
