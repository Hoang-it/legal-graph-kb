"""Multi-model inference: chạy 2 elite arms × N OpenAI models trên dataset 200 câu.

Output mỗi (arm, model) 1 thư mục:
    data/eval/multimodel/results/{arm}__{model_safe}/A{stt}.json

Idempotent — skip nếu file đã tồn tại (--force để chạy lại).

Real API + real Prolog: dùng cùng OpenAILLMClient/SWI-Prolog như eval cũ,
chỉ swap model parameter. Khi reasoning model reject `temperature`/`response_format`,
elite_pipelines tự fallback (xem `_TokenTrackingLLMClient._chat_with_fallback`).

CLI:
    python -m experiments.run_multimodel_inference \\
        --models gpt-4.1,gpt-4o,gpt-5,gpt-5-mini \\
        --arms elite_no_retrieval,elite_graphrag \\
        --n 1            # smoke
    python -m experiments.run_multimodel_inference --n 10  # pilot
    python -m experiments.run_multimodel_inference --n 200 # full
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

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

QUESTIONS_PATH = Path("data/eval/questions_200.json")
OUT_ROOT = Path("data/eval/multimodel/results")

DEFAULT_MODELS = ("gpt-4.1", "gpt-4o", "gpt-5", "gpt-5-mini")
DEFAULT_ARMS = ("elite_no_retrieval", "elite_graphrag")
ALL_ARMS = ("elite_no_retrieval", "elite_ontology", "elite_graphrag")


def model_safe(model: str) -> str:
    """Chuyển model name về form an toàn cho filesystem.

    'gpt-4.1' -> 'gpt-4_1'   'gpt-4o' -> 'gpt-4o'   'gpt-5-mini' -> 'gpt-5-mini'
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model)


def combo_dir(arm: str, model: str) -> Path:
    return OUT_ROOT / f"{arm}__{model_safe(model)}"


def load_questions(n: int | None = None) -> list[dict]:
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        qs = json.load(f)
    if n:
        qs = qs[:n]
    return qs


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_pipeline(arm: str, model: str, shared_rag=None):
    """Khởi tạo pipeline cho 1 (arm, model). shared_rag chỉ dùng cho elite_graphrag."""
    from experiments.elite_pipelines import (
        EliteNoRetrievalPipeline,
        EliteOntologyPipeline,
        EliteGraphRAGPipeline,
    )

    if arm == "elite_no_retrieval":
        return EliteNoRetrievalPipeline(model=model)
    if arm == "elite_ontology":
        return EliteOntologyPipeline(model=model)
    if arm == "elite_graphrag":
        return EliteGraphRAGPipeline(model=model, rag_pipeline=shared_rag)
    raise ValueError(f"Unknown arm: {arm}")


def _run_combo(
    arm: str,
    model: str,
    questions: list[dict],
    force: bool,
    verbose: bool,
    shared_rag=None,
) -> dict:
    """Chạy 1 (arm, model) trên tất cả questions. Trả về summary dict."""
    out_dir = combo_dir(arm, model)
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


def main() -> int:
    p = argparse.ArgumentParser(
        description="Chạy elite arms × multiple OpenAI models trên N câu hỏi."
    )
    p.add_argument("--n", type=int, default=200, help="Số câu đầu tiên (default 200).")
    p.add_argument("--models", type=str,
                   default=",".join(DEFAULT_MODELS),
                   help=f"Comma-separated. Default: {','.join(DEFAULT_MODELS)}")
    p.add_argument("--arms", type=str,
                   default=",".join(DEFAULT_ARMS),
                   help=f"Comma-separated. Default: {','.join(DEFAULT_ARMS)}")
    p.add_argument("--force", action="store_true", help="Chạy lại dù file đã có.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    invalid = [a for a in arms if a not in ALL_ARMS]
    if invalid:
        raise SystemExit(f"Unknown arm(s): {invalid}. Valid: {list(ALL_ARMS)}")

    questions = load_questions(args.n)
    print(f"Loaded {len(questions)} questions từ {QUESTIONS_PATH}")
    print(f"Models : {models}")
    print(f"Arms   : {arms}")
    print(f"Output : {OUT_ROOT}")
    print(f"Total combos: {len(models)} × {len(arms)} = {len(models) * len(arms)}")
    print()

    # Share RagPipeline across all elite_graphrag combos to amortize warm-up
    shared_rag = None
    if "elite_graphrag" in arms:
        from src.rag_query import RagPipeline
        shared_rag = RagPipeline()
        _ = shared_rag.embed_model  # warm up
        print("RagPipeline warmed (shared across elite_graphrag combos)\n",
              flush=True)

    summaries = []
    for arm in arms:
        for model in models:
            print(f"\n=== COMBO: arm={arm}  model={model} ===")
            try:
                s = _run_combo(arm, model, questions, args.force, args.verbose,
                               shared_rag=shared_rag if arm == "elite_graphrag" else None)
                summaries.append(s)
            except Exception as e:
                print(f"!! COMBO {arm}/{model} crashed: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                summaries.append({"arm": arm, "model": model,
                                  "fatal_error": f"{type(e).__name__}: {e}"})

    if shared_rag is not None:
        try:
            shared_rag.close()
        except Exception:
            pass

    print("\n\n=== SUMMARY ===")
    for s in summaries:
        print(f"  {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
