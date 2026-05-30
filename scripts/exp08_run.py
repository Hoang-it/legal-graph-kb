"""Experiment 08 runner — HyDE retrieval-only 4-arm A/B on full 200 questions.

Arms (all retrieval-only, no LLM):

- ``dense``             → :meth:`V5RetrievalPipeline.retrieve_dense_only`
                          (BGE-M3 LoRA + clause_vec_tuned, dense_k=100)
- ``dense_hyde``        → :meth:`V5RetrievalPipeline.retrieve_dense_only_hyde`
                          (same, but dense query = embedding of Qwen-generated
                          hypothetical doc)
- ``full_rerank``       → :meth:`V5RetrievalPipeline.retrieve_only` on a
                          pipeline WITHOUT hyde (full v5 scaled, mirrors exp 07)
- ``full_rerank_hyde``  → :meth:`V5RetrievalPipeline.retrieve_only` on a
                          pipeline WITH hyde (same pipeline, dense channel
                          uses HyDE embedding)

Two pipeline instances are constructed (one with ``hyde=None``, one with
``hyde=QwenHydeGenerator(...)``). Sharing a single instance + flipping the
encoder per call would also work, but two instances keep each arm's
config_snapshot honest and avoid race conditions in the cache. They share
the same Neo4j driver indirectly (each opens its own driver — Neo4j is
fine with this).

Records land under
``experiments/08_hyde_retrieval/results/<arm>/A<stt>.json`` —
idempotent + ``--force`` overwrite + ``--stt`` subset (matches exp07_run.py
shape). HyDE generation cache (``artifacts/hyde/...``) is shared across
arms ``dense_hyde`` and ``full_rerank_hyde`` so each question's
hypothetical doc is generated exactly once per (model, prompt, n,
max_new_tokens) combo.

Designed for Colab Free T4 — see ``notebooks/exp08_hyde_colab.ipynb``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

load_dotenv()
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

EXP_DIR = _REPO / "experiments" / "08_hyde_retrieval"
QUESTIONS_PATH = _REPO / "data" / "eval" / "questions_200.json"

# Shared knobs across all 4 arms — mirror exp 07 so K up to 100 is meaningful.
SHARED = {
    "adapter_path": "models/bge-m3-bhxh-lora",
    "dense_index": "clause_vec_tuned",
    "reranker_model": "BAAI/bge-reranker-base",
}

DENSE_CFG = {
    **SHARED,
    "dense_k": 100,
    "sparse_k": 30,            # ignored by retrieve_dense_only*
    "top_after_fusion": 50,
    "rerank1_top_k": 15,
    "rerank2_top_k": 12,
    "rrf_k": 60,
    "max_hops": 3,
    "per_seed_neighbors": 10,
    "temporal_mode": "strict_today_default",
}
DENSE_TOP_K = 100

FULL_CFG = {
    **SHARED,
    "dense_k": 100,
    "sparse_k": 100,
    "top_after_fusion": 150,
    "rerank1_top_k": 50,
    "rerank2_top_k": 100,
    "rrf_k": 60,
    "max_hops": 3,
    "per_seed_neighbors": 15,
    "temporal_mode": "strict_today_default",
}

# HyDE generator config — defaults from plan §D1, D2, D9, D10
HYDE_CFG = {
    "model_id": "Qwen/Qwen2.5-3B-Instruct",
    "n": 1,
    "max_new_tokens": 400,
    "batch_size": 4,
    "dtype": "fp16",
    "prompt_path": "runtime/hyde_generate.md",
}

ARMS = ("dense", "dense_hyde", "full_rerank", "full_rerank_hyde")
HYDE_ARMS = {"dense_hyde", "full_rerank_hyde"}


def _parse_stt(expr: str) -> list[int]:
    out: list[int] = []
    for token in expr.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(token))
    return out


def _load_questions(stt_list: list[int]) -> list[dict]:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    by_stt = {q["stt"]: q for q in questions}
    if not stt_list:
        return sorted(questions, key=lambda q: q["stt"])
    return [by_stt[s] for s in stt_list if s in by_stt]


def _write_record(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _pending_questions_for_arm(
    questions: list[dict], arm: str, out_root: Path, force: bool
) -> list[dict]:
    """Return only questions that don't already have a record (or all when --force)."""
    if force:
        return list(questions)
    pending = []
    for q in questions:
        out_path = out_root / arm / f"A{q['stt']}.json"
        if not out_path.exists():
            pending.append(q)
    return pending


def _prewarm_hyde_for_pending(
    hyde, questions: list[dict], hyde_arms_requested: list[str], out_root: Path, force: bool
) -> None:
    """Batch-generate HyDE docs for every question that any HyDE arm still
    needs to process. This populates the cache up-front so the per-arm
    per-question loop becomes a pure cache-read — no model forward pass
    interleaved with Neo4j calls, which would thrash the GPU."""
    if hyde is None:
        return
    pending_qs: dict[int, dict] = {}  # stt → question
    for arm in hyde_arms_requested:
        for q in _pending_questions_for_arm(questions, arm, out_root, force):
            pending_qs[q["stt"]] = q
    if not pending_qs:
        print("  HyDE prewarm: all docs already cached for pending arms.")
        return
    ordered = [pending_qs[s] for s in sorted(pending_qs)]
    print(f"  HyDE prewarm: generating for {len(ordered)} unique questions "
          f"(batch_size={hyde.batch_size}) ...", flush=True)
    t0 = time.time()
    _ = hyde.generate_batch([q["question"] for q in ordered])
    print(f"  HyDE prewarm: done in {time.time() - t0:.1f}s.", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stt", type=str, default="",
                   help="Comma/range stt list (e.g. '1-5,10'); empty = full dataset.")
    p.add_argument("--arms", type=str, default=",".join(ARMS),
                   help=f"Subset of {ARMS}; default = all four.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing records.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--skip-prewarm", action="store_true",
                   help="Skip batched HyDE pre-generation. Records will still "
                        "generate lazily on first dense call per question.")
    p.add_argument("--hyde-dtype", type=str, default=HYDE_CFG["dtype"],
                   choices=["fp16", "bf16", "4bit"],
                   help="Qwen dtype — '4bit' = bitsandbytes fallback for tight VRAM.")
    args = p.parse_args()

    arms_subset = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms_subset:
        if a not in ARMS:
            print(f"ERROR: unknown arm {a!r}; valid: {ARMS}", file=sys.stderr)
            return 2

    stt_list = _parse_stt(args.stt) if args.stt else []
    questions = _load_questions(stt_list)
    hyde_arms_requested = [a for a in arms_subset if a in HYDE_ARMS]
    need_hyde = bool(hyde_arms_requested)

    print(f"Probe size       : {len(questions)} questions")
    print(f"Arms             : {arms_subset}")
    print(f"HyDE arms active : {hyde_arms_requested or '(none — skipping Qwen)'}")
    print(f"Full cfg         : dense_k={FULL_CFG['dense_k']} sparse_k={FULL_CFG['sparse_k']} "
          f"top_after_fusion={FULL_CFG['top_after_fusion']} "
          f"rerank1_top_k={FULL_CFG['rerank1_top_k']} "
          f"rerank2_top_k={FULL_CFG['rerank2_top_k']} "
          f"per_seed_neighbors={FULL_CFG['per_seed_neighbors']}")
    if need_hyde:
        print(f"HyDE cfg         : model={HYDE_CFG['model_id']} n={HYDE_CFG['n']} "
              f"max_new_tokens={HYDE_CFG['max_new_tokens']} "
              f"batch_size={HYDE_CFG['batch_size']} dtype={args.hyde_dtype}")

    from src.retrieval.hyde import QwenHydeGenerator
    from src.retrieval.pipeline import V5RetrievalPipeline

    hyde = None
    if need_hyde:
        cfg = {**HYDE_CFG, "dtype": args.hyde_dtype}
        hyde = QwenHydeGenerator(**cfg)
        print(f"  HyDE prompt_sha : {hyde.prompt_sha}")
        print(f"  HyDE cache_dir  : {hyde.cache_dir}")

    # Two pipeline instances — keep their config_snapshot honest. The
    # non-hyde pipe runs arms {dense, full_rerank}; the hyde pipe runs
    # {dense_hyde, full_rerank_hyde}. They share the same BGE-M3 weights
    # only if both are loaded into the same GPU — they each construct
    # their own SentenceTransformer, which is intentional: we want the
    # config_snapshot to reflect the actual pipeline used. The cost is
    # ~1.2 GB extra VRAM for the second BGE-M3 instance; the T4 16 GB
    # still has headroom (BGE-M3 1.2 GB × 2 + reranker 0.5 GB + Qwen 6 GB
    # ≈ 9 GB) per plan §"Why Colab Free T4".
    print("Warming up BGE-M3 + reranker (non-hyde pipeline) ...", flush=True)
    pipe_plain = V5RetrievalPipeline(**FULL_CFG)
    _ = pipe_plain.embed_model
    if "full_rerank" in arms_subset:
        _ = pipe_plain.reranker.model

    pipe_hyde = None
    if need_hyde:
        print("Warming up BGE-M3 + reranker (hyde pipeline) ...", flush=True)
        pipe_hyde = V5RetrievalPipeline(hyde=hyde, **FULL_CFG)
        _ = pipe_hyde.embed_model
        if "full_rerank_hyde" in arms_subset:
            _ = pipe_hyde.reranker.model

    out_root = EXP_DIR / "results"

    # HyDE prewarm — fill the cache for every pending question across both
    # HyDE arms in one batched pass.
    if hyde is not None and not args.skip_prewarm:
        _prewarm_hyde_for_pending(hyde, questions, hyde_arms_requested, out_root, args.force)

    counts = {a: {"done": 0, "skipped": 0, "failed": 0} for a in arms_subset}
    t_total = time.time()

    try:
        for i, q in enumerate(questions, 1):
            stt = q["stt"]
            for arm in arms_subset:
                out_path = out_root / arm / f"A{stt}.json"
                if out_path.exists() and not args.force:
                    counts[arm]["skipped"] += 1
                    continue
                try:
                    t = time.time()
                    if arm == "dense":
                        ans = pipe_plain.retrieve_dense_only(
                            q["question"], top_k=DENSE_TOP_K
                        )
                        cfg_used = DENSE_CFG
                    elif arm == "dense_hyde":
                        assert pipe_hyde is not None
                        ans = pipe_hyde.retrieve_dense_only_hyde(
                            q["question"], top_k=DENSE_TOP_K
                        )
                        cfg_used = {**DENSE_CFG, "hyde": HYDE_CFG}
                    elif arm == "full_rerank":
                        ans = pipe_plain.retrieve_only(q["question"])
                        cfg_used = FULL_CFG
                    elif arm == "full_rerank_hyde":
                        assert pipe_hyde is not None
                        ans = pipe_hyde.retrieve_only(q["question"])
                        cfg_used = {**FULL_CFG, "hyde": HYDE_CFG}
                    else:
                        raise RuntimeError(f"unreachable: arm={arm!r}")
                    elapsed = round(time.time() - t, 3)
                    record = {
                        "arm": arm,
                        "stt": stt,
                        "question": q["question"],
                        "gold_citations_raw": q.get("gold_citations_raw"),
                        "config_used": cfg_used,
                        "retrieval_only": asdict(ans),
                    }
                    _write_record(out_path, record)
                    counts[arm]["done"] += 1
                    if args.verbose or i % 10 == 0:
                        n_final = len(ans.final_article_ids)
                        print(
                            f"  [{arm:<18} {i:>3}/{len(questions)}] stt={stt} "
                            f"({elapsed:.1f}s, final={n_final})",
                            flush=True,
                        )
                except Exception as e:  # noqa: BLE001
                    counts[arm]["failed"] += 1
                    print(
                        f"  X [{arm} stt={stt}] {type(e).__name__}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    err_path = out_root / arm / f"A{stt}.error.json"
                    _write_record(
                        err_path,
                        {"arm": arm, "stt": stt, "error": f"{type(e).__name__}: {e}"},
                    )
    finally:
        pipe_plain.close()
        if pipe_hyde is not None:
            pipe_hyde.close()

    print()
    for arm in arms_subset:
        c = counts[arm]
        print(f"{arm:<18} done={c['done']} skipped={c['skipped']} failed={c['failed']}")
    print(f"Total wall time: {time.time() - t_total:.1f}s")

    return 0 if all(c["failed"] == 0 for c in counts.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
