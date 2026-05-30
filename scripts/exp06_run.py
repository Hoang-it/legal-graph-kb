"""Experiment 06 runner — retrieval-only A/B on full 200 questions.

For every question in ``data/eval/questions_200.json``, performs TWO real
retrieval calls and saves one record per arm:

- Arm ``dense``       : ``V5RetrievalPipeline.retrieve_dense_only`` —
                        pure BGE-M3 dense via ``clause_vec_tuned``.
- Arm ``full_rerank`` : ``V5RetrievalPipeline.retrieve_only`` — current
                        production pipeline (M2 baseline knobs).

No LLM, no E2E. Records land under
``experiments/06_retrieval_dense_vs_full/results/<arm>/A<stt>.json``.

Re-runnable / idempotent — existing record files are skipped unless
``--force`` is passed.

Usage::

    python scripts/exp06_run.py                  # full 200, both arms
    python scripts/exp06_run.py --stt 1-5        # pilot
    python scripts/exp06_run.py --arms dense     # only one arm
    python scripts/exp06_run.py --force          # overwrite existing
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

load_dotenv()
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

EXP_DIR = _REPO / "experiments" / "06_retrieval_dense_vs_full"
QUESTIONS_PATH = _REPO / "data" / "eval" / "questions_200.json"

# Shared knobs — both arms use the same encoder + dense index so the A/B
# isolates the contribution of sparse + temporal + RRF + rerank + expand.
SHARED = {
    "adapter_path": "models/bge-m3-bhxh-lora",
    "dense_index": "clause_vec_tuned",
    "reranker_model": "BAAI/bge-reranker-base",
}

# Arm `dense` — pure dense, but we raise dense_k so the metric script can
# report precision/recall up to K=50 without truncation.
DENSE_CFG = {
    **SHARED,
    "dense_k": 50,
    "sparse_k": 30,            # ignored by retrieve_dense_only
    "top_after_fusion": 50,    # ignored
    "rerank1_top_k": 15,       # ignored
    "rerank2_top_k": 12,       # ignored
    "rrf_k": 60,               # ignored
    "max_hops": 3,             # ignored
    "per_seed_neighbors": 10,  # ignored
    "temporal_mode": "strict_today_default",  # ignored
}
DENSE_TOP_K = 50  # passed explicitly to retrieve_dense_only

# Arm `full_rerank` — current production M2 baseline.
FULL_CFG = {
    **SHARED,
    "dense_k": 30,
    "sparse_k": 30,
    "top_after_fusion": 50,
    "rerank1_top_k": 15,
    "rerank2_top_k": 12,
    "rrf_k": 60,
    "max_hops": 3,
    "per_seed_neighbors": 10,
    "temporal_mode": "strict_today_default",
}

ARMS = ("dense", "full_rerank")


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
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--stt",
        type=str,
        default="",
        help="Probe stt range. Empty = all 200. Examples: '1-5', '1-200', '1,10,42'.",
    )
    p.add_argument(
        "--arms",
        type=str,
        default=",".join(ARMS),
        help=f"Comma-separated subset of arms. Default: '{','.join(ARMS)}'.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing records.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    arms_subset = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms_subset:
        if a not in ARMS:
            print(f"ERROR: unknown arm {a!r}; valid: {ARMS}", file=sys.stderr)
            return 2

    stt_list = _parse_stt(args.stt) if args.stt else []
    questions = _load_questions(stt_list)
    print(f"Probe size : {len(questions)} questions")
    print(f"Arms       : {arms_subset}")

    # Lazy import after env hygiene.
    from src.retrieval import V5RetrievalPipeline

    # Build ONE pipeline — both arms share the same encoder/index/reranker so
    # we want them to live in the same process to amortise warm-up cost.
    print("Warming up BGE-M3 + reranker ...", flush=True)
    pipe = V5RetrievalPipeline(**FULL_CFG)
    _ = pipe.embed_model
    if "full_rerank" in arms_subset:
        _ = pipe.reranker.model

    out_root = EXP_DIR / "results"
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
                        ans = pipe.retrieve_dense_only(q["question"], top_k=DENSE_TOP_K)
                        cfg_used = DENSE_CFG
                    else:  # full_rerank
                        ans = pipe.retrieve_only(q["question"])
                        cfg_used = FULL_CFG
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
                            f"  [{arm:<12} {i:>3}/{len(questions)}] stt={stt} "
                            f"({elapsed:.1f}s, final={n_final})",
                            flush=True,
                        )
                except Exception as e:  # noqa: BLE001
                    counts[arm]["failed"] += 1
                    print(
                        f"  ✗ [{arm} stt={stt}] {type(e).__name__}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    err_path = out_root / arm / f"A{stt}.error.json"
                    _write_record(
                        err_path,
                        {"arm": arm, "stt": stt, "error": f"{type(e).__name__}: {e}"},
                    )
    finally:
        pipe.close()

    print()
    for arm in arms_subset:
        c = counts[arm]
        print(
            f"{arm:<12} done={c['done']} skipped={c['skipped']} failed={c['failed']}"
        )
    print(f"Total wall time: {time.time() - t_total:.1f}s")

    return 0 if all(c["failed"] == 0 for c in counts.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
