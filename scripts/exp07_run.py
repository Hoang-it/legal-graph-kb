"""Experiment 07 runner — extended-K retrieval-only A/B on full 200 questions.

Same shape as ``scripts/exp06_run.py`` but with both arms scaled up so that
the final retrieved list can support K up to 100:

- Arm ``dense``       : dense_k = 100 (vs 50 in exp 06).
- Arm ``full_rerank`` : dense_k = 100, sparse_k = 100,
                        top_after_fusion = 150, rerank1_top_k = 50,
                        per_seed_neighbors = 15, rerank2_top_k = 100
                        (vs 12 in exp 06).

No LLM, no E2E. Records land under
``experiments/07_retrieval_extended_k/results/<arm>/A<stt>.json``.

Re-runnable / idempotent. Same CLI surface as exp06_run.py.
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

EXP_DIR = _REPO / "experiments" / "07_retrieval_extended_k"
QUESTIONS_PATH = _REPO / "data" / "eval" / "questions_200.json"

SHARED = {
    "adapter_path": "models/bge-m3-bhxh-lora",
    "dense_index": "clause_vec_tuned",
    "reranker_model": "BAAI/bge-reranker-base",
}

DENSE_CFG = {
    **SHARED,
    "dense_k": 100,
    "sparse_k": 30,            # ignored by retrieve_dense_only
    "top_after_fusion": 50,    # ignored
    "rerank1_top_k": 15,       # ignored
    "rerank2_top_k": 12,       # ignored
    "rrf_k": 60,               # ignored
    "max_hops": 3,             # ignored
    "per_seed_neighbors": 10,  # ignored
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
    p.add_argument("--stt", type=str, default="")
    p.add_argument("--arms", type=str, default=",".join(ARMS))
    p.add_argument("--force", action="store_true")
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
    print(f"Full cfg   : dense_k={FULL_CFG['dense_k']} sparse_k={FULL_CFG['sparse_k']} "
          f"top_after_fusion={FULL_CFG['top_after_fusion']} "
          f"rerank1_top_k={FULL_CFG['rerank1_top_k']} "
          f"rerank2_top_k={FULL_CFG['rerank2_top_k']} "
          f"per_seed_neighbors={FULL_CFG['per_seed_neighbors']}")

    from src.retrieval import V5RetrievalPipeline

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
                    else:
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
        print(f"{arm:<12} done={c['done']} skipped={c['skipped']} failed={c['failed']}")
    print(f"Total wall time: {time.time() - t_total:.1f}s")

    return 0 if all(c["failed"] == 0 for c in counts.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
