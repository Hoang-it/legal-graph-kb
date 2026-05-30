"""Week 1 retrieval-only runner.

Runs :meth:`V5RetrievalPipeline.retrieve_only` on a probe set and saves the
per-stage article-ID pool (dense, sparse, post-temporal, RRF, rerank1,
expanded, final) into one JSON record per question.

No LLM call, no citation parsing — pure retrieval audit.

Each config (knob set) writes into its own subfolder so the audit script can
compare them side-by-side without ambiguity.

Usage examples::

    # Baseline M2 (matches Sprint 2 published numbers)
    python scripts/run_retrieval_only.py baseline_m2

    # K-tuned variant
    python scripts/run_retrieval_only.py k_tuned

    # Combined Quick Wins
    python scripts/run_retrieval_only.py k_tuned_temporal_relax

All configs use the same probe set (stt 1..30 by default; pass --stt 1..150
for the locked test split).
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

# Configs are intentionally declarative — every knob lives here and nowhere
# else. Adding a new variant means appending a single dict.
CONFIGS: dict[str, dict] = {
    "baseline_m2": {
        # Reproduces Sprint 2 M2 numbers (overall 0.333, in_corpus 0.506).
        "adapter_path": "models/bge-m3-bhxh-lora",
        "dense_index": "clause_vec_tuned",
        "reranker_model": "BAAI/bge-reranker-base",
        "dense_k": 30,
        "sparse_k": 30,
        "top_after_fusion": 50,
        "rerank1_top_k": 15,
        "rerank2_top_k": 12,
        "rrf_k": 60,
        "max_hops": 3,
        "per_seed_neighbors": 10,
        "temporal_mode": "strict_today_default",
    },
    "k_tuned": {
        # QW1 — bigger candidate pools at every retrieval stage. Final remains 12.
        "adapter_path": "models/bge-m3-bhxh-lora",
        "dense_index": "clause_vec_tuned",
        "reranker_model": "BAAI/bge-reranker-base",
        "dense_k": 50,
        "sparse_k": 50,
        "top_after_fusion": 100,
        "rerank1_top_k": 25,
        "rerank2_top_k": 12,
        "rrf_k": 60,
        "max_hops": 3,
        "per_seed_neighbors": 10,
        "temporal_mode": "strict_today_default",
    },
    "temporal_relax": {
        # QW2 — date-less queries skip the temporal filter (so L58_2014 isn't
        # systematically dropped for the bulk of the probe).
        "adapter_path": "models/bge-m3-bhxh-lora",
        "dense_index": "clause_vec_tuned",
        "reranker_model": "BAAI/bge-reranker-base",
        "dense_k": 30,
        "sparse_k": 30,
        "top_after_fusion": 50,
        "rerank1_top_k": 15,
        "rerank2_top_k": 12,
        "rrf_k": 60,
        "max_hops": 3,
        "per_seed_neighbors": 10,
        "temporal_mode": "skip_when_no_date",
    },
    "k_tuned_temporal_relax": {
        # QW1 + QW2 stacked.
        "adapter_path": "models/bge-m3-bhxh-lora",
        "dense_index": "clause_vec_tuned",
        "reranker_model": "BAAI/bge-reranker-base",
        "dense_k": 50,
        "sparse_k": 50,
        "top_after_fusion": 100,
        "rerank1_top_k": 25,
        "rerank2_top_k": 12,
        "rrf_k": 60,
        "max_hops": 3,
        "per_seed_neighbors": 10,
        "temporal_mode": "skip_when_no_date",
    },
}

OUTPUT_ROOT = Path("experiments/05_v5_retrieval_audit")


def _load_questions(stt_list: list[int]) -> list[dict]:
    questions = json.loads(Path("data/eval/questions_200.json").read_text(encoding="utf-8"))
    by_stt = {q["stt"]: q for q in questions}
    return [by_stt[s] for s in stt_list if s in by_stt]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "config",
        choices=list(CONFIGS.keys()),
        help="Knob set name (declared in CONFIGS).",
    )
    p.add_argument(
        "--stt",
        type=str,
        default="1-30",
        help="Probe stt range. Examples: '1-30' (default), '1-150', '1,5,10'.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing records instead of skipping.",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # Parse stt expression.
    stt_list: list[int] = []
    for token in args.stt.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            stt_list.extend(range(int(lo), int(hi) + 1))
        else:
            stt_list.append(int(token))

    questions = _load_questions(stt_list)
    print(f"Config       : {args.config}")
    print(f"Probe size   : {len(questions)} questions")
    cfg = CONFIGS[args.config]
    print(f"Knobs        :")
    for k, v in cfg.items():
        print(f"  {k:<22} {v}")

    out_dir = OUTPUT_ROOT / args.config
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import so import errors surface alongside config print.
    from src.retrieval import V5RetrievalPipeline

    pipe = V5RetrievalPipeline(**cfg)
    _ = pipe.embed_model
    _ = pipe.reranker.model

    n_done = n_skipped = n_failed = 0
    t_total = time.time()
    try:
        for i, q in enumerate(questions, 1):
            stt = q["stt"]
            out_path = out_dir / f"A{stt}.json"
            if out_path.exists() and not args.force:
                n_skipped += 1
                continue
            try:
                ans = pipe.retrieve_only(q["question"])
                record = {
                    "config": args.config,
                    "stt": stt,
                    "question": q["question"],
                    "gold_citations_raw": q.get("gold_citations_raw"),
                    "retrieval_only": asdict(ans),
                }
                out_path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                n_done += 1
                if args.verbose or i % 5 == 0:
                    a = ans
                    print(
                        f"  [{args.config:<24} {i:>3}/{len(questions)}] stt={stt} "
                        f"({a.elapsed_s:.1f}s, dense={len(a.dense_article_ids)}, "
                        f"sparse={len(a.sparse_article_ids)}, "
                        f"fused={len(a.fused_article_ids)}, "
                        f"rerank1={len(a.rerank1_article_ids)}, "
                        f"final={len(a.final_article_ids)})",
                        flush=True,
                    )
            except Exception as e:  # noqa: BLE001
                n_failed += 1
                print(f"  ✗ [{args.config} {stt}] {type(e).__name__}: {e}", file=sys.stderr)
                (out_path.with_suffix(".error.json")).write_text(
                    json.dumps(
                        {"config": args.config, "stt": stt, "error": f"{type(e).__name__}: {e}"},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
    finally:
        pipe.close()

    print(
        f"\n{args.config} done: {n_done} new, {n_skipped} skipped, {n_failed} failed "
        f"({time.time() - t_total:.1f}s)"
    )
    return 0 if not n_failed else 2


if __name__ == "__main__":
    sys.exit(main())
