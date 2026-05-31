"""Experiment 10 runner — HyDE prompts under GRACE-R, 3-arm A/B/C.

Clone of ``scripts/exp09_run.py`` with two changes:

1. ``EXP_DIR`` → ``experiments/10_hyde_gracer``.
2. ``LEGAL_KG_PROMPTS_DIR`` is exported BEFORE the HyDE generators are
   constructed, pointing at this experiment's ``prompts_override/``
   directory. The prompt loader (``src.prompts.resolve_prompt_path``)
   checks the override dir first, so HyDE1 / HyDE2 pick up the
   GRACE-R rewrites without touching the canonical prompts/ tree.

Cache: shared with exp 08/09 under ``artifacts/hyde/openai__gpt-4o-mini/``
and ``artifacts/hyde2/openai__gpt-4o-mini/``. The new prompts have
different ``prompt_sha`` so cache keys are disjoint — no collision
with prior experiments, and re-running exp10 hits this experiment's
cache for free.

Arms (all retrieval-only at article granularity, identical to exp 09):

- ``dense``        → :meth:`V5RetrievalPipeline.retrieve_dense_only`
                     (BGE-M3 LoRA + clause_vec_tuned, dense_k=100,
                     raw question — deterministic, no LLM)
- ``dense_hyde``   → :meth:`V5RetrievalPipeline.retrieve_dense_only_hyde`
                     (HyDE1 with GRACE-R prompt override)
- ``dense_hyde2``  → :meth:`V5RetrievalPipeline.retrieve_dense_only_hyde2`
                     (HyDE2 with GRACE-R grounded prompt override,
                     same seed_k=5 as exp 09)

Pre-flight cost estimate: ~$0.0005 × N for HyDE1 + $0.0007 × N for
HyDE2 = $0.24 total for full 200 cold. Default ``--cost-cap`` = $0.50.

``--pilot-5`` runs the first 5 stt for a quick smoke test (~$0.006).
``--pilot-50`` reuses exp 08's stratified 50 list (for parity with
exp 08/09 pilots).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean

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

EXP_DIR = _REPO / "experiments" / "10_hyde_gracer"
EXP08_DIR = _REPO / "experiments" / "08_hyde_retrieval"
QUESTIONS_PATH = _REPO / "data" / "eval" / "questions_200.json"
PILOT_50_PATH_EXP08 = EXP08_DIR / "pilot_50_stt.json"

# Export prompt override BEFORE any generator import touches src.prompts.
# Path must be absolute so it works regardless of cwd.
PROMPTS_OVERRIDE_DIR = EXP_DIR / "prompts_override"
if not PROMPTS_OVERRIDE_DIR.exists():
    raise FileNotFoundError(
        f"GRACE-R prompts not found at {PROMPTS_OVERRIDE_DIR} — "
        "check experiments/10_hyde_gracer/prompts_override/runtime/"
    )
os.environ["LEGAL_KG_PROMPTS_DIR"] = str(PROMPTS_OVERRIDE_DIR)

# Pipeline knobs — IDENTICAL to exp 09 so dense / dense_hyde2 seed
# retrieval is comparable across the two experiments.
SHARED = {
    "adapter_path": "models/bge-m3-bhxh-lora",
    "dense_index": "clause_vec_tuned",
    "reranker_model": "BAAI/bge-reranker-base",
}

DENSE_CFG = {
    **SHARED,
    "dense_k": 100,
    "sparse_k": 30,
    "top_after_fusion": 50,
    "rerank1_top_k": 15,
    "rerank2_top_k": 12,
    "rrf_k": 60,
    "max_hops": 3,
    "per_seed_neighbors": 10,
    "temporal_mode": "strict_today_default",
}
DENSE_TOP_K = 100

HYDE1_CFG = {
    "model": "gpt-4o-mini",
    "n": 1,
    "max_tokens": 700,
    "temperature": 0.0,
    "concurrency": 5,
    "prompt_path": "runtime/hyde_generate.md",
}

HYDE2_CFG = {
    "model": "gpt-4o-mini",
    "n": 1,
    "max_tokens": 700,
    "temperature": 0.0,
    "concurrency": 5,
    "prompt_path": "runtime/hyde_generate_grounded.md",
}

HYDE2_SEED_K = 5

ARMS = ("dense", "dense_hyde", "dense_hyde2")
HYDE1_ARMS = {"dense_hyde"}
HYDE2_ARMS = {"dense_hyde2"}


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------


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


def _load_pilot_50_from_exp08() -> dict:
    if not PILOT_50_PATH_EXP08.exists():
        raise FileNotFoundError(
            f"Expected exp 08's pilot list at {PILOT_50_PATH_EXP08} — "
            f"run scripts/exp08_run.py --pilot-50 first."
        )
    return json.loads(PILOT_50_PATH_EXP08.read_text(encoding="utf-8"))


def _write_record(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _pending_questions_for_arm(
    questions: list[dict], arm: str, out_root: Path, force: bool
) -> list[dict]:
    if force:
        return list(questions)
    pending = []
    for q in questions:
        out_path = out_root / arm / f"A{q['stt']}.json"
        if not out_path.exists():
            pending.append(q)
    return pending


# ---------------------------------------------------------------------------
# Pre-warm helpers
# ---------------------------------------------------------------------------


def _prewarm_hyde1(
    hyde, questions: list[dict], out_root: Path, force: bool
) -> None:
    pending = _pending_questions_for_arm(questions, "dense_hyde", out_root, force)
    if not pending:
        print("  HyDE1 prewarm: all docs already on disk for pending arm.")
        return
    print(f"  HyDE1 prewarm: generating for {len(pending)} unique questions "
          f"(concurrency={hyde.concurrency}) ...", flush=True)
    t0 = time.time()
    _ = hyde.generate_batch([q["question"] for q in pending])
    dt = time.time() - t0
    cs = hyde.cost_summary()
    print(f"  HyDE1 prewarm: done in {dt:.1f}s. "
          f"api_calls={cs['api_calls']} cache_hits={cs['cache_hits']} "
          f"cost=${cs['total_cost_usd']:.4f}", flush=True)


def _prewarm_hyde2(
    pipe_dense, hyde2, questions: list[dict], out_root: Path, force: bool,
    seed_k: int,
) -> None:
    pending = _pending_questions_for_arm(questions, "dense_hyde2", out_root, force)
    if not pending:
        print("  HyDE2 prewarm: all docs already on disk for pending arm.")
        return
    print(f"  HyDE2 prewarm: computing pass-1 seeds for {len(pending)} questions ...",
          flush=True)
    t_seed = time.time()
    triples: list[tuple[str, list[str], list[str]]] = []
    for q in pending:
        rows = pipe_dense.retriever._dense_search(q["question"], seed_k)
        if not rows:
            raise RuntimeError(
                f"HyDE2 prewarm: pass-1 returned 0 rows for stt={q['stt']} — "
                f"check Neo4j connection + dense index."
            )
        triples.append(
            (q["question"], [r["text"] for r in rows], [r["clause_id"] for r in rows])
        )
    dt_seed = time.time() - t_seed
    print(f"  HyDE2 prewarm: seeds done in {dt_seed:.1f}s. "
          f"Now batch-generating HyDE2 docs (concurrency={hyde2.concurrency}) ...",
          flush=True)
    t_llm = time.time()
    _ = hyde2.generate_batch(triples)
    dt_llm = time.time() - t_llm
    cs = hyde2.cost_summary()
    print(f"  HyDE2 prewarm: LLM done in {dt_llm:.1f}s. "
          f"api_calls={cs['api_calls']} cache_hits={cs['cache_hits']} "
          f"cost=${cs['total_cost_usd']:.4f}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stt", type=str, default="",
                   help="Comma/range stt list (e.g. '1-5,10'); empty = full dataset (200) "
                        "or pilot-N if a --pilot-* flag is given.")
    p.add_argument("--pilot-5", action="store_true",
                   help="First 5 stt from the dataset for a quick smoke test (~$0.006).")
    p.add_argument("--pilot-50", action="store_true",
                   help="Use exp 08's stratified 50-question pilot subset "
                        "(experiments/08_hyde_retrieval/pilot_50_stt.json).")
    p.add_argument("--arms", type=str, default=",".join(ARMS),
                   help=f"Subset of {ARMS}; default = all three.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing records.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--hyde-model", type=str, default=HYDE1_CFG["model"],
                   help="OpenAI model id used by both HyDE1 and HyDE2 generators.")
    p.add_argument("--cost-cap", type=float, default=0.50,
                   help="Abort pre-flight if cost estimate exceeds this USD "
                        "(default $0.50).")
    p.add_argument("--skip-prewarm", action="store_true",
                   help="Skip batched HyDE pre-generation; per-question lazy mode.")
    args = p.parse_args()

    if args.pilot_5 and args.pilot_50:
        print("ERROR: --pilot-5 and --pilot-50 are mutually exclusive.", file=sys.stderr)
        return 2

    arms_subset = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms_subset:
        if a not in ARMS:
            print(f"ERROR: unknown arm {a!r}; valid: {ARMS}", file=sys.stderr)
            return 2

    # Resolve question set.
    pilot_info: dict | None = None
    if args.pilot_5:
        all_qs = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
        all_qs_sorted = sorted(all_qs, key=lambda q: q["stt"])
        stt_list = [q["stt"] for q in all_qs_sorted[:5]]
        print(f"Pilot subset    : --pilot-5 → first 5 stt = {stt_list}")
    elif args.pilot_50:
        pilot_info = _load_pilot_50_from_exp08()
        stt_list = pilot_info["stt_list"]
        print(f"Pilot subset    : n={pilot_info['n']} seed={pilot_info['seed']} "
              f"quotas={pilot_info['quotas']} (reused from exp 08)")
    elif args.stt:
        stt_list = _parse_stt(args.stt)
    else:
        stt_list = []
    questions = _load_questions(stt_list)

    need_hyde1 = "dense_hyde" in arms_subset
    need_hyde2 = "dense_hyde2" in arms_subset

    print(f"Prompts override: {PROMPTS_OVERRIDE_DIR}")
    print(f"Probe size      : {len(questions)} questions")
    print(f"Arms            : {arms_subset}")
    print(f"HyDE1 active    : {need_hyde1}")
    print(f"HyDE2 active    : {need_hyde2} (seed_k={HYDE2_SEED_K})")
    print(f"Dense cfg       : dense_k={DENSE_CFG['dense_k']} "
          f"adapter={SHARED['adapter_path']} index={SHARED['dense_index']}")

    est_cost = 0.0
    if need_hyde1:
        est_cost += 0.0005 * len(questions)
    if need_hyde2:
        est_cost += 0.0007 * len(questions)
    if est_cost > 0:
        print(f"Estimated cost  : ~${est_cost:.4f} (assuming all cache misses)")
        if est_cost > args.cost_cap:
            print(f"ABORT: estimated cost ${est_cost:.4f} exceeds "
                  f"--cost-cap ${args.cost_cap:.4f}", file=sys.stderr)
            return 3

    from src.retrieval.hyde import OpenAIHydeGenerator
    from src.retrieval.hyde2 import OpenAIGroundedHydeGenerator
    from src.retrieval.pipeline import V5RetrievalPipeline

    hyde1 = None
    if need_hyde1:
        cfg = {**HYDE1_CFG, "model": args.hyde_model}
        hyde1 = OpenAIHydeGenerator(**cfg)
        print(f"  HyDE1 prompt_sha: {hyde1.prompt_sha}")
        print(f"  HyDE1 prompt path: {hyde1.prompt_source_path}")
        print(f"  HyDE1 cache_dir : {hyde1.cache_dir}")

    hyde2 = None
    if need_hyde2:
        cfg = {**HYDE2_CFG, "model": args.hyde_model}
        hyde2 = OpenAIGroundedHydeGenerator(**cfg)
        print(f"  HyDE2 prompt_sha: {hyde2.prompt_sha}")
        print(f"  HyDE2 prompt path: {hyde2.prompt_source_path}")
        print(f"  HyDE2 cache_dir : {hyde2.cache_dir}")

    print("Warming up BGE-M3 (shared across pipelines) ...", flush=True)
    pipe_dense = V5RetrievalPipeline(**DENSE_CFG)
    _ = pipe_dense.embed_model

    pipe_hyde = None
    if need_hyde1:
        pipe_hyde = V5RetrievalPipeline(hyde=hyde1, **DENSE_CFG)
        pipe_hyde._embed_model = pipe_dense._embed_model
        print("  pipe_hyde: shared embed_model.", flush=True)

    pipe_hyde2 = None
    if need_hyde2:
        pipe_hyde2 = V5RetrievalPipeline(
            hyde2=hyde2, hyde2_seed_k=HYDE2_SEED_K, **DENSE_CFG
        )
        pipe_hyde2._embed_model = pipe_dense._embed_model
        print("  pipe_hyde2: shared embed_model.", flush=True)

    out_root = EXP_DIR / "results"

    if not args.skip_prewarm:
        if hyde1 is not None:
            _prewarm_hyde1(hyde1, questions, out_root, args.force)
        if hyde2 is not None:
            _prewarm_hyde2(
                pipe_dense, hyde2, questions, out_root, args.force, HYDE2_SEED_K
            )

    counts = {a: {"done": 0, "skipped": 0, "failed": 0, "latencies": []}
              for a in arms_subset}
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
                        ans = pipe_dense.retrieve_dense_only(
                            q["question"], top_k=DENSE_TOP_K
                        )
                        cfg_used = {**DENSE_CFG, "mode": "dense_only"}
                    elif arm == "dense_hyde":
                        assert pipe_hyde is not None
                        ans = pipe_hyde.retrieve_dense_only_hyde(
                            q["question"], top_k=DENSE_TOP_K
                        )
                        cfg_used = {
                            **DENSE_CFG,
                            "mode": "dense_only_hyde",
                            "hyde": {**HYDE1_CFG, "model": args.hyde_model},
                        }
                    elif arm == "dense_hyde2":
                        assert pipe_hyde2 is not None
                        ans = pipe_hyde2.retrieve_dense_only_hyde2(
                            q["question"], top_k=DENSE_TOP_K
                        )
                        cfg_used = {
                            **DENSE_CFG,
                            "mode": "dense_only_hyde2",
                            "hyde2": {**HYDE2_CFG, "model": args.hyde_model},
                            "hyde2_seed_k": HYDE2_SEED_K,
                        }
                    else:
                        raise RuntimeError(f"unreachable: arm={arm!r}")
                    elapsed = round(time.time() - t, 3)
                    counts[arm]["latencies"].append(elapsed)
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
                            f"  [{arm:<14} {i:>3}/{len(questions)}] stt={stt} "
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
        pipe_dense.close()
        if pipe_hyde is not None:
            pipe_hyde.close()
        if pipe_hyde2 is not None:
            pipe_hyde2.close()

    print()
    print("=" * 78)
    print("Per-arm summary")
    print("=" * 78)
    for arm in arms_subset:
        c = counts[arm]
        avg = round(mean(c["latencies"]), 3) if c["latencies"] else None
        print(f"  {arm:<14} done={c['done']} skipped={c['skipped']} "
              f"failed={c['failed']} avg_latency={avg}s")
    print(f"Total wall time: {time.time() - t_total:.1f}s")

    if hyde1 is not None:
        cs = hyde1.cost_summary()
        print()
        print("=" * 78)
        print("HyDE1 LLM cost summary")
        print("=" * 78)
        print(f"  model            : {cs['model_id']}")
        print(f"  API calls (cold) : {cs['api_calls']}")
        print(f"  Cache hits       : {cs['cache_hits']}")
        print(f"  Prompt tokens    : {cs['prompt_tokens']:,}")
        print(f"  Completion tokens: {cs['completion_tokens']:,}")
        print(f"  Cached tokens    : {cs['cached_tokens']:,}")
        print(f"  TOTAL COST       : ${cs['total_cost_usd']:.6f}")

    if hyde2 is not None:
        cs = hyde2.cost_summary()
        print()
        print("=" * 78)
        print("HyDE2 LLM cost summary")
        print("=" * 78)
        print(f"  model            : {cs['model_id']}")
        print(f"  API calls (cold) : {cs['api_calls']}")
        print(f"  Cache hits       : {cs['cache_hits']}")
        print(f"  Prompt tokens    : {cs['prompt_tokens']:,}")
        print(f"  Completion tokens: {cs['completion_tokens']:,}")
        print(f"  Cached tokens    : {cs['cached_tokens']:,}")
        print(f"  TOTAL COST       : ${cs['total_cost_usd']:.6f}")
        total_cost = cs["total_cost_usd"] + (
            hyde1.cost_summary()["total_cost_usd"] if hyde1 is not None else 0.0
        )
        if total_cost > args.cost_cap:
            print(f"  WARNING: total cost ${total_cost:.4f} exceeded --cost-cap "
                  f"${args.cost_cap:.4f}", file=sys.stderr)

    return 0 if all(c["failed"] == 0 for c in counts.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
