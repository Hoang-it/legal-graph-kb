"""Experiment 13 runner — semantic-grounded HyDE (concept frame) 3-arm.

Retrieval-only, article granularity, on the TUNED stack (BGE-M3 LoRA +
``clause_vec_tuned``) — same stack as exp 08/09 so numbers are comparable.

Arms:
- ``dense``               → :meth:`V5RetrievalPipeline.retrieve_dense_only` (raw question)
- ``dense_hyde``          → :meth:`...retrieve_dense_only_hyde` (HyDE1 — the bar)
- ``dense_hyde_semantic`` → :meth:`...retrieve_dense_only_hyde_semantic` (NEW)

The semantic arm grounds the HyDE doc on a **BHXH concept frame** built by
``runtime.retrievers.semantic_context.build_semantic_context`` (query →
concepts + KG entities from ``ontology_kg_full.json``; NO dense clause seed).

Reuses exp 09's runner helpers + the exact tuned dense config so ``dense`` /
``dense_hyde`` are byte-comparable to exp 08/09 (and HyDE1 hits the shared
``artifacts/hyde/`` cache for $0). The semantic HyDE has its own cache at
``artifacts/hyde_semantic/``.

    python -m scripts.exp13_run --pilot-50
    python -m scripts.exp13_run --stt 1-5 --arms dense_hyde_semantic --verbose
"""
from __future__ import annotations

import argparse
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

# Reuse exp 09's runner helpers + tuned dense config verbatim.
from scripts.exp09_run import (  # noqa: E402
    DENSE_CFG,
    DENSE_TOP_K,
    HYDE1_CFG,
    SHARED,
    _load_pilot_50_from_exp08,
    _load_questions,
    _parse_stt,
    _pending_questions_for_arm,
    _write_record,
)

EXP_DIR = _REPO / "experiments" / "13_hyde_semantic"

HYDE_SEMANTIC_CFG = {
    "model": "gpt-4o-mini",
    "n": 1,
    "max_tokens": 700,
    "temperature": 0.0,
    "concurrency": 5,
    "prompt_path": "runtime/hyde_generate_semantic.md",
}

ARMS = ("dense", "dense_hyde", "dense_hyde_semantic")
HYDE1_ARMS = {"dense_hyde"}
SEMANTIC_ARMS = {"dense_hyde_semantic"}


def _prewarm_hyde1(hyde, questions, out_root, force) -> None:
    pending = _pending_questions_for_arm(questions, "dense_hyde", out_root, force)
    if not pending:
        print("  HyDE1 prewarm: all docs already on disk.")
        return
    print(f"  HyDE1 prewarm: {len(pending)} questions ...", flush=True)
    t0 = time.time()
    _ = hyde.generate_batch([q["question"] for q in pending])
    cs = hyde.cost_summary()
    print(f"  HyDE1 prewarm done in {time.time()-t0:.1f}s: api={cs['api_calls']} "
          f"hits={cs['cache_hits']} cost=${cs['total_cost_usd']:.4f}", flush=True)


def _prewarm_semantic(hyde_semantic, questions, ctx_by_stt, out_root, force) -> None:
    pending = _pending_questions_for_arm(questions, "dense_hyde_semantic", out_root, force)
    if not pending:
        print("  HyDE-semantic prewarm: all docs already on disk.")
        return
    triples = [
        (q["question"], ctx_by_stt[q["stt"]].frame_text, ctx_by_stt[q["stt"]].context_key_ids)
        for q in pending
    ]
    print(f"  HyDE-semantic prewarm: {len(triples)} questions "
          f"(concurrency={hyde_semantic.concurrency}) ...", flush=True)
    t0 = time.time()
    _ = hyde_semantic.generate_batch(triples)
    cs = hyde_semantic.cost_summary()
    print(f"  HyDE-semantic prewarm done in {time.time()-t0:.1f}s: api={cs['api_calls']} "
          f"hits={cs['cache_hits']} cost=${cs['total_cost_usd']:.4f}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stt", type=str, default="")
    p.add_argument("--pilot-50", action="store_true",
                   help="Use exp 08's stratified 50-question pilot (identical strata).")
    p.add_argument("--arms", type=str, default=",".join(ARMS))
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--hyde-model", type=str, default=HYDE1_CFG["model"])
    p.add_argument("--cost-cap", type=float, default=0.50)
    p.add_argument("--skip-prewarm", action="store_true")
    args = p.parse_args()

    arms_subset = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms_subset:
        if a not in ARMS:
            print(f"ERROR: unknown arm {a!r}; valid: {ARMS}", file=sys.stderr)
            return 2

    if args.pilot_50:
        pilot = _load_pilot_50_from_exp08()
        stt_list = pilot["stt_list"]
        print(f"Pilot subset    : n={pilot['n']} seed={pilot['seed']} "
              f"quotas={pilot['quotas']} (reused from exp 08)")
    elif args.stt:
        stt_list = _parse_stt(args.stt)
    else:
        stt_list = []
    questions = _load_questions(stt_list)

    need_hyde1 = "dense_hyde" in arms_subset
    need_semantic = "dense_hyde_semantic" in arms_subset

    print(f"Probe size      : {len(questions)} questions")
    print(f"Arms            : {arms_subset}")
    print(f"Dense cfg       : dense_k={DENSE_CFG['dense_k']} "
          f"adapter={SHARED['adapter_path']} index={SHARED['dense_index']}")

    est = (0.0005 * len(questions) if need_hyde1 else 0.0) + \
          (0.0005 * len(questions) if need_semantic else 0.0)
    if est > 0:
        print(f"Estimated cost  : ~${est:.4f} (all cache misses)")
        if est > args.cost_cap:
            print(f"ABORT: est ${est:.4f} > --cost-cap ${args.cost_cap:.4f}", file=sys.stderr)
            return 3

    # Build semantic contexts up front (pure, deterministic, no network).
    ctx_by_stt: dict[int, object] = {}
    if need_semantic:
        from runtime.retrievers.semantic_context import build_semantic_context
        print("Building semantic contexts (concept frames) ...", flush=True)
        n_match = 0
        for q in questions:
            ctx = build_semantic_context(q["question"])
            ctx_by_stt[q["stt"]] = ctx
            n_match += int(ctx.concept_match)
        print(f"  concept_match: {n_match}/{len(questions)} "
              f"({n_match/max(len(questions),1):.1%})", flush=True)

    from src.retrieval.hyde import OpenAIHydeGenerator
    from src.retrieval.hyde_semantic import OpenAISemanticHydeGenerator
    from src.retrieval.pipeline import V5RetrievalPipeline

    hyde1 = OpenAIHydeGenerator(**{**HYDE1_CFG, "model": args.hyde_model}) if need_hyde1 else None
    hyde_sem = (
        OpenAISemanticHydeGenerator(**{**HYDE_SEMANTIC_CFG, "model": args.hyde_model})
        if need_semantic else None
    )
    if hyde1:
        print(f"  HyDE1 cache    : {hyde1.cache_dir}")
    if hyde_sem:
        print(f"  HyDE-sem sha   : {hyde_sem.prompt_sha[:12]}  cache: {hyde_sem.cache_dir}")

    print("Warming BGE-M3 (shared) ...", flush=True)
    pipe_dense = V5RetrievalPipeline(**DENSE_CFG)
    _ = pipe_dense.embed_model

    pipe_hyde = None
    if need_hyde1:
        pipe_hyde = V5RetrievalPipeline(hyde=hyde1, **DENSE_CFG)
        pipe_hyde._embed_model = pipe_dense._embed_model
    pipe_sem = None
    if need_semantic:
        pipe_sem = V5RetrievalPipeline(hyde_semantic=hyde_sem, **DENSE_CFG)
        pipe_sem._embed_model = pipe_dense._embed_model

    out_root = EXP_DIR / "results"
    if not args.skip_prewarm:
        if hyde1 is not None:
            _prewarm_hyde1(hyde1, questions, out_root, args.force)
        if hyde_sem is not None:
            _prewarm_semantic(hyde_sem, questions, ctx_by_stt, out_root, args.force)

    counts = {a: {"done": 0, "skipped": 0, "failed": 0, "latencies": []} for a in arms_subset}
    prov = {"concept_match": 0, "fallback": 0}
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
                    record_extra: dict = {}
                    if arm == "dense":
                        ans = pipe_dense.retrieve_dense_only(q["question"], top_k=DENSE_TOP_K)
                        cfg = {**DENSE_CFG, "mode": "dense_only"}
                    elif arm == "dense_hyde":
                        ans = pipe_hyde.retrieve_dense_only_hyde(q["question"], top_k=DENSE_TOP_K)
                        cfg = {**DENSE_CFG, "mode": "dense_only_hyde",
                               "hyde": {**HYDE1_CFG, "model": args.hyde_model}}
                    elif arm == "dense_hyde_semantic":
                        ctx = ctx_by_stt[stt]
                        ans = pipe_sem.retrieve_dense_only_hyde_semantic(
                            q["question"], ctx.frame_text, ctx.context_key_ids, top_k=DENSE_TOP_K)
                        cfg = {**DENSE_CFG, "mode": "dense_only_hyde_semantic",
                               "hyde_semantic": {**HYDE_SEMANTIC_CFG, "model": args.hyde_model}}
                        record_extra["semantic_context"] = {
                            "concept_match": ctx.concept_match,
                            "n_concepts": ctx.n_concepts,
                            "n_kg_entities": ctx.n_kg_entities,
                            "laws": ctx.laws,
                            "frame_text": ctx.frame_text,
                        }
                        prov["concept_match"] += int(ctx.concept_match)
                        prov["fallback"] += int(not ctx.concept_match)
                    else:
                        raise RuntimeError(f"unreachable arm={arm!r}")

                    elapsed = round(time.time() - t, 3)
                    counts[arm]["latencies"].append(elapsed)
                    _write_record(out_path, {
                        "arm": arm, "stt": stt, "question": q["question"],
                        "gold_citations_raw": q.get("gold_citations_raw"),
                        "config_used": cfg, "retrieval_only": asdict(ans),
                        **record_extra,
                    })
                    counts[arm]["done"] += 1
                    if args.verbose or i % 10 == 0:
                        print(f"  [{arm:<20} {i:>3}/{len(questions)}] stt={stt} "
                              f"({elapsed:.1f}s, final={len(ans.final_article_ids)})", flush=True)
                except Exception as e:  # noqa: BLE001
                    counts[arm]["failed"] += 1
                    print(f"  X [{arm} stt={stt}] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                    _write_record(out_root / arm / f"A{stt}.error.json",
                                  {"arm": arm, "stt": stt, "error": f"{type(e).__name__}: {e}"})
    finally:
        pipe_dense.close()
        if pipe_hyde is not None:
            pipe_hyde.close()
        if pipe_sem is not None:
            pipe_sem.close()

    print("\n" + "=" * 78 + "\nPer-arm summary\n" + "=" * 78)
    for arm in arms_subset:
        c = counts[arm]
        avg = round(mean(c["latencies"]), 3) if c["latencies"] else None
        line = (f"  {arm:<20} done={c['done']} skipped={c['skipped']} "
                f"failed={c['failed']} avg_latency={avg}s")
        if arm == "dense_hyde_semantic":
            line += f"  concept_match={prov['concept_match']} fallback={prov['fallback']}"
        print(line)
    if hyde_sem is not None:
        cs = hyde_sem.cost_summary()
        print(f"  HyDE-semantic cost: api={cs['api_calls']} hits={cs['cache_hits']} "
              f"total=${cs['total_cost_usd']:.4f}")
    print(f"Total wall time: {time.time() - t_total:.1f}s")
    return 0 if all(c["failed"] == 0 for c in counts.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
