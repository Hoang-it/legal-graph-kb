"""Experiment 14 runner — FAIR-PROMPT re-test of grounded + semantic HyDE.

Re-runs the grounding arms of exp 09 (HyDE2, clause-text grounding) and
exp 13 (semantic, concept-frame grounding) with their prompts brought to
**byte-parity** with HyDE1's skeleton — same role, same ~20-term BHXH
vocabulary scaffold, same prohibitions, same examples — the ONLY delta being
the grounding block, so the comparison isolates the grounding variable
(prompt scaffold held constant; phrase-level vocabulary reuse allowed).

The parity prompts live as OVERRIDES under
``experiments/14_hyde_fair_prompts/prompts_override/`` and are selected via
``LEGAL_KG_PROMPTS_DIR`` (set at import time below). HyDE1 has NO override →
``load_prompt`` falls back to the canonical (frozen) ``hyde_generate.md``, so
``dense_hyde`` here is bit-identical to the exp 08/09/13 bar and reuses the
``artifacts/hyde/`` cache for $0. The grounded/semantic parity prompts have a
new ``prompt_sha`` → fresh cache entries (no collision with exp 09/13 docs).

Canonical prompts under ``prompts/`` and the frozen exp 09/13 records are
left UNTOUCHED — this is a new experiment, per the project rule against
mutating canonical prompts that committed baselines depend on.

Arms (retrieval-only, article granularity, TUNED stack — same dense config
as exp 08/09/13 so numbers are comparable):

- ``dense``               → retrieve_dense_only (raw question)
- ``dense_hyde``          → retrieve_dense_only_hyde (HyDE1 — the bar, canonical prompt)
- ``dense_hyde2``         → retrieve_dense_only_hyde2 (grounded — PARITY prompt)
- ``dense_hyde_semantic`` → retrieve_dense_only_hyde_semantic (semantic — PARITY prompt)

    python -m scripts.exp14_run --pilot-50      # 4 arms on exp 08's stratified pilot-50
    python -m scripts.exp14_metrics             # in_corpus headline + S1/S2 per challenger

Cache-aware: generation is inline (no prewarm); re-runs are $0.
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

EXP_DIR = _REPO / "experiments" / "14_hyde_fair_prompts"
OVERRIDE_DIR = EXP_DIR / "prompts_override"

# Select the PARITY prompts for the grounded + semantic arms. HyDE1 has no
# override file here → src.prompts.load_prompt falls back to the canonical
# prompts/runtime/hyde_generate.md (the frozen bar). Must be set BEFORE the
# generators load their prompts (they do so at __init__).
os.environ["LEGAL_KG_PROMPTS_DIR"] = str(OVERRIDE_DIR)

# Reuse exp 09's tuned dense config + helpers verbatim so dense / dense_hyde
# are byte-comparable to exp 08/09/13.
from scripts.exp09_run import (  # noqa: E402
    DENSE_CFG,
    DENSE_TOP_K,
    HYDE1_CFG,
    HYDE2_CFG,
    HYDE2_SEED_K,
    SHARED,
    _load_pilot_50_from_exp08,
    _load_questions,
    _parse_stt,
    _write_record,
)

HYDE_SEMANTIC_CFG = {
    "model": "gpt-4o-mini",
    "n": 1,
    "max_tokens": 700,
    "temperature": 0.0,
    "concurrency": 5,
    "prompt_path": "runtime/hyde_generate_semantic.md",
}

ARMS = ("dense", "dense_hyde", "dense_hyde2", "dense_hyde_semantic")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stt", type=str, default="")
    p.add_argument("--pilot-50", action="store_true",
                   help="Use exp 08's stratified 50-question pilot (identical strata as exp 13).")
    p.add_argument("--arms", type=str, default=",".join(ARMS))
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--hyde-model", type=str, default=HYDE1_CFG["model"])
    p.add_argument("--cost-cap", type=float, default=0.50)
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
              f"quotas={pilot['quotas']} (reused from exp 08 → identical strata as exp 13)")
    elif args.stt:
        stt_list = _parse_stt(args.stt)
    else:
        stt_list = []
    questions = _load_questions(stt_list)

    need_hyde1 = "dense_hyde" in arms_subset
    need_hyde2 = "dense_hyde2" in arms_subset
    need_semantic = "dense_hyde_semantic" in arms_subset

    print(f"Override dir    : {OVERRIDE_DIR}")
    print(f"Probe size      : {len(questions)} questions")
    print(f"Arms            : {arms_subset}")
    print(f"Dense cfg       : dense_k={DENSE_CFG['dense_k']} top_k={DENSE_TOP_K} "
          f"adapter={SHARED['adapter_path']} index={SHARED['dense_index']}")

    # Cost: only the two challenger arms hit OpenAI on cache-miss (HyDE1 cache
    # from exp 08 → $0). ~$0.0005/question/challenger-arm.
    est = (0.0005 * len(questions) if need_hyde2 else 0.0) + \
          (0.0005 * len(questions) if need_semantic else 0.0)
    if est > 0:
        print(f"Estimated cost  : ~${est:.4f} (all cache misses; re-runs $0)")
        if est > args.cost_cap:
            print(f"ABORT: est ${est:.4f} > --cost-cap ${args.cost_cap:.4f}", file=sys.stderr)
            return 3

    # Build semantic contexts up front (pure, deterministic, no network). The
    # frame builder is unchanged from exp 13 — only the PROMPT differs here.
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
    from src.retrieval.hyde2 import OpenAIGroundedHydeGenerator
    from src.retrieval.hyde_semantic import OpenAISemanticHydeGenerator
    from src.retrieval.pipeline import V5RetrievalPipeline

    hyde1 = OpenAIHydeGenerator(**{**HYDE1_CFG, "model": args.hyde_model}) if need_hyde1 else None
    hyde2 = (OpenAIGroundedHydeGenerator(**{**HYDE2_CFG, "model": args.hyde_model})
             if need_hyde2 else None)
    hyde_sem = (OpenAISemanticHydeGenerator(**{**HYDE_SEMANTIC_CFG, "model": args.hyde_model})
                if need_semantic else None)

    # Sanity: confirm WHICH prompt file each generator resolved to. The two
    # challengers MUST point at the override dir; HyDE1 MUST stay canonical.
    if hyde1:
        print(f"  HyDE1 prompt   : {hyde1.prompt_source_path}  (expect canonical prompts/)")
    if hyde2:
        print(f"  HyDE2 prompt   : {hyde2.prompt_source_path}  sha={hyde2.prompt_sha[:12]}")
        assert "prompts_override" in str(hyde2.prompt_source_path), \
            "HyDE2 did not resolve to the parity override — check LEGAL_KG_PROMPTS_DIR."
    if hyde_sem:
        print(f"  semantic prompt: {hyde_sem.prompt_source_path}  sha={hyde_sem.prompt_sha[:12]}")
        assert "prompts_override" in str(hyde_sem.prompt_source_path), \
            "semantic did not resolve to the parity override — check LEGAL_KG_PROMPTS_DIR."

    print("Warming BGE-M3 (shared across arms) ...", flush=True)
    pipe_dense = V5RetrievalPipeline(**DENSE_CFG)
    _ = pipe_dense.embed_model

    pipe_hyde = pipe_hyde2 = pipe_sem = None
    if need_hyde1:
        pipe_hyde = V5RetrievalPipeline(hyde=hyde1, **DENSE_CFG)
        pipe_hyde._embed_model = pipe_dense._embed_model
    if need_hyde2:
        pipe_hyde2 = V5RetrievalPipeline(hyde2=hyde2, hyde2_seed_k=HYDE2_SEED_K, **DENSE_CFG)
        pipe_hyde2._embed_model = pipe_dense._embed_model
    if need_semantic:
        pipe_sem = V5RetrievalPipeline(hyde_semantic=hyde_sem, **DENSE_CFG)
        pipe_sem._embed_model = pipe_dense._embed_model

    out_root = EXP_DIR / "results"
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
                               "hyde": {**HYDE1_CFG, "model": args.hyde_model},
                               "prompt_source": "canonical"}
                    elif arm == "dense_hyde2":
                        ans = pipe_hyde2.retrieve_dense_only_hyde2(q["question"], top_k=DENSE_TOP_K)
                        cfg = {**DENSE_CFG, "mode": "dense_only_hyde2", "seed_k": HYDE2_SEED_K,
                               "hyde2": {**HYDE2_CFG, "model": args.hyde_model},
                               "prompt_source": "parity_override"}
                    elif arm == "dense_hyde_semantic":
                        ctx = ctx_by_stt[stt]
                        ans = pipe_sem.retrieve_dense_only_hyde_semantic(
                            q["question"], ctx.frame_text, ctx.context_key_ids, top_k=DENSE_TOP_K)
                        cfg = {**DENSE_CFG, "mode": "dense_only_hyde_semantic",
                               "hyde_semantic": {**HYDE_SEMANTIC_CFG, "model": args.hyde_model},
                               "prompt_source": "parity_override"}
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
        for pp in (pipe_hyde, pipe_hyde2, pipe_sem):
            if pp is not None:
                pp.close()

    print("\n" + "=" * 78 + "\nPer-arm summary\n" + "=" * 78)
    for arm in arms_subset:
        c = counts[arm]
        avg = round(mean(c["latencies"]), 3) if c["latencies"] else None
        line = (f"  {arm:<20} done={c['done']} skipped={c['skipped']} "
                f"failed={c['failed']} avg_latency={avg}s")
        if arm == "dense_hyde_semantic":
            line += f"  concept_match={prov['concept_match']} fallback={prov['fallback']}"
        print(line)
    for gen, name in ((hyde2, "HyDE2-fair"), (hyde_sem, "semantic-fair")):
        if gen is not None:
            cs = gen.cost_summary()
            print(f"  {name:<14} cost: api={cs['api_calls']} hits={cs['cache_hits']} "
                  f"total=${cs['total_cost_usd']:.4f}")
    print(f"Total wall time: {time.time() - t_total:.1f}s")
    return 0 if all(c["failed"] == 0 for c in counts.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
