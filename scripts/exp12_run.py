"""Experiment 12 runner — HyDE × CypherWalk retrieval-only 2×2 factorial.

Follow-on to exp 11. The Cypher walk seeds from a raw-question vector
search; this experiment varies the seed (raw vs HyDE hypothetical-doc
embedding) to isolate whether seed quality changes the walk's effect on
retrieval — a weak-seed factor vs an intrinsic walk-behaviour factor.

2×2 factorial — seed {raw, hyde} × walk {off, on}, ALL on the *same*
vanilla `RagPipeline` / `clause_vec` stack as exp 11 (NOT the v5 tuned
index of exp 08), so the HyDE effect is isolated:

    dense_vanilla       raw seed, top-12, no walk      (= exp 11 baseline)
    dense_hyde          HyDE seed, top-12, no walk
    cypher_walk         raw seed (8) + walk → RRF 12   (= exp 11 arm)
    cypher_walk_hyde    HyDE seed (8) + walk → RRF 12  ← NEW combined arm

HyDE generator: OpenAI gpt-4o-mini, n=1, max_tokens=700, T=0, prompt
`runtime/hyde_generate.md` — identical config to exp 08, so its disk cache
(artifacts/hyde/) is reused for $0 on overlapping questions. `dense_hyde`
and `cypher_walk_hyde` share the same cached HyDE doc per question.

Same pilot subset as exp 11 (pilot_50_stt.json copied in), idempotent +
--force + --stt, cost-capped. Records at
experiments/12_hyde_cypher_walk/results/<arm>/A<stt>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

# Reuse the exp 11 run-helpers verbatim so the dense_vanilla / cypher_walk
# arms are byte-for-byte the same logic (only the seed differs for HyDE).
from scripts.exp11_run import (  # noqa: E402
    _dedupe,
    _load_questions,
    _parse_stt,
    _write_record,
    run_cypher_walk,
    run_dense_vanilla,
)

EXP_DIR = _REPO / "experiments" / "12_hyde_cypher_walk"
QUESTIONS_PATH = _REPO / "data" / "eval" / "questions_200.json"
PILOT_50_PATH = EXP_DIR / "pilot_50_stt.json"

ARMS = ("dense_vanilla", "dense_hyde", "cypher_walk", "cypher_walk_hyde")
HYDE_ARMS = {"dense_hyde", "cypher_walk_hyde"}
OPENAI_ARMS = {"dense_hyde", "cypher_walk", "cypher_walk_hyde"}

TOP_K = 12
SEED_K = 8
MAX_REPAIR_ROUNDS = 2

HYDE_CFG = {
    "model": "gpt-4o-mini",
    "n": 1,
    "max_tokens": 700,
    "temperature": 0.0,
    "concurrency": 5,
    "prompt_path": "runtime/hyde_generate.md",
    "cache_dir": str(_REPO / "artifacts" / "hyde"),
}


def _load_pilot() -> list[int]:
    if not PILOT_50_PATH.exists():
        print(f"FAIL: {PILOT_50_PATH} missing (copy it from exp 11).", file=sys.stderr)
        sys.exit(1)
    payload = json.loads(PILOT_50_PATH.read_text(encoding="utf-8"))
    print(f"Pilot subset    : n={payload['n']} seed={payload.get('seed')} "
          f"quotas={payload.get('quotas')}")
    return payload["stt_list"]


def run_dense_hyde(rag, encoder, question: str) -> dict:
    """HyDE seed top-12, no walk — mirrors run_dense_vanilla but the dense
    query is the HyDE hypothetical-doc embedding."""
    t0 = time.time()
    qvec = encoder(question)
    hits = rag.vector_search_by_vector(qvec, top_k=TOP_K)
    final_clause_ids = [h.clause_id for h in hits]
    final_article_ids = _dedupe(h.article_id for h in hits)
    return {
        "final_article_ids": final_article_ids,
        "final_clause_ids": final_clause_ids,
        "elapsed_s": round(time.time() - t0, 3),
        "provenance": {"n_vector": len(hits), "vector_clause_ids": final_clause_ids,
                       "seed_mode": "hyde"},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stt", type=str, default="", help="Comma/range stt; empty = pilot-50.")
    p.add_argument("--arms", type=str, default=",".join(ARMS))
    p.add_argument("--force", action="store_true")
    p.add_argument("--cypher-model", type=str, default=None)
    p.add_argument("--hyde-model", type=str, default=HYDE_CFG["model"])
    p.add_argument("--cost-cap", type=float, default=0.75)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    arms_subset = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms_subset:
        if a not in ARMS:
            print(f"ERROR: unknown arm {a!r}; valid: {ARMS}", file=sys.stderr)
            return 2

    stt_list = _parse_stt(args.stt) if args.stt else _load_pilot()
    questions = _load_questions(stt_list)

    need_hyde = any(a in HYDE_ARMS for a in arms_subset)
    need_openai = any(a in OPENAI_ARMS for a in arms_subset)
    print(f"Probe size      : {len(questions)} questions")
    print(f"Arms            : {arms_subset}")
    print(f"Config          : top_k={TOP_K} seed_k={SEED_K} max_repair_rounds={MAX_REPAIR_ROUNDS}")

    if need_openai:
        # HyDE: <=1 call/q (cached). Cypher walk: <=(repair+1) calls/q.
        cyp_q = len(questions) * (MAX_REPAIR_ROUNDS + 1) * (
            sum(1 for a in arms_subset if a in {"cypher_walk", "cypher_walk_hyde"}))
        hyde_q = len(questions) if need_hyde else 0
        est = 0.0005 * (cyp_q + hyde_q)
        print(f"Estimated cost  : ~${est:.4f} (worst case; HyDE cache reuse lowers it)")
        if est > args.cost_cap:
            print(f"ABORT: est ${est:.4f} > --cost-cap ${args.cost_cap:.4f}", file=sys.stderr)
            return 3

    from runtime.rag_query import RagPipeline

    print("Connecting Neo4j + warming BGE-M3 ...", flush=True)
    rag = RagPipeline()
    _ = rag.embed_model

    encoder = None
    hyde = None
    if need_hyde:
        from src.retrieval.hyde import OpenAIHydeGenerator

        hyde = OpenAIHydeGenerator(**{**HYDE_CFG, "model": args.hyde_model})
        print(f"  HyDE prompt_sha={hyde.prompt_sha[:12]}  cache_dir={hyde.cache_dir}")
        # Prewarm: batch-generate HyDE docs for all pilot questions (cache-aware).
        print("  HyDE prewarm ...", flush=True)
        _ = hyde.generate_batch([q["question"] for q in questions])
        cs = hyde.cost_summary()
        print(f"  HyDE prewarm done: api_calls={cs['api_calls']} cache_hits={cs['cache_hits']} "
              f"cost=${cs['total_cost_usd']:.4f}", flush=True)
        encoder = hyde.embed_query_callable(rag.embed_model)

    from runtime.retrievers.cypher_walk import CypherWalkRetriever

    retr_raw = retr_hyde = None
    if "cypher_walk" in arms_subset:
        retr_raw = CypherWalkRetriever(rag, top_k_seed=SEED_K, top_k_final=TOP_K,
                                       max_repair_rounds=MAX_REPAIR_ROUNDS,
                                       cypher_model=args.cypher_model)
    if "cypher_walk_hyde" in arms_subset:
        retr_hyde = CypherWalkRetriever(rag, top_k_seed=SEED_K, top_k_final=TOP_K,
                                        max_repair_rounds=MAX_REPAIR_ROUNDS,
                                        cypher_model=args.cypher_model,
                                        seed_query_encoder=encoder)

    out_root = EXP_DIR / "results"
    counts = {a: {"done": 0, "skipped": 0, "failed": 0, "latencies": []} for a in arms_subset}
    prov_tally = {a: {"cypher_used": 0, "fallback_used": 0} for a in arms_subset}
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
                    if arm == "dense_vanilla":
                        ro = run_dense_vanilla(rag, q["question"])
                        cfg = {"mode": "dense_vanilla", "top_k": TOP_K, "seed_mode": "raw"}
                    elif arm == "dense_hyde":
                        ro = run_dense_hyde(rag, encoder, q["question"])
                        cfg = {"mode": "dense_hyde", "top_k": TOP_K, "seed_mode": "hyde",
                               "hyde": {**HYDE_CFG, "model": args.hyde_model}}
                    elif arm == "cypher_walk":
                        ro = run_cypher_walk(retr_raw, q["question"])
                        cfg = {"mode": "cypher_walk", "top_k_final": TOP_K, "top_k_seed": SEED_K,
                               "seed_mode": "raw", "cypher_model": retr_raw.cypher_model}
                    elif arm == "cypher_walk_hyde":
                        ro = run_cypher_walk(retr_hyde, q["question"])
                        cfg = {"mode": "cypher_walk_hyde", "top_k_final": TOP_K, "top_k_seed": SEED_K,
                               "seed_mode": "hyde", "cypher_model": retr_hyde.cypher_model,
                               "hyde": {**HYDE_CFG, "model": args.hyde_model}}
                    else:
                        raise RuntimeError(f"unreachable arm={arm!r}")

                    if arm in {"cypher_walk", "cypher_walk_hyde"}:
                        if ro["provenance"]["cypher_used"]:
                            prov_tally[arm]["cypher_used"] += 1
                        if ro["provenance"]["fallback_used"]:
                            prov_tally[arm]["fallback_used"] += 1

                    counts[arm]["latencies"].append(ro["elapsed_s"])
                    _write_record(out_path, {
                        "arm": arm, "stt": stt, "question": q["question"],
                        "gold_citations_raw": q.get("gold_citations_raw"),
                        "config_used": cfg, "retrieval_only": ro,
                    })
                    counts[arm]["done"] += 1
                    if args.verbose or i % 10 == 0:
                        extra = (f" cyp_new={ro['provenance']['n_cypher_new']}"
                                 if arm in {"cypher_walk", "cypher_walk_hyde"}
                                 else f" final={len(ro['final_article_ids'])}")
                        print(f"  [{arm:<18} {i:>3}/{len(questions)}] stt={stt} "
                              f"({ro['elapsed_s']:.1f}s{extra})", flush=True)
                except Exception as e:  # noqa: BLE001
                    counts[arm]["failed"] += 1
                    print(f"  X [{arm} stt={stt}] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                    _write_record(out_root / arm / f"A{stt}.error.json",
                                  {"arm": arm, "stt": stt, "error": f"{type(e).__name__}: {e}"})
    finally:
        rag.close()

    print()
    print("=" * 78)
    print("Per-arm summary")
    print("=" * 78)
    for arm in arms_subset:
        c = counts[arm]
        avg = round(mean(c["latencies"]), 3) if c["latencies"] else None
        line = (f"  {arm:<18} done={c['done']} skipped={c['skipped']} "
                f"failed={c['failed']} avg_latency={avg}s")
        if arm in {"cypher_walk", "cypher_walk_hyde"}:
            n_run = c["done"] or 1
            line += (f"  cypher_used={prov_tally[arm]['cypher_used']} "
                     f"fallback={prov_tally[arm]['fallback_used']}")
        print(line)
    if hyde is not None:
        cs = hyde.cost_summary()
        print(f"  HyDE cost: api_calls={cs['api_calls']} cache_hits={cs['cache_hits']} "
              f"total=${cs['total_cost_usd']:.4f}")
    print(f"Total wall time: {time.time() - t_total:.1f}s")
    return 0 if all(c["failed"] == 0 for c in counts.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
