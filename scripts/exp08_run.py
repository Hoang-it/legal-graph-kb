"""Experiment 08 runner — HyDE retrieval-only 4-arm A/B (gpt-4o-mini).

Arms (all retrieval-only, no LLM generator):

- ``dense``             → :meth:`V5RetrievalPipeline.retrieve_dense_only`
                          (BGE-M3 LoRA + clause_vec_tuned, dense_k=100)
- ``dense_hyde``        → :meth:`V5RetrievalPipeline.retrieve_dense_only_hyde`
                          (same, but dense query = embedding of gpt-4o-mini
                          hypothetical doc)
- ``full_rerank``       → :meth:`V5RetrievalPipeline.retrieve_only` on a
                          pipeline WITHOUT hyde (full v5 scaled, mirrors exp 07)
- ``full_rerank_hyde``  → :meth:`V5RetrievalPipeline.retrieve_only` on a
                          pipeline WITH hyde (same pipeline, dense channel
                          uses HyDE embedding)

Two pipeline instances are constructed (one with ``hyde=None``, one with
``hyde=OpenAIHydeGenerator(...)``). They share the HyDE disk cache so
``dense_hyde`` and ``full_rerank_hyde`` get the same hypothetical doc
per question and never pay twice.

``--pilot-50`` flag selects a stratified 50-question subset (in_corpus /
mixed / ooc / unparseable proportional to the full dataset, seed=0).
The selection is persisted at
``experiments/08_hyde_retrieval/pilot_50_stt.json`` so re-runs and the
metrics + funnel scripts converge on the same subset.

Records land under
``experiments/08_hyde_retrieval/results/<arm>/A<stt>.json``. Idempotent
+ ``--force`` overwrite + ``--stt`` subset (same shape as exp07_run.py).

Cost summary is printed at the end: per-arm + total. Aborts pre-flight
if estimated cost exceeds ``--cost-cap`` (default $0.50, well above the
~$0.025 plan estimate for pilot 50).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
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

EXP_DIR = _REPO / "experiments" / "08_hyde_retrieval"
QUESTIONS_PATH = _REPO / "data" / "eval" / "questions_200.json"
PILOT_50_PATH = EXP_DIR / "pilot_50_stt.json"

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

# HyDE generator defaults per plan §D1–D8
HYDE_CFG = {
    "model": "gpt-4o-mini",
    "n": 1,
    "max_tokens": 700,
    "temperature": 0.0,
    "concurrency": 5,
    "prompt_path": "runtime/hyde_generate.md",
}

ARMS = ("dense", "dense_hyde", "full_rerank", "full_rerank_hyde")
HYDE_ARMS = {"dense_hyde", "full_rerank_hyde"}


# ---------------------------------------------------------------------------
# Stratified sampler — keep in sync with categorize() in exp08_metrics.py
# ---------------------------------------------------------------------------

_RE_CODE = re.compile(r"\d+/\d{4}/(?:QH\d+|N[ĐD]-CP|NQ-CP|TT-[A-Z]+|CP|TTg)")


def _categorize(raw, in_corpus_codes: set[str]) -> str:
    if not raw:
        return "unparseable"
    if isinstance(raw, list):
        raw = "\n".join(str(x) for x in raw)
    hits = _RE_CODE.findall(raw)
    if not hits:
        return "unparseable"
    in_kg = sum(1 for h in hits if h in in_corpus_codes)
    if in_kg == len(hits):
        return "in_corpus"
    if in_kg == 0:
        return "ooc"
    return "mixed"


def build_or_load_pilot_50(seed: int = 0) -> dict:
    """Return the canonical pilot-50 selection.

    First call writes ``pilot_50_stt.json`` with seed + per-stratum quotas
    + sorted stt list. Subsequent calls load the cached file — re-runs
    and metrics/funnel scripts all see the same 50 stt.
    """
    if PILOT_50_PATH.exists():
        return json.loads(PILOT_50_PATH.read_text(encoding="utf-8"))

    from src.legal_metadata import load_law_metadata

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    in_corpus_codes = {m.full_id for m in load_law_metadata().values()}
    by_cat: dict[str, list[int]] = {"in_corpus": [], "mixed": [], "ooc": [], "unparseable": []}
    for q in questions:
        by_cat[_categorize(q.get("gold_citations_raw"), in_corpus_codes)].append(q["stt"])
    n_total = 50
    n_dataset = len(questions)
    quotas: dict[str, int] = {}
    for c, ids in by_cat.items():
        if not ids:
            quotas[c] = 0
        else:
            quotas[c] = max(1, n_total * len(ids) // n_dataset)
    # Reconcile to exactly n_total by adjusting in_corpus (the dominant
    # stratum) up or down.
    while sum(quotas.values()) < n_total:
        quotas["in_corpus"] += 1
    while sum(quotas.values()) > n_total:
        quotas["in_corpus"] -= 1
    rng = random.Random(seed)
    chosen: list[int] = []
    for cat, k in quotas.items():
        if k <= 0:
            continue
        pool = sorted(by_cat[cat])
        chosen.extend(rng.sample(pool, min(k, len(pool))))
    chosen.sort()
    payload = {
        "seed": seed,
        "n": len(chosen),
        "n_dataset": n_dataset,
        "quotas": quotas,
        "stt_list": chosen,
    }
    PILOT_50_PATH.parent.mkdir(parents=True, exist_ok=True)
    PILOT_50_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


# ---------------------------------------------------------------------------
# CLI helpers
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


def _prewarm_hyde(
    hyde, questions: list[dict], hyde_arms_requested: list[str], out_root: Path, force: bool
) -> None:
    """Batch-generate HyDE docs for every question that any HyDE arm still
    needs to process. Populates the disk cache up-front so the per-arm
    per-question loop becomes a pure cache-read — cleaner cost accounting
    and lower wall time (asyncio.Semaphore(5) parallelises the API)."""
    pending_qs: dict[int, dict] = {}
    for arm in hyde_arms_requested:
        for q in _pending_questions_for_arm(questions, arm, out_root, force):
            pending_qs[q["stt"]] = q
    if not pending_qs:
        print("  HyDE prewarm: all docs already cached for pending arms.")
        return
    ordered = [pending_qs[s] for s in sorted(pending_qs)]
    print(f"  HyDE prewarm: generating for {len(ordered)} unique questions "
          f"(concurrency={hyde.concurrency}) ...", flush=True)
    t0 = time.time()
    _ = hyde.generate_batch([q["question"] for q in ordered])
    dt = time.time() - t0
    cs = hyde.cost_summary()
    print(f"  HyDE prewarm: done in {dt:.1f}s. "
          f"api_calls={cs['api_calls']} cache_hits={cs['cache_hits']} "
          f"cost=${cs['total_cost_usd']:.4f}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stt", type=str, default="",
                   help="Comma/range stt list (e.g. '1-5,10'); empty = full dataset (200) or pilot-50 if --pilot-50.")
    p.add_argument("--pilot-50", action="store_true",
                   help="Use stratified 50-question pilot subset "
                        "(experiments/08_hyde_retrieval/pilot_50_stt.json).")
    p.add_argument("--arms", type=str, default=",".join(ARMS),
                   help=f"Subset of {ARMS}; default = all four.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing records.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--hyde-model", type=str, default=HYDE_CFG["model"],
                   help="OpenAI model id for HyDE generator.")
    p.add_argument("--cost-cap", type=float, default=0.50,
                   help="Abort pre-flight if cost estimate exceeds this USD (default $0.50).")
    p.add_argument("--skip-prewarm", action="store_true",
                   help="Skip batched HyDE pre-generation; per-question lazy mode.")
    args = p.parse_args()

    arms_subset = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms_subset:
        if a not in ARMS:
            print(f"ERROR: unknown arm {a!r}; valid: {ARMS}", file=sys.stderr)
            return 2

    # Resolve question set.
    pilot_info: dict | None = None
    if args.pilot_50:
        pilot_info = build_or_load_pilot_50()
        stt_list = pilot_info["stt_list"]
        print(f"Pilot subset    : n={pilot_info['n']} seed={pilot_info['seed']} "
              f"quotas={pilot_info['quotas']}")
    elif args.stt:
        stt_list = _parse_stt(args.stt)
    else:
        stt_list = []
    questions = _load_questions(stt_list)

    hyde_arms_requested = [a for a in arms_subset if a in HYDE_ARMS]
    need_hyde = bool(hyde_arms_requested)

    print(f"Probe size      : {len(questions)} questions")
    print(f"Arms            : {arms_subset}")
    print(f"HyDE arms active: {hyde_arms_requested or '(none — skipping OpenAI)'}")
    print(f"Full cfg        : dense_k={FULL_CFG['dense_k']} sparse_k={FULL_CFG['sparse_k']} "
          f"top_after_fusion={FULL_CFG['top_after_fusion']} "
          f"rerank1_top_k={FULL_CFG['rerank1_top_k']} "
          f"rerank2_top_k={FULL_CFG['rerank2_top_k']} "
          f"per_seed_neighbors={FULL_CFG['per_seed_neighbors']}")
    if need_hyde:
        # Pre-flight cost estimate (very rough: $0.0005/call at default config).
        # Both HyDE arms share the cache so it's 1 call per UNIQUE question.
        est_cost = 0.0005 * len(questions)
        print(f"HyDE cfg        : model={args.hyde_model} n={HYDE_CFG['n']} "
              f"max_tokens={HYDE_CFG['max_tokens']} temperature={HYDE_CFG['temperature']} "
              f"concurrency={HYDE_CFG['concurrency']}")
        print(f"Estimated cost  : ~${est_cost:.4f} (assuming all cache misses)")
        if est_cost > args.cost_cap:
            print(f"ABORT: estimated cost ${est_cost:.4f} exceeds --cost-cap ${args.cost_cap:.4f}",
                  file=sys.stderr)
            return 3

    from src.retrieval.hyde import OpenAIHydeGenerator
    from src.retrieval.pipeline import V5RetrievalPipeline

    hyde = None
    if need_hyde:
        cfg = {**HYDE_CFG, "model": args.hyde_model}
        hyde = OpenAIHydeGenerator(**cfg)
        print(f"  HyDE prompt_sha : {hyde.prompt_sha}")
        print(f"  HyDE cache_dir  : {hyde.cache_dir}")

    # Two pipeline instances — kept separate so per-arm config_snapshot
    # honestly reflects which arm produced each record. To avoid loading
    # BGE-M3 (~1.2 GB) and the reranker (~0.5 GB) twice on a small GPU,
    # we SHARE the weight-bearing components across the two instances.
    # The pipelines still differ in `query_encoder` (None vs HyDE closure)
    # via the lazy `retriever` property, which is what defines per-arm
    # behaviour — weight sharing has no functional effect.
    need_reranker = "full_rerank" in arms_subset or "full_rerank_hyde" in arms_subset
    print("Warming up BGE-M3 + reranker (shared across pipelines) ...", flush=True)
    pipe_plain = V5RetrievalPipeline(**FULL_CFG)
    _ = pipe_plain.embed_model
    if need_reranker:
        _ = pipe_plain.reranker.model

    pipe_hyde = None
    if need_hyde:
        pipe_hyde = V5RetrievalPipeline(hyde=hyde, **FULL_CFG)
        # Share weights — must happen BEFORE pipe_hyde.retriever is first
        # accessed so the lazy encoder closure captures the shared model.
        pipe_hyde._embed_model = pipe_plain._embed_model
        if need_reranker:
            pipe_hyde._reranker = pipe_plain._reranker
        print("  pipe_hyde: shared embed_model + reranker (no extra VRAM).",
              flush=True)

    out_root = EXP_DIR / "results"

    if hyde is not None and not args.skip_prewarm:
        _prewarm_hyde(hyde, questions, hyde_arms_requested, out_root, args.force)

    counts = {a: {"done": 0, "skipped": 0, "failed": 0, "latencies": []} for a in arms_subset}
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
                        cfg_used = {**DENSE_CFG, "hyde": {**HYDE_CFG, "model": args.hyde_model}}
                    elif arm == "full_rerank":
                        ans = pipe_plain.retrieve_only(q["question"])
                        cfg_used = FULL_CFG
                    elif arm == "full_rerank_hyde":
                        assert pipe_hyde is not None
                        ans = pipe_hyde.retrieve_only(q["question"])
                        cfg_used = {**FULL_CFG, "hyde": {**HYDE_CFG, "model": args.hyde_model}}
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
    print("=" * 78)
    print("Per-arm summary")
    print("=" * 78)
    for arm in arms_subset:
        c = counts[arm]
        avg = round(mean(c["latencies"]), 3) if c["latencies"] else None
        print(f"  {arm:<18} done={c['done']} skipped={c['skipped']} "
              f"failed={c['failed']} avg_latency={avg}s")
    print(f"Total wall time: {time.time() - t_total:.1f}s")

    if hyde is not None:
        cs = hyde.cost_summary()
        print()
        print("=" * 78)
        print("HyDE LLM cost summary (gpt-4o-mini)")
        print("=" * 78)
        print(f"  model            : {cs['model_id']}")
        print(f"  API calls (cold) : {cs['api_calls']}")
        print(f"  Cache hits       : {cs['cache_hits']}")
        print(f"  Prompt tokens    : {cs['prompt_tokens']:,}")
        print(f"  Completion tokens: {cs['completion_tokens']:,}")
        print(f"  Cached tokens    : {cs['cached_tokens']:,}")
        print(f"  TOTAL COST       : ${cs['total_cost_usd']:.6f}")
        if cs["total_cost_usd"] > args.cost_cap:
            print(f"  WARNING: cost ${cs['total_cost_usd']:.4f} exceeded --cost-cap "
                  f"${args.cost_cap:.4f}", file=sys.stderr)

    return 0 if all(c["failed"] == 0 for c in counts.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
