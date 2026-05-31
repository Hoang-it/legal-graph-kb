"""Experiment 08 dry-run — single-question HyDE eyeball + manual GATE.

Loads stt=2 (or `--stt`) from data/eval/questions_200.json, then:

1. Constructs `OpenAIHydeGenerator` (gpt-4o-mini default).
2. Generates 1 hypothetical legal-document passage (cache-aware).
   Prints the full text + per-call usage (prompt/completion/cached tokens)
   + cost in USD + the OpenAI snapshot id returned (e.g.
   ``gpt-4o-mini-2024-07-18``) so the user can inspect:
     - style matches formal legal text (not casual chat / not Q-and-A);
     - NO 'Điều X', 'Khoản Y', numbered citations;
     - NO personal names / dates / numbers from the question;
     - topic is on-target.
3. Runs BGE-M3 LoRA dense search both ways:
     - vanilla: encode the raw question;
     - HyDE:    encode the hypothetical doc.
   Prints top-12 article-deduped lists side-by-side, plus the same
   reference column from exp 06's full_rerank/A<stt>.json (frozen
   baseline) when available.
4. Prints delta: gold article rank in HyDE-top-12 vs vanilla-top-12;
   articles added by HyDE; articles removed; Jaccard overlap.
5. Re-runs HyDE end-to-end so the user can confirm the second invocation
   hits the disk cache (no API call, no cost).

Hand off to user with the printed report. GATE checklist:
  - [ ] Hypothetical style matches formal legal text
  - [ ] NO 'Điều X', 'Khoản Y', numbered citations in the doc
  - [ ] NO personal names / dates / specific numbers from the question
  - [ ] Topic relevant to question's legal area
  - [ ] Gold article (L58_2014.A2 for stt=2) still in HyDE top-12
  - [ ] No suspiciously off-topic articles climbing into top-3

If any fail: revise prompts/runtime/hyde_generate.md, re-run, loop.

The OpenAI call is real — no mocking, no synthetic fixtures. Requires
$OPENAI_API_KEY in env or .env. Single dry-run call costs ~$0.0005.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

QUESTIONS_PATH = _REPO / "data" / "eval" / "questions_200.json"
EXP06_FULL_REF_DIR = (
    _REPO / "experiments" / "06_retrieval_dense_vs_full" / "results" / "full_rerank"
)

# Match exp 06 dense settings so vanilla-vs-HyDE is apples-to-apples on the
# same dense_k budget. The reference A<stt>.json files were produced with
# dense_k=30.
SHARED_CFG = {
    "adapter_path": "models/bge-m3-bhxh-lora",
    "dense_index": "clause_vec_tuned",
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

COMPARE_K = 12


def _load_question(stt: int) -> dict:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    for q in questions:
        if int(q["stt"]) == stt:
            return q
    raise KeyError(f"stt={stt} not found in {QUESTIONS_PATH}")


def _load_reference_dense_top12(stt: int) -> list[str]:
    """Read exp06 dense_article_ids from full_rerank/A<stt>.json (if exists)."""
    ref = EXP06_FULL_REF_DIR / f"A{stt}.json"
    if not ref.exists():
        return []
    rec = json.loads(ref.read_text(encoding="utf-8"))
    ids = (rec.get("retrieval_only") or {}).get("dense_article_ids") or []
    return list(ids)[:COMPARE_K]


def _rank_of(article_id: str, lst: list[str]) -> int | None:
    for i, a in enumerate(lst, start=1):
        if a == article_id:
            return i
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stt", type=int, default=2,
                   help="Question stt to use as the dry-run probe (default 2).")
    p.add_argument("--gold", type=str, default="L58_2014.A2",
                   help="Gold article id to track for rank-delta (default L58_2014.A2 — stt=2).")
    p.add_argument("--model", type=str, default="gpt-4o-mini",
                   help="OpenAI model id (default gpt-4o-mini).")
    p.add_argument("--n", type=int, default=1,
                   help="N hypothetical docs to generate (plan D3 default 1).")
    p.add_argument("--max-tokens", type=int, default=700)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    print("=" * 78)
    print(f"Experiment 08 dry-run (stt={args.stt}, gold={args.gold}, model={args.model})")
    print("=" * 78)

    q = _load_question(args.stt)
    print()
    print("[1/5] Question")
    print("-" * 78)
    print(q["question"])
    print(f"  gold_citations_raw: {q.get('gold_citations_raw')}")
    print()

    from src.retrieval.hyde import OpenAIHydeGenerator

    print("[2/5] HyDE generator")
    print("-" * 78)
    hyde = OpenAIHydeGenerator(
        model=args.model,
        n=args.n,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(f"  model        : {hyde.model}")
    print(f"  prompt_path  : {hyde.prompt_source_path}")
    print(f"  prompt_sha   : {hyde.prompt_sha}")
    print(f"  cache_dir    : {hyde.cache_dir}")
    print(f"  cache_key    : {hyde._cache_key(q['question'])}")

    # Construct TWO pipeline instances so the vanilla-vs-HyDE comparison
    # is honest. A single pipeline with hyde=hyde would route
    # retrieve_dense_only through the HyDE encoder too (the retriever's
    # query_encoder is per-instance, not per-method), making both rows
    # identical. Mirrors the two-pipeline pattern in scripts/exp08_run.py.
    print()
    print("[3/5] V5RetrievalPipeline (warm-up BGE-M3 + Neo4j) — 2 instances")
    print("-" * 78)
    from src.retrieval.pipeline import V5RetrievalPipeline

    pipe_plain = V5RetrievalPipeline(**SHARED_CFG)  # hyde=None
    pipe_hyde = V5RetrievalPipeline(hyde=hyde, **SHARED_CFG)
    _ = pipe_plain.embed_model
    _ = pipe_hyde.embed_model

    # --- First HyDE run: generates + caches (or cache-hits if re-run) ---
    print()
    print("[4/5] HyDE generation + dense comparison")
    print("-" * 78)
    t0 = time.time()
    docs = hyde.generate(q["question"])
    t_first = time.time() - t0
    cs1 = hyde.cost_summary()
    print(f"  generate() wall time  : {t_first:.2f}s")
    print(f"  api_calls / cache_hits: {cs1['api_calls']} / {cs1['cache_hits']}")
    print(f"  prompt / completion t : {cs1['prompt_tokens']} / {cs1['completion_tokens']}")
    print(f"  cost so far           : ${cs1['total_cost_usd']:.6f}")
    print()
    for i, doc in enumerate(docs, 1):
        print(f"--- Hypothetical doc #{i} (len={len(doc)} chars) ---")
        print(doc)
        print()

    try:
        # Vanilla dense search (raw-question embedding) — uses pipe_plain
        # whose retriever has query_encoder=None, so embed_model.encode is
        # called directly on the question.
        t = time.time()
        vanilla = pipe_plain.retrieve_dense_only(
            q["question"], top_k=SHARED_CFG["dense_k"]
        )
        t_vanilla = time.time() - t

        # HyDE dense search — uses pipe_hyde whose retriever's
        # query_encoder is the closure from hyde.embed_query_callable().
        # generate() inside the closure cache-hits because we populated above.
        t = time.time()
        hyde_dense = pipe_hyde.retrieve_dense_only_hyde(
            q["question"], top_k=SHARED_CFG["dense_k"]
        )
        t_hyde = time.time() - t

        vanilla_top = vanilla.final_article_ids[:COMPARE_K]
        hyde_top = hyde_dense.final_article_ids[:COMPARE_K]
        ref_top = _load_reference_dense_top12(args.stt)

        max_rows = max(len(vanilla_top), len(hyde_top), len(ref_top), COMPARE_K)
        print(f"  {'rank':>4}  {'vanilla(now)':<22}  {'HyDE(now)':<22}  {'exp06 ref':<22}")
        print(f"  {'-'*4}  {'-'*22}  {'-'*22}  {'-'*22}")
        for i in range(max_rows):
            v = vanilla_top[i] if i < len(vanilla_top) else ""
            h = hyde_top[i] if i < len(hyde_top) else ""
            r = ref_top[i] if i < len(ref_top) else ""
            tag_v = " *" if v == args.gold else "  "
            tag_h = " *" if h == args.gold else "  "
            tag_r = " *" if r == args.gold else "  "
            print(f"  {i + 1:>4}  {v:<20}{tag_v}  {h:<20}{tag_h}  {r:<20}{tag_r}")
        print(f"  latency vanilla={t_vanilla:.2f}s   HyDE={t_hyde:.2f}s   "
              "(HyDE includes embedding-of-generated-doc; LLM gen was cache-hit)")

        # Delta summary
        v_set = set(vanilla_top)
        h_set = set(hyde_top)
        added = [a for a in hyde_top if a not in v_set]
        removed = [a for a in vanilla_top if a not in h_set]
        rank_v = _rank_of(args.gold, vanilla_top)
        rank_h = _rank_of(args.gold, hyde_top)
        rank_r = _rank_of(args.gold, ref_top)
        print()
        print("  --- Delta ---")
        print(f"  Gold {args.gold!r} rank:  exp06_ref={rank_r}  vanilla_now={rank_v}  HyDE_now={rank_h}")
        print(f"  Added by HyDE   : {added}")
        print(f"  Removed by HyDE : {removed}")
        print(f"  Jaccard(vanilla, HyDE) = "
              f"{(len(v_set & h_set) / max(1, len(v_set | h_set))):.3f}")

        # --- Cache-hit confirmation ---
        print()
        print("[5/5] Cache-hit confirmation (re-run HyDE generate())")
        print("-" * 78)
        t = time.time()
        docs2 = hyde.generate(q["question"])
        t_second = time.time() - t
        cs2 = hyde.cost_summary()
        print(f"  generate() second wall time : {t_second:.4f}s")
        print(f"  api_calls / cache_hits      : {cs2['api_calls']} / {cs2['cache_hits']}")
        print(f"  total cost (unchanged?)     : ${cs2['total_cost_usd']:.6f}")
        same = docs == docs2
        speedup = (t_first / max(t_second, 1e-9))
        print(f"  identical to first call     : {same}")
        print(f"  speedup vs first call       : {speedup:.1f}×")
        # If cache hit: t_second should be small (<50ms) AND cost+api_calls
        # should be unchanged vs cs1.
        no_new_call = cs2["api_calls"] == cs1["api_calls"]
        cost_unchanged = abs(cs2["total_cost_usd"] - cs1["total_cost_usd"]) < 1e-12
        if no_new_call and cost_unchanged and same:
            print("  OK — cache hit confirmed (no new API call, no cost delta).")
        else:
            print("  WARNING: cache miss — investigate cache key / file write.")

    finally:
        pipe_plain.close()
        pipe_hyde.close()

    print()
    print("=" * 78)
    print("User GATE checklist:")
    print("  [ ] Hypothetical style matches formal legal text")
    print("  [ ] NO 'Điều X', 'Khoản Y', numbered citations in the doc")
    print("  [ ] NO personal names / dates / specific numbers from the question")
    print("  [ ] Topic is relevant to the question's legal area")
    print(f"  [ ] Gold {args.gold} still in HyDE top-{COMPARE_K}")
    print("  [ ] No suspiciously off-topic articles climbing into top-3")
    print()
    print("If any fail → revise prompts/runtime/hyde_generate.md and re-run.")
    print(f"Final cost: ${hyde.cost_summary()['total_cost_usd']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
