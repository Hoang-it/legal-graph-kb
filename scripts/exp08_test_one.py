"""Experiment 08 Phase 3 dry-run — single-question HyDE eyeball + manual GATE.

Loads stt=2 from data/eval/questions_200.json, then for the same question:

1. Loads Qwen 2.5 3B Instruct, prints pre/post-load VRAM (CUDA only).
2. Generates 1 hypothetical legal-document passage (cache-aware).
   Prints the full text so the user can inspect:
     - style matches formal legal text (not casual chat / not Q-and-A);
     - NO 'Điều X', 'Khoản Y', numbered citations;
     - NO personal names / dates / numbers from the question;
     - topic is on-target.
3. Runs BGE-M3 LoRA dense search both ways:
     - vanilla: encode the raw question;
     - HyDE:    encode the hypothetical doc.
   Prints top-12 article-deduped lists side-by-side.
4. Compares to experiments/06_retrieval_dense_vs_full/results/full_rerank/
   A2.json :: retrieval_only.dense_article_ids (frozen reference).
5. Prints delta: gold article rank in HyDE-top-12 vs vanilla-top-12;
   articles added by HyDE; articles removed.
6. Re-runs HyDE end-to-end so the user can confirm the second invocation
   hits the disk cache (no model forward pass).

Hand off to user with the printed report. User checklist (plan §Phase 3):
  - [ ] Hypothetical style matches legal text
  - [ ] No 'Điều X', 'Khoản Y', numbered citations
  - [ ] Content relevant to the question's legal topic
  - [ ] Gold article (L58_2014.A2 for stt=2) still in top-12 after HyDE
  - [ ] No suspiciously off-topic articles climbing into top-3

If any fail: revise prompts/runtime/hyde_generate.md, re-run, loop.

Designed for Colab — uses the same retrieval pipeline as exp08_run.py so
HyDE wiring is exercised end-to-end (not a synthetic test).
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
EXP06_FULL_REF = (
    _REPO
    / "experiments"
    / "06_retrieval_dense_vs_full"
    / "results"
    / "full_rerank"
    / "A2.json"
)

# Match exp 06 dense settings so vanilla-vs-HyDE is apples-to-apples on the
# same dense_k budget. The reference A2.json was produced with dense_k=30.
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

# We compare top-12 article-deduped lists.
COMPARE_K = 12


def _load_question(stt: int) -> dict:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    for q in questions:
        if int(q["stt"]) == stt:
            return q
    raise KeyError(f"stt={stt} not found in {QUESTIONS_PATH}")


def _load_reference_dense_top12(stt: int) -> list[str]:
    """Read exp06 dense_article_ids from full_rerank/A<stt>.json.

    Plan §Phase 3 says to compare against
    experiments/06_retrieval_dense_vs_full/results/full_rerank/A2.json ::
    dense_article_ids — that field is the article-deduped dense pool from
    the same dense_k=30, clause_vec_tuned encoder, so it is a fair
    reference baseline for the dry-run.
    """
    if not EXP06_FULL_REF.exists():
        return []
    rec = json.loads(EXP06_FULL_REF.read_text(encoding="utf-8"))
    ids = (rec.get("retrieval_only") or {}).get("dense_article_ids") or []
    return list(ids)[:COMPARE_K]


def _vram_line(label: str, info: dict) -> str:
    if not info:
        return f"  {label}: (CUDA not available)"
    return (
        f"  {label}: alloc={info['allocated_mb']:.1f} MB "
        f"reserved={info['reserved_mb']:.1f} MB "
        f"max_alloc={info['max_allocated_mb']:.1f} MB"
    )


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
    p.add_argument("--n", type=int, default=1,
                   help="N hypothetical docs to generate (plan §D2 default 1).")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--dtype", type=str, default="fp16",
                   choices=["fp16", "bf16", "4bit"])
    args = p.parse_args()

    print("=" * 78)
    print(f"Experiment 08 — Phase 3 dry-run (stt={args.stt}, gold={args.gold})")
    print("=" * 78)

    q = _load_question(args.stt)
    print()
    print("[1/6] Question")
    print("-" * 78)
    print(q["question"])
    print(f"  gold_citations_raw: {q.get('gold_citations_raw')}")
    print()

    # Construct HyDE generator first so VRAM probes capture pre-load state.
    from src.retrieval.hyde import QwenHydeGenerator

    print("[2/6] HyDE generator")
    print("-" * 78)
    hyde = QwenHydeGenerator(
        n=args.n,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
    )
    print(f"  model_id    : {hyde.model_id}")
    print(f"  prompt_path : {hyde.prompt_source_path}")
    print(f"  prompt_sha  : {hyde.prompt_sha}")
    print(f"  cache_dir   : {hyde.cache_dir}")
    print(_vram_line("VRAM pre-load", hyde.cuda_memory_mb()))

    # Construct pipeline with hyde wired in — this is the production-shape
    # entry point. retrieve_dense_only ignores hyde, retrieve_dense_only_hyde
    # uses it. We construct ONCE so embed_model + hyde model share GPU.
    print()
    print("[3/6] V5RetrievalPipeline (warm-up)")
    print("-" * 78)
    from src.retrieval.pipeline import V5RetrievalPipeline

    pipe = V5RetrievalPipeline(hyde=hyde, **SHARED_CFG)
    _ = pipe.embed_model  # force BGE-M3 load
    print(_vram_line("VRAM after BGE-M3 load (Qwen not loaded yet)",
                     hyde.cuda_memory_mb()))

    # --- First HyDE run: generates + caches ---
    print()
    print("[4/6] HyDE first generation (model forward pass expected)")
    print("-" * 78)
    t0 = time.time()
    docs = hyde.generate(q["question"])
    t_first = time.time() - t0
    print(f"  generate() wall time : {t_first:.2f}s")
    print(_vram_line("VRAM after Qwen load + generate", hyde.cuda_memory_mb()))
    print()
    for i, doc in enumerate(docs, 1):
        print(f"--- Hypothetical doc #{i} (len={len(doc)} chars) ---")
        print(doc)
        print()

    try:
        # --- Vanilla dense search (raw-question embedding) ---
        print("[5/6] Dense search — vanilla vs HyDE (top-12 article-deduped)")
        print("-" * 78)
        t = time.time()
        vanilla = pipe.retrieve_dense_only(q["question"], top_k=SHARED_CFG["dense_k"])
        t_vanilla = time.time() - t

        # --- HyDE dense search (cache hit on docs above) ---
        t = time.time()
        hyde_dense = pipe.retrieve_dense_only_hyde(
            q["question"], top_k=SHARED_CFG["dense_k"]
        )
        t_hyde = time.time() - t

        vanilla_top = vanilla.final_article_ids[:COMPARE_K]
        hyde_top = hyde_dense.final_article_ids[:COMPARE_K]
        ref_top = _load_reference_dense_top12(args.stt)

        # Side-by-side table
        max_rows = max(len(vanilla_top), len(hyde_top), len(ref_top), COMPARE_K)
        header = f"  {'rank':>4}  {'vanilla(now)':<22}  {'HyDE(now)':<22}  {'exp06 ref':<22}"
        print(header)
        print(f"  {'-' * 4}  {'-' * 22}  {'-' * 22}  {'-' * 22}")
        for i in range(max_rows):
            v = vanilla_top[i] if i < len(vanilla_top) else ""
            h = hyde_top[i] if i < len(hyde_top) else ""
            r = ref_top[i] if i < len(ref_top) else ""
            tag_v = " *" if v == args.gold else "  "
            tag_h = " *" if h == args.gold else "  "
            tag_r = " *" if r == args.gold else "  "
            print(f"  {i + 1:>4}  {v:<20}{tag_v}  {h:<20}{tag_h}  {r:<20}{tag_r}")
        print(f"  latency vanilla={t_vanilla:.2f}s   HyDE={t_hyde:.2f}s   "
              f"(HyDE includes embedding-of-generated-doc; gen was cache-hit)")

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
        print(f"  Added by HyDE (in HyDE-top12, not in vanilla-top12): {added}")
        print(f"  Removed by HyDE (in vanilla-top12, not in HyDE-top12): {removed}")
        print(f"  Jaccard(vanilla, HyDE) = "
              f"{(len(v_set & h_set) / max(1, len(v_set | h_set))):.3f}")

        # --- Cache-hit confirmation ---
        print()
        print("[6/6] Cache-hit confirmation (re-run HyDE generation)")
        print("-" * 78)
        t = time.time()
        docs2 = hyde.generate(q["question"])
        t_second = time.time() - t
        print(f"  generate() second wall time : {t_second:.4f}s")
        same = docs == docs2
        # A real cache hit returns in O(ms) — no model forward. We expect
        # at least a 50× speedup vs the cold call.
        speedup = (t_first / t_second) if t_second > 0 else float("inf")
        print(f"  identical to first call     : {same}")
        print(f"  speedup vs first call       : {speedup:.1f}×")
        if t_second > 1.0:
            print("  WARNING: second call took >1s — cache may have missed. Investigate.")
        else:
            print("  OK — cache hit confirmed.")

    finally:
        pipe.close()

    print()
    print("=" * 78)
    print("User GATE checklist (plan §Phase 3):")
    print("  [ ] Hypothetical style matches formal legal text")
    print("  [ ] NO 'Điều X', 'Khoản Y', numbered citations in the doc")
    print("  [ ] NO personal names / dates / specific numbers from the question")
    print("  [ ] Topic is relevant to the question's legal area")
    print(f"  [ ] Gold {args.gold} still in HyDE top-{COMPARE_K}")
    print("  [ ] No suspiciously off-topic articles climbing into top-3")
    print()
    print("If any fail → revise prompts/runtime/hyde_generate.md and re-run.")
    print("If all pass → proceed to Phase 4 (scaffold) + Phase 5 (pilot 5).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
