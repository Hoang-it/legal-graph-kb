"""Experiment 11 runner — CypherWalkRetriever retrieval-only 3-arm audit.

REDO of the mis-built ``graphrag_cypher`` E2E arm (commit a10f609). This
script is retrieval-only: NO answer generation, NO BERTScore, NO citation
parsing. It mirrors the exp 06/07/08 convention (custom run + metrics +
funnel scripts, not ``python -m eval_core run``).

Arms (plan §5.2), all built on the *vanilla* ``RagPipeline`` dense channel
(``clause_vec``) so the comparison stays apples-to-apples with vanilla
graphrag and pulls in no v5 changes:

- ``dense_vanilla``      → ``RagPipeline.vector_search(top_k=12)``. Baseline.
- ``dense_then_expand``  → ``vector_search(top_k=12)`` + ``RagPipeline.expand``
                           REFERENCES/CITES_EXTERNAL refs folded in at the
                           article level (vanilla graphrag's retrieval side,
                           minus the LLM).
- ``cypher_walk``        → ``CypherWalkRetriever.retrieve`` (vector seed →
                           LLM outward Cypher walk → fallback expand → RRF).

Only ``cypher_walk`` calls OpenAI; the other two are free + offline-LLM.

``--pilot-50`` selects a 50-question subset stratified by L41-presence
(l41_only / mixed_l41_other / no_l41 — plan §5.3), persisted at
``experiments/11_graphrag_cypher/pilot_50_stt.json`` so re-runs and the
metrics/funnel scripts converge on the same subset.

Records land at ``experiments/11_graphrag_cypher/results/<arm>/A<stt>.json``.
Idempotent + ``--force`` + ``--stt`` subset (same shape as exp08_run.py).
"""
from __future__ import annotations

import argparse
import json
import os
import random
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

from src.ids import parse_id  # noqa: E402

EXP_DIR = _REPO / "experiments" / "11_graphrag_cypher"
QUESTIONS_PATH = _REPO / "data" / "eval" / "questions_200.json"
REGISTRY_PATH = _REPO / "data" / "legal_sources.yaml"
PILOT_50_PATH = EXP_DIR / "pilot_50_stt.json"

ARMS = ("dense_vanilla", "dense_then_expand", "cypher_walk")
OPENAI_ARMS = {"cypher_walk"}

TOP_K = 12
SEED_K = 8
MAX_REPAIR_ROUNDS = 2


# ---------------------------------------------------------------------------
# Article-level helpers
# ---------------------------------------------------------------------------


def _article_of(node_id: str) -> str | None:
    """Article id for a structural Clause/Article id, else None."""
    try:
        p = parse_id(node_id)
    except ValueError:
        return None
    if p.get("article") is None:
        return None
    return f"{p['law']}.A{p['article']}"


def _dedupe(seq) -> list[str]:
    return list(dict.fromkeys(x for x in seq if x))


# ---------------------------------------------------------------------------
# Stratified pilot — keyed on L41 presence in gold (plan §5.3)
# ---------------------------------------------------------------------------


def _l41_stratum(gold_articles: list[str]) -> str:
    if not gold_articles:
        return "empty_gold"
    laws = {a.split(".")[0] for a in gold_articles}
    if laws == {"L41_2024"}:
        return "l41_only"
    if "L41_2024" in laws:
        return "mixed_l41_other"
    return "no_l41"


def _load_gold_articles() -> dict[int, list[str]]:
    from eval_core.gold import validate_gold_citations

    ok, summary = validate_gold_citations(
        questions_path=QUESTIONS_PATH,
        registry_path=REGISTRY_PATH,
        out_dir=EXP_DIR / "metrics",
    )
    if not ok:
        print(f"FAIL: gold validation; see {summary['errors_path']}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(Path(summary["normalized_path"]).read_text(encoding="utf-8"))
    return {int(k): v.get("gold_articles") or [] for k, v in data["records"].items()}


def build_or_load_pilot_50(seed: int = 0) -> dict:
    if PILOT_50_PATH.exists():
        return json.loads(PILOT_50_PATH.read_text(encoding="utf-8"))

    gold = _load_gold_articles()
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    by_cat: dict[str, list[int]] = {
        "l41_only": [], "mixed_l41_other": [], "no_l41": [], "empty_gold": [],
    }
    for q in questions:
        by_cat[_l41_stratum(gold.get(q["stt"], []))].append(q["stt"])

    n_total = 50
    n_dataset = len(questions)
    quotas: dict[str, int] = {}
    for c, ids in by_cat.items():
        quotas[c] = (max(1, n_total * len(ids) // n_dataset) if ids else 0)
    # Reconcile to exactly n_total by adjusting the dominant stratum.
    dominant = max(by_cat, key=lambda c: len(by_cat[c]))
    while sum(quotas.values()) < n_total:
        quotas[dominant] += 1
    while sum(quotas.values()) > n_total:
        quotas[dominant] -= 1

    rng = random.Random(seed)
    chosen: list[int] = []
    for cat, k in quotas.items():
        if k <= 0:
            continue
        pool = sorted(by_cat[cat])
        chosen.extend(rng.sample(pool, min(k, len(pool))))
    chosen.sort()
    payload = {
        "seed": seed, "n": len(chosen), "n_dataset": n_dataset,
        "stratify_by": "l41_presence", "quotas": quotas, "stt_list": chosen,
    }
    PILOT_50_PATH.parent.mkdir(parents=True, exist_ok=True)
    PILOT_50_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


# ---------------------------------------------------------------------------
# CLI helpers (same shape as exp08_run.py)
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


# ---------------------------------------------------------------------------
# Per-arm retrieval
# ---------------------------------------------------------------------------


def run_dense_vanilla(rag, question: str) -> dict:
    t0 = time.time()
    hits = rag.vector_search(question, top_k=TOP_K)
    final_clause_ids = [h.clause_id for h in hits]
    final_article_ids = _dedupe(h.article_id for h in hits)
    return {
        "final_article_ids": final_article_ids,
        "final_clause_ids": final_clause_ids,
        "elapsed_s": round(time.time() - t0, 3),
        "provenance": {
            "n_vector": len(hits),
            "vector_clause_ids": final_clause_ids,
        },
    }


def run_dense_then_expand(rag, question: str) -> dict:
    """Vanilla graphrag's retrieval side: vector hits + expand refs folded in.

    ``RagPipeline.expand`` returns semantic edges (entity→entity, no
    clause/article id) + REFERENCES/CITES_EXTERNAL refs. At the article
    level the refs add the target Articles/Clauses' articles; semantic
    edges add nothing structural. We fold the ref-target articles in after
    the vector-hit articles (rank-preserving, deduped) — exactly what the
    vanilla ``build_context`` surfaces, without the LLM.
    """
    t0 = time.time()
    hits = rag.vector_search(question, top_k=TOP_K)
    vec_clause_ids = [h.clause_id for h in hits]
    vec_article_ids = _dedupe(h.article_id for h in hits)

    expansion = rag.expand(vec_clause_ids)
    ref_article_ids: list[str] = []
    for ref in expansion.get("refs", []):
        art = _article_of(ref.get("dst", ""))
        if art:
            ref_article_ids.append(art)
    ref_article_ids = [a for a in _dedupe(ref_article_ids) if a not in set(vec_article_ids)]

    final_article_ids = _dedupe(vec_article_ids + ref_article_ids)
    return {
        "final_article_ids": final_article_ids,
        # clause-level identity is only pinned for the vector hits; refs are
        # article-level neighbours — recorded separately for honesty.
        "final_clause_ids": vec_clause_ids,
        "elapsed_s": round(time.time() - t0, 3),
        "provenance": {
            "n_vector": len(hits),
            "n_semantic_edges": len(expansion.get("edges", [])),
            "n_refs": len(expansion.get("refs", [])),
            "n_ref_articles_added": len(ref_article_ids),
            "ref_articles_added": ref_article_ids,
        },
    }


def _articles_of(clause_ids) -> list[str]:
    return _dedupe(a for a in (_article_of(c) for c in clause_ids) if a)


def run_cypher_walk(retriever, question: str) -> dict:
    res = retriever.retrieve(question)
    final_clause_ids = [h.clause_id for h in res.hits]
    final_article_ids = _dedupe(h.article_id for h in res.hits)
    return {
        "final_article_ids": final_article_ids,
        "final_clause_ids": final_clause_ids,
        "elapsed_s": res.elapsed_s,
        # Per-stage article projections — consumed by exp11_funnel.py.
        "seed_article_ids": _articles_of(res.seed_clause_ids),
        "cypher_new_article_ids": _articles_of(res.cypher_new_clause_ids),
        "fallback_article_ids": _articles_of(res.fallback_clause_ids),
        "provenance": {
            "n_seed": res.n_seed,
            "n_cypher_new": res.n_cypher_new,
            "n_fallback_added": res.n_fallback_added,
            "cypher_used": res.cypher_used,
            "fallback_used": res.fallback_used,
            "n_cypher_attempts": len(res.cypher_attempts),
            "seed_clause_ids": res.seed_clause_ids,
            "cypher_new_clause_ids": res.cypher_new_clause_ids,
            "fallback_clause_ids": res.fallback_clause_ids,
            "cypher_attempts": [asdict(a) for a in res.cypher_attempts],
            "hit_sources": [h.source for h in res.hits],
            "elapsed_breakdown": res.elapsed_breakdown,
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stt", type=str, default="",
                   help="Comma/range stt list (e.g. '1-5,10'); empty = full 200 "
                        "or pilot-50 if --pilot-50.")
    p.add_argument("--pilot-50", action="store_true",
                   help="Use the L41-stratified 50-question pilot subset.")
    p.add_argument("--arms", type=str, default=",".join(ARMS),
                   help=f"Subset of {ARMS}; default = all three.")
    p.add_argument("--force", action="store_true", help="Overwrite existing records.")
    p.add_argument("--cypher-model", type=str, default=None,
                   help="OpenAI model id for the Cypher generator (default = $OPENAI_MODEL).")
    p.add_argument("--cost-cap", type=float, default=0.50,
                   help="Abort pre-flight if Cypher-LLM cost estimate exceeds this USD.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    arms_subset = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms_subset:
        if a not in ARMS:
            print(f"ERROR: unknown arm {a!r}; valid: {ARMS}", file=sys.stderr)
            return 2

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

    need_openai = any(a in OPENAI_ARMS for a in arms_subset)
    print(f"Probe size      : {len(questions)} questions")
    print(f"Arms            : {arms_subset}")
    print(f"Config          : top_k={TOP_K} seed_k={SEED_K} max_repair_rounds={MAX_REPAIR_ROUNDS}")

    if need_openai:
        # Worst case = (max_repair_rounds + 1) Cypher LLM calls per question.
        est_calls = len(questions) * (MAX_REPAIR_ROUNDS + 1)
        est_cost = 0.0005 * est_calls
        print(f"Cypher LLM      : model={args.cypher_model or os.getenv('OPENAI_MODEL','gpt-4o-mini')} "
              f"(<= {est_calls} calls worst-case)")
        print(f"Estimated cost  : ~${est_cost:.4f} (worst case, all repairs exhausted)")
        if est_cost > args.cost_cap:
            print(f"ABORT: estimated cost ${est_cost:.4f} exceeds --cost-cap "
                  f"${args.cost_cap:.4f}", file=sys.stderr)
            return 3

    from runtime.rag_query import RagPipeline

    print("Connecting Neo4j + warming BGE-M3 ...", flush=True)
    rag = RagPipeline()
    _ = rag.embed_model  # warm once; shared across arms

    retriever = None
    if "cypher_walk" in arms_subset:
        from runtime.retrievers.cypher_walk import CypherWalkRetriever

        retriever = CypherWalkRetriever(
            rag, top_k_seed=SEED_K, top_k_final=TOP_K,
            max_repair_rounds=MAX_REPAIR_ROUNDS, cypher_model=args.cypher_model,
        )

    out_root = EXP_DIR / "results"
    counts = {a: {"done": 0, "skipped": 0, "failed": 0, "latencies": []} for a in arms_subset}
    cypher_used_n = fallback_used_n = 0
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
                        cfg = {"mode": "dense_vanilla", "top_k": TOP_K, "dense_index": "clause_vec"}
                    elif arm == "dense_then_expand":
                        ro = run_dense_then_expand(rag, q["question"])
                        cfg = {"mode": "dense_then_expand", "top_k": TOP_K,
                               "dense_index": "clause_vec", "expand": "REFERENCES|CITES_EXTERNAL"}
                    elif arm == "cypher_walk":
                        assert retriever is not None
                        ro = run_cypher_walk(retriever, q["question"])
                        cfg = {"mode": "cypher_walk", "top_k_final": TOP_K, "top_k_seed": SEED_K,
                               "max_repair_rounds": MAX_REPAIR_ROUNDS,
                               "cypher_model": retriever.cypher_model, "rrf_k": retriever.rrf_k}
                        if ro["provenance"]["cypher_used"]:
                            cypher_used_n += 1
                        if ro["provenance"]["fallback_used"]:
                            fallback_used_n += 1
                    else:
                        raise RuntimeError(f"unreachable arm={arm!r}")

                    counts[arm]["latencies"].append(ro["elapsed_s"])
                    record = {
                        "arm": arm, "stt": stt, "question": q["question"],
                        "gold_citations_raw": q.get("gold_citations_raw"),
                        "config_used": cfg, "retrieval_only": ro,
                    }
                    _write_record(out_path, record)
                    counts[arm]["done"] += 1
                    if args.verbose or i % 10 == 0:
                        prov = ro["provenance"]
                        extra = (f" cyp_new={prov['n_cypher_new']} cyp_used={prov['cypher_used']}"
                                 if arm == "cypher_walk" else
                                 f" final={len(ro['final_article_ids'])}")
                        print(f"  [{arm:<18} {i:>3}/{len(questions)}] stt={stt} "
                              f"({ro['elapsed_s']:.1f}s{extra})", flush=True)
                except Exception as e:  # noqa: BLE001
                    counts[arm]["failed"] += 1
                    print(f"  X [{arm} stt={stt}] {type(e).__name__}: {e}",
                          file=sys.stderr, flush=True)
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
        print(f"  {arm:<18} done={c['done']} skipped={c['skipped']} "
              f"failed={c['failed']} avg_latency={avg}s")
    if "cypher_walk" in arms_subset:
        n_run = counts["cypher_walk"]["done"]
        if n_run:
            print(f"  cypher_walk provenance: cypher_used={cypher_used_n}/{n_run} "
                  f"({cypher_used_n / n_run:.0%}) fallback_used={fallback_used_n}/{n_run} "
                  f"({fallback_used_n / n_run:.0%})")
    print(f"Total wall time: {time.time() - t_total:.1f}s")
    return 0 if all(c["failed"] == 0 for c in counts.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
