# Decision 001 — Retrieval default: `full_rerank` arm at K=12

- **Status**: Accepted
- **Date**: 2026-05-30
- **Owner**: Nguyễn Hữu Hoàng
- **Affects**: `src/retrieval/pipeline.py::V5RetrievalPipeline` defaults,
  every downstream GraphRAG E2E experiment, retrieval-only audit baselines.
- **Supersedes**: nothing (first explicit decision record for retrieval K).

## Decision

The production retrieval pipeline for GraphRAG and Logic-LM GraphRAG arms
uses **the full `V5RetrievalPipeline` (`full_rerank` arm)** — i.e. BGE-M3
LoRA dense + Lucene BM25 sparse + temporal filter + RRF + cross-encoder
rerank1 (top-15 seeds) + REFERS_TO graph hop + rerank2 — with the final
top-K context window set to **`rerank2_top_k = 12` articles**.

This is the *current* default in
[`src/retrieval/pipeline.py`](../../src/retrieval/pipeline.py). This
document ratifies it with empirical evidence so future PRs don't drift
the default without re-running the audit.

## Context — why decide now

After exp 06 (K=12 retrieval A/B) and exp 07 (K extended to 100), there
was an open question: should the default K be raised (to capture more of
the recall ceiling), or stay at 12? The marginal-gain analysis below
settles it.

There was also a parallel question: is the `dense_only` arm a viable
alternative (much faster, simpler)? The evidence below answers no for
production, while marking it as the preferred *retrieval-only audit*
baseline.

## Alternatives considered

| Option | Recall ceiling | F1@retrieval | Latency | LLM-fit |
|---|---|---|---|---|
| **A. `full_rerank` K=12 (chosen)** | 0.357 overall / 0.447 in_corpus | **0.102** | 1.20s | fits 7k-char context |
| B. `full_rerank` K=30 | 0.462 / 0.583 | 0.047 | 2.87s | exceeds context — truncated |
| C. `full_rerank` K=50 | 0.504 / 0.634 | 0.042 | ~3.5s | severe truncation |
| D. `dense_only` K=12 (50-pool) | 0.317 / 0.383 | 0.069 | **0.16s** | fits but lower NDCG |
| E. `dense_only` K=50 | 0.501 / 0.618 | 0.033 | 0.16s | exceeds context |

Macro numbers from
[experiments/06_retrieval_dense_vs_full](../../experiments/06_retrieval_dense_vs_full/report/academic_report.md)
and
[experiments/07_retrieval_extended_k](../../experiments/07_retrieval_extended_k/report/academic_report.md),
n=200 BHXH questions, gold parsed strict via
[`src/citations.parse_gold_citations_raw`](../../src/citations.py).

## Evidence

### 1. Marginal recall per added K plateaus after K=30 (in_corpus, full_rerank)

| K range | ΔR | ΔR/ΔK |
|---|---:|---:|
| 12→20 | +0.082 | **0.0102** (elbow) |
| 20→30 | +0.059 | **0.0059** (elbow) |
| 30→50 | +0.051 | 0.0026 (plateau) |
| 50→70 | +0.004 | 0.0002 (zero) |
| 70→100 | 0.000 | 0 (pool exhausted) |

Source:
[`experiments/07_retrieval_extended_k/metrics/academic_metrics.json`](../../experiments/07_retrieval_extended_k/metrics/academic_metrics.json).

### 2. K=12 wins all rank-aware metrics at the natural context budget

In-corpus stratum (n=151), exp 06:

| metric | dense K=12 | full_rerank K=12 | rel |
|---|---:|---:|---:|
| R@12 | 0.383 | **0.447** | +17% |
| P@12 | 0.049 | **0.077** | +59% |
| F1@12 | 0.083 | **0.126** | +52% |
| NDCG@10 | 0.210 | **0.278** | +33% |
| R-Precision | 0.064 | **0.129** | **+103%** |
| MRR | 0.210 | **0.265** | +26% |

Source:
[`experiments/06_retrieval_dense_vs_full/metrics/academic_metrics.json`](../../experiments/06_retrieval_dense_vs_full/metrics/academic_metrics.json).

### 3. NDCG `full_rerank` dominates dense at every K

| K | NDCG@K dense (in_corpus) | NDCG@K full_rerank | Δ |
|---:|---:|---:|---:|
| 12 | 0.219 | 0.264 | +21% |
| 30 | 0.257 | 0.300 | +17% |
| 50 | 0.276 | 0.312 | +13% |
| 100 | 0.284 | 0.313 | +10% |

Reranker pushes gold higher in the ranking regardless of pool size.
Even when dense overtakes on raw recall (K≥70), full_rerank keeps the
ranking-quality lead.

### 4. Context budget gates K in production

[`src/retrieval/pipeline.py`](../../src/retrieval/pipeline.py) has
`MAX_CONTEXT_CHARS = 7000`. With ~500 chars per article header+body,
~12 articles fills the budget. Raising K above 12 means the LLM still
only sees the top 12-15 due to truncation — the extra retrieval cost
buys nothing for the generator.

Exp 05 finding (Sprint 2 audit,
[`experiments/05_v5_retrieval_audit/README.md`](../../experiments/05_v5_retrieval_audit/README.md)):
LLM citation-loss rate increased when rerank2 candidate scores
clustered tightly (0.93-0.97 range with M2 + bigger pool), the
generator "got confused" picking the right top. K=12 keeps the cluster
tight in a way the generator handles best.

### 5. Latency budget

- `full_rerank` K=12: 1.20s/question (exp 06)
- `full_rerank` K=100: 2.87s/question (exp 07) — 2.4× slower for no
  E2E win because of context-budget truncation.
- `dense_only` K=12: 0.16s/question.

`full_rerank` K=12 is 7.5× slower than dense at K=12 but buys real
ranking quality (R-Precision +103% in_corpus). At K=30+ the rerank
overhead doesn't pay back at the generator.

## Consequences

### Positive
- Maximises F1, NDCG, R-Precision, MRR within the context budget.
- Matches the LLM-context capacity (7k chars), so retrieval and
  generator are co-tuned.
- Keeps the default identical to what `experiments/04_v5_sprint2_m2/`
  shipped — no migration cost for existing baselines or inheritance
  chains.

### Negative / accepted tradeoffs
- **Retrieval ceiling capped at R≈0.45 in_corpus**. To raise the
  ceiling, the bottlenecks are *not* K — they are:
  - Out-of-corpus questions (8/200 = 4%) — needs corpus expansion
    (add Bộ luật Lao động, Nghị định 152/2006, etc.) per plan v5 §11,
    blocked on user decision.
  - Unparseable gold (36/200 = 18%) — registry alias gaps
    in [`data/legal_sources.yaml`](../../data/legal_sources.yaml).
    Quick win, no re-embed needed.
- Dense arm is faster but is the wrong primary because of NDCG/R-Prec
  gap. Reserved as an audit baseline + ablation only.

### Operational
- The K value lives in **one place**: the `rerank2_top_k=12` default in
  `V5RetrievalPipeline.__init__`. Per-experiment overrides go through
  the constructor argument, not by editing the default.
- Audit experiments (retrieval-only) may set higher K transparently for
  ceiling measurement; they must not propagate that K into production
  arms via inheritance.

## When to revisit

This decision becomes stale and should be re-audited if **any** of the
following changes:

1. **Corpus grows**: new laws ingested into Neo4j. New OOC distribution
   may change |gold| stats and shift the |gold|/K asymmetry.
2. **`MAX_CONTEXT_CHARS` is raised** (e.g. switching generator to
   GPT-4o-128k or Claude-Sonnet with bigger context). The K=12 cap is
   directly tied to 7000 chars — a wider window invites K=20-30.
3. **Reranker model changes** (currently `BAAI/bge-reranker-base`).
   A stronger reranker may keep ranking quality at higher K, weakening
   the case for K=12.
4. **Registry / corpus fixes land** that reduce the unparseable bucket
   (36/200) significantly. With more parseable gold the absolute ceiling
   rises and the elbow may shift.
5. **A new arm** outperforms `full_rerank` on R-Precision or NDCG at
   K=12 — e.g. an HyDE / decomposed retrieval variant on the plan v5
   roadmap.

The re-audit recipe: re-run
[`scripts/exp06_run.py`](../../scripts/exp06_run.py) and
[`scripts/exp07_run.py`](../../scripts/exp07_run.py) into fresh
experiment folders, regenerate the marginal-gain table, update this
decision record (new dated section) or supersede it with Decision 002.

## References

- [Plan v5 — general retrieval](../plans/v5_general_retrieval.md)
- [Experiment 04 — Sprint 2 M2 (current production baseline)](../../experiments/04_v5_sprint2_m2/README.md)
- [Experiment 05 — retrieval audit (30-probe)](../../experiments/05_v5_retrieval_audit/README.md)
- [Experiment 06 — dense vs full_rerank at K=12](../../experiments/06_retrieval_dense_vs_full/README.md)
- [Experiment 07 — extended K to 100](../../experiments/07_retrieval_extended_k/README.md)
- [`src/retrieval/pipeline.py`](../../src/retrieval/pipeline.py) — implementation default lives here.
