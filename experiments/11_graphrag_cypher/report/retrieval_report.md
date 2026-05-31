# Experiment 11 (REDO) — `CypherWalkRetriever` retrieval-only audit

Dataset: 50 BHXH questions. Metric granularity: **article**.
Arms: `dense_vanilla`, `dense_then_expand`, `cypher_walk`.
Retrieval-only: no answer generation, no BERTScore, no citation parsing.
Plan: [`docs/plans/exp11_cypher_walk_retriever.md`](../../../docs/plans/exp11_cypher_walk_retriever.md).

## Honest limitations (read first)

- **Clause-level recall is NOT reported.** The gold dataset carries `0` clause-level (khoản) citations — gold is article-level only (`gold_items == gold_articles` for all questions, granularity=`tuple`). A clause-level number would require fabricating clause gold, so only article-level metrics appear below.
- All three arms reuse the **vanilla** `clause_vec` dense channel (not the v5 tuned index/reranker), so numbers are NOT comparable to exp 06/07/08 absolute values — only to each other within exp 11.

## Overall macro (n=50)

### Citation recall@K (article-level)
| arm | n | @5 | @12 | @20 | @all |
|---|---:|---:|---:|---:|---:|
| dense_vanilla | 50 | 0.1483 | 0.2067 | 0.2067 | 0.2067 |
| dense_then_expand | 50 | 0.1483 | 0.2583 | 0.2750 | 0.2750 |
| cypher_walk | 50 | 0.1150 | 0.1650 | 0.1650 | 0.1650 |

### Citation precision@K
| arm | n | @5 | @12 | @20 | @all |
|---|---:|---:|---:|---:|---:|
| dense_vanilla | 50 | 0.0520 | 0.0433 | 0.0433 | 0.0433 |
| dense_then_expand | 50 | 0.0520 | 0.0414 | 0.0422 | 0.0422 |
| cypher_walk | 50 | 0.0400 | 0.0395 | 0.0395 | 0.0395 |

### Citation F1@K
| arm | n | @5 | @12 | @20 | @all |
|---|---:|---:|---:|---:|---:|
| dense_vanilla | 50 | 0.0726 | 0.0685 | 0.0685 | 0.0685 |
| dense_then_expand | 50 | 0.0726 | 0.0684 | 0.0705 | 0.0705 |
| cypher_walk | 50 | 0.0559 | 0.0603 | 0.0603 | 0.0603 |

### NDCG@K (binary relevance)
| arm | n | @5 | @12 | @20 | @all |
|---|---:|---:|---:|---:|---:|
| dense_vanilla | 50 | 0.1023 | 0.1247 | 0.1247 | 0.1247 |
| dense_then_expand | 50 | 0.1023 | 0.1406 | 0.1461 | 0.1461 |
| cypher_walk | 50 | 0.0835 | 0.1038 | 0.1038 | 0.1038 |

### Rank-aware (K-independent) + latency

| arm | n | R-Precision | MRR | avg elapsed (s) |
|---|---:|---:|---:|---:|
| dense_vanilla | 50 | 0.0483 | 0.1226 | 0.2980 |
| dense_then_expand | 50 | 0.0483 | 0.1294 | 0.2070 |
| cypher_walk | 50 | 0.0450 | 0.1025 | 5.8350 |

## CypherWalk provenance — is the graph actually walked? (plan §5.3)

| quantity | value |
|---|---:|
| n (cypher_walk records) | 50 |
| **cypher_used rate** (≥1 NEW clause beyond seed) | **0.7600** (38/50) |
| mean n_cypher_new (all) | 11.1800 |
| **mean n_cypher_new (when cypher_used)** | **14.7105** |
| fallback_used rate | 0.2400 |
| mean n_fallback_added | 0.9200 |
| mean cypher_attempts | 1.8600 |

## Pre-commitment check (plan §5.5) — stated before the run

| prediction | threshold | observed | within prediction? |
|---|---|---:|:-:|
| cypher_used rate | ≥ 0.30 | 0.7600 | ✓ |
| mean n_cypher_new (when used) | 1.5–3.0 | 14.7105 | ✗ AUDIT |
| recall@12 lift vs dense_vanilla (all strata) | +0.00 to +0.05 | -0.0417 | ✗ AUDIT |
| recall@12 lift on no_l41 stratum | near 0 (≤ +0.05) | 0.0000 | ✓ |

> If recall@12 lift > +0.05 across all strata, the plan says treat as **suspicious and audit before celebrating** — do not rationalise.

## Stratified by L41 presence in gold

### l41_only

_recall@K_

| arm | n | @5 | @12 | @20 | @all |
|---|---:|---:|---:|---:|---:|
| dense_vanilla | 28 | 0.1458 | 0.2321 | 0.2321 | 0.2321 |
| dense_then_expand | 28 | 0.1458 | 0.3244 | 0.3423 | 0.3423 |
| cypher_walk | 28 | 0.1101 | 0.1458 | 0.1458 | 0.1458 |

### mixed_l41_other

_recall@K_

| arm | n | @5 | @12 | @20 | @all |
|---|---:|---:|---:|---:|---:|
| dense_vanilla | 9 | 0.2037 | 0.2593 | 0.2593 | 0.2593 |
| dense_then_expand | 9 | 0.2037 | 0.2593 | 0.2963 | 0.2963 |
| cypher_walk | 9 | 0.1296 | 0.2963 | 0.2963 | 0.2963 |

### no_l41

_recall@K_

| arm | n | @5 | @12 | @20 | @all |
|---|---:|---:|---:|---:|---:|
| dense_vanilla | 13 | 0.1154 | 0.1154 | 0.1154 | 0.1154 |
| dense_then_expand | 13 | 0.1154 | 0.1154 | 0.1154 | 0.1154 |
| cypher_walk | 13 | 0.1154 | 0.1154 | 0.1154 | 0.1154 |

## Notes

- Recall denominator = |gold articles|; questions with empty gold skipped.
- Precision denominator = |retrieved@K|; empty retrieved set → precision = 0.
- `dense_then_expand` final set = vector-hit articles ∪ REFERENCES/CITES_EXTERNAL ref-target articles.
- For the per-stage cypher_walk funnel, run `python -m scripts.exp11_funnel`.
