# Experiment 13 — Semantic-grounded HyDE (concept frame), retrieval-only

Dataset: 50 BHXH questions. Tuned stack (BGE-M3 LoRA + `clause_vec_tuned`). Headline stratum = **in_corpus** (gold ⊆ indexed laws).
Arms: `dense` (raw), `dense_hyde` (HyDE1 = the bar), `dense_hyde_semantic` (concept frame, no dense seed).

## In-corpus (headline)

### recall@K
| arm | n | @1 | @5 | @10 | @12 | @20 | @all |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 38 | 0.0746 | 0.2456 | 0.3904 | 0.4154 | 0.4944 | 0.7224 |
| dense_hyde | 38 | 0.1140 | 0.3716 | 0.4812 | 0.5207 | 0.5294 | 0.7005 |
| dense_hyde_semantic | 38 | 0.1053 | 0.3289 | 0.3985 | 0.4248 | 0.5873 | 0.7014 |

### precision@K  (precision@1 is the headline; @2+ is cardinality-capped, see §0)
| arm | n | @1 | @5 | @10 | @12 | @20 | @all |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 38 | 0.1053 | 0.0579 | 0.0447 | 0.0461 | 0.0329 | 0.0198 |
| dense_hyde | 38 | 0.1316 | 0.1000 | 0.0658 | 0.0592 | 0.0368 | 0.0220 |
| dense_hyde_semantic | 38 | 0.1053 | 0.0842 | 0.0526 | 0.0461 | 0.0408 | 0.0229 |

| arm | n | R-Precision | MRR | NDCG@12 |
|---|---:|---:|---:|---:|
| dense | 38 | 0.0746 | 0.2075 | 0.2318 |
| dense_hyde | 38 | 0.1479 | 0.2781 | 0.3134 |
| dense_hyde_semantic | 38 | 0.1529 | 0.2550 | 0.2673 |

## Pre-registered success criteria (plan §8, in_corpus)

| check | rule | value | verdict |
|---|---|---:|:-:|
| S1 no-regression vs HyDE1 | sem R@12 − HyDE1 R@12 ≥ −0.01 | -0.0959 | FAIL |
| S2 beats raw dense | sem R@12 − dense R@12 ≥ +0.03 | 0.0094 | FAIL |
| headline Δ (win) | sem R@12 − HyDE1 R@12 ≥ +0.02 | -0.0959 | — |

> Decision rule: **win = S1 ∧ S2 ∧ (headline Δ ≥ +0.02)**. Δ > +0.05 → audit before celebrating.

**North-star (NOT this experiment's pass/fail, plan §9):** recall@12 ≈ 0.55–0.60 / R-Prec ≈ 0.30 are above the HyDE1 baseline and need the full rerank pipeline and/or corpus expansion — separate levers.

## Semantic-frame provenance (dense_hyde_semantic)

- concept_match_rate (in_corpus): **0.8421** (fallback 0.1579)
- mean concepts/q: 1.4740; mean KG entities/q: 0.4210

## Overall macro (all strata)

### recall@K
| arm | n | @1 | @5 | @10 | @12 | @20 | @all |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 50 | 0.0567 | 0.1867 | 0.2967 | 0.3157 | 0.3757 | 0.5590 |
| dense_hyde | 50 | 0.0867 | 0.2824 | 0.3757 | 0.4057 | 0.4124 | 0.5424 |
| dense_hyde_semantic | 50 | 0.0800 | 0.2500 | 0.3029 | 0.3229 | 0.4464 | 0.5430 |

## Stratified recall@12

| arm | in_corpus | mixed | ooc | unparseable |
|---|---:|---:|---:|---:|
| dense | 0.4154 | 0.0000 | 0.0000 | 0.0000 |
| dense_hyde | 0.5207 | 0.5000 | 0.0000 | 0.0000 |
| dense_hyde_semantic | 0.4248 | 0.0000 | 0.0000 | 0.0000 |
