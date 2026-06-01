# Experiment 14 — Fair-prompt re-test (grounded + semantic HyDE), retrieval-only

Dataset: 50 BHXH questions. Tuned stack (BGE-M3 LoRA + `clause_vec_tuned`). Headline stratum = **in_corpus**.
Parity prompts (grounded + semantic) share HyDE1's vocab scaffold; only the grounding block differs. HyDE1 = canonical (frozen) prompt = the bar.

## In-corpus (headline)

### recall@K
| arm | n | @1 | @5 | @10 | @12 | @20 | @all |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 38 | 0.0746 | 0.2456 | 0.3904 | 0.4154 | 0.4944 | 0.7224 |
| dense_hyde | 38 | 0.1140 | 0.3716 | 0.4812 | 0.5207 | 0.5294 | 0.7005 |
| dense_hyde2 | 38 | 0.1053 | 0.3189 | 0.4066 | 0.4330 | 0.5031 | 0.6172 |
| dense_hyde_semantic | 38 | 0.0921 | 0.3239 | 0.4731 | 0.4731 | 0.5739 | 0.6830 |

### precision@K  (precision@1 is the headline; @2+ is cardinality-capped)
| arm | n | @1 | @5 | @10 | @12 | @20 | @all |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 38 | 0.1053 | 0.0579 | 0.0447 | 0.0461 | 0.0329 | 0.0198 |
| dense_hyde | 38 | 0.1316 | 0.1000 | 0.0658 | 0.0592 | 0.0368 | 0.0220 |
| dense_hyde2 | 38 | 0.1053 | 0.0842 | 0.0553 | 0.0482 | 0.0355 | 0.0196 |
| dense_hyde_semantic | 38 | 0.1053 | 0.0789 | 0.0632 | 0.0526 | 0.0395 | 0.0228 |

| arm | n | R-Precision | MRR | NDCG@12 |
|---|---:|---:|---:|---:|
| dense | 38 | 0.0746 | 0.2075 | 0.2318 |
| dense_hyde | 38 | 0.1479 | 0.2781 | 0.3134 |
| dense_hyde2 | 38 | 0.1128 | 0.2188 | 0.2552 |
| dense_hyde_semantic | 38 | 0.1046 | 0.2249 | 0.2630 |

## Pre-registered success criteria (in_corpus) — per challenger vs HyDE1 bar

| challenger | R@12 | S1 (≥ HyDE1−0.01) | S2 (≥ dense+0.03) | fair-win (Δ≥+0.02) |
|---|---:|:-:|:-:|:-:|
| HyDE2-fair (grounded) | 0.4330 | FAIL (-0.0877) | FAIL (0.0176) | — |
| semantic-fair (concept frame) | 0.4731 | FAIL (-0.0476) | PASS (0.0577) | — |

> HyDE1 bar R@12 (in_corpus) = **0.5207**; raw dense = 0.4154.
> Read the prompt-parity effect by comparing each challenger's R@12 here against
> its value in the FROZEN exp 13 (semantic) / exp 09 (grounded) report on the
> SAME pilot-50. A narrower gap ⇒ part of the earlier loss was the dropped-vocab
> confound, not the grounding idea. Differences are point estimates — pair a
> bootstrap CI before any firm claim.

## Semantic-frame provenance (dense_hyde_semantic) — frame builder UNCHANGED vs exp 13

- concept_match_rate (in_corpus): **0.8421** (fallback 0.1579)
- mean concepts/q: 1.4740; mean KG entities/q: 0.4210

## Overall macro (all strata)

### recall@K
| arm | n | @1 | @5 | @10 | @12 | @20 | @all |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 50 | 0.0567 | 0.1867 | 0.2967 | 0.3157 | 0.3757 | 0.5590 |
| dense_hyde | 50 | 0.0867 | 0.2824 | 0.3757 | 0.4057 | 0.4124 | 0.5424 |
| dense_hyde2 | 50 | 0.0800 | 0.2424 | 0.3090 | 0.3290 | 0.3824 | 0.4790 |
| dense_hyde_semantic | 50 | 0.0700 | 0.2462 | 0.3695 | 0.3695 | 0.4462 | 0.5290 |

## Stratified recall@12

| arm | in_corpus | mixed | ooc | unparseable |
|---|---:|---:|---:|---:|
| dense | 0.4154 | 0.0000 | 0.0000 | 0.0000 |
| dense_hyde | 0.5207 | 0.5000 | 0.0000 | 0.0000 |
| dense_hyde2 | 0.4330 | 0.0000 | 0.0000 | 0.0000 |
| dense_hyde_semantic | 0.4731 | 0.5000 | 0.0000 | 0.0000 |
