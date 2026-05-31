# Experiment 12 — HyDE × CypherWalk 2×2 (retrieval-only, article-level)

Dataset: 50 BHXH questions (same pilot subset as exp 11).
All arms on the **vanilla `clause_vec`** stack (not v5 tuned) so the HyDE
effect is isolated; numbers are NOT comparable to exp 08 absolute values.
Plan motivation: exp 11 showed `cypher_walk` loses — does a HyDE seed fix it?

- **Clause-level recall NOT reported**: 0 clause-level gold cites (article-level only).

## Overall macro (n=50)

### Citation recall@K (article-level)
| arm | n | @5 | @12 | @20 | @all |
|---|---:|---:|---:|---:|---:|
| dense_vanilla | 50 | 0.1483 | 0.2067 | 0.2067 | 0.2067 |
| dense_hyde | 50 | 0.1857 | 0.2790 | 0.2790 | 0.2790 |
| cypher_walk | 50 | 0.1150 | 0.1650 | 0.1650 | 0.1650 |
| cypher_walk_hyde | 50 | 0.1717 | 0.2157 | 0.2157 | 0.2157 |

### NDCG@K
| arm | n | @5 | @12 | @20 | @all |
|---|---:|---:|---:|---:|---:|
| dense_vanilla | 50 | 0.1023 | 0.1247 | 0.1247 | 0.1247 |
| dense_hyde | 50 | 0.1142 | 0.1493 | 0.1493 | 0.1493 |
| cypher_walk | 50 | 0.0786 | 0.0989 | 0.0989 | 0.0989 |
| cypher_walk_hyde | 50 | 0.0971 | 0.1146 | 0.1146 | 0.1146 |

### Rank-aware + latency

| arm | n | R-Prec | MRR | avg elapsed (s) |
|---|---:|---:|---:|---:|
| dense_vanilla | 50 | 0.0483 | 0.1226 | 0.2970 |
| dense_hyde | 50 | 0.0490 | 0.1287 | 0.1940 |
| cypher_walk | 50 | 0.0450 | 0.0965 | 5.0700 |
| cypher_walk_hyde | 50 | 0.0350 | 0.0967 | 4.9810 |

## 2×2 interaction — recall@12 (the headline)

|  | no walk | + cypher walk | walk effect (Δ) |
|---|---:|---:|---:|
| **raw seed** | 0.2067 | 0.1650 | -0.0417 |
| **HyDE seed** | 0.2790 | 0.2157 | -0.0633 |
| **HyDE effect (Δ)** | 0.0723 | 0.0507 | |

- Best single arm (excl. combo) recall@12 = **0.2790**; combo `cypher_walk_hyde` = **0.2157** (Δ vs best = -0.0633).

## Pre-commitment check (stated in README before the run)

| prediction | threshold | observed | verdict |
|---|---|---:|:-:|
| walk hurts even on HyDE seed (cypher_walk_hyde − dense_hyde) | ≤ +0.02 | -0.0633 | ✓ |
| combo does not beat best single arm | ≤ +0.05 | -0.0633 | ✓ |
| cypher_walk: +cypher_new adds ~0 gold (Σ gold seed→+cypher) | ≈ equal | 14→15 | ✓ |
| cypher_walk_hyde: +cypher_new adds ~0 gold (Σ gold seed→+cypher) | ≈ equal | 20→21 | ✓ |

## Per-stage gold-hit funnel — both cypher arms

### cypher_walk (n=50)

| stage | recall@all | gold-hits (Σ) |
|---|---:|---:|
| seed | 0.1583 | 14 |
| +cypher_new | 0.1783 | 15 |
| final | 0.1650 | 15 |

### cypher_walk_hyde (n=50)

| stage | recall@all | gold-hits (Σ) |
|---|---:|---:|
| seed | 0.2390 | 20 |
| +cypher_new | 0.2590 | 21 |
| final | 0.2157 | 17 |

## CypherWalk provenance

| arm | cypher_used | mean n_cypher_new (when used) | fallback_used |
|---|---:|---:|---:|
| cypher_walk | 0.7800 | 14.8718 | 0.2200 |
| cypher_walk_hyde | 0.8600 | 14.7674 | 0.1400 |

## Stratified recall@12 by L41 presence

| arm | l41_only | mixed_l41_other | no_l41 |
|---|---:|---:|---:|
| dense_vanilla | 0.2321 | 0.2593 | 0.1154 |
| dense_hyde | 0.3244 | 0.2630 | 0.1923 |
| cypher_walk | 0.1458 | 0.2963 | 0.1154 |
| cypher_walk_hyde | 0.2351 | 0.2444 | 0.1538 |
