# Funnel — `full_rerank` arm at K=12 (exp 06, n=200)

Per-stage retrieval recall + rank-aware metrics. Computed by
[`scripts/exp06_funnel.py`](../../../scripts/exp06_funnel.py) from
the 200 records in `results/full_rerank/`. Macro-averaged across
questions with non-empty gold.

## Overall (all 200) (n=200)

| stage | avg \|pool\| | recall@12 | recall@all | NDCG@12 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 18.58 | 0.3026 | 0.3734 | 0.1714 | 0.1631 |
| sparse | 23.25 | 0.1820 | 0.2653 | 0.0849 | 0.0715 |
| dense ∪ sparse | 36.02 | 0.3026 | 0.4320 | — | — |
| post_temporal | 24.27 | 0.3334 | 0.3920 | — | — |
| fused (RRF) | 24.23 | 0.2970 | 0.3920 | 0.1825 | 0.1771 |
| rerank1 (top-15) | 10.92 | 0.3640 | 0.3653 | 0.2265 | 0.2176 |
| expanded | 14.56 | 0.3782 | 0.4146 | 0.2307 | 0.2208 |
| final (rerank2, top-12) | 9.15 | 0.3568 | 0.3568 | 0.2231 | 0.2141 |

**Stage-to-stage gold count (sum over all questions):**

| from | to | total gold in `from` | total gold in `to` | Δ | cause |
|---|---|---:|---:|---:|---|
| dense ∪ sparse | post_temporal | 137 | 125 | -12 | temporal filter |
| post_temporal | fused (RRF) | 125 | 125 | +0 | RRF top-50 cap |
| fused (RRF) | rerank1 (top-15) | 125 | 113 | -12 | rerank1 (15-seed cap) |
| rerank1 (top-15) | expanded | 113 | 132 | +19 | graph expansion (additive) |
| expanded | final (rerank2, top-12) | 132 | 109 | -23 | rerank2 (12-final cap) |

## in_corpus stratum (n=151)

| stage | avg \|pool\| | recall@12 | recall@all | NDCG@12 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 18.42 | 0.3779 | 0.4662 | 0.2102 | 0.1950 |
| sparse | 23.51 | 0.2118 | 0.3221 | 0.0971 | 0.0794 |
| dense ∪ sparse | 36.14 | 0.3779 | 0.5396 | — | — |
| post_temporal | 23.86 | 0.4165 | 0.4866 | — | — |
| fused (RRF) | 23.82 | 0.3674 | 0.4866 | 0.2216 | 0.2096 |
| rerank1 (top-15) | 10.85 | 0.4562 | 0.4579 | 0.2834 | 0.2695 |
| expanded | 14.46 | 0.4750 | 0.5157 | 0.2890 | 0.2729 |
| final (rerank2, top-12) | 9.12 | 0.4467 | 0.4467 | 0.2789 | 0.2649 |

**Stage-to-stage gold count (sum over all questions):**

| from | to | total gold in `from` | total gold in `to` | Δ | cause |
|---|---|---:|---:|---:|---|
| dense ∪ sparse | post_temporal | 127 | 115 | -12 | temporal filter |
| post_temporal | fused (RRF) | 115 | 115 | +0 | RRF top-50 cap |
| fused (RRF) | rerank1 (top-15) | 115 | 105 | -10 | rerank1 (15-seed cap) |
| rerank1 (top-15) | expanded | 105 | 121 | +16 | graph expansion (additive) |
| expanded | final (rerank2, top-12) | 121 | 101 | -20 | rerank2 (12-final cap) |

## mixed stratum (n=5)

| stage | avg \|pool\| | recall@12 | recall@all | NDCG@12 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 19.6 | 0.1917 | 0.3583 | 0.1747 | 0.3167 |
| sparse | 24.2 | 0.3833 | 0.3833 | 0.2238 | 0.2848 |
| dense ∪ sparse | 36.8 | 0.1917 | 0.4833 | — | — |
| post_temporal | 26.4 | 0.2583 | 0.4833 | — | — |
| fused (RRF) | 26.4 | 0.2833 | 0.4833 | 0.2263 | 0.3356 |
| rerank1 (top-15) | 11.4 | 0.2833 | 0.2833 | 0.2305 | 0.3667 |
| expanded | 17.2 | 0.2833 | 0.5083 | 0.2305 | 0.3884 |
| final (rerank2, top-12) | 10 | 0.2833 | 0.2833 | 0.2299 | 0.3667 |

**Stage-to-stage gold count (sum over all questions):**

| from | to | total gold in `from` | total gold in `to` | Δ | cause |
|---|---|---:|---:|---:|---|
| dense ∪ sparse | post_temporal | 7 | 7 | +0 | temporal filter |
| post_temporal | fused (RRF) | 7 | 7 | +0 | RRF top-50 cap |
| fused (RRF) | rerank1 (top-15) | 7 | 5 | -2 | rerank1 (15-seed cap) |
| rerank1 (top-15) | expanded | 5 | 8 | +3 | graph expansion (additive) |
| expanded | final (rerank2, top-12) | 8 | 5 | -3 | rerank2 (12-final cap) |

## ooc stratum (n=8)

| stage | avg \|pool\| | recall@12 | recall@all | NDCG@12 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 16.38 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sparse | 18.38 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dense ∪ sparse | 28.88 | 0.0000 | 0.0000 | — | — |
| post_temporal | 18.62 | 0.0000 | 0.0000 | — | — |
| fused (RRF) | 18.62 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rerank1 (top-15) | 9.38 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| expanded | 12.12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| final (rerank2, top-12) | 7.75 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**Stage-to-stage gold count (sum over all questions):**

| from | to | total gold in `from` | total gold in `to` | Δ | cause |
|---|---|---:|---:|---:|---|
| dense ∪ sparse | post_temporal | 0 | 0 | +0 | temporal filter |
| post_temporal | fused (RRF) | 0 | 0 | +0 | RRF top-50 cap |
| fused (RRF) | rerank1 (top-15) | 0 | 0 | +0 | rerank1 (15-seed cap) |
| rerank1 (top-15) | expanded | 0 | 0 | +0 | graph expansion (additive) |
| expanded | final (rerank2, top-12) | 0 | 0 | +0 | rerank2 (12-final cap) |

## unparseable stratum (n=36)

| stage | avg \|pool\| | recall@12 | recall@all | NDCG@12 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 19.58 | 0.0694 | 0.0694 | 0.0460 | 0.0440 |
| sparse | 23.14 | 0.0694 | 0.0694 | 0.0336 | 0.0243 |
| dense ∪ sparse | 36.97 | 0.0694 | 0.0694 | — | — |
| post_temporal | 26.92 | 0.0694 | 0.0694 | — | — |
| fused (RRF) | 26.92 | 0.0694 | 0.0694 | 0.0528 | 0.0583 |
| rerank1 (top-15) | 11.47 | 0.0694 | 0.0694 | 0.0375 | 0.0278 |
| expanded | 15.19 | 0.0694 | 0.0694 | 0.0375 | 0.0278 |
| final (rerank2, top-12) | 9.5 | 0.0694 | 0.0694 | 0.0375 | 0.0278 |

**Stage-to-stage gold count (sum over all questions):**

| from | to | total gold in `from` | total gold in `to` | Δ | cause |
|---|---|---:|---:|---:|---|
| dense ∪ sparse | post_temporal | 3 | 3 | +0 | temporal filter |
| post_temporal | fused (RRF) | 3 | 3 | +0 | RRF top-50 cap |
| fused (RRF) | rerank1 (top-15) | 3 | 3 | +0 | rerank1 (15-seed cap) |
| rerank1 (top-15) | expanded | 3 | 3 | +0 | graph expansion (additive) |
| expanded | final (rerank2, top-12) | 3 | 3 | +0 | rerank2 (12-final cap) |

## Notes

- `dense ∪ sparse` is the pre-temporal-filter pool used by the audit. It is a SET (rank-aware metrics not meaningful), so NDCG/MRR are not reported for it.
- `post_temporal` ordering = dense pool (first), then sparse pool minus dense — i.e. the order the retriever stored. Rank-aware metrics on it reflect that synthetic ordering, not a true ranking.
- `expanded` = rerank1 seeds (in rerank1 score order) followed by REFERS_TO neighbours (in graph traversal order). Same caveat as post_temporal — the rank is partly mechanical.
- `rerank1` and `final (rerank2)` are TRUE rankings — NDCG/MRR there reflect the cross-encoder's decisions.
- Gold counts in the funnel use `round(recall@all × |gold|)`. Sum across questions, so a single gold article can be counted multiple times if multiple questions share it.
