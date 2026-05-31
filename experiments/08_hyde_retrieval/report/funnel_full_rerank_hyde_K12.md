# Funnel — `full_rerank_hyde` arm at K=12 (exp 08, n=200)

Per-stage retrieval recall + rank-aware metrics for the HyDE-
augmented full pipeline. Computed by
[`scripts/exp08_funnel.py`](../../../scripts/exp08_funnel.py) from
the records in `results/full_rerank_hyde/`. Macro-averaged across
questions with non-empty gold.

Note: only the DENSE channel uses the HyDE doc embedding. Sparse
channel keeps the raw question (plan §D3), so any lift in the
post-RRF stages reflects the dense-side improvement propagating
through the rest of the pipeline.

## Overall (all 200) (n=200)

| stage | avg \|pool\| | recall@12 | recall@all | NDCG@12 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 47.06 | 0.3707 | 0.5466 | 0.2308 | 0.2244 |
| sparse | 66.94 | 0.1820 | 0.4280 | 0.0849 | 0.0764 |
| dense ∪ sparse | 92.2 | 0.3707 | 0.5958 | — | — |
| post_temporal | 61.88 | 0.3941 | 0.5373 | — | — |
| fused (RRF) | 61.86 | 0.3688 | 0.5373 | 0.2148 | 0.2133 |
| rerank1 | 30.11 | 0.3391 | 0.4885 | 0.2044 | 0.1996 |
| expanded | 37.55 | 0.3391 | 0.5298 | 0.2044 | 0.2005 |
| final (rerank2) | 37.55 | 0.3391 | 0.5298 | 0.2042 | 0.2001 |

**Stage-to-stage gold count (sum over all questions):**

| from | to | total gold in `from` | total gold in `to` | Δ | cause |
|---|---|---:|---:|---:|---|
| dense ∪ sparse | post_temporal | 199 | 181 | -18 | temporal filter |
| post_temporal | fused (RRF) | 181 | 181 | +0 | RRF top-150 cap |
| fused (RRF) | rerank1 | 181 | 159 | -22 | rerank1 (50-seed cap) |
| rerank1 | expanded | 159 | 177 | +18 | graph expansion (additive) |
| expanded | final (rerank2) | 177 | 177 | +0 | rerank2 (100-final cap) |

## in_corpus stratum (n=151)

| stage | avg \|pool\| | recall@12 | recall@all | NDCG@12 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 46.54 | 0.4648 | 0.6905 | 0.2846 | 0.2708 |
| sparse | 68.09 | 0.2118 | 0.5377 | 0.0971 | 0.0860 |
| dense ∪ sparse | 92.27 | 0.4648 | 0.7557 | — | — |
| post_temporal | 60.93 | 0.4927 | 0.6782 | — | — |
| fused (RRF) | 60.91 | 0.4592 | 0.6782 | 0.2594 | 0.2480 |
| rerank1 | 29.81 | 0.4241 | 0.6170 | 0.2562 | 0.2482 |
| expanded | 37.17 | 0.4241 | 0.6683 | 0.2562 | 0.2491 |
| final (rerank2) | 37.17 | 0.4241 | 0.6683 | 0.2560 | 0.2486 |

**Stage-to-stage gold count (sum over all questions):**

| from | to | total gold in `from` | total gold in `to` | Δ | cause |
|---|---|---:|---:|---:|---|
| dense ∪ sparse | post_temporal | 188 | 170 | -18 | temporal filter |
| post_temporal | fused (RRF) | 170 | 170 | +0 | RRF top-150 cap |
| fused (RRF) | rerank1 | 170 | 149 | -21 | rerank1 (50-seed cap) |
| rerank1 | expanded | 149 | 166 | +17 | graph expansion (additive) |
| expanded | final (rerank2) | 166 | 166 | +0 | rerank2 (100-final cap) |

## mixed stratum (n=5)

| stage | avg \|pool\| | recall@12 | recall@all | NDCG@12 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 46.6 | 0.2917 | 0.5083 | 0.2507 | 0.4313 |
| sparse | 67.6 | 0.3833 | 0.3833 | 0.2238 | 0.2848 |
| dense ∪ sparse | 91.6 | 0.2917 | 0.5083 | — | — |
| post_temporal | 62.8 | 0.3833 | 0.5083 | — | — |
| fused (RRF) | 62.8 | 0.3833 | 0.5083 | 0.3945 | 0.7087 |
| rerank1 | 32.6 | 0.2583 | 0.4083 | 0.1678 | 0.2900 |
| expanded | 43.8 | 0.2583 | 0.5083 | 0.1678 | 0.2949 |
| final (rerank2) | 43.8 | 0.2583 | 0.5083 | 0.1678 | 0.2943 |

**Stage-to-stage gold count (sum over all questions):**

| from | to | total gold in `from` | total gold in `to` | Δ | cause |
|---|---|---:|---:|---:|---|
| dense ∪ sparse | post_temporal | 8 | 8 | +0 | temporal filter |
| post_temporal | fused (RRF) | 8 | 8 | +0 | RRF top-150 cap |
| fused (RRF) | rerank1 | 8 | 7 | -1 | rerank1 (50-seed cap) |
| rerank1 | expanded | 7 | 8 | +1 | graph expansion (additive) |
| expanded | final (rerank2) | 8 | 8 | +0 | rerank2 (100-final cap) |

## ooc stratum (n=8)

| stage | avg \|pool\| | recall@12 | recall@all | NDCG@12 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 40.88 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sparse | 50.75 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dense ∪ sparse | 74.12 | 0.0000 | 0.0000 | — | — |
| post_temporal | 51.12 | 0.0000 | 0.0000 | — | — |
| fused (RRF) | 51.12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rerank1 | 24.25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| expanded | 29.5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| final (rerank2) | 29.5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**Stage-to-stage gold count (sum over all questions):**

| from | to | total gold in `from` | total gold in `to` | Δ | cause |
|---|---|---:|---:|---:|---|
| dense ∪ sparse | post_temporal | 0 | 0 | +0 | temporal filter |
| post_temporal | fused (RRF) | 0 | 0 | +0 | RRF top-150 cap |
| fused (RRF) | rerank1 | 0 | 0 | +0 | rerank1 (50-seed cap) |
| rerank1 | expanded | 0 | 0 | +0 | graph expansion (additive) |
| expanded | final (rerank2) | 0 | 0 | +0 | rerank2 (100-final cap) |

## unparseable stratum (n=36)

| stage | avg \|pool\| | recall@12 | recall@all | NDCG@12 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 50.69 | 0.0694 | 0.0694 | 0.0538 | 0.0509 |
| sparse | 65.64 | 0.0694 | 0.0694 | 0.0336 | 0.0243 |
| dense ∪ sparse | 96 | 0.0694 | 0.0694 | — | — |
| post_temporal | 68.11 | 0.0694 | 0.0694 | — | — |
| fused (RRF) | 68.11 | 0.0694 | 0.0694 | 0.0502 | 0.0463 |
| rerank1 | 32.31 | 0.0694 | 0.0694 | 0.0375 | 0.0278 |
| expanded | 40.08 | 0.0694 | 0.0694 | 0.0375 | 0.0278 |
| final (rerank2) | 40.08 | 0.0694 | 0.0694 | 0.0375 | 0.0278 |

**Stage-to-stage gold count (sum over all questions):**

| from | to | total gold in `from` | total gold in `to` | Δ | cause |
|---|---|---:|---:|---:|---|
| dense ∪ sparse | post_temporal | 3 | 3 | +0 | temporal filter |
| post_temporal | fused (RRF) | 3 | 3 | +0 | RRF top-150 cap |
| fused (RRF) | rerank1 | 3 | 3 | +0 | rerank1 (50-seed cap) |
| rerank1 | expanded | 3 | 3 | +0 | graph expansion (additive) |
| expanded | final (rerank2) | 3 | 3 | +0 | rerank2 (100-final cap) |

## Notes

- `dense ∪ sparse` is the pre-temporal-filter pool (SET, not ranked) — NDCG/MRR not reported.
- `post_temporal` ordering = dense pool (first) then sparse minus dense — synthetic order, rank-aware metrics partly mechanical.
- `expanded` = rerank1 seeds (rerank1-score order) followed by REFERS_TO neighbours (graph order) — same caveat.
- `rerank1` and `final (rerank2)` are TRUE rankings — NDCG/MRR reflect the cross-encoder's decisions.
- To compare HyDE vs no-HyDE side-by-side, also run `python scripts/exp06_funnel.py` (or `exp08`'s own no-HyDE control via `--arm full_rerank` if you add that mode in future).
