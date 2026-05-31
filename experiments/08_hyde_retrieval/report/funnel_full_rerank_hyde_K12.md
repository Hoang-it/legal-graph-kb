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

## Overall (all 200) (n=50)

| stage | avg \|pool\| | recall@12 | recall@all | NDCG@12 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 47.84 | 0.4057 | 0.5424 | 0.2419 | 0.2136 |
| sparse | 71.58 | 0.1595 | 0.3457 | 0.0761 | 0.0628 |
| dense ∪ sparse | 98.2 | 0.4057 | 0.5852 | — | — |
| post_temporal | 67.28 | 0.3357 | 0.5052 | — | — |
| fused (RRF) | 67.28 | 0.3029 | 0.5052 | 0.1355 | 0.1129 |
| rerank1 | 31.32 | 0.3124 | 0.4486 | 0.1888 | 0.1878 |
| expanded | 40.06 | 0.3124 | 0.4886 | 0.1888 | 0.1889 |
| final (rerank2) | 40.06 | 0.3124 | 0.4886 | 0.1888 | 0.1883 |

**Stage-to-stage gold count (sum over all questions):**

| from | to | total gold in `from` | total gold in `to` | Δ | cause |
|---|---|---:|---:|---:|---|
| dense ∪ sparse | post_temporal | 43 | 38 | -5 | temporal filter |
| post_temporal | fused (RRF) | 38 | 38 | +0 | RRF top-150 cap |
| fused (RRF) | rerank1 | 38 | 33 | -5 | rerank1 (50-seed cap) |
| rerank1 | expanded | 33 | 38 | +5 | graph expansion (additive) |
| expanded | final (rerank2) | 38 | 38 | +0 | rerank2 (100-final cap) |

## in_corpus stratum (n=38)

| stage | avg \|pool\| | recall@12 | recall@all | NDCG@12 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 47.37 | 0.5207 | 0.7005 | 0.3134 | 0.2781 |
| sparse | 71.74 | 0.2099 | 0.4549 | 0.1001 | 0.0826 |
| dense ∪ sparse | 97.84 | 0.5207 | 0.7569 | — | — |
| post_temporal | 65.76 | 0.4286 | 0.6516 | — | — |
| fused (RRF) | 65.76 | 0.3985 | 0.6516 | 0.1783 | 0.1475 |
| rerank1 | 30.87 | 0.4110 | 0.5902 | 0.2485 | 0.2471 |
| expanded | 39.92 | 0.4110 | 0.6297 | 0.2485 | 0.2480 |
| final (rerank2) | 39.92 | 0.4110 | 0.6297 | 0.2485 | 0.2472 |

**Stage-to-stage gold count (sum over all questions):**

| from | to | total gold in `from` | total gold in `to` | Δ | cause |
|---|---|---:|---:|---:|---|
| dense ∪ sparse | post_temporal | 42 | 37 | -5 | temporal filter |
| post_temporal | fused (RRF) | 37 | 37 | +0 | RRF top-150 cap |
| fused (RRF) | rerank1 | 37 | 33 | -4 | rerank1 (50-seed cap) |
| rerank1 | expanded | 33 | 37 | +4 | graph expansion (additive) |
| expanded | final (rerank2) | 37 | 37 | +0 | rerank2 (100-final cap) |

## mixed stratum (n=1)

| stage | avg \|pool\| | recall@12 | recall@all | NDCG@12 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 54 | 0.5000 | 0.5000 | 0.1846 | 0.1111 |
| sparse | 74 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dense ∪ sparse | 112 | 0.5000 | 0.5000 | — | — |
| post_temporal | 73 | 0.5000 | 0.5000 | — | — |
| fused (RRF) | 73 | 0.0000 | 0.5000 | 0.0000 | 0.0435 |
| rerank1 | 33 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| expanded | 51 | 0.0000 | 0.5000 | 0.0000 | 0.0244 |
| final (rerank2) | 51 | 0.0000 | 0.5000 | 0.0000 | 0.0213 |

**Stage-to-stage gold count (sum over all questions):**

| from | to | total gold in `from` | total gold in `to` | Δ | cause |
|---|---|---:|---:|---:|---|
| dense ∪ sparse | post_temporal | 1 | 1 | +0 | temporal filter |
| post_temporal | fused (RRF) | 1 | 1 | +0 | RRF top-150 cap |
| fused (RRF) | rerank1 | 1 | 0 | -1 | rerank1 (50-seed cap) |
| rerank1 | expanded | 0 | 1 | +1 | graph expansion (additive) |
| expanded | final (rerank2) | 1 | 1 | +0 | rerank2 (100-final cap) |

## ooc stratum (n=2)

| stage | avg \|pool\| | recall@12 | recall@all | NDCG@12 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 24.5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sparse | 38.5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dense ∪ sparse | 54 | 0.0000 | 0.0000 | — | — |
| post_temporal | 44.5 | 0.0000 | 0.0000 | — | — |
| fused (RRF) | 44.5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rerank1 | 16.5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| expanded | 19 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| final (rerank2) | 19 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**Stage-to-stage gold count (sum over all questions):**

| from | to | total gold in `from` | total gold in `to` | Δ | cause |
|---|---|---:|---:|---:|---|
| dense ∪ sparse | post_temporal | 0 | 0 | +0 | temporal filter |
| post_temporal | fused (RRF) | 0 | 0 | +0 | RRF top-150 cap |
| fused (RRF) | rerank1 | 0 | 0 | +0 | rerank1 (50-seed cap) |
| rerank1 | expanded | 0 | 0 | +0 | graph expansion (additive) |
| expanded | final (rerank2) | 0 | 0 | +0 | rerank2 (100-final cap) |

## unparseable stratum (n=9)

| stage | avg \|pool\| | recall@12 | recall@all | NDCG@12 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 54.33 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sparse | 78 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dense ∪ sparse | 108 | 0.0000 | 0.0000 | — | — |
| post_temporal | 78.11 | 0.0000 | 0.0000 | — | — |
| fused (RRF) | 78.11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rerank1 | 36.33 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| expanded | 44.11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| final (rerank2) | 44.11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**Stage-to-stage gold count (sum over all questions):**

| from | to | total gold in `from` | total gold in `to` | Δ | cause |
|---|---|---:|---:|---:|---|
| dense ∪ sparse | post_temporal | 0 | 0 | +0 | temporal filter |
| post_temporal | fused (RRF) | 0 | 0 | +0 | RRF top-150 cap |
| fused (RRF) | rerank1 | 0 | 0 | +0 | rerank1 (50-seed cap) |
| rerank1 | expanded | 0 | 0 | +0 | graph expansion (additive) |
| expanded | final (rerank2) | 0 | 0 | +0 | rerank2 (100-final cap) |

## Notes

- `dense ∪ sparse` is the pre-temporal-filter pool (SET, not ranked) — NDCG/MRR not reported.
- `post_temporal` ordering = dense pool (first) then sparse minus dense — synthetic order, rank-aware metrics partly mechanical.
- `expanded` = rerank1 seeds (rerank1-score order) followed by REFERS_TO neighbours (graph order) — same caveat.
- `rerank1` and `final (rerank2)` are TRUE rankings — NDCG/MRR reflect the cross-encoder's decisions.
- To compare HyDE vs no-HyDE side-by-side, also run `python scripts/exp06_funnel.py` (or `exp08`'s own no-HyDE control via `--arm full_rerank` if you add that mode in future).
