# Funnel — `cypher_walk` arm at K=12 (exp 11, n=50)

Per-stage **article-level** retrieval recall + gold-hit counts for the
Cypher-walk retriever. Computed by
[`scripts/exp11_funnel.py`](../../../scripts/exp11_funnel.py) from the
per-stage projections in `results/cypher_walk/`. Macro-averaged across
questions with non-empty gold.

Key question: does `+ cypher_new` raise recall above `seed (vector)`? If
`recall@all` for `+ cypher_new` equals `seed (vector)`, the walk surfaced
no gold the seed missed (the previous attempt's failure mode).

## Overall (all 50) (n=50)

| stage | avg \|pool\| | recall@12 | recall@all | gold-hits (Σ) |
|---|---:|---:|---:|---:|
| seed (vector) | 6.58 | 0.1583 | 0.1583 | 14 |
| + cypher_new | 10.8 | 0.1583 | 0.1583 | 14 |
| + fallback | 6.98 | 0.1817 | 0.1817 | 17 |
| final (fused top-K) | 7.64 | 0.1650 | 0.1650 | 15 |

## l41_only stratum (n=28)

| stage | avg \|pool\| | recall@12 | recall@all | gold-hits (Σ) |
|---|---:|---:|---:|---:|
| seed (vector) | 6.64 | 0.1458 | 0.1458 | 6 |
| + cypher_new | 10.46 | 0.1458 | 0.1458 | 6 |
| + fallback | 7.04 | 0.1756 | 0.1756 | 8 |
| final (fused top-K) | 7.11 | 0.1458 | 0.1458 | 6 |

## mixed_l41_other stratum (n=9)

| stage | avg \|pool\| | recall@12 | recall@all | gold-hits (Σ) |
|---|---:|---:|---:|---:|
| seed (vector) | 6.33 | 0.2593 | 0.2593 | 6 |
| + cypher_new | 11.67 | 0.2593 | 0.2593 | 6 |
| + fallback | 6.89 | 0.2963 | 0.2963 | 7 |
| final (fused top-K) | 8.11 | 0.2963 | 0.2963 | 7 |

## no_l41 stratum (n=13)

| stage | avg \|pool\| | recall@12 | recall@all | gold-hits (Σ) |
|---|---:|---:|---:|---:|
| seed (vector) | 6.62 | 0.1154 | 0.1154 | 2 |
| + cypher_new | 10.92 | 0.1154 | 0.1154 | 2 |
| + fallback | 6.92 | 0.1154 | 0.1154 | 2 |
| final (fused top-K) | 8.46 | 0.1154 | 0.1154 | 2 |

## Notes

- `+ cypher_new` = seed ∪ articles the Cypher walk surfaced beyond seed.
- `+ fallback` = seed ∪ articles the vanilla expand fallback added (populated only on questions where the Cypher walk found 0 new clauses).
- `final (fused top-K)` = the RRF-fused top-K actually returned — can be
  below `+ cypher_new`/`+ fallback` recall@all because it is K-capped.
- Gold-hits (Σ) = sum over questions of |stage pool ∩ gold| — the absolute
  count of gold articles present at each stage.
