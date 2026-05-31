# exp 12 — HyDE × CypherWalk 2×2 (retrieval-only)

> Follow-on to [exp 11](../11_graphrag_cypher/README.md), which found
> `cypher_walk` **loses** (recall@12 −0.0417 vs vanilla) because the LLM
> Cypher walk surfaces new clauses that carry **no gold**. exp 11 seeded the
> walk from a raw-question vector search. This experiment swaps in a **HyDE**
> seed to test whether the loss was a *weak-seed* problem or an intrinsic
> *walk-adds-noise* problem.

## Design — 2×2 factorial

seed {raw, HyDE} × walk {off, on}, all on the **same vanilla
`RagPipeline` / `clause_vec`** stack as exp 11 (not the v5 tuned index exp 08
used), so the HyDE contribution is isolated and the dense/walk arms are
directly comparable to exp 11.

|  | no walk | + cypher walk |
|---|---|---|
| **raw seed** | `dense_vanilla` | `cypher_walk` |
| **HyDE seed** | `dense_hyde` | `cypher_walk_hyde` ← new |

- HyDE generator: OpenAI `gpt-4o-mini`, n=1, max_tokens=700, T=0, prompt
  [`prompts/runtime/hyde_generate.md`](../../prompts/runtime/hyde_generate.md)
  — identical config to exp 08, so its `artifacts/hyde/` disk cache is reused
  (overlapping questions cost $0). `dense_hyde` and `cypher_walk_hyde` share
  the same cached HyDE doc per question.
- Only the **seed embedding** differs between raw and HyDE arms; the
  `clause_vec` index, the Cypher walk, the validator, the fallback and the
  RRF fusion are unchanged. Wired via the new optional `seed_query_encoder`
  on [`CypherWalkRetriever`](../../runtime/retrievers/cypher_walk.py) +
  `RagPipeline.vector_search_by_vector`.

## Honest limitations

- **Clause-level recall NOT measurable** (0/200 gold cites carry khoản) —
  article-level only, same as exp 11.
- Absolute numbers **not** comparable to exp 08 (that used BGE-M3 LoRA +
  `clause_vec_tuned`; this uses base BGE-M3 + `clause_vec`).
- Same 50-question pilot subset as exp 11 (`pilot_50_stt.json` copied in):
  `l41_only=28`, `mixed_l41_other=9`, `no_l41=13`.

## Pre-commitment predictions (stated BEFORE the run)

Based on exp 11's finding that the walk surfaces noise, not gold:

1. **HyDE may help the seed** but I'm genuinely unsure of sign/size on the
   base `clause_vec` stack (exp 08's HyDE was on the tuned stack) — no firm
   threshold, reported as-is.
2. **The walk still hurts on a HyDE seed**: `cypher_walk_hyde − dense_hyde`
   **≤ +0.02** (the walk adds noise regardless of seed quality). If it
   *exceeds +0.05*, that's a surprise → **audit before celebrating**.
3. **The combination does not beat the best single arm**:
   `cypher_walk_hyde` ≤ max(`dense_vanilla`, `dense_hyde`, `cypher_walk`)
   **+0.05**.
4. **`+cypher_new` still adds ~0 gold** even with a HyDE seed (per-stage
   funnel: gold-hits at `seed` ≈ gold-hits at `+cypher_new`).

The metrics report emits a pass/`AUDIT` table against 2–4.

## How to run

```powershell
python -m scripts.exp12_run            # 4 arms on the pilot-50 (only HyDE/cypher arms hit OpenAI)
python -m scripts.exp12_metrics        # 2×2 table + funnel + pre-commitment check
```

## Result — 50-question pilot (2026-06-01)

Same pilot subset as exp 11. 0 failures. HyDE fully served from the exp 08
cache (**$0**); Cypher arms cost **$0.072** total (gpt-4o-mini). Report:
[`report/retrieval_report.md`](report/retrieval_report.md).

### 2×2 — recall@12 (article-level, n=50)

|  | no walk | + cypher walk | walk effect (Δ) |
|---|---:|---:|---:|
| **raw seed** | 0.2067 | 0.1650 | **−0.0417** |
| **HyDE seed** | **0.2790** | 0.2157 | **−0.0633** |
| **HyDE effect (Δ)** | **+0.0723** | +0.0507 | |

**Two clean findings:**

1. **HyDE on the seed is the real win** — `dense_hyde` 0.2790 vs `dense_vanilla`
   0.2067 = **+0.0723 recall@12** (+35% rel), NDCG@12 +0.025, and it's *faster*
   (cached HyDE doc → embed only). It even lifts the hard `no_l41` stratum
   0.1154 → **0.1923** — the subset *nothing* in exp 11 could move.
2. **The Cypher walk is harmful, and worse on a better seed** — walk Δ is
   −0.0417 on the raw seed and **−0.0633 on the HyDE seed**. The combination
   `cypher_walk_hyde` (0.2157) is **strictly dominated** by `dense_hyde`
   (0.2790): the walk throws away ~0.063 of HyDE's 0.072 gain.

### Why — the funnel shows the walk *evicts* gold

`cypher_walk_hyde` gold-hits (Σ over 50): seed **20** → +cypher_new **21**
(+1) → final (RRF top-12) **17**. The walk surfaces ~15 noise clauses/q
(`mean n_cypher_new=14.77`); RRF ranks them competitively and the top-12 cap
**drops 4 gold articles the HyDE seed had already found**. The walk doesn't
merely fail to add gold — it displaces gold. The better the seed, the more
there is to lose.

### Pre-commitment — all ✓ (predictions in the section above held)

| prediction | observed | verdict |
|---|---:|:-:|
| walk hurts even on HyDE seed (≤ +0.02) | −0.0633 | ✓ |
| combo ≤ best single arm + 0.05 | −0.0633 | ✓ |
| +cypher_new adds ~0 gold (raw / hyde) | 14→15 / 20→21 | ✓ |

### Conclusion

Combining HyDE with the Cypher walk is **not worth it** — HyDE *alone* is far
better and the walk actively undoes it. Confirms exp 11's negative result for
the walk across a second, stronger seed, and adds a strong **positive** result
for HyDE on the vanilla dense channel (cheap, faster, +0.072 recall@12,
+0.077 on `no_l41`). **Recommendation: drop the Cypher walk; pursue HyDE on
the dense seed.** A full-200 run would confirm, but the pilot signal is
unambiguous.
