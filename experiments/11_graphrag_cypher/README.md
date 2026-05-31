# exp 11 (REDO) — `CypherWalkRetriever` retrieval-only audit

> **Status: 50-question pilot run (2026-06-01). Result: `cypher_walk` LOSES.**
> The component works mechanically (it surfaces new clauses 76% of the time,
> fixing the previous attempt's `n_cypher_new=0`), but the new clauses are
> **noise, not gold** — `cypher_walk` recall@12 is **−0.0417 below**
> `dense_vanilla`. The incidental winner is `dense_then_expand` (+0.0517).
> See "Result" below.
>
> Plan: [`docs/plans/exp11_cypher_walk_retriever.md`](../../docs/plans/exp11_cypher_walk_retriever.md).

## Result — 50-question pilot (2026-06-01)

Stratified pilot (`l41_only=28`, `mixed_l41_other=9`, `no_l41=13`), seed=0.
gpt-4o-mini Cypher generator, T=0. Cost ≈ **$0.036** (188.5K prompt + 13.1K
completion tokens). 0 failures. Metrics: [`report/retrieval_report.md`](report/retrieval_report.md),
[`metrics/academic_metrics.json`](metrics/academic_metrics.json). Funnel:
[`report/funnel_cypher_walk_K12.md`](report/funnel_cypher_walk_K12.md).

### Article-level recall@K (n=50)

| arm | R@5 | R@12 | R@20 | NDCG@12 | MRR |
|---|---:|---:|---:|---:|---:|
| dense_vanilla | 0.1483 | 0.2067 | 0.2067 | 0.1247 | 0.1226 |
| **dense_then_expand** | 0.1483 | **0.2583** | **0.2750** | **0.1406** | **0.1294** |
| cypher_walk | 0.1150 | 0.1650 | 0.1650 | 0.1038 | 0.1025 |

`cypher_walk` is **worst on every metric**. `dense_then_expand` (vanilla
graphrag's REFERENCES/CITES_EXTERNAL expansion, no LLM) is **best**:
recall@12 +0.0517 vs vanilla.

### Pre-commitment check (plan §5.5) — 2 of 4 FAILED

| prediction | threshold | observed | verdict |
|---|---|---:|:-:|
| cypher_used rate | ≥ 0.30 | 0.7600 (38/50) | ✓ |
| mean n_cypher_new (when used) | 1.5–3.0 | **14.71** | ✗ AUDIT |
| recall@12 lift vs dense_vanilla | +0.00 to +0.05 | **−0.0417** | ✗ AUDIT |
| recall@12 lift on no_l41 stratum | near 0 | 0.0000 | ✓ |

### Funnel — why it loses (the honest mechanism)

| stage | avg \|pool\| | recall@all | gold-hits (Σ) |
|---|---:|---:|---:|
| seed (vector, top-8) | 6.58 | 0.1583 | 14 |
| + cypher_new | 10.8 | **0.1583** | **14** |
| + fallback | 6.98 | 0.1817 | 17 |
| final (fused top-12) | 7.64 | 0.1650 | 15 |

The Cypher walk adds ~4 new articles per question (≈15 new clauses, mean
`n_cypher_new=14.71`) but **gold-hits stays 14 → 14** — *not one* of the new
articles is gold. The LLM writes broad traversals (e.g. "cousin-article
clauses CONTAINS keyword", LIMIT 30) that dump loosely-related clauses; RRF
(k=60) then interleaves that noise with the precise dense seed and the top-12
cap drops good hits. Net: 0.1650 < the 0.2067 you get by just taking the
top-12 dense hits. The `+ fallback` gold gain (14→17) comes from the 12
questions where Cypher found nothing and vanilla expand kicked in — which is
exactly why `dense_then_expand` wins.

### Stratified recall@12

| arm | l41_only (28) | mixed (9) | no_l41 (13) |
|---|---:|---:|---:|
| dense_vanilla | 0.2321 | 0.2593 | 0.1154 |
| dense_then_expand | **0.3244** | 0.2593 | 0.1154 |
| cypher_walk | 0.1458 | **0.2963** | 0.1154 |

`no_l41` is **identical across all three arms** — the KG has no useful
outward edges reaching non-L41 gold, so neither expansion nor the walk helps
the hard subset. `cypher_walk` only ties the best on the small `mixed`
stratum (n=9).

### Conclusion

The hypothesis "LLM graph-walking improves retrieval" is **refuted** on this
pilot. The redo fixed the previous attempt's degeneracy (walks now surface
new clauses), but those clauses carry no gold, so fusing them in degrades
recall. **Precondition for plan §8 (plug into Logic-LM) is NOT met** —
`cypher_walk` shows no useful lift, so it should not be promoted. If anything
is worth pursuing it is `dense_then_expand` (cheap, no LLM, +0.05 recall@12).
A full-200 run could confirm, but the pilot signal is a clean negative for
the Cypher walk.

## What

A **retrieval-layer** component — peer of `RagPipeline.vector_search` and
`V5RetrievalPipeline`. It takes a question and returns a ranked set of
clauses + provenance. **No LLM render, no citation parsing, no answer
generation.** ([`runtime/retrievers/cypher_walk.py`](../../runtime/retrievers/cypher_walk.py))

Three retrieval-only arms (plan §5.2), all on the **vanilla `clause_vec`**
dense channel (no v5 tuned index / reranker), so they are comparable to
vanilla graphrag and to each other — but **not** to exp 06/07/08 absolute
numbers:

| Arm | Pipeline |
|---|---|
| `dense_vanilla` | `RagPipeline.vector_search(top_k=12)` — baseline |
| `dense_then_expand` | `vector_search(top_k=12)` + `RagPipeline.expand` REFERENCES/CITES_EXTERNAL refs folded in at the article level (vanilla graphrag's retrieval side, minus the LLM) |
| `cypher_walk` | `CypherWalkRetriever.retrieve` — vector seed → LLM **outward** Cypher walk → fallback expand → RRF (k=60) |

### The one change that matters vs the deprecated arm

The previous attempt's Cypher prompt pinned every row with
`WHERE r.source_clause IN $seed_ids`, which anchors results to the seed by
construction → it surfaced **0 new clauses** (mean `n_cypher_new = 0`). The
redo:

- **forbids the pin** in the validator
  ([`validate_cypher`](../../runtime/retrievers/cypher_walk.py)) — a query
  using `<rel>.source_clause IN $seed_ids` is rejected;
- **requires node-identity seeding** (`<node>.id IN $seed_ids`) so the walk
  starts at the seed nodes and traverses OUTWARD;
- **requires** the `RETURN` to expose `target_clause_id` / `target_article_id`
  of a *traversed* node (not the seed anchor).

`n_cypher_new` (NEW clauses surfaced beyond the seed) is therefore the key
diagnostic — if it stays 0, the walk is doing nothing the vector seed didn't
already do.

## Why

Vanilla GraphRAG does neighbour *expansion* but the LLM never *queries* the
graph. Hypothesis: letting the LLM author a schema-constrained walk
conditional on the question surfaces gold the vector seed missed — but only
where the relevant outward edges exist. The KG's semantic edges are sparse
(40% of L41 Articles have 0 semantic edges; many gold laws are
Nghị định/Thông tư with even thinner coverage), so the realistic expectation
is a small, measurable lift on a subset, not a large one. This audit is built
to make that *measurable*, not to mask it.

## Metrics

Article-level **recall@K / precision@K / F1@K / NDCG@K** for K ∈ {5, 12, 20,
all}, plus R-Precision and MRR. Stratified by **L41 presence in gold**:
`l41_only`, `mixed_l41_other`, `no_l41` (the hard subset). Plus the
`cypher_walk` provenance diagnostics: `cypher_used` rate, mean `n_cypher_new`
(overall + conditional on `cypher_used`), `fallback_used` rate, mean
`cypher_attempts`. Funnel: per-stage article pools + gold-hit counts
(seed → +cypher_new → +fallback → final).

### Honest limitations

- **Clause-level recall is NOT measurable on this dataset.** Verified against
  `eval_core.gold`: **0 / 200** gold citations carry khoản-level detail
  (`gold_items == gold_articles` for all 200, granularity = `tuple`). The
  plan (§5.3) asked for clause-level recall, but the gold has none. Reporting
  one would mean fabricating clause gold. We report article-level only and
  the metrics script records the probe (`clause_level_note`) so this stays
  visible. Provenance still keeps `final_clause_ids` for inspection.
- Absolute numbers are **not** comparable to exp 06/07/08 (those use the v5
  tuned index + reranker; exp 11 uses vanilla `clause_vec`).
- The 50-question pilot is stratified by **L41 presence**, not by the
  in-KG/ooc axis exp 08 used — a different (more direct) cut for this
  hypothesis. The actual stratum sizes come from the data, not the plan's
  pre-registration estimate (110 / 37 / 53 L41-only/mixed/no-L41 across the
  full 200).

## Pre-commitment predictions (plan §5.5)

Stated before the run so the result can't be rationalised. The metrics script
emits a pass/`AUDIT` table against these:

- `cypher_used` rate (≥1 NEW clause beyond seed): **predict ≥ 30%**.
- mean `n_cypher_new` given `cypher_used=True`: **predict 1.5–3.0**.
- recall@12 lift vs `dense_vanilla` (all strata): **predict +0.00 to +0.05**.
- recall@12 lift on the `no_l41` stratum: **predict near 0**.

> If recall@12 lift > +0.05 across all strata, treat as **suspicious and
> audit before celebrating** (the metrics report flags it).

## How to run (not yet executed)

```powershell
# Pilot — 50 stratified questions (only cypher_walk hits OpenAI):
python -m scripts.exp11_run --pilot-50
python -m scripts.exp11_metrics            # auto-filters to the pilot subset
python -m scripts.exp11_funnel             # cypher_walk per-stage funnel

# Full 200 once the pilot signal is useful:
python -m scripts.exp11_run
python -m scripts.exp11_metrics --full
python -m scripts.exp11_funnel --full
```

The runner is idempotent (`--force` to overwrite) and aborts pre-flight if
the Cypher-LLM cost estimate exceeds `--cost-cap` (default $0.50).

## Deprecation — the prior `graphrag_cypher` E2E arm (commit `a10f609`)

The original exp 11 shipped a `graphrag_cypher` **E2E arm** that was
architecturally misplaced (graphrag-family render instead of a retrieval
component) and whose BERTScore numbers confounded three unrelated changes
(Cypher walk + context layout + answer prompt). Plan §1 and §7 explain why.

Per plan §7, the cleanup of that arm — moving
[`runtime/graphrag_cypher.py`](../../runtime/graphrag_cypher.py), removing
`"graphrag_cypher"` from `eval_core/arms.py` + `eval_core/inference.py`,
deleting `prompts/runtime/graphrag_cypher/`, and wiping the old
`results/graphrag_cypher/` + `metrics/academic_metrics.*` +
`report/academic_report.md` — happens **after** the retrieval-only redo has
been run, not before, to avoid leaving the repo in a half-cleaned state. It is
therefore **out of scope for this change** (retrieval-layer only).

⚠️ Until that cleanup: the files under `results/graphrag_cypher/`, the
existing `metrics/academic_metrics.*`, and `report/academic_report.md` are
**leftovers from the deprecated E2E arm**. They measure the wrong thing on a
wrongly-built component — **do not cite them.** When `exp11_metrics.py` runs,
it overwrites `metrics/academic_metrics.*` and writes
`report/retrieval_report.md` (the deprecated `report/academic_report.md`
should be removed in the §7 cleanup).

## Files

- Component: [`runtime/retrievers/cypher_walk.py`](../../runtime/retrievers/cypher_walk.py) (+ [`runtime/retrievers/__init__.py`](../../runtime/retrievers/__init__.py))
- Prompt: [`prompts/runtime/cypher_walk/cypher_gen.md`](../../prompts/runtime/cypher_walk/cypher_gen.md)
- Scripts: [`scripts/exp11_run.py`](../../scripts/exp11_run.py), [`scripts/exp11_metrics.py`](../../scripts/exp11_metrics.py), [`scripts/exp11_funnel.py`](../../scripts/exp11_funnel.py)
