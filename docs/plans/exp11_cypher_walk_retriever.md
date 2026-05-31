# exp 11 — `CypherWalkRetriever` — retrieval-layer component (REDO)

> **Status**: planned, not implemented. Supersedes the original `graphrag_cypher`
> arm shipped in commit `a10f609` — that arm is architecturally misplaced
> (graphrag-family E2E instead of retrieval-layer component) and its
> reported BERTScore metrics confound 3 unrelated changes. Section
> [§7 — Disposition of current `graphrag_cypher`](#7-disposition-of-current-graphrag_cypher-arm)
> explains the cleanup.

## 1. Honest accounting — why this plan exists

The first attempt (commit `a10f609`) made 3 layered mistakes:

1. **Wrong family.** The canonical E2E of this system is the Logic-LM
   pipeline ([`runtime/logic_lm_pipelines.py:634`](../../runtime/logic_lm_pipelines.py)):
   `retrieve → LLM Prolog gen → Prolog solver verify → IRAC render → citations`.
   I instead copied the shape of `RagPipeline.ask()` — which is a
   baseline *comparator*, not the canonical E2E — and added a Cypher loop
   between retrieve and render. The result was a graphrag-family arm
   that bypassed the solver entirely. There is no place in the
   reasoning architecture for such an arm.
2. **Wrong layer.** "Walking the graph with Cypher" is a **retrieval**
   concern (which clauses surface for downstream reasoning) — not a
   rendering concern. By placing it next to render I made it impossible
   to plug the same component into the Logic-LM family later.
3. **Wrong metric framing.** Retrieval components are audited
   retrieval-only (`recall@K`, `NDCG@K`, `MRR`) — see
   [`experiments/06_retrieval_dense_vs_full`](../../experiments/06_retrieval_dense_vs_full),
   [`07_retrieval_extended_k`](../../experiments/07_retrieval_extended_k),
   [`08_hyde_retrieval`](../../experiments/08_hyde_retrieval). I ran the
   arm E2E and reported BERTScore + citation_recall, which confound 3
   changes (Cypher walk + context layout + answer prompt) and tell us
   nothing isolatable about retrieval.

This plan reframes exp 11 so it audits retrieval-only, builds the
component at the right architectural layer, and stays comparable to
the existing retrieval audits.

## 2. Architectural position

`CypherWalkRetriever` sits **at the retrieval layer**, peer with the
existing retrieval components:

```
Retrieval layer (each returns ranked clauses + provenance):
├── RagPipeline.vector_search           — vanilla dense (graphrag baseline)
├── V5RetrievalPipeline                 — dense + sparse + RRF + CE rerank + REFERS_TO expand
├── OntologyRetrieval                   — for logic_lm_ontology
├── GraphRAGAsLogicLMRetriever          — adapts vanilla GraphRAG for Logic-LM
└── CypherWalkRetriever  ← NEW          — vector seed + LLM-Cypher walk
```

It produces a ranked clause set. Nothing else. **No LLM render. No
citation parsing. No answer generation.**

Adapting it for Logic-LM (so it can power `logic_lm_graphrag_cypher` in a
future experiment) is a separate concern and is **out of scope for
exp 11** — see [§8](#8-out-of-scope-future-logic-lm-integration).

## 3. `CypherWalkRetriever` — interface contract

```python
@dataclass
class RetrievedClause:
    clause_id: str              # e.g. "L41_2024.A64.K1"
    article_id: str
    article_number: int
    article_title: str
    clause_number: int
    text: str
    score: float                # final rank score (fused)
    source: str                 # "vector" | "cypher" | "fallback_expand"

@dataclass
class CypherWalkResult:
    hits: list[RetrievedClause]              # top-K final
    n_seed: int                              # vector seed count
    n_cypher_new: int                        # NEW clauses surfaced by Cypher
                                             # (beyond seed) — KEY signal
    n_fallback_added: int                    # clauses added by vanilla expand
                                             # when Cypher returned 0 new
    cypher_used: bool                        # ≥1 NEW clause from Cypher
    fallback_used: bool                      # vanilla expand kicked in
    cypher_attempts: list[CypherAttempt]     # per-round provenance
    elapsed_breakdown: dict[str, float]      # vector_s / cypher_s / fuse_s

class CypherWalkRetriever:
    def __init__(self,
                 rag: RagPipeline,
                 top_k_seed: int = 8,
                 top_k_final: int = 12,
                 max_repair_rounds: int = 2,
                 cypher_model: str | None = None): ...

    def retrieve(self, question: str) -> CypherWalkResult: ...
```

The contract is intentionally narrower than `RagPipeline.ask`: input
is a question, output is a ranked set of clauses + provenance. No
`answer`, no `citations`. This makes it a drop-in replacement for any
component that today calls `vector_search`.

## 4. Pipeline design (retrieval-only)

The CRITICAL change vs the previous attempt: the Cypher pattern must
be allowed — and encouraged — to **surface clauses BEYOND the seed
set**. The previous attempt's canonical pattern `WHERE r.source_clause
IN $seed_ids` pinned every result row to a seed clause and produced
mean `n_cypher_new = 0`. That defeats the purpose.

### 4.1 Stages

```
question
  │
  ▼
[1] vector_search(question, top_k_seed)            → seed_clauses (≤ 8)
  │
  ▼
[2] LLM Cypher gen (constrained, with repair)
       schema-whitelisted
       MUST traverse from seeds OUTWARD
       MUST NOT pin r.source_clause IN $seed_ids
       MUST RETURN target.id (new candidate clause/article) + score signal
  │
  ▼
[3] execute on Neo4j (READ_ACCESS, timeout 15s)
       success criterion: ≥1 row whose clause is NOT in seed set
  │  (else → repair, up to 2 rounds)
  │
  ▼ if all Cypher rounds yield 0 new clauses → flag fallback_used=True
[4] fallback_expand = RagPipeline.expand(seed_clause_ids)
       returns the same edge expansion that vanilla graphrag uses
       (kept for honest comparison: when Cypher fails we degrade to baseline,
        not to nothing)
  │
  ▼
[5] fuse and rank → top_k_final clauses
       inputs:
       - seed_clauses with vector_score (cosine 0-1)
       - cypher_new_clauses with cypher_score (constant 1.0 if surfaced,
         OR a hop-based decay 1/(1+hop) if we add hop bookkeeping)
       - fallback_expand neighbors with expand_score (constant)
       method:
       - Reciprocal Rank Fusion (RRF, k=60) — matches v5 sprint 1 convention
       - re-fetch full Clause text for cypher_new_clauses (vector hits
         already carry text)
  │
  ▼
return CypherWalkResult(hits=top_k_final, ...)
```

### 4.2 Cypher generation prompt — what changes vs the previous attempt

The previous prompt encouraged the pin pattern `WHERE r.source_clause
IN $seed_ids` and gave 3 worked examples that all used it. The new
prompt MUST:

- Start patterns from seed clauses/articles and traverse to **other**
  clauses/articles via `:REFERENCES | :REFERS_TO | :HAS_CLAUSE |
  :HAS_POINT | :BELONGS_TO | :NEXT` AND semantic edges.
- Forbid the pin pattern explicitly. Validator (see §4.4) rejects any
  Cypher whose only WHERE clause is the seed pin without a follow-up
  outward traversal.
- Require the RETURN clause to expose `target_clause_id` (or
  `target_article_id` — which we expand to its first clause downstream)
  that is NOT in `$seed_ids`.

Few-shot examples to swap in (each shows a different outward
traversal):

```cypher
-- Q: "Khoản 1 Điều 64 viện dẫn những Điều nào của Bộ luật Lao động?"
MATCH (src:Clause)-[r:REFERS_TO|CITES_EXTERNAL]->(tgt)
WHERE src.id IN $seed_ids
RETURN tgt.id AS target_clause_id, type(r) AS relation_type,
       coalesce(tgt.title, tgt.code, '') AS target_label,
       r.span AS evidence
LIMIT 20
```

```cypher
-- Q: "Quy định nào về quỹ BHXH liên quan đến chế độ hưu trí?"
-- Strategy: from seed clauses, hop to Article-cousins via shared Section,
-- then collect new Clauses about Fund.
MATCH (seed:Clause)<-[:HAS_CLAUSE]-(a:Article)-[:IN_SECTION]->(sec:Section)
      <-[:IN_SECTION]-(cousin:Article)-[:HAS_CLAUSE]->(new:Clause)
WHERE seed.id IN $seed_ids
  AND new.id <> seed.id
  AND toLower(new.text) CONTAINS 'quỹ'
RETURN new.id AS target_clause_id, 'cousin_clause' AS relation_type,
       cousin.title AS target_label
LIMIT 20
```

### 4.3 Forbid the seed-pin escape

Validator rejects a query that contains `r.source_clause IN $seed_ids`
(or equivalent) unless the same query also has at least one MATCH
edge that doesn't have `r.` as its source. Concretely: parse the
WHERE clauses, look for the pin predicate, and reject if no other
clause/article identifier appears in RETURN beyond what we can
derive from the seeds. Implementation detail to nail down at code
time; the principle is: **a query that cannot surface a new clause is
not a graph walk**.

### 4.4 Validator — same hygiene as before plus the no-pin rule

Keep all checks from the previous validator (READ-only keyword
denylist, whitelisted node labels + edge types, `$seed_ids`
reference, `LIMIT ≤ 30`) AND add:

- RETURN must include either `target_clause_id` OR `target_article_id`.
- Reject queries where the only constraint is the pin pattern.

### 4.5 Fusion (Stage 5) details

- Use RRF with `k=60` (matches `V5RetrievalPipeline`). Each source
  produces a ranked list; final score = `Σ 1/(k + rank_i)`.
- Inputs to fusion:
  - seed list (vector_search order)
  - cypher new-clause list (Cypher row order)
  - fallback list (only if `fallback_used`) (expand order)
- Output truncated to `top_k_final` (default 12 — matches the K=12
  used by exp 07/08).

## 5. Experiment 11 (REDO) — retrieval-only audit

### 5.1 Scope

Pure retrieval audit. NO answer generation. NO BERTScore. NO citation
parsing.

Match the convention established by exp 06/07/08:

```
python -m scripts.exp11_run         # run each retrieval arm on N questions
                                    # save hits-per-question as JSON
python -m scripts.exp11_metrics     # compute recall@K, NDCG@K, MRR
python -m scripts.exp11_funnel      # stage-wise pool sizes + gold counts
```

(Or wire through `eval_core` if we add a retrieval-metric path —
that's a follow-up, not blocking. exp 08 used a custom script;
this can too.)

### 5.2 Arms — 3 retrieval-only configurations

| Arm | Pipeline | What we hold constant vs vary |
|---|---|---|
| `dense_vanilla` | `RagPipeline.vector_search(top_k=12)` | Baseline. Same as the dense channel of exp 08. |
| `dense_then_expand` | `vector_search(top_k=12)` + `RagPipeline.expand` neighbours folded into result | Vanilla graphrag's retrieval-side behaviour (what `RagPipeline.ask` feeds the LLM, but without the LLM). |
| `cypher_walk` | `CypherWalkRetriever.retrieve(top_k=12)` | The new component. |

The first two are reproductions of behaviour the codebase already has
— they exist to make `cypher_walk` directly comparable without
pulling in any v5 changes.

### 5.3 Metrics

Article-level **and** clause-level recall@K for K ∈ {5, 12, 20}, plus
NDCG@K and MRR. Stratify by gold-law:

- L41-only gold (15/50 in pilot)
- mixed L41 + other (8/50)
- no-L41 gold (27/50) — this is the hard subset; the KG has the laws
  too (L58, L45, ND143_2018, …) so retrieval can succeed if it
  surfaces them.

Plus the new-to-this-arm provenance metrics (CypherWalkResult fields):

- `cypher_used` rate across the dataset
- mean `n_cypher_new` (clauses surfaced by Cypher BEYOND seed)
- `fallback_used` rate
- mean `cypher_attempts` length

These are the diagnostic signals that say whether the component is
actually doing graph traversal vs degenerating to vector + fallback.

### 5.4 Pilot then full

Same convention as exp 08: 50-question stratified pilot first, then
full 200 if pilot signals are useful. Cost on gpt-4o-mini at 50
questions ≈ ~$0.05 — same envelope as the original (mis-built)
attempt.

### 5.5 Pre-commitment predictions

Stated up front so the result can't be rationalised:

- `cypher_used` rate (≥1 NEW clause beyond seed) — **predict ≥ 30%**.
  Lower than the 76% reported in the mis-built arm because we now
  require new clauses, not just any row.
- `n_cypher_new` mean conditional on `cypher_used=True` — **predict
  1.5–3 new clauses** per success.
- recall@12 (article-level, all strata) — **predict +0 to +0.05 vs
  `dense_vanilla`**. The KG's semantic edges are sparse; we don't
  expect a large lift, only a measurable one on the subset where
  outward edges exist.
- On the no-L41 stratum (n=27 in pilot, the hard case) — predict
  recall@12 lift **near zero** because the relevant outward edges
  (REFERS_TO into L58 / L45) are present for only some clauses.

If recall@12 lift > +0.05 across all strata, treat as suspicious and
audit before celebrating.

### 5.6 Layout of `experiments/11_graphrag_cypher/` (post-REDO)

```
experiments/11_graphrag_cypher/
├── config.yaml                  — REWRITTEN: retrieval-only, points
│                                  at retrieval scripts not eval_core run
├── README.md                    — REWRITTEN: scope is retrieval audit
├── pilot_50_stt.json            — (if we stratify-pilot like exp 08)
├── results/
│   ├── dense_vanilla/A<stt>.json    — hits + scores per question
│   ├── dense_then_expand/A<stt>.json
│   └── cypher_walk/A<stt>.json      — hits + CypherWalkResult fields
├── metrics/
│   ├── retrieval_metrics.json    — recall@K / NDCG@K / MRR per arm + stratum
│   ├── retrieval_metrics.csv
│   └── gold_citations_normalized.json
└── report/
    ├── retrieval_report.md       — stratum tables + funnel
    └── funnel_cypher_walk_K12.md — stage-wise pool sizes + gold counts
```

## 6. File-by-file implementation checklist

Once this plan is approved:

1. **`runtime/retrievers/__init__.py`** (new package). Houses
   retrieval-layer components separately from arm pipelines.
   `RagPipeline` stays in place for backward compat (it's an arm).
2. **`runtime/retrievers/cypher_walk.py`** — `CypherWalkRetriever`
   class + `CypherWalkResult`, `RetrievedClause`, `CypherAttempt`
   dataclasses. Pure retrieval. Reuses `RagPipeline.vector_search`
   and `RagPipeline.driver` for Neo4j access.
3. **`prompts/runtime/cypher_walk/cypher_gen.md`** — REWRITTEN prompt
   per §4.2 (forbid pin, encourage outward traversal, new few-shots).
4. **`scripts/exp11_run.py`** — run the 3 arms over a question list,
   save per-question JSON of hits + provenance.
5. **`scripts/exp11_metrics.py`** — compute recall@K / NDCG@K / MRR
   from saved JSON, write CSV + JSON to `metrics/`.
6. **`scripts/exp11_funnel.py`** (optional, follow exp 08 style) —
   stage-wise pool sizes and gold-count deltas.
7. **`experiments/11_graphrag_cypher/config.yaml`** — REWRITTEN
   per §5.6 (retrieval-only, no `arms` block consumed by eval_core).
8. **`experiments/11_graphrag_cypher/README.md`** — REWRITTEN: state
   that exp 11 is retrieval-only; link this plan; record the
   deprecation of the prior E2E arm.

## 7. Disposition of current `graphrag_cypher` arm

The arm shipped in commit `a10f609`:

- `runtime/graphrag_cypher.py`
- `prompts/runtime/graphrag_cypher/*.md`
- `eval_core/arms.py` entry `"graphrag_cypher"`
- `eval_core/inference.py` runner `run_graphrag_cypher`
- `experiments/11_graphrag_cypher/results/graphrag_cypher/A*.json` (50 records)
- `experiments/11_graphrag_cypher/metrics/*` + `report/academic_report.md`

It is **architecturally misplaced** and its reported metrics confound
3 unrelated changes. Proposed cleanup at code time (NOT now):

- Move `runtime/graphrag_cypher.py` to `runtime/_deprecated/graphrag_cypher.py`
  with a docstring pointing at this plan. Do not delete — preserve the
  history of the mistake.
- Remove `"graphrag_cypher"` from `ALL_ARMS` in `eval_core/arms.py`
  AND from `ARM_RUNNERS` in `eval_core/inference.py`. Failing to do
  so leaves a broken import path after the move.
- Delete `prompts/runtime/graphrag_cypher/` — the prompts encode the
  pin pattern that this plan is explicitly correcting against. Keeping
  them around is a footgun for any reader who later confuses the two
  components.
- Wipe `experiments/11_graphrag_cypher/results/graphrag_cypher/` and
  `experiments/11_graphrag_cypher/metrics/` and
  `experiments/11_graphrag_cypher/report/`. The previous numbers are
  not a "frozen baseline" — they measure the wrong thing on a wrongly
  built component and should not be preservable as a number anyone
  could cite.
- Add the deprecation note to the new `README.md` so anyone reading
  the experiment folder sees the history without having to read git
  log.

These steps are listed for completeness — they happen **after** the
new `CypherWalkRetriever` is in and exp 11 has been re-run, not
before. Don't leave the repo in an in-between state.

## 8. Out-of-scope: future Logic-LM integration

After `CypherWalkRetriever` is audited at the retrieval layer and
shows useful lift, the natural next step is to plug it into the
Logic-LM family:

- New file: `runtime/cypher_walk_logic_lm_adapter.py` — wraps
  `CypherWalkRetriever` and emits `RetrievedKnowledgeContext`
  (chunks-for-Prolog).
- New file: `runtime/logic_lm_pipelines.py` gets a 4th class
  `LogicLMGraphRAGCypherPipeline` modelled on `LogicLMGraphRAGPipeline`,
  but its `retriever` is the adapter above.
- New arm `logic_lm_graphrag_cypher` in `eval_core/arms.py`.
- A separate experiment (exp 12 or later) compares
  `logic_lm_graphrag_cypher` vs `logic_lm_graphrag` inherited from
  `01_initial_eval`, holding the Prolog generation + solver + IRAC
  render constant. Headline metrics for that experiment are the
  canonical Logic-LM ones (`prolog_first_try_solution_rate`,
  `repair_invoked_rate`, `repair_success_rate`, citation_recall on
  multi-law-aware parser, BERTScore on `plain_answer`).

This experiment is the right place to claim — or refute — that
"walking the graph improves legal reasoning". Exp 11 is a
precondition: it only establishes that the component is worth
plugging in.

## 9. What this plan does NOT change

- No edit to `runtime/rag_query.py`, `runtime/logic_lm_pipelines.py`,
  `runtime/logic_lm/`, `src/retrieval.py`, or any `eval_core` file
  beyond removing the deprecated arm name.
- No new prompt under `prompts/runtime/logic_lm/`.
- No change to the Logic-LM Prolog generation + solver flow.
- No new node label or edge type in the KG schema. The Cypher walks
  use what's already there. Sparse semantic edges are a constraint
  of the KG — that's a separate problem (see
  [`docs/known_issues_kg_build.md`](../known_issues_kg_build.md)).

## 10. Approval checklist

Before any code lands:

- [ ] Plan reviewed.
- [ ] Architecture position (§2) agreed: retriever layer, peer with
      `vector_search` / `V5RetrievalPipeline`.
- [ ] Scope of exp 11 (§5) agreed: retrieval-only, no E2E.
- [ ] Cleanup plan for the deprecated arm (§7) agreed.
- [ ] Pre-commitment predictions (§5.5) accepted — if numbers blow
      past them, audit before report.

---

*Author: Claude (Opus 4.7).
Plan dated 2026-06-01. Replaces the implicit plan that
shipped with commit `a10f609`.*
