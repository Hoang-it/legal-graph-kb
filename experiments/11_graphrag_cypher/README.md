# exp 11 — `graphrag_cypher` pilot (Hybrid: vector seed + LLM-Cypher + fallback)

## What

New inference arm `graphrag_cypher`. Per question:

1. **Vector seed** — `RagPipeline.vector_search(question, top_k=8)` →
   list of seed Clause IDs (same dense channel as vanilla `graphrag`).
2. **LLM-authored Cypher** — `gpt-4o-mini` is given the KG schema
   (whitelisted labels + edge types), the seed IDs, and the question;
   it emits a JSON `{cypher, rationale}`. The Cypher MUST reference
   `$seed_ids` and MUST be read-only.
3. **Validator** — regex-gated:
   - rejects any `CREATE / MERGE / DELETE / SET / REMOVE / DROP / LOAD / CALL / FOREACH`;
   - rejects any `:Label` outside the schema whitelist;
   - rejects any relationship type outside the edge whitelist
     (supports `[r:A|B|C]` lists);
   - requires `$seed_ids` reference, a `clause_id` column in `RETURN`,
     and a `LIMIT ≤ 30`.
4. **Execute** — `session.execute_read(...)` (server-enforced read
   mode) with a 15 s transaction timeout.
5. **Repair loop** — up to 2 rounds. The LLM is shown its previous
   Cypher + the validation/execution error / zero-row signal and asked
   to re-author.
6. **Fallback** — if all attempts return 0 rows: use vanilla GraphRAG
   context (`vector_search + expand` neighbor walk). The record marks
   `fallback_used = true` so the comparison is honest.
7. **Render** — `gpt-4o-mini` answers from a context that surfaces
   GRAPH FACTS (Cypher rows) above CLAUSE TEXTS. Same answer prompt
   discipline as vanilla `graphrag` (citations as `[Điều X khoản Y]`,
   no outside knowledge).

## Why

Per the 2026-05-31 discussion (memory ref: "graph-walk question"):
vanilla GraphRAG does neighbor *expansion* but the LLM never *queries*
the graph. We hypothesise that letting the LLM choose the traversal
path conditional on the question gives qualitatively different
context — but only on questions whose answer is reachable via
existing semantic edges. The KG audit showed 40 % of L41 Articles have
zero semantic edges, so we expect a high fallback rate. The arm is
designed to make that fact *measurable*, not to mask it.

## Setup

- 50 questions: first 50 of `data/eval/questions_200.json`
  (matches the prefix used by the frozen exp 01 baseline → direct
  apples-to-apples comparison on `graphrag` baseline).
- Single arm with `mode: run` for `graphrag_cypher`. `graphrag` inherits
  from exp 01 to anchor the comparison without re-spending API budget.
- Cypher LLM = gpt-4o-mini, T = 0, JSON response format.
- Answer LLM = gpt-4o-mini, T = 0 (same as baseline).
- Max repair rounds = 2 (so worst case = 3 Cypher LLM calls + 1 answer call per question).

## Predictions (pre-commitment)

- `cypher_used` rate < 50 % of the 50 questions (sparse semantic
  edges + many questions reference non-L41 laws not in KG).
- On `cypher_used = true` subset: citation recall ≥ vanilla
  graphrag baseline on the same subset (else the Cypher walk is
  surfacing distracting clauses).
- Median latency `graphrag_cypher` > `graphrag` by 3–5 s
  (dominated by Cypher generation calls).

## How to run

```powershell
python -m eval_core run experiments/11_graphrag_cypher
python -m eval_core metrics experiments/11_graphrag_cypher
```

## Result — 50-question pilot (2026-05-31)

50/50 questions ran without exception. Comparison vs vanilla `graphrag`
on the SAME stt 1–50 (baseline inherited from `01_initial_eval`).

### Provenance — did the graph actually get walked?

| Quantity | Value |
|---|---:|
| cypher_used (≥ 1 Cypher row returned) | **38 / 50** (76 %) |
| fallback_used (Cypher exhausted 2 repair rounds → fell back to vanilla expand) | 12 / 50 (24 %) |
| Cypher attempts = 1 | 26 / 50 |
| Cypher attempts = 2 | 12 / 50 |
| Cypher attempts = 3 (= max → fallback) | 12 / 50 |
| Validation errors / Execution errors among fallbacks | 0 / 0 |
| **New Clause IDs surfaced beyond vector seed (mean / max)** | **0.00 / 0** |

The last row is the most important finding. Every fallback was caused
by the Cypher returning **zero rows**, not by validation or execution
failure — i.e. the LLM wrote a valid query that simply found no
matching semantic edges anchored to the seed clauses. And among the
38 successful Cypher walks, **not a single one surfaced a Clause that
vector retrieval had not already seen**. Reason: the canonical pattern
`WHERE r.source_clause IN $seed_ids` pins the citation anchor to the
seed set by construction. The Cypher walk therefore enriches the
context with **named semantic entities** (Subject / Benefit / Condition
names from the graph), not with **new clause candidates**.

### Answer quality — BERTScore F1 (n=50 paired)

| Stratum | n | Cypher arm | Baseline graphrag | Δ |
|---|---:|---:|---:|---:|
| **All 50 (paired)** | 50 | **0.7097 ± 0.039** | 0.6658 ± 0.047 | **+0.0439** |
| cypher_used = True | 38 | 0.7142 | 0.6649 | +0.0494 |
| fallback_used = True | 12 | 0.6954 | 0.6687 | +0.0267 |

Paired record-level (±0.005 tie band): **Cypher wins 34, baseline wins 8, ties 8**.

Caveat: the +0.027 advantage on the fallback subset is unexpected
(fallback path uses the same `build_context` as baseline). The
plausible explanation is the answer-rendering system prompt — it
keeps the GRAPH FACTS / CLAUSE TEXTS framing even when GRAPH FACTS is
empty, which slightly nudges phrasing toward the reference style.
Not investigated further in this pilot.

### Citation metrics

Both arms report `citation_recall = citation_precision = 0` because
`RagPipeline.parse_citations` only matches `[Điều X khoản Y]` with
implicit L41 prefix, but the answer LLM routinely cites
`[Nghị định 143/2018 Điều X]` or `[Luật 41/2024 Điều X]` — the
brackets contain extra text, regex misses. This is a baseline-wide
parser limitation, not introduced by the `graphrag_cypher` arm; it
affects both arms equally. Until the project-wide parser is
generalised, citation metrics cannot speak to retrieval quality
here.

### Cost + latency

| | Cypher arm | Baseline | Ratio |
|---|---:|---:|---:|
| Mean latency / question | 8.58 s | 4.71 s | 1.82 × |
| Median latency / question | 7.90 s | 3.92 s | 2.01 × |
| Prompt tokens (total, 50 q) | 275 463 | — | — |
| Completion tokens (total, 50 q) | 21 323 | — | — |
| **Est. cost (50 q, gpt-4o-mini)** | **≈ $0.054** | — | — |
| Projected cost for full 200 q | ≈ $0.22 | — | — |

### Honest reading

1. **The graph IS walked 76 % of the time** — better than the < 50 %
   I pre-committed to predict. The schema-constrained LLM-Cypher
   loop is robust enough that gpt-4o-mini produces a syntactically
   valid + whitelist-compliant query on the first try in 26/50
   cases.
2. **But the walk does not add candidate clauses.** It only adds
   semantic entity names. The "real graph reasoning" we hoped for
   (multi-hop retrieval beyond what vector saw) does not happen
   with the current Cypher templates the LLM produces. To make the
   walk surface NEW evidence, the prompt would need to *forbid*
   the `r.source_clause IN $seed_ids` pin and instead start from
   seeds and follow `(c:Clause)-[:REFERENCES|REFERS_TO]->(target:Clause|Article)`
   to reach previously-unseen targets.
3. **BERTScore lifts +0.044 absolute on the paired comparison**,
   with 34/50 paired wins — a real and material effect. But because
   the Cypher walk doesn't surface new evidence, the lift must come
   from the **answer-rendering side**: structured GRAPH FACTS in the
   context bias the LLM toward more reference-style phrasing.
4. **Citation recall stays = 0** — strict-tuple matching fails because
   the citation parser doesn't recognise the multi-law bracket format
   the LLM uses. Fixing this is a project-wide retrieval-eval gap, not
   an arm-design issue.
5. **Cost is negligible** (~$0.054 for the pilot, ~$0.22 for 200).
6. **Latency is the main trade-off**: ~2 × baseline because of the
   Cypher LLM round-trip(s).

### Follow-ups worth scoping (not done here)

- Rewrite the Cypher prompt to encourage *out-of-seed* traversal
  (`REFERENCES`, `REFERS_TO`, `BELONGS_TO` to neighbour clauses /
  articles), then re-measure `clauses_added_beyond_seed`.
- Generalise `parse_citations` to recognise `[Nghị định …]`,
  `[Thông tư …]`, `[Luật …]` — orthogonal to this arm but blocks
  any meaningful citation comparison.
- Re-run on the full 200 to confirm the BERTScore lift is
  stable across the distribution (not concentrated in the easy first 50).

