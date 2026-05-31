# exp 13 — Semantic-grounded HyDE (concept-anchored, no dense seed)

> **Status**: planned, not implemented. Branch `exp/13-hyde-semantic`.
> Retrieval-only audit on the **tuned stack** (BGE-M3 LoRA +
> `clause_vec_tuned`), **in-corpus scoped** — built like
> [exp 09](../../experiments/09_hyde2_grounded/README.md), not exp 11/12.
>
> One-line question: *does grounding the HyDE hypothesis on the KG's
> **concept layer** (query→concept frame, no dense clause seed) beat
> plain HyDE1 — where grounding on **clause text from a dense seed**
> (exp 09 / HyDE2) regressed?*

---

## 0. Target & scope decision (2026-06-01) — read first

The original targets proposed for this experiment — **recall@5 > 0.60**
and **precision@2 > 0.80** — were reality-checked against the dataset
and current index. Findings (measured, not estimated):

- **`precision@2 > 0.80` is mathematically impossible on this gold set.**
  97/200 questions (48.5%) have exactly **1** gold article → for them
  `precision@2 ≤ 0.5` by definition. The perfect-oracle ceiling for
  mean precision@2 is **0.7575** (full 200) / 0.78 (pilot-50). 0.80 is
  *above the ceiling* — no retriever can reach it. (Measured now: 0.05.)
- **`recall@5 > 0.60` is blocked by corpus coverage.** The index holds
  only 4 laws (`L41_2024`, `L58_2014`, `L45_2019`, `ND143_2018`); the
  gold spans 28 laws. **38.1% of all gold-article mentions are in laws
  that never appear in retrieval** (`LVL_2025`=66, `LVL_2013`=50, +22
  decrees/circulars) → unreachable at any K. Total gold recall is
  capped ≈ **0.62** even with K→∞ and a perfect retriever; recall@5 is
  far tighter. (Measured recall@5 now, best vanilla arm: 0.19.)

**Decision (chosen by user):** *re-scope + change metrics, no new data.*

1. **Scope to the in-corpus stratum** — score on questions whose gold ⊆
   the 4 indexed laws (the honest-denominator framing exp 08/09 already
   use via `categorize(...)` + `in_corpus_codes`). Mixed / ooc /
   unparseable reported separately, never folded into the headline.
2. **Drop `precision@2`** as a target (cardinality-capped). Headline
   precision = **precision@1** (ceiling 1.0) + **R-Precision**.
   `precision@2` may still be *reported* with its ceiling annotation.
3. **recall@5 → recall@{10,12}** as the headline recall K.
4. **Move exp 13 onto the tuned stack** (BGE-M3 LoRA + `clause_vec_tuned`
   via `V5RetrievalPipeline`), where the in-corpus baselines are
   strongest — not the vanilla `clause_vec` of exp 11/12.

The aspirational `recall@12 ≈ 0.6` / `R-Prec ≈ 0.3` survive as a
**labelled north-star** (§9), explicitly above the known HyDE1 baseline
and reachable only with extra levers (full rerank pipeline and/or corpus
expansion) that are out of scope here.

---

## 1. Where this sits in the HyDE lineage

| exp | grounding of the hypothetical doc | stack | result (in_corpus) |
|---|---|---|---|
| 08 `dense_hyde` | **none** (zero-shot HyDE1 from the question) | tuned | **WINS** R@12 0.38→0.47 |
| 09 `dense_hyde2` | **clause TEXT** from a top-5 **dense** seed | tuned | **LOSES** vs HyDE1 (R@12 0.42; S1 regression) |
| **13 `dense_hyde_semantic`** ← this | **concept FRAME** from `match_query_concept_ids` + ontology — **no dense seed** | tuned | TBD |

exp 09's loss was **not** clause-text verbosity. The documented root
cause (exp 09 README, stt=56) is that the **pass-1 dense seed on the
raw question landed in the wrong domain cluster** and grounding
*amplified* that error. This experiment removes the dense seed entirely
and anchors on canonical BHXH concepts instead.

## 2. Hypothesis & the bar to beat

**Hypothesis.** A hypothesis grounded on a *concept frame* writes in
canonical BHXH vocabulary aligned to the question's actual topic,
embedding nearer the correct article cluster than a raw-question dense
seed could — recovering exp 09's intended benefit without its failure
mode.

**The bar is HyDE1 on the tuned in_corpus stratum**, i.e. the real
measured numbers (exp 09, n=151):

| metric (in_corpus) | dense (raw) | **dense_hyde (HyDE1) = the bar** | dense_hyde2 (lost) |
|---|---:|---:|---:|
| recall@12 | 0.3832 | **0.4736** | 0.4210 |
| recall@30 | — | 0.6066 | 0.5203 |
| R-Precision | 0.0635 | **0.1326** | 0.1019 |
| NDCG@12 | — | 0.2944 | 0.2437 |
| MRR | — | 0.2843 | 0.2192 |

exp 09 *failed* its no-regression-vs-HyDE1 gate. exp 13's primary
success criterion is to **pass that gate** (§9).

## 3. Architectural position

`dense_hyde_semantic` is a **retrieval-layer** arm on the tuned stack,
a peer of HyDE1, scored **in-corpus**. It produces a ranked
article set — **no LLM render, no citation parsing, no answer
generation** (same scope as exp 08/09). The "top-K (K=5) context for the
next flow" the user described is the downstream answer stage, out of
scope here.

Built like exp 09: three `V5RetrievalPipeline` instances on
`clause_vec_tuned`; metrics via the exp 09 `categorize` + in_corpus
machinery. Absolute numbers are comparable to exp 08/09 (same tuned
stack) — **not** to exp 11/12 (vanilla `clause_vec`).

## 4. Pipeline design (retrieval-only)

```
question
  │
  ▼
[1] get_semantic_relevance(question)                         → SemanticContext
      a. concepts   = match_query_concept_ids(question)      (query→concept ids)
      b. frame      = render concepts + parents/children     (the CONCEPT FRAME)
                      from concept_specs_by_id()
      c. (optional) anchored = OntologyRetrieval.retrieve(question, k=N)
                      → top-N concept-/keyword-scored snippets
      d. context_key_ids = sorted(concept_ids) + sorted(anchored chunk ids)
  │   NO dense clause seed anywhere in this step.
  ▼
[2] semantic HyDE generation (cache-aware)
      OpenAISemanticHydeGenerator.generate(question, frame[, anchored])
      gpt-4o-mini, n=1, max_tokens=700, T=0
      prompt = prompts/runtime/hyde_generate_semantic.md
      cache  = artifacts/hyde_semantic/  (key includes context_key_ids hash)
  │   (fallback) concepts == ∅ AND anchored == ∅ → plain HyDE1 doc, flag fallback_used
  ▼
[3] BGE-M3 **LoRA** encode the hypothetical doc → query vector
  ▼
[4] dense top-k over **clause_vec_tuned** (V5RetrievalPipeline)
      via new V5 method retrieve_dense_only_hyde_semantic
  ▼
[5] article-dedupe → ranked article ids (report @5/@10/@12, in_corpus)
```

### 4.1 What goes in `{context}` — the concept frame

Default context = **concept frame + a few concept-anchored snippets**:

```
KHUNG KHÁI NIỆM (chủ đề pháp lý liên quan đến tình huống):
- Lương hưu  ⊂ Bảo hiểm xã hội
    • Tỷ lệ lương hưu, Tuổi nghỉ hưu, Nghỉ hưu trước tuổi, Điều chỉnh lương hưu
- Đóng BHXH  ⊂ Bảo hiểm xã hội
    • Tiền lương tháng đóng BHXH, Bảo lưu thời gian đóng

DỮ LIỆU NỀN (trích, để dùng đúng thuật ngữ — KHÔNG sao chép số hiệu Điều/Khoản):
[1] <snippet from OntologyRetrieval top chunk>
[2] ...
```

- The frame (labels + `is_a` hierarchy + sibling concepts) is the
  **pure semantic-relation signal**, present whenever ≥1 concept matches.
- Snippets give lexical grounding via concept-/keyword-selection
  (`OntologyRetrieval`), **not** dense-seed selection → no exp 09 bias.
- **Ablation toggle** (run only if the blended arm is inconclusive):
  `frame-only` (drop snippets). Not part of the headline run.

### 4.2 Generator prompt — `prompts/runtime/hyde_generate_semantic.md`

Inherits all HyDE1 hard constraints (no `Điều X`/`Khoản Y`/`Điểm`, no
fake citations, no proper nouns/numbers from the question, no Q&A
framing). Adds a `{context}` block (the concept frame + optional
snippets) and: *"Viết đoạn văn bản pháp luật giả định **bám sát các
khái niệm và quan hệ trong KHUNG KHÁI NIỆM**; dùng đúng thuật ngữ BHXH
chuẩn của các khái niệm đó."* Uses `{context}`+`{question}` so it reuses
the `OpenAIGroundedHydeGenerator` machinery unchanged.

### 4.3 Cache + determinism

New namespace `artifacts/hyde_semantic/` (never collides with
`artifacts/hyde` / `hyde2`). Cache key = `sha256(question | prompt_sha |
n | model | max_tokens | temperature | hash(context_key_ids))`. Concept
matching + `OntologyRetrieval` are deterministic → stable key →
idempotent, $0 on re-run.

## 5. Arms — 3 retrieval-only configurations (in-corpus scored)

| Arm | Pipeline (tuned stack) | Source of records |
|---|---|---|
| `dense` | `V5RetrievalPipeline.retrieve_dense_only` (raw) | **inherit exp 08/09** (skip-when-on-disk; $0) |
| `dense_hyde` | `retrieve_dense_only_hyde` (HyDE1) | **inherit exp 08/09** (HyDE1 cache; $0) — *the bar* |
| `dense_hyde_semantic` | §4 pipeline (new V5 method) | **new** (this experiment) |

Only the grounding of the HyDE doc differs between `dense_hyde` and
`dense_hyde_semantic` → clean single-variable comparison on an
identical stack + pilot.

*(Optional stretch arm, §9):* `full_rerank_semantic_hyde` =
semantic-HyDE dense channel inside the full V5 pipeline
(dense+sparse+RRF+CE-rerank+expand). The rerank pipeline is the lever
that could push recall@12 toward the north-star band; added only if the
dense arm lands close.

## 6. Context builder design

Same as §4.1. `build_semantic_context(question)` is pure + deterministic;
no LLM, no dense seed. Fallback to HyDE1 when no concept matches and
`OntologyRetrieval` returns nothing (flagged in provenance).

## 7. Metrics & stratification

Headline computed on the **in_corpus** stratum (gold ⊆ indexed laws),
reusing exp 09's `categorize` + `in_corpus_codes = {m.full_id for m in
load_law_metadata()}`. Strata `in_corpus / mixed / ooc / unparseable`
reported separately.

Metrics: **recall@{5,10,12}** (headline recall = @12; @5 reported for
continuity with the original ask), **precision@1**, **R-Precision**,
NDCG@12, MRR. `precision@2` reported **with its 0.76 ceiling
annotation** so the cap is visible, not as a target.

Provenance diagnostics: `concept_match_rate`, mean `n_concepts`,
`fallback_used` rate, per-stratum `concept_match_rate`.

## 8. Pre-commitment predictions (stated BEFORE the run)

vs HyDE1 on **in_corpus** (the bar from §2):

1. **Sanity gate S1 (no regression vs HyDE1):** semantic recall@12 ≥
   HyDE1 recall@12 − 0.01 (= ≥ 0.4636). *This is the gate exp 09
   failed.* If S1 fails → second negative; report it, don't rationalise.
2. **Sanity gate S2 (beats raw dense):** semantic recall@12 ≥ dense
   (raw) + 0.03 (= ≥ 0.4132).
3. **Headline Δ:** semantic recall@12 − HyDE1 recall@12 ∈ **[−0.01,
   +0.04]**. HyDE1 already encodes strong BHXH priors, so the concept
   frame's marginal lift is bounded. **Δ > +0.05 → audit before
   celebrating** (don't trust a too-good result).
4. `concept_match_rate` (in_corpus) **≥ 0.85**; `fallback_used` ≤ 0.15.

**Decision rule (primary):** win = S1 ∧ S2 ∧ (headline Δ ≥ +0.02).

## 9. North-star vs pre-registered bar (honesty note)

The user's aspiration is `recall@12 ≈ 0.55–0.60` and `R-Precision ≈
0.30` in_corpus. Stated plainly: **these are above the strong HyDE1
baseline** (0.47 / 0.13). A HyDE-grounding change alone has never moved
recall@12 by the +0.10 needed (exp 08's HyDE1 lift was +0.09 over *raw*,
and grounding in exp 09 went *backwards*). So:

- **Pre-registered success = §8** (beat HyDE1). That is the claim this
  experiment can honestly make or refute.
- **North-star = the 0.55–0.60 / 0.30 band**, reachable only with extra
  levers, each its own experiment:
  - the **full V5 rerank+expand pipeline** (optional stretch arm §5);
  - **corpus expansion** (ingest `LVL_2025/2013` + missing decrees via
    the `add-legal-document` B1–B6 pipeline) — the only lever that lifts
    the full-dataset recall ceiling above ~0.62.

Do not report a north-star "miss" as an experiment failure: the
experiment tests S1–S3, the north-star tracks the roadmap.

## 10. Coverage / feasibility caveats

- The in_corpus scope makes the *denominator* honest but does **not**
  change that 38% of full-dataset gold is un-indexed (§0). Any
  full-dataset number remains coverage-capped ≈ 0.62.
- The concept ontology is **L41-only** (`corpus_2024.jsonl`); on
  in_corpus questions whose gold is L58/L45/ND143 the frame may match
  only generic concepts. Mitigation is structural: step [4] searches the
  full `clause_vec_tuned` (all 4 indexed laws). Measured per-stratum, not
  assumed. No ontology rebuild in this experiment.

## 11. File-by-file implementation checklist

Once approved (modelled on exp 09, not exp 11/12):

1. **`runtime/retrievers/semantic_context.py`** (new) — pure
   `build_semantic_context(question, ontology_retrieval=None,
   top_n_snippets=N) -> SemanticContext{concept_ids, frame_text,
   snippet_ids, snippet_text, context_key_ids}`. Wraps
   `match_query_concept_ids`, `concept_specs_by_id`, optional
   `OntologyRetrieval`. No dense seed.
2. **`prompts/runtime/hyde_generate_semantic.md`** (new) — §4.2 prompt.
3. **`src/retrieval/hyde_semantic.py`** (new) —
   `OpenAISemanticHydeGenerator(OpenAIGroundedHydeGenerator)`: default
   `prompt_path=runtime/hyde_generate_semantic.md`,
   `cache_dir=artifacts/hyde_semantic`; `generate(question, frame_text,
   context_key_ids)` maps `frame_text→{context}` + `context_key_ids`
   into the seed-hash slot. Reuses cache/async/cost machinery.
4. **`src/retrieval/pipeline.py`** — add `retrieve_dense_only_hyde_semantic`
   (peer of `retrieve_dense_only_hyde2`): build context → generate doc →
   LoRA-embed → dense top-k over `clause_vec_tuned`. Construct a
   `pipe_hyde_semantic` instance (analogous to exp 09's `pipe_hyde2`).
5. **`scripts/exp13_run.py`** — 3 arms over the exp 08/09 pilot-50
   (`--pilot-50` / full), inherit `dense` + `dense_hyde` from exp 08/09
   via skip-when-on-disk, run `dense_hyde_semantic`. Prewarm semantic-HyDE
   docs batch. Records → `experiments/13_hyde_semantic/results/<arm>/A<stt>.json`.
6. **`scripts/exp13_metrics.py`** — fork exp 09 metrics: `in_corpus`
   stratum headline; recall@{5,10,12}, precision@1, R-Precision, NDCG@12,
   MRR; precision@2-with-ceiling note; provenance (§7); §8 pass/AUDIT table.
7. **`experiments/13_hyde_semantic/{config.yaml,README.md}`** —
   retrieval-only (`arms: {}`); README records §0 decision, §8
   predictions, §9 north-star, links exp 09 (failed precedent) + exp 08
   (HyDE win).
8. **`experiments/13_hyde_semantic/pilot_50_stt.json`** — copy exp 08/09's
   list so strata are identical (HyDE1-vs-semantic on the same questions).

Disk policy: `results/` gitignored; `metrics/` + `report/` tracked. No
output outside the experiment folder. No `data/` writes.

## 12. Cost

| | value |
|---|---:|
| `dense` (inherit exp 08/09, local) | $0 |
| `dense_hyde` (HyDE1 cache) | $0 |
| `dense_hyde_semantic` LLM (pilot-50 cold, gpt-4o-mini, 1 call/q) | ~$0.03 |
| **Pilot-50 total (cold)** | **~$0.03** |
| Re-run | $0 (cache) |
| Full-200 (only if pilot signal is useful) | ~$0.12 cold |

Cost cap enforced in `exp13_run.py` (default $0.50; abort before spend if
estimate exceeds cap), same as exp 08/09.

## 13. What this plan does NOT change

- No edit to `runtime/rag_query.py`, `runtime/logic_lm/`, or `eval_core`.
- No KG schema change; no ontology rebuild; no corpus expansion (that is
  the separate north-star lever, §9).
- No Cypher walk (exp 12 concluded it hurts — unused here).
- No end-to-end arm, no answer generation, no judge metrics.
- No in-place edit of existing `prompts/` files; the semantic prompt is new.
- `precision@2` metric definition is **not** changed — it is dropped as a
  *target* and annotated with its ceiling (changing the metric to flatter
  it would violate the metric-discipline rule).

## 14. Approval checklist

- [ ] §0 decision agreed: in-corpus scope + tuned stack + precision@1 /
      R-Precision + recall@{10,12}; original 0.6/0.8 retired (0.8
      impossible, 0.6 coverage-blocked).
- [ ] §2 bar agreed: beat HyDE1 (tuned in_corpus 0.4736 R@12 / 0.1326 R-Prec).
- [ ] §8 pre-commitment predictions accepted (S1 gate is the headline).
- [ ] §9 north-star understood as roadmap, not this experiment's pass/fail.

---

*Author: Claude (Opus 4.8). Plan dated 2026-06-01. Branch
`exp/13-hyde-semantic`. Descends from exp 09 (negative, clause-text
grounding) + exp 08 (HyDE1 win). Target re-scoped per the 2026-06-01
feasibility analysis (§0).*
