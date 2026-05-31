# Plan v5 — General, scalable legal citation retrieval

> **Status: design under review, post-grilling rev 2. Not committed; updated locally
> until Sprint 1 (vanilla pipeline + audit) lands.**
> Supersedes: `plan_phase6_completion.md`, `plan_phase6_prolog_multilaw.md`, the
> entire `logic_lm_graphrag_logic_rt_v*` arm family (v1 → v4).
> Owner: Hoàng — UIT MSc thesis.

## 1. Why v4 was discarded

v4 (`logic_lm_graphrag_logic_rt_v4`) was a workaround layer. Its core mechanism —
`data/legal_issue_taxonomy.yaml` + `src/legal_issue_planner.py` + per-domain
slot dependency map — encoded a hand-curated rule set per legal domain. That
blocks the only scaling axis that matters for the thesis: number of laws and
number of questions.

Evidence from the 50-question phase6 cut:

| Bucket (in-corpus gold = 31) | Count | Root cause |
|---|---:|---|
| Article-level hit | 13 | — |
| Predicted but wrong article | 11 | retrieval miss for 9/11; LLM pick lệch for 2/11 |
| Blocked by `evidence_gap` (source_family hard gate) | 3 | taxonomy filter too narrow |
| Blocked by `citation_validation_failed` | 6 | validator over-strict |
| Other (OOC-declared false, unknown gold) | 2 | — |

73% of all failing in-corpus cases have the gold article missing from the
retrieval context the LLM saw. Validator/planner over-strictness only inflated
the loss; the underlying retriever is the bottleneck.

## 2. Problem decomposition

Citation retrieval for Vietnamese legal QA is **three problems stacked**:

1. **Domain-adapted dense retrieval.** BGE-M3 vanilla never saw Vietnamese
   statutory text during pretraining; lexical/syntactic distance between user
   phrasing and statutory phrasing is large.
2. **Multi-evidence aggregation for legal reasoning chains.** Median gold
   citations per question = 2–4. Reasoning is a chain (applicability →
   eligibility → quantity → procedure) and frequently crosses laws.
3. **Temporal / version-aware filtering.** Event date in the question selects
   which law version is in force. This is a property of node metadata, not a
   rule.

## 3. Design principles

1. **No YAML / Python file may contain patterns keyed by law name or legal
   domain.** Adding Luật Đất đai = re-run indexing, zero code change.
2. **Every ranking / filter decision is driven by signal already in the
   data** (embedding score, BM25 score, `effective_from/until`, citation
   edges). No handwritten rules.
3. **Retrieval recall improves through representation learning** when
   needed, not through widening thresholds or hardcoded include lists.
4. **Each module is measurable in isolation**: bottlenecks must surface from
   metrics, not from inspection.
5. **The citation graph is first-class structure.** Cross-law dependencies
   emerge from `REFERS_TO` edges.
6. **Build vanilla first; add modules only when audit data justifies them.**
   Defer fine-tune, defer verifier, defer query decomposition. Add when an
   empirical signal — not a hunch — calls for them.

## 4. Architecture — sprint-based, conditional

### Sprint 1 — Vanilla pipeline (always)

```
M1 Indexing (offline)
  ├─ Dense vectors: BGE-M3 dense (vanilla)
  ├─ Sparse vectors: BGE-M3 sparse OR Lucene BM25
  ├─ Metadata: law_code, effective_from/until, parent_article_id
  └─ Citation graph: REFERS_TO / CITES_EXTERNAL as first-class edges
                       ↓
M4 Hybrid retrieval (online)
  ├─ raw query → dense top-30 + sparse top-30
  ├─ Temporal filter native in Cypher (effective_from <= event_date <
  │   effective_until)
  └─ RRF fusion → top-50
                       ↓
M5 Rerank + graph expansion
  ├─ Cross-encoder rerank top-50 → top-15
  ├─ For each of top-15, follow REFERS_TO 2-3 hops in Neo4j
  ├─ Add referenced articles to pool, dedupe, re-rerank
  └─ Final top-12 → into LLM context
                       ↓
Generator
  ├─ GPT-4o-mini, strict prompt (no invention, cite only from context)
  └─ Output: answer + citation_ids
```

End of Sprint 1: `citation_recall` + `citation_precision` measured on 30-case
probe and full 150-test.

### Sprint 2 — Conditional additions

Decisions in Sprint 2 are driven entirely by Sprint 1 audit. Possible
additions (zero, one, or several):

| Trigger | Add | Cost |
|---|---|---|
| Probe shows gold systematically missing from top-12 even with 2-3 hop expansion | **M2 — Fine-tune BGE-M3** (LoRA on synthetic Q–clause pairs, then re-eval) | ~$10–20 GPU, ~$5 synthetic data, ~1 week |
| Multi-aspect queries (A1, A47-style) fail systematically | **M3 — HyDE** (1 LLM call sinh hypothetical answer → embed → retrieve, fused) | ~$0.001/query, ~3-5s latency, 2 days build |
| Precision e2e < 80% (tangential cites) | **M6 — Verifier** (entailment check, drop low-confidence cites) | ~$0.002/query, +1s latency, 3 days build |
| Reasoning-heavy cases (A30-style) > expected 5–8% | **M5+ hops** OR document as limitation | depends |

### Sprint 3 — Final evaluation + paper

Full 150-test run on all baselines (v4 archived + Sprint 2 finalized v5)
with adaptive granularity policy. Stratified analysis: in-corpus vs OOC vs
reasoning-heavy.

## 5. Metric framework

### Primary metric: `citation_recall` and `citation_precision` end-to-end

Defined in `eval_core/metrics.py:compute_citation_metrics()` (the module
was renamed from `evaluation.compute_academic_metrics` during the
2026-05 refactor; the contract is unchanged).

**Granularity policy: STRICT TUPLE-EQUAL** (decision: 2026-05-31, owner).

A predicted citation matches a gold citation **iff** the full tuple
`(law_id, article_n, clause_n, point_letter)` matches exactly — no
component may differ, be missing, or be over-specified. A wrong khoản
might not exist in the law; a missing khoản leaves the reader unable
to locate the rule. Strict matching aligns the metric with how a
lawyer reads a citation.

| Gold | Arm cite | Verdict | Why |
|---|---|---|---|
| `L58_2014, Điều 2` | `L58_2014, Điều 2` | **HIT** | exact |
| `L58_2014, Điều 2` | `L58_2014, Điều 2 khoản 1` | **MISS** | over-specified (arm may have hallucinated a khoản) |
| `L58_2014, Điều 2 khoản 1` | `L58_2014, Điều 2 khoản 1` | **HIT** | exact |
| `L58_2014, Điều 2 khoản 1` | `L58_2014, Điều 2` | **MISS** | missing khoản |
| `L58_2014, Điều 2 khoản 1` | `L58_2014, Điều 2 khoản 2` | **MISS** | wrong khoản |
| `L58_2014, Điều 2 khoản 1` | `L41_2024, Điều 2 khoản 1` | **MISS** | wrong law |

**Scope distinction — strict applies ONLY to E2E citation metrics**:

- **E2E citation metrics** (LLM emits citation string → parsed → tuple)
  use strict tuple-equal. This is the **primary metric** for thesis
  acceptance gates §10.
- **Retrieval-only experiments** (exp 06 / 07 / 08, no LLM) continue to
  use **article-deduped diagnostic**. They answer "did dense surface the
  right Điều at all?" — bottleneck analysis before rerank + LLM commit
  to a specific khoản. The pipeline is also allowed to fetch sibling
  clauses of a hit article to widen LLM context (this is a feature,
  not a metric leak: extra clauses become context for the LLM, but
  the LLM still has to emit a strict-correct citation to score).
- **`law_id` MUST match at every layer** (both retrieval-only diagnostic
  and E2E primary). Wrong law is always a miss — non-negotiable, since
  citing the wrong statute is the most basic form of legal
  misinformation.

**Implementation plan** (deferred; task #21 — not coded yet):

1. **`eval_core/gold.py:validate_gold_citations`** — keep `clause_n`
   and `point_letter` on the normalized gold records. Currently the
   normalizer rolls everything up into `gold_articles: list[str]` (e.g.
   `["L58_2014.A2"]`) and drops khoản/điểm. After the change, emit
   `gold_citations_normalized: list[dict]` with full tuple per citation
   AND keep `gold_articles` as a derived field for backward compat with
   retrieval-only audits.
2. **`eval_core/metrics.py:compute_citation_metrics`** — comparison key
   becomes the full 4-tuple. Recall denominator = |gold_tuples|;
   precision denominator = |predicted_tuples|.
3. **`src/citations.py:parse_displayed_citations`** — verify it already
   emits the 4-tuple from canonical citation format
   (`"Điều X khoản Y điểm z"`). It should — the parser predates this
   policy and was always granularity-aware.
4. **`tests/test_academic_metrics.py`** — add fixtures covering the 6
   example rows above. Each fixture asserts the per-citation match
   verdict so a future refactor cannot silently regress the policy.
5. **Re-run `python -m eval_core metrics experiments/01_initial_eval`**
   to re-aggregate the frozen v4 baseline under the new policy. v4
   records (`results/*.json`) are immutable; only the metrics
   aggregation (`metrics/academic_metrics.json` + report) gets
   rewritten. This is the **A/B blocker**: any v5-vs-v4 comparison
   after this date MUST cite the post-rewrite v4 numbers.
6. **Document** in `docs/changelog.md`: "2026-05-31 — citation metric
   switched to strict tuple-equal policy; v4 baselines re-aggregated
   in same commit. All prior published v4 numbers are deprecated."
7. **Retrieval-only metric scripts (`scripts/exp{06,07,08}_metrics.py`)
   stay article-deduped** — explicit comment that they are diagnostic,
   not the primary thesis metric.

**Expected baseline shift**: v4 article-only recall will drop when
re-aggregated, because strict-tuple is uniformly stricter than the old
implicit policy. The §10 acceptance tier numbers (70/80, 85/90, 95/95)
are interpretive targets — they may need re-calibration based on what
the strict-tuple v4 baseline actually reads. Do NOT change the
acceptance tiers preemptively; first see the re-aggregated v4 number,
then decide if a tier shift is warranted (and document the rationale).

### No `retrieval_recall@K` as standalone study

`retrieval_recall@K` does not directly improve `citation_recall` — they live
at different pipeline layers. The relationship is an upper-bound diagnostic,
which can be derived post-hoc from any pipeline run's retrieval log.
Phase 0 as a separate study is dropped.

If/when a retrieval-vs-generator bottleneck analysis is needed (e.g.,
during Sprint 1 audit), it is computed per-case from the existing
`retrieval` field in result records, not as a separate experiment.

### Test set: 150 / 50 stratified split

- 200 questions in `data/eval/questions_200.json`
- Split 150 test / 50 dev, stratified by gold corpus type (in / out / mixed)
- Dev: threshold calibration (verifier, if added), early stopping
  (fine-tune, if added). Dev never used for final report numbers.
- Test: final paper metrics.

Power: 150 detects ≥ 8% recall difference at p<0.05 binomial. Acceptable
for relative-improvement claims; insufficient for absolute claims with
small effect sizes.

### Target — multi-tier

| Tier | recall | precision | Meaning |
|---|---:|---:|---|
| Minimum | 70% | 80% | Defensible thesis: improvement over v4 |
| Target | 85% | 90% | Successful v5 |
| Stretch | 95% | 95% | Excellent — competitive with claimed industry benchmarks |

Paper frames results against these tiers; no single arbitrary number.

## 6. Reasoning-heavy failure mode scope

Audit of v4 50-case cut identifies ~5–8% of test set as
**reasoning-required**, not retrieval-required (A30 needs "báo giảm → thẻ"
inference; A22 needs temporal + transitional law inference). These cases
are *not* solved by better embedding.

**v5 strategy**:
- Extend M5 graph expansion to **2-3 hops** instead of 1, leveraging the
  REFERS_TO edges already extracted by `rule_extract.py`.
- Some reasoning cases will still miss. Accept as a documented limitation
  in the paper; do *not* add a dedicated "reasoning module" for v5.
- **Sanity check before committing 2-3 hop**: measure REFERS_TO edge
  coverage in the current graph; if many cross-law refs are missing,
  fix at extractor level first.

## 7. Phased plan

| Sprint | Goal | Output | Time |
|---|---|---|---|
| **1** | Build vanilla v5 pipeline (M1+M4+M5+generator); run on 30-case probe + full 150 test | `src/retrieval/` module, recall/precision numbers, per-case audit | 2 weeks |
| **2** | Conditional additions (M2 fine-tune, M3 HyDE, M6 verifier) — based on Sprint 1 audit | Modules added with A/B vs Sprint 1 vanilla | 2 weeks |
| **3** | Full eval on 150 test, A/B against v4 baseline (re-aggregated under adaptive policy), paper-ready tables and analysis | Thesis chapter, final metric report | 2 weeks |

Total: **6 weeks** full-time. Matches budget cap.

## 8. Budget

| Item | Estimate | Cap |
|---|---|---:|
| Synthetic data ($0.001/pair × 10k pairs) | $5–10 | only if M2 triggered |
| GPU rental (A100 × 8h) for fine-tune | $10–15 | only if M2 triggered |
| Eval API costs (150 × 5 arms × $0.005 + ablations) | $20–40 | recurring |
| Probe + iteration | $5–10 | recurring |
| **Total** | | **$200 hard cap** |

Time: **6 weeks hard cap**. Money: **$200 hard cap**. If a sprint blocks,
report + propose alternative scope reduction rather than overrun.

## 9. Risk register

| Risk | Mitigation |
|---|---|
| Synthetic data drift from real query distribution | If M2 triggered, sample 30 synthetic queries, spot-check style vs real test queries; consider seeding from 50 dev cases. |
| Cross-encoder rerank latency on top-100 | Cap rerank candidates at 50; switch to BGE-M3 ColBERT-style late interaction if > 5s/query. |
| HyDE hallucination (if M3 triggered) | Low temperature, prompt constraint "trả lời theo style văn bản luật"; verify on 10 examples. |
| REFERS_TO graph coverage thin → 2-3 hop unproductive | Measure edge coverage upfront; if low, fix `rule_extract.py` before expanding hops. |
| Verifier same-source bias (if M6 triggered) | Use different model family (Claude or local NLI) instead of same OpenAI family. |
| Sample size 150 insufficient for absolute claims | Frame paper as relative improvement vs v4 baseline; document multi-tier targets. |

## 10. Acceptance gates

Cross-cutting (apply at end of Sprint 3):

| Metric | Threshold (Minimum tier) | Measured by |
|---|---|---|
| `citation_recall` end-to-end on 150 test (**strict tuple policy** — see §5) | ≥ 0.70 | `eval_core.metrics.compute_citation_metrics` |
| `citation_precision` end-to-end (strict tuple policy) | ≥ 0.80 | same |
| OOC detection F1 | ≥ 0.80 | derived |
| Median latency per question | ≤ 30s | end-to-end |
| Reproducibility: re-index with 1 new law (e.g., Luật Việc làm) requires zero code change | pass | manual test |

Target tier (85/90) and stretch tier (95/95) are bonus.

**Tier re-calibration caveat**: tiers were authored under the old
implicit (article-only) policy. The strict tuple-equal policy is
uniformly stricter, so the **re-aggregated v4 baseline numbers**
(after task #21) may sit well below the Minimum tier — that doesn't
necessarily invalidate v4 quality, only the tier definition. Decision
to keep / shift tiers will be made AFTER the re-aggregation reads,
not preemptively. Any tier shift must be documented in
`docs/changelog.md` with the v4 baseline numbers under both policies
for traceability.

## 11. Open questions, settled or deferred

1. ~~Synthetic data via `gpt-4o-mini` accepted?~~ → Deferred until M2 triggered.
2. ~~Adding M3 + M6 LLM calls accepted given latency/cost?~~ → Deferred; only
   if Sprint 1 audit motivates.
3. **Corpus scope**: stay at 3 laws (BHXH 41, BHXH 58, BLLĐ 45). Multi-law
   scaling test happens at Sprint 3 acceptance gate.

End of plan.
