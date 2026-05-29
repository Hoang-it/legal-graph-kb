# Plan v5 — General, scalable legal citation retrieval

> **Status: design under review, post-grilling rev 2. Not committed; updated locally
> until Sprint 1 (vanilla pipeline + audit) lands.**
> Supersedes: `plan_phase6_completion.md`, `plan_phase6_prolog_multilaw.md`, the
> entire `elite_graphrag_logic_rt_v*` arm family (v1 → v4).
> Owner: Hoàng — UIT MSc thesis.

## 1. Why v4 was discarded

v4 (`elite_graphrag_logic_rt_v4`) was a workaround layer. Its core mechanism —
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

Defined in `evaluation.compute_academic_metrics:compute_citation_metrics()`.

**Granularity policy: adaptive.**
- Gold cites with khoản → match strict (Điều + khoản must both match)
- Gold cites article-only (no khoản) → match lenient (Điều only)
- Implementation note: requires updating `compute_citation_metrics()` to read
  gold granularity per-citation and switch comparison key dynamically. v4
  baselines must be re-aggregated under the new policy before A/B.

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
| `citation_recall` end-to-end on 150 test (adaptive policy) | ≥ 0.70 | `evaluation.compute_academic_metrics` |
| `citation_precision` end-to-end | ≥ 0.80 | same |
| OOC detection F1 | ≥ 0.80 | derived |
| Median latency per question | ≤ 30s | end-to-end |
| Reproducibility: re-index with 1 new law (e.g., Luật Việc làm) requires zero code change | pass | manual test |

Target tier (85/90) and stretch tier (95/95) are bonus.

## 11. Open questions, settled or deferred

1. ~~Synthetic data via `gpt-4o-mini` accepted?~~ → Deferred until M2 triggered.
2. ~~Adding M3 + M6 LLM calls accepted given latency/cost?~~ → Deferred; only
   if Sprint 1 audit motivates.
3. **Corpus scope**: stay at 3 laws (BHXH 41, BHXH 58, BLLĐ 45). Multi-law
   scaling test happens at Sprint 3 acceptance gate.

End of plan.
