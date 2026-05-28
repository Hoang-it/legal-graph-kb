# Plan v5 — General, scalable legal citation retrieval

> Status: **approved by user, not yet implemented**.
> Supersedes: `plan_phase6_completion.md`, `plan_phase6_prolog_multilaw.md`, the entire `elite_graphrag_logic_rt_v*` arm family (v1 → v4).
> Owner: Hoàng — UIT MSc thesis.

## 1. Why v4 must be discarded

v4 (`elite_graphrag_logic_rt_v4`) was a workaround layer. Its core mechanism — `data/legal_issue_taxonomy.yaml` + `src/legal_issue_planner.py` + per-domain slot dependency map — encodes a hand-curated rule set per legal domain (maternity, pension, lump_sum, …). This blocks scaling along the only axis that matters for the thesis: **number of laws and number of questions**.

Concretely, evidence from the 50-question phase6 cut:

| Bucket (in-corpus gold = 31) | Count | Root cause |
|---|---:|---|
| Article-level hit | 13 | — |
| Predicted but wrong article | 11 | retrieval miss (gold not in top-K context) for 9/11; LLM pick lệch for 2/11 |
| Blocked by `evidence_gap` (source_family hard gate) | 3 | taxonomy filter too narrow |
| Blocked by `citation_validation_failed` | 6 | validator rules `missing_required_slot_citation` / `used_citation_without_claim` rejecting otherwise valid envelopes |
| Other (OOC-declared false, unknown gold) | 2 | — |

**73% of all failing cases have the gold article missing from the retrieval context delivered to the LLM**. Validator and planner over-strictness only inflate the loss; the underlying retriever is the bottleneck. v4 cannot be patched into a general solution — its abstraction is wrong.

## 2. Problem decomposition

Citation retrieval for Vietnamese legal QA is **three problems stacked**, not one:

1. **Domain-adapted dense retrieval**. BGE-M3 vanilla never saw Vietnamese statutory text during pretraining; lexical and syntactic distance between user-question phrasing and statutory phrasing is large.
2. **Multi-evidence aggregation for legal reasoning chains**. Median gold cites = 2–4 per question. Reasoning is a chain (applicability → eligibility → quantity → procedure) and frequently crosses laws (e.g. L41 BHXH ↔ L45 BLLĐ).
3. **Temporal / version-aware filtering**. Event date in the question selects which law version is in force (sự kiện 2014 → L58; sự kiện 2026 → L41). This is a property of node metadata, not a rule.

v4 conflates all three behind a YAML taxonomy.

## 3. Design principles for v5

1. **No YAML / Python file may contain patterns keyed by law name or legal domain.** Adding Luật Đất đai = re-run the indexing pipeline, edit zero lines of code.
2. **Every ranking / filter decision is driven by signal already in the data**: embedding score, BM25 score, `effective_from/until` metadata, citation edges. No handwritten rules.
3. **Retrieval recall improves through representation learning**, not by widening thresholds or hardcoded include lists.
4. **Each module is measurable in isolation**: retriever recall@K, reranker recall@K, verifier precision. Bottlenecks must surface from metrics, not from inspection.
5. **The citation graph is first-class structure**. Cross-law dependencies emerge from `REFERS_TO` edges, never from `if domain == pension then include L45.A169`.

## 4. v5 architecture — six modules

### M1 · Indexing layer (offline, scales O(n) with number of laws)

Each clause / article node carries:
- **Dense vector** (BGE-M3 dense, eventually fine-tuned — see M2)
- **Sparse vector** (BGE-M3 sparse + BM25) — catches rare statutory terms (`điều khoản chuyển tiếp`, `trợ cấp một lần`) where dense models smear meaning
- **Metadata**: `law_code`, `effective_from`, `effective_until`, `parent_article_id`, `position_in_article`
- **Citation edges**: `REFERS_TO`, `CITES_EXTERNAL` (already extracted by B2 rule extractor) — promoted to first-class Neo4j relationships, queryable in retrieval

Adding a new law = parse → embed → load. No code change.

### M2 · Domain-adapted retriever + reranker — core effort

This is ~70% of v5 effort and is the precondition for hitting recall ≥ 80%.

- **Synthetic Q–Clause data**. For each Khoản (≈1,585 currently), GPT-4o-mini generates 5–8 natural-language questions targeting that clause. Yields ~10k query-positive pairs. Cost ≈ $3-5 OpenAI.
- **Hard negative mining**. For each query, BM25 retrieves top-50, remove positives, take 15 lexically-close-but-wrong as hard negatives.
- **Fine-tune BGE-M3** with Multiple Negatives Ranking Loss (MNRL) + in-batch negatives. Output: a checkpoint specific to Vietnamese legal corpus.
- **Fine-tune cross-encoder reranker** (BGE-reranker-v2-m3 base) with margin MSE loss on the same hard negatives.
- **Held-out eval**: 20% of data is test split → measure recall@10 / recall@50 / recall@100 pre vs post fine-tune.

**Accept gate**: fine-tuned retriever recall@100 ≥ 0.95 on held-out synthetic; reranker recall@10 ≥ 0.85. If gates miss, debug data quality, not model architecture.

### M3 · LLM query decomposition (no taxonomy)

One LLM call (`gpt-4o-mini`, ~$0.001/query) returns JSON:

```json
{
  "sub_questions": ["...", "..."],
  "event_dates": ["2014-12"],
  "subjects": ["lao động nữ"],
  "actions": ["hưởng chế độ thai sản", "hồ sơ"]
}
```

The prompt is **domain-agnostic**: it asks the LLM to decompose along legal aspects of the question, without enumerating domains. Each `sub_question` is an independent retrieval intent. This replaces v4's hardcoded `LegalQueryPlan`; generalisation cost is essentially zero.

### M4 · Multi-query hybrid retrieval with native temporal filter

For each `sub_question`:
- top-30 dense + top-30 sparse
- **Reciprocal Rank Fusion** → top-50
- **Temporal filter in Cypher pre-rerank**: `WHERE article.effective_from <= $event_date AND (article.effective_until IS NULL OR article.effective_until > $event_date)`

Union all sub_question candidates → dedupe → top-100 globally.

### M5 · Cross-encoder rerank + citation-graph expansion

- Cross-encoder reranks top-100 → top-15
- For each of top-15, follow `REFERS_TO` 1 hop in Neo4j → add referenced articles
- Rerank the expanded pool → final top-12

Cross-law dependencies (e.g. L41.A64 → L45.A169 for retirement age) emerge naturally from edges. If the B2 rule extractor missed an edge, fix it at B2 (data quality), never patch at runtime.

### M6 · Answer + self-verification loop

- LLM (`gpt-4o`) answers and cites from top-12
- **Verifier** (second LLM call or NLI model): for each `(claim, cited_clause_text)`, score entailment — does the clause actually support the claim?
- Citations kept iff entailment score > threshold; tangential cites dropped.
- **Coverage check**: if any M3 `sub_question` is not covered by retained citations → re-retrieve for that sub_question with higher K → iterate, max 2 rounds.

The verifier is the general mechanism for hitting precision ≥ 90%: measure entailment, do not encode validator rules.

## 5. Phased plan

| Phase | Goal | Output | Duration |
|---|---|---|---|
| **0** | Prove the bottleneck: raw hybrid retrieval (vanilla BGE-M3 dense + BM25, RRF) on 200 questions, measure recall@100 vs in-corpus gold | Decision: skip vs invest in Phase 1 | 1 day |
| **1** | Synthetic data + fine-tune retriever + reranker | 2 model checkpoints + held-out metrics | 1–2 weeks |
| **2** | New retrieval pipeline (M1 + M4 + M5) replacing v4 evidence builder | `src/retrieval/` module + recall@K eval | 4 days |
| **3** | LLM query decomposition (M3) | `src/query_decompose/` module | 2 days |
| **4** | Verifier + iterative re-retrieval (M6) | `src/verifier/` module | 3 days |
| **5** | Full 200-question eval, A/B against v3/v4 baseline rows, paper-ready metrics | Final report + thesis chapter | 3 days |

Total ≈ 3–4 weeks of full-time work for one person.

## 6. Phase 0 — first thing to do once cleanup lands

Single script:

> Run hybrid retrieval (BGE-M3 dense + BM25 sparse, RRF fusion, top-100 clauses) on the 200-question evaluation set. Measure recall@100 at article level on the in-corpus subset.

Three outcomes:
- **recall@100 ≥ 0.90** → embedding is already fine; skip Phase 1, go straight to Phase 2-4.
- **recall@100 ∈ [0.70, 0.90)** → Phase 1 fine-tune is justified, target lifting to ≥ 0.95.
- **recall@100 < 0.70** → corpus problem (parsing errors, embedding mismatch); fix data before fine-tuning.

~50 lines of Python on existing infra, ~15 min to run.

## 7. Acceptance gates for v5

| Metric | Threshold | Measured by |
|---|---|---|
| Article-level recall@10 (in-corpus gold) | ≥ 0.85 | M5 output vs gold |
| Citation precision after verifier | ≥ 0.90 | verifier-kept × entailment-correct |
| OOC detection F1 | ≥ 0.90 | gold-OOC vs predicted-OOC |
| Median latency per question | ≤ 30s | end-to-end |
| Scale test: indexing a new law (e.g. Luật Việc làm) requires zero code change | pass | run pipeline on 50 new synthetic Qs |

## 8. Risks

- **Synthetic data drift** — LLM-generated queries may differ stylistically from real user queries. Mitigation: spot-check 100 samples; filter low-reranker-score samples.
- **GPU cost** — BGE-M3 ≈ 568M params, batch 16 fp16, ≈10–15h on RTX 3090. Rentable A100 colab/runpod at ~$1–2/h × ~6h is the alternative.
- **Eval API cost** — 200 questions × 3 LLM calls × ~$0.005 ≈ $3/run. Negligible.
- **Reranker latency** — cross-encoder over 100 candidates ≈ 3–5s. Fallback: BGE-M3 ColBERT-style late interaction (vector-space rerank, no LLM).

## 9. Open questions, to confirm before Phase 0 starts

1. Synthetic data via `gpt-4o-mini` accepted? (Yes assumed unless human-curated is required.)
2. Adding M3 + M6 LLM calls (3 calls per query total, ~$0.01) accepted given the latency/cost?
3. Corpus scope for the thesis: stay at 3 laws (BHXH 41, BHXH 58, BLLĐ 45) for v5 launch, or expand immediately? Expansion requires source docx files staged in `data/raw/`.

## 10. What gets removed from the repo before v5 implementation begins

The cleanup pass (separate task) removes: the v1–v4 phase0/phase6 prolog runtime arms, `legal_issue_taxonomy.yaml`, `legal_issue_planner.py`, `legal_evidence.py`, `legal_citation_validator.py`, the Prolog stack (`extract_prolog.py`, `load_prolog.py`, `validate_prolog.py`, `prolog_utils.py`), and the elite_pipelines arm wiring for `*_logic_rt_v*`. Baseline arms (`graphrag`, `llm_only`, `elite_no_retrieval`, `elite_ontology`, `elite_graphrag`) are kept as comparison anchors for the v5 paper.

End of plan.
