# 05 — v5 Retrieval-only Audit (Stage A, post-hoc)

## What

Tách `retrieval_recall@K` (= "gold article có vào top-K candidate set không")
khỏi `e2e_citation_recall` (= "LLM có cite đúng từ context không"). Cost $0, no
new API calls — script đọc records đã có từ Sprint 0/1/2.

## Why

[Plan v5 §5](../../docs/plans/v5_general_retrieval.md#5-metric-framework):
*"retrieval_recall@K does not directly improve citation_recall — they live at
different pipeline layers"*.

E2E recall = `retrieval_recall@K × LLM_cite_rate × parser_strictness`. Sprint 2
A/B Δ in-corpus +17% E2E nhưng dense-only A/B chỉ -1.1%. Cần biết: bottleneck
hiện tại là retrieval ceiling hay LLM citing? Trả lời quyết định Sprint 3 scope.

## Method

Script: [`scripts/audit_retrieval.py`](../../scripts/audit_retrieval.py).

- 30-probe (stt 1..30) — same set used in Sprint 1 + Sprint 2 inference.
- Extract retrieved article ids từ record:
  - `graphrag` (S0): `vector_hits[*].clause_id` → parse to article (8 entries max).
  - `graphrag_v5` (S1): `hits[*]` (seeds + neighbours), `article_id` / `target_id` (12 max).
  - `graphrag_v5_m2` (S2): same as S1.
- Compute `recall@K` for K in {5, 10, 12, all-available} vs strict gold_articles.
- Compute E2E recall từ citation_ids đã parsed.
- Stratify theo gold corpus type (in_corpus / mixed / ooc / unparseable).

## Headline results

### Table 1 — Macro retrieval@K vs E2E (30-probe overall)

| arm | n | @5 | @10 | @12 | @all | E2E | gap (@12 − E2E) |
|---|---:|---:|---:|---:|---:|---:|---:|
| graphrag (S0) | 30 | 0.174 | 0.174 | 0.174 | 0.174 | 0.094 | **+0.079** |
| graphrag_v5 (S1) | 30 | 0.282 | 0.306 | 0.306 | 0.306 | 0.236 | **+0.070** |
| **graphrag_v5_m2 (S2)** | 30 | 0.243 | 0.329 | **0.333** | 0.333 | 0.224 | **+0.110** |

(graphrag flatlines: only 5-8 hits stored, recall maxes out at K=5.)

### Table 2 — Stratified by corpus type

| category | arm | n | @5 | @10 | @12 | E2E |
|---|---|---:|---:|---:|---:|---:|
| **in_corpus** | graphrag (S0) | 14 | 0.304 | 0.304 | 0.304 | 0.179 |
| **in_corpus** | graphrag_v5 (S1) | 14 | 0.429 | 0.446 | 0.446 | 0.321 |
| **in_corpus** | **graphrag_v5_m2 (S2)** | 14 | 0.321 | 0.506 | **0.506** | **0.375** |
| mixed | graphrag (S0) | 3 | 0.153 | 0.153 | 0.153 | 0.111 |
| mixed | graphrag_v5 (S1) | 3 | 0.319 | 0.472 | 0.472 | 0.361 |
| mixed | graphrag_v5_m2 (S2) | 3 | 0.431 | 0.431 | 0.472 | 0.319 |
| ooc | all arms | 7 | 0.000 | — | — | 0.000 |
| unparseable | graphrag (S0) | 6 | 0.083 | 0.083 | 0.083 | 0.000 |
| unparseable | graphrag_v5 (S1) | 6 | 0.250 | 0.250 | 0.250 | 0.250 |
| unparseable | graphrag_v5_m2 (S2) | 6 | 0.250 | 0.250 | 0.250 | 0.083 |

## Honest interpretation

### Finding 1 — M2 IS lifting retrieval (genuine retrieval contribution)

**In-corpus retrieval@12**: M2 = 0.506 vs vanilla v5 = 0.446 = **+13.5% relative**.

Sprint 2 e2e in-corpus +17% rel không phải artifact của LLM/parser variance — M2
genuinely retrieve more correct articles. Dense-only A/B (-1.1% trên dev) đã
undersold thật sự M2 contribution vì:
- Dense-only đo isolated dense path (1 trong 4 retrieval signals)
- Full pipeline (sparse + RRF + temporal + rerank + graph hop) amplify M2 signal

### Finding 2 — LLM citation rate là vấn đề tách biệt và tăng nặng dần

| arm | retrieval@12 | E2E | citation loss % |
|---|---:|---:|---:|
| graphrag (S0) | 0.174 | 0.094 | **46%** loss |
| graphrag_v5 (S1) | 0.306 | 0.236 | 23% loss |
| **graphrag_v5_m2 (S2)** | 0.333 | 0.224 | **33%** loss |

→ M2 retrieves nhiều hơn nhưng LLM cite proportional ít hơn. Có thể vì:
- M2 candidates rerank2 score gần nhau hơn (0.93-0.97 vs vanilla 0.97-0.99) →
  LLM khó pick top.
- Context có nhiều clause "tương tự" (cùng đề tài) → LLM confuse cite đúng cái.

### Finding 3 — Retrieval ceiling vẫn dưới gate Plan §10

- Plan §10 minimum tier: e2e recall ≥ 0.70 overall.
- Best retrieval ceiling hiện tại: M2 overall = 0.333, in-corpus = 0.506.
- Even with **PERFECT LLM citing** (gap = 0), max E2E achievable hiện tại ≈ 0.50
  on in-corpus.
- → Hit 0.70 overall **bắt buộc** trong scope hiện tại = retrieval lift thêm
  (M3 HyDE hoặc M2 v3) HOẶC corpus expansion (Plan §11 chốt 3 luật, blocked).

### Finding 4 — Sprint 0 graphrag baseline bị giới hạn measurement

`vector_hits` chỉ lưu 5-8 entries (top vector_search) → recall không thể đo
beyond K=8. Reading code: actual `n_vector_hits=8` (top-K=8 set ở Sprint 0), nhưng
top-5 saved to record. → Sprint 0 number @5..@12 cùng giá trị 0.174 là artifact
của storage, không phải retrieval ceiling thật.

## Sprint 3 redirect (replace plan trước đó)

Stage A đủ signal để bypass Stage B (full 150 retrieval-only). Sprint 3 scope:

### Phase 9a (PROCEED) — Full 150-test E2E M2 + vanilla v5

- Lý do: M2 retrieval contribution confirmed (+13.5% in-corpus). 150-test cho
  statistical power (CI ±4-5 pts) thay vì 30-probe ±10-15 pts.
- Cost: graphrag_v5_m2 trên 120 stt mới = ~$0.5, ~13 min. graphrag_v5 trên 120
  stt mới = ~$0.5, ~80 min.
- Output: paper-grade numbers + stratified report.

### Phase 9b (SKIP) — Reranker-only & M2-only ablation arms

- Lý do: Stage A đã isolate M2 contribution at retrieval level (0.506 vs 0.446
  in-corpus). Diff-in-diff với reranker swap không add unique info — reranker
  swap chủ yếu là latency win (9× speedup), không retrieval quality.
- Save: ~$2.5, ~3h work.

### Phase 10 (PROCEED) — Generator-side investigation

- Lý do mới từ Finding 2: LLM citation loss = 33% trên M2. Đây là biggest single
  source of E2E underperformance.
- Hypothesis testing trên 150 test:
  - **Prompt v3**: more aggressive template enforcement + explicit "cite từ
    cluster top-3 most relevant".
  - **Reduce context size**: top-12 → top-8 → ép LLM focus.
  - **M6 verifier (claude-haiku)**: post-hoc drop low-confidence cites — giảm FP
    đồng thời (precision lift).
- Cost: ~$1, ~30 min per variant.

### Phase 11 (PROCEED) — Per-question fail-mode taxonomy

- 5 fail modes: OOC / multi-hop / terminology gap / parser miss / LLM cite-loss.
- Output: tables for thesis chapter limitations section.

## Decision: bottleneck is mixed, not purely either side

| Layer | Contribution to E2E loss | Sprint 3 priority |
|---|---|---|
| OOC (corpus scope) | 23% of probe = 0 recall | Plan §11 blocked, document as limitation |
| Retrieval ceiling | M2 retrieval = 0.333 overall | Sprint 3 = full 150-test on M2 (no more ablation) |
| **LLM cite loss** | 33% loss from retrieval to E2E | Sprint 3 = generator-side experiments (Phase 10) |

## Artifacts

- [stage_a_results.json](stage_a_results.json) — per-record retrieved articles, gold articles, recall@K, e2e_recall, category.
- [scripts/audit_retrieval.py](../../scripts/audit_retrieval.py) — script (idempotent, re-runnable, deterministic).

## Next action

Recommended sequence:
1. Sprint 3 Phase 9a: full 150 test trên graphrag_v5_m2 + graphrag_v5.
2. Sprint 3 Phase 10: generator-side experiments per Finding 2.
3. Sprint 3 Phase 11: fail-mode taxonomy on 150-test results.

Skip ablation arms (reranker-only, M2-only) — Stage A đã isolate.
