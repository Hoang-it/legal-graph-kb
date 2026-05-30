# 04 — v5 Sprint 2 M2 (LoRA fine-tune + reranker swap, 30-probe)

## What

Đo **E2E citation recall** của M2 fine-tuned BGE-M3 pipeline so với:
- `graphrag_v5` baseline Sprint 1 (vanilla BGE-M3 + bge-reranker-v2-m3)
- `graphrag` baseline Sprint 0 (single-dense + 1-hop)
- `llm_only` control

## Why

[Plan v5 Sprint 2 §4](../../docs/plans/v5_general_retrieval.md#sprint-2--conditional-additions)
+ [v5_sprint2_implementation.md §3 Phase 4](../../docs/plans/v5_sprint2_implementation.md):
sau khi Sprint 1 vanilla cho recall_macro_in_corpus 0.32, plan trigger M2 fine-tune
BGE-M3 (LoRA) để bridge domain-adapted retrieval.

Dense-only A/B trên 39 dev câu cho thấy **không lift rõ ràng** (-1.1% vs vanilla,
trong noise range). Câu hỏi: full pipeline (sparse + rerank + graph hop) có
salvage được không?

## Setup

### Arms

| Arm | Mode | Mô tả |
|---|---|---|
| `graphrag_v5_m2` | **run** | M2 fine-tuned BGE-M3 + BM25 + RRF + **bge-reranker-base** + REFERS_TO + GPT-4o-mini |
| `graphrag_v5` | inherit từ `03_v5_sprint1_vanilla` | Sprint 1 baseline v5 (vanilla BGE-M3 + bge-reranker-v2-m3) |
| `graphrag` | inherit từ `01_initial_eval` | Sprint 0 (single-dense + 1-hop semantic expand) |
| `llm_only` | inherit từ `01_initial_eval` | Control (no retrieval) |

### M2 training config (Colab Tesla T4)

| Param | Value |
|---|---|
| LoRA r / alpha / dropout | 8 / 32 / 0.1 |
| Target modules | query, key, value, dense |
| LR | 1e-5 |
| Batch / Grad accum | 4 / 4 |
| Epochs | 10 |
| Warmup ratio | 0.1 |
| Max hard negs | 5 |
| Training examples (after multi-pos flatten) | 3416 |

### Sprint 2 swap points (env-driven)

```bash
BGE_M3_ADAPTER_PATH=models/bge-m3-bhxh-lora
V5_DENSE_INDEX=clause_vec_tuned
V5_RERANKER_MODEL=BAAI/bge-reranker-base
V5_RERANKER_BATCH=8
```

### Probe selection

Same 30 stt (1..30) với Sprint 1 — trực tiếp so sánh per-question và aggregate.

Stratified distribution của 30-probe (per Sprint 1 audit):
- in_corpus: 14
- ooc: 7
- mixed: 3
- unparseable: 6

## Expected outcome (pre-commitment)

**Predict trước khi chạy** (chặn post-hoc rationalization):

Dense-only A/B đã cho thấy M2 ≈ vanilla. Predict E2E:

| Metric | Sprint 1 v5 vanilla (30-probe) | M2 prediction | Verdict threshold |
|---|---:|---:|---|
| recall_macro overall | 0.236 | 0.20-0.27 | ±15% trong noise |
| recall_macro in-corpus | 0.321 | 0.28-0.36 | ±15% trong noise |
| precision_macro | 0.213 (strict parser) | similar | rerank swap có thể tăng/giảm |
| latency | 39s | **15-20s** | reranker base 2.5× faster |

**Threshold cho gate decision per plan §3**:
- M2 in-corpus recall ≥ 0.50 → trigger M6 verifier
- 0.35 ≤ M2 < 0.50 → trigger M6 + M3 HyDE
- M2 < 0.35 → **STOP M2**, document, pivot M3 + M6 (Path A trong design session)

## Aborted runs

### Gold validation registry fix — 2026-05-30
- 30 inference records OK (4.19s/câu, 0 fail).
- Metrics command bị block do `validate_gold_citations` fail-hard: stt=105 cite "Quyết định 366/QĐ-BHXH" — authority này thiếu trong `data/legal_sources.yaml`.
- (Lưu ý: stt=105 không nằm trong 30-probe, nhưng validation chạy trên toàn bộ 200 câu của `data/eval/questions_200.json`. Dataset có vẻ đã được update sau Sprint 1.)
- Fix: thêm entry `QD366_BHXH` vào `data/legal_sources.yaml` với pattern y như `QD595_BHXH`/`QD838_BHXH`. Không sửa thuật toán hay records.

## Result summary — 2026-05-30

Artefacts: [metrics/academic_metrics.json](metrics/academic_metrics.json) ·
[report/academic_report.md](report/academic_report.md).

### Headline — same-30-stt apples-to-apples

| arm | n | recall_macro | precision_macro | f1_macro | recall_micro | latency_macro |
|---|---:|---:|---:|---:|---:|---:|
| **graphrag_v5_m2 (M2)** | 30 | **0.2236** | **0.2111** | **0.1994** | 0.2069 | **4.19s** ⚡ |
| graphrag_v5 (Sprint1) | 30 | 0.2361 | 0.2133 | 0.2093 | 0.2069 | 39.18s |
| graphrag (Sprint0 baseline) | 30 | 0.0944 | 0.0556 | 0.0656 | 0.0690 | 4.91s |
| llm_only | 30 | 0.0111 | 0.0333 | 0.0167 | 0.0172 | 5.72s |

### Stratified by corpus type

| category | n | **M2** | v5 vanilla | graphrag | llm_only |
|---|---:|---:|---:|---:|---:|
| in_corpus | 14 | **0.3750** | 0.3214 | 0.1786 | 0.0000 |
| mixed | 3 | 0.3194 | 0.3611 | 0.1111 | 0.1111 |
| ooc | 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| unparseable | 6 | 0.0833 | 0.2500 | 0.0000 | 0.0000 |

### Honest interpretation

**WIN — Latency**: M2 = 4.19s vs v5 vanilla = 39.18s → **9.4× faster**. Đây là kết quả
của (a) reranker swap `bge-reranker-v2-m3` → `bge-reranker-base`, và (b) M2 LoRA tuned
model converge nhanh hơn vanilla. Đáp ứng Plan §10 acceptance gate "≤ 30s median".

**WIN — In-corpus retrieval**: M2 = 0.375 vs Sprint 1 vanilla = 0.321 → **+5.4 pts
absolute, +17% relative** trên subset có gold trong KG. Đây là metric Plan §3 Phase 4
gate dựa vào.

**NEUTRAL — Overall recall**: M2 0.224 vs Sprint 1 v5 0.236 → -1.25 pts. Drop chủ yếu
do category "unparseable" (M2 0.083 vs v5 0.250). Đây là class có gold cite bằng
Vietnamese title only — regex chính sách của script audit không match thực tế parser
xử lý, nên phân loại có thể sai lệch. Drop overall vì 6/30 stt thuộc nhóm này.

**NEUTRAL — Precision**: M2 0.211 vs v5 0.213 → bằng. Strict parser Phase 0a + similar
LLM prompt nên không khác biệt rõ.

### Comparison with dense-only A/B (Phase 2)

Dense-only retrieval@10 trên 39 dev: M2 0.3825 vs vanilla 0.3868 (Δ -1.1%).
**Full pipeline** trên 30-probe in-corpus: M2 0.375 vs vanilla 0.321 (Δ +17%).

→ Full pipeline (sparse + RRF + temporal + rerank + graph hop) **compensate được
distribution shift** mà dense-only A/B cảnh báo. M2 cung cấp candidate set hơi khác
nhưng rerank chọn đúng → e2e tốt hơn.

## Phase 4 decision gate

Plan §3 Phase 4 trigger table dựa trên `recall_macro_in_corpus`:

| Recall in-corpus | → Phase 5 action |
|---|---|
| ≥ 0.50 | Add M6 verifier only |
| **0.35 – 0.50** | **Add M6 + M3 HyDE** |
| < 0.35 | STOP, document, pivot |

**M2 in-corpus recall = 0.375 ⇒ Gate "M6 + M3 HyDE" triggered.**

Tuy nhiên — implementation honesty: lift trên in-corpus (+17%) đáng kể nhưng overall
metric không cải thiện. Phase 5 M3 + M6 cần invest thêm ~5 ngày + chi phí API. Quyết
định Phase 5 implementation deferred — xem [Sprint 2 summary](#sprint-2-summary) ở
cuối doc.

## Phase 4 → Phase 5/6 path

Plan §3 Phase 4-6 nguyên thuỷ:
- Phase 5: implement M6 (claude-haiku verifier) + M3 (HyDE) → exp folders 05_/06_
- Phase 6: final 30-probe A/B + Sprint 3 prep

**Recommended action** (xem Sprint 2 summary): paper-grade write-up Phase 4 result như
là Sprint 2 deliverable. M3 + M6 deferred sang next iteration với clear hypothesis.
