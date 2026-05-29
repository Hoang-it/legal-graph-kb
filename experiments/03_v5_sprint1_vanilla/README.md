# 03 — v5 Sprint 1 vanilla (30-probe)

## What

Đo recall/precision **end-to-end** của vanilla v5 pipeline trên **30-câu probe** để có baseline
số liệu cho Sprint 2 conditional additions.

## Why

[Plan v5 §1](../../docs/plans/v5_general_retrieval.md#1-why-v4-was-discarded) đã chứng minh
73% case fail của v4 là retrieval miss (gold không vào top-K LLM thấy). Sprint 1 thay path
single-dense (`runtime/rag_query.py:vector_search`) bằng pipeline M4+M5 vanilla — không
fine-tune, không HyDE, không verifier — để đo *exactly* phần improvement nào đến từ
"retrieval better" thuần túy.

Sprint 2 chỉ trigger module thêm khi audit Sprint 1 cho thấy *cần*.

## Setup

### Arms

| Arm | Mode | Mô tả |
|---|---|---|
| `graphrag_v5` | **run** | New: dense top-30 ∥ BM25 top-30 → temporal filter → RRF → CE rerank top-15 → REFERS_TO 2-3 hop → re-rerank → top-12 → GPT-4o-mini |
| `graphrag` | inherit từ `01_initial_eval` | Baseline để A/B |
| `llm_only` | inherit từ `01_initial_eval` | Control (no retrieval) |

### Dataset

- `data/eval/questions_200.json`, **first 30** (deterministic seed). Stratified split 150/50
  chính thức chỉ chạy ở Sprint 3 (§5 plan).
- Gold granularity check: 200/200 câu chỉ có gold **article-only** (parse từ `gold_citations_raw`).
  Adaptive-granularity metric của plan §5 vì vậy moot cho dataset này — defer per skill Rule 2
  (metric change cần separate commit + baseline re-aggregate).

### Implementation choices (đã chốt trong design session)

| Decision | Chọn | Lý do |
|---|---|---|
| Sparse retriever | Neo4j FULLTEXT `clause_fulltext` (Lucene BM25) | Đã có ở schema, 0 offline work mới |
| Cross-encoder | `BAAI/bge-reranker-v2-m3` | Cùng family BGE-M3, multilingual, ~568M params chạy được trên RTX 3050 4GB |
| K params | dense=30, sparse=30, RRF top-50, rerank-1 top-15, rerank-2 top-12 | Đúng plan §4 |
| RRF k | 60 | Default literature (Cormack 2009) |
| Hop expansion | REFERS_TO 1..3 hops | Đúng plan §6 |
| Temporal filter | `Law.effective_date` + `REPEALS` edges; event_date detect từ question text (year mention) → fallback today | Data-driven, no hardcoded law list |
| Metric | Existing `compute_citation_metrics` (article-level) | Adaptive granularity moot — see Dataset note |

### Pre-flight audit findings (chạy trước impl)

| Signal | Value | Tác động Sprint 1 |
|---|---|---|
| Total `REFERS_TO` edges | **25** | Hop expansion sẽ contribute rất ít |
| Unique src clauses | 16 / 1585 (1.0%) | Chỉ ~1% clause có ANY out-edge REFERS_TO |
| Unique dst articles | **2** | Expansion trỏ về chỉ 2 article duy nhất |
| Cross-law / intra-law | 15 / 10 | Hầu hết là L41 → L45 (chỉ 1 cặp Điều) và L45 → L45 self-cite |

→ §6 risk register cảnh báo này HIỆN HỮU. Sprint 1 vẫn implement 2-3 hop để đo *quantitatively*
contribution của graph expansion; quyết định fix `offline/rule_extract.py` deferred sang
output Sprint 1.

## Expected outcome (pre-commitment)

**Predict trước khi chạy** (để chặn post-hoc rationalization):

- `citation_recall` (macro): expect ≥ 0.30 (cải thiện 2-3× so với baseline 0.129).
- `citation_precision`: expect ≥ 0.20 (BM25 + rerank giảm noise vs single-dense).
- Hop expansion contribution: expect ~0 chunks added/query trên ≥80% câu (do coverage thin).
- Latency: expect 4–8s/query (CE rerank + extra Neo4j calls).

**Threshold change-of-mind**:
- Nếu recall < 0.20 → vanilla v5 không đủ; Sprint 2 phải có M2 (fine-tune BGE-M3).
- Nếu precision < 0.15 → noise quá nhiều; xem xét M6 verifier.
- Nếu recall ≥ 0.50 → vanilla đã rất tốt, Sprint 2 chỉ cần marginal tuning.

## Aborted runs

### Run 1 — 2026-05-29, CUDA OOM 27/30
- 3/30 câu thành công (A1, A2, A3), 27/30 fail với `RuntimeError: CUDA error: out of memory`.
- Nguyên nhân: RTX 3050 4 GB VRAM không kham nổi BGE-M3 (1.5 GB fp32) + BGE-reranker-v2-m3
  (~1.1 GB) cộng peak activation của reranker batch=16; fragmentation tích lũy sau ~3 query.
- **Không sửa thuật toán / kiến trúc.** Fix config-level duy nhất:
  - Default reranker batch 16 → 4 (env `V5_RERANKER_BATCH`).
  - Gọi `torch.cuda.empty_cache()` cuối mỗi `CrossEncoderReranker.rerank()`.
- Xoá `A*.error.json` trước khi resume; giữ A1.json/A2.json/A3.json đã thành công.

## Result summary — Run 2 (2026-05-29, n=30, apples-to-apples)

Artefacts: [metrics/academic_metrics.json](metrics/academic_metrics.json) ·
[report/academic_report.md](report/academic_report.md).

### Headline — same-30-stt comparison

Cả 3 arm tính lại trên đúng 30 stt v5 đã chạy. Cột `eval_core/metrics.py`,
chính sách article-level (adaptive granularity moot vì gold dataset 0% có khoản).

| arm | n | recall_macro | precision_macro | f1_macro | recall_micro | precision_micro | latency_macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| **graphrag_v5** | 30 | **0.2361** | **0.1867** | **0.1915** | 0.2069 | 0.2182 | 39.18s |
| graphrag (inherit) | 30 | 0.0944 | 0.0556 | 0.0656 | 0.0690 | 0.0851 | 4.91s |
| llm_only (inherit) | 30 | 0.0111 | 0.0333 | 0.0167 | 0.0172 | 0.0435 | 5.72s |

→ **v5 cải thiện 2.5× recall, 3.4× precision, 2.9× f1 so với baseline graphrag**
trên cùng 30 câu. Cost: 8× latency (5s → 39s) — bottleneck ở 2 lần cross-encoder rerank.

### Stratified by corpus type (signal Sprint 2)

Phân nhóm 30 câu theo gold_citations_raw có nằm trong 3-luật-KG (L41_2024 / L58_2014 / L45_2019)
hay không:

| category | n | v5 recall_ma | v5 precision_ma | baseline recall_ma | baseline precision_ma |
|---|---:|---:|---:|---:|---:|
| **in-corpus** (mọi gold ∈ KG) | 14 | **0.3214** | **0.2309** | 0.1786 | 0.0833 |
| OOC (gold không có trong KG) | 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mixed | 3 | 0.3611 | 0.5952 | 0.1111 | 0.1667 |
| unparseable gold_code | 6 | — | — | — | — |

**Quan sát chính**:
- 14 câu in-corpus: v5 recall **0.32** vs baseline 0.18 — improvement thực tại pipeline retrieval.
- 7 câu OOC: cả 2 arm 0.0 — đúng, ground truth không thể đạt khi KG thiếu Nghị định/Thông tư.
  Đây không phải lỗi retrieval; đây là **corpus scope problem**.
- 23% gold (7/30) là OOC → kết quả overall 0.236 bị kéo xuống bởi nhóm này.

### So với prediction (pre-commit, [§Expected outcome](#expected-outcome-pre-commitment))

| Predict | Reality | Verdict |
|---|---|---|
| recall ≥ 0.30 | 0.236 overall, **0.321 in-corpus** | UNDER overall, MET in-corpus |
| precision ≥ 0.20 | 0.187 overall, **0.231 in-corpus** | UNDER overall, MET in-corpus |
| Hop expansion ~0/query | mean **3.53 / query**, 27/30 query có ≥1 neighbor | **WRONG** — REFERS_TO thưa nhưng seeds rơi đúng cụm có ref → contribution thật |
| Latency 4–8s/query | **~42s/query** | FAR OVER — cross-encoder bottleneck |

### Per-stage audit (mean trên 30 query)

| stage | mean | median | ghi chú |
|---|---:|---:|---|
| n_dense | 30.0 | 30 | tối đa K |
| n_sparse | 30.0 | 30 | tối đa K |
| n_dropped_by_temporal | 21.2 | 19 | gần ⅓ candidate bị temporal filter loại (L58_2014 không-in-force khi default-today) |
| n_after_fusion | 27.2 | 29 | sau RRF |
| n_seeds | 14.0 | 15 | gần đủ cap 15 |
| n_neighbors_added | 3.53 | 3 | REFERS_TO expansion **thực sự contribute** dù coverage thưa |
| retrieve (Cypher + BGE-M3 encode) | 0.82s | 0.50 | nhanh hơn predict |
| rerank1 | 18.01s | 15.12 | bottleneck |
| expand | 0.07s | 0.02 | negligible |
| rerank2 | 17.71s | 13.35 | bottleneck |
| llm (gpt-4o-mini) | 5.30s | 4.65 | bình thường |

Cross-encoder rerank = **90% latency**. Batch=4 trên RTX 3050 4GB là tối ưu hiện tại;
giảm latency triệt để cần GPU lớn hơn HOẶC switch sang bge-reranker-base (~278M).

### Per-record outcome (gold ⟂ predicted)

| outcome | count |
|---|---:|
| Full hit (recall=1.0) | 5/30 |
| Partial hit | 5/30 |
| Total miss (recall=0.0) | 20/30 |

Trong 20 total miss, ≥7 là OOC. Còn lại ~13 câu có gold in-corpus nhưng pipeline vẫn miss
→ retrieval recall@K thực vẫn là vấn đề lớn nhất.

### Edge case observed

`stt=24` (bà Thiên Kim, sự kiện 1995): pipeline detect `event_date=1995-12-31` → KHÔNG luật nào trong KG
in-force tại 1995 (L58 từ 2016, L45 từ 2021, L41 từ 2025) → temporal filter drop tất cả 60/60
candidate → pipeline đúng đắn trả `"không có đủ thông tin để trả lời chính xác câu hỏi này"`. Behavior đúng theo design — không phải bug.

## Decision — Sprint 2 trigger analysis

Tham chiếu [plan §4 trigger table](../../docs/plans/v5_general_retrieval.md#sprint-2--conditional-additions):

| Module | Trigger threshold | Quan sát Sprint 1 | Trigger? |
|---|---|---|---|
| **M2** Fine-tune BGE-M3 | Gold systematically missing from top-12 cả khi đã expand | 13/30 in-corpus miss; recall ceiling 0.32 << 0.70 gate | **YES** |
| **M3** HyDE | Multi-aspect queries fail systematically | Chưa diagnose; cần per-case audit (deferred) | Cần thêm signal |
| **M6** Verifier | e2e precision < 80% (tangent cites) | precision 0.187 << 0.80 | **YES** (low priority — recall mới là bottleneck) |
| **M5+ hops** OR document limitation | Reasoning case > 5–8% | Chưa enumerate; deferred | Defer |
| **Corpus expansion** (không có trong plan §4) | OOC rate quá cao | **23% OOC** trên 30 stt — kéo overall metric xuống | Cần đánh giá phạm vi: load thêm Nghị định / Thông tư |

### Conclusion (honest)

**Vanilla v5 hoạt động đúng plan và cải thiện rõ rệt so với single-dense baseline**
(2.5–3× trên mọi metric, in-corpus subset đạt 0.32 recall vs 0.18). Nhưng absolute level
**vẫn xa minimum acceptance gate** §10 (0.70 recall). Sprint 2 phải trigger **M2 fine-tune**;
M6 verifier có thể defer cho tới khi recall ≥ 0.50.

**Bất ngờ tích cực**: hop expansion qua REFERS_TO contribute thực (mean 3.53 neighbor/query)
dù §6 sanity audit cho thấy chỉ 25 edge — vì seeds rơi đúng vào cluster có edge.

**Vấn đề structural**: 23% câu là OOC. Plan §11 đã chốt "stay at 3 laws". Nhưng nếu Sprint 3
chấp nhận gate 0.70 recall trên 150-test stratified, **bắt buộc** hoặc (a) tăng corpus
(load Nghị định 115/2015, 152/2006, ... — additive offline), hoặc (b) báo cáo metric tách
in-corpus vs OOC như paper-grade analysis.

**Bottleneck operational**: cross-encoder rerank chiếm 90% latency và là nguyên nhân OOM
ban đầu. Cân nhắc cho Sprint 2: bge-reranker-base (~278M) hoặc move CE inference sang batch=1
+ fp16; hoặc accept latency và scale GPU.

