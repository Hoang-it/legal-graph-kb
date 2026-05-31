# 08 — HyDE retrieval (gpt-4o-mini, stratified 50-question pilot)

## What

Test [HyDE (Hypothetical Document Embeddings, Gao et al. 2022)](https://arxiv.org/abs/2212.10496)
trên kênh dense BGE-M3 của `V5RetrievalPipeline`. Generator: OpenAI
`gpt-4o-mini` (N=1 doc/câu, max_tokens=700, temperature=0). Plug-in
point: **chỉ thay query embedding của dense channel** — sparse channel
giữ nguyên câu hỏi gốc (plan §D4).

**Scope**: PILOT 50 câu stratified (in_corpus / mixed / ooc /
unparseable theo tỉ lệ tự nhiên của dataset 200; seed=0). Mục tiêu:
đo cost + retrieval lift trước khi quyết định scale full 200.

Bốn arms (đều retrieval-only, không LLM generator):

| arm | method | mô tả |
|---|---|---|
| `dense` | `retrieve_dense_only` | BGE-M3 LoRA + `clause_vec_tuned`, dense_k=100 |
| `dense_hyde` | `retrieve_dense_only_hyde` | giống `dense`, nhưng query = embedding của HyDE doc |
| `full_rerank` | `retrieve_only` (no hyde) | full v5 scaled (giống exp 07) |
| `full_rerank_hyde` | `retrieve_only` (with hyde) | giống `full_rerank`, dense embed = HyDE doc |

## Why pivot từ Qwen → gpt-4o-mini

Plan ban đầu ([`docs/plans/hyde_qwen_colab.md`](../../docs/plans/hyde_qwen_colab.md))
dùng Qwen 2.5 3B trên Colab Free T4 để tránh API cost. Sau khi
implement, user judge hướng 7B fallback (nếu 3B chất lượng kém trên VN
legal text) không khả thi do VRAM tight. Pivot sang gpt-4o-mini:

- **Cost rẻ + transparent**: ~$0.025 cho pilot 50 câu, ~$0.10 cho full
  200. Mọi call lưu lại usage + cost_usd trong cache → audit chính xác.
- **Reproducible**: OpenAI snapshot id (`gpt-4o-mini-2024-07-18`) ghi
  trong cache payload field `model_returned`.
- **Không cần Colab / GPU**: chạy local, dùng OpenAI SDK đã pin sẵn
  trong `pyproject.toml`.
- **Cache LLM**: yêu cầu user — mọi response persist ở
  `artifacts/hyde/openai__gpt-4o-mini/<sha>.json`, re-run = $0.

## Why HyDE for this project

Funnel `full_rerank` K=12 in_corpus (n=151) — xem
[`experiments/06_retrieval_dense_vs_full/report/funnel_full_rerank_K12.md`](../06_retrieval_dense_vs_full/report/funnel_full_rerank_K12.md)
— cho thấy dense là nguồn signal chủ đạo. Rerank1 nâng R@12 thêm +8.9pp
nhưng chỉ có thể re-rank những gì dense + sparse đã surface. **Dense
tốt hơn → rerank pool tốt hơn → final tốt hơn.**

200 câu BHXH viết kiểu kể chuyện đời thường (`"Bà Minh Châu (Long An)
ký hợp đồng lao động theo diện làm việc bán thời gian…"`) trong khi
clause KG là văn bản pháp luật trang trọng (`"Người lao động làm việc
theo hợp đồng lao động không xác định thời hạn…"`). HyDE được thiết
kế đúng cho style-gap này: thay vì embed câu hỏi, sinh một đoạn văn
bản pháp luật giả định *sẽ* trả lời nó, rồi embed + search bằng đoạn
đó. Hypothetical doc nằm gần clause thật hơn trong không gian
embedding.

## Setup

- **Dataset**: 50 câu stratified, seed=0 — list persisted ở
  [`pilot_50_stt.json`](pilot_50_stt.json) (sinh tự động lần đầu chạy).
- **Generator**: `gpt-4o-mini` (constructor-overridable), N=1,
  max_tokens=700, temperature=0, concurrency=5 (asyncio.Semaphore).
- **Prompt**: [`prompts/runtime/hyde_generate.md`](../../prompts/runtime/hyde_generate.md)
  — Vietnamese system+user, target 200–400 từ, **cấm tuyệt đối**
  "Điều X", "Khoản Y", tên người, ngày tháng cụ thể.
- **Encoder + index**: `models/bge-m3-bhxh-lora` + `clause_vec_tuned`.
- **Sparse**: Lucene FULLTEXT `clause_fulltext` (chỉ dùng cho 2 arm
  có sparse).
- **Reranker**: `BAAI/bge-reranker-base`.
- **Pipeline scaled** như exp 07: dense_k=100, sparse_k=100,
  top_after_fusion=150, rerank1_top_k=50, per_seed_neighbors=15,
  rerank2_top_k=100.
- **Runner**: [`scripts/exp08_run.py --pilot-50`](../../scripts/exp08_run.py).
- **Metrics**: [`scripts/exp08_metrics.py`](../../scripts/exp08_metrics.py)
  (auto-filter `pilot_50_stt.json` khi có).
- **Funnel**: [`scripts/exp08_funnel.py`](../../scripts/exp08_funnel.py)
  (auto-filter tương tự).
- **Dry-run gate**: [`scripts/exp08_test_one.py`](../../scripts/exp08_test_one.py)
  — chạy 1 câu, in hypothetical doc + cost + cache-hit confirmation.
- **Cache**: `artifacts/hyde/openai__gpt-4o-mini/<sha256>.json` — schema
  lưu model_returned + prompt_sha + usage + cost_usd.

## Cost estimate (pre-flight)

Per call (gpt-4o-mini, no cache hit, default config):
- Input ≈ 850 tokens × $0.15/M = $0.000128
- Output ≈ 600 tokens × $0.60/M = $0.00036
- **≈ $0.0005/call**

Pilot 50: ~$0.025. Re-runs = $0 (cache hit).

Pre-flight cap = $0.50 (runner aborts before any spend nếu estimate
vượt cap — safety net cho future config drift).

## Expected outcome

HyDE coi là **thắng** ở pilot N=50 (indicative, không conclusive) nếu
thoả ít nhất 1 trong 3:

- `dense_hyde` nâng R@12 in_corpus **≥ +3pp tuyệt đối** so với `dense`.
- `dense_hyde` nâng NDCG@10 in_corpus **≥ +5% tương đối** so với `dense`.
- `full_rerank_hyde` nâng R-Precision in_corpus **≥ +15% tương đối**
  so với `full_rerank`.

**Strong signal trên pilot 50** → scale lên full 200, sau đó xem xét
ADR 002.
**Weak / null signal** (lift trong noise ± 1pp) → dừng, viết kết quả
âm vào "Result summary" bên dưới, KHÔNG tốn $0.10 cho full run.
**Pilot cost > $0.10** → bất thường, debug prompt/max_tokens trước khi
mở rộng.

## Risks

| risk | mức | mitigation |
|---|---|---|
| gpt-4o-mini Vietnamese chưa đủ chất / bịa "Điều X" | low–medium | Phase 3 dry-run gate; prompt cấm tuyệt đối + ví dụ negative |
| Cost overrun bất ngờ | low | --cost-cap + cache + cost summary cuối run |
| LoRA-tuned BGE-M3 train Q→clause, doc-style HyDE có thể underperform | medium | Pilot 50 đủ để thấy lift hay không; nếu null thì ablation `clause_vec` vanilla là next step |
| Pilot N=50 high variance, conclusion sai về full 200 | inherent | Stratified sample giữ tỉ lệ stratum; 3 criteria + magnitude check trước khi tuyên thắng |

## Result summary

Chạy ngày 2026-05-31. Pilot 50 stratified (in_corpus=38, mixed=1,
ooc=2, unparseable=9). 4/4 arms 0 failures. Wall time 228.6s.

Artifacts:
- [metrics/academic_metrics.json](metrics/academic_metrics.json)
- [metrics/academic_metrics.csv](metrics/academic_metrics.csv)
- [metrics/gold_citations_normalized.json](metrics/gold_citations_normalized.json)
- [report/academic_report.md](report/academic_report.md)
- [report/funnel_full_rerank_hyde_K12.md](report/funnel_full_rerank_hyde_K12.md)
- [pilot_50_stt.json](pilot_50_stt.json) — selection list (seed=0)

### HyDE LLM cost (gpt-4o-mini)

| | value |
|---|---:|
| Pilot 50 run | **$0.00** (toàn bộ cache hit từ prewarm trước đó) |
| Cumulative session (dry-run + prewarm + run) | **$0.0122** |
| Plan estimate | $0.025 — dưới hẳn |
| API snapshot returned | `gpt-4o-mini-2024-07-18` |
| Cache dir | `artifacts/hyde/openai__gpt-4o-mini/` (51 entries) |

### Success criteria verdict (in_corpus n=38)

| # | Criterion | Threshold | Result | Verdict |
|---|---|---:|---:|:---:|
| 1 | `dense_hyde` R@12 − `dense` R@12 (abs) | +0.030 | **+0.1053** | ✅ **PASS** (3.5× margin) |
| 2 | `dense_hyde` NDCG@12 / `dense` NDCG@12 − 1 | +5.0% rel | **+35.2%** | ✅ **PASS** (7× margin) |
| 3 | `full_hyde` R-Prec / `full_rerank` R-Prec − 1 | +15.0% rel | **−0.5%** | ❌ FAIL |

**2/3 criteria pass strongly → HyDE thắng ở pilot.**

### In-corpus stratum (n=38) — full table

| metric | dense | dense_hyde | Δrel | full_rerank | full_rerank_hyde | Δrel |
|---|---:|---:|---:|---:|---:|---:|
| R@12 | 0.4154 | **0.5207** | **+25.3%** | 0.4248 | 0.4110 | −3.2% |
| R@30 | 0.5207 | 0.5514 | +5.9% | 0.5338 | 0.5376 | +0.7% |
| R@100 | 0.7224 | 0.7005 | −3.0% | 0.5683 | 0.6297 | +10.8% |
| P@12 | 0.0461 | 0.0592 | +28.4% | 0.0482 | 0.0504 | +4.6% |
| NDCG@12 | 0.2318 | **0.3134** | **+35.2%** | 0.2487 | 0.2485 | −0.1% |
| R-Prec | 0.0746 | **0.1479** | **+98.3%** | 0.1310 | 0.1303 | −0.5% |
| MRR | 0.2075 | **0.2781** | **+34.0%** | 0.2315 | 0.2472 | +6.8% |

### Funnel insight (full_rerank_hyde, K=12, in_corpus n=38)

| stage | avg pool | R@12 | NDCG@12 | MRR |
|---|---:|---:|---:|---:|
| dense (HyDE) | 47.8 | **0.521** | **0.313** | **0.278** |
| sparse (raw question) | 71.6 | 0.210 | 0.100 | 0.083 |
| dense ∪ sparse | 98.2 | 0.521 | — | — |
| post_temporal | 67.3 | 0.429 | — | — |
| fused (RRF) | 67.3 | 0.399 | 0.178 | 0.148 |
| rerank1 (top-50) | 31.3 | 0.411 | 0.249 | 0.247 |
| expanded | 40.1 | 0.411 | 0.249 | 0.248 |
| final (rerank2, top-100) | 40.1 | 0.411 | 0.249 | 0.247 |

Stage-to-stage gold delta (in_corpus): temporal mất 5 hits, rerank1 mất
4 hits, graph expansion cứu 4 hits, rerank2 net 0. **HyDE-augmented
dense bắt đầu ở 52% R@12 nhưng pipeline downstream ép xuống 41%** —
gần bằng `full_rerank` không HyDE.

### Findings

1. **HyDE thắng MẠNH ở kênh dense thuần** — R@12 in_corpus tăng từ
   41.5% → 52.1% (+10.5pp), NDCG@12 0.23 → 0.31 (+35%), R-Prec gần
   gấp đôi (0.075 → 0.148). 2/3 criteria pass with very large margins.

2. **HyDE bị "hấp thụ" bởi pipeline full_rerank** — chỉ R@100 còn lift
   +10.8%, mọi metric ở K ≤ 30 và rank-aware gần như không đổi. Lý do:
   cross-encoder reranker đã tự làm gần hết phần dense-quality lift.
   Funnel chứng minh: HyDE-dense in_corpus R@12=0.52 → sau temporal +
   rerank2 chỉ còn 0.41 (gần bằng non-HyDE full 0.42).

3. **Trade-off ở high-K**: `dense_hyde` R@100 thấp hơn `dense` vanilla
   (−3pp). HyDE doc-embedding chuyên ép top-results lên cao, đuôi K cao
   kém hơn raw-question encoding. **dense_hyde + dense union** có thể
   là arm tiếp theo đáng thử nếu cần cả top precision + high-K recall.

4. **Cost**: cumulative $0.0122 cho toàn bộ pilot 50 cycle (dry-run +
   prewarm + 4-arm). Cache hit toàn bộ trên run chính → mỗi lần re-run
   = $0. Full 200 estimate: ~$0.05.

### Quyết định scale full 200

**Strong signal → recommend scale full 200.** Lý do:
- 2/3 criteria pass với margin rất lớn (3.5× và 7×) — không phải noise.
- Stratum in_corpus (n=38) đã đủ thấy lift; full 200 in_corpus =151 sẽ
  cho thấy magnitude ổn định hơn, đặc biệt cho thesis chapter.
- Cost trivial (~$0.05).

Sau full 200 confirm:
- Draft `docs/decisions/002_hyde_retrieval.md`: mở `dense_hyde` thành
  arm tuỳ chọn bên cạnh `dense` hiện hữu (sau cờ config).
- KHÔNG mở `full_rerank_hyde` — cost LLM thêm không đổi metric.
- Next experiment idea (exp 09): `dense_hyde ∪ dense` union arm — lift
  cả top precision (HyDE) + high-K recall (vanilla).
