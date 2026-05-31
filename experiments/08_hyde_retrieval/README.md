# 08 — HyDE retrieval (gpt-4o-mini, pilot 50 + full 200)

## What

Test [HyDE (Hypothetical Document Embeddings, Gao et al. 2022)](https://arxiv.org/abs/2212.10496)
trên kênh dense BGE-M3 của `V5RetrievalPipeline`. Generator: OpenAI
`gpt-4o-mini` (N=1 doc/câu, max_tokens=700, temperature=0). Plug-in
point: **chỉ thay query embedding của dense channel** — sparse channel
giữ nguyên câu hỏi gốc (plan §D4).

**Scope (đã mở rộng)**:

- **Pilot 50** stratified (seed=0): cost + lift gate trước khi scale.
  Numbers committed ở commit `098d32d`, giữ nguyên dưới phần
  "Pilot 50 result summary" để so sánh stability.
- **Full 200** (idempotent over pilot 50): chạy ngày 2026-05-31 sau khi
  pilot đạt 2/3 success criteria với margin lớn. Numbers ở phần
  "Full 200 result summary" — đè vào `metrics/` + `report/`.

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

- **Dataset**:
  - Pilot 50 câu stratified, seed=0 — list persisted ở
    [`pilot_50_stt.json`](pilot_50_stt.json).
  - Full 200 chạy bằng `python scripts/exp08_run.py` (không cờ
    `--pilot-50`); idempotent: pilot records skip, 150 câu mới được
    tạo cho mỗi arm.
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
- **Runner**: [`scripts/exp08_run.py`](../../scripts/exp08_run.py)
  (`--pilot-50` cho pilot, không cờ cho full 200).
- **Metrics**: [`scripts/exp08_metrics.py`](../../scripts/exp08_metrics.py)
  — auto-filter theo `pilot_50_stt.json` nếu có; pass `--full` để
  override và score mọi record on-disk (dùng sau khi full 200 run xong).
- **Funnel**: [`scripts/exp08_funnel.py`](../../scripts/exp08_funnel.py)
  — cùng cờ `--full`.
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

## Pilot 50 result summary (commit `098d32d`, immutable history)

Chạy ngày 2026-05-31. Pilot 50 stratified (in_corpus=38, mixed=1,
ooc=2, unparseable=9). 4/4 arms 0 failures. Wall time 228.6s.

Pilot artifacts đã bị overwrite bởi full 200 run trong `metrics/` +
`report/`. Numbers dưới đây giữ lại snapshot commit `098d32d` để
audit stability pilot→full.

Selection list (seed=0): [pilot_50_stt.json](pilot_50_stt.json).

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

### Quyết định scale full 200 (đã thực hiện)

**Strong signal → scale full 200.** Lý do tại pilot:
- 2/3 criteria pass với margin rất lớn (3.5× và 7×) — không phải noise.
- Stratum in_corpus (n=38) đã đủ thấy lift; full 200 in_corpus =151 sẽ
  cho thấy magnitude ổn định hơn, đặc biệt cho thesis chapter.
- Cost trivial (~$0.05).

Full 200 đã chạy → kết quả ở phần "Full 200 result summary" bên dưới.

## Full 200 result summary

Chạy ngày 2026-05-31 (cùng session với pilot, ngay sau task #21).
Idempotent: 50 pilot records on disk → skip; chạy thêm 150 câu × 4
arms = 600 retrievals. **4/4 arms, 0 failures.**

Strata: in_corpus=151, mixed=5, ooc=8, unparseable=36 (tỉ lệ matching
proportion pilot).

| | value |
|---|---:|
| Wall time | **654.4s** (10.9 min) |
| HyDE API calls (cold, run này) | 149 |
| HyDE cache hits (run này) | 301 |
| HyDE cost (run này) | **$0.0355** |
| Cumulative cache size sau run | 200 entries |
| Plan estimate cost | $0.075 — về dưới hẳn (shared cache hit cao hơn dự đoán) |
| Latency dense | 0.057s |
| Latency dense_hyde | 0.075s (HyDE call cache-hit) |
| Latency full_rerank | 2.128s |
| Latency full_rerank_hyde | 2.147s |

Artifacts (đè vào pilot, vẫn đường dẫn cũ):
- [metrics/academic_metrics.json](metrics/academic_metrics.json) (n=200)
- [metrics/academic_metrics.csv](metrics/academic_metrics.csv)
- [report/academic_report.md](report/academic_report.md)
- [report/funnel_full_rerank_hyde_K12.md](report/funnel_full_rerank_hyde_K12.md)

### Success criteria verdict (in_corpus n=151)

| # | Criterion | Threshold | Pilot 50 | **Full 200** | Verdict |
|---|---|---:|---:|---:|:---:|
| 1 | `dense_hyde` R@12 − `dense` R@12 (abs) | +0.030 | +0.1053 | **+0.0904** | ✅ **PASS** (3.0× margin) |
| 2 | `dense_hyde` NDCG@12 / `dense` NDCG@12 − 1 | +5.0% rel | +35.2% | **+34.7%** | ✅ **PASS** (~7× margin) |
| 3 | `full_hyde` R-Prec / `full_rerank` R-Prec − 1 | +15.0% rel | −0.5% | **−0.6%** | ❌ FAIL |

**Stability: pilot magnitude giữ vững ở full N.** C1 hơi yếu hơn pilot
(0.105 → 0.090) — vẫn cách threshold 3.0× — nhất quán với việc
in_corpus n từ 38 → 151 làm noise giảm, không phải HyDE yếu đi.

### In-corpus stratum (n=151) — full 200 numbers

| metric | dense | dense_hyde | Δrel | full_rerank | full_rerank_hyde | Δrel |
|---|---:|---:|---:|---:|---:|---:|
| R@12 | 0.3832 | **0.4736** | **+23.6%** | 0.4419 | 0.4241 | −4.0% |
| R@30 | 0.5311 | **0.6066** | **+14.2%** | 0.5830 | 0.5861 | +0.5% |
| R@100 | 0.6592 | **0.7016** | **+6.4%** | 0.6388 | 0.6683 | +4.6% |
| P@12 | 0.0486 | **0.0607** | **+24.9%** | 0.0585 | 0.0568 | −2.9% |
| NDCG@12 | 0.2186 | **0.2944** | **+34.7%** | 0.2640 | 0.2560 | −3.0% |
| R-Prec | 0.0635 | **0.1326** | **+108.8%** | 0.1207 | 0.1200 | −0.6% |
| MRR | 0.2122 | **0.2843** | **+34.0%** | 0.2507 | 0.2486 | −0.8% |

Đáng chú ý: ở pilot, `dense_hyde` R@100 thấp hơn `dense` vanilla
(−3pp). **Ở full 200, R@100 đảo dấu thành +6.4%.** Đây là phát hiện mới
— pilot 50 n=38 không đủ stable để kết luận về trade-off high-K. Lý
do giả thuyết: HyDE không chỉ ép top-K cao mà còn nâng đuôi K
moderate (30–100), pilot N nhỏ làm điều này invisible.

### Funnel insight (full_rerank_hyde, K=12, in_corpus n=151)

Pre-flight: dense channel HyDE-augmented bắt đầu ở R@12=0.465. Sau
toàn pipeline xuống 0.424 — cùng pattern absorb như pilot.

| stage | avg pool | R@12 | NDCG@12 | MRR |
|---|---:|---:|---:|---:|
| dense (HyDE) | 46.54 | **0.4648** | **0.2846** | **0.2708** |
| sparse (raw question) | (xem report) | 0.2118 | 0.0971 | 0.0860 |
| dense ∪ sparse | 91.06 | 0.4648 | — | — |
| post_temporal | 61.31 | 0.4927 | — | — |
| fused (RRF) | 61.30 | 0.4592 | 0.2594 | 0.2480 |
| rerank1 (top-50) | 29.92 | 0.4241 | 0.2562 | 0.2482 |
| expanded | 37.86 | 0.4241 | 0.2562 | 0.2491 |
| final (rerank2, top-100) | 37.86 | 0.4241 | 0.2560 | 0.2486 |

Stage-to-stage gold delta (in_corpus): temporal mất ~18 hits overall
(toàn 200 còn 181/199 = 91%), rerank1 mất ~22 hits, graph expansion
cứu lại +18, rerank2 net 0. Cùng shape pilot — không có regression
mới.

### Findings on full 200

1. **HyDE thắng MẠNH ổn định ở kênh dense thuần.** Cả 3 metric chủ
   chốt (R@12, NDCG@12, R-Prec, MRR) đều giữ lift +23–109% trên N
   3× lớn hơn pilot. Đây là evidence cấp **thesis-ready**.
2. **HyDE bị "hấp thụ" bởi pipeline full_rerank** — y hệt pilot.
   `full_rerank_hyde` thực tế **kém hơn nhẹ** `full_rerank` ở mọi
   metric K ≤ 30 (−0.6% đến −4%), chứ không phải neutral như pilot
   gợi ý. **Khẳng định: không nên ship HyDE trên nhánh rerank.**
3. **R@100 trade-off đảo dấu** — pilot dự báo `dense_hyde` R@100 kém
   hơn `dense` (−3pp); full 200 cho thấy thực tế là **+6.4%**. Pilot
   N=38 quá nhỏ để phát hiện điều này. Bài học: dùng pilot để gate
   "có lift hay không", đừng dùng để dự báo signed-trade-off ở K
   biên.
4. **Cost ngoài kỳ vọng theo hướng tốt**: kế hoạch dự $0.075 cho 150
   câu mới × 1 LLM call, thực tế $0.0355 vì 2 arm HyDE share cache
   cùng prompt_sha → 1 call cho 2 arm. Cache hit ratio = 301/(149+
   301) = 67%.

### Cập nhật khuyến nghị (thay block "Sau full 200 confirm" trước đây)

- **`dense_hyde` đủ evidence để promote thành arm chính thức**: tạo
  `docs/decisions/002_hyde_retrieval.md` (ADR) khuyến nghị mở
  `dense_hyde` thành option config-flag cạnh `dense`. Cost LLM:
  ~$0.0005/câu sau khi cache nguội, $0/câu khi cache hit.
- **`full_rerank_hyde` đã rõ là LOSE** (chứ không phải neutral) — KHÔNG
  ship. Không cần thêm experiment đặc biệt cho arm này.
- **Exp 09 candidate** vẫn còn giá trị: `dense_hyde ∪ dense` union
  hoặc `dense_hyde` làm seed-set cho rerank thay vì query encoding —
  hỏi liệu có cách nào dùng HyDE-lift mà KHÔNG bị reranker hấp thụ.
  Nhưng đây không phải gate cho thesis.
- **Item B kế tiếp** (E2E HyDE arm): full 200 retrieval-only không
  thay thế E2E. Cần `dense_hyde_e2e` arm chạy `V5RetrievalPipeline.ask`
  với LLM generator để đo citation recall/precision dưới
  `academic_v2`. Đó là số chủ chốt cho thesis chapter — không phải
  retrieval R@12.
