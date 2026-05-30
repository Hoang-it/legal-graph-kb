# 07 — Retrieval-only A/B at extended K (K ∈ {12, 20, 30, 50, 70, 100, all})

## What

Mở rộng exp 06 sang K cao. Mỗi stage của full_rerank được scale up để arm
thực sự emit ≥100 articles/câu, nên K=70 và K=100 mới có ý nghĩa.

- **Arm `dense`**: BGE-M3 LoRA + `clause_vec_tuned`, **dense_k = 100**
  (so 50 ở exp 06). Article-dedup. Không sparse / temporal / rerank /
  expand.
- **Arm `full_rerank`** (scaled): dense_k = 100, sparse_k = 100,
  top_after_fusion = 150, rerank1_top_k = 50, per_seed_neighbors = 15,
  max_hops = 3, **rerank2_top_k = 100** (so 12 ở exp 06). Reranker
  giữ `BAAI/bge-reranker-base`. Temporal mode `strict_today_default`.

## Why

Exp 06 cho thấy `full_rerank` thắng ở K ≤ 12 nhưng bị cap recall ceiling
tại 12 articles. Câu hỏi mở: **nếu cho phép retrieve sâu hơn, full_rerank
có giữ lợi thế rank-quality (NDCG, MRR) đồng thời bắt kịp dense về recall
ceiling không?** Hay tới K cao thì hai arm hội tụ và rerank chỉ có giá trị
ở top?

Đồng thời cung cấp toàn bộ metric phổ thông (recall, precision, F1,
R-Precision, MRR, NDCG@K) ở K ∈ {12, 20, 30, 50, 70, 100, all} để có cái
nhìn đầy đủ cho thesis chapter retrieval analysis.

## Setup

- Dataset: 200 câu (`data/eval/questions_200.json`).
- Encoder + index: `models/bge-m3-bhxh-lora` + `clause_vec_tuned`.
- Sparse: Lucene FULLTEXT `clause_fulltext`.
- Runner: [`scripts/exp07_run.py`](../../scripts/exp07_run.py).
- Metrics: [`scripts/exp07_metrics.py`](../../scripts/exp07_metrics.py).
- Gold: re-derived in-place via `eval_core.gold.validate_gold_citations`.

## Expected outcome

- Ở K ≤ 30: `full_rerank` vẫn dẫn đầu R, P, F1, NDCG, R-Prec, MRR (carry-over
  từ exp 06).
- Ở K = 70, 100: hai arm hội tụ về recall. Precision của cả hai giảm mạnh
  (|gold|/K asymmetry).
- NDCG@K: `full_rerank` vẫn nhỉnh hơn ở mọi K vì reranker xếp hạng tốt hơn
  ngay cả khi pool lớn — discount log2(rank) penalty gold position thấp.
- Latency `full_rerank` tăng ~3-4× so exp 06 (rerank pool to hơn).

## Result summary

Chạy ngày 2026-05-30, 200/200 câu, 0 failure. dense ~0.15s/câu,
full_rerank ~2.87s/câu (chậm hơn exp 06 ~2.4× do rerank pool 100 docs).

Artifacts:
- [metrics/academic_metrics.json](metrics/academic_metrics.json)
- [metrics/academic_metrics.csv](metrics/academic_metrics.csv)
- [metrics/gold_citations_normalized.json](metrics/gold_citations_normalized.json)
- [report/academic_report.md](report/academic_report.md)

### Overall macro (n=200)

| metric | K=12 | K=20 | K=30 | K=50 | K=70 | K=100 | all |
|---|---:|---:|---:|---:|---:|---:|---:|
| **R — dense** | 0.317 | 0.383 | 0.435 | 0.501 | **0.532** | **0.532** | **0.532** |
| **R — full_rerank** | **0.353** | **0.417** | **0.462** | **0.504** | 0.508 | 0.508 | 0.508 |
| **P — dense** | 0.040 | 0.030 | 0.023 | 0.017 | 0.016 | 0.016 | 0.016 |
| **P — full_rerank** | **0.047** | **0.033** | **0.025** | **0.022** | **0.022** | **0.022** | **0.022** |
| **F1 — dense** | 0.068 | 0.054 | 0.042 | 0.033 | 0.032 | 0.032 | 0.032 |
| **F1 — full_rerank** | **0.080** | **0.060** | **0.047** | **0.042** | **0.042** | **0.042** | **0.042** |
| **NDCG — dense** | 0.188 | 0.206 | 0.219 | 0.233 | 0.239 | 0.239 | 0.239 |
| **NDCG — full_rerank** | **0.210** | **0.228** | **0.238** | **0.248** | **0.249** | **0.249** | **0.249** |

K-independent rank-aware:

| arm | R-Precision | MRR | latency |
|---|---:|---:|---:|
| dense | 0.068 | 0.186 | 0.15s |
| **full_rerank** | **0.092** | **0.202** | 2.87s |

### In-corpus stratum (n=151)

| metric | K=12 | K=20 | K=30 | K=50 | K=70 | K=100 |
|---|---:|---:|---:|---:|---:|---:|
| **R — dense** | 0.383 | 0.468 | 0.531 | 0.618 | **0.659** | **0.659** |
| **R — full_rerank** | **0.442** | **0.524** | **0.583** | **0.634** | 0.639 | 0.639 |
| **NDCG — dense** | 0.219 | 0.243 | 0.257 | 0.276 | 0.284 | 0.284 |
| **NDCG — full_rerank** | **0.264** | **0.286** | **0.300** | **0.312** | **0.313** | **0.313** |
| **R-Prec — dense** | 0.064 | | | | | |
| **R-Prec — full_rerank** | **0.121** | | | | | |
| **MRR — dense** | 0.212 | | | | | |
| **MRR — full_rerank** | **0.251** | | | | | |

### Findings

1. **`full_rerank` thắng ở K ≤ 50 cho mọi metric** (R, P, F1, NDCG). Trên
   in_corpus K=30: NDCG +17% rel, R +10% rel, F1 +12% rel. Hiệu ứng
   rerank rõ nhất ở K nhỏ.

2. **Crossover ở K ≈ 50-70**: dense bắt đầu vượt full_rerank về RECALL.
   K=70 overall: dense 0.532 vs full 0.508 (+4.7% rel cho dense). K=70
   in_corpus: dense 0.659 vs full 0.639. Lý do:
   - Dense pool=100 → có nhiều cơ hội bắt gold ở sâu.
   - full_rerank cap ở 100 articles thực, nhưng rerank2 ưu tiên chất
     lượng top → các candidate "đúng nhưng score thấp" bị cắt bớt; sau
     temporal filter còn lại ~37-42 final → mất gold luật cũ.

3. **NDCG `full_rerank` vẫn dẫn ở MỌI K** kể cả khi recall đã thua.
   K=100 overall: NDCG full 0.249 vs dense 0.239. Có nghĩa rerank xếp
   hạng tốt hơn ngay cả khi không retrieve đủ — gold được full_rerank
   đặt ở rank thấp hơn (sớm hơn) trung bình.

4. **Precision sụp đổ theo `|gold|/K`** đúng như dự đoán. P@100 chỉ
   0.022. Không có ý nghĩa benchmark standalone — chỉ để confirm tỉ lệ.

5. **R-Precision in_corpus full_rerank = 0.121 vs dense 0.064** (gấp
   1.9×). Đây là phương sai ổn định nhất: full_rerank đặt 12% gold đúng
   vào top-|gold| positions, dense chỉ 6%.

6. **Latency**: dense 0.15s (rẻ tiền), full_rerank 2.87s. Tỉ số 19×.
   Khi cần K cao + ưu tiên throughput → dense pool=100 (rẻ + recall@70
   nhỉnh hơn).

### Verdict tổng hợp (so với prediction)

| Prediction | Result | Verdict |
|---|---|---|
| K ≤ 30: full_rerank dẫn đầu mọi metric | đúng | ✓ |
| K=70, 100: 2 arm hội tụ về recall | dense vượt full @K=70 | sai chiều — dense thắng nhẹ |
| NDCG: full_rerank vẫn nhỉnh ở mọi K | đúng | ✓ |
| Latency full ~3-4× exp 06 | 2.4× thực tế | ✓ gần đúng |

### Next steps gợi ý

- **Hybrid combine arm** mới: dense top-100 ∪ full_rerank top-30 → có thể
  hưởng lợi cả recall ceiling của dense + ranking quality của full.
- **`temporal_mode=skip_when_no_date`** ablation: confirm temporal filter
  đang chặn gold luật cũ ở full_rerank K cao.
- **NDCG@K vs E2E citation recall** correlation analysis trên các
  experiments e2e — kiểm tra NDCG có phải proxy tốt cho downstream LLM
  citing không.
