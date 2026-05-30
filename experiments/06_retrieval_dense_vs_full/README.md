# 06 — Retrieval-only A/B: dense vs full (rerank+expand) on 200 questions

## What

Đo **citation recall + citation precision + F1 ở mức article** trên 200 câu
trong `data/eval/questions_200.json` cho hai cấu hình retrieval-only:

- **Arm `dense`** — chỉ BGE-M3 LoRA dense (Neo4j vector `clause_vec_tuned`),
  `dense_k=50`, article-dedupe; không sparse / không temporal / không RRF /
  không rerank / không graph hop.
- **Arm `full_rerank`** — pipeline production hiện tại (M2): dense + Lucene
  BM25 sparse + temporal filter + RRF + bge-reranker-base rerank (top-15
  seed) + REFERS_TO 1..3-hop expand + rerank pass 2 (top-12 final).

Không có call LLM, không E2E. Chỉ retrieve từ graph.

## Why

Mỗi tầng trên top của dense (sparse, temporal, RRF, rerank, graph expand)
đều phát sinh latency và phụ thuộc. Cần một bài đo **mức retrieval thuần**
trên đủ kích cỡ (200 câu) để biết:

1. Sparse + RRF + rerank + graph expand thực sự nâng recall/precision bao
   nhiêu so với dense-only?
2. Tradeoff recall vs precision khác nhau ở K nào?

Experiment 05 đã đo retrieval audit nhưng chỉ trên 30-probe + dùng records
inference đã có; đây là run retrieval-only chuyên dụng trên full 200, không
kế thừa, không workaround.

## Setup

- Dataset: 200 câu (`data/eval/questions_200.json`).
- Encoder + dense index: `models/bge-m3-bhxh-lora` + `clause_vec_tuned`
  (production setup hiện tại).
- Sparse: Lucene FULLTEXT `clause_fulltext`.
- Reranker (arm `full_rerank` only): `BAAI/bge-reranker-base`.
- Temporal mode: `strict_today_default` (dùng ngày trong câu, nếu không có
  thì today; filter các Law không hiệu lực tại ngày đó).
- Runner: [`scripts/exp06_run.py`](../../scripts/exp06_run.py).
- Metric script: [`scripts/exp06_metrics.py`](../../scripts/exp06_metrics.py).
- Gold: re-derived in-place via `eval_core.gold.validate_gold_citations`
  vào `metrics/gold_citations_normalized.json` (self-contained, không lệ
  thuộc experiment trước).

## Expected outcome

Predictions trước khi chạy (để chống post-hoc rationalization):

- `full_rerank` sẽ thắng `dense` ở recall@12 với Δ ≥ +5pp (sparse + RRF
  nhặt thêm các văn bản đồng nghĩa mà dense miss; reranker đẩy gold lên
  cao hơn).
- Precision@K=12: `full_rerank` ≥ `dense` nhờ rerank loại nhiễu — Δ ≥ +5pp.
- Ở K rộng (K=30, all): hai arm sẽ tiến gần nhau (sparse + temporal đã
  bù phần lớn miss của dense, graph expand chỉ thêm ở K cao).
- OOC stratum (gold ngoài KG): cả 2 arm = 0 recall (không có cách nào
  retrieve được).

Threshold đổi kết luận:

- Nếu `full_rerank` không nhỉnh hơn `dense` quá +2pp recall@12 trên
  in_corpus stratum → toàn bộ stack sparse+RRF+rerank+expand không
  worth latency overhead → cần rethink Sprint 3 plan.
- Nếu precision@12 của `full_rerank` < `dense` → reranker đang đưa noise
  lên top (sai kỳ vọng).

## Result summary

Chạy ngày 2026-05-30, 200 / 200 câu, 0 failure, wall time 254.5s
(dense ~0.16s/câu, full_rerank ~1.20s/câu).

Artifacts:
- [metrics/academic_metrics.json](metrics/academic_metrics.json)
- [metrics/academic_metrics.csv](metrics/academic_metrics.csv)
- [metrics/gold_citations_normalized.json](metrics/gold_citations_normalized.json)
- [report/academic_report.md](report/academic_report.md)

### Headline (overall macro, n=200)

| arm | R@5 | R@10 | R@12 | R@all | P@12 | F1@12 | avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense (50-pool) | 0.2169 | 0.2902 | 0.3166 | **0.4312** | 0.0400 | 0.0685 | 0.16s |
| **full_rerank** (top-12) | **0.2488** | **0.3556** | **0.3568** | 0.3568 | **0.0627** | **0.1021** | 1.20s |

**Rank-aware (không bị cap bởi |gold|/K asymmetry)**:

| arm | R-Precision | MRR | NDCG@10 | NDCG@all |
|---|---:|---:|---:|---:|
| dense | 0.068 | 0.185 | 0.180 | 0.219 |
| **full_rerank** | **0.102** | **0.214** | **0.223** | **0.223** |

### In-corpus stratum (n=151) — phần fair-compare nhất

| arm | R@12 | R@all | P@12 | F1@12 | R-Prec | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.383 | **0.526** | 0.049 | 0.083 | 0.064 | 0.210 | 0.210 |
| **full_rerank** | **0.447** | 0.447 | **0.077** | **0.126** | **0.129** | **0.265** | **0.278** |

### Findings

1. **`full_rerank` thắng ở mọi K ≤ 12 cho cả recall, precision, F1.**
   F1@12 in_corpus = 0.126 vs 0.083 (rel +52%). Sparse + RRF + rerank2
   tăng đậm cả precision và recall khi K nhỏ — chứng tỏ rerank đang đẩy
   gold lên đầu và loại bớt nhiễu. **Rank-aware confirm**: NDCG@10
   in_corpus 0.278 vs 0.210 (rel +33%), R-Precision 0.129 vs 0.064
   (rel +103%) — rerank đẩy gold lên top mạnh nhất ở positions đầu.

2. **Dense có recall ceiling cao hơn (R@all)** — 0.431 vs 0.357 overall,
   0.526 vs 0.447 in_corpus. Lý do: `full_rerank` bị cap ở 12 article
   final, và temporal filter (`strict_today_default`) drop hết L58_2014
   ở những câu không có ngày → mất gold thuộc luật cũ. Dense không filter
   nên giữ.

3. **Latency 7.5× chậm hơn** (1.20s vs 0.16s) là cost của sparse +
   rerank + graph expand. Worth trade ở precision/F1, không worth ở
   recall ceiling.

4. **OOC** (8 câu): cả hai arm = 0 — không retrieve nổi luật ngoài
   corpus. Tương đương Stage A của experiment 05.

5. **Unparseable** (36 câu, đa số gold là L58_2014 hoặc Bộ luật Lao
   động không có alias chính xác trong registry): dense (R@12=0.125)
   nhỉnh hơn full_rerank (0.069) — confirm finding #2 về temporal
   filter ăn gold luật cũ.

### Prediction check

| Prediction (README before run) | Result | Verdict |
|---|---|---|
| full_rerank R@12 hơn dense ≥ +5pp | +3.2pp overall, +6.4pp in_corpus | partial — đúng on in_corpus |
| P@12 full_rerank ≥ dense ≥ +5pp | +2.3pp overall, +2.9pp in_corpus | dưới threshold |
| K rộng (@all) 2 arm gần nhau | dense BEATS full @all (0.43 vs 0.36) | sai — full bị cap 12 + temporal drop |
| OOC = 0 cả 2 | confirm | đúng |

### Next steps gợi ý

- Tăng `rerank2_top_k` của full_rerank lên 20-25 để recall@all bắt kịp
  dense, đo lại precision drop.
- Test `temporal_mode=skip_when_no_date` (đã có sẵn) — cứu được phần
  unparseable + L58_2014 gold mà không hy sinh date-specific queries.
- Tách finding #2 thành ablation: full_rerank không có temporal filter
  vs có — cần để biết temporal filter có đang gây hại nhiều hơn lợi
  trên dataset này không.
