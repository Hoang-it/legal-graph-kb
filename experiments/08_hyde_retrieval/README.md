# 08 — HyDE retrieval (Qwen 2.5 3B Instruct on Colab Free T4)

## What

Test [HyDE (Hypothetical Document Embeddings, Gao et al. 2022)](https://arxiv.org/abs/2212.10496)
trên kênh dense BGE-M3 của `V5RetrievalPipeline`. Generator chạy local
trên Colab Free T4 với `Qwen/Qwen2.5-3B-Instruct` (fp16, N=1 doc/câu,
max_new_tokens=400). Plug-in point: **chỉ thay query embedding của dense
channel** — sparse channel giữ nguyên câu hỏi gốc (plan §D3, cô lập
đóng góp của HyDE về dense).

Bốn arms (đều retrieval-only, không LLM, không E2E):

| arm | method | mô tả |
|---|---|---|
| `dense` | `retrieve_dense_only` | BGE-M3 LoRA + `clause_vec_tuned`, dense_k=100 |
| `dense_hyde` | `retrieve_dense_only_hyde` | giống `dense`, nhưng query = embedding của HyDE doc |
| `full_rerank` | `retrieve_only` (no hyde) | full v5 scaled (giống exp 07) |
| `full_rerank_hyde` | `retrieve_only` (with hyde) | giống `full_rerank`, dense embed = HyDE doc |

## Why

Funnel `full_rerank` K=12 in_corpus (n=151) — xem
[`experiments/06_retrieval_dense_vs_full/report/funnel_full_rerank_K12.md`](../06_retrieval_dense_vs_full/report/funnel_full_rerank_K12.md)
— cho thấy dense là nguồn signal chủ đạo. Rerank1 nâng R@12 thêm +8.9pp
(đóng góp lớn nhất pipeline) nhưng chỉ có thể re-rank những gì dense +
sparse đã surface. **Dense tốt hơn → rerank pool tốt hơn → final tốt hơn.**

200 câu BHXH viết kiểu kể chuyện đời thường (`"Bà Minh Châu (Long An) ký
hợp đồng lao động theo diện làm việc bán thời gian…"`) trong khi clause
KG là văn bản pháp luật trang trọng (`"Người lao động làm việc theo hợp
đồng lao động không xác định thời hạn…"`). HyDE được thiết kế đúng cho
style-gap này: thay vì embed câu hỏi, sinh một đoạn văn bản pháp luật giả
định *sẽ* trả lời nó, rồi embed + search bằng đoạn đó. Hypothetical doc
nằm gần clause thật hơn trong không gian embedding.

## Setup

- Dataset: 200 câu (`data/eval/questions_200.json`).
- Generator: `Qwen/Qwen2.5-3B-Instruct`, dtype=fp16, N=1,
  max_new_tokens=400, batch_size=4.
- Prompt: [`prompts/runtime/hyde_generate.md`](../../prompts/runtime/hyde_generate.md)
  — Vietnamese system+user, target 200–400 từ, **cấm tuyệt đối** "Điều X",
  "Khoản Y", tên người, ngày tháng cụ thể (plan §D5).
- Encoder + index: `models/bge-m3-bhxh-lora` + `clause_vec_tuned`.
- Sparse: Lucene FULLTEXT `clause_fulltext` (chỉ dùng cho 2 arm có sparse).
- Reranker: `BAAI/bge-reranker-base`.
- Pipeline scaled như exp 07: dense_k=100, sparse_k=100,
  top_after_fusion=150, rerank1_top_k=50, per_seed_neighbors=15,
  rerank2_top_k=100.
- Runner: [`scripts/exp08_run.py`](../../scripts/exp08_run.py) (Colab).
- Metrics: [`scripts/exp08_metrics.py`](../../scripts/exp08_metrics.py).
- Funnel: [`scripts/exp08_funnel.py`](../../scripts/exp08_funnel.py).
- Notebook: [`notebooks/exp08_hyde_colab.ipynb`](../../notebooks/exp08_hyde_colab.ipynb).
- Cache: `artifacts/hyde/<model_id_safe>/<sha256>.json` — re-run = free,
  schema lưu model_revision + prompt_sha để audit.

## Expected outcome

HyDE coi là **thắng** nếu thoả ít nhất 1 trong 3 (plan §Success criteria):

- `dense_hyde` nâng R@12 in_corpus **≥ +3pp tuyệt đối** so với `dense`.
- `dense_hyde` nâng NDCG@10 in_corpus **≥ +5% tương đối** so với `dense`.
- `full_rerank_hyde` nâng R-Precision in_corpus **≥ +15% tương đối** so với
  `full_rerank` (bằng cỡ exp 06 thấy được từ rerank).

**Thắng → ADR 002**: HyDE-with-Qwen thành optional default sau cờ config,
cùng revisit triggers như ADR 001.

**Không thắng** (3 metric đều trong noise): viết kết quả âm vào README,
KHÔNG đổi production.

## Risks (xem plan §Risks)

| risk | mức | mitigation |
|---|---|---|
| Qwen 2.5 3B Vietnamese chưa đủ chất | medium | Phase 3 dry-run eyeball; thay 7B nếu cần |
| OOM T4 (BGE-M3 + reranker + Qwen 3B > 16 GB) | medium | Default fp16; fallback `dtype="4bit"` qua bitsandbytes |
| Qwen bịa "Điều X / Khoản Y" | medium | Prompt cấm tuyệt đối + Phase 3 gate |
| Qwen paraphrase câu hỏi thay vì sinh doc | medium | Prompt: bỏ tên / số liệu cá nhân + Phase 3 gate |
| LoRA-tuned BGE-M3 train Q→clause, doc-style HyDE có thể underperform | medium | Ablation tuỳ chọn: re-run HyDE với vanilla `clause_vec` |
| Colab disconnect mid-run | medium | Drive persistence + idempotent runner — lossless |

## Result summary

_(Điền sau khi chạy full 200 + metrics + funnel.)_

- Liên kết `metrics/academic_metrics.json` + `report/academic_report.md`.
- State arm nào thắng, magnitude, p-value nếu có.
- Verdict so với 3 success criteria ở trên.
- Ghi chú per-stratum (in_corpus / mixed / ooc / unparseable).
