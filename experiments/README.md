# Experiments — GraphRAG vs LLM-only

So sánh **GraphRAG** (Neo4j + BGE-M3 + GPT-4o-mini) với **LLM thuần** (chỉ GPT-4o-mini, không context) trên 200 câu hỏi BHXH thu thập từ FB group.

## Dataset

`data/eval/questions_200.json` — 200 câu đầu từ sheet `Tiền xử lý` của file
`_Câu hỏi BHXH trên group facebook.xlsx`. Cấu trúc mỗi câu:
- `stt`, `question`, `group`
- `gold_answer` (198/200 có)
- `gold_citations_raw` (196/200 có) — đa số chỉ Luật BHXH 2024 (kế thừa Luật 58/2014 ở vài điểm)

**Lưu ý version**: KG build từ Luật **41/2024/QH15** mới. ~151/200 câu refer Luật 2024 (khớp KG), 2 câu refer Luật 2014 cũ, còn lại không rõ. Sẽ tag mỗi câu + breakdown trong report.

## Pipeline

```
inference (eval_core.inference)
   ├── arm A: GraphRAG  → runtime.rag_query.RagPipeline.ask()
   └── arm B: LLM-only  → runtime.llm_only.LlmOnlyPipeline.ask()
                          (cùng SYSTEM prompt yêu cầu citation, KHÔNG có context)

→ data/eval/results/{graphrag,llm_only}/A{stt}.json

validate_gold_citations (eval_core/gold.py)
   └── Parse gold_citations_raw strict theo registry; fail nếu thiếu/sai

eval_core.runners (multi-arm orchestrator)
   ├── Load result folders + selected arms
   ├── Attach validated gold_articles and group records by experiment arm
   └── Call eval_core.metrics.compute_academic_metrics(records) per experiment arm

eval_core.metrics.compute_academic_metrics(records)
   ├── Dataset-based: citation_recall, citation_precision, citation_f1
   ├── Answer display: citation_display_rate từ citation_ids vs answer
   ├── Objective: latency_s, prolog_first_try_solution_rate,
   │   repair_invoked_rate, repair_success_rate
   └── Semantic: BERTScore vs gold_answer (fail-soft nếu thiếu dependency/model)

→ metrics/academic_metrics.json + metrics/academic_metrics.csv
→ metrics/academic_report.md
→ metrics/academic/gold_citations_normalized.json
```

## Metrics hiện tại

Headline metrics không dùng judge model. Citation recall/precision/F1 so sánh
`record["citation_ids"]` với `gold_citations_raw` sau khi parse strict theo
article-level authority. `citation_display_rate` đo citation ID nào trong
pipeline thật sự được thể hiện trong answer với đủ văn bản + điều/khoản.
BERTScore giữ chuẩn Zhang et al. ICLR 2020 và chạy fail-soft.

Judge-model metrics không chạy trong main experiment. Entrypoint
`eval_core.judge` hiện fail-closed để tránh vô tình dùng lại
công thức cũ; khi cần judge metrics sẽ thiết kế rubric riêng.

## Giới hạn (caveats — sẽ ghi rõ trong report)

1. **Gold citation quality**: `gold_citations_raw` là source of truth; validator sẽ dừng nếu thiếu/sai hoặc không parse được.
2. **Law/source mismatch**: predicted citation phải đúng văn bản và đúng điều. Khác văn bản là sai hoàn toàn.
3. **BERTScore**: chỉ là semantic reference vs `gold_answer`; headline citation metrics vẫn lấy từ dataset gold.
4. **Judge metrics**: không nằm trong main workflow và không được tính ngầm.

## Chạy

```powershell
# 1. Inference main experiment arms
python -m eval_core.inference --arms main --n 200

# 2. Validate gold citations; lệnh experiment metrics cũng tự gọi bước này
python -m eval_core.gold

# 3. Compute deterministic academic metrics for the same main arms
python -m eval_core.runners --arms main
```

Đổi thư mục lưu toàn bộ artifact bằng `--output-dir <folder>`.

Pilot trước với `--n 10` để xác nhận pipeline.

## Tính trung thực

- Metrics hiện tại dùng registry citation chung trong `data/legal_sources.yaml` và
  parser chung `src/citations.py`; không hardcode authority trong từng script.
- Mọi LLM call thật → API.
- Mọi metric headline tính từ output thật + dataset gold; judge metrics tách riêng.
- Per-question result lưu đầy đủ (question, answer, citations) → reproducible/auditable.
