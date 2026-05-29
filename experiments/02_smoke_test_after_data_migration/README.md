# 02 — Smoke test sau khi migrate data/

## What
Smoke test 5 questions × 5 arms × gpt-4o-mini để verify hệ thống vẫn
chạy được sau đợt refactor folder `data/`:

- `data/raw/` → `data/graph/raw/`
- `data/interim/` → `data/graph/interim/`
- `data/processed/` → `data/graph/processed/`
- `data/logic_lm/` → `data/ontology/` (đổi tên)
- `data/logic_lm/generated/programs/` → `artifacts/logic_lm/programs/` (di chuyển ra khỏi `data/`)

Đây KHÔNG phải experiment khoa học — chỉ infrastructure verification.

## Why
Migration vừa rồi đụng ~15 file source + config + docs. Pytest pass nhưng
test chỉ phủ unit-level provenance / schema, không chạy E2E pipeline.
Cần verify LLM call thật, Prolog call thật, và ontology/graph KG load
đúng từ path mới.

## Setup
- 5 arms × 5 questions × gpt-4o-mini (single model, không multimodel)
- Dataset: 5 record đầu của `data/eval/questions_200.json`
- No prompt overrides
- Neo4j Aura phải up (cho `graphrag` + `logic_lm_graphrag`)
- SWI-Prolog 9.x trên PATH (cho `logic_lm_*` arms)

## Expected outcome
Pass criteria:
- Mỗi arm sinh đúng 5 record JSON dưới `results/<arm>/A*.json`
- `prompt_tokens > 0` cho mọi record (chứng minh LLM API call thật)
- `prolog_program` non-empty cho logic_lm arms (chứng minh Prolog call thật)
- `metrics/academic_metrics.json` + `report/academic_report.md` được sinh

Headline metrics không quan trọng (n=5 quá nhỏ); chỉ cần pipeline complete.

Fail criteria: any arm raises `FileNotFoundError` trỏ vào path cũ
`data/raw/`, `data/interim/`, `data/processed/`, hoặc `data/logic_lm/`.

## Result summary

**Pipeline complete: 25/25 records, 0 failures.** Migration pass verification.

Bug phát hiện trong lần chạy đầu: [runtime/logic_lm_pipelines.py:289](../../runtime/logic_lm_pipelines.py:289)
còn bare-import `from llm.client import _chunk_lines` — stale từ thời
`elite/` chưa được rename sang absolute path. Triệu chứng: 3 logic_lm
arms silent fail (`prompt_tokens=0, prolog_program=""`, status =
`unable_to_conclude`). Đã fix → `from runtime.logic_lm.llm.client import _chunk_lines`.

Sau khi fix, rerun thành công. Bằng chứng real calls:

| Arm | n | LLM tokens (range) | Prolog programs (chars) | Real call evidence |
|---|---:|---|---|---|
| graphrag | 5/5 | (not tracked in record) | n/a | elapsed 4–16s + Neo4j vector hits + citations |
| llm_only | 5/5 | prompt 255–286, completion 139–277 | n/a | Real OpenAI usage stats |
| logic_lm_no_retrieval | 5/5 | prompt 2.7K–4.3K | 134–845 chars | 3 success, 2 invalid_query (real Prolog validator fired) |
| logic_lm_ontology | 5/5 | prompt 3.6K–6.5K | 549–994 chars | 4 success, 1 syntax_error (real SWI-Prolog crashed on syntax) |
| logic_lm_graphrag | 5/5 | prompt 3.4K–5K | 429–2117 chars | 5/5 success, Prolog `step(conclusion, ..., based_on(source_c000))` trace |

Reports:
- [metrics/academic_metrics.json](metrics/academic_metrics.json)
- [report/academic_report.md](report/academic_report.md)

Headline (n=5 — không có ý nghĩa statistical):
- logic_lm_graphrag macro citation_f1 = 0.1333 (best), display_rate = 1.0
- BERTScore tất cả arms tính được trên CUDA / bert-base-multilingual-cased
- 0 BERTScore fail-soft (model load thành công)

Pipeline data migration → ✅ pass. Mọi path (`data/graph/raw`,
`data/graph/interim`, `data/graph/processed`, `data/ontology/`) resolve
đúng, không có FileNotFoundError trỏ vào path cũ.
