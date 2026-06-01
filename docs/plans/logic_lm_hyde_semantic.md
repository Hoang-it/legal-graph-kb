# Plan — Arm `logic_lm_hyde_semantic`: feed HyDE-semantic hypothesis vào Logic-LM

> **Status (cập nhật 2026-06-01):** Backend arm ĐÃ implement + **smoke-verified** (1 câu,
> cả 2 arm, gọi thật Neo4j + BGE-M3 + OpenAI + SWI-Prolog) — danh sách file ở
> `docs/plans/ui_logic_lm_chatbot.md` §"Trạng thái hiện tại". Experiment folder
> `experiments/02_logic_lm_hyde_semantic/` đã tạo (config valid). CHƯA chạy eval (pilot/full).
> Khi chạy eval xong, WHAT/WHY + kết quả nằm trong
> `experiments/02_logic_lm_hyde_semantic/README.md` (plan này không chứa kết quả — Rule 6).

## Context (vì sao)

`dense_hyde_semantic` là retrieve chính, Logic-LM là tầng sinh. Hiện
`dense_hyde_semantic` sinh ra một **hypothesis** (đoạn "văn bản luật giả định"
200–400 từ, bám concept-frame BHXH) nhưng chỉ dùng để tính embedding rồi **vứt đi**.
Logic-LM (`_attempt` → rule-gen) chỉ nhận 2 input: `training_question` + `retrieved_chunks`.

Mục tiêu: cho bước **sinh Prolog** của Logic-LM nhận thêm **hypothesis** làm input thứ 3
(định hướng khái niệm/điều kiện/thuật ngữ để dựng rule), clause để cite vẫn đến từ
`dense_hyde_semantic`. ⇒ arm QA mới `logic_lm_hyde_semantic`, đo theo contract qa, cô lập
đóng góp của hypothesis bằng A/B với arm control không hypothesis.

## Tuân thủ luật skill (ràng buộc thiết kế)

- **Arm mới + experiment mới** — không sửa experiment cũ, không sửa 3 arm Logic-LM hiện có.
- **Backward-compatible**: thay đổi base Logic-LM là **no-op khi hypothesis rỗng** →
  `logic_lm_no_retrieval/ontology/graphrag` cho prompt byte-identical như cũ.
- **Không sửa** `prompts/runtime/logic_lm/rule_gen.md` — tạo biến thể mới.
- **Không đụng code shared** `runtime/logic_lm/` — điểm chèn hypothesis nằm trọn trong
  `_TokenTrackingLLMClient` ở `runtime/logic_lm_pipelines.py`.
- **Rule 5**: chỉ forecast cost, không dự đoán số; success criterion là decision-rule đặt trước.
- **Rule 6**: kết quả nằm trong `experiments/02_.../`; so sánh chéo bằng copy folder sang `experiments_repo/`.

## Điểm tích hợp đã xác minh

- Rule-gen user-message ráp ở `_TokenTrackingLLMClient._logic_with_tracking`
  (`runtime/logic_lm_pipelines.py:272`): `user_parts = [training_question_line,
  retrieved_chunks_header, chunk_lines]`. **Chỗ chèn hypothesis duy nhất.**
- `_attempt` shared (`runtime/logic_lm/pipelines/program_pipeline.py:85`) chỉ dựng payload
  `{question, chunks, [previous_*]}` rồi gọi `llm.generate` — **không** format message → không sửa.
  Hypothesis do client thêm ⇒ phủ cả lần đầu lẫn repair round.
- `_LogicLMBasePipeline.ask()` gọi `retrieve()` **trước** `_make_llm()` → khi tạo client,
  hypothesis (do retriever sinh) đã sẵn sàng.
- Adapter clause→chunk có mẫu sẵn: `runtime/graphrag_retriever_adapter.py`.
  `_dense_search_by_vector` trả rows đủ field `clause_id/text/score/article_id/article_n/clause_n/law_id`.
- `RetrievedKnowledgeChunk(id, text, document, article, clause, point)` —
  `runtime/logic_lm/knowledge/hybrid_retrieval.py`.
- `experiments/` chỉ có `_template` + `01_hyde_source_variants` → experiment mới là **`02`**;
  không có baseline qa trong repo để `inherit` → so sánh chéo qua `experiments_repo`.

---

## Các bước thực thi

### B1 — Lộ hypothesis + clause rows từ tầng retrieval
**File mới** `runtime/retrievers/dense_hyde_semantic_logic_adapter.py` —
class `DenseHydeSemanticAsLogicLMRetriever`:
- ctor: dựng `V5RetrievalPipeline(hyde_semantic=OpenAISemanticHydeGenerator(cache_dir=
  artifacts/logic_lm_hyde_semantic/hyde_semantic, ...))`, `ontology_path =
  data/ontology/ontology_kg_full.json`; warm BGE-M3.
- `retrieve(question, top_k) -> RetrievedKnowledgeContext`:
  1. `ctx = build_semantic_context(question, ontology_path)`
  2. `rows, docs = pipe.dense_hyde_semantic_rows(question, ctx.frame_text, ctx.context_key_ids, top_k)`;
     `self.last_hypothesis = docs[0]`, `self.last_semantic_context = ctx`.
  3. map rows → `RetrievedKnowledgeChunk(id=clause_id, text, document=pipe._law_display(law_id),
     article=str(article_n), clause=str(clause_n), point=None)`; `scores[clause_id]=score`.

**Sửa (additive)** `src/retrieval/pipeline.py` — thêm method public nhỏ
`dense_hyde_semantic_rows(question, frame_text, context_key_ids, top_k) -> (rows, docs)`
(cùng cơ chế `retrieve_dense_only_hyde_semantic` nhưng trả rows chưa dedupe + docs). Không sửa method cũ.

### B2 — Cho Logic-LM nhận hypothesis (backward-compatible)
**File** `runtime/logic_lm_pipelines.py`:
- `_TokenTrackingLLMClient.__init__(..., hypothesis: Optional[str] = None)` → lưu `self._hypothesis`.
- `_logic_with_tracking`: **sau** block chunks, nếu `self._hypothesis` → append header
  (`"# KHUNG PHÁP LÝ GIẢ ĐỊNH (chỉ định hướng — KHÔNG trích dẫn, KHÔNG lấy fact)"`) + hypothesis.
  (None → không thêm → arm cũ không đổi.)
- `_LogicLMBasePipeline`: hook `_rule_gen_hypothesis() -> str` (default `""`); `_make_llm()`
  truyền `hypothesis=(self._rule_gen_hypothesis() or None)`.
- `LogicLMAnswer`: thêm field `hypothesis: str = ""`; `ask()` set `hypothesis=self._rule_gen_hypothesis()`.
- **Class** `LogicLMHydeSemanticPipeline` (arm `logic_lm_hyde_semantic`): retriever=adapter,
  `prompt_override=load_prompt("runtime/logic_lm/rule_gen_hyde_semantic.md")`,
  `_rule_gen_hypothesis()` trả `retriever.last_hypothesis`.
- **Class control** `LogicLMHydeSemanticNoHypPipeline` (arm `logic_lm_hyde_semantic_nohyp`):
  cùng adapter (cùng retrieval+cache), không prompt_override, `_rule_gen_hypothesis()` trả `""`.

### B3 — Prompt mới (canonical)
**File mới** `prompts/runtime/logic_lm/rule_gen_hyde_semantic.md` — copy `rule_gen.md`, thêm
Input `hypothesis` + rule: *use hypothesis ONLY to choose predicates/conditions/terms; do NOT
extract any user fact, threshold, number, or citation from it; thresholds & citations vẫn chỉ
từ `retrieved_chunks`*. IRAC render KHÔNG nhận hypothesis.

### B4 — Đăng ký arm
- `eval_core/arms.py`: thêm `"logic_lm_hyde_semantic"`, `"logic_lm_hyde_semantic_nohyp"` vào
  `ALL_ARMS` (KHÔNG vào `MAIN_EXPERIMENT_ARMS`).
- `eval_core/inference.py`: thêm `run_logic_lm_hyde_semantic[_nohyp]` (dùng `_run_logic_lm`) +
  map vào `ARM_RUNNERS`; thêm `"hypothesis": ans.hypothesis` vào record.

### B5 — Experiment `experiments/02_logic_lm_hyde_semantic/`
Copy `_template/`; `config.yaml`: `family: qa`, `recompute: eval_core`,
`dataset.questions: data/eval/questions_200.json`, `n: null`, `parent: null`,
`arms: { logic_lm_hyde_semantic: {mode: run}, logic_lm_hyde_semantic_nohyp: {mode: run} }`.
README: What/Why + success-criterion (decision-rule) + cost forecast. `.gitignore`: ignore `results/`.

> Tùy chọn tiết kiệm: bỏ arm control → giảm ~½ cost, so sánh qua leaderboard `experiments_repo`.

---

## Cost forecast (Rule 5 — chỉ cost)
treatment/q ≈ 1 HyDE (cache) + 1–3 rule-gen + 1 IRAC ≈ 2–5 calls; control/q chia sẻ cache HyDE ≈ 2–4.
2 arm × 200 q ≈ ≤ ~1,800 gpt-4o-mini chat completions (~≤700 tok out/call); cỡ vài USD; wall-clock
~chục phút; re-run trùng input ⇒ ~$0 nhờ cache.

## Success criterion (pre-registered decision-rule, KHÔNG dự đoán số)
Sau khi có `metrics/academic_metrics.json`: **adopt** treatment thay control chỉ khi citation **F1
không giảm** VÀ (citation **recall tăng** HOẶC **prolog_success tăng**) mà **unable_to_conclude
không tăng**. Ngược lại → giữ control.

## Verification (e2e)
smoke 1 câu qua `LogicLMHydeSemanticPipeline.ask` (kiểm prolog_success/citations/hypothesis≠rỗng) →
pilot `n:8` `eval_core all` (treatment có field `hypothesis`, control rỗng) → full → `eval_core
metrics` → `experiment_contract validate` → backward-compat check `logic_lm_graphrag` không đổi →
copy sang `experiments_repo` + `expkit leaderboard --all`.

## Touch list
- **Mới**: `runtime/retrievers/dense_hyde_semantic_logic_adapter.py`,
  `prompts/runtime/logic_lm/rule_gen_hyde_semantic.md`,
  `experiments/02_logic_lm_hyde_semantic/{config.yaml,README.md,.gitignore}`
- **Sửa (additive/no-op cho arm cũ)**: `src/retrieval/pipeline.py`,
  `runtime/logic_lm_pipelines.py`, `eval_core/arms.py`, `eval_core/inference.py`
