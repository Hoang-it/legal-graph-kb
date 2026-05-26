# Methodology Fixes — Post-Audit Patches (2026-05-26)

Document hóa 7 issues phương pháp đã phát hiện qua audit + patches áp dụng.
**Không re-run inference; không gọi judge mới**. Tất cả fixes là post-processing
trên `metrics.json` từ cached judge outputs.

---

## 1. API error contamination (HIGHEST severity)

**Phát hiện**: 90 records (R1 `elite_ontology`: 13, R2 `elite_graphrag × gpt-5-mini`: 77)
có `prompt_tokens=0 AND completion_tokens=0` — silent OpenAI "Connection error."
được pipeline ghi nhận như `unable_to_conclude` (giả như Prolog failure).

**Tác động ban đầu**:
- R2 GR × gpt-5-mini "prolog_success" được báo cáo 0.595 → thực ra 0.9675 (n=123 sạch).
- "Cost reversal" ($0.0084 < $0.0121) chỉ là artifact của 77 zero-token records.
- Pairwise "NR beats GR for gpt-5-mini 52.5% vs 30.5%" thực ra ngược lại: GR wins 68.5%
  trên 89 consistent-verdict cases sạch (judge so sánh real NR vs broken GR "[Pipeline
  không trả về kết luận]" → NR luôn thắng trivially).

**Fix**:
- `experiments/audit_apply_fixes_v2.py`: tag `r["api_error"]=True` cho records 0-token (không phải graphrag arm dùng estimated tokens).
- Aggregate cells trong reports loại bỏ API-error records.
- Pairwise: skip pairwise data trên broken GR records.
- Tách "API error rate" section riêng để minh bạch infrastructure failures.

**Pipeline issue chưa fix** (out of scope cho post-processing): `elite/src/pipelines/program_pipeline.py:_attempt` catch `Exception` blindly → cần distinguish `openai.APIConnectionError` / `RateLimitError` / `APITimeoutError` và retry với backoff.

---

## 2. BERTScore F1 bias với structured output

**Vấn đề**: BERTScore so sánh embeddings token. Elite arms output IRAC text với
headers `Issue:/Rule:/Application:/Conclusion:` — format dramatically khác
free prose của graphrag/llm_only. Lexical/structural overlap với gold (free prose)
không fair.

**Decision ban đầu (2026-05-26)**: **Drop BERTScore** cho elite arms.

**Decision mới (2026-05-27)**: **Plain-answer rendering** — modify IRAC prompt
để LLM produce CẢ IRAC + plain_answer trong **1 LLM call duy nhất**. Compute
BERTScore trên `plain_answer` (prose form, comparable với prose baselines).

### Plain-answer pipeline (2026-05-27 update)

- **New prompt**: `experiments/prompts/irac_with_plain.md` — output JSON
  với 2 fields:
  - `irac`: structured 4-section analysis (preserved cho user inspection)
  - `plain_answer`: 2-4 câu prose tự nhiên với inline `[Điều X khoản Y]` citations
- **Pipeline patches**:
  - `elite_pipelines.py:_TokenTrackingLLMClient._irac_with_tracking` accept
    `override_irac_prompt`, parse JSON response
  - `EliteAnswer` dataclass thêm `plain_answer` field
  - `_EliteBasePipeline.__init__` default `enable_plain_answer=True` → tự load
    new prompt nếu file tồn tại
- **Run scripts** (`run_inference.py`, `run_multimodel_inference.py`): save
  `plain_answer` vào record JSON
- **Compute metrics patches**:
  - `m_answer_relevance`: use `plain_answer` if present (else fall back to `answer`).
    Cache key suffix `_plain` để tránh hit stale cache.
    Tag `_used_plain_answer: True` để generate_report aware.
  - `compute_bertscore_all`: same — feed `plain_answer` when available.
  - Tag `_used_plain_answer: True`.
- **Report logic**: `generate_report.py` + `generate_multimodel_report.py`:
  - Default cho elite arms: `dropped (IRAC bias)`
  - Override: nếu metric record's `_used_plain_answer=True` → fall through
    to fair value rendering
- **Backfill script** (`experiments/rerender_plain_answer.py`):
  - Optional 1-time backfill cho records cũ (1435 records, ~$0.72 with
    gpt-4o-mini, ~4h sequential)
  - Skip records với failure IRAC hoặc api_error
  - Idempotent (skip nếu `plain_answer` already set)

### Status hiện tại

- ✅ Infrastructure ready cho all future runs
- ✅ Pipeline verified với 1 smoke + 3 pilot records (R1 elite_graphrag)
- ⚠️ Backfill chưa chạy cho 1432 records cũ → reports hiện tại vẫn show
  `dropped (IRAC bias)` cho elite arms
- 📝 To enable elite BERTScore/AR in reports: run
  `python -m experiments.rerender_plain_answer --combos all`, then re-run
  `compute_metrics.py` (cache đã dùng `_plain` suffix → re-compute từ scratch).

**Sample output quality (gpt-4o-mini, smoke test 1 câu):**

Question: "Tôi đã đóng BHXH 12 năm, có đủ điều kiện hưởng lương hưu không?"

IRAC field: Issue/Rule/Application/Conclusion 4 sections (~150 từ tiếng Việt)

plain_answer field:
> Bạn đã đóng BHXH 12 năm thì chưa đủ điều kiện hưởng lương hưu hằng tháng theo [Điều 141 khoản 10] Luật BHXH 2024 — luật yêu cầu tối thiểu 15 năm. Bạn có thể tiếp tục đóng thêm 3 năm nữa để đủ điều kiện, hoặc tham khảo chế độ BHXH một lần nếu phù hợp.

Output 615 completion tokens / 3773 prompt tokens — overhead ~200 tokens
so với original IRAC-only render.

**Alternative đã consider**: normalize IRAC về prose (strip headers, flatten).
Reject vì: changes content non-trivially (drop sections), không capture LLM's
own conversational interpretation. Single-call plain_answer là cleaner — LLM
quyết định cách phrase prose dựa trên semantic understanding của IRAC nó tạo.

---

## 3. RAGAs answer_relevance bias

**Vấn đề tương tự (2)**: AR (Es 2024) sinh 3 questions từ answer rồi đo cosine sim với original Q. IRAC có "Issue:" section thường chứa câu hỏi ban đầu → trivially high relevance.

Spot-check raw cache xác nhận: AR cho elite generate questions sát với Issue field. AR cho graphrag generate questions tổng quát hơn.

**Decision**: **Drop AR** cho elite arms (cùng cách như BERTScore).

---

## 4. RAGAs faithfulness cho elite_no_retrieval

**Câu hỏi**: Context input cho faithfulness judge là gì khi `retrieved=None`?

**Trace `m_faithfulness` ([compute_metrics.py:284-330](../experiments/compute_metrics.py)):
```python
cits = list(dict.fromkeys(record.get("citation_ids") or []))
ctx_map = neo.get_texts(cits)  # ← Neo4j text của cited IDs
context_block = "\n\n".join(f"[{cid}]:\n{txt[:600]}" for cid, txt in ctx_map.items())
```

→ Context = **Neo4j text của citation IDs đã được populate trên record**, KHÔNG dùng gold answer, KHÔNG dùng IRAC's own claims.

**Đối với elite_no_retrieval**:
- Citation IDs từ fallback parser của Prolog `legal_source(...)` facts (LLM training knowledge / fabrication)
- Faithfulness judge: "do the answer's claims follow from Neo4j's text of those articles?"
- Đây là test fair: context độc lập với LLM claim.

**Issue thực sự**: n_valid=35/200 (small subset) → selection bias. LLM chỉ "đủ tự tin" để cite trong 35/200 cases; faithfulness=0.81 trên subset này nói "khi nó cite, claims align với Neo4j text" — không phải "elite_no_retrieval is faithful overall".

**Decision**: Giữ metric NHƯNG report cụ thể n_valid trong cell (đã fix bởi audit_apply_fixes_v2). Cảnh báo trong caveats rằng faithfulness của elite_no_retrieval đo trên subset where LLM chose to cite — not whole population.

**Đề xuất tương lai**: significance test on this metric (C2 trong significance.md) cho thấy n=29 paired là too small → CI [-0.097, 0.207] includes 0 → KHÔNG defensible. Cần increase samples hoặc drop khỏi paper headline.

---

## 5. Pairwise reporting: consistent-verdict denominator

**Vấn đề**: Báo cáo cũ ghi "elite_graphrag wins 29.5% (gpt-4o)" trên total 200. Đúng hơn:
- Signal chỉ tồn tại khi judge consistent (both ab + ba pick same arm)
- 200 − n_split = n_consistent
- Tỉ lệ ý nghĩa: wins / n_consistent

**Fix**: Mọi pairwise section trong reports giờ format theo 2 levels:
1. **Full breakdown** (n=200): hiển thị split / wins / tie raw
2. **Consistent-verdict subset** (n=n_consistent): wins %

Ví dụ R2 gpt-5-mini (sau API filter):
- Full (n=123): GR=61 (49.6%), split=34 (27.6%), NR=28 (22.8%)
- Consistent subset (n=89): **GR=68.5%, NR=31.5%** ← signal thật

**Bonus reverse**: trong R2 trước đây "NR beats GR for gpt-5-mini" (105 vs 61), sau fix
+ API filter → **GR beats NR (61 vs 28 on consistent subset)**. Headline reversed.

---

## 6. Hallucination metric (Magesh 2025) — tách 2 metrics

**Vấn đề**: Original formula:
```python
n_halu_total = n_misstate + n_unsup + n_invented
denom = max(1, n_claims + n_invented)
hallucination_rate = n_halu_total / denom
```
+ Edge case: `1.0 if n_invented > 0 and not valid_texts else None` → records có chỉ
1 invented cit (e.g. cite Điều 999 không tồn tại) bị gán full hallucination 1.0.

→ Conflate **content lying** (misstate, unsupported) với **citation invention** (cite article không tồn tại). Hai dimensions khác nhau, nên đo riêng.

**Fix** ([compute_metrics.py:430-484](../experiments/compute_metrics.py)):
```python
content_hallucination_rate = (n_misstate + n_unsupported) / max(1, n_claims)
invented_citation_rate     = n_invented / max(1, n_total_citations)
hallucination_rate         = (giữ formula cũ cho backward compat)
```

`audit_apply_fixes_v2.py` recompute từ existing record fields (không cần judge call mới).

Cả 2 metrics giờ xuất hiện trong aggregate table với direction lower-better.

---

## 7. Significance testing — 5 top claims

**Vấn đề**: Reports cũ tuyên bố winners theo macro mean without hypothesis testing.

**Fix**: `experiments/compute_significance.py` chạy:
- McNemar test cho paired binary outcomes (Prolog success, pairwise consensus)
- Bootstrap 95% CI 10k resamples cho continuous metrics (faithfulness diff)
- Bonferroni correction: α_bonf = 0.05/5 = **0.01**

**Top 5 claims tested**:

| # | Claim | Test | Result | Defensible? |
|---|---|---|---:|---|
| C1 | llm_only beats graphrag pairwise (R1) | McNemar | p < 0.0001 | ✓ **DEFENSIBLE** |
| C2 | graphrag faithfulness > NR (R1) | Bootstrap CI | CI=[-0.097, 0.207] includes 0 | ✗ DROP (n=29 too small) |
| C3a | NR prolog_success > Ontology (R1) | McNemar | p=0.89 | ✗ DROP (essentially tied) |
| C3b | NR prolog_success > GraphRAG (R1) | McNemar | p=0.0035 | ✓ **DEFENSIBLE** |
| C4 | NR vs GR pairwise (gpt-5-mini) | McNemar | p=0.0006, **GR wins** | ✓ **DEFENSIBLE (reversed!)** |
| C5 | GR vs NR prolog_success (gpt-5-mini, clean) | McNemar | p=0.69 | ✗ DROP (no diff) |

**3/6 claims defensible, 3 must NOT be paper headlines** (insufficient evidence at α_bonf=0.01).

Key reversal: C4 — originally claim "NR beats GR for gpt-5-mini" → after API filter + consistent-verdict subset, GR beats NR (p=0.0006, significant). The original was contamination artifact.

---

## Summary of changes by file

| File | Change |
|---|---|
| `experiments/compute_metrics.py:430-484` | Split `m_hallucination` into 3 fields (content_halu, invented_cit, legacy) |
| `experiments/compute_metrics.py:518-535` | Fixed `_vote` (previous audit, included here for completeness) |
| `experiments/generate_report.py` | METRIC_SPECS expanded, ELITE_BIASED_METRICS gating, API-error filter, new pairwise format |
| `experiments/generate_multimodel_report.py` | Same as above + R2 API-error filter on pairwise |
| `experiments/audit_apply_fixes_v2.py` (NEW) | Post-process metrics.json: split halluc + tag api_error |
| `experiments/compute_significance.py` (NEW) | McNemar + bootstrap + Bonferroni |
| `reports/experiment_report.md` | Regenerated |
| `reports/multimodel_report.md` | Regenerated |
| `reports/significance.md` (NEW) | 5 claims × test results |
| `data/eval/metrics.json` | Backed up `.bak_pre_v2`, then patched |
| `data/eval/multimodel/metrics.json` | Same |

## Recommended paper headline claims (post-audit)

1. **llm_only beats graphrag** trong pairwise judge (R1, p<0.0001) ← C1
2. **elite NR has higher prolog_success than elite GR** với gpt-4o-mini (R1, p=0.0035) ← C3b
3. **GR (with retrieval) beats NR (no retrieval) cho gpt-5-mini** trong pairwise (R2, p=0.0006) — **reversed from cũ** ← C4

**KHÔNG được claim**:
- Anything về BERTScore / AR cho elite arms (biased)
- "elite_graphrag drops gpt-5-mini's prolog_success" (artifact của API errors)
- "graphrag faithfulness > elite_no_retrieval" (n=29 too small, CI includes 0)
- "NR vs Ontology prolog_success differs" (basically equal, p=0.89)
