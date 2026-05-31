# Plan — exp 09: HyDE2 (retrieval-grounded iterative HyDE)

- **Status**: PROPOSED. Not yet implemented. Last updated 2026-05-31.
- **Owner**: Nguyễn Hữu Hoàng
- **Branch (proposed)**: `exp/09-hyde2` (cut from `exp/08-hyde` after item A lands).
- **Parent plan**: [`v5_general_retrieval.md`](v5_general_retrieval.md).
- **Related**: exp 08 ([`hyde_gpt4o_mini.md`](hyde_gpt4o_mini.md), [`experiments/08_hyde_retrieval/README.md`](../../experiments/08_hyde_retrieval/README.md)) — standard HyDE baseline.

## 1. Hypothesis

Standard HyDE (exp 08, arm `dense_hyde`) sinh hypothetical doc CHỈ từ
câu hỏi gốc. LLM phải tự đoán vocabulary BHXH chuẩn nào sẽ xuất hiện
trong điều luật đúng. Khi đoán sai (vd: dùng cụm từ thông thường thay
vì thuật ngữ chuyên ngành như "mức bình quân tiền lương tháng đóng
BHXH"), embedding lệch khỏi cluster của clause thật.

**HyDE2 (proposed)**: thực hiện **2 pass retrieval**.

- Pass 1: dùng `dense` thuần (BGE-M3+LoRA + raw question, top-5 clause)
  để lấy 5 đoạn clause THẬT từ KG.
- Pass 2: feed 5 đoạn clause đó vào prompt LLM round 2 như "context";
  yêu cầu LLM sinh hypothetical doc **dùng đúng vocabulary xuất hiện
  trong context**. Embed doc round 2 → dense search lần nữa.

**H1**: HyDE2 nâng R@12 / NDCG@12 / R-Prec ở in_corpus thêm một mức có
ý nghĩa so với HyDE1, vì doc round 2 grounded vào vocabulary thực của
KG → embedding gần cluster clause thật hơn.

**H0** (null): grounding không giúp / hại do (a) context dilute focus,
(b) seed dense thuần đã có hết signal, HyDE1 đã exploit đủ — round 2
chỉ noise.

## 2. Why this experiment

Exp 08 confirm:
- `dense_hyde` thắng MẠNH `dense` ở mọi metric in_corpus (R@12 +23.6%,
  R-Prec +109%).
- Nhưng `dense_hyde` R@12 in_corpus = 0.474 — gold ở K=12 vẫn miss ~52%.
  Còn nhiều dư địa cải thiện.
- Funnel cho thấy `dense` thuần đã có R@all = 0.69 trên in_corpus —
  signal có, nhưng dense embedding không xếp đúng top-12.

HyDE2 trực tiếp tấn công gap này: nếu LLM thấy 5 clause thật trong
context, nó nên sinh hypothetical doc gần cluster clause thật hơn → top
ranking cải thiện.

Literature parallel: Gao+22 mentions "iterative HyDE" như follow-up;
nhiều RAG paper sau (eg. ITER-RETGEN, FLARE) chứng minh iterative
generation+retrieval thắng single-shot ở domain có vocabulary gap (như
legal/medical).

## 3. Decisions locked

### D-EXP09-1 — Seed retriever (pass 1) = `dense` thuần

User-confirmed 2026-05-31. Rationale:
- Cost cực thấp (~50ms/Q, $0).
- Seed cleanliness cao: lift round 2 attributable thẳng cho HyDE2 logic,
  không bị confound bởi reranker (loại bỏ vs `full_rerank` seed) hoặc
  HyDE-stacking noise (loại bỏ vs `dense_hyde` seed).
- So sánh đối xứng với arm hiện hữu `dense_hyde` (cả hai dùng dense
  pure).

### D-EXP09-2 — Context size = top-5 clause

User-confirmed. Rationale:
- 5 clause × ~200–300 từ ≈ 1500 tokens, đủ trong 4k context window của
  gpt-4o-mini với max_tokens=700 output.
- Focus density cao: LLM ít risk salad-bar từ nhiều passages khác chủ
  đề.
- Pre-flight phép tính token: input prompt mới ≈ 2400 tokens (system 850
  + context 1500 + question 50) × $0.15/M = $0.00036; output 600 ×
  $0.60/M = $0.00036 → ~$0.0007/call (≈ +40% so với HyDE1 $0.0005).

### D-EXP09-3 — Citation rule: GIỮ cấm tuyệt đối "Điều X / Khoản Y"

User-confirmed. Rationale:
- HyDE2 hypothetical doc vẫn ở dạng paraphrase, không leak citation
  string từ context retrieved.
- Bảo vệ metric audit trong item B sau này: nếu doc round 2 chứa
  citation từ context, và LLM E2E generator ở item B copy citation đó
  ra, ta không phân biệt được lift là từ HyDE vocabulary lift hay từ
  citation leak qua context channel.
- Trade-off: bỏ qua việc dense embedding có thể tận dụng citation
  string xuất hiện trong text (BGE-M3 có thể).

### D-EXP09-4 — Arms = 3 (dense-side only)

User-confirmed:
- `dense` — baseline (inherit từ exp 08 records, idempotent).
- `dense_hyde` — HyDE1 (inherit từ exp 08 records).
- `dense_hyde2` — HyDE2 (new arm, fresh 200 records).

Rationale: A/B/C clean. Câu hỏi cốt lõi "grounding có giúp?" trả lời
trực tiếp bằng dense_hyde2 vs dense_hyde. Không cần full_rerank arms
vì exp 08 đã chứng minh reranker hấp thụ HyDE1; không có lý do
expensive lặp lại để xem nó cũng hấp thụ HyDE2.

Trade-off: không trả lời được "rerank2 có hấp thụ HyDE2 không?". Nếu
HyDE2 dense-side thắng đáng kể, đó sẽ là exp 10 follow-up.

### D-EXP09-5 — HyDE generator config = giữ exp 08 defaults

- Model: `gpt-4o-mini` (snapshot id audit ở `model_returned`).
- N=1 hypothetical doc.
- max_tokens=700.
- temperature=0.0.
- concurrency=5.

Rationale: minimize variable count, isolate "grounding" là biến độc
lập. Nếu HyDE2 thắng, follow-up ablation có thể vary N hoặc temperature.

### D-EXP09-6 — Cache key MUST include seed clause IDs

Critical implementation detail. HyDE2 cache key =
`sha256(question + prompt_sha + n + model + max_tokens + temperature +
seed_clause_ids_sorted_hash)`.

Lý do:
- Pass 1 (dense thuần) deterministic given (BGE-M3 model version, index
  version, question, k=5).
- Nhưng nếu sau này tune LoRA hay re-build index, seed clause IDs đổi
  → hypothetical doc cũ trở thành stale.
- Hash seed clause IDs vào cache key tự động invalidate khi index/model
  đổi, KHÔNG cần manual purge.

Hash spec: `sha256(",".join(sorted(top_5_clause_ids)))`.

Cache dir: `artifacts/hyde2/openai__gpt-4o-mini/<sha>.json` (separate
namespace để không lẫn với HyDE1 cache).

## 4. Method — step by step

Cho mỗi question q:

### Pass 1 — Seed retrieval (deterministic, no LLM)
1. Encode q bằng BGE-M3+LoRA → `vec_q` (1024-d, L2-normalized).
2. Cypher `db.index.vector.queryNodes('clause_vec_tuned', 5, vec_q)`
   → 5 clause rows: `(clause_id, text, score, article_id, law_id, ...)`.
3. **Không temporal filter ở pass 1** — context cho LLM nên đại diện
   "what dense thuần thấy", kể cả clause khả năng outdated. Filtering
   diễn ra ở pass 2 retrieval.
4. Output: `seeds: list[ClauseRow]` với `len(seeds)==5`.

### Pass 2 — LLM-grounded HyDE generation
5. Compute cache key như D-EXP09-6.
6. Nếu cache hit → load hypothetical doc cũ, skip 7–9.
7. Build prompt:
   - System (giống HyDE1 prompt, có chỉnh): nhấn mạnh "dùng đúng
     vocabulary trong CONTEXT bên dưới". Reuse phần cấm "Điều X/Khoản
     Y/tên người" nguyên xi.
   - User: gồm `## CONTEXT` (5 đoạn clause + article numbers MỚI BỊ
     CẤM xuất hiện trong output) + `## TÌNH HUỐNG` (câu hỏi q).
8. Gọi `gpt-4o-mini` (n=1, max_tokens=700, temperature=0).
9. Cache atomic write (tmp → os.replace).
10. Output: 1 hypothetical doc text.

### Pass 2 retrieval — final
11. Encode hypothetical doc bằng cùng BGE-M3+LoRA → `vec_hyde2`.
12. Cypher `db.index.vector.queryNodes('clause_vec_tuned', 100, vec_hyde2)`
    → top-100 clause.
13. Article-dedupe in rank order → `final_article_ids`.
14. Record:
    ```json
    {
      "arm": "dense_hyde2",
      "stt": N,
      "question": "...",
      "gold_citations_raw": [...],
      "config_used": {
        "dense_k": 100, "seed_k": 5, "adapter_path": "...",
        "dense_index": "clause_vec_tuned",
        "hyde2": {
          "model": "gpt-4o-mini", "n": 1, "max_tokens": 700,
          "temperature": 0.0, "prompt_sha": "<sha>",
          "seed_clause_ids": [...], "seed_clause_ids_hash": "<sha>"
        }
      },
      "retrieval_only": {
        "final_article_ids": [...],
        "n_final": <int>,
        "elapsed_s": <float>,
        "elapsed_breakdown": {
          "seed_retrieve": <float>, "hyde_generate": <float>,
          "final_retrieve": <float>
        }
      }
    }
    ```
15. Metrics + funnel scripts đọc records giống exp 08 (article-deduped
    diagnostic) → reuse `eval_core.gold` + `eval_core.metrics`.

## 5. Implementation surface

### New files
- `prompts/runtime/hyde_generate_grounded.md` — prompt round 2.
  Mirror structure của `hyde_generate.md` (single file với
  `===== USER =====` sentinel). User template chứa
  `{context}` + `{question}` placeholders.
- `src/retrieval/hyde2.py` — `OpenAIGroundedHydeGenerator` class.
  - Extend `OpenAIHydeGenerator` (subclass): override `generate()` để
    nhận `context_passages: list[str]` arg, build prompt khác, cache
    key khác.
  - Method `embed_query_callable(embed_model, seed_retriever)` returns
    closure `question → np.ndarray`:
    1. Gọi `seed_retriever.dense_only(question, k=5)` → seed_clause_ids
       + seed_texts.
    2. Gọi `self.generate(question, context_passages=seed_texts,
       seed_clause_ids=seed_clause_ids)`.
    3. Encode doc → return vector.
- `src/retrieval/pipeline.py` — thêm method
  `retrieve_dense_only_hyde2(question, top_k=100)` đối xứng với
  `retrieve_dense_only_hyde`.
- `experiments/09_hyde2_grounded/` — folder mới (copy template):
  - `config.yaml` — declare arms: `dense` inherit từ `01_initial_eval`
    nếu cần; `dense_hyde` inherit từ `08_hyde_retrieval`; `dense_hyde2`
    mode=run.
  - `README.md` — WHAT/WHY (cite plan này).
  - `.gitignore` — `results/` ignored by default (không phải baseline).
- `scripts/exp09_run.py` — runner. Pattern giống exp08_run.py:
  - Construct hai `V5RetrievalPipeline`: pipe_dense (no hyde, dùng cho
    arm `dense` HOẶC ăn inherit), pipe_hyde2 (với
    `OpenAIGroundedHydeGenerator`).
  - **WEIGHT SHARING** BGE-M3 + dense index session giữa hai pipe để
    tránh OOM (giống exp 08).
  - Cờ `--pilot-50` reuse `experiments/08_hyde_retrieval/pilot_50_stt.json`
    để strata giống hệt exp 08 (so sánh head-to-head fair).
  - Cờ `--cost-cap` default $0.50 (HyDE2 estimate full 200 = $0.14, dư).
- `scripts/exp09_metrics.py` — fork từ exp08_metrics.py. Hỗ trợ cờ
  `--full` ngay từ đầu (rút kinh nghiệm exp 08). 3 arms thay vì 4.
- `scripts/exp09_funnel.py` — chỉ có 1 stage retrieve (no rerank/expand),
  funnel rút gọn: `dense_thuần (pass 1)` → `dense_hyde2 (pass 2)` →
  `final`. Hoặc skip funnel cho exp 09 (chỉ dense, funnel ít nghĩa hơn
  cho full_rerank).

### No changes to
- `eval_core/` — metric engine không đổi. `academic_v2` strict tuple chỉ
  liên quan ở item B của plan exp 08, exp 09 vẫn dùng article-deduped
  diagnostic.
- `prompts/runtime/hyde_generate.md` — HyDE1 prompt giữ nguyên cho
  reproducibility exp 08.
- `data/`, schema, KG — không động.

### Test surface
- Unit test `OpenAIGroundedHydeGenerator`:
  - Cache key đổi khi seed clause IDs đổi (mock seed_retriever).
  - Cache key KHÔNG đổi khi context_passages reorder nhưng seed IDs
    giữ nguyên (consistent).
- Smoke test: 1 câu pilot, assert hypothetical doc không chứa "Điều
  \d+", "Khoản \d+", tên proper noun.

## 6. Success criteria

In_corpus stratum (n=151 full 200) — comparison **vs `dense_hyde`**:

| # | Criterion | Threshold | Note |
|---|---|---:|---|
| 1 | `dense_hyde2` R@12 − `dense_hyde` R@12 (abs) | +0.030 | Mirror exp 08 C1 threshold |
| 2 | `dense_hyde2` NDCG@12 / `dense_hyde` NDCG@12 − 1 | +5.0% rel | Mirror exp 08 C2 |
| 3 | `dense_hyde2` R-Precision / `dense_hyde` R-Precision − 1 | +15.0% rel | Strict precision-sensitive |

**Sanity check (must hold to claim HyDE2 win)**:
| # | Criterion | Threshold |
|---|---|---:|
| S1 | `dense_hyde2` R@12 ≥ `dense_hyde` R@12 (no regression) | abs Δ ≥ 0 |
| S2 | `dense_hyde2` R@12 ≥ `dense` R@12 (still beats baseline) | abs Δ ≥ +0.030 |

### Decision rule

**HyDE2 win** = (S1 + S2 hold) AND (≥1 of 1/2/3 passes).

→ Recommend gom `dense_hyde2` thành arm chính thức + draft ADR
`docs/decisions/003_hyde2_grounded.md`. Plan exp 10: thử nghiệm
`dense_hyde2` trong full_rerank pipeline (xem rerank có hấp thụ HyDE2
không).

**HyDE2 neutral** (S1 hold nhưng no criterion passes) = grounding
không thêm signal. Kết luận: `dense_hyde` (single-shot HyDE) là sweet
spot cho dense-only retrieval. Document, không tốn engineering cho ADR.

**HyDE2 lose** (S1 hoặc S2 fail) = grounding hại — hypothesis:
- Context dilute focus (hypothetical doc bị seed clause cũ kéo lệch).
- Pre-existing knowledge của gpt-4o-mini về BHXH đủ tốt; ép seed clause
  vào lại làm noisy.
- → Try `seed_k=3`? Try `temperature=0.3`? Là follow-up tách riêng,
  không phải claim HyDE2 thắng.

## 7. Cost estimate

| Item | calc | total |
|---|---|---:|
| Pass 1 seed retrieval | ~50ms × 200 = 10s, no LLM | $0 |
| Pass 2 LLM cold (199 unique) | ~$0.0007 × 199 = $0.139 | $0.14 |
| Pass 2 dense final retrieval | ~75ms × 200 = 15s, no LLM | $0 |
| **Total exp 09 cold** | | **~$0.14** |
| Re-run sau code change | cache hit if prompt_sha + seed IDs giữ nguyên | $0 |
| Wall time full 200 | ~6 min (no rerank/expand, no full pipeline) | |

Plan estimate pessimistic: $0.20 ceiling (giả sử prompt token cao hơn
dự tính). `--cost-cap=$0.50` an toàn.

## 8. Risks + mitigations

| Risk | Mức | Mitigation |
|---|---|---|
| Pass 1 dense thuần seed kém → context bẩn → HyDE2 doc lệch | medium | Đo trực tiếp R@5 in_corpus của dense thuần ở exp 08: ~0.30. ~30% câu có ít nhất 1 gold trong seed. Đủ cho LLM grounding signal kể cả khi 4/5 seeds là near-miss. Nếu thí nghiệm lose, ablation `seed_k` lên 10 hoặc 12 là next step. |
| LLM ignore context, reproduce HyDE1 doc | low | Prompt yêu cầu explicit "dùng vocabulary trong CONTEXT". Smoke-test 5 câu, đọc doc, confirm grounding. |
| LLM leak "Điều X" từ context | low–medium | Prompt cấm tuyệt đối (kế thừa HyDE1 design). Post-generation regex check: nếu match `Điều\s+\d+` → log warning, optionally reject + regenerate với temperature=0.1. Document trong `OpenAIGroundedHydeGenerator.generate()` docstring. |
| Cost overrun | low | --cost-cap + cost summary cuối run. Estimate $0.14 với cap $0.50 → 3.5× margin. |
| Cache key bug (seed IDs ordering) | low | Sort seed_clause_ids trước khi hash. Unit test check. |
| Confound: pilot 50 reuse pilot 50 từ exp 08 = same questions, không truly independent test | inherent | Cố ý — muốn pilot exp 09 stratum giống hệt exp 08 để head-to-head clean. Magnitude check ở full 200 (n=151 in_corpus) đủ stable cho thesis. |

## 9. Open questions

(none — D-EXP09-1..6 đã chốt với user 2026-05-31.)

## 10. Follow-ups (out of scope for exp 09)

1. **Exp 10 candidate**: nếu HyDE2 thắng ở dense-side, lặp lại với
   `full_rerank_hyde2`. Test giả thuyết: liệu reranker có hấp thụ
   HyDE2 nhẹ hơn HyDE1 vì doc round 2 có grounded vocabulary?
2. **N-hop iterative HyDE**: pass 3 = re-feed top-5 từ pass 2 retrieval
   vào LLM lần nữa. Probably diminishing returns + cost x2, nhưng giá
   trị literature.
3. **E2E HyDE2** (mirror item B của exp 08 plan): chạy
   `V5RetrievalPipeline.ask` với HyDE2-augmented retrieval, đo
   citation recall/precision dưới `academic_v2` strict tuple. Đó mới
   là số chốt cho thesis.
4. **HyDE2 prompt ablation**: thử bỏ constraint "không Điều X" — đo
   metric leak risk thực tế bằng tỷ lệ output match `Điều\s+\d+`.
   Hữu ích nếu HyDE2 lose vì hypothesis "constraint quá chặt làm doc
   khô".

## 11. References

- [`v5_general_retrieval.md`](v5_general_retrieval.md) — parent plan.
- [`hyde_gpt4o_mini.md`](hyde_gpt4o_mini.md) — HyDE1 design notes.
- [`exp08_followups_and_strict_metric.md`](exp08_followups_and_strict_metric.md) —
  exp 08 handoff (item B is the E2E follow-up; this plan is parallel,
  not a replacement).
- [`experiments/08_hyde_retrieval/README.md`](../../experiments/08_hyde_retrieval/README.md) —
  exp 08 result summary (pilot 50 + full 200).
- Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without Relevance
  Labels" — HyDE original.
- Shao et al. 2023, "Enhancing Retrieval-Augmented Large Language Models
  with Iterative Retrieval-Generation Synergy" (ITER-RETGEN) —
  iterative pattern literature.
