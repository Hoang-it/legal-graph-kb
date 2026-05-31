# 09 — HyDE2 retrieval-grounded iterative HyDE (gpt-4o-mini)

## What

Two-pass retrieval-and-generate flow on the BGE-M3 dense channel:

1. **Pass 1 — seed retrieval**: `V5RetrievalPipeline.retrieve_dense_only`
   on the **raw question**, top-5 clauses.
2. **Pass 2 — grounded HyDE**: `OpenAIGroundedHydeGenerator` (gpt-4o-mini,
   n=1, max_tokens=700, T=0) reads those 5 clauses as context and writes
   a hypothetical legal-document passage that uses the exact BHXH
   vocabulary present in the context.
3. **Pass 3 — final retrieval**: BGE-M3 embedding of the grounded
   passage → dense top-100 → article-deduped.

Three arms:

| arm | method | source of records |
|---|---|---|
| `dense` | `retrieve_dense_only` (raw question) | exp 09 runner (cheap; re-runs from cache for inheritance-style audit) |
| `dense_hyde` | `retrieve_dense_only_hyde` (HyDE1) | inherit from exp 08 via shared HyDE1 cache + skip-when-on-disk |
| `dense_hyde2` | `retrieve_dense_only_hyde2` (HyDE2) | exp 09 runner (new) |

## Why

Exp 08 (`08_hyde_retrieval/`) confirmed HyDE1 lifts R@12 in_corpus from
0.38 → 0.47 (+24%). But `dense` already retrieves the gold somewhere
in the top-100 for ~69% of in_corpus questions — HyDE1 just promotes
some of them above K=12. Funnel suggests there's headroom: gold is
in-pool but not top-ranked. **HyDE2 grounding** is the hypothesis that
feeding the LLM real clause text helps it use exact corpus vocabulary,
narrowing the question-to-clause embedding gap further.

Plan + locked design decisions: [`docs/plans/exp09_hyde2_grounded.md`](../../docs/plans/exp09_hyde2_grounded.md).

## Setup

- **Dataset**: 200 BHXH questions. Pilot 50 reuses exp 08's stratified
  seed=0 list at [`../08_hyde_retrieval/pilot_50_stt.json`](../08_hyde_retrieval/pilot_50_stt.json)
  so HyDE1-vs-HyDE2 comparison is on identical strata.
- **Seed retriever**: BGE-M3+LoRA raw, top-5 (deterministic given index
  + model version).
- **Generator**: `gpt-4o-mini` snapshot id `gpt-4o-mini-2024-07-18`
  (audited in cache payload `model_returned`).
- **Generator prompt**: [`../../prompts/runtime/hyde_generate_grounded.md`](../../prompts/runtime/hyde_generate_grounded.md)
  — inherits HyDE1 constraints (no Điều X / Khoản Y / proper nouns /
  paraphrase), adds a CONTEXT section + "use vocabulary from context"
  instruction.
- **Encoder + index**: `models/bge-m3-bhxh-lora` + `clause_vec_tuned`
  (same as exp 06/07/08, fair comparison).
- **Runner**: [`../../scripts/exp09_run.py`](../../scripts/exp09_run.py)
  (`--pilot-50` for pilot, no flag for full 200).
- **Metrics**: [`../../scripts/exp09_metrics.py`](../../scripts/exp09_metrics.py)
  (`--full` flag from day 1).
- **Cache**: `artifacts/hyde2/openai__gpt-4o-mini/<sha>.json`. Key
  includes `sorted(seed_clause_ids)` hash → invalidates automatically
  when index/model changes.

## Success criteria (plan §6)

Comparison **vs `dense_hyde`** on in_corpus stratum (n=151 full 200):

| # | Criterion | Threshold |
|---|---|---:|
| 1 | `dense_hyde2` R@12 − `dense_hyde` R@12 (abs) | +0.030 |
| 2 | `dense_hyde2` NDCG@12 / `dense_hyde` NDCG@12 − 1 | +5.0% rel |
| 3 | `dense_hyde2` R-Prec / `dense_hyde` R-Prec − 1 | +15.0% rel |

Sanity checks (must hold to claim HyDE2 win):

| # | Criterion |
|---|---|
| S1 | `dense_hyde2` R@12 ≥ `dense_hyde` R@12 (no regression) |
| S2 | `dense_hyde2` R@12 ≥ `dense` R@12 + 0.030 (still beats baseline) |

**Decision rule**: HyDE2 win = (S1 + S2 hold) AND (≥1 of 1/2/3 passes).

## Cost estimate

| | value |
|---|---:|
| Pass 1 (dense, 200 Q × 50ms) | 10s / $0 |
| Pass 2 LLM (full 200 cold) | ~$0.14 |
| Pass 3 (dense, 200 Q × 75ms) | 15s / $0 |
| **Total full 200 (cold)** | **~$0.14** |
| Re-run | $0 (cache hit) |

## Risks (plan §8 summary)

- LLM ignores context and reproduces HyDE1-style doc → mitigated by
  prompt emphasis on "use context vocabulary" + smoke-test inspection.
- LLM leaks `Điều X` from context → mitigated by explicit prompt
  constraint (kept from HyDE1) + optional post-generation regex check.
- Cache key bug → unit-tested for ordering invariance + change
  detection.

## Result summary — pilot 50 + full 200 (2026-05-31)

**HyDE2 LOSES to HyDE1 across every in_corpus metric**, at both N=38
(pilot) and N=151 (full). The pilot result was not noise — the
magnitude grew slightly at full N. This is a clean negative finding;
plan §6 anticipated it explicitly under "HyDE2 lose".

### Cost + ops
| | pilot 50 | full 200 |
|---|---:|---:|
| HyDE1 cost (cache from exp 08) | $0 | $0 |
| HyDE2 cost (cold LLM calls) | $0.0150 (49 cold) | $0.0469 (150 cold) |
| HyDE2 cumulative cost | $0.015 | $0.062 |
| Wall time (with HyDE prewarms) | 26.7s | ~245s (~4 min) |
| Records | 150 (50 × 3) | 600 (200 × 3) |
| Failures | 0 | 0 |
| HyDE2 cache size after | 50 entries | 200 entries |

### Success criteria verdict (in_corpus n=151, full 200)

| # | Criterion | Threshold | **Full 200 Δ** | Pilot 50 Δ | Verdict |
|---|---|---:|---:|---:|:---:|
| 1 | `dense_hyde2` R@12 − `dense_hyde` R@12 (abs) | +0.030 | **−0.0526** | −0.0395 | ❌ **FAIL** |
| 2 | NDCG@12 rel Δ vs HyDE1 | +5% | **−17.2%** | −16.4% | ❌ FAIL |
| 3 | R-Precision rel Δ vs HyDE1 | +15% | **−23.2%** | −20.4% | ❌ FAIL |

Sanity:

| # | Check | Full 200 | Verdict |
|---|---|---:|:---:|
| S1 | HyDE2 ≥ HyDE1 (no regression) | −0.0526 | ❌ FAIL |
| S2 | HyDE2 ≥ dense + 0.030 (still beats baseline) | +0.0378 | ✅ tight |

**Decision rule** (plan §6): win = S1 ∧ S2 ∧ (≥1 of 1/2/3 passes). S1
fails → HyDE2 is a regression vs HyDE1.

### In-corpus stratum (n=151) — full table

| metric | dense | dense_hyde | dense_hyde2 | hyde2 vs dense | hyde2 vs hyde |
|---|---:|---:|---:|---:|---:|
| R@12 | 0.3832 | **0.4736** | 0.4210 | +0.0378 | −0.0526 (−11.1%) |
| R@30 | 0.5311 | **0.6066** | 0.5203 | −0.0108 | −0.0863 (−14.2%) |
| R@100 | 0.6592 | **0.7016** | 0.5989 | −0.0603 | −0.1027 (−14.6%) |
| P@12 | 0.0486 | **0.0607** | 0.0557 | +0.0071 | −0.0050 (−8.2%) |
| NDCG@12 | 0.2186 | **0.2944** | 0.2437 | +0.0251 | −0.0507 (−17.2%) |
| R-Prec | 0.0635 | **0.1326** | 0.1019 | +0.0384 | −0.0307 (−23.2%) |
| MRR | 0.2122 | **0.2843** | 0.2192 | +0.0070 | −0.0651 (−22.9%) |

**Đáng chú ý**: `dense_hyde2` R@100 = 0.5989, **THẤP HƠN cả dense
baseline 0.6592**. Grounding không chỉ hại ở top-K mà còn co recall
ceiling — seed bias kéo HyDE2 doc lệch domain xa hơn cả raw question
encoding ở high-K tail.

### Failure mode — qualitative inspection (stt=56)

Câu hỏi: "Bên em có 1 nhóm cộng tác viên đến làm công việc đóng hàng
mà thời gian không cố định..." — về BHXH cho nhân viên thời vụ.

Gold: `L41_2024.A2` (Luật BHXH 2024 — Điều 2, Đối tượng áp dụng),
`L45_2019.A13` (Bộ luật Lao động 2019 — Điều 13, HĐLĐ).

- **HyDE1** top-12 chứa `L41_2024.A2` ở rank 11 → hit 1/2 gold.
- **HyDE2** top-12 chứa 0/2 gold. Toàn bộ top-12 bị lệch sang
  `L45_2019.A105/A111/A146/A97` (thủ tục tranh chấp + thực hiện
  HĐLĐ).
- **HyDE2 seed clauses** (top-5 dense thuần trên raw question):
  `['L41_2024.A33.K5', 'L45_2019.A105.K2', 'L45_2019.A105.K1',
  'L45_2019.A146.K1', 'L45_2019.A97.K1']` — **3/5 từ L45_2019** (Bộ
  luật Lao động), về quy trình thủ tục, KHÔNG về đối tượng tham gia
  BHXH.

Diễn giải: pass-1 dense thuần trên raw question (chứa vocabulary
hằng-ngày "cộng tác viên", "đóng hàng", "thời gian không cố định")
matched nhầm với cluster văn-bản-thủ-tục-lao-động thay vì cluster
BHXH-eligibility. 5 seed lệch domain → HyDE2 doc round 2 grounded
vào vocabulary lao động/thủ tục → final retrieval thừa kế bias.

**HyDE1 không gặp vấn đề này** vì không consult seed — gpt-4o-mini
tự dùng prior BHXH của nó để viết doc về "đối tượng tham gia BHXH",
embedding gần `L41_2024.A2` hơn.

### Diễn giải kết luận

Plan §6 đã anticipate failure mode này dưới mục "HyDE2 lose":
> _Context dilute focus (hypothetical doc bị seed clause cũ kéo lệch).
> Pre-existing knowledge của gpt-4o-mini về BHXH đủ tốt; ép seed
> clause vào lại làm noisy._

Cả hai giả thuyết đều được khẳng định bằng:
- C1−C3 fail với magnitude lớn (>−10% rel ở mọi K).
- R@100 dưới cả dense baseline → noise từ context không chỉ
  ranking-noise mà recall-noise.
- Inspection stt=56 cho thấy mechanism cụ thể.

### Khuyến nghị

1. **Document HyDE2 (basic seed_k=5) là negative result.** Không
   draft ADR; không scale lên full_rerank arms; không bổ sung vào
   pipeline production.
2. **Ablation candidates** (nếu sau này muốn cứu HyDE2 — chưa ưu
   tiên cho thesis):
   - `seed_k=3`: giảm noise. Đoán: marginally better, vẫn lose vs
     HyDE1.
   - **Same-law filter**: trong pass-1, chỉ giữ seed clauses cùng
     `law_id` với top-1 (giả thiết top-1 là dense most-confident).
     Sẽ giảm domain bias từ L45_2019 vs L41_2024 trong stt=56.
   - **Re-rank seeds bằng cross-encoder** trước khi đưa vào prompt
     (đắt hơn nhưng có thể cứu khi seed quality biến động).
   - **HyDE2 conditional**: chỉ áp dụng cho câu hỏi mà HyDE1 confidence
     thấp; default = HyDE1.
3. **Thesis narrative**: kết quả này có giá trị scientific (tiết
   lộ failure mode của naive iterative HyDE) — đáng đưa vào chương
   với caveat "naive grounding hurts when seed retrieval is
   domain-noisy on raw question".
4. **Không cần exp 10 trên `full_rerank_hyde2`** — vì HyDE2 dense
   side đã thua, full_rerank pipeline sẽ tiếp tục hấp thụ HyDE2 lift
   (vốn không có) → metric đoán được kết quả mà không cần chạy.

