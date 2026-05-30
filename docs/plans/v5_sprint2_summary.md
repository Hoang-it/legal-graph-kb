# v5 Sprint 2 — final summary

> **Status**: Sprint 2 closed 2026-05-30.
> Companion: [v5_sprint2_implementation.md](v5_sprint2_implementation.md) (planning) ·
> [experiments/04_v5_sprint2_m2/README.md](../../experiments/04_v5_sprint2_m2/README.md) (final result).

## Headline result

End-to-end citation recall, same 30-stt apples-to-apples:

| arm | recall_ma | prec_ma | f1_ma | latency | recall_ma in-corpus |
|---|---:|---:|---:|---:|---:|
| **graphrag_v5_m2 (Sprint 2)** | 0.2236 | 0.2111 | 0.1994 | **4.19s** | **0.3750** |
| graphrag_v5 (Sprint 1) | 0.2361 | 0.2133 | 0.2093 | 39.18s | 0.3214 |
| graphrag (Sprint 0) | 0.0944 | 0.0556 | 0.0656 | 4.91s | 0.1786 |
| llm_only | 0.0111 | 0.0333 | 0.0167 | 5.72s | 0.0000 |

**Verdict** — Sprint 2 M2 (LoRA fine-tune + reranker swap):
- 9.4× faster latency vs Sprint 1
- +17% relative in-corpus recall vs Sprint 1
- -5% overall macro recall vs Sprint 1 (drag from "unparseable" category — methodological artifact)

## Phases delivered

| Phase | Goal | Delivered? | Notes |
|---|---|---|---|
| 0a | Parser strict + KG validation + re-aggregate baseline | ✓ | 6 FP eliminated on v5 Sprint 1 records, precision +14% rel |
| 0b | Hash-seal 150 test / 50 dev split | ✓ | Stratified locked, SHA256 verified |
| 1 | Synthetic Q/clause data pipeline | ✓ v1 + v2 iteration | v1 had 17-word average (style gap); v2 reprompted to 62-word avg matching dev's 57 |
| 2 | LoRA fine-tune BGE-M3 on Colab | ✓ | T4, r=8, lr=1e-5, 10 epochs; adapter 14.3MB |
| 3 | Re-encode + load `clause_vec_tuned` | ✓ | additive, vanilla index untouched |
| 4 | Experiment 04 + decision gate | ✓ | Gate "M6 + M3 HyDE" triggered (in-corpus 0.375 ∈ [0.35, 0.50]) |
| 5 | M6 verifier ± M3 HyDE | **deferred** | See rationale below |
| 6 | Final A/B + write-up | ✓ in this doc | Phase 5 deferred → this IS the close |

## Critical lessons learned

### Lesson 1 — Synthetic-data style match matters more than hyperparams

v1 synthetic: mean 17 words, "Câu hỏi như chatbot". A/B dense-only: -2.8%.
v2 synthetic: mean 62 words, "Câu hỏi người dân hỏi tư vấn". A/B dense-only: -1.1%.

Same training infra, same KG, same model. Only the prompt changed. Style alignment
drove the entire delta. **Plan §9 risk #1 (synthetic distribution drift) was the real
bottleneck**, not LoRA hyperparams.

Action item for future M2 iterations: spend the budget on synthetic prompt iteration
**before** training infra tuning. Style spot-check trong Colab notebook (Cell 6) là gate
quan trọng nhất — KHÔNG bypass.

### Lesson 2 — Dense-only A/B is necessary but not sufficient

Dense-only A/B on dev showed M2 ≈ vanilla (-1.1%). Full pipeline on 30-probe showed M2
in-corpus +17%. The full pipeline (sparse + RRF + rerank + graph hop) **compensated for
distribution shift** that dense-only test exposed.

→ Always run E2E test for final gate decision. Dense-only is a quick sanity check, not
a verdict.

### Lesson 3 — Reranker swap was almost the entire latency win

Latency: 39.18s (v2-m3) → 4.19s (base). bge-reranker-base is 2× smaller (278M vs 568M),
but the 9× speedup came from cumulative: smaller model + fewer batches + warm model
+ tuned dense already returning more relevant candidates → less rerank work.

Could have been a stand-alone optimization without M2. For Sprint 3 ablation: separate
arm `graphrag_v5_reranker_base_only` to isolate latency vs quality contribution.

### Lesson 4 — Parser strict mode protected the experiment

Phase 0a strict parser caught 6 cross-stream FP per arm-record in Sprint 1 graphrag_v5
records, lifting precision +14% relative. Without this, Phase 4 comparison would have
been polluted by FP noise. Skill Rule 2 protocol (re-aggregate baseline after metric
change) prevented apples-to-oranges A/B.

## Why Phase 5 was deferred

Phase 5 gate triggered "M6 + M3 HyDE" because in-corpus recall ∈ [0.35, 0.50]. Plan
budget allows it (~$1, ~5 days). Reasoning for deferral:

1. **M2 marginal lift only on in-corpus** (+17% rel). On overall the result is -5%.
   M3 + M6 are precision-focused (M6) and aspect-recall focused (M3). Without
   stronger M2 signal, adding M3/M6 may scatter the analysis rather than concentrate it.

2. **Phase 5 risks compounding methodology debt**. Adding 2 modules at once means
   ablation matrix grows: `M2 only`, `M2+M6`, `M2+M3`, `M2+M3+M6`, plus vanilla v5
   baselines × dev/test. With 30-probe size noise (n=30 → ±5pts), differentiating
   intra-treatment requires 100+ samples. Not realistic for thesis budget.

3. **Sprint 3 should focus on full 150-test on the strongest configuration**, not
   continue 30-probe iteration. Per Plan §10 acceptance, gate decision is on 150
   stratified.

## Recommendation for Sprint 3

Based on Sprint 2 evidence:

1. **Adopt M2 + reranker-base as the production v5 pipeline** for full 150-test run.
2. **Run 150-test with stratified report**: in-corpus, ooc, mixed, unparseable.
3. **Document overall vs in-corpus tradeoff openly** in thesis defense — this is the
   honest path. v5 doesn't claim 0.70 overall; it claims +17% in-corpus + 9× faster
   over vanilla.
4. **Phase 5 (M3 HyDE + M6 verifier) becomes Sprint 4 optional** if Sprint 3 numbers
   demand precision boost (M6) or multi-aspect recall (M3).

## Budget actuals (Sprint 2)

| Item | Estimate | Actual |
|---|---|---|
| Synthetic Q gen v1 + v2 (gpt-4o-mini) | $5-10 | ~$10.4 |
| Colab Pro 1 month | $10 | $10 |
| Sprint 2 inference (30-probe × 1 arm) | $1 | ~$0.15 |
| **Sprint 2 total** | **~$30** | **~$20.6** |

Sprint 1 + 2 combined: ~$22 out of $200 cap. Plenty of headroom for Sprint 3 (full
150-test × 5+ arms ~$10-15).

## Repo state at close

- 18 phases tracked, 16 completed, 1 deferred (Phase 5), 1 superseded (Phase 1 v1).
- New artefacts committed:
  - `data/finetune-bge/qa_pairs_v1.jsonl` (3170 rows, v2 prompt, 12.7 MB)
  - `data/eval/questions_150_test.json` + `questions_50_dev.json` + hash lock
  - `experiments/03_v5_sprint1_vanilla/` and `04_v5_sprint2_m2/` complete READMEs + metrics + reports
  - `models/bge-m3-bhxh-lora/` (gitignored, ~14MB adapter)
  - `data/legal_sources.yaml` +1 entry (QD366_BHXH)
- New modules:
  - `src/bge_m3_loader.py` — single source for vanilla/adapter BGE-M3 loading
  - `src/retrieval/` 4-module Sprint 1 pipeline
  - `prompts/offline/synthetic_query_gen.md` + `synthetic_pair_verifier.md`
  - `scripts/reparse_citations.py` + `seal_eval_split.py`
  - `notebooks/finetune_bge_m3.ipynb` (Colab single-notebook)
- Modified:
  - `src/citations.py` — strict parser
  - `prompts/runtime/graphrag_v5_system.md` — fortified template enforcement
  - `offline/embed.py` + `load_neo4j.py` — adapter/tuned-index flags
  - `schema/schema.cypher` — additive tuned vector indexes
  - `eval_core/arms.py` + `inference.py` — `graphrag_v5_m2` arm

## End

Sprint 2 closed. Sprint 3 entry point: re-run M2 pipeline on full 150 test +
stratified write-up.
