# v5 Sprint 2 — Implementation plan

> **Status**: draft, post Sprint 1 audit ([experiments/03_v5_sprint1_vanilla/README.md](../../experiments/03_v5_sprint1_vanilla/README.md)).
> Supersedes nothing — supplements [docs/plans/v5_general_retrieval.md](v5_general_retrieval.md) §4 Sprint 2 conditional section.
> Owner: Hoàng — UIT MSc thesis.

## 1. Sprint 1 signal recap (driver of Sprint 2 design)

| Signal | Sprint 1 | Plan §10 gate | Gap |
|---|---:|---:|---:|
| recall macro (overall) | 0.236 | 0.70 | -0.46 |
| recall macro (in-corpus 14 stt) | 0.321 | 0.70 | -0.38 |
| precision macro | 0.187 | 0.80 | -0.61 |
| OOC rate | 23% | — | structural |
| latency median | 39s | 30s | -9s |
| `REFERS_TO` hop contribution | 3.5/query | — | surprisingly usable |

Bottleneck: retrieval (gold không vào top-12 ngay cả khi đã expand graph). Precision low part do parser FP — Phase 0a fix.

## 2. Decisions chốt (qua design session)

| Item | Quyết định | Rationale |
|---|---|---|
| Corpus | Giữ 3 luật (L41/L58/L45) | Plan §11 + budget cap; OOC tách report |
| M2 fine-tune | YES — primary intervention | Recall gap 0.38 không bridge được bằng prompt tuning |
| M6 verifier | Conditional, after M2 | Precision-side fix; ưu tiên recall trước |
| M3 HyDE | Conditional, after audit | Cần per-case failure mode signal |
| Corpus expansion | NO | Plan §11 chốt |
| Fine-tune compute | Colab Pro ($10/tháng) | RTX 3050 4GB không kham được fine-tune |
| Reranker swap | bge-reranker-v2-m3 → bge-reranker-base | Latency gate §10 ≤30s; trade 4% NDCG cho 2.5× speed |
| Training data source | Synthetic Q (gpt-4o-mini) + KG clause text | Promptagator/InPars/GPL precedent; no labeled dataset Vietnamese legal exists |
| False-negative defense | Multi-positive + LLM verifier + distance filter | Plan §2 median gold 2-4/Q → can't single-positive |
| Eval split | Hash-seal 150 test / 50 dev stratified | Plan §5 contract |
| Parser FP | 4-layer strict + KG validation | "Citation phải đúng luật" — user constraint |

## 3. Phased implementation (2 weeks, $30 ước tính)

### Phase 0 — Pre-flight (Day 1, blocking)

**0a. Parser FP fix** ([src/citations.py](../../src/citations.py))

4 layer defense:
1. Strict template parse: chỉ accept `[<full law title with code>, Điều X[ khoản Y[ điểm z]]]` cùng trong 1 `[...]` block. Refuse loose window matching.
2. Post-parse KG validation: mỗi citation_id phải verify với Neo4j (Article tồn tại, khoản/điểm là con). Drop nếu fail.
3. Prompt update [prompts/runtime/graphrag_v5_system.md](../../prompts/runtime/graphrag_v5_system.md): cấm tuyệt đối inline citation form "Điều X của Luật Y" — chỉ template bracket.
4. New metric `citation_validity_rate` = % parsed cite tồn tại trong KG.

**Commits sequence per skill Rule 2**:
```
commit 1: fix(citations): strict bracket parser + post-parse KG validation
commit 2: chore(metrics): re-aggregate 01_initial_eval baseline post parser-fix
commit 3: feat(metrics): add citation_validity_rate
```

**0b. Hash-seal eval split** ([data/eval/](../../data/eval/))

Script `scripts/seal_eval_split.py`:
- Categorize 200 câu: in-corpus / OOC / mixed / unparseable.
- Stratified 150 test / 50 dev (proportional per category).
- Write `questions_150_test.json` + `questions_50_dev.json` + `eval_split_hashes.json` (SHA256).
- Commit + lock — never touch again.

### Phase 1 — Synthetic Q data generation (Day 2-3)

**Script**: `offline/build_synthetic_qa.py`

Pipeline per clause (1585 total):
```
for clause in KG:
    # Step A: Sinh 2 query
    queries = gpt-4o-mini.gen(
        clause.text,
        prompt="Sinh 2 câu hỏi tự nhiên người dân BHXH có thể hỏi..."
    )  # ~$0.0001/clause

    # Step B: Cypher candidates (same Article + same Chapter)
    candidates = neo4j.query(
        f"MATCH (cl {{id: '{clause.id}'}})<-[:HAS_CLAUSE]-(art)
           -[:HAS_CLAUSE]->(sib) WHERE sib <> cl
           OPTIONAL MATCH (art)<-[:HAS_ARTICLE]-(ch)
                          -[:HAS_ARTICLE]->(other_art)
                          -[:HAS_CLAUSE]->(ch_sib)
           RETURN sib, ch_sib LIMIT 15"
    )

    # Step C (Layer 3 distance filter):
    candidates = [c for c in candidates
                  if 0.3 ≤ cosine(emb_vanilla(Q), emb_vanilla(c.text)) ≤ 0.85]

    # Step D (Layer 2 LLM verifier):
    for c in candidates:
        label = gpt-4o-mini.classify(Q, c.text)  # YES/PARTIAL/NO
        if label == YES: positives.append(c)
        elif label == NO: negatives.append(c)
        # PARTIAL → drop

    # Step E: format multi-positive row
    output_row = {
        "query": Q,
        "pos": [clause.text] + [p.text for p in positives],
        "neg": [n.text for n in negatives] + [random_distant.text] * 3,
        "_meta": {"source_clause_id": clause.id, ...}
    }
```

**Output**: `data/synthetic/qa_pairs_v1.jsonl` ~3170 rows.

**Cost estimate**:
- Q generation: 1585 × $0.0001 = $0.16
- LLM verifier: 1585 × 2 Q × ~8 candidates × $0.0002 = $5.07
- **Total ~$5.2** (well within plan §8 $5-10 budget)

**Audit deliverables**:
- `data/synthetic/stats.json`: mean pos/row, mean neg/row, % multi-positive (target 20-40%).
- `data/synthetic/style_audit.md`: 30 synthetic Q side-by-side với 30 dev real Q.

**Leak protection invariants**:
- Synthetic Q sinh **chỉ** từ clause text. Không feed real Q content vào prompt.
- LLM verifier không thấy `questions_200.json`.
- Hash của 50_dev kiểm tra trước-sau training — không thay đổi.

### Phase 2 — LoRA fine-tune BGE-M3 (Day 4-6)

**Notebook**: `notebooks/finetune_bge_m3_colab.ipynb`

```python
# Colab Pro A100 (40GB) hoặc V100 (16GB)
from FlagEmbedding import BGEM3FlagModel, FineTuneArgs
from peft import LoraConfig

base = "BAAI/bge-m3"
lora_cfg = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["query", "key", "value", "dense"],
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION",
)

trainer_args = FineTuneArgs(
    train_data="data/synthetic/qa_pairs_v1.jsonl",
    output_dir="models/bge-m3-bhxh-lora",
    num_train_epochs=2,
    per_device_train_batch_size=4,    # tightly per V100 VRAM
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    temperature=0.02,                  # InfoNCE temperature
    use_lora=True,
    lora_config=lora_cfg,
    eval_strategy="steps",
    eval_steps=100,
    eval_data="data/eval/questions_50_dev.json",  # ONLY for ckpt selection — no leak
    metric_for_best_model="recall@10_dev",
)
```

**Training time**: ~2h trên A100, ~4h trên V100. Adapter output ~50MB.

**Style spot-check** (in-notebook before training):
- Render 30 random synthetic Q + 30 random dev Q side-by-side.
- Manual mark: high-overlap / medium / low style match.
- Abort condition: nếu >50% mismatch on critical dimensions (formality, length, narrative-vs-direct) → revise Q generation prompt.

**Output**:
- `models/bge-m3-bhxh-lora/adapter_model.safetensors`
- `models/bge-m3-bhxh-lora/training_log.json`
- `notebooks/finetune_bge_m3_colab.ipynb` (tracked)

### Phase 3 — Re-encode + load tuned index (Day 7)

**Offline changes (additive, không phá B5/B6 cũ)**:

[offline/embed.py](../../offline/embed.py) — thêm `--adapter-path` flag:
```bash
python -m offline.embed --adapter-path models/bge-m3-bhxh-lora \
    --output data/graph/processed/embeddings_tuned.parquet
```

[offline/load_neo4j.py](../../offline/load_neo4j.py) — thêm `--load-tuned` flag:
```bash
python -m offline.load_neo4j --load-tuned \
    --embeddings data/graph/processed/embeddings_tuned.parquet \
    --index-name clause_vec_tuned
```

[schema/schema.cypher](../../schema/schema.cypher) — thêm VECTOR INDEX `clause_vec_tuned` (1024-d cosine), giữ `clause_vec` cũ song song để A/B.

### Phase 4 — Experiment 04_v5_sprint2_m2 + decision gate (Day 8-9)

**Folder**: `experiments/04_v5_sprint2_m2/`

```yaml
# config.yaml
name: "v5 Sprint 2 — M2 fine-tuned BGE-M3 (30-probe)"
date: "..."
dataset:
  questions: data/eval/questions_150_test.json
  n: 30
parent: 03_v5_sprint1_vanilla
arms:
  graphrag_v5:    { mode: inherit }     # baseline v5 vanilla
  graphrag_v5_m2: { mode: run, model: gpt-4o-mini }   # NEW M2 arm
  graphrag:       { mode: inherit }     # original baseline
  llm_only:       { mode: inherit }     # control
```

**Wiring**:
- `graphrag_v5_m2` runner = `graphrag_v5` runner nhưng `HybridRetriever.DENSE_INDEX = 'clause_vec_tuned'` + reranker swap `bge-reranker-base`.
- Configurable via env: `V5_DENSE_INDEX=clause_vec_tuned V5_RERANKER_MODEL=BAAI/bge-reranker-base`.

**Decision gate** (after Phase 4 metrics computed):

| M2 recall macro in-corpus | → Sprint 2 Phase 5 |
|---|---|
| ≥ 0.50 | Add M6 verifier (push precision) |
| 0.35 – 0.50 | Add M6 + M3 HyDE (recall mở rộng + precision filter) |
| < 0.35 | **STOP**. Document M2 vanilla insufficient. Possible Sprint 3 = pivot scope. |

### Phase 5 — Conditional M6 ± M3 (Day 10-12)

**M6 — Verifier** (if triggered):
- `runtime/v5_verifier.py`: per citation_id predicted, call `claude-haiku-4-5` (different family, không OpenAI bias) với prompt:
  ```
  Câu hỏi: {Q}
  Câu trả lời: {answer text}
  Citation: {clause text} (id={cid})
  Citation này có entail (suport) câu trả lời không? YES/PARTIAL/NO
  ```
- Drop citation nếu NO. Output: `citation_ids_verified` field.
- Cost: ~$0.001/citation, 30-probe ~$0.1.
- Folder: `experiments/05_v5_sprint2_m2_m6/`.

**M3 — HyDE** (if triggered):
- `src/retrieval/hyde.py`: wrap HybridRetriever:
  ```python
  hypothetical = gpt-4o-mini.gen(f"Trả lời ngắn câu hỏi luật BHXH: {Q}")
  emb_aug = emb(Q) + 0.5 * emb(hypothetical)
  retrieve dense path using emb_aug
  # sparse path không đổi
  ```
- Fuse trong RRF cùng dense/sparse path.
- Cost: ~$0.0005/query, 30-probe ~$0.015.
- Folder: `experiments/06_v5_sprint2_full/`.

### Phase 6 — Final A/B + write-up (Day 13-14)

**Compare on same 30 stt**:
| arm | recall_macro | precision_macro | latency | notes |
|---|---|---|---|---|
| graphrag (baseline) | ~0.094 | ~0.056 | 5s | inherit |
| graphrag_v5 (vanilla) | ~0.236 | ~0.187 | 39s | inherit |
| graphrag_v5_m2 | ? | ? | ? | Phase 4 |
| graphrag_v5_full | ? | ? | ? | Phase 5 (if any) |

**Write-up** trong README của final experiment folder:
- Lift breakdown per intervention (M2 alone, M2+M6, M2+M6+M3).
- Stratified in-corpus / OOC.
- Per-case failure analysis if recall ceiling reached.
- Decision Sprint 3: full 150-test với cấu hình thắng, hay pivot scope.

## 4. Files / modules touched

### New files
```
docs/plans/v5_sprint2_implementation.md     ← this file
scripts/seal_eval_split.py                  ← Phase 0b
data/eval/questions_150_test.json           ← Phase 0b output (committed)
data/eval/questions_50_dev.json             ← Phase 0b output (committed)
data/eval/eval_split_hashes.json            ← Phase 0b lock
offline/build_synthetic_qa.py               ← Phase 1
data/synthetic/qa_pairs_v1.jsonl            ← Phase 1 output (committed? size-dependent)
data/synthetic/stats.json                   ← Phase 1 audit
data/synthetic/style_audit.md               ← Phase 1 audit
notebooks/finetune_bge_m3_colab.ipynb       ← Phase 2
models/bge-m3-bhxh-lora/                    ← Phase 2 output (gitignored, link release)
runtime/v5_verifier.py                      ← Phase 5 (if M6)
src/retrieval/hyde.py                       ← Phase 5 (if M3)
experiments/04_v5_sprint2_m2/               ← Phase 4
experiments/05_v5_sprint2_m2_m6/            ← Phase 5 (conditional)
experiments/06_v5_sprint2_full/             ← Phase 5 (conditional)
```

### Modified files
```
src/citations.py                            ← Phase 0a strict parser + KG validation
prompts/runtime/graphrag_v5_system.md       ← Phase 0a prompt tighten
docs/changelog.md                           ← Phase 0a parser change rationale
eval_core/metrics.py                        ← Phase 0a citation_validity_rate metric
schema/schema.cypher                        ← Phase 3 clause_vec_tuned index
offline/embed.py                            ← Phase 3 --adapter-path flag
offline/load_neo4j.py                       ← Phase 3 --load-tuned flag
eval_core/inference.py                      ← Phase 4 graphrag_v5_m2 runner
eval_core/arms.py                           ← Phase 4 register graphrag_v5_m2
experiments/01_initial_eval/metrics/        ← Phase 0a re-aggregated
```

## 5. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Synthetic Q distribution gap với real Q | **HIGH** | Style spot-check Phase 2 + held-out 150 test never seen |
| LLM verifier classifies PARTIAL→DROP overly aggressive → ít hard negative | MEDIUM | Distance filter Layer 3 safety net; tune threshold post-hoc |
| LoRA r=16 capacity quá nhỏ cho 3k pairs | LOW | Monitor train/val loss; bump r=32 nếu underfit |
| Colab session timeout 12h | LOW | Save checkpoint mỗi 500 step; resume if disconnect |
| Reranker swap (v2-m3 → base) tụt quality > 4% | MEDIUM | Pin baseline result Phase 4; nếu drop quá → revert chỉ Phase 5 |
| M6 (claude-haiku) drift API hoặc giá tăng | LOW | Cost cap $0.5 cho 30-probe; fallback gpt-4o-mini với rotate prompt |
| Schema migration `clause_vec_tuned` xung đột với existing index | LOW | Additive, `IF NOT EXISTS` clause; rollback bằng DROP INDEX |
| Parser strict mode reject legitimate citation từ baseline → recall tụt | MEDIUM | Re-aggregate 01_initial_eval trước cite số mới (Rule 2 protocol) |
| Catastrophic forgetting pretrain knowledge sau LoRA | LOW | LoRA base frozen by design; eval trên general MTEB Vietnamese tasks nếu cần (Sprint 3) |

## 6. Acceptance criteria (Sprint 2 end)

| Metric | Sprint 2 threshold | Source |
|---|---|---|
| recall_macro_in-corpus trên 30-probe | ≥ 0.50 (= "M2 working") | Phase 4 |
| precision_macro overall trên 30-probe | ≥ 0.30 (parser fix lift baseline) | Phase 4 |
| citation_validity_rate | ≥ 0.95 | Phase 0a metric |
| latency_median | ≤ 30s (Plan §10 gate) | Phase 4 với reranker base |
| All experiments có README đầy đủ với pre-commitment + result | qualitative | Phase 6 |
| 50 dev hash bằng trước-sau training | bit-identical | Phase 0b lock |

Nếu **ANY** gate fail → Phase 6 write-up phân tích root cause; Sprint 3 lên kế hoạch fix hoặc giảm scope.

## 7. Budget tracking

| Item | Estimate | Cap |
|---|---|---:|
| Synthetic Q generation (gpt-4o-mini) | $0.16 | — |
| LLM verifier candidates | $5.07 | — |
| Colab Pro 1 tháng | $10 | — |
| Sprint 2 inference (30-probe × ~3 arms × multiple runs) | ~$1 | — |
| M6 verifier API (claude-haiku) | ~$0.5 if triggered | — |
| M3 HyDE generator | ~$0.1 if triggered | — |
| **Sprint 2 total** | **~$17** | $200 cap |

Còn ~$183 dự trữ cho Sprint 3 (full 150 test × ~5 arms = ~$5-10) + iteration buffer.

## 8. Open questions for confirm trước khi proceed

1. **Schema migration timing**: chốt thêm `clause_vec_tuned` ngay từ Phase 0 hay sau Phase 2? Đề xuất sau Phase 2 (cần có model trước mới biết encode được). [DECISION: defer to Phase 3 — additive, ad-hoc.]
2. **Sprint 2 inference run on 30 probe (same as Sprint 1) hay 50 dev?**: 30 probe để A/B trực tiếp với Sprint 1; 50 dev rộng hơn nhưng dev không phải test set. Đề xuất: chạy cả 2, report 30 probe làm headline, 50 dev làm sanity.
3. **Reranker swap**: thực hiện ngay Phase 4 (cùng với M2) hay tách thành ablation riêng? Đề xuất: gộp vào Phase 4 — interest là pipeline as a whole, không phải từng component isolated.

End of plan.
