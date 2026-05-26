# Colab Notebooks — Legal KB Experiment

Run experiment pipeline trên Google Colab Pro (T4 GPU, 16GB VRAM).

## 📓 Available notebooks (chọn 1)

### Option A: 2 notebooks tách theo experiment (recommended)

| Notebook | Scope | Parallelism | Time |
|---|---|---|---|
| **`colab_r1_5arm.ipynb`** | 1 GPT model × 5 arms × 200 q = **1000 inferences** | 5 arms concurrent | ~3h |
| **`colab_r2_multimodel.ipynb`** | N GPT models × 2 elite arms × 200 q | **2D**: arms × models concurrent (6+ combos) | ~3-4h |

**R1 use case**: Test 1 model toàn diện qua 5 retrieval/reasoning variants → so sánh approach.
**R2 use case**: Compare nhiều models trên cùng 2 elite arms → so sánh model capability.

### Option B: All-in-one (everything in 1 file)

| Notebook | Scope |
|---|---|
| `colab_full_pipeline.ipynb` | R1 + R2 together — full matrix |

### Option C: 3-notebook split (legacy)

| `01_colab_setup.ipynb` | `02_colab_inference.ipynb` | `03_colab_metrics_report.ipynb` |
|---|---|---|

## ✅ Both R1 + R2 notebooks support

### 17 metrics
| Category | Metrics |
|---|---|
| Citation | `citation_validity`, `citation_recall`, `citation_precision` |
| RAGAS | `faithfulness`, `answer_relevance` (RAGAS Es EACL'24) |
| Hallucination | `content_hallucination_rate`, `invented_citation_rate`, `hallucination_rate` (legacy) (Magesh JELS'25) |
| Embedding | `bertscore_f1`, `bertscore_p`, `bertscore_r` (Zhang ICLR'20) |
| Resource | `cost_usd`, `latency_s` |
| Prolog (Logic-LM) | `prolog_success_rate`, `first_try_success_rate`, `repair_invoked_rate`, `avg_repair_rounds` (Pan EMNLP'23) |
| Pairwise | `consensus`, `vote_ab`, `vote_ba` (Zheng NeurIPS'23) |
| Quality | `api_error_rate` |
| Statistical | McNemar, Bootstrap 95% CI, Bonferroni correction |

### Parallel execution

| Notebook | Parallel dim 1 | Parallel dim 2 |
|---|---|---|
| R1 | arms (5 concurrent) | — |
| R2 | arms (2) | models (N) → **arms × models combos all parallel** |

### Visualizations 📈

| # | Viz | R1 | R2 |
|---|---|:---:|:---:|
| 1 | Prolog success bar chart | per arm | per (model, arm) grouped |
| 2 | Metric matrix heatmap (RdYlGn) | arm × metric | (model, arm) × metric |
| 3 | Cost vs faithfulness scatter | bubble = latency | bubble + marker per arm |
| 4 | Latency box plot (log scale) | per arm | per (model, arm) |
| 5 | Pairwise consensus stacked bar | vs graphrag baseline | GR vs NR per model |
| 6 | Sortable summary DataFrame (styled) | per arm | per (model, arm) |
| 7 | Markdown reports inline (FINAL_REPORT, significance) | ✓ | ✓ |
| **R2-only** | Faithfulness Δ (GR − NR) per model | — | ✓ bar chart |

## ⚙️ Configurable knobs

| Notebook | Knobs |
|---|---|
| R1 | `R1_MODEL` (1 model), `JUDGE_MODEL`, `N_QUESTIONS`, `MAX_PARALLEL`, `BACKFILL_PLAIN`, `ARMS` (5) |
| R2 | `R2_MODELS` (list), `JUDGE_MODEL`, `N_QUESTIONS`, `MAX_PARALLEL`, `BACKFILL_PLAIN`, `ARMS` (2) |
| All | `REPO_GDRIVE`, `RESULTS_GDRIVE` |

## Prerequisites

1. **Google Colab Pro subscription** (T4 GPU minimum)
2. **OpenAI API key** với credit ($20+ recommended). **Tier 4+** cho R2 (6+ parallel)
3. **Neo4j Aura account** (free tier: aura.io → tạo AuraDB instance)
4. **Code repo** push lên GDrive: `/MyDrive/legal-graph-kb`
5. **Colab secrets** (🔑 icon):
   - `OPENAI_API_KEY`
   - `NEO4J_URI`
   - `NEO4J_USER`
   - `NEO4J_PASSWORD`

## Speedup vs local RTX 3050 Laptop

| Phase | Local | Colab T4 | Speedup |
|---|---|---|---|
| Inference parallel | Blocked by paging file → seq | True parallel | **3-4×** |
| BGE-M3 embedding | 4GB VRAM bottleneck | 16GB headroom | **5-8×** |
| BERTScore (1000 records) | ~10 min | ~2 min | **5×** |
| Full workflow R1 | ~6h | ~3h | **2×** |
| Full workflow R2 | ~12h | ~3-4h | **3×** |

## Cost

- Colab Pro: prepaid subscription
- Neo4j Aura free tier: $0
- OpenAI API: R1 ~$2-5 (depends on `R1_MODEL`), R2 ~$15-20 (gpt-5-mini reasoning)
- Backfill plain_answer: ~$0.30-0.50

## Caveats

- Colab session timeout 12-24h (Pro). R1 ~3h, R2 ~3-4h → 1 session đủ
- GDrive IO chậm hơn local SSD ~3× — notebook auto-symlink results → GDrive
- OpenAI rate limits cứng cap parallelism — Tier 4+ recommended cho R2 6+ concurrent
- Pipeline idempotent: timeout → re-run sẽ skip records đã có
