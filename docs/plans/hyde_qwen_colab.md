# Plan — HyDE retrieval with Qwen 2.5 3B Instruct on Colab Free (experiment 08)

- **Status**: Accepted (2026-05-30), ready to implement
- **Owner**: Nguyễn Hữu Hoàng
- **Target experiment**: `experiments/08_hyde_retrieval/`
- **Discussion thread**: design iterated through gpt-4o-mini (v1) → local Qwen
  (v2) → Colab Free Qwen (v3, this plan)

## TL;DR

Implement HyDE (Hypothetical Document Embeddings, Gao et al. 2022) on the
BGE-M3 dense retrieval channel of `V5RetrievalPipeline`, generating the
hypothetical document **locally on Colab Free T4** with Qwen 2.5 3B
Instruct (no OpenAI API). Compare 4 retrieval arms — `dense`,
`dense_hyde`, `full_rerank`, `full_rerank_hyde` — on full 200 BHXH
questions with the exp 07 metric suite (recall, precision, F1, NDCG @K
+ R-Precision, MRR for K ∈ {12, 20, 30, 50, 70, 100, all}).

## Context

### Why HyDE for this project

`full_rerank` funnel analysis at K=12 (in_corpus stratum, n=151) — see
[`experiments/06_retrieval_dense_vs_full/report/funnel_full_rerank_K12.md`](../../experiments/06_retrieval_dense_vs_full/report/funnel_full_rerank_K12.md)
— shows the dense channel is the dominant signal source. Cross-encoder
rerank1 lifts R@12 by +8.9pp (the biggest single contribution in the
pipeline), but it can only re-rank what dense + sparse surfaced. Better
dense candidates → better rerank pool → better final.

The 200 BHXH questions are written in casual storytelling style
("Bà Minh Châu (Long An) ký hợp đồng lao động theo diện làm việc bán
thời gian…") while the KG clauses are formal legal text
("Người lao động làm việc theo hợp đồng lao động không xác định thời
hạn…"). HyDE is designed exactly for this style gap: instead of
embedding the question, generate a hypothetical formal-legal-style
passage that *would* answer it, then embed and search with that. The
hypothetical sits closer to real clauses in embedding space.

### Why Qwen 2.5 3B Instruct

BGE-M3 is encoder-only — it cannot generate text. HyDE requires a
generative model. After ruling out (a) gpt-4o-mini (user prefers
reproducible, no API), (b) local 3B model (insufficient local VRAM):
Qwen 2.5 3B Instruct on Colab Free T4 is the chosen combination.

Why Qwen 2.5 specifically (vs Phi-3 / Llama 3.2 / VinaLLaMA):
- Multilingual pre-training includes explicit Vietnamese
- Instruction-tuned with ChatML chat template
- Apache 2.0 license (thesis + future open-source OK)
- 6 GB fp16, fits T4 16 GB alongside BGE-M3 + reranker

Why Colab Free T4:
- 16 GB VRAM clears the budget (~10 GB total: 1.2 GB BGE-M3 + 0.5 GB
  reranker + 6 GB Qwen + ~2 GB activations)
- $0 cost
- 12 h session limit + 90 min idle disconnect — accepted risk (D14)

## Design decisions (locked)

| # | Decision | Value | Rationale |
|---:|---|---|---|
| D1 | Generator model | `Qwen/Qwen2.5-3B-Instruct` | Vietnamese-aware, instruction-tuned, fits T4 |
| D2 | Hypothetical doc count N | 1 | Paper used 8, but N=1 sufficient to detect lift; can scale later. Cost = wall time, not money |
| D3 | HyDE plug-in point | Replace dense query embedding only. Sparse/BM25 keeps raw question text | Isolates HyDE contribution; BM25 with hypothetical doc shifts term distribution and contaminates the ablation |
| D4 | Dense index | `clause_vec_tuned` (LoRA-adapted BGE-M3) | Match production. Optional ablation with vanilla `clause_vec` if HyDE underperforms — LoRA was fine-tuned on Q→clause pairs, may have weakened doc-style query encoding |
| D5 | Prompt design | Vietnamese system+user, target 200-400 từ formal legal style, **forbid fabricating "Điều X / Khoản Y"**, drop personal details (names, dates) from the question | Fabricated citations contaminate BM25 (if reused) and shift embedding away from real clauses. Personal details ("Bà Minh Châu", "53 tuổi") aren't legal terms — generator should focus on the underlying rule |
| D6 | Cache | `artifacts/hyde/<model_id_safe>/<sha256(question + prompt_sha + n + max_new_tokens)>.json` | Re-run experiment = free. Schema includes model commit hash + revision for audit |
| D7 | Baseline reproduction | Re-run `dense` and `full_rerank` inside exp 08 (no `mode: inherit` from exp 06) | Strict no-inherit per user request. Cost ~0 since both arms are retrieval-only and Neo4j-only |
| D8 | Metric K-set | K ∈ {12, 20, 30, 50, 70, 100, all} + R-Precision + MRR + NDCG@K per K | Matches exp 07 — comparable across experiments |
| D9 | Precision mode | fp16. Fallback 4-bit (bitsandbytes) only if OOM on T4 | T4 has 16 GB, fp16 should fit. 4-bit costs 3-5% quality. |
| D10 | Batch size | Generation `batch=4` | T4 throughput optimisation. 200 q × ~7 s single → ~3 s/q batched effective |
| D11 | Storage on Colab | Mount Google Drive. Clone repo to `/content/drive/MyDrive/legal-graph-kb/`. All cache + results land in Drive automatically | Survives session disconnect. ~7 GB total (repo + HF cache + results) fits Drive Free 15 GB |
| D12 | Repo transfer | Clone from GitHub via PAT (user confirmed GitHub repo exists) | Cleanest. Branch strategy: dedicated `exp/08-hyde` branch, PR back to `main` on completion |
| D13 | Secrets | Colab Secrets sidebar: `GITHUB_PAT`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` | Never hardcode in notebook cells; clear all outputs before commit |
| D14 | Idle protection | JavaScript ping every 60s in a notebook cell. User keeps tab active. Accepted disconnect risk | Free tier — no Pro upgrade per user confirmation |
| D15 | Sync results to local | `git push` from Colab to `exp/08-hyde` branch on completion. User pulls locally for ADR + changelog work | Cleanest. Drive UI download is fallback |

## Success criteria

After full 200 run + metrics, **HyDE is considered a win** if any one of:

- `dense_hyde` lifts in_corpus R@12 by **≥ +3pp absolute** over `dense`
- `dense_hyde` lifts in_corpus NDCG@10 by **≥ +5% relative** over `dense`
- `full_rerank_hyde` lifts in_corpus R-Precision by **≥ +15% relative**
  over `full_rerank` (matches the magnitude exp 06 saw from rerank itself)

**Win → ADR 002**: HyDE-with-Qwen becomes a documented optional default
behind a config flag, with the same revisit triggers as ADR 001.

**No win** (all three metrics within ± noise): document negative result
in the experiment README. Do NOT change production.

## Code surface — file by file

### New files

1. **`prompts/runtime/hyde_generate.md`**
   - System: define role as expert in Vietnamese social insurance law,
     instruct to write a legal-text-style passage answering the
     hypothetical case. Explicit prohibitions: no "Điều X", "Khoản Y",
     "theo quy định tại"; no personal names; no specific dates/numbers
     from the question.
   - User: `{question}` — interpolated at call time via
     `src.prompts.load_prompt`.
   - Target output: 200-400 từ, formal "Người lao động", "BHXH bắt
     buộc", "chế độ thai sản" terminology.

2. **`src/retrieval/hyde.py`**
   ```python
   class QwenHydeGenerator:
       def __init__(
           self,
           model_id: str = "Qwen/Qwen2.5-3B-Instruct",
           n: int = 1,
           cache_dir: str | Path = "artifacts/hyde",
           prompt_path: str = "runtime/hyde_generate.md",
           dtype: str = "fp16",            # fp16 | bf16 | 4bit
           batch_size: int = 4,
           max_new_tokens: int = 400,
           device: str = "cuda",
           seed: int = 0,
       ): ...

       def generate(self, question: str) -> list[str]:
           """Returns N hypothetical docs. Cache-aware."""

       def generate_batch(self, questions: list[str]) -> list[list[str]]:
           """Batched single-pass generation for the runner."""

       def embed_query_callable(self, embed_model) -> Callable[[str], np.ndarray]:
           """Returns: question → mean-pooled embedding of N HyDE docs."""

       # Internals
       def _load_model(self): ...           # lazy, idempotent
       def _apply_chat_template(self, q): ...
       def _cache_key(self, question): ...  # sha256
       def _cache_get(self, key): ...
       def _cache_put(self, key, payload): ...
   ```

   Cache file schema:
   ```json
   {
     "question": "...",
     "model_id": "Qwen/Qwen2.5-3B-Instruct",
     "model_revision": "<commit-sha-from-HF>",
     "prompt_sha": "<sha256-of-prompt-file>",
     "n": 1,
     "max_new_tokens": 400,
     "generated_at": "2026-05-31T...",
     "generated_docs": ["..."]
   }
   ```

### Modified files

3. **`src/retrieval/hybrid_retriever.py`**
   - Add optional constructor param:
     `query_encoder: Callable[[str], np.ndarray] | None = None`.
   - Modify `_dense_search`:
     ```python
     if self._query_encoder is not None:
         q_emb = self._query_encoder(query).tolist()
     else:
         q_emb = self._embed_model.encode(
             [query], normalize_embeddings=True, show_progress_bar=False,
         )[0].tolist()
     ```
   - Default `None` keeps existing behaviour byte-for-byte.

4. **`src/retrieval/pipeline.py`**
   - Add constructor param:
     `hyde: QwenHydeGenerator | None = None`.
   - When `hyde is not None`, build encoder closure with
     `hyde.embed_query_callable(self.embed_model)` and pass into the
     `HybridRetriever` constructor.
   - Add public method:
     ```python
     def retrieve_dense_only_hyde(self, question: str, top_k: int | None = None
                                  ) -> RetrievalOnlyAnswer:
         """Pure dense retrieval using HyDE embedding. Requires hyde to be set."""
     ```
     Mirror of `retrieve_dense_only` but uses `self.hyde.embed_query_callable`.

### New scripts

5. **`scripts/exp08_test_one.py`** — Phase 3 dry-run for stt=2.
   - Load Qwen, print pre/post VRAM via `torch.cuda.memory_allocated()`.
   - Generate hypothetical, print full text.
   - Embed + Neo4j dense search, print top-12 article IDs.
   - Compare to `experiments/06_retrieval_dense_vs_full/results/full_rerank/A2.json::dense_article_ids`.
   - Print delta: gold rank in HyDE top-12 vs vanilla, articles added,
     articles removed.
   - Run twice — second invocation must be cache-hit (no model call).

6. **`scripts/exp08_run.py`** — same shape as `exp07_run.py` with 4 arms:
   - `dense`         → `retrieve_dense_only`
   - `dense_hyde`    → `retrieve_dense_only_hyde`
   - `full_rerank`   → `retrieve_only` (no hyde)
   - `full_rerank_hyde` → `retrieve_only` with `hyde` set
   - Pipeline constructed once with `hyde=QwenHydeGenerator()` — non-HyDE
     arms swap to a parallel pipeline without hyde (or temporarily clear
     the encoder; cleaner = two pipeline instances).

7. **`scripts/exp08_metrics.py`** — clone `exp07_metrics.py`,
   - `ARMS = ("dense", "dense_hyde", "full_rerank", "full_rerank_hyde")`
   - same K-set, same metrics
   - update output paths to exp 08 folder

8. **`scripts/exp08_funnel.py`** — clone `exp06_funnel.py`,
   - Operates on `full_rerank_hyde` records
   - Useful to see whether HyDE changes the rerank1 / rerank2 lift profile

### New notebook

9. **`notebooks/exp08_hyde_colab.ipynb`**
   - Phase 0 cells: runtime check, Drive mount, repo clone, install, env
     vars, model pre-download, idle keepalive
   - Phase 3 cell: run `exp08_test_one.py`, manual gate
   - Phase 5 cell: pilot 5
   - Phase 6 cell: full 200 (subprocess streaming)
   - Phase 7 cell: metrics
   - Phase 8 cell: funnel
   - Phase 9 cell: `git add` + `git push exp/08-hyde`
   - **Important**: clear all outputs before commit (PATs / Neo4j
     credentials must not leak)

### New experiment folder

10. **`experiments/08_hyde_retrieval/`**
    - `config.yaml` — name, description (Qwen 2.5 3B, Colab T4), date,
      `parent: null`, `arms: {}` (retrieval-only audit, eval_core CLI
      not used)
    - `README.md` — What / Why / Setup / Expected outcome / Result
      summary (fill after run)
    - `.gitignore` — `results/` ignored

## Phase plan (chronological)

### Phase 0 — Colab setup (one-time per branch, ~10 min)

In notebook cells 0.1 → 0.7:
1. Runtime → Change runtime type → T4 GPU
2. `nvidia-smi` verify
3. `drive.mount('/content/drive')`
4. `git clone https://{PAT}@github.com/{user}/legal-graph-kb.git /content/drive/MyDrive/legal-graph-kb` (one-time)
5. `%cd /content/drive/MyDrive/legal-graph-kb && git checkout exp/08-hyde`
6. `%pip install -e ".[dev,eval]" --quiet && %pip install bitsandbytes accelerate --quiet`
7. Set env vars from `userdata.get(...)` for Neo4j
8. `huggingface_hub.snapshot_download('Qwen/Qwen2.5-3B-Instruct', cache_dir='/content/drive/MyDrive/hf_cache')`
9. `os.environ['HF_HOME'] = '/content/drive/MyDrive/hf_cache'`
10. Idle keepalive JS (optional)

### Phase 1 — HyDE module (local commit)

Create `src/retrieval/hyde.py` + `prompts/runtime/hyde_generate.md`.
Test import locally (no GPU needed for syntactic test). Commit.

### Phase 2 — Pipeline wire (local commit)

Modify `HybridRetriever` + `V5RetrievalPipeline` as in section above.
Verify existing tests in `tests/` still pass on local CPU (vanilla
path unchanged). Commit.

### Phase 3 — 1-record dry-run + GATE (Colab)

Run `scripts/exp08_test_one.py` on Colab. **Hand off to user** with:
- Hypothetical doc full text printed
- Top-12 dense vanilla vs dense_hyde for stt=2
- Gold rank delta

User checklist:
- [ ] Hypothetical style matches legal text (formal terminology)
- [ ] No "Điều X", "Khoản Y", numbered citations
- [ ] Content relevant to the question's legal topic
- [ ] Gold article (`L58_2014.A2` for stt=2) still in top-12 after HyDE
- [ ] No suspiciously "off-topic" articles climbing into top-3

**If any fail**: revise prompt in `prompts/runtime/hyde_generate.md`,
re-run dry-run. Loop until OK.

### Phase 4 — Experiment 08 scaffold

Create `experiments/08_hyde_retrieval/{config.yaml, README.md, .gitignore}`.
Commit.

### Phase 5 — Pilot 5

`!python scripts/exp08_run.py --stt 1-5 --verbose` in notebook. Verify
4 arms produce records, cache populated, no OOM.

### Phase 6 — Full 200

`!python scripts/exp08_run.py` in notebook subprocess. ~30-35 min on
T4 first run. Drive persistence means session disconnect is recoverable
(idempotent runner skips done records on resume).

### Phase 7 — Metrics

`!python scripts/exp08_metrics.py` writes
`experiments/08_hyde_retrieval/metrics/academic_metrics.{json,csv}` +
`report/academic_report.md`.

### Phase 8 — Funnel

`!python scripts/exp08_funnel.py` writes
`experiments/08_hyde_retrieval/report/funnel_full_rerank_hyde_K12.md`.

### Phase 9 — Sync back

```python
!git -C /content/drive/MyDrive/legal-graph-kb add experiments/08_hyde_retrieval docs/
!git -C /content/drive/MyDrive/legal-graph-kb commit -m "exp 08 — HyDE results"
!git -C /content/drive/MyDrive/legal-graph-kb push origin exp/08-hyde
```

User pulls locally, opens PR `exp/08-hyde → main`.

### Phase 10 — Report + ADR (local)

Update `experiments/08_hyde_retrieval/README.md` Result Summary.
Update `docs/changelog.md`.
If HyDE wins per success criteria → create `docs/decisions/002_hyde_retrieval.md` (or update ADR 001).

## Risks

| risk | severity | mitigation |
|---|---|---|
| Qwen 2.5 3B Vietnamese quality insufficient | medium | Phase 3 dry-run eyeball; swap to Qwen 2.5 7B (needs ≥ 16 GB → tight on T4 free) or fall back to gpt-4o-mini if catastrophic |
| OOM on T4 (BGE-M3 + reranker + Qwen 3B exceeds 16 GB during batch generation) | medium | Default fp16 should fit. Fallback to `dtype="4bit"` via bitsandbytes — costs 3-5% quality |
| Generation latency too slow (CPU fallback if T4 unavailable) | high if T4 missing | Phase 0 hard-check; abort if no GPU. Don't try CPU — would take hours |
| Qwen fabricates `Điều X` / `Khoản Y` despite prompt | medium | Prompt explicit prohibition + negative example. Phase 3 gate catches it |
| Qwen paraphrases the question instead of generating a doc | medium | Prompt: "bỏ qua tên/số liệu cá nhân, chỉ viết quy định pháp luật áp dụng chung". Phase 3 gate catches it |
| LoRA-tuned BGE-M3 (`clause_vec_tuned`) was trained Q→clause; doc-style HyDE input may underperform | medium | Optional ablation: re-run HyDE with vanilla `clause_vec` index. Document in README |
| Colab session disconnect mid-full-200 | medium (free tier) | Drive persistence + idempotent runner = lossless. Just re-run cell |
| Drive 15 GB quota exceeded | low | Estimated 7 GB total. Monitor. Worst case: delete `~/.cache/huggingface/` after experiment |
| OOC stratum (8 questions) still 0 recall after HyDE | low — known | Stratified report isolates; doesn't kill macro |

## Cost

- **API**: $0 — fully local on Colab
- **Compute**: $0 — Colab Free
- **Storage**: 7 GB / 15 GB Drive (repo 500 MB + HF cache 6 GB + results 10 MB)
- **Wall time** (user-attended):
  - First session: ~45-60 min (setup + first download + run)
  - Resumed sessions: ~15-20 min (cache hits)

## Prerequisites (user actions before Phase 0)

1. **GitHub**: confirm `main` branch is pushed up to date.
   Create branch `exp/08-hyde` from `main`.
2. **Colab Secrets** (sidebar key icon → New secret):
   - `GITHUB_PAT` — fine-grained PAT with repo write access
   - `NEO4J_URI` — copy from local `.env`
   - `NEO4J_USER` — `neo4j` typically
   - `NEO4J_PASSWORD` — from local `.env`
   - `NEO4J_DATABASE` — `neo4j` typically
3. **Drive**: ensure ≥ 8 GB free.
4. **Cookie**: enable third-party cookies for the Colab tab (Drive mount
   requirement on some browsers).

## References

- [`docs/plans/v5_general_retrieval.md`](v5_general_retrieval.md) — parent retrieval plan
- [`docs/decisions/001_retrieval_k_and_arm.md`](../decisions/001_retrieval_k_and_arm.md) — current `full_rerank` K=12 default
- [`experiments/06_retrieval_dense_vs_full/README.md`](../../experiments/06_retrieval_dense_vs_full/README.md) — dense vs full_rerank A/B
- [`experiments/07_retrieval_extended_k/README.md`](../../experiments/07_retrieval_extended_k/README.md) — extended K analysis
- [`experiments/06_retrieval_dense_vs_full/report/funnel_full_rerank_K12.md`](../../experiments/06_retrieval_dense_vs_full/report/funnel_full_rerank_K12.md) — per-stage funnel that motivated HyDE
- [Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without Relevance Labels"](https://arxiv.org/abs/2212.10496) — HyDE paper
- [Qwen 2.5 Instruct model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
