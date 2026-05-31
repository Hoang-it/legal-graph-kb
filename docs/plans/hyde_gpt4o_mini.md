# Plan — HyDE retrieval with gpt-4o-mini + stratified 50-question pilot (experiment 08)

- **Status**: Accepted (2026-05-31), implementation in flight
- **Owner**: Nguyễn Hữu Hoàng
- **Target experiment**: [`experiments/08_hyde_retrieval/`](../../experiments/08_hyde_retrieval/)
- **Supersedes**: [`docs/plans/hyde_qwen_colab.md`](hyde_qwen_colab.md) (Qwen 2.5 3B on Colab Free T4)
- **Why pivot**: Qwen 7B fallback infeasible on Colab Free T4 VRAM
  (BGE-M3 + reranker co-residency tight); Qwen 3B output quality on
  Vietnamese legal text felt risky for a thesis-grade comparison.
  gpt-4o-mini is reproducible (snapshot id audit-able), cheap
  (~$0.025 / 50-q pilot, ~$0.10 / 200-q full), and removes Colab/GPU
  dependency.

## TL;DR

Implement HyDE (Hypothetical Document Embeddings, Gao et al. 2022) on
the BGE-M3 dense retrieval channel of `V5RetrievalPipeline`. Generator
runs locally calling OpenAI `gpt-4o-mini`. All LLM responses persisted
to `artifacts/hyde/openai__gpt-4o-mini/<sha>.json` so re-runs cost $0.
Compare 4 retrieval arms — `dense`, `dense_hyde`, `full_rerank`,
`full_rerank_hyde` — on a **stratified 50-question pilot** with the
exp 07 metric suite (recall, precision, F1, NDCG @K + R-Precision,
MRR for K ∈ {12, 20, 30, 50, 70, 100, all}). Decision after pilot:
scale to full 200 only if at least one of three success criteria
shows a strong signal.

## Context

### Why HyDE for this project

`full_rerank` funnel analysis at K=12 in_corpus (n=151) — see
[`experiments/06_retrieval_dense_vs_full/report/funnel_full_rerank_K12.md`](../../experiments/06_retrieval_dense_vs_full/report/funnel_full_rerank_K12.md)
— shows dense is the dominant signal source. Rerank1 lifts R@12 by
+8.9pp but it can only re-rank what dense + sparse surfaced. Better
dense candidates → better rerank pool → better final.

The 200 BHXH questions are written in casual storytelling style
("Bà Minh Châu (Long An) ký hợp đồng lao động theo diện làm việc bán
thời gian…") while the KG clauses are formal legal text
("Người lao động làm việc theo hợp đồng lao động không xác định thời
hạn…"). HyDE is designed exactly for this style gap.

### Why gpt-4o-mini (after Qwen pivot)

- **Cost trivial**: gpt-4o-mini pricing ($0.15/M input, $0.60/M
  output). Per-call ≈ $0.0005. 50 q × $0.0005 = $0.025. 200 q = $0.10.
- **Reproducible**: snapshot id (`gpt-4o-mini-2024-07-18`) returned
  in every response is stored in the cache payload (`model_returned`).
  Re-runs hit disk cache → byte-for-byte identical docs.
- **No GPU / Colab**: works on the existing dev environment. Same
  OpenAI SDK + tenacity retry pattern already used by `offline/llm_extract.py`.
- **Cache LLM**: user requirement. Every successful API call writes
  an atomic JSON payload (question + prompt_sha + n + model + max_tokens
  + temperature + generated_docs + usage + cost_usd + model_returned).
  Re-runs with the same knobs = $0.

### Why a stratified 50-question pilot

The full 200-question dataset is a known fixed cost ($0.10) and
runtime (~30 min including reranker on full pipeline). The pilot
exists so we can:

1. Validate the prompt + cache + cost wiring with a small spend (~$0.025).
2. Detect a strong lift signal cheaply — if `dense_hyde` adds nothing
   on 50 stratified questions, full 200 is unlikely to flip the verdict.
3. Eyeball stratum-level behaviour (in_corpus / mixed / ooc /
   unparseable) — the OOC stratum's 0 recall ceiling won't change.

If pilot fails all three success criteria → write a negative-result
README, do NOT spend the $0.10 + half-hour on full 200.

## Design decisions (locked)

| # | Decision | Value | Rationale |
|---:|---|---|---|
| D1 | Generator class | `OpenAIHydeGenerator` | Same public surface as the deleted `QwenHydeGenerator` — `HybridRetriever` + `V5RetrievalPipeline` wiring unchanged |
| D2 | Model | `gpt-4o-mini` (constructor-overridable) | User requirement |
| D3 | N (HyDE docs/question) | 1 | Pilot focused on cost; can scale later |
| D4 | HyDE plug-in point | Replace dense query embedding only | Sparse channel keeps raw question — isolates HyDE contribution |
| D5 | Dense index | `clause_vec_tuned` | Match production (BGE-M3 LoRA) |
| D6 | Prompt | [`prompts/runtime/hyde_generate.md`](../../prompts/runtime/hyde_generate.md) **unchanged** | Model-agnostic; same prompt → fair comparison if we later try other models |
| D7 | Temperature | 0 | Reproducible |
| D8 | max_tokens | 700 | 200–400 từ Vietnamese ≈ 400–800 tokens; buffer for stop |
| D9 | Cache key | `sha256(question + prompt_sha + n + model + max_tokens + temperature)` | Every knob that influences output goes into the key |
| D10 | Cache path | `artifacts/hyde/openai__<model_safe>/<sha>.json` | Namespaced by model so a future ablation doesn't clobber existing docs |
| D11 | Cost formula | `(prompt − cached) × $0.15/M + cached × $0.075/M + completion × $0.60/M` | **Reused verbatim** from [`offline/llm_extract.py:637-644`](../../offline/llm_extract.py) |
| D12 | Concurrency | `generate_batch`: `AsyncOpenAI` + `Semaphore(5)` + tenacity retry. `generate` (single): sync, used by `HybridRetriever` encoder closure | Matches project default `OPENAI_CONCURRENCY=5` |
| D13 | Retry policy | `stop_after_attempt(5)` + `wait_exponential(multiplier=2, min=4, max=60)` on `RateLimitError | APIError` | Verbatim from `offline/llm_extract.py:289-294` |
| D14 | Pilot N | 50 | User chốt — small enough to be cheap, large enough to detect ≥3pp lift |
| D15 | Stratified sample | Proportional `floor(50 × n_stratum / 200)` + remainder to in_corpus, seed=0. Persisted at [`experiments/08_hyde_retrieval/pilot_50_stt.json`](../../experiments/08_hyde_retrieval/pilot_50_stt.json) | Reproducible + transparent; metrics + funnel auto-filter to this subset |
| D16 | Cost cap | $0.50 pre-flight (well above $0.025 estimate). Runner aborts before any spend if estimate exceeds cap | Safety guard against config drift / runaway max_tokens |
| D17 | `model_revision` audit | Store `resp.model` (e.g. `gpt-4o-mini-2024-07-18`) in cache payload | OpenAI doesn't expose commit SHA; snapshot id is the audit proxy |

## Cache payload schema

```json
{
  "question": "Bà Minh Châu (Long An) ...",
  "model_id": "gpt-4o-mini",
  "model_returned": "gpt-4o-mini-2024-07-18",
  "prompt_sha": "1d90cf...",
  "n": 1,
  "max_tokens": 700,
  "temperature": 0.0,
  "generated_at": "2026-05-31T...",
  "generated_docs": ["Người lao động làm việc theo ..."],
  "usage": {
    "prompt_tokens": 850,
    "completion_tokens": 612,
    "cached_tokens": 0
  },
  "cost_usd": 0.00049525
}
```

Cache writes are atomic (`*.tmp → os.replace`) so a Ctrl+C / network
blip mid-write never leaves a partial JSON.

## Success criteria

After pilot 50 run + metrics, **HyDE is considered a (pilot-level)
win** if any one of:

- `dense_hyde` lifts in_corpus **R@12 by ≥ +3pp absolute** over `dense`.
- `dense_hyde` lifts in_corpus **NDCG@12 by ≥ +5% relative** over `dense`
  (we use K=12 instead of K=10 — K=10 isn't in the K-set; same direction).
- `full_rerank_hyde` lifts in_corpus **R-Precision by ≥ +15% relative**
  over `full_rerank`.

**Strong pilot signal** → scale full 200 → if confirmed → ADR 002.
**Weak / null signal** (3 metrics within ±1pp noise) → document the
negative result in [`experiments/08_hyde_retrieval/README.md`](../../experiments/08_hyde_retrieval/README.md);
do NOT change production.
**Pilot cost > $0.10** → bất thường, debug before scaling.

## Code surface

### New files (already implemented)

1. [`prompts/runtime/hyde_generate.md`](../../prompts/runtime/hyde_generate.md) — model-agnostic Vietnamese system+user prompt.
2. [`src/retrieval/hyde.py`](../../src/retrieval/hyde.py) — `OpenAIHydeGenerator` class.
3. [`scripts/exp08_test_one.py`](../../scripts/exp08_test_one.py) — single-question dry-run + manual GATE.
4. [`scripts/exp08_run.py`](../../scripts/exp08_run.py) — 4-arm runner + `--pilot-50` + cost summary.
5. [`scripts/exp08_metrics.py`](../../scripts/exp08_metrics.py) — K-set + 3-criterion table + auto pilot filter.
6. [`scripts/exp08_funnel.py`](../../scripts/exp08_funnel.py) — per-stage funnel for `full_rerank_hyde` + auto pilot filter.

### Modified files (Phase-2 wiring, unchanged from Qwen iteration)

7. [`src/retrieval/hybrid_retriever.py`](../../src/retrieval/hybrid_retriever.py) — `query_encoder` kwarg.
8. [`src/retrieval/pipeline.py`](../../src/retrieval/pipeline.py) — `hyde: OpenAIHydeGenerator | None` param + `retrieve_dense_only_hyde` method.

### Experiment folder

9. [`experiments/08_hyde_retrieval/config.yaml`](../../experiments/08_hyde_retrieval/config.yaml)
10. [`experiments/08_hyde_retrieval/README.md`](../../experiments/08_hyde_retrieval/README.md)
11. `experiments/08_hyde_retrieval/.gitignore` — `results/` ignored
12. `experiments/08_hyde_retrieval/pilot_50_stt.json` — auto-created on first `--pilot-50` run

## Stratified sampler (algorithm — implemented in `scripts/exp08_run.py`)

```python
def build_or_load_pilot_50(seed=0):
    if PILOT_50_PATH.exists():
        return json.loads(PILOT_50_PATH.read_text())  # idempotent
    by_cat = {"in_corpus": [], "mixed": [], "ooc": [], "unparseable": []}
    for q in questions:
        by_cat[_categorize(q.get("gold_citations_raw"), in_corpus_codes)].append(q["stt"])
    quotas = {c: max(1, 50 * len(by_cat[c]) // 200) for c in by_cat if by_cat[c]}
    while sum(quotas.values()) < 50: quotas["in_corpus"] += 1
    while sum(quotas.values()) > 50: quotas["in_corpus"] -= 1
    rng = random.Random(seed)
    chosen = sorted(s for c, k in quotas.items() for s in rng.sample(sorted(by_cat[c]), k))
    # ... persist payload with seed + quotas + stt_list ...
```

Expected quotas on the 200-q dataset (rough):
- in_corpus 151 → 37–38
- mixed 30 → 7
- ooc 8 → 2
- unparseable 11 → 2–3
- Total = 50

## Verification

1. **Import smoke**: `python -c "from src.retrieval.hyde import OpenAIHydeGenerator; g = OpenAIHydeGenerator(); print(g.prompt_sha[:16])"`
2. **Tests**: `python -m pytest tests/ -q` — must keep 173/175 passing
   (same 2 pre-existing failures unrelated to this change).
3. **Single-question dry run**:
   `python scripts/exp08_test_one.py --stt 2 --gold L58_2014.A2`
   Inspect hypothetical doc + verify cost ≈ $0.0005 + verify second
   `generate()` is cache-hit (no new API call, no cost delta).
4. **Pilot 50 full pipeline**:
   `python scripts/exp08_run.py --pilot-50 --verbose`
   4 arms, prints per-arm latencies + total HyDE cost summary. Must be
   < $0.10 (cap warning).
5. **Metrics**: `python scripts/exp08_metrics.py` — auto-filters to
   pilot 50, prints 3-criterion table.
6. **Funnel**: `python scripts/exp08_funnel.py` — per-stage for
   `full_rerank_hyde` on pilot 50.

## Decision tree after pilot

- **All 3 criteria fail** (no metric within +1pp / +5% rel of threshold)
  → write negative result. Do NOT spend on full 200. Consider follow-ups:
  - LoRA-tuned BGE-M3 (`clause_vec_tuned`) was trained Q→clause; HyDE
    doc-style query may mismatch. Run ablation with vanilla `clause_vec`.
  - Try different prompt (more terse / different VN legal terminology).
- **≥1 criterion passes weakly** (within +1pp / +5% rel of threshold)
  → uncertain; consider running full 200 to reduce variance OR repeat
  pilot with different seed to gauge variance.
- **≥1 criterion passes strongly** → scale to full 200. After full
  confirms → draft `docs/decisions/002_hyde_retrieval.md` (or update
  ADR 001) making HyDE optional default behind a config flag.

## References

- [Gao et al. 2022 — "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE paper)](https://arxiv.org/abs/2212.10496)
- [`docs/plans/v5_general_retrieval.md`](v5_general_retrieval.md) — parent retrieval plan
- [`docs/plans/hyde_qwen_colab.md`](hyde_qwen_colab.md) — superseded plan (Qwen on Colab)
- [`docs/decisions/001_retrieval_k_and_arm.md`](../decisions/001_retrieval_k_and_arm.md) — current `full_rerank` K=12 default
- [`experiments/06_retrieval_dense_vs_full/README.md`](../../experiments/06_retrieval_dense_vs_full/README.md) — dense vs full_rerank A/B
- [`experiments/07_retrieval_extended_k/README.md`](../../experiments/07_retrieval_extended_k/README.md) — extended K analysis
- [`offline/llm_extract.py`](../../offline/llm_extract.py) — source of truth for OpenAI cost formula + tenacity retry pattern
