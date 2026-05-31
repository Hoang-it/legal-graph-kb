# Session handoff — exp 08 HyDE pilot done, strict-metric policy locked

- **Status**: Active. Created 2026-05-31 at end of pilot 50 session.
- **Owner**: Nguyễn Hữu Hoàng
- **Branch**: `exp/08-hyde` (local commits ahead of `origin/exp/08-hyde`)
- **Purpose**: Single canonical handoff so a future session can resume
  without replaying the chat. Covers (1) current state, (2) policy
  decisions locked, (3) pending implementation tasks with priority +
  reasoning, (4) operational notes on HyDE LLM cache.

If you are resuming this work, read:
1. This doc (state + next steps).
2. [`v5_general_retrieval.md`](v5_general_retrieval.md) §5 (strict
   citation metric policy) and §10 (acceptance gates).
3. [`hyde_gpt4o_mini.md`](hyde_gpt4o_mini.md) (HyDE design, supersedes
   `hyde_qwen_colab.md`).
4. [`experiments/08_hyde_retrieval/README.md`](../../experiments/08_hyde_retrieval/README.md)
   (pilot result summary + verdict).

## 1. Current state snapshot (2026-05-31)

### Branch + worktree
- Active branch: `exp/08-hyde`.
- Local working tree has uncommitted changes from this session:
  - `src/retrieval/hyde.py` (rewritten Qwen → OpenAIHydeGenerator).
  - `src/retrieval/pipeline.py` (type hint + field-name fix).
  - `scripts/exp08_{test_one,run,metrics,funnel}.py` (OpenAI-aware).
  - `experiments/08_hyde_retrieval/{config.yaml,README.md,pilot_50_stt.json}`.
  - `docs/plans/hyde_qwen_colab.md` (header marked Superseded).
  - `docs/plans/hyde_gpt4o_mini.md` (new canonical plan).
  - `docs/plans/v5_general_retrieval.md` §5 + §10 (strict metric policy).
  - `notebooks/exp08_hyde_colab.ipynb` (deleted).
- **None of the above is committed yet.** Decide on commit boundaries
  before starting the next session.

### Experiment 08 — pilot 50 done
- Stratified sample (seed=0): in_corpus=38, mixed=1, ooc=2, unparseable=9.
- 4 arms × 50 questions = 200 retrieval calls, **0 failures**.
- Wall time: 228.6s. HyDE LLM cost cumulative: **$0.0122**.
- Records: `experiments/08_hyde_retrieval/results/{dense,dense_hyde,full_rerank,full_rerank_hyde}/A*.json`
  (gitignored by experiment-local `.gitignore` — see `experiments/README.md`).
- Metrics: `experiments/08_hyde_retrieval/metrics/{academic_metrics.json,csv}` (committed).
- Report: `experiments/08_hyde_retrieval/report/{academic_report.md,funnel_full_rerank_hyde_K12.md}` (committed).

### Pilot verdict — strong signal for `dense_hyde`, neutral for `full_rerank_hyde`
**In-corpus stratum (n=38)**:

| Criterion | Threshold | Result | Verdict |
|---|---:|---:|:---:|
| C1: `dense_hyde` R@12 − `dense` R@12 (abs) | +0.030 | +0.1053 | ✅ PASS (3.5× margin) |
| C2: `dense_hyde` NDCG@12 / `dense` NDCG@12 − 1 | +5.0% | +35.2% | ✅ PASS (7× margin) |
| C3: `full_hyde` R-Prec / `full_rerank` R-Prec − 1 | +15.0% | −0.5% | ❌ FAIL |

`dense_hyde` R-Prec almost doubles (0.075 → 0.148). `full_rerank_hyde`
is flat — the cross-encoder reranker already absorbs the dense-side
lift from HyDE.

Full numbers in [`experiments/08_hyde_retrieval/README.md`](../../experiments/08_hyde_retrieval/README.md)
"Result summary" section.

## 2. Decisions locked in this session

### D-EXP08-1 — Switch HyDE generator from Qwen to gpt-4o-mini
- Rationale: Qwen 3B → 7B fallback infeasible on Colab Free T4;
  gpt-4o-mini is reproducible (snapshot id audit-able), trivial cost
  (~$0.025 / 50-q pilot, ~$0.10 / full 200), no GPU dependency.
- Old plan superseded: `docs/plans/hyde_qwen_colab.md` (header marked).
- New canonical plan: `docs/plans/hyde_gpt4o_mini.md`.

### D-EXP08-2 — HyDE LLM responses persisted to disk
- **Single location**: `artifacts/hyde/openai__<model_safe>/<sha256>.json`.
- Cache key = sha256 of `(question + prompt_sha + n + model + max_tokens + temperature)`.
- Schema: question, model_id, model_returned (OpenAI snapshot id),
  prompt_sha, generation knobs, generated_at, generated_docs[],
  usage (prompt/completion/cached tokens), cost_usd.
- Atomic writes (tmp → `os.replace`). 51 entries after pilot 50 run.
- **Gitignored** (root `.gitignore` ignores `artifacts/`).
  - Implication: re-clone on a new machine = lost cache = re-pay LLM
    cost for those questions.
  - If reproducibility for thesis demands shipping the cache, unignore
    `artifacts/hyde/` specifically, OR move cache to `data/hyde_cache/`.
- Generated text **does NOT live** in experiment records under
  `experiments/08_hyde_retrieval/results/<arm>/A<stt>.json` — records
  carry only the HyDE config snapshot (model, n, max_tokens, prompt_sha).
  To inspect what was actually generated for a question, compute its
  cache key and read the JSON in `artifacts/hyde/`.

### D-METRIC-1 — Strict tuple-equal citation matching for E2E
- Decision date: 2026-05-31. See [`v5_general_retrieval.md`](v5_general_retrieval.md) §5.
- A predicted citation matches a gold citation **iff** the full tuple
  `(law_id, article_n, clause_n, point_letter)` matches exactly.
- No component may differ, be missing, or be over-specified.
  Particularly: arm cite `Điều 2 khoản 1` when gold is `Điều 2` = MISS
  (arm may have hallucinated a non-existent khoản).
- **Scope**:
  - **E2E citation metrics** (LLM emits citation → parsed → tuple) use
    strict tuple. This is the **primary metric** for §10 gates.
  - **Retrieval-only experiments (exp 06/07/08)** stay article-deduped
    diagnostic. Pipeline is allowed to fetch sibling clauses of a hit
    article to widen LLM context — not a metric leak because the LLM
    still has to emit strict-correct citation to score.
  - `law_id` MUST match at every layer.
- See [Task #21 implementation plan below](#task-21--implement-strict-tuple-citation-metric).

### D-EXP08-3 — Two-pipeline pattern in HyDE runner
- `scripts/exp08_run.py` constructs two `V5RetrievalPipeline` instances
  (one with `hyde=None`, one with `hyde=OpenAIHydeGenerator(...)`).
  Required because `pipe.retriever.query_encoder` is per-instance — a
  shared pipeline would route ALL arms through HyDE.
- BGE-M3 weights + reranker SHARED across the two pipes to avoid OOM
  on 4 GB GPU (sharing is purely a memory optimisation; per-arm config
  snapshot remains honest).
- Same pattern lifted into `scripts/exp08_test_one.py` after the
  initial dry-run accidentally compared HyDE-vs-HyDE.

## 3. Pending work — ordered by priority

### Task #21 — Implement strict tuple citation metric (BLOCKER for Sprint 3)
- **Why blocker**: any v5 vs v4 A/B after 2026-05-31 must cite v4
  numbers re-aggregated under strict tuple. Mixing two metric
  definitions invalidates A/B claims for the thesis.
- **Not yet coded** (user explicitly deferred during this session).
- **Steps** (verbatim from `v5_general_retrieval.md` §5):
  1. `eval_core/gold.py:validate_gold_citations` — keep `clause_n` +
     `point_letter` on normalized gold (currently rolled into
     `gold_articles: list[str]`). Emit
     `gold_citations_normalized: list[dict]` with full tuple. Keep
     `gold_articles` as derived field for retrieval-only audits.
  2. `eval_core/metrics.py:compute_citation_metrics` — comparison key
     becomes the full 4-tuple. Recall denominator = |gold_tuples|;
     precision denominator = |predicted_tuples|.
  3. Verify `src/citations.py:parse_displayed_citations` emits the
     4-tuple from canonical citation format ("Điều X khoản Y điểm z").
  4. `tests/test_academic_metrics.py` — add fixtures covering the
     6-row example table in §5 (6 verdicts: 1 HIT case-2, plus 4 MISS
     cases, plus 1 trivial HIT). Each fixture asserts a per-citation
     verdict so a future refactor cannot silently regress.
  5. Re-run `python -m eval_core metrics experiments/01_initial_eval`
     to re-aggregate the frozen v4 baseline under the new policy.
     Records (`results/*.json`) are immutable; only the metrics
     aggregation (`metrics/academic_metrics.json` + report) gets
     rewritten.
  6. `docs/changelog.md` — entry: "2026-XX-XX — citation metric
     switched to strict tuple-equal; v4 baselines re-aggregated in
     same commit." Include before-and-after v4 recall/precision so
     the published numbers in any old paper draft can be reconciled.
  7. Leave `scripts/exp{06,07,08}_metrics.py` article-deduped with an
     explicit comment "diagnostic, not primary metric — see v5 plan §5".
- **Acceptance**: tests pass; v4 baseline numbers in
  `experiments/01_initial_eval/metrics/academic_metrics.json` change
  (likely DOWN, since strict > article-only). After re-aggregation,
  decide whether §10 acceptance tiers (70/80, 85/90, 95/95) need
  shift — do NOT shift preemptively, see §10 caveat.
- **Effort**: ~3–4h pure code, $0 (no API calls).

### Other pending items per v5 plan §4 + §10

In priority order (assuming Task #21 lands first):

| # | Item | Why | Cost / effort |
|---|---|---|---|
| A | Scale HyDE pilot 50 → full 200 | Confirm pilot magnitude on full dataset. Pilot N=50 is suggestive; thesis chapter needs full-200 numbers. | ~$0.05, ~15 min. Idempotent — runner skips done records. |
| B | Add E2E arms (`dense_hyde_e2e`, `full_rerank_hyde_e2e`) | exp 08 is retrieval-only. v5 §5 primary metric is **E2E citation_recall/precision**, not retrieval recall. Need to run `V5RetrievalPipeline.ask()` (with LLM generator) on HyDE arms to know whether retrieval lift translates to citation lift. | ~$0.20–0.30 for 200 × 2 arms × gpt-4o-mini. |
| C | OOC detection metric | v5 §10 gate: F1 ≥ 0.80. Not yet implemented. Question with OOC gold → arm must declare "không có trong corpus" instead of citing. | ~1 day code + integration. |
| D | E2E latency metric | v5 §10 gate: median ≤ 30s/question. Only retrieval latency measured so far (2.2s). | Drop-in metric once E2E arms exist. |
| E | M6 Verifier (conditional) | v5 §4 Sprint 2 trigger: precision E2E < 80%. Decide AFTER E2E numbers exist. | Skip if E2E precision already ≥ 80%. |
| F | 150 / 50 stratified split seal | v5 §5: 150 test for paper, 50 dev for calibration. Currently using all 200. `scripts/seal_eval_split.py` exists — verify if it has been run; if not, run + commit splits. | ~1h. |
| G | Sprint 3 final eval | All baselines on 150-test, A/B v4-vs-v5 under strict policy, paper-ready tables, OOC F1, multi-tier framing. | 1–2 weeks per v5 plan. |

### Decision tree for next session

```
Did Task #21 land?
├─ NO → Code Task #21 first. Without it, any number from items A–G
│       is unreportable for the thesis.
└─ YES → Pick branch:
         ├─ Item A (full 200 HyDE retrieval) — fastest, $0.05, 15 min.
         │       Confirms pilot. Cheap insurance before investing in E2E.
         ├─ Item B (E2E HyDE arms) — main answer to "does HyDE actually
         │       lift the metric the thesis cares about?". $0.20–0.30,
         │       ~30 min including code.
         └─ Items C–G are Sprint 3 territory — defer until A + B done.
```

## 4. Operational notes

### Re-running anything in exp 08 is cheap because of cache
- All 50 pilot questions are cached in `artifacts/hyde/openai__gpt-4o-mini/`.
- Re-running `python scripts/exp08_run.py --pilot-50` after a change
  to retrieval code costs **$0** for HyDE.
- Only invalidates if you change `prompts/runtime/hyde_generate.md`
  (prompt_sha changes), or change `model` / `n` / `max_tokens` /
  `temperature` constructor args.

### Cache behaviour to be aware of
- 4-tuple cache key is content-derived; no LRU eviction. Cache grows
  monotonically. At ~2 KB/entry, full 200 = ~400 KB. Negligible.
- A future ablation that varies the prompt will write a SECOND set of
  entries (different prompt_sha → different cache keys). Old prompt's
  cache stays. Tidy up manually if needed.

### Gitignore exception decision pending (D-EXP08-2 follow-up)
- Default: cache is local-only, not shipped.
- If thesis reproducibility requires shipping the cache:
  - Option 1: add `!artifacts/hyde/` to root `.gitignore` (small but
    pollutes repo). Recommended only if total cache < 1 MB.
  - Option 2: move cache target dir to `data/hyde_cache/` (tracked by
    convention). Requires changing `cache_dir` default in
    `OpenAIHydeGenerator.__init__` and re-running to populate the new
    location. Old cache becomes dead but takes no repo space.
  - **Defer until thesis publishing**. Local-only is fine for now.

### Branch hygiene
- `exp/08-hyde` has 1 committed commit (`ae0fc4e`) carrying the old
  Qwen implementation. The current session's rewrite is uncommitted.
- Suggested commit boundaries before next session:
  1. One commit for the rewrite + docs (logical pairing: hyde.py +
     pipeline.py + scripts/exp08_*.py + experiments/08_hyde_retrieval/* +
     docs/plans/hyde_*.md + docs/plans/v5_general_retrieval.md + this
     handoff doc).
  2. Pilot results files (`metrics/`, `report/`,
     `pilot_50_stt.json`) — commit separately so the rewrite commit
     stays code-only.
  3. README "Result summary" update goes in commit (2).
- After commits, push `exp/08-hyde` to origin. PR `exp/08-hyde → main`
  is premature — wait for Task #21 + full 200 + E2E before merging.

## 5. References

- [`docs/plans/v5_general_retrieval.md`](v5_general_retrieval.md) —
  parent plan; §5 strict citation metric, §10 acceptance gates.
- [`docs/plans/hyde_gpt4o_mini.md`](hyde_gpt4o_mini.md) — current HyDE plan.
- [`docs/plans/hyde_qwen_colab.md`](hyde_qwen_colab.md) — superseded
  Qwen plan (kept for design-history).
- [`experiments/08_hyde_retrieval/README.md`](../../experiments/08_hyde_retrieval/README.md) —
  exp 08 result summary (pilot 50).
- [`experiments/08_hyde_retrieval/metrics/academic_metrics.json`](../../experiments/08_hyde_retrieval/metrics/academic_metrics.json) —
  raw numbers.
- [`docs/decisions/001_retrieval_k_and_arm.md`](../decisions/001_retrieval_k_and_arm.md) —
  current production default; HyDE win could trigger ADR 002.
- [`src/retrieval/hyde.py`](../../src/retrieval/hyde.py) —
  `OpenAIHydeGenerator` (sync `generate`, async `generate_batch`,
  cache management, cost tracking).
- [`offline/llm_extract.py:637-644`](../../offline/llm_extract.py) —
  source of truth for gpt-4o-mini cost formula (reused in `hyde.py`).
