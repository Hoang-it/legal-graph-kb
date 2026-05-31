# Session handoff — exp 08 HyDE pilot + strict-metric v2 landed

- **Status**: Active. Last updated 2026-05-31 (post task #21).
- **Owner**: Nguyễn Hữu Hoàng
- **Branch**: `exp/08-hyde` (3 commits ahead of `origin/exp/08-hyde`)
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
5. [`docs/changelog.md`](../changelog.md) Unreleased — the
   "academic_v1 → academic_v2" entry has the full v4 baseline shift table.

## 1. Current state snapshot (2026-05-31, end of session)

### Branch + worktree
- Active branch: `exp/08-hyde`.
- **3 commits ahead of `origin/exp/08-hyde`, working tree clean.**

  | SHA | Title |
  |---|---|
  | `8fe0a9d` | feat(retrieval): HyDE với gpt-4o-mini + persistent LLM cache; plan strict tuple metric |
  | `098d32d` | data(exp08): pilot 50 HyDE results — metrics + funnel + stratified seed |
  | `6c03617` | feat(metrics): strict tuple-equal citation matching (academic_v2) + v4 baseline re-aggregated |

- Branch not yet pushed to origin. PR `exp/08-hyde → main` is
  premature — wait until full 200 + E2E confirm before merging.

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

### Task #21 — Strict tuple citation metric ✅ DONE (commit `6c03617`)

Sprint 3 A/B unblocked. v4 baseline re-aggregated under `academic_v2`;
v4 records still immutable (`experiments/01_initial_eval/results/*.json`
untouched), only the metrics + report rewritten.

Tests: **175/175 pass** (excluding the 2 pre-existing failures
unrelated to this change: `test_main_arm_preset_is_shared_between_runner_and_metrics`
and `test_columns_dung_format`).

v4 baseline macro citation recall — academic_v1 → academic_v2:

| arm | v1 | v2 | Δ |
|---|---:|---:|---:|
| graphrag | 0.1120 | **0.0000** | −0.1120 |
| llm_only | 0.0067 | 0.0000 | −0.0067 |
| logic_lm_no_retrieval | 0.0023 | 0.0000 | −0.0023 |
| logic_lm_ontology | 0.0073 | 0.0000 | −0.0073 |
| logic_lm_graphrag | 0.0175 | 0.0000 | −0.0175 |
| logic_lm_graphrag__gpt-4_1 | 0.1565 | **0.0338** | −0.1227 |
| logic_lm_graphrag__gpt-4o | 0.1407 | 0.0260 | −0.1147 |
| logic_lm_graphrag__gpt-5-mini | 0.0785 | 0.0315 | −0.0470 |
| logic_lm_no_retrieval__gpt-4_1 | 0.0433 | 0.0008 | −0.0425 |
| logic_lm_no_retrieval__gpt-4o | 0.0400 | 0.0033 | −0.0367 |
| logic_lm_no_retrieval__gpt-5-mini | 0.0081 | 0.0000 | −0.0081 |

Reading: 5/11 arms collapse to 0.0 because they cite with `khoản`
granularity while every gold cite in the 200-q dataset is article-only.
Logic-LM with explicit `__gpt-4_*` answerers retain some recall
because Prolog forces citation cardinality to match facts loaded
into the program. See `docs/changelog.md` Unreleased "Changed" entry
for full rationale.

§10 tier re-calibration: NOT done preemptively per the §10 caveat.
Decide after seeing actual v2 v5 numbers in Sprint 3.

### Now-immediate items (priority order, post task #21)

| # | Item | Why | Cost / effort | Owner notes |
|---|---|---|---|---|
| **A** | **Scale HyDE pilot 50 → full 200** (retrieval-only) | Confirm pilot magnitude on full dataset. Pilot N=50 is suggestive; thesis chapter needs full-200 numbers + per-stratum stability. Cache covers 50/200 questions → only 150 new LLM calls. | ~$0.075, ~15 min wall time. `python scripts/exp08_run.py` (no `--pilot-50`) runs the full set; idempotent — skips done records, prewarm only the new 150. | Cheap insurance before investing engineering time in #B. |
| **B** | **Add E2E HyDE arms** (`dense_hyde_e2e`, `full_rerank_hyde_e2e`) | exp 08 today is retrieval-only — measures `final_article_ids` lift. v5 §5 primary metric is **E2E citation recall/precision under academic_v2** (the thesis-defining number). Need to run `V5RetrievalPipeline.ask()` (with LLM generator) on HyDE-augmented retrieval to see whether retrieval lift translates to citation lift. Adapter mismatch caveat: the LLM may still emit khoản with article-only gold → strict-tuple MISS even if retrieval is perfect. | ~$0.20–0.30 for 200 × 2 arms × gpt-4o-mini. ~30 min code + 10 min run. | Needs a new runner script (or extend exp08_run.py with an `--e2e` flag) that records `answer` + `citation_ids` + `latency_s` so academic_v2 metrics apply directly. |
| **C** | **150 / 50 stratified split seal** | v5 §5: 150 test for paper, 50 dev for calibration. Currently scripts read `data/eval/questions_200.json` raw. `scripts/seal_eval_split.py` exists but unknown whether it was already run + committed. | ~1h verify + run + commit | Audit first: `ls data/eval/questions_*.json`. If `questions_150_test.json` + `questions_50_dev.json` already exist + are stratified seed-locked, mark done. Otherwise run the sealer + commit, then point all exp 08 runner / metrics paths at the test split. |
| **D** | **OOC detection F1 metric** | v5 §10 gate: F1 ≥ 0.80. Question with OOC gold → arm must declare "không có trong corpus" instead of citing. Currently no metric computes this. The 8 OOC questions in the 200-q dataset are a known weak stratum (0 recall under all arms; rerank funnels confirmed). | ~1 day code + integration | New helper in `eval_core/metrics.py`: classify per-record arm output as `{cites_any, declares_ooc, silent}`. Compare to gold OOC flag. Compute F1 over the OOC class. Add to `aggregate.macro`. |
| **E** | **E2E latency metric** | v5 §10 gate: median ≤ 30s/question E2E. Retrieval-only latency measured (full_rerank ~2.2s); LLM generator latency not. | Drop-in once #B exists | The `elapsed_s` field is already collected by the inference runners; just surface its median into the aggregate. |
| **F** | **M6 Verifier (conditional)** | v5 §4 Sprint 2 trigger: precision E2E < 80%. Decide AFTER #B numbers exist. | Skip if E2E precision already ≥ 80% under v2; otherwise ~3 days build | A wrong-but-confident citation is the most dangerous failure mode for a legal QA system. If E2E precision is poor, consider Claude or local NLI verifier to drop low-confidence cites. |
| **G** | **Sprint 3 final eval + thesis chapter** | All baselines on 150-test, A/B v4-vs-v5 (both under `academic_v2`), paper-ready tables, OOC F1, multi-tier framing (70/80, 85/90, 95/95). | 1–2 weeks per v5 plan §7 | The end-state for the v5 plan. Depends on A + B + C + D done. |

### Decision tree (post task #21)

```
Did the full 200 HyDE retrieval-only run finish (item A)?
├─ NO → run item A first ($0.075, 15 min). Confirms pilot signal at
│       higher N before investing engineering in #B.
└─ YES → Did E2E HyDE arm runner exist (item B)?
         ├─ NO → write the runner + run on full 200 ($0.20-0.30,
         │       ~30 min code + 10 min run). This produces the
         │       thesis primary number.
         └─ YES → branch on item B result:
                  ├─ E2E precision ≥ 80% under v2 → skip M6 verifier
                  │       (item F); proceed to items C + D + G.
                  └─ E2E precision < 80% → triage:
                        ├─ Low recall → add HyDE to dense further (no
                        │   action; result speaks).
                        ├─ Low precision (over-citing) → build M6
                        │   verifier (item F), re-run E2E, then G.
                        └─ Both low → document as limitation; G.
```

### Task #21 follow-up — items NOT in scope but flagged for future

These came up while implementing task #21; flagged so future sessions
don't re-discover the same thing.

1. **`prolog_law_id` registry alias completeness**. The v4 logic_lm
   arms scored higher under v1 partly because Prolog-emitted
   citations sometimes match wrong-khoản gold by accident. Under
   v2 the bound is tight. If a Sprint 3 ablation wants to maximise
   Logic-LM v4 → re-run with the post-2026-05 registry which has
   added laws (`ND143_2018`, `QD838_BHXH`) — those law_ids did not
   exist when v4 records were produced, so any v4 citation referencing
   them parses as `BAD_ID` under the strict tuple. Acceptable for
   audit, mention as caveat in any v4-vs-v5 narrative.
2. **`scripts/exp{06,07,08}_metrics.py` use a local article-only
   diagnostic** that bypasses `eval_core.metrics`. Per the v5 §5
   scope distinction this is correct (retrieval-only ≠ primary
   metric). Do NOT migrate them to `academic_v2` — they would
   become misleading. The header comment in each file already says
   "diagnostic, not primary".
3. **`expected_summary.json` fixture had no version field before**.
   Now carries an `_note` field for reader context. Future metric
   policy changes should follow the same pattern: bump
   `METRIC_VERSION`, regenerate fixture, add a dated `_note`.

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
- `exp/08-hyde` is 3 commits ahead of `origin/exp/08-hyde`. Sequence:
  1. `8fe0a9d` — Qwen → gpt-4o-mini rewrite + docs (incl. this handoff
     in its first version)
  2. `098d32d` — pilot 50 results (metrics + funnel + stratified seed)
  3. `6c03617` — task #21 strict metric + v4 baseline re-aggregation
- **Not yet pushed** to origin. PR `exp/08-hyde → main` is premature
  — wait until item A (full 200) + item B (E2E) confirm before merging.
- If pushing now: `git push origin exp/08-hyde` is safe (these commits
  modify code + docs + frozen-experiment metrics, no records / no
  secrets).

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
