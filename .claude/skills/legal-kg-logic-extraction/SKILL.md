---
name: legal-kg-logic-extraction
description: Continuing work on the Legal KG project — extending GraphRAG with logic-based extraction (multi-hop traversal + symbolic facts + structured queries) per the plan in reports/plan_logic_extraction.md. Use when user asks to implement Phase 1-5 of logic extraction, design Neo4j schema for legal logic, build LLM-based legal info extractor, evaluate new arm vs baselines, or any continuation of the legal KB experiment work.
---

# Legal KG Logic Extraction — Continuation Skill

## When to invoke
- User asks to implement any phase of `reports/plan_logic_extraction.md`
- Questions about extending GraphRAG beyond semantic search
- Designing Cypher queries for legal KG with new node/edge types
- Evaluating elite arm Prolog reliability improvements
- Adding new experiments to the eval framework
- Debugging audit-flagged issues (API errors, selection bias, pairwise interpretation)
- Any work touching `src/rag_query.py`, `experiments/elite_pipelines.py`, or `experiments/compute_metrics.py`

## Project context (read this first)

This is a Vietnamese Legal Knowledge Graph QA system over **Luật Bảo hiểm xã hội 41/2024/QH15** (Vietnam Social Insurance Law 2024). The repo has gone through **4 audit rounds** that found + fixed serious methodological issues. The next phase is **logic extraction** (Layer A + B + C per the plan doc).

### Project state (2026-05-27)

| Component | Status |
|---|---|
| KG (Neo4j Aura) | 543 Clauses, 141 Articles, vector index `clause_vec` (1024d BGE-M3) |
| R1 experiment (5 arms × 200q × gpt-4o-mini) | Done — 1000 records committed to repo |
| R2 experiment (3 models × 2 elite arms × 200q) | Done — 1200 records committed |
| Audit fixes | All 4 rounds applied; metrics.json post-audit committed |
| Colab notebooks | Done — `colab_r1_5arm.ipynb`, `colab_r2_multimodel.ipynb`, all-in-one variant |
| Plain answer rendering | Pipeline patched; backfill script ready (not yet run on existing data) |
| **Logic extraction** | **Plan only** — `reports/plan_logic_extraction.md`. Next phase = implement |

### Defensible claims (from significance.md, α_bonf=0.01)
1. **C1**: llm_only beats graphrag pairwise (R1, p<0.0001)
2. **C3b**: elite_no_retrieval > elite_graphrag prolog_success với gpt-4o-mini (p=0.0035)
3. **C4**: elite_graphrag beats elite_no_retrieval pairwise cho gpt-5-mini (p=0.0006, **reversed từ báo cáo cũ**)

### Active problems logic extraction sẽ giải quyết
- elite_graphrag prolog_success 65.5% (vs 78% no_retrieval với gpt-4o-mini) — retrieval HURTS Prolog generation cho weak models
- gpt-5-mini abstention: 108/198 records LLM declares `legal_source(...,article: none,...)` vì không có article numbers cụ thể
- graphrag citation_recall caps at 0.90 — vector sim không recall đủ multi-article questions
- Cross-reference miss: không follow REFERS_TO edges

---

## Codebase architecture

```
legal-graph-kb/
├── src/                    # Main pipeline (DO NOT modify elite/ subdir)
│   ├── rag_query.py        # RagPipeline — semantic vector_search
│   ├── load_neo4j.py       # KG loader from merged_graph.json + embeddings.parquet
│   ├── embed.py            # BGE-M3 embedding
│   └── chat.py             # Interactive REPL
├── elite/                  # Logic-LLM Prolog subproject — DO NOT modify internals
│   └── src/                # IRAC_RENDER_PROMPT, LOGIC_LLM_RULE_GEN_PROMPT, OpenAILLMClient
├── experiments/            # Eval framework — modify here freely
│   ├── run_inference.py             # R1 single-arm runner
│   ├── run_multimodel_inference.py  # R2 arm × model runner
│   ├── elite_pipelines.py           # Wrapper for elite arm (3 variants)
│   ├── graphrag_retriever_adapter.py # Wrap RagPipeline → elite retriever
│   ├── llm_only.py                  # Pure LLM baseline
│   ├── compute_metrics.py           # 17 metrics + cached judge
│   ├── compute_multimodel_metrics.py # R2-specific
│   ├── generate_report.py           # R1 report
│   ├── generate_multimodel_report.py # R2 report
│   ├── compute_significance.py      # McNemar + Bootstrap + Bonferroni
│   ├── audit_apply_fixes.py         # Re-aggregate pairwise from cache (fix _vote bug)
│   ├── audit_apply_fixes_v2.py      # Split halluc + tag api_error
│   ├── audit_reaggregate.py         # Post-process: n_valid + micro-average
│   ├── audit_repair_pairwise.py     # Diagnostic re-parse cached judges
│   ├── rerender_plain_answer.py     # Backfill plain_answer cho elite records
│   ├── text_normalize.py            # IRAC → prose helper (unused, alternative to plain_answer)
│   └── prompts/
│       ├── elite_no_retrieval.md    # Arm C prompt
│       └── irac_with_plain.md       # IRAC + plain_answer combined render
├── data/
│   ├── eval/
│   │   ├── questions_200.json       # Dataset (committed via .gitignore exception)
│   │   ├── results/{arm}/A*.json    # R1 inference records (committed)
│   │   ├── metrics.json             # R1 post-audit metrics (committed)
│   │   ├── judge_cache.jsonl        # Cached LLM-as-Judge results (committed)
│   │   ├── elite_corpus_2024.jsonl  # Elite corpus from Luật 2024
│   │   ├── elite_ontology_2024.json # Elite ontology (concepts + thresholds)
│   │   └── multimodel/
│   │       ├── metrics.json         # R2 (committed)
│   │       ├── judge_cache.jsonl    # R2 cache (committed)
│   │       └── results/{combo}/A*.json
│   ├── interim/structured_law.json  # KG source (committed via exception)
│   └── processed/
│       ├── merged_graph.json        # KG ready-to-load (committed)
│       └── embeddings.parquet       # BGE-M3 vectors (committed)
├── notebooks/                       # Colab Pro notebooks
│   ├── colab_r1_5arm.ipynb          # 1 model × 5 arms parallel
│   ├── colab_r2_multimodel.ipynb    # 2D parallel (arms × models)
│   ├── colab_full_pipeline.ipynb    # all-in-one
│   └── README.md                    # 3 options + speedup table
└── reports/
    ├── FINAL_REPORT.md              # Consolidated honest report (post-audit)
    ├── methodology_fixes.md         # 7 audit issues + fixes
    ├── plan_logic_extraction.md     # ← NEXT PHASE PLAN (read this thoroughly)
    └── experiment_report.md         # Auto-generated R1 report
```

---

## Critical knowledge — gotchas + audit findings

### 1. Pairwise `_vote` bug (FIXED, but understand the history)
**Old bug**: `_vote(w, a_first=False)` inverted vote_ba because `_ask()` keeps label-content pairing stable (only display order swaps). Resulted in ALL pairwise consensus = "split".
**Fix**: `experiments/compute_metrics.py:518-535` simplified to `w='a' → record_a`, no `a_first` param.
**Affected**: 1381/1400 records re-aggregated via `audit_apply_fixes.py`. Reports regenerated.
**Important**: when designing new pairwise tests, label A always = ans_a regardless of position. Never use `a_first` in vote logic.

### 2. API error contamination (90 records affected)
**Issue**: 77 R2 GR×gpt-5-mini + 13 R1 elite_ontology records have prompt_tokens=0 + completion_tokens=0 due to OpenAI "Connection error.". Pipeline's `_attempt()` catches `Exception` blindly and marks as `unable_to_conclude`.
**Mitigation**: `audit_apply_fixes_v2.py` tags `api_error=True` on these records. All aggregates filter them out.
**Future pipeline work**: catch `openai.APIConnectionError`/`RateLimitError`/`APITimeoutError` separately with backoff retry. See `elite/src/pipelines/program_pipeline.py:_attempt`.

### 3. Selection bias in cells (n_valid << n_total)
**Issue**: `elite_no_retrieval citation_validity = 0.9936` was n=52/200 only. Comparing macro means across arms với different n_valid is apples-to-oranges.
**Fix**: every cell now shows `(n=valid/total)`. Cells < 30 marked "insufficient".
**For new metrics**: always report n_valid alongside mean.

### 4. BERTScore + answer_relevance IRAC bias
**Issue**: Elite arms output IRAC structured text (Issue:/Rule:/...) — format dramatically different from gold prose. BERTScore + AR self-similarity unfair to compare.
**Decision**: drop for elite arms unless `plain_answer` available (new IRAC+plain prompt).
**Cell behavior**: if `_used_plain_answer=True` in metric record → compute, else show "dropped (IRAC bias)".

### 5. Hallucination metric split (3 things conflated)
**Old formula**: `(misstate + unsupported + invented) / max(1, n_claims + n_invented)` + edge case `1.0 if n_invented > 0`.
**Fix**: split into 2 metrics in `compute_metrics.py:m_hallucination`:
- `content_hallucination_rate` = `(misstate + unsupported) / max(1, n_claims)` — judge-based
- `invented_citation_rate` = `n_invented / max(1, n_total_citations)` — deterministic
- Legacy `hallucination_rate` kept for backward compat.

### 6. gpt-5-mini abstention behavior
**Finding**: 108/198 success records have `based_on(source_X)` in Prolog trace nhưng `citation_ids=[]` because gpt-5-mini honestly declares `legal_source(...,article: none,...)`. Fallback parser regex `article_(\d+)` không match `none`.
**This is NOT pipeline bug** — it's reasoning model behavior. Document in caveats. Logic extraction (next phase) sẽ fix this by giving model explicit article numbers from KG.

### 7. compute_metrics judge model = generator
**Issue**: `JUDGE_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")` — same env var as generator. To use different judge, set OPENAI_MODEL khác trước khi run compute_metrics.
**Colab notebooks handle this**: pass `env={'OPENAI_MODEL': JUDGE_MODEL}` to subprocess.

---

## Required skills (knowledge checklist)

### Critical (must have)

| Skill | Why | Where to learn |
|---|---|---|
| **Neo4j Cypher** | Schema additions, multi-hop queries, vector index | Existing `src/load_neo4j.py` examples; Neo4j docs |
| **Prolog basics** (SWI-Prolog 9.x) | Elite Logic-LM pipeline + extracted facts will be Prolog-style | `elite/src/pipelines/program_pipeline.py` |
| **Vietnamese legal terminology** | "Mức bình quân tiền lương", "BHXH", "NLĐ/NSDLĐ", IRAC | Existing prompts in `experiments/prompts/`; `data/interim/structured_law.json` |
| **OpenAI API tool use** | Generator + judge calls, env var management, rate limits | Existing `compute_metrics.py:_judge_call` patterns |
| **BGE-M3 embedding** | Vector retrieval, 1024-d Vietnamese-aware | `src/embed.py` |
| **LLM-based information extraction** | Prompt engineering cho structured output (JSON schema) | `experiments/prompts/irac_with_plain.md` is a good example |
| **Statistical tests** | McNemar, Bootstrap CI, Bonferroni | `experiments/compute_significance.py` |
| **Audit mindset** | Distinguish real findings từ artifacts (API errors, selection bias) | `reports/methodology_fixes.md` |

### Important (should have)

| Skill | Why | Where to learn |
|---|---|---|
| **Python concurrency** (ThreadPoolExecutor, subprocess) | Parallel arm execution | `notebooks/colab_r1_5arm.ipynb:Phase 2` |
| **JSON schema design** | Structured output from LLM extraction | Plan §4 in `plan_logic_extraction.md` |
| **Pandas + Matplotlib + Seaborn** | Visualization cells trong notebooks | `notebooks/colab_*.ipynb:Phase 4` |
| **Regex** (especially Vietnamese) | Number/date/percentage extraction | `experiments/compute_metrics.py:_CIT_PAT` |
| **Git workflow** | Branch + commit conventions | `commit history` |

### Nice-to-have

| Skill | Why |
|---|---|
| **Legal NLP background** | Adapt techniques from Western legal NLP to Vietnamese |
| **Graph algorithms** | Optimal multi-hop traversal cho `traverse()` API |
| **Information retrieval theory** | Hybrid search ranking (combine vector + structured) |
| **Active learning** | Cheaper manual annotation gates |

---

## Files to read FIRST when continuing

**In this order**:

1. **`reports/plan_logic_extraction.md`** — THE plan you're implementing. 13 sections covering everything.

2. **`reports/methodology_fixes.md`** — 7 audit issues that shaped current pipeline. Understanding these prevents repeating mistakes.

3. **`reports/FINAL_REPORT.md`** — Current honest baselines + defensible claims. Your improvements compete against these numbers.

4. **`src/rag_query.py`** — Current `RagPipeline` you'll extend with `logic_search()`, `traverse()`, `hybrid_search()`.

5. **`experiments/elite_pipelines.py`** — Current `EliteGraphRAGPipeline`. You'll add `EliteGraphRAGLogicPipeline` parallel to it.

6. **`experiments/compute_metrics.py`** — All metrics + judge cache. Understand cache key conventions before adding new metrics.

7. **`experiments/prompts/irac_with_plain.md`** — Example of well-structured prompt that emits JSON. Pattern to follow for extraction prompts.

8. **`data/eval/elite_ontology_2024.json`** — Existing concept ontology from elite arm. Has `CONCEPT_SPECS` you should reuse for canonical predicates.

9. **`elite/src/pipelines/program_pipeline.py`** — Elite's Prolog generation flow. Understand `_attempt`, repair loop, predicate validation BEFORE building logic-aware variant.

---

## Common gotchas

1. **Pairwise cache keys**: must include `record_a_arm + record_b_arm + swap_id` to prevent collision (see `audit_apply_fixes.py` for the fix). If you add new pairwise comparisons, use the same convention.

2. **`run_inference.py` is idempotent** — re-running skips existing records. If you want to FORCE re-run (e.g., after prompt change), pass `--force` flag.

3. **Colab subprocess env**: `subprocess.run(env=...)` doesn't inherit shell env. Always pass `os.environ.copy()` then add overrides.

4. **OpenAI rate limits**: Tier 4+ for 6+ concurrent jobs. gpt-5-mini reasoning model is slow (~60s/q) — don't expect parallelism to speed it up much.

5. **Neo4j Aura free tier auto-pauses** after 3 days inactivity. Resume via web UI or any query.

6. **`reports/*.md` is gitignored** — to commit a report, add explicit `!reports/your_file.md` exception in `.gitignore`.

7. **plain_answer field** in records: not present in pre-2026-05-27 inference runs. Generate via `python -m experiments.rerender_plain_answer --combos all` (~$0.72).

8. **Elite `legal_source(...,article: none,...)` is intentional** for gpt-5-mini reasoning model — NOT a bug, just behavioral difference. Fallback parser regex skips these → empty citation_ids. Mitigation = logic extraction (give explicit article numbers from KG).

9. **Statistical significance**: McNemar exact binomial only valid for PAIRED outcomes. If you compare independent samples, use chi-square. Always Bonferroni-correct if running > 1 test.

10. **Don't trust macro mean blindly** — always inspect n_valid. `reports/report_v2.md` has micro-average column for citation metrics where this matters.

---

## Recommended workflow cho next phase

### Implementation order (per `plan_logic_extraction.md` Phase 1)

1. **Read all "first" files** above (~2-3h)
2. **Design schema** (Phase 1 of plan):
   - Lock 6 new node labels + 6 edge types
   - Map to existing `CONCEPT_SPECS` from elite arm
   - Document in `reports/logic_extraction_schema.md`
3. **Annotate 30 gold clauses** (manual, 2-3h):
   - Sample 6 clauses from each of 5 BHXH chapters
   - Use schema designed above
   - Save as `data/gold/logic_30_clauses.json`
4. **Build extractor** (Phase 2):
   - `experiments/extract_logic.py` — LLM-based với gpt-4o-mini
   - Regex helpers cho numbers/dates
   - Validation script (compare extracted vs gold)
   - **Accuracy gate ≥80%** before scaling
5. **Iterate prompt** until pass gate
6. **Full extract** 543 clauses → push Neo4j
7. **Query API** (Phase 3):
   - Add methods to `src/rag_query.py`
   - Unit tests
8. **New arm** (Phase 4):
   - `EliteGraphRAGLogicPipeline` trong `elite_pipelines.py`
   - New prompt: "assemble Prolog from THESE facts"
9. **Evaluate** (Phase 5):
   - Run 200q × 5 arms (3 baselines + 2 new)
   - Compute metrics, significance tests
   - Generate report

### Branch + commit hygiene
- Create branch `feature/logic-extraction` để isolate
- Commit per phase (1 commit per major deliverable)
- Use commit message convention from existing history (`feat:`, `data:`, `fix:`)
- Include `Co-Authored-By: Claude` if AI-assisted

---

## Key environment + setup

```bash
# Python
python3.10+ với venv

# Required env vars (in .env or Colab secrets):
OPENAI_API_KEY=sk-...
NEO4J_URI=neo4j+s://xxx.databases.neo4j.io  # Aura cloud, NOT localhost cho Colab
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j
EMBED_DEVICE=cuda  # or cpu nếu không có GPU
OPENAI_MODEL=gpt-4o-mini  # generator (override per-job khi cần)

# Required services:
- Neo4j Aura instance (or local 5.x với vector index support)
- OpenAI API access (Tier 4+ recommended cho parallelism)
- SWI-Prolog 9.x (`apt-get install swi-prolog`)

# Required Python packages:
pip install neo4j sentence-transformers openai python-dotenv bert-score \
            pandas matplotlib seaborn tqdm pyarrow numpy
```

---

## Success criteria cho logic extraction phase

**Hard requirements** (must hit to ship):
- Extraction accuracy ≥ 80% trên gold annotations (Phase 2 gate)
- `prolog_success_rate(logic) ≥ prolog_success_rate(semantic) + 5pp`
- McNemar test p < 0.01 (Bonferroni for 1-2 claims)
- No backward compat break (existing arms still work)

**Soft targets**:
- `faithfulness` improvement ≥ 3pp
- `abstention_rate` (gpt-5-mini) reduce ≥ 30%
- `citation_recall` improvement ≥ 2pp via multi-hop

---

## When you finish

Update `reports/methodology_fixes.md` with any new audit findings discovered.
Update `reports/FINAL_REPORT.md` if new defensible claims emerge.
Add new defensible claims to `experiments/compute_significance.py:CLAIMS`.
Document new pipeline classes / scripts trong this skill file (update for next session).
