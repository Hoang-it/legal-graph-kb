---
name: legal-kg-logic-extraction
description: Continuing work on the Legal KG project — Vietnamese Social-Insurance-Law GraphRAG + Logic-LM stack with offline/runtime/prompts split. Use when the user asks to extend the inference pipeline, add a new logic-LM arm, run the academic-metrics evaluation, design Neo4j schema, work on the v5 retrieval plan, or touch any file under offline/, runtime/, evaluation/, or experiments/.
---

# Legal KG — Continuation Skill

## When to invoke
- User asks to extend or modify the inference runtime in `runtime/`
- Designing or migrating schema in `schema/schema.cypher`
- Adding / rewriting prompts under `prompts/`
- Running or evolving the academic-metrics evaluation in `evaluation/` + `experiments/`
- Following up on `reports/plan_v5_general_retrieval.md` (current planning doc)
- Touching `runtime/rag_query.py`, `runtime/logic_lm_pipelines.py`, `runtime/logic_lm/`, or the offline B1–B6 pipeline in `offline/`

## Project context (read first)

This is a Vietnamese Legal Knowledge Graph QA system over **Luật Bảo hiểm xã hội 41/2024/QH15** (Vietnam Social Insurance Law 2024). Pipeline is two-stage:

- **Offline (`offline/`)** — B1–B6: parse `.docx` → rule-extract → LLM-extract → merge → embed (BGE-M3 1024-d) → load Neo4j.
- **Runtime (`runtime/`)** — B7: GraphRAG, LLM-only baseline, three Logic-LM arms (no-retrieval / ontology / graphrag), plus the IRAC + plain-answer renderer.

The project has been refactored: the old `elite/` package was renamed to `logic_lm` (commit `428302d`), then in the current session the codebase was split into top-level `offline/`, `runtime/`, and a centralized `prompts/` tree.

### Project state

| Component | Status |
|---|---|
| KG (Neo4j) | 543 Clauses, 141 Articles, vector index `clause_vec` (1024-d BGE-M3) |
| Inference arms | `graphrag`, `llm_only`, `logic_lm_no_retrieval`, `logic_lm_ontology`, `logic_lm_graphrag` |
| R1 / R2 results in `data/eval/results/` and `data/eval/multimodel/results/` | Committed |
| Headline evaluation | Academic metrics only (citation recall/precision/F1, display rate, latency, BERTScore, 3 Prolog rates). Judge metrics fail-closed by design. |
| Active plan | `reports/plan_v5_general_retrieval.md` — general/scalable citation retrieval (Sprint 1 = vanilla pipeline + audit) |

---

## Codebase architecture (post-refactor)

```
legal-graph-kb/
├── offline/                          # B1–B6 build-time pipeline
│   ├── parse_docx.py                 # B1 — deterministic .docx parser
│   ├── rule_extract.py               # B2 — regex extraction
│   ├── llm_extract.py                # B3 — OpenAI semantic extraction
│   ├── merge_normalize.py            # B4 — dedup + validate
│   ├── embed.py                      # B5 — BGE-M3 embeddings
│   ├── load_neo4j.py                 # B6 — load Neo4j (constraints + vector index)
│   └── build_logic_lm_corpus_2024.py # Build Logic-LM corpus + ontology
│
├── runtime/                          # Inference runtime
│   ├── rag_query.py                  # B7 — RagPipeline (vector_search, expand, fetch_facts, ask)
│   ├── chat.py                       # Interactive REPL (rich)
│   ├── llm_only.py                   # Pure LLM baseline pipeline
│   ├── logic_lm/                     # Symbolic-LM hybrid package
│   │   ├── config/                   # base/cli/index/llm/pipeline/prolog/prompt/retrieval/schema settings
│   │   ├── indexes/                  # HNSW dense + keyword sparse
│   │   ├── knowledge/                # BHXH ontology, ontology retrieval, hybrid retrieval
│   │   ├── llm/                      # LLM client + factory
│   │   ├── pipelines/                # program_pipeline (Prolog generation + repair loop)
│   │   ├── services/                 # encoder
│   │   ├── solvers/                  # SWI-Prolog wrapper
│   │   └── cli/                      # answer_with_program, build_bhxh_ontology, query_ontology
│   ├── logic_lm_pipelines.py         # 3 arm-aware wrappers (NoRetrieval / Ontology / GraphRAG)
│   ├── graphrag_retriever_adapter.py # Adapt RagPipeline as logic-LM retriever
│   ├── run_inference.py              # Batch inference orchestrator (single arm or 'main')
│   ├── run_multimodel_inference.py   # 2D arm × model orchestrator
│   └── rerender_plain_answer.py      # Backfill plain_answer field on legacy records
│
├── src/                              # Shared utilities (used by offline AND runtime)
│   ├── ids.py                        # ID convention + reverse parser
│   ├── schema.py                     # Pydantic models (provenance invariants)
│   ├── legal_metadata.py             # Multi-law metadata registry
│   ├── citations.py                  # Citation parsing + registry
│   └── prompts.py                    # Prompt loader (with LEGAL_KG_PROMPTS_DIR override)
│
├── prompts/                          # SINGLE SOURCE OF TRUTH for all system prompts
│   ├── offline/
│   │   └── llm_extract.md            # B3 LLM extraction prompt (SYSTEM/USER sections)
│   └── runtime/
│       ├── graphrag_system.md        # GraphRAG generator system prompt
│       ├── llm_only_system.md        # LLM-only baseline system prompt
│       └── logic_lm/
│           ├── rule_gen.md           # Default Prolog generator
│           ├── rule_gen_no_retrieval.md  # No-retrieval ablation variant
│           ├── irac_render.md        # Default IRAC renderer
│           └── irac_with_plain.md    # IRAC + plain_answer combined renderer
│
├── experiments/                      # Eval orchestration + arm definitions
│   ├── arms.py                       # ALL_ARMS, MAIN_EXPERIMENT_ARMS, parse_run_arms/parse_metrics_arms
│   ├── compute_academic_metrics.py   # Experiment-owned loader; delegates to evaluation/
│   ├── text_normalize.py             # IRAC → prose helper for BERTScore fairness
│   └── README.md                     # Eval pipeline diagram + caveats
│
├── evaluation/                       # Headline metric engine (deterministic, dataset-based)
│   ├── compute_academic_metrics.py   # citation recall/precision/F1, display rate, latency, BERTScore, prolog rates
│   ├── validate_gold_citations.py    # Strict parse of gold_citations_raw against registry
│   ├── compute_judge_metrics.py      # Fail-closed placeholder (judge metrics intentionally NOT in main)
│   └── samples/                      # Worked-example fixtures
│
├── data/
│   ├── legal_metadata.yaml           # Multi-law metadata source of truth
│   ├── legal_sources.yaml            # Citation authority registry
│   ├── raw/                          # Source .docx files
│   ├── interim/                      # B1–B3 intermediate JSON (gitignored)
│   ├── processed/                    # merged_graph.json + embeddings.parquet (committed)
│   ├── logic_lm/                     # Logic-LM intermediate data
│   └── eval/
│       ├── questions_200.json        # 200 BHXH questions (committed)
│       ├── logic_lm_corpus_2024.jsonl    # Logic-LM corpus
│       ├── logic_lm_ontology_2024.json   # Logic-LM ontology (concepts + thresholds)
│       ├── academic_metrics.{json,csv}   # Headline metrics (committed)
│       ├── academic/                 # gold_citations_normalized.json
│       ├── results/{arm}/A*.json     # R1 inference records (committed)
│       └── multimodel/results/{arm__model}/A*.json   # R2 records (committed)
│
├── schema/schema.cypher              # Neo4j constraints + vector indexes
├── scripts/                          # PowerShell wrappers (chat, install_b5, install_bge_m3) + verify_b5.py
└── reports/
    └── plan_v5_general_retrieval.md  # ← CURRENT PLAN (vanilla → audit → conditional modules)
```

---

## Key design rules (don't violate)

### Provenance invariants
- Every semantic node MUST have non-empty `mentioned_in` (Clause.id list).
- Every semantic edge MUST have `source_clause` + `source_text` (substring-verified).
- Structural ID format is fixed; parse with `src.ids` — never construct ad hoc.
- LLM extraction output is validated against Pydantic schema in `src/schema.py` AND post-checked that `source_text` appears verbatim inside `source_clause`. Fabrication = drop edge.

### Headline-metric discipline
- Main experiment = `python -m runtime.run_inference --arms main --n 200` + `python -m experiments.compute_academic_metrics --arms main`.
- Citation metrics compare `record["citation_ids"]` to `gold_citations_raw` after strict parse via `src/citations.py`. No per-script authority hardcoding.
- BERTScore runs fail-soft (skips if dep/model missing); citation metrics never silently fail — `validate_gold_citations` fails-hard before any metric runs.
- `evaluation.compute_judge_metrics` is **intentionally fail-closed** as a placeholder. Don't reintroduce judge metrics into the main flow without redesigning the rubric and getting buy-in.

### Prompt management
- Every system prompt lives in `prompts/` and is read through `src.prompts.load_prompt(rel_path)`. No long string literals in Python.
- Override via env var: `LEGAL_KG_PROMPTS_DIR=<dir>` — per-file fallback to the canonical `prompts/`. Use this for ablation runs instead of editing the canonical files.
- The Logic-LM IRAC renderer prefers `prompts/runtime/logic_lm/irac_with_plain.md` when present (produces both IRAC + plain_answer); falls back to `irac_render.md` if absent. This conditional is intentional — preserved from the original codebase.

### Module placement
- "Where do I put a new file?" — does it run **before** any user question (data prep, indexing, schema migration)? → `offline/`. Does it run **per user question** (retrieval, generation, repair)? → `runtime/`. Eval orchestration → `experiments/`. Deterministic metric engine → `evaluation/`. Reusable across both stages (IDs, schemas, registries, loader) → `src/`.

---

## Files to read FIRST when continuing

In this order:

1. **`reports/plan_v5_general_retrieval.md`** — current planning doc (vanilla → audit → conditional modules). Sprint 1 is "vanilla pipeline + audit" before any new module ships.
2. **`runtime/rag_query.py`** — `RagPipeline.vector_search` / `expand` / `fetch_facts` / `traverse` / `ask` / `verify_citations`. The retrieval surface you'll extend.
3. **`runtime/logic_lm_pipelines.py`** — three arm wrappers (`LogicLMNoRetrievalPipeline`, `LogicLMOntologyPipeline`, `LogicLMGraphRAGPipeline`). Each returns a `LogicLMAnswer` dataclass.
4. **`runtime/logic_lm/pipelines/program_pipeline.py`** — actual Prolog generation + repair loop (`_attempt`, `_validate_predicate_inputs`, `_verify`).
5. **`evaluation/compute_academic_metrics.py`** — exactly which metrics are headline + how `gold_articles` flow through. Don't add new metrics without reading this first.
6. **`src/citations.py`** — citation parsing + canonical formatting + the authority registry. All eval scripts must use this; never write a per-script parser.
7. **`prompts/runtime/logic_lm/irac_with_plain.md`** — well-structured JSON-emitting prompt; good template for new structured-output prompts.
8. **`schema/schema.cypher`** — current Neo4j constraints + vector indexes. Any new node label / edge type lands here first.

---

## Required skills (knowledge checklist)

### Critical
| Skill | Why | Where to learn |
|---|---|---|
| Neo4j Cypher | Schema, multi-hop, vector index | `offline/load_neo4j.py`, `schema/schema.cypher` |
| Prolog (SWI 9.x) | Logic-LM rule generation + repair loop | `runtime/logic_lm/pipelines/program_pipeline.py`, `runtime/logic_lm/solvers/prolog_solver.py` |
| Vietnamese legal terminology | "Mức bình quân tiền lương", "BHXH", "NLĐ/NSDLĐ", IRAC | Prompts in `prompts/`; `data/interim/structured_law.json` |
| OpenAI SDK | Generator + structured output + retries | `runtime/logic_lm/llm/client.py`, `offline/llm_extract.py` |
| BGE-M3 embedding | 1024-d Vietnamese-aware vectors, cosine = dot product (normalized) | `offline/embed.py` |
| Prompt engineering for structured JSON | Logic-LM emits parseable JSON envelopes | `prompts/runtime/logic_lm/rule_gen.md`, `irac_with_plain.md` |

### Important
| Skill | Why | Where to learn |
|---|---|---|
| Pydantic v2 | All schemas in `src/schema.py` | Pydantic docs + existing models |
| PowerShell on Windows | Primary dev environment is Windows | `scripts/*.ps1` |
| Pandas / Parquet | `embeddings.parquet`, metric CSVs | `offline/embed.py`, `evaluation/compute_academic_metrics.py` |
| Regex (Vietnamese) | ID / citation / number parsing | `src/ids.py`, `src/citations.py` |

### Nice-to-have
| Skill | Why |
|---|---|
| Information retrieval theory | Plan v5 §4 hybrid + rerank stages |
| Cross-encoder rerankers | Sprint 2+ conditional module in plan v5 |
| Temporal logic | `effective_from / effective_until` filtering across law versions |

---

## Common gotchas

1. **Idempotency**: every offline step skips when output exists. Pass `--force` to re-run (e.g. `python -m offline.embed --force`).
2. **OPENAI_BASE_URL empty string**: every runtime entry-point defensively pops `OPENAI_BASE_URL` if it's blank, otherwise the SDK uses the empty string as the URL and you get `APIConnectionError`. Keep this pop when adding new runtime entry-points.
3. **OpenAI API errors silently mark `unable_to_conclude`**: `runtime/logic_lm/pipelines/program_pipeline.py:_attempt` catches `Exception` broadly. When OpenAI throws `APIConnectionError` / `RateLimitError`, the record looks like a Prolog failure. Watch for `prompt_tokens=0 + completion_tokens=0` as a signal of API errors vs real reasoning failures.
4. **gpt-5-mini `legal_source(...,article: none,...)`**: reasoning model honestly declares unknown article numbers as `none`. Fallback regex `article_(\d+)` does not match → `citation_ids=[]`. Document as a model-behaviour caveat; mitigation = give explicit article numbers from KG (the v5 retrieval improvements).
5. **Path("experiments/prompts/...")**: gone. All prompts now under `prompts/`; load via `src.prompts.load_prompt(rel)`.
6. **Module paths**: nothing in `runtime/` may import from `offline/` (and vice versa). Both may import from `src/`. Cross-imports between offline and runtime should go through `src/` shared utilities.
7. **REPO_ROOT**: `runtime/logic_lm/cli/*.py` uses `Path(__file__).resolve().parents[3]` — depth from CLI file to repo root is exactly 3 levels (`cli → logic_lm → runtime → repo`). Don't change this without auditing the CLI entry-points.
8. **plain_answer backfill**: pre-2026-05-27 records lack `plain_answer`. Generate via `python -m runtime.rerender_plain_answer --combos all` (~$0.72 on gpt-4o-mini).
9. **B5 embedding env**: torch 2.6.0+cu124, datasets 3.0.1, pyarrow 17 on Windows — pin all three together (see project memory `feedback_b5_pin_versions.md`). Wrong combo causes 3 cascading errors.
10. **Reports directory**: only `reports/*.md` files explicitly listed in `.gitignore` exception are tracked. `plan_v5_general_retrieval.md` is the current canonical plan.

---

## Recommended workflow

### Adding a new inference arm
1. Decide the arm name (e.g. `logic_lm_decomposed`). Add it to `ALL_ARMS` and `MAIN_EXPERIMENT_ARMS` in `experiments/arms.py` if it belongs to the headline set.
2. Build the pipeline class in `runtime/<your_arm>.py` returning `LogicLMAnswer` (or `RagAnswer` shape), so the inference orchestrator can route it uniformly.
3. If you need a new prompt: add `.md` under `prompts/runtime/<your_arm>/...` and load via `src.prompts.load_prompt(...)`.
4. Wire a runner in `runtime/run_inference.py:ARM_RUNNERS`.
5. Pilot: `python -m runtime.run_inference --arms <your_arm> --n 10`.
6. Full: `python -m runtime.run_inference --arms <your_arm> --n 200`.
7. Headline metrics: `python -m experiments.compute_academic_metrics --arms <your_arm>` (or `main` to include it in the comparison set).

### Adding a new node label / edge type
1. Edit `schema/schema.cypher` (constraints + index if needed). Idempotent `IF NOT EXISTS`.
2. Add Pydantic model in `src/schema.py`.
3. Extend the loader in `offline/load_neo4j.py` with a matching UNWIND/MERGE block.
4. Run `python -m offline.load_neo4j --apply-schema` (and `--reset` only if you're rebuilding from scratch — destructive).
5. Sanity Cypher: confirm count + provenance roundtrip.

### Running the full pipeline end-to-end
```powershell
python -m offline.parse_docx
python -m offline.rule_extract
python -m offline.llm_extract
python -m offline.merge_normalize
python -m offline.embed
python -m offline.load_neo4j --apply-schema
python -m offline.build_logic_lm_corpus_2024   # only if logic-LM arms will run

python -m runtime.run_inference --arms main --n 200
python -m evaluation.validate_gold_citations
python -m experiments.compute_academic_metrics --arms main
```

### Commit hygiene
- Use existing commit-message style: `feat:`, `fix:`, `refactor:`, `docs:`, `data:`, `chore:`.
- One coherent change per commit. The repo history is small and audited — don't bundle unrelated changes.
- Prompts and code that depend on them belong in the same commit.

---

## Environment + setup

```bash
# Python 3.10–3.12 (pyproject `requires-python = ">=3.10,<3.13"`)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,eval]"

# .env (project root)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini          # generator (override per-job when needed)
NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j
EMBED_DEVICE=cuda                  # or cpu
HF_HUB_DISABLE_SYMLINKS_WARNING=1  # silences HF warnings on Windows
# Optional — point at an alternate prompts directory for ablations:
# LEGAL_KG_PROMPTS_DIR=experiments/my_ablation_prompts

# External services
- Neo4j 5.x (Aura cloud or local with vector-index support)
- OpenAI API (Tier 4+ for safe concurrency)
- SWI-Prolog 9.x (logic-LM arms only)
```

---

## When you finish a task

- Update `reports/plan_v5_general_retrieval.md` if the plan itself changed.
- Update this skill file when the architecture or workflow changes (file paths, arm list, eval flow).
- If a new defensible empirical claim emerges, document it next to the metrics CSV / JSON it depends on.
