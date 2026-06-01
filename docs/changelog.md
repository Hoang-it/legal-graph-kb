# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — Retrieval metrics consolidated into `eval_core` (single source of truth)

All metric computation now lives in `eval_core`, for **both** families. The
retrieval family previously had one bespoke `scripts/exp<NN>_metrics.py` per
experiment; the pure IR primitives (recall/precision/r_precision/mrr/ndcg/
categorize) were copy-pasted across `exp06–11` (with drift) and `exp12–14`
imported them cross-experiment — a maintenance hazard and a self-containment
(Rule 6) risk. This change reverses the 2026-06-01 "retrieval producer is not
generic / no refactor" decision.

- `eval_core/retrieval_metrics.py` — **new** generic, config-driven retrieval
  metric engine. Reads the experiment's `retrieval:` block (`arms`, `ks`,
  `record_field`, `latency_field`, `pilot_subset`), scores `results/<arm>/
  A*.json`, writes `metrics/academic_metrics.json` (`overall_macro` +
  `stratified` + `Ks`) + CSV + report. Primitives ported **verbatim** from the
  retired `exp09_metrics.py` — no metric definition changed, so numbers are
  unaffected (Rule 2).
- `eval_core/cli.py` — `metrics` (and `all`) dispatch on `config.family`:
  retrieval → `retrieval_metrics`, qa → the existing arm runners. `run`/`all`
  print a guard for retrieval (Tier-1 inference is not owned by `eval_core`).
  Added `--full` (ignore `retrieval.pilot_subset`).
- `experiment_contract.py` — the retrieval recompute **default** is now
  `eval_core_metrics` (was `module: scripts.exp<NN>_metrics`); both families
  recompute via `python -m eval_core metrics <exp>`. Docstrings/examples updated.
- Deleted all 22 orphaned `scripts/exp*.py` (9 `_metrics` + 9 `_run` + 3
  `_funnel` + `exp08_test_one`). The metric scripts are superseded by the
  engine above; the run/funnel scripts were already orphaned (their
  `experiments/<NN>/` targets had been purged).
- `experiments/_template/config.yaml`, `CONTRACT.md`, and the
  `legal-kg-logic-extraction` skill — updated to the unified flow (retrieval
  metrics via `eval_core`, config `retrieval:` block, no per-experiment script).
- `tests/test_retrieval_metrics.py` — **new** (primitives pinned to hand-computed
  values + end-to-end shape). `tests/test_experiment_contract.py` — retrieval
  default now asserts `eval_core_metrics`; real-folder checks rebuilt on
  synthetic fixtures (no dependency on purged experiment data).

### Changed — Citation metric engine: strict tuple-equal (`academic_v2`)

`eval_core/metrics.py` compares `pred_items ∩ gold_items` on the full
4-tuple `(law_id, article, clause, point)` instead of an article-only
intersection.

- `eval_core/gold.py` — normalizer emits `gold_items` (full tuple)
  alongside `gold_articles` (article-deduped, kept for backward compat);
  granularity flag set to `"tuple"`.
- `eval_core/metrics.py` — `compute_citation_metrics` rewritten;
  `_coerce_gold_articles` → `_coerce_gold_items` with legacy fallback;
  `METRIC_VERSION = "academic_v2"`.
- `eval_core/runners.py` — gold-attachment helpers accept either the old
  (`list[str]`) or new (`dict`) gold-map shape for backward compat.
- `tests/test_academic_metrics.py`, `tests/test_evaluation_sample_metrics.py`,
  `tests/fixtures/academic_metrics/expected_summary.json` — updated to the
  `academic_v2` arithmetic.

Strict tuple applies to E2E citation metrics only (LLM emits citation →
parsed → tuple); `law_id` must match at every layer. Frozen baseline records
remain immutable — only their aggregation logic changed.

### Changed — Citation parser strict mode

`src/citations.parse_displayed_citations` accepts a citation **only** when
the authority alias and `Điều X[ khoản Y[ điểm z]]` co-occur inside the
**same** `[...]` block; inline mentions outside brackets, and brackets with
multiple authorities, are rejected.

- `src/citations._BRACKET_BLOCK_RE` — single source of truth for strict mode.
- `prompts/runtime/graphrag_v5_system.md` — enforces template emission with
  explicit DO/DON'T examples.
- `scripts/reparse_citations.py` — re-parses an experiment's records with the
  current strict parser, preserving the pre-strict list under
  `citation_ids_pre_strict_parser`.

### Added — multi-law KG ingestion (ND143_2018 + QD838_BHXH)

First non-QH legal documents loaded into the KG. Adds 2 entries to
`data/legal_metadata.yaml` + 2 raw `.docx` to `data/graph/raw/`.

- **Schema drift fixed**: `src/schema.py::SemanticEdge._must_be_clause_id`
  regex relaxed from `^L\d+_\d{4}…` to `^[A-Z][A-Z0-9_]*…` to accept every
  source prefix in the registry (ND/QĐ/TT/CV/…). 5 new tests in
  `tests/test_schema.py`.
- **Parser limitation flagged**: some raw files don't load under the
  "cover decree + attached procedure" / "main document + appendix" patterns;
  proposed opt-in YAML `appendix_markers`. See
  [docs/known_issues_kg_build.md](known_issues_kg_build.md).
- **Operational gotcha**: an empty `OPENAI_BASE_URL=` in `.env` makes the SDK
  fail with `APIConnectionError`; pop the env var when blank.

### Added — eval & retrieval infrastructure

- `scripts/seal_eval_split.py` — stratified test/dev split with a SHA256 lock
  (`--verify` asserts lock integrity); outputs under `data/eval/`
  (exception-listed in `.gitignore`).
- `prompts/offline/synthetic_query_gen.md` + `synthetic_pair_verifier.md`,
  `offline/build_synthetic_qa.py` — async, idempotent, resumable synthetic
  Q/clause generator (reads only `data/graph/processed/merged_graph.json`);
  output `data/finetune-bge/qa_pairs_v1.jsonl`.
- `notebooks/finetune_bge_m3.ipynb` — LoRA (r=16) fine-tune flow for BGE-M3
  (XLM-RoBERTa `query/key/value/dense` targets).
- `src/bge_m3_loader.py` — single BGE-M3 loader with optional LoRA adapter,
  shared by `offline/embed.py` and `src/retrieval/pipeline.py` so corpus and
  query encoding stay symmetric.
- `offline/embed.py`, `offline/load_neo4j.py` — adapter / embeddings / index
  flags so tuned vectors load into `n.embedding_tuned` next to `n.embedding`.
- `schema/schema.cypher` — additive `*_vec_tuned` indexes (`IF NOT EXISTS`).
- `src/retrieval/{hybrid_retriever,pipeline}.py` — `dense_index` /
  `reranker_model` / `adapter_path` constructor args + env fall-backs
  (`BGE_M3_ADAPTER_PATH`, `V5_DENSE_INDEX`, `V5_RERANKER_MODEL`).

### Added — academic metrics pipeline

Deterministic metric engine over preloaded record lists with strict
`gold_citations_raw` validation, a shared citation registry, citation
recall/precision/F1, citation display rate, latency, BERTScore, and Prolog
reliability rates.

### Changed — evaluation uses academic metrics only

Main experiment evaluation uses academic metrics only; legacy judge-based
metrics and their generated reports/data artifacts were removed from the
main workflow.

## [0.1.0] — 2026-05-25

Initial public release.

### Added
- **B1** `src/parse_docx.py` — deterministic .docx → JSON tree
  (11 chapters · 13 sections · 141 articles · 543 clauses · 359 points).
- **B2** `src/rule_extract.py` — regex extraction (271 internal refs + 146 self
  refs + 30 external refs + 12 definitions + 9 amendments) with byte-for-byte
  `source_clause` + `char_offset` + `span` provenance.
- **B3** `src/llm_extract.py` — OpenAI `gpt-4o-mini` semantic extraction with
  JSON-schema strict mode; post-extraction `source_text` substring validation
  drops 104/347 unsupported edges (~30%).
- **B4** `src/merge_normalize.py` — dedup canonical IDs, filter orphan edges
  → 1,334 nodes (17 labels) + 1,942 edges (21 types).
- **B5** `src/embed.py` — BGE-M3 1024-d embeddings for 1,043 Article/Clause/Point
  units (~60s on RTX 3050).
- **B6** `src/load_neo4j.py` — idempotent MERGE load into Neo4j 5.x with
  existence constraints enforcing `source_clause` on every semantic edge.
- **B7** `src/rag_query.py` — RAG pipeline: vector search → graph expansion
  → GPT-4o-mini answer with `[Điều X khoản Y]` citations + reverse-DB verify.
- **CLI**: `src/chat.py` interactive REPL with `/sources`, `/verify`, `/save`,
  rich-formatted output (`scripts/chat.ps1` wrapper for UTF-8 on Windows).
- **Eval framework** (`experiments/`): per-question result records on the BHXH
  question set for later metric computation.
- **Tests**: 95 unit/integration tests covering provenance integrity
  (byte-for-byte source verification, dst-must-exist, reverse-DB lookups).
- **Docs**: `README.md`, `docs/neo4j-setup.md`,
  `prompts/offline/llm_extract.md` (was `prompts/extract_v1.md` pre-refactor),
  `docs/experiments.md` (was `experiments/README.md` pre-refactor),
  `data/graph/processed/extraction_summary.md` (was `reports/extraction_summary.md` pre-refactor).

### Security
- `.env` excluded via `.gitignore` since first commit; `.env.example`
  documents required keys.
- All LLM responses post-validated against ground-truth Clause text → no
  fabricated content can persist to graph.

[Unreleased]: https://github.com/USER/legal-graph-kb/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/USER/legal-graph-kb/releases/tag/v0.1.0
