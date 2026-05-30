# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned — HyDE retrieval with Qwen 2.5 3B on Colab Free (experiment 08)

**Scope**: Hypothetical Document Embeddings on the BGE-M3 dense channel
of `V5RetrievalPipeline`. Generator = Qwen 2.5 3B Instruct local on
Colab Free T4 (no OpenAI API). 4 arms — `dense`, `dense_hyde`,
`full_rerank`, `full_rerank_hyde` — same metric suite as experiment 07.

**Motivation**: experiment 06 funnel showed dense is the dominant
signal source, but BHXH questions are written in casual narrative style
while KG clauses are formal legal text. HyDE should bridge that style
gap.

**Plan**: [`docs/plans/hyde_qwen_colab.md`](plans/hyde_qwen_colab.md) —
fully self-contained: 15 locked decisions (D1-D15), file-by-file code
surface, phase plan, gate criteria, risks, prerequisites. Ready for
implementation in a fresh session.

**Status**: design accepted 2026-05-30. Implementation phases 1-10
pending. Branch strategy: `exp/08-hyde` from `main`.

**Success criteria** (in_corpus stratum): R@12 ≥ +3pp absolute, or
NDCG@10 ≥ +5% rel, or R-Precision ≥ +15% rel. Win triggers ADR 002.

### Decided — retrieval default = `full_rerank` arm at K=12 (Decision 001)

**Scope**: Ratifies the existing `V5RetrievalPipeline.rerank2_top_k=12`
default + the full M2 pipeline (hybrid + RRF + rerank + REFERS_TO expand
+ rerank2) as the production retrieval arm. No code change — this
documents the choice with evidence so future PRs do not drift the
default without re-running the audit.

**Evidence**: experiments 06 (K=12 A/B) and 07 (K extended to 100) on
200 BHXH questions show:
- `full_rerank` K=12 wins all rank-aware metrics on in_corpus stratum:
  R-Precision +103%, NDCG@10 +33%, F1@12 +52% vs `dense_only`.
- Marginal recall per added K plateaus after K=30 (ΔR/ΔK drops from
  0.010 in 12→20 to 0.0002 in 50→70). Going to K=30+ buys recall but
  exceeds the `MAX_CONTEXT_CHARS=7000` LLM budget — extra retrieval
  cost not visible to generator.

**Doc**: [docs/decisions/001_retrieval_k_and_arm.md](decisions/001_retrieval_k_and_arm.md).

**Triggers to revisit** are listed in the decision record — corpus
growth, context-budget bump, reranker swap, registry alias fixes, or
a new arm beating R-Precision at K=12.

### Added — exp 06 + exp 07: retrieval-only A/B experiments

- [experiments/06_retrieval_dense_vs_full](../experiments/06_retrieval_dense_vs_full/):
  `dense_only` vs `full_rerank` at K∈{5,10,12,20,30,all} on full 200
  questions. Adds R-Precision, MRR, NDCG@10 rank-aware metrics on top
  of recall / precision / F1.
- [experiments/07_retrieval_extended_k](../experiments/07_retrieval_extended_k/):
  same arms scaled up (full pipeline rerank2_top_k=100, dense_k=100)
  at K∈{12,20,30,50,70,100,all}. Reveals the elbow at K=20-30 and the
  plateau after K=50 that underpins Decision 001.

**New code**: [`src/retrieval/pipeline.py::retrieve_dense_only`](../src/retrieval/pipeline.py),
[`scripts/exp06_run.py`](../scripts/exp06_run.py),
[`scripts/exp06_metrics.py`](../scripts/exp06_metrics.py),
[`scripts/exp07_run.py`](../scripts/exp07_run.py),
[`scripts/exp07_metrics.py`](../scripts/exp07_metrics.py).

### Added — multi-law KG ingestion (ND143_2018 + QD838_BHXH)

**Scope**: First non-QH legal documents loaded into KG. Adds 2 entries to
`data/legal_metadata.yaml` + 2 raw `.docx` to `data/graph/raw/`. KG grows from
3 laws (486 art, 1585 cl) to 5 laws (507 art, 1645 cl).

**Hidden drift fixed**: `src/schema.py::SemanticEdge._must_be_clause_id` còn
regex literal `^L\d+_\d{4}…` — drift cùng category với `ids._ID_PATTERN` đã
relax session trước. Relax thành `^[A-Z][A-Z0-9_]*…` để chấp nhận mọi source
prefix trong registry (ND/QĐ/TT/CV/...). 5 test mới trong `tests/test_schema.py`.

**Documents flagged for follow-up PR**: 4 trong 6 file raw không load được
do parser limitation — pattern "cover decree + attached procedure" (2 QĐ-BHXH)
hoặc "main document + appendix" (2 NĐ). Đề xuất: thêm field YAML
`appendix_markers: [PHỤ LỤC, QUY TRÌNH]` (data-driven, opt-in, ~10 dòng code).

**Operational gotcha**: `OPENAI_BASE_URL=` empty string trong `.env` làm SDK
fail với APIConnectionError. Workaround inline:
`OPENAI_BASE_URL=https://api.openai.com/v1 python -m ...`.
Đề xuất fix code: pop env var nếu rỗng trong `offline/llm_extract.py`.

**Reference**: full report [docs/known_issues_kg_build.md](known_issues_kg_build.md).

### Changed — citation parser strict mode (v5 Sprint 2 Phase 0a)

**Rationale**: v5 Sprint 1 audit showed loose `parse_displayed_citations` cross-stream
matching produced false positives (e.g. inline mention "Bộ luật Lao động" paired with
"Điều 64" from a separate sentence → spurious `L45_2019.A64`). User-stated requirement:
*"citation phải đúng luật, đúng điều khoản. ko thể đúng điều khoản mà sai luật được"*.

**Change**: `src/citations.parse_displayed_citations` now accepts citations **only**
when authority alias and `Điều X[ khoản Y[ điểm z]]` co-occur inside the **same**
square-bracket `[...]` block. Inline mentions outside brackets, and brackets with
multiple authorities, are rejected.

**Impact on frozen baseline `experiments/01_initial_eval/`** (re-aggregated, skill Rule 2):

| arm | recall_macro old (loose) | recall_macro new (strict) | Δ |
|---|---:|---:|---:|
| graphrag (own regex parser, untouched) | 0.1292 | 0.1120 | -0.017 (metric-engine drift only) |
| llm_only | 0.0142 | 0.0067 | -0.0075 |
| logic_lm_no_retrieval | — | 0.0023 | — |
| logic_lm_ontology | — | 0.0073 | — |
| logic_lm_graphrag | — | 0.0175 | — |

**Impact on `experiments/03_v5_sprint1_vanilla/`** (graphrag_v5 arm re-aggregated):

| metric | old (loose) | new (strict) | Δ |
|---|---:|---:|---:|
| recall_macro | 0.2361 | 0.2361 | 0 (no true positive dropped) |
| precision_macro | 0.1867 | 0.2133 | +0.027 (+14% relative) |
| f1_macro | 0.1915 | 0.2093 | +0.018 |

**Why the asymmetry**: v5 vanilla prompt already enforced template emission; strict
parser only dropped FPs. Logic-LM arms emit verbose IRAC text with inline mentions;
strict mode rejects those mentions → recall tumbles. This reveals that prior numbers
were partially-inflated by loose-mode catching inline cites; new numbers reflect what
the LLM emits in canonical citation format.

**Re-parse tool**: `scripts/reparse_citations.py` walks an experiment's records,
re-parses `record["answer"]` with the current strict parser, and writes back
`citation_ids` / `citations` (keeps the original list under
`citation_ids_pre_strict_parser` for one-time audit).

### Added — citation parser strict mode (Phase 0a continued)

- `src/citations._BRACKET_BLOCK_RE` — only-source-of-truth regex for the new strict mode.
- `prompts/runtime/graphrag_v5_system.md` rewritten to enforce template emission
  with explicit DO/DON'T examples.

### Added — v5 Sprint 2 Phase 0b: hash-sealed eval split

- `scripts/seal_eval_split.py` — stratified 150 test / 50 dev split with SHA256 lock.
- Strata (gold_citations_raw corpus type): in_corpus 151, mixed 5, ooc 9, unparseable 35.
- Outputs (now exception-listed in `.gitignore`):
  - `data/eval/questions_150_test.json` (n=150)
  - `data/eval/questions_50_dev.json` (n=50)
  - `data/eval/eval_split_hashes.json` (lock record)
- Use `--verify` in CI to assert lock integrity.

### Added — v5 Sprint 2 Phase 1: synthetic Q/clause training data

- `prompts/offline/synthetic_query_gen.md` + `synthetic_pair_verifier.md`.
- `offline/build_synthetic_qa.py` — async pipeline, idempotent, resumable. Per clause:
  query generation → graph proximity candidates → vanilla BGE-M3 distance filter
  → LLM verifier (YES/PARTIAL/NO) → multi-positive row with hard negatives + easy
  random negatives. Eval-leak invariant: reads only `data/graph/processed/merged_graph.json`.
- Output `data/finetune-bge/qa_pairs_v1.jsonl` (committed via `.gitignore` exception).
- Cost estimate (1585 clauses, gpt-4o-mini): ~$5.2.

### Added — v5 Sprint 2 Phase 2: Colab fine-tune notebook

- `notebooks/finetune_bge_m3.ipynb` — single-notebook flow:
  clone repo → style spot-check → LoRA (r=16) → train (2 epoch, MNRL) →
  dev recall@K sanity → save adapter to Drive.
- LoRA target_modules = `[query, key, value, dense]` on XLM-RoBERTa transformer
  inside BGE-M3.

### Added — v5 Sprint 2 Phase 3: tuned-index plumbing

- `src/bge_m3_loader.py` — single loader for BGE-M3 with optional LoRA adapter,
  used by both `offline/embed.py` and `src/retrieval/pipeline.py` so corpus
  and query encoding stay symmetric.
- `offline/embed.py` — `--adapter-path`, `--output`, `--batch-size` flags.
- `offline/load_neo4j.py` — `--embeddings`, `--embed-prop`, `--embeddings-only`
  flags so tuned vectors load into `n.embedding_tuned` next to vanilla
  `n.embedding`.
- `schema/schema.cypher` — added `article_vec_tuned`, `clause_vec_tuned`,
  `point_vec_tuned` indexes (additive, `IF NOT EXISTS`).
- `src/retrieval/hybrid_retriever.py` — `dense_index` / `sparse_index`
  constructor params (default keeps Sprint 1 behaviour).
- `src/retrieval/pipeline.py` — `adapter_path`, `dense_index`, `reranker_model`
  constructor + env fall-backs (`BGE_M3_ADAPTER_PATH`, `V5_DENSE_INDEX`,
  `V5_RERANKER_MODEL`). One env var flip swaps the entire encoding stack.

### Added — academic metrics pipeline (original Unreleased line)

- Academic metrics pipeline:
  `evaluation.compute_academic_metrics` core over preloaded record lists,
  experiment-owned result loading in `experiments.compute_academic_metrics`,
  strict `gold_citations_raw` validation, shared citation registry, citation
  recall/precision/F1, citation display rate, latency, BERTScore, and Prolog
  reliability.

### Changed
- Main experiment evaluation now uses only academic metrics. Legacy judge-based
  metrics and their generated reports/data artifacts were removed from the main
  workflow.

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
- **Eval framework** (`experiments/`): GraphRAG vs LLM-only on 200 BHXH
  questions with generated per-question results for later metric computation.
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
