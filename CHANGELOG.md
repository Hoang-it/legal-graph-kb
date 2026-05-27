# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Academic metrics pipeline:
  `experiments.compute_academic_metrics`, strict `gold_citations_raw` validation,
  shared citation registry, citation recall/precision/F1, citation display rate,
  latency, BERTScore, and Prolog reliability.

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
- **Docs**: `README.md`, `docs/neo4j-setup.md`, `prompts/extract_v1.md`,
  `experiments/README.md`, `reports/extraction_summary.md`.

### Security
- `.env` excluded via `.gitignore` since first commit; `.env.example`
  documents required keys.
- All LLM responses post-validated against ground-truth Clause text → no
  fabricated content can persist to graph.

[Unreleased]: https://github.com/USER/legal-graph-kb/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/USER/legal-graph-kb/releases/tag/v0.1.0
