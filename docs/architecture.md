# Architecture

## Highlights

- **End-to-end pipeline**: parse `.docx` → trích quan hệ ngữ nghĩa (rule + LLM)
  → load Neo4j với vector index → RAG Q&A có citation.
- **Provenance bất biến**: every semantic node/edge truy ngược được về
  Điều/Khoản gốc, verify byte-for-byte cả khi build và query. See
  [contributing.md § Provenance principle](contributing.md).
- **3 lớp chống bịa**: Pydantic schema → DB constraints → post-extraction
  substring check (đã loại 158/745 edges LLM bịa trên dataset gốc).
- **Eval framework** theo academic metrics: gold citation recall/precision/F1,
  citation display rate, latency, BERTScore, 3 Prolog reliability rates trên
  200 câu hỏi BHXH thực tế.
- **Multilingual native**: BGE-M3 1024-d embeddings, GPT-4o-mini generator,
  system prompt + UI tiếng Việt.

## Pipeline (B1 → B7)

```
┌──────────────────────────────────────────────────────────────┐
│                   data/raw/Luật-...docx                      │
└─────────────────────────────┬────────────────────────────────┘
                              │
            ┌─────────────────▼─────────────────┐
  B1 PARSE  │  offline/parse_docx.py            │  → structured_law.json
            │  (regex deterministic; 0 LLM)     │     (1,068 structural nodes)
            └─────────────────┬─────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
 B2 RULE EXTRACT       B3 LLM EXTRACT         (skip — chỉ structural)
 offline/rule_extract  offline/llm_extract
 - REFERENCES 387      - Subject 45
 - CITES_EXTERNAL 30   - Benefit 34
 - AMENDS 9            - Obligation 72
 - DEFINES 12          - 243 semantic edges
       │                      │
       └──────────┬───────────┘
                  ▼
   B4 MERGE  offline/merge_normalize.py    → merged_graph.json
            (dedup + filter orphan)          1,334 nodes + 1,942 edges
                  │
       ┌──────────┼──────────┐
       │                     │
       ▼                     ▼
 B5 EMBED              B6 LOAD NEO4J
 offline/embed.py      offline/load_neo4j.py
 BGE-M3 → 1043 vec     UNWIND/MERGE + vector index
       │                     │
       └──────────┬──────────┘
                  ▼
   B7 RAG  runtime/rag_query.py / runtime/chat.py
   user Q → vector search → expand graph → GPT-4o-mini → answer + cited [Điều X khoản Y]
```

## Project structure

```
legal-graph-kb/
├── offline/                     # B1–B6 build-time pipeline (data prep)
│   ├── parse_docx.py            # B1 — deterministic docx parser
│   ├── rule_extract.py          # B2 — regex extraction
│   ├── llm_extract.py           # B3 — OpenAI semantic extraction
│   ├── merge_normalize.py       # B4 — dedup + validate
│   ├── embed.py                 # B5 — BGE-M3 embeddings
│   ├── load_neo4j.py            # B6 — load Neo4j
│   └── build_logic_lm_corpus_2024.py  # logic-LM corpus + ontology build
├── runtime/                     # Inference runtime (B7 + chat + logic-LM arms)
│   ├── rag_query.py             # B7 — GraphRAG pipeline + ask()
│   ├── chat.py                  # Interactive REPL
│   ├── llm_only.py              # LLM-only baseline pipeline
│   ├── logic_lm/                # Symbolic-LLM hybrid package
│   ├── logic_lm_pipelines.py    # 3 arm-aware logic-LM wrappers
│   └── graphrag_retriever_adapter.py  # Adapt RagPipeline as logic-LM retriever
├── eval_core/                   # Shared experiment infrastructure
│   ├── experiment.py            # Experiment class + inheritance
│   ├── paths.py                 # Standard experiment-folder layout
│   ├── arms.py                  # Arm definitions + CLI parsing
│   ├── inference.py             # Per-arm inference orchestrator
│   ├── multimodel.py            # arm × model matrix orchestrator
│   ├── rerender.py              # plain_answer backfill
│   ├── gold.py                  # Gold-citation validator
│   ├── metrics.py               # Deterministic metric engine
│   ├── report.py                # CSV + Markdown writers
│   ├── runners.py               # Multi-arm metric loader
│   ├── judge.py                 # Fail-closed placeholder
│   └── cli.py / __main__.py     # `python -m eval_core <cmd> <exp>`
├── experiments/                 # One folder per experiment
│   ├── _template/               # Starter skeleton
│   └── 01_initial_eval/         # Frozen R1+R2 baseline (committed records)
├── src/                         # Shared utilities (used everywhere)
│   ├── ids.py                   # ID convention + reverse parser
│   ├── schema.py                # Pydantic models (provenance invariants)
│   ├── legal_metadata.py        # Multi-law metadata registry
│   ├── citations.py             # Citation parsing + registry
│   └── prompts.py               # Prompt loader (with override env var)
├── prompts/                     # Single source of truth for ALL system prompts
│   ├── offline/llm_extract.md           # B3 LLM extraction
│   ├── runtime/graphrag_system.md       # GraphRAG generator
│   ├── runtime/llm_only_system.md       # LLM-only baseline
│   └── runtime/logic_lm/                # Logic-LM Prolog gen + IRAC render variants
├── schema/schema.cypher         # Neo4j constraints + vector indexes
├── docs/                        # All project documentation (this folder)
├── scripts/                     # PowerShell wrappers (Windows-friendly)
│   ├── install_b5.ps1           #   B5 deps + model pre-download
│   ├── verify_b5.py             #   B5 env verify
│   └── chat.ps1                 #   chat REPL with UTF-8 console
├── tests/                       # 110+ pytest cases (provenance focus)
├── data/                        # KG + raw law + ontology (NOT experiment output)
│   ├── raw/                     #   Source law .docx
│   ├── interim/                 #   (ignored) B1-B3 intermediate JSON
│   ├── processed/               #   (ignored) B4-B5 final artifacts + extraction_summary
│   ├── logic_lm/                #   Logic-LM corpus + ontology
│   └── eval/questions_200.json  #   200-question benchmark (input only)
├── pyproject.toml               # Build, deps, ruff, pytest, coverage, mypy
├── .github/                     # CI + issue/PR templates
├── README.md                    # Brief entry point — points here
└── LICENSE                      # MIT
```

## Roadmap

- [ ] Hybrid search: vector + fulltext keyword (Neo4j fulltext index đã có sẵn)
- [ ] Multi-document KG (gộp với Luật BHXH cũ 58/2014 để truy vết kế thừa)
- [ ] Optional judge metrics module sau khi chốt rubric riêng
- [ ] Web UI thay vì REPL
- [ ] Stratified eval theo loại câu hỏi (định nghĩa / quyền / thủ tục / chế độ)

## Acknowledgements

- [Neo4j](https://neo4j.com/) — graph database + vector search
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — multilingual embeddings
- [OpenAI](https://openai.com/) — GPT-4o-mini for extraction + generation
- [BERTScore](https://github.com/Tiiiger/bert_score) — semantic reference metric
- ĐH Công nghệ Thông tin (UIT, ĐHQG TP.HCM) — research support
