# legal-graph-kb

[![CI](https://github.com/USER/legal-graph-kb/actions/workflows/ci.yml/badge.svg)](https://github.com/USER/legal-graph-kb/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Neo4j 5.x](https://img.shields.io/badge/Neo4j-5.x-008cc1.svg)](https://neo4j.com/)

> **Knowledge Graph + GraphRAG cho Luật Bảo hiểm xã hội số 41/2024/QH15** (Việt Nam) — pipeline 7 bước từ văn bản `.docx` thuần → KG có provenance đầy đủ → chatbot Q&A trích dẫn `[Điều X khoản Y]` đáng tin cậy.

## ✨ Điểm nổi bật

- **Pipeline end-to-end**: parse `.docx` → trích quan hệ ngữ nghĩa (rule + LLM) → load Neo4j với vector index → RAG Q&A có citation
- **Provenance bất biến**: mọi node/edge ngữ nghĩa truy ngược được về Điều/Khoản gốc, verify byte-for-byte cả khi build và query (xem [CONTRIBUTING § Provenance principle](CONTRIBUTING.md))
- **3 lớp chống bịa**: Pydantic schema → DB constraints → post-extraction substring check (loại 158/745 edges LLM bịa)
- **Eval framework** theo academic metrics mới: gold citation recall/precision/F1,
  citation display rate, latency, BERTScore và Prolog reliability trên 200 câu hỏi BHXH thực tế
- **Multilingual native**: BGE-M3 1024-d embeddings, GPT-4o-mini generator, system prompt + UI tiếng Việt
- **CLI + REPL**: 8 entry-points + interactive chat với pretty output (rich)

## 🏗️ Kiến trúc

```
┌──────────────────────────────────────────────────────────────┐
│                   data/raw/Luật-...docx                      │
└─────────────────────────────┬────────────────────────────────┘
                              │
            ┌─────────────────▼─────────────────┐
  B1 PARSE  │  src/parse_docx.py                │  → structured_law.json
            │  (regex deterministic; 0 LLM)     │     (1,068 structural nodes)
            └─────────────────┬─────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
 B2 RULE EXTRACT       B3 LLM EXTRACT         (skip — chỉ structural)
 src/rule_extract      src/llm_extract
 - REFERENCES 387      - Subject 45
 - CITES_EXTERNAL 30   - Benefit 34
 - AMENDS 9            - Obligation 72
 - DEFINES 12          - 243 semantic edges
       │                      │
       └──────────┬───────────┘
                  ▼
   B4 MERGE  src/merge_normalize.py    → merged_graph.json
            (dedup + filter orphan)       1,334 nodes + 1,942 edges
                  │
       ┌──────────┼──────────┐
       │                     │
       ▼                     ▼
 B5 EMBED              B6 LOAD NEO4J
 src/embed.py          src/load_neo4j.py
 BGE-M3 → 1043 vec     UNWIND/MERGE + vector index
       │                     │
       └──────────┬──────────┘
                  ▼
   B7 RAG  src/rag_query.py / src/chat.py
   user Q → vector search → expand graph → GPT-4o-mini → answer + cited [Điều X khoản Y]
```

## 🚀 Quickstart

### Prerequisites

- Python **3.10 – 3.12**
- [Neo4j Desktop 5.x](https://neo4j.com/download/) + APOC plugin
- [OpenAI API key](https://platform.openai.com/api-keys)
- (Khuyến nghị) GPU CUDA — RTX class trở lên cho B5 (CPU fallback hoạt động nhưng chậm)
- Windows: PowerShell 5.1+; macOS/Linux: bash/zsh

### Install

```powershell
# 1. Clone
git clone https://github.com/USER/legal-graph-kb.git
cd legal-graph-kb

# 2. Virtual env
python -m venv .venv
.\.venv\Scripts\Activate.ps1            # Windows
# source .venv/bin/activate              # Linux/macOS

# 3. Install package + deps
pip install -e .

# 4. (Windows + GPU) install torch CUDA wheel + pre-download BGE-M3 (~4 GB)
.\scripts\install_b5.ps1 -Backend cu121

# 5. Setup secrets
Copy-Item .env.example .env
# → mở .env, điền OPENAI_API_KEY + NEO4J_PASSWORD
```

### Run the pipeline

| # | Command | Output |
|---|---|---|
| B1 | `python -m src.parse_docx` | `data/interim/structured_law.json` |
| B2 | `python -m src.rule_extract` | `data/interim/{internal,external}_refs.json`, `definitions.json`, `amendments.json` |
| B3 | `python -m src.llm_extract` | `data/interim/llm_extractions/A*.json` |
| B4 | `python -m src.merge_normalize` | `data/processed/merged_graph.json` + `reports/extraction_summary.md` |
| B5 | `python -m src.embed` | `data/processed/embeddings.parquet` |
| B6 | `python -m src.load_neo4j` | Nodes + edges + embeddings nạp vào Neo4j |
| B7 | `python -m src.rag_query -q "..."` | Câu trả lời với citation |

Mỗi step idempotent: chạy lại an toàn, skip output đã có (override bằng `--force`).

### Chat REPL

```powershell
.\scripts\chat.ps1                       # Windows
# python -m src.chat                      # cross-platform
```

```
bạn> Người sử dụng lao động có những trách nhiệm gì về bảo hiểm xã hội?

╭──── Trả lời (3.2s) ────────────────────────────────────────────╮
│ Người sử dụng lao động có các trách nhiệm sau:                  │
│ 1. Kê khai và nộp hồ sơ tham gia BHXH bắt buộc...               │
│    [Luật BHXH 2024 (41/2024/QH15), Điều 28 khoản 1]             │
│ 2. Đóng BHXH cho người lao động...                              │
│    [Luật BHXH 2024 (41/2024/QH15), Điều 117 khoản 1]            │
│ ...                                                              │
╰──────────────────────────────────────────────────────────────────╯
Citations: [Luật BHXH 2024 (41/2024/QH15), Điều 28 khoản 1]  ✓
```

Lệnh trong REPL: `/help`, `/sources`, `/verify`, `/save chat.md`, `/quit`.

## 📊 Evaluation (200 câu BHXH real)

```bash
python -m experiments.run_inference --arms main --n 200
python -m experiments.compute_academic_metrics
```

Output mặc định: `metrics/academic_metrics.json`, `metrics/academic_metrics.csv`,
và `metrics/academic_report.md`. Có thể đổi thư mục bằng `--output-dir`. Script sẽ dừng
nếu `gold_citations_raw` thiếu hoặc không parse được.

Headline metrics là deterministic/dataset-based: citation recall,
citation precision, citation F1, citation display rate, latency, BERTScore
và 3 prolog rates. Judge-model metrics được tách khỏi headline và không chạy
trong main experiment. Entrypoint `evaluation.compute_judge_metrics` hiện fail-closed
cho đến khi judge metrics được thiết kế và duyệt riêng.

## 📁 Project structure

```
legal-graph-kb/
├── src/                         # Pipeline modules (B1-B7 + chat)
│   ├── parse_docx.py            # B1 — deterministic docx parser
│   ├── rule_extract.py          # B2 — regex extraction
│   ├── llm_extract.py           # B3 — OpenAI semantic extraction
│   ├── merge_normalize.py       # B4 — dedup + validate
│   ├── embed.py                 # B5 — BGE-M3 embeddings
│   ├── load_neo4j.py            # B6 — load Neo4j
│   ├── rag_query.py             # B7 — RAG pipeline + ask()
│   ├── chat.py                  # Interactive REPL
│   ├── ids.py                   # ID convention + reverse parser
│   └── schema.py                # Pydantic models (provenance invariants)
├── experiments/                 # GraphRAG vs LLM-only eval framework
├── tests/                       # 95+ pytest cases (provenance focus)
├── schema/schema.cypher         # Neo4j constraints + vector indexes
├── prompts/extract_v1.md        # LLM extraction prompt template
├── docs/neo4j-setup.md          # Neo4j Desktop installation guide
├── scripts/                     # PowerShell wrappers (Windows-friendly)
│   ├── install_b5.ps1           #   B5 deps + model pre-download
│   ├── verify_b5.py             #   B5 env verify
│   └── chat.ps1                 #   chat REPL with UTF-8 console
├── data/raw/                    # Source law document
├── data/interim/                # (ignored) B1-B3 intermediate JSON
├── data/processed/              # (ignored) B4-B5 final artifacts
├── reports/                     # (ignored except this README) Markdown reports
├── pyproject.toml               # Build, deps, ruff, pytest, coverage, mypy
├── .github/                     # CI + issue/PR templates
├── README.md                    # ← bạn đang đọc
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE                      # MIT
```

## 🔬 Development

```bash
# Editable install + dev tools (ruff, mypy, pytest)
pip install -e ".[dev]"

# Lint + format (CI enforced)
ruff check .
ruff format .

# Tests — fast (CI runs these)
pytest tests/test_ids.py tests/test_schema.py tests/test_parse_docx.py

# Tests — full (cần Neo4j live + OpenAI API + BGE-M3 model)
pytest

# Type check
mypy src/

# Eval extras
pip install -e ".[eval]"
```

Xem [CONTRIBUTING.md](CONTRIBUTING.md) cho hướng dẫn chi tiết và provenance principle.

## 🗺️ Roadmap

- [ ] Hybrid search: vector + fulltext keyword (Neo4j fulltext index đã có sẵn)
- [ ] Multi-document KG (gộp với Luật BHXH cũ 58/2014 để truy vết kế thừa)
- [ ] Optional judge metrics module sau khi chốt rubric riêng
- [ ] Web UI thay vì REPL
- [ ] Stratified eval theo loại câu hỏi (định nghĩa / quyền / thủ tục / chế độ)

## ⚖️ Legal disclaimer

**Đây là project học thuật / nghiên cứu.** Output của hệ thống KHÔNG phải tư vấn pháp lý chuyên nghiệp. Khi cần áp dụng trong tình huống pháp lý cụ thể, hãy tham khảo luật sư hoặc cơ quan có thẩm quyền. Tác giả không chịu trách nhiệm về các quyết định pháp lý dựa trên output của hệ thống này.

Văn bản gốc *Luật Bảo hiểm xã hội số 41/2024/QH15* là văn bản pháp luật công khai của Quốc hội Việt Nam.

## 🤝 Contributing

PR / issue welcome. Đọc [CONTRIBUTING.md](CONTRIBUTING.md) trước. Tuân thủ [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## 📜 License

[MIT](LICENSE) © Nguyễn Hữu Hoàng

## 🙏 Acknowledgements

- [Neo4j](https://neo4j.com/) — graph database + vector search
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — multilingual embeddings
- [OpenAI](https://openai.com/) — GPT-4o-mini for extraction + generation
- [BERTScore](https://github.com/Tiiiger/bert_score) — semantic reference metric
- ĐH Công nghệ Thông tin (UIT, ĐHQG TP.HCM) — research support
