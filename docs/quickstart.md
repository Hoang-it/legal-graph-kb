# Quickstart

## Prerequisites

- Python **3.10 – 3.12**
- [Neo4j Desktop 5.x](https://neo4j.com/download/) + APOC plugin (see [`neo4j-setup.md`](neo4j-setup.md))
- [OpenAI API key](https://platform.openai.com/api-keys)
- (Khuyến nghị) GPU CUDA — RTX class trở lên cho B5 (CPU fallback hoạt động nhưng chậm)
- Windows: PowerShell 5.1+; macOS/Linux: bash/zsh

## Install

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

## Build the KG (B1–B6, one-time)

| # | Command | Output |
|---|---|---|
| B1 | `python -m offline.parse_docx` | `data/graph/interim/structured_law.json` |
| B2 | `python -m offline.rule_extract` | `data/graph/interim/{internal,external}_refs.json`, `definitions.json`, `amendments.json` |
| B3 | `python -m offline.llm_extract` | `data/graph/interim/llm_extractions/A*.json` |
| B4 | `python -m offline.merge_normalize` | `data/graph/processed/merged_graph.json` + `data/graph/processed/extraction_summary.md` |
| B5 | `python -m offline.embed` | `data/graph/processed/embeddings.parquet` |
| B6 | `python -m offline.load_neo4j` | Nodes + edges + embeddings nạp vào Neo4j |

Each step is idempotent: re-runs skip existing outputs (override with `--force`).

## Run an experiment (B7 inference + metrics)

The repo ships with [`experiments/01_initial_eval/`](../experiments/01_initial_eval/)
— 5 single-model arms × 200 questions + a 2-arm × 3-model multimodel matrix,
records committed.

```powershell
# Recompute metrics + report from committed records (no API calls)
python -m eval_core metrics experiments/01_initial_eval

# Create a new experiment that inherits the baseline
Copy-Item -Recurse experiments/_template experiments/02_my_idea
# Edit experiments/02_my_idea/config.yaml (set parent: 01_initial_eval, your arm: mode: run)
# Edit experiments/02_my_idea/README.md (WHAT/WHY + expected outcome)

python -m eval_core all experiments/02_my_idea
```

See [`eval_core.md`](eval_core.md) and [`experiments.md`](experiments.md) for the
full Experiment lifecycle, inheritance, prompt overrides, and git policy.

## Interactive chat REPL

```powershell
.\scripts\chat.ps1                       # Windows wrapper
# python -m runtime.chat                 # cross-platform
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

REPL commands: `/help`, `/sources`, `/verify`, `/save chat.md`, `/quit`.

## Development

```bash
# Editable install + dev tools (ruff, mypy, pytest)
pip install -e ".[dev]"

# Lint + format (CI enforced)
ruff check .
ruff format .

# Tests — fast (CI runs these)
pytest tests/test_ids.py tests/test_schema.py tests/test_parse_docx.py

# Tests — full (needs live Neo4j + OpenAI API + BGE-M3 model)
pytest

# Type check
mypy src/ offline/ runtime/ eval_core/

# Eval extras
pip install -e ".[eval]"
```

See [`contributing.md`](contributing.md) for the provenance principle and PR
conventions.
