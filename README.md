# legal-graph-kb

[![CI](https://github.com/USER/legal-graph-kb/actions/workflows/ci.yml/badge.svg)](https://github.com/USER/legal-graph-kb/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Neo4j 5.x](https://img.shields.io/badge/Neo4j-5.x-008cc1.svg)](https://neo4j.com/)

> **Knowledge Graph + GraphRAG cho Luật Bảo hiểm xã hội số 41/2024/QH15** (Việt Nam) — pipeline 7 bước từ văn bản `.docx` thuần → KG có provenance đầy đủ → chatbot Q&A trích dẫn `[Điều X khoản Y]` đáng tin cậy.

## Documentation

All project documentation lives in [`docs/`](docs/). Start there.

- 🚀 [Quickstart](docs/quickstart.md) — install, env vars, first inference run.
- 🏗️ [Architecture](docs/architecture.md) — pipeline diagram (B1–B7), packages, prompt loader, experiment model.
- 🧪 [eval_core](docs/eval_core.md) and [experiments](docs/experiments.md) — how to add an experiment, inheritance, headline metrics.
- 🗺️ [Plans](docs/plans/) — design plans for upcoming work.
- 📜 [Changelog](docs/changelog.md) · [Contributing](docs/contributing.md) · [Code of conduct](docs/code-of-conduct.md).

## Legal disclaimer

**Đây là project học thuật / nghiên cứu.** Output của hệ thống KHÔNG phải tư vấn pháp lý chuyên nghiệp. Khi cần áp dụng trong tình huống pháp lý cụ thể, hãy tham khảo luật sư hoặc cơ quan có thẩm quyền. Văn bản gốc *Luật Bảo hiểm xã hội số 41/2024/QH15* là văn bản pháp luật công khai của Quốc hội Việt Nam.

## License

[MIT](LICENSE) © Nguyễn Hữu Hoàng
