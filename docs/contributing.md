# Contributing to legal-graph-kb

Cảm ơn bạn quan tâm tới project! Đây là project demo/research, mọi đóng góp
đều được đón nhận.

## Development setup

```bash
# 1. Clone
git clone https://github.com/USER/legal-graph-kb.git
cd legal-graph-kb

# 2. Virtual env
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# 3. Install (editable + dev deps)
pip install -e ".[dev]"

# 4. (Optional) install eval deps (cho experiments/)
pip install -e ".[eval]"

# 5. Setup secrets
cp .env.example .env
# → mở .env, điền OPENAI_API_KEY + NEO4J_PASSWORD
```

Lưu ý: `torch` cần wheel riêng tuỳ GPU/CPU. Trên Windows + RTX, dùng:

```powershell
.\scripts\install_b5.ps1 -Backend cu121   # tự pick đúng wheel
```

Xem `scripts/install_b5.ps1` cho chi tiết. Linux/macOS: cài torch theo
[pytorch.org instructions](https://pytorch.org/get-started/locally/).

## Tests

```bash
# Fast tests (không cần infra) — luôn xanh, ~3 giây
pytest tests/test_ids.py tests/test_schema.py tests/test_parse_docx.py

# Full suite (cần Neo4j live + OpenAI key + BGE-M3 model)
pytest

# Skip integration tests
pytest -m "not integration"
```

CI chạy đúng tập fast tests trên Python 3.10/3.11/3.12.

## Code style

Project dùng [ruff](https://docs.astral.sh/ruff/) cho cả lint và format
(thay cho black + flake8 + isort).

```bash
ruff check .            # lint
ruff format .           # format
ruff check . --fix      # auto-fix các issue có thể
```

CI fail nếu `ruff check` báo error hoặc `ruff format --check` thấy code chưa
format. Chạy `ruff format .` trước khi commit.

## Type check (optional)

```bash
mypy src/
```

Codebase chưa type-strict; type checking là khuyến khích, không bắt buộc.

## Commit messages

Khuyến nghị theo [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: <description>` — tính năng mới
- `fix: <description>` — sửa bug
- `docs: <description>` — chỉ docs
- `refactor: <description>` — refactor không đổi behavior
- `test: <description>` — thêm/sửa test
- `chore: <description>` — build, CI, deps, etc.

Ví dụ: `feat(rag): add hybrid keyword+vector search`

## Pull requests

1. **Fork** repo, tạo branch từ `main`:
   `git checkout -b feat/your-feature`
2. **Code + test** thay đổi. Đảm bảo:
   - `pytest tests/test_ids.py tests/test_schema.py tests/test_parse_docx.py` xanh
   - `ruff check .` không có error
   - `ruff format --check .` không có diff
3. **`docs/changelog.md`** — thêm 1 dòng dưới `## [Unreleased]`
4. **Push + open PR** với mô tả ngắn gọn (vấn đề giải quyết, cách thay đổi,
   test bạn đã chạy)
5. PR được review → merge sau khi CI xanh

## Bug reports / feature requests

Mở [GitHub Issue](https://github.com/USER/legal-graph-kb/issues) dùng
template có sẵn (`.github/ISSUE_TEMPLATE/`).

Khi báo bug, cung cấp:
- Phiên bản Python (`python --version`)
- Phiên bản các deps liên quan (`pip show neo4j openai sentence-transformers`)
- OS (Windows/Linux/macOS) + GPU nếu liên quan
- Steps to reproduce
- Expected vs actual behavior
- Stack trace nếu có

## Provenance principle (project core invariant)

Đây là invariant quan trọng nhất của project — **PR nào vi phạm sẽ bị
reject**:

> Mọi node và relationship trong knowledge graph PHẢI có thể truy ngược về
> điều luật gốc trong `data/raw/Luật-41-2024-QH15.docx`. Cụ thể:
>
> - Structural nodes (`Article`/`Clause`/`Point`) lưu nguyên văn trong field
>   `text` (verified byte-for-byte trong test).
> - Semantic nodes có `mentioned_in: list[Clause.id]` (≥1).
> - Semantic edges có `source_clause: Clause.id` + `source_text: str` (PHẢI
>   là substring của text Clause).
> - Reference edges có `source_clause` + `span` + `char_offset`.

Tất cả enforced qua:
1. Pydantic models (`src/schema.py`)
2. Cypher constraints (`schema/schema.cypher`)
3. Post-extraction validation trong các script (loại bỏ output không pass)
4. Tests (`tests/test_*.py`)

## License

Bằng việc đóng góp, bạn đồng ý license đóng góp của mình theo
[MIT License](LICENSE) của project.
