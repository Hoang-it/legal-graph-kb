# Pull request

## Tóm tắt

<!-- Một-hai câu mô tả thay đổi và lý do. -->

## Loại thay đổi

- [ ] Bug fix (non-breaking)
- [ ] Feature mới (non-breaking)
- [ ] Breaking change (vỡ API hiện tại — cần bump version)
- [ ] Docs / chore / refactor

## Issue liên quan

<!-- "Closes #123" hoặc "Refs #456" — nếu có. -->

## Đã làm

<!-- Bullet list các thay đổi chính. -->

-
-
-

## Đã test

- [ ] `pytest tests/test_ids.py tests/test_schema.py tests/test_parse_docx.py` xanh (fast tests)
- [ ] `ruff check .` không error
- [ ] `ruff format --check .` không diff
- [ ] (Nếu sửa B6/B7) test integration với Neo4j live
- [ ] (Nếu sửa B3) chạy lại pilot 10 câu để confirm chất lượng

## Provenance check

> Reminder: mọi node/edge mới PHẢI có `source_clause` + reverse-traceable.
> Xem CONTRIBUTING.md mục "Provenance principle".

- [ ] Tôi đã đọc Provenance principle và xác nhận PR không vi phạm
- [ ] (Nếu thêm semantic edge/node mới) đã update schema constraints + test
- [ ] N/A — PR chỉ ảnh hưởng UX/tooling/docs

## Changelog

- [ ] Đã thêm entry vào `CHANGELOG.md` mục `[Unreleased]`
- [ ] Không cần (PR chỉ là docs/chore/internal refactor)

## Screenshots / output (nếu áp dụng)

<!-- Trước/sau cho UI changes, hoặc paste output mới cho CLI. -->
