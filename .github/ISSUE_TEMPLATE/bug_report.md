---
name: Bug report
about: Báo lỗi hành vi không mong đợi
title: "[BUG] "
labels: bug
assignees: ""
---

## Mô tả bug

<!-- Một-hai câu mô tả vấn đề. -->

## Bước tái hiện

1. Chạy lệnh `...`
2. Thấy output `...`
3. Mong đợi `...`

## Hành vi mong đợi

<!-- Nên xảy ra gì? -->

## Hành vi thực tế

<!-- Đang xảy ra gì? Paste stack trace nếu có. -->

```
<paste stack trace ở đây>
```

## Môi trường

- OS: <!-- Windows 11 / Ubuntu 22.04 / macOS 14 -->
- Python: <!-- `python --version` -->
- GPU: <!-- RTX 3050 / CPU-only / ... -->
- Phiên bản các package liên quan:
  - `neo4j` =
  - `openai` =
  - `sentence-transformers` =
  - `torch` =

## Bước nào trong pipeline?

- [ ] B1 — parse_docx
- [ ] B2 — rule_extract
- [ ] B3 — llm_extract
- [ ] B4 — merge_normalize
- [ ] B5 — embed
- [ ] B6 — load_neo4j
- [ ] B7 — rag_query / chat
- [ ] experiments/

## Thông tin thêm

<!-- Screenshot, config, log file... -->
