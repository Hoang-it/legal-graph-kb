# Known issues — KG build/merge pipeline

Tổng hợp các khó khăn / thách thức / hidden drift phát sinh khi thực hiện load
các văn bản dưới luật (NĐ/QĐ) vào KG, và các vấn đề hệ thống quan sát được khi
build/merge graph. Chia 3 nhóm: **đã fix trong session này**, **document-specific
parser limits cần PR**, **operational gotchas**.

> Last updated: 2026-05-30 — session add-legal-document attempt for 6 raw files
> trong `data/graph/raw/` (3 Nghị định + 3 Quyết định BHXH).

---

## A. Issues đã fix trong session

### A.1 — Hidden regex drift: `SemanticEdge.source_clause` reject ID NĐ/QĐ/TT

- **File**: [`src/schema.py`](../src/schema.py) — `SemanticEdge._must_be_clause_id`.
- **Triệu chứng**: B3 (`offline/llm_extract.py`) chạy thành công về mặt JSON
  schema OpenAI, nhưng `validate_extraction` raise `ValidationError` cho mọi
  semantic edge của luật mới:

  ```
  source_clause phải là Clause.id hoặc Point.id, nhận: ND143_2018.A5.K1
  ```

  ⇒ Tất cả 18 Article của ND143_2018 đầu tiên fail B3 với cùng pattern.

- **Root cause**: Pydantic validator hardcode regex `^L\d+_\d{4}\.A\d+\.K\d+(\.[a-zđ])?$`
  — drift y hệt regex cũ trong `ids._ID_PATTERN` đã được relax ở session trước
  (Phase 1 của parse_id fix). Chứng minh **multi-law refactor chưa quét toàn bộ
  surface** — vẫn còn 1 chỗ giữ giả định "chỉ Luật QH".

- **Fix**: Đổi regex prefix thành `[A-Z][A-Z0-9_]*`, đồng bộ với
  `ids._ID_PATTERN` + `citations._INTERNAL_ID_RE`. **5 test mới** trong
  `tests/test_schema.py` cover NĐ/QĐ/TT prefix + reject lowercase/digit prefix.

- **Bài học**: Sau bất kỳ refactor multi-law nào, **grep toàn repo** cho regex
  literal `L\d+_\d{4}` / `L\\d+_\\d{4}` để không sót drift. Hiện không còn match
  (đã verify post-fix).

---

## B. Document-specific parser limitations (cần PR riêng)

4 document trong `data/graph/raw/` không load được vì parser
[`offline/parse_docx.py`](../offline/parse_docx.py) chưa cover pattern thực tế
của văn bản dưới luật Việt Nam. Per [add-legal-document skill](../.claude/skills/add-legal-document/SKILL.md)
Rule §7, **không sửa code trong luồng add-law** — flag lại cho PR riêng.

### B.1 — Pattern "cover decree + attached procedure"

Áp dụng cho:
- `366_QD-BHXH_704900.docx` — 3 Điều cover + 6 Điều quy trình đính kèm.
- `595_QD-BHXH_348047.docx` — 3 Điều cover + 51 Điều quy trình + 8 Chương riêng.

Cấu trúc văn bản:

```
QUYẾT ĐỊNH số 366/QĐ-BHXH
Điều 1. Ban hành kèm theo Quyết định này: Quy trình thu BHXH ...
Điều 2. Quyết định này có hiệu lực thi hành từ ngày ...
Điều 3. Trưởng Ban Quản lý ... có trách nhiệm thi hành.

QUY TRÌNH                       ← document đính kèm
Điều 1. Phạm vi điều chỉnh      ← Điều numbering BẮT ĐẦU LẠI từ 1
Điều 2. Các từ viết tắt
Điều 3. Phân cấp quản lý
...
```

Hiện tại parser regex match cả 2 nhóm "Điều 1." → tạo trùng ID
`<KEY>.A1`. Validate `Article number không liên tiếp 1..N` raise
`AssertionError`.

### B.2 — Pattern "main document + appendix với cấu trúc Khoản giả"

Áp dụng cho:
- `146_2018_ND-CP_357505.docx` — 43 Điều chính + 8 Điều mẫu hợp đồng KCB BHYT
  ở phụ lục.
- `158_2025_ND-CP_634792.docx` — 45 Điều + PHỤ LỤC I/II.

Cấu trúc văn bản (vd ND158_2025):

```
Điều 45. Trách nhiệm tổ chức thi hành
1. Bộ trưởng Bộ Nội vụ có trách nhiệm hướng dẫn ...
2. Bộ trưởng Bộ Tài chính có trách nhiệm chỉ đạo ...
3. Các Bộ trưởng ... có trách nhiệm thi hành.

PHỤ LỤC I                                                    ← marker không phải Điều
DANH MỤC CÔNG VIỆC KHAI THÁC THAN TRONG HẦM LÒ
1. Khai thác mỏ hầm lò.                                      ← match RE_CLAUSE → trùng K1
2. Khoan đá bằng búa máy cầm tay trong hầm lò.               ← trùng K2 của Điều 45
3. Đội viên cứu hộ mỏ.
...
```

Parser state machine sau Điều 45 còn `current_article = A45`. PHỤ LỤC I
không match Chương/Mục/Điều regex → fall into continuation. Item "1.",
"2.", "3." match `RE_CLAUSE = r"^(\d+)\s*\.\s+(.+)$"` → tạo Clause
trùng ID `<KEY>.A45.K1`. Validate phát hiện `Trùng ID:
ND158_2025.A45.K1` → fail.

### B.3 — Đề xuất giải pháp PR

**Thêm field YAML `appendix_markers`** vào `LawMetadata`:

```yaml
laws:
  ND158_2025:
    ...
    appendix_markers:
      - "PHỤ LỤC"
      - "PHỤ LỤC I"
      - "MẪU"
  QD366_BHXH:
    ...
    appendix_markers:
      - "QUY TRÌNH"
      - "PHỤ LỤC"
```

Parser logic: khi gặp paragraph khớp 1 marker trong list (case-insensitive,
trim), set `st.in_postamble = True` ⇒ mọi paragraph sau đó đi vào
`st.postamble` (không tạo node Article/Clause/Point). Pattern này
**reuse** cơ chế `in_postamble` đã có cho `RE_RATIFICATION` ở
[parse_docx.py:300-306](../offline/parse_docx.py#L300).

**Backward compat**: field default `()` ⇒ luật QH hiện tại không đổi behavior.
**Opt-in** y hệt `allow_no_chapter` / `llm_skip_articles` đã làm trước đó —
data-driven, không hardcode marker text trong code.

**Phạm vi sửa**: ~10 dòng code (1 field dataclass + 1 condition check trong
`_handle_paragraph`) + tests. Tương đương Phase 1 của
`allow_no_chapter` skill.

**Test plan**: dùng python-docx in-memory build fixture cho 2 pattern
(cover-decree, main+appendix), assert chỉ Điều của main content được giữ.

---

## C. Operational gotchas

### C.1 — `OPENAI_BASE_URL=` empty string trong `.env`

- **File**: `.env` line 9 — `OPENAI_BASE_URL=` (giá trị rỗng).
- **Triệu chứng**: B3 (`offline/llm_extract.py`) fail toàn bộ với
  `APIConnectionError: Connection error.`
- **Root cause**: OpenAI SDK đọc env var `OPENAI_BASE_URL` ngay cả khi
  `AsyncOpenAI()` không truyền `base_url=` argument. Empty string làm URL
  ⇒ kết nối fail.
- **Status code**: `BASE_URL = os.getenv("OPENAI_BASE_URL") or None` ở
  [`offline/llm_extract.py:46`](../offline/llm_extract.py#L46) **không bảo vệ**
  vì SDK đọc env trực tiếp, bypass biến `BASE_URL` của chúng ta.
- **Đã document** trong skill [`legal-kg-logic-extraction`](../.claude/skills/legal-kg-logic-extraction/SKILL.md)
  §Gotchas #2: *"every runtime entry-point defensively pops `OPENAI_BASE_URL`
  if it's blank"*. **Nhưng `offline/llm_extract.py` chưa implement pop**.
- **Workaround data-side** (đã dùng trong session): set inline trước khi gọi:

  ```bash
  OPENAI_BASE_URL='https://api.openai.com/v1' python -m offline.llm_extract --law <KEY>
  ```

- **Đề xuất fix code** (out of scope skill add-law):

  ```python
  # offline/llm_extract.py — đầu file, sau load_dotenv()
  if os.environ.get("OPENAI_BASE_URL", "") == "":
      os.environ.pop("OPENAI_BASE_URL", None)
  ```

### C.2 — Article skipped "no_clauses (lead_text only)" cho cover decree ngắn

- **Triệu chứng**: B3 cho QD838_BHXH skip cả 3 Điều với
  `skipped_reason: "no_clauses (lead_text only)"`. Graph kết quả có 3
  Article node nhưng 0 semantic edge từ LLM.
- **Root cause**: QD838 là quyết định 3 Điều rất ngắn (công bố danh
  mục), mỗi Điều chỉ có 1 đoạn text duy nhất (lead_text), không có
  "1.", "2." numbering. Parser sinh `clauses=[]`. `extract_article`
  ([llm_extract.py:457-468](../offline/llm_extract.py#L457)) skip vì
  không thể anchor edge vào Clause.
- **Behavior này đúng** theo design provenance: không có Clause anchor ⇒
  không thể trace ngược source_text ⇒ skip thay vì sinh edge bịa.
- **Hệ quả**: 3 Điều cover decree chỉ có structural node + text, không
  có semantic enrichment. **Chấp nhận được** với loại văn bản này — nội
  dung pháp lý chính nằm trong DANH MỤC đính kèm (table, không trong
  numbered clauses).

### C.3 — `RE_RATIFICATION` không bắt được "Nghị định này có hiệu lực ..."

- **Quan sát**: `RE_RATIFICATION = r"Luật\s+này\s+được\s+Quốc\s+hội.*thông\s+qua"`
  ([parse_docx.py:51](../offline/parse_docx.py#L51)) chỉ match Luật QH.
  Với NĐ/QĐ, parser không transition vào `in_postamble` ⇒ "Nơi nhận:",
  "TM. CHÍNH PHỦ", "KT. THỦ TƯỚNG", v.v. bị append vào continuation của
  Điều cuối.
- **Tác động**: Chất lượng nhẹ (embedding của Điều cuối bị pollute bởi
  trailing administrative text). Không crash. Không trùng ID.
- **Đề xuất fix** (cùng PR với B.3): thêm `RE_NGHI_DINH_POSTAMBLE = r"^(Nơi nhận|TM\.\s|KT\.\s)"`
  hoặc dùng `appendix_markers` để bao gồm postamble markers.

### C.4 — Pre-existing test failures (không liên quan session này)

Không phải session phát sinh, nhưng verify lại để baseline:

| Test | Lý do fail | Đề xuất |
|---|---|---|
| `test_academic_metrics.py::test_main_arm_preset_is_shared...` | `MAIN_EXPERIMENT_ARMS` thêm `graphrag_v5` / `graphrag_v5_m2` mà test cũ không update | Update assertion list trong test |
| `test_embed.py::test_columns_dung_format` | Cột `embed_text_preview` được thêm vào parquet (debugging field) nhưng test cũ assert exact 4 columns | Update assertion |

---

## D. Summary — load attempt 6 files (session 2026-05-30)

| File | Result | Issue ref |
|---|---|---|
| 143_2018_ND-CP_346012.docx | ✅ ND143_2018 loaded (18 art, 60 clauses, 19 semantic edges) | A.1 (fixed inline), C.1 (worked around) |
| 838_QD-BHXH_612897.docx | ✅ QD838_BHXH loaded (3 art structural-only) | A.1, C.1, C.2 (by-design) |
| 158_2025_ND-CP_634792.docx | ❌ Skipped — `Trùng ID: ND158_2025.A45.K1` | B.2 — needs `appendix_markers` PR |
| 146_2018_ND-CP_357505.docx | ❌ Skipped — Article numbering không liên tiếp (43+8 dup) | B.2 |
| 366_QD-BHXH_704900.docx | ❌ Skipped — cover decree + attached procedure | B.1 |
| 595_QD-BHXH_348047.docx | ❌ Skipped — cover decree + attached procedure (with chapters) | B.1 |

**Coverage**: 2/6 files (33%). 4 file blocked bởi cùng 1 parser limitation (B.1 + B.2 cùng chung pattern "marker bắt đầu content thứ cấp") ⇒ **1 PR** thêm `appendix_markers` field sẽ unblock cả 4.

---

## E. Outstanding work (next session)

1. **PR thêm `appendix_markers`** vào `LawMetadata` + `parse_docx` — unblock 4 file còn lại.
2. **PR pop `OPENAI_BASE_URL=""` env var** trong `offline/llm_extract.py` — eliminate workaround.
3. **PR update 2 pre-existing test failures** (`test_academic_metrics`, `test_embed`) — restore green suite.
4. **Optional**: thêm postamble markers cho NĐ ("Nơi nhận", "TM. CHÍNH PHỦ") để tránh pollute Điều cuối.
5. **Audit grep** cho `L\d+_\d{4}` regex literal toàn repo để đảm bảo không còn drift còn sót sau A.1.

Mỗi outstanding item nên tách 1 PR riêng để dễ review + revert nếu cần.
