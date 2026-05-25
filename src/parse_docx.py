"""B1 — Parse Luật .docx thành JSON cây Chương → Mục → Điều → Khoản → Điểm.

NGUYÊN TẮC: hoàn toàn DETERMINISTIC. Chỉ dùng regex trên text gốc của file
docx. KHÔNG suy đoán, KHÔNG gọi LLM, KHÔNG sinh nội dung mới. Mọi chuỗi
trong JSON output đều phải đối chiếu được 1-1 với nội dung trong docx
(chỉ strip whitespace 2 đầu của paragraph; nối các dòng wrap của title
bằng dấu cách).

Cấu trúc output (data/interim/structured_law.json):
    {
      "law": {id, code, title, issuer, issued_date, effective_date, session},
      "preamble": [str],
      "postamble": [str],
      "chapters": [{
        id, number, roman, title,
        sections: [{id, number, title, chapter_id}],   # có thể rỗng
        articles: [{
          id, number, title, chapter_id, section_id, lead_text, text,
          clauses: [{id, number, text, full_text, article_id,
                     points: [{id, letter, text, clause_id}]}],
          tables: [{id, article_id, rows}],
        }],
      }]
    }

Bảo đảm cứng (assertion):
- Đúng 11 Chương, 13 Mục, 141 Điều.
- Article numbers liên tiếp 1..141.
- Mọi Điều/Khoản/Điểm có text non-empty.
- Không có ID trùng.
Fail-fast nếu sai.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

from src import ids

# ---------------------------------------------------------------------------
# Regex (cố tình chặt để tránh false-positive)
# ---------------------------------------------------------------------------

_ROMAN = "IVXLC"
RE_CHAPTER = re.compile(rf"^Chương\s+([{_ROMAN}]+)\s*(.*)$")
RE_SECTION = re.compile(r"^Mục\s+(\d+)\s*(.*)$")
RE_ARTICLE = re.compile(r"^Điều\s+(\d+)\s*\.\s*(.*)$")
RE_CLAUSE = re.compile(r"^(\d+)\s*\.\s+(.+)$")
RE_POINT = re.compile(r"^([a-zđ])\)\s+(.+)$")

ROMAN_MAP = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
}

# Metadata patterns
RE_LAW_CODE = re.compile(r"Luật\s+số:\s*(\d+/\d{4}/QH\d+)")
RE_DATE = re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})")
RE_SESSION = re.compile(r"khóa\s+([IVXLC]+),?\s+kỳ\s+họp\s+(?:thứ\s+)?(\d+)")
RE_EFFECTIVE = re.compile(
    r"có\s+hiệu\s+lực\s+thi\s+hành\s+từ\s+ngày\s+" r"(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})"
)
# Marker postamble: đoạn kết "Luật này được Quốc hội ... thông qua ngày..."
RE_RATIFICATION = re.compile(r"Luật\s+này\s+được\s+Quốc\s+hội.*thông\s+qua")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def iter_block_items(doc):
    """Yield ('p', Paragraph) hoặc ('tbl', Table) theo thứ tự xuất hiện."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield "p", Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield "tbl", Table(child, doc)


def table_rows(tbl) -> list[list[str]]:
    return [[cell.text.strip() for cell in row.cells] for row in tbl.rows]


def fmt_date(day: str, month: str, year: str) -> str:
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _is_boundary(text: str) -> bool:
    """Trả True nếu paragraph này mở đầu Chương/Mục/Điều mới."""
    if RE_ARTICLE.match(text):
        return True
    if RE_SECTION.match(text):
        return True
    m = RE_CHAPTER.match(text)
    return bool(m and m.group(1).upper() in ROMAN_MAP)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


@dataclass
class _State:
    chapters: list[dict] = field(default_factory=list)
    current_chapter: dict | None = None
    current_section: dict | None = None
    current_article: dict | None = None
    current_clause: dict | None = None

    # Title buffer: dùng cho cả Chương + Mục.
    # `title_buffer` là list dòng đang tích lũy, `title_owner` là dict được set.
    title_buffer: list[str] | None = None
    title_owner: dict | None = None

    preamble: list[str] = field(default_factory=list)
    postamble: list[str] = field(default_factory=list)
    preamble_tables: list[list[list[str]]] = field(default_factory=list)
    postamble_tables: list[list[list[str]]] = field(default_factory=list)

    seen_first_chapter: bool = False
    in_postamble: bool = False


def _finalize_title(st: _State) -> None:
    """Gộp title_buffer thành 1 chuỗi và gán cho owner."""
    if st.title_buffer is not None and st.title_owner is not None:
        st.title_owner["title"] = " ".join(t.strip() for t in st.title_buffer if t.strip()).strip()
    st.title_buffer = None
    st.title_owner = None


def _handle_paragraph(text: str, st: _State, law: str) -> None:
    # ---- 0. Đã vào postamble: append tất cả ----
    if st.in_postamble:
        st.postamble.append(text)
        return

    # ---- 1. Detect ratification → enter postamble ----
    if RE_RATIFICATION.search(text):
        _finalize_title(st)
        st.in_postamble = True
        st.current_article = None
        st.current_clause = None
        st.postamble.append(text)
        return

    # ---- 2. Đang collect title buffer? ----
    if st.title_buffer is not None:
        if _is_boundary(text):
            _finalize_title(st)
            # fall through để xử lý Chương/Mục/Điều mới
        else:
            st.title_buffer.append(text)
            return

    # ---- 3. Chương ----
    m = RE_CHAPTER.match(text)
    if m and m.group(1).upper() in ROMAN_MAP:
        roman = m.group(1).upper()
        n = ROMAN_MAP[roman]
        ch = {
            "id": ids.chapter_id(law, n),
            "number": n,
            "roman": roman,
            "title": "",
            "sections": [],
            "articles": [],
        }
        st.chapters.append(ch)
        st.current_chapter = ch
        st.current_section = None
        st.current_article = None
        st.current_clause = None
        st.seen_first_chapter = True
        st.title_buffer = []
        st.title_owner = ch
        inline = m.group(2).strip()
        if inline:
            st.title_buffer.append(inline)
        return

    # ---- 4. Mục ----
    m = RE_SECTION.match(text)
    if m and st.current_chapter is not None:
        sec_n = int(m.group(1))
        sec = {
            "id": ids.section_id(law, st.current_chapter["number"], sec_n),
            "number": sec_n,
            "chapter_id": st.current_chapter["id"],
            "title": "",
        }
        st.current_chapter["sections"].append(sec)
        st.current_section = sec
        st.current_article = None
        st.current_clause = None
        st.title_buffer = []
        st.title_owner = sec
        inline = m.group(2).strip()
        if inline:
            st.title_buffer.append(inline)
        return

    # ---- 5. Điều ----
    m = RE_ARTICLE.match(text)
    if m and st.current_chapter is not None:
        art_n = int(m.group(1))
        art_title = m.group(2).strip()
        art = {
            "id": ids.article_id(law, art_n),
            "number": art_n,
            "title": art_title,
            "chapter_id": st.current_chapter["id"],
            "section_id": st.current_section["id"] if st.current_section else None,
            "lead_text": "",
            "clauses": [],
            "tables": [],
        }
        st.current_chapter["articles"].append(art)
        st.current_article = art
        st.current_clause = None
        return

    # ---- 6. Khoản (chỉ trong Điều) ----
    if st.current_article is not None:
        m = RE_CLAUSE.match(text)
        if m:
            cl_n = int(m.group(1))
            cl = {
                "id": ids.clause_id(law, st.current_article["number"], cl_n),
                "number": cl_n,
                "article_id": st.current_article["id"],
                "text": m.group(2).strip(),
                "points": [],
            }
            st.current_article["clauses"].append(cl)
            st.current_clause = cl
            return

    # ---- 7. Điểm (chỉ trong Khoản) ----
    if st.current_clause is not None:
        m = RE_POINT.match(text)
        if m:
            pt = {
                "id": ids.point_id(
                    law,
                    st.current_article["number"],
                    st.current_clause["number"],
                    m.group(1),
                ),
                "letter": m.group(1),
                "clause_id": st.current_clause["id"],
                "text": m.group(2).strip(),
            }
            st.current_clause["points"].append(pt)
            return

    # ---- 8. Continuation / fallback ----
    if not st.seen_first_chapter:
        st.preamble.append(text)
        return

    if st.current_clause is not None:
        if st.current_clause["points"]:
            st.current_clause["points"][-1]["text"] += "\n" + text
        else:
            st.current_clause["text"] += "\n" + text
        return

    if st.current_article is not None:
        if st.current_article["clauses"]:
            st.current_article["clauses"][-1]["text"] += "\n" + text
            st.current_clause = st.current_article["clauses"][-1]
        else:
            sep = "\n" if st.current_article["lead_text"] else ""
            st.current_article["lead_text"] += sep + text
        return

    # Không thuộc đâu rõ ràng → vào postamble (an toàn)
    st.postamble.append(text)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def _build_article_full_text(art: dict) -> str:
    """Ghép nguyên văn 1 Điều theo đúng thứ tự trong docx."""
    parts = [f"Điều {art['number']}. {art['title']}".rstrip()]
    if art["lead_text"]:
        parts.append(art["lead_text"])
    for cl in art["clauses"]:
        lines = [f"{cl['number']}. {cl['text']}"]
        for pt in cl["points"]:
            lines.append(f"{pt['letter']}) {pt['text']}")
        cl["full_text"] = "\n".join(lines)
        parts.append(cl["full_text"])
    return "\n".join(parts)


def _extract_metadata(st: _State, law_code: str) -> dict:
    meta = {
        "id": ids.law_id(law_code),
        "code": law_code,
        "title": None,
        "issuer": None,
        "issued_date": None,
        "effective_date": None,
        "session": None,
    }

    # Preamble table → issuer + code
    flat_pre_tbl = "\n".join(cell for tbl in st.preamble_tables for row in tbl for cell in row)
    if "QUỐC HỘI" in flat_pre_tbl:
        meta["issuer"] = "Quốc hội"
    m = RE_LAW_CODE.search(flat_pre_tbl)
    if m:
        meta["code"] = m.group(1)
        meta["id"] = ids.law_id(m.group(1))

    # Title: paragraph "LUẬT" + dòng kế tiếp
    for i, p in enumerate(st.preamble):
        if p.strip().upper() == "LUẬT" and i + 1 < len(st.preamble):
            meta["title"] = f"Luật {st.preamble[i + 1].strip()}"
            break

    # Issued date + session: từ postamble (paragraph ratification)
    flat_post = "\n".join(st.postamble)
    m = RE_DATE.search(flat_post)
    if m:
        meta["issued_date"] = fmt_date(m.group(1), m.group(2), m.group(3))
    m = RE_SESSION.search(flat_post)
    if m:
        meta["session"] = f"Khóa {m.group(1)}, kỳ họp thứ {m.group(2)}"

    # Effective date: trong text Điều 140
    all_text = "\n".join(art["text"] for ch in st.chapters for art in ch["articles"])
    m = RE_EFFECTIVE.search(all_text)
    if m:
        meta["effective_date"] = fmt_date(m.group(1), m.group(2), m.group(3))

    return meta


def parse_docx(path: Path, law_code: str = "41/2024/QH15") -> dict:
    doc = Document(str(path))
    law = ids.law_id(law_code)
    st = _State()

    for kind, block in iter_block_items(doc):
        if kind == "tbl":
            rows = table_rows(block)
            if st.in_postamble:
                st.postamble_tables.append(rows)
            elif not st.seen_first_chapter:
                st.preamble_tables.append(rows)
            elif st.current_article is not None:
                idx = len(st.current_article["tables"]) + 1
                st.current_article["tables"].append(
                    {
                        "id": ids.table_id(law, st.current_article["number"], idx),
                        "article_id": st.current_article["id"],
                        "rows": rows,
                    }
                )
            else:
                # Chương đã set nhưng chưa vào Điều — hiếm
                st.postamble_tables.append(rows)
            continue

        text = block.text.strip()
        if not text:
            continue
        _handle_paragraph(text, st, law)

    # Đóng title buffer nếu còn (an toàn)
    _finalize_title(st)

    # Build full text mỗi Article
    for ch in st.chapters:
        for art in ch["articles"]:
            art["text"] = _build_article_full_text(art)

    meta = _extract_metadata(st, law_code)

    return {
        "law": meta,
        "preamble": st.preamble,
        "postamble": st.postamble,
        "chapters": st.chapters,
    }


# ---------------------------------------------------------------------------
# Validation (fail-fast)
# ---------------------------------------------------------------------------


def validate(result: dict, expect_chapters=11, expect_sections=13, expect_articles=141) -> None:
    chapters = result["chapters"]
    assert (
        len(chapters) == expect_chapters
    ), f"Số Chương sai: mong đợi {expect_chapters}, thực tế {len(chapters)}"

    n_sec = sum(len(ch["sections"]) for ch in chapters)
    assert n_sec == expect_sections, f"Số Mục sai: mong đợi {expect_sections}, thực tế {n_sec}"

    n_art = sum(len(ch["articles"]) for ch in chapters)
    assert n_art == expect_articles, f"Số Điều sai: mong đợi {expect_articles}, thực tế {n_art}"

    seen_ids: set[str] = set()
    article_numbers: list[int] = []
    for ch in chapters:
        assert ch["title"], f"Chương {ch['roman']} thiếu title"
        assert ch["id"] not in seen_ids
        seen_ids.add(ch["id"])
        for sec in ch["sections"]:
            assert sec["title"], f"Mục {sec['id']} thiếu title"
            assert sec["id"] not in seen_ids, f"Trùng ID: {sec['id']}"
            seen_ids.add(sec["id"])
        for art in ch["articles"]:
            assert art["id"] not in seen_ids, f"Trùng ID: {art['id']}"
            seen_ids.add(art["id"])
            article_numbers.append(art["number"])
            assert art["title"], f"Điều {art['number']} thiếu title"
            assert art["text"].strip(), f"Điều {art['number']} text rỗng"
            for cl in art["clauses"]:
                assert cl["id"] not in seen_ids, f"Trùng ID: {cl['id']}"
                seen_ids.add(cl["id"])
                assert cl["text"].strip(), f"{cl['id']} text rỗng"
                for pt in cl["points"]:
                    assert pt["id"] not in seen_ids, f"Trùng ID: {pt['id']}"
                    seen_ids.add(pt["id"])
                    assert pt["text"].strip(), f"{pt['id']} text rỗng"

    assert article_numbers == list(
        range(1, expect_articles + 1)
    ), "Article number không liên tiếp 1..N"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    src = Path("data/raw/Luật-41-2024-QH15.docx")
    out = Path("data/interim/structured_law.json")

    if not src.exists():
        print(f"FAIL: không tìm thấy {src}", file=sys.stderr)
        return 1

    print(f"Đọc {src} ...")
    result = parse_docx(src)

    print("\n=== METADATA ===")
    for k, v in result["law"].items():
        print(f"  {k}: {v}")

    print("\n=== STATS ===")
    n_ch = len(result["chapters"])
    n_sec = sum(len(ch["sections"]) for ch in result["chapters"])
    n_art = sum(len(ch["articles"]) for ch in result["chapters"])
    n_cl = sum(len(art["clauses"]) for ch in result["chapters"] for art in ch["articles"])
    n_pt = sum(
        len(cl["points"])
        for ch in result["chapters"]
        for art in ch["articles"]
        for cl in art["clauses"]
    )
    n_tbl_art = sum(
        len(art.get("tables", [])) for ch in result["chapters"] for art in ch["articles"]
    )
    print(f"  Chương : {n_ch}")
    print(f"  Mục    : {n_sec}")
    print(f"  Điều   : {n_art}")
    print(f"  Khoản  : {n_cl}")
    print(f"  Điểm   : {n_pt}")
    print(f"  Bảng trong Điều  : {n_tbl_art}")
    print(f"  Bảng preamble    : {len(result.get('preamble') and [None]) if False else '—'}")
    print()
    for ch in result["chapters"]:
        n_sec_ch = len(ch["sections"])
        n_art_ch = len(ch["articles"])
        sec_info = f", {n_sec_ch} mục" if n_sec_ch else ""
        print(f"  Chương {ch['roman']:<5} ({n_art_ch:>3} điều{sec_info}) — {ch['title'][:80]}")
        for sec in ch["sections"]:
            n_in_sec = sum(1 for a in ch["articles"] if a.get("section_id") == sec["id"])
            print(f"      Mục {sec['number']} ({n_in_sec} điều) — {sec['title'][:80]}")

    print("\n=== VALIDATE ===")
    try:
        validate(result)
        print("  OK — 11 chương, 13 mục, 141 điều; không trùng ID; mọi text non-empty.")
    except AssertionError as e:
        print(f"  FAIL: {e}", file=sys.stderr)
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu: {out} ({out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
