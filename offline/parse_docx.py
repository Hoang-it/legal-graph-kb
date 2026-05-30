"""B1 - deterministic parser for Vietnamese legal Word documents.

The parser is intentionally structural only: it reads paragraphs/tables and
uses regex state transitions for Chapter -> Section -> Article -> Clause ->
Point. It does not call an LLM and it does not infer missing legal content.

Phase 6 makes the parser multi-law generic. Law metadata, expected counts,
source path, and output path come from `data/legal_metadata.yaml` or CLI args.
Legacy `.doc` files are converted to `.docx` through a real document engine
(LibreOffice if available, otherwise Microsoft Word COM on Windows).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docx import Document
from docx.oxml.ns import qn

from src import ids
from src.legal_metadata import LawMetadata, load_law_metadata, load_order, metadata_for


# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------

_ROMAN = "IVXLCDM"
RE_CHAPTER = re.compile(rf"^Chương\s+([{_ROMAN}]+)\s*(.*)$", re.IGNORECASE)
RE_SECTION = re.compile(r"^Mục\s+(\d+)\s*(.*)$", re.IGNORECASE)
RE_ARTICLE = re.compile(r"^Điều\s+(\d+)\s*\.\s*(.*)$", re.IGNORECASE)
RE_CLAUSE = re.compile(r"^(\d+)\s*\.\s+(.+)$")
RE_POINT = re.compile(r"^([a-zđ])\)\s+(.+)$", re.IGNORECASE)

RE_LAW_CODE = re.compile(r"(?:Luật|Bộ luật)\s+số:\s*(\d+/\d{4}/QH\d+)", re.IGNORECASE)
RE_DATE = re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})")
RE_SESSION = re.compile(r"khóa\s+([IVXLCDM]+),?\s+kỳ\s+họp\s+(?:thứ\s+)?(\d+)", re.IGNORECASE)
RE_EFFECTIVE = re.compile(
    r"có\s+hiệu\s+lực\s+thi\s+hành\s+từ\s+ngày\s+"
    r"(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
    re.IGNORECASE,
)
RE_RATIFICATION = re.compile(r"Luật\s+này\s+được\s+Quốc\s+hội.*thông\s+qua", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Document readers
# ---------------------------------------------------------------------------


def _candidate_soffice_paths() -> list[Path]:
    paths = []
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            paths.append(Path(found))
    paths.extend(
        [
            Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
            Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
        ]
    )
    return [p for p in paths if p.exists()]


def _convert_with_soffice(source: Path, out_dir: Path) -> Path | None:
    for soffice in _candidate_soffice_paths():
        cmd = [
            str(soffice),
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            str(out_dir),
            str(source),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        converted = out_dir / f"{source.stem}.docx"
        if proc.returncode == 0 and converted.exists():
            return converted
    return None


def _ps_quote(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _convert_with_word_com(source: Path, out_dir: Path) -> Path | None:
    converted = out_dir / f"{source.stem}.docx"
    script = f"""
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {{
  $doc = $word.Documents.Open({_ps_quote(source)}, $false, $true)
  $doc.SaveAs2({_ps_quote(converted)}, 16)
  $doc.Close($false)
}} finally {{
  $word.Quit()
}}
"""
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode == 0 and converted.exists():
        return converted
    return None


def ensure_docx(source: Path, out_dir: Path = Path("data/graph/interim/converted_sources")) -> Path:
    """Return a `.docx` path for a `.docx` or `.doc` source.

    `.doc` conversion is a real conversion step. If no document engine can
    perform the conversion, the caller gets a hard failure with diagnostics.
    """
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source document not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".docx":
        return source
    if suffix != ".doc":
        raise ValueError(f"Unsupported source extension {suffix!r}: {source}")

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    converted = out_dir / f"{source.stem}.docx"
    if converted.exists() and converted.stat().st_mtime >= source.stat().st_mtime:
        return converted.resolve()

    converted_by_soffice = _convert_with_soffice(source, out_dir)
    if converted_by_soffice:
        return converted_by_soffice.resolve()

    converted_by_word = _convert_with_word_com(source, out_dir)
    if converted_by_word:
        return converted_by_word.resolve()

    raise RuntimeError(
        "Cannot convert .doc source to .docx. Install LibreOffice/soffice or "
        "Microsoft Word COM support, then rerun parser. Source: "
        f"{source}"
    )


def open_document(source: Path) -> Document:
    return Document(str(ensure_docx(source)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def roman_to_int(value: str) -> int | None:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for ch in reversed(value.upper()):
        cur = vals.get(ch)
        if cur is None:
            return None
        if cur < prev:
            total -= cur
        else:
            total += cur
            prev = cur
    return total if total > 0 else None


def iter_block_items(doc):
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
    if RE_ARTICLE.match(text) or RE_SECTION.match(text):
        return True
    m = RE_CHAPTER.match(text)
    return bool(m and roman_to_int(m.group(1)))


def _fallback_metadata(law_code: str) -> LawMetadata:
    law_id = ids.law_id(law_code)
    return LawMetadata(
        id=law_id,
        code=law_id,
        full_id=law_code if "/" in law_code else law_id,
        title="",
        canonical_title=law_id,
        type="law",
        hierarchy_level="luật",
        priority=100,
        source_file=Path(""),
    )


def _resolve_metadata(law_code: str | None, metadata: LawMetadata | None) -> LawMetadata:
    if metadata is not None:
        return metadata
    try:
        return metadata_for(law_code or ids.LAW_ID_DEFAULT)
    except Exception:
        return _fallback_metadata(law_code or "41/2024/QH15")


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

    title_buffer: list[str] | None = None
    title_owner: dict | None = None

    preamble: list[str] = field(default_factory=list)
    postamble: list[str] = field(default_factory=list)
    preamble_tables: list[list[list[str]]] = field(default_factory=list)
    postamble_tables: list[list[list[str]]] = field(default_factory=list)

    seen_first_chapter: bool = False
    in_postamble: bool = False
    in_nested_quote: bool = False

    # Khi document không có dòng "Chương …" (NĐ/QĐ/TT — opt-in qua YAML
    # `allow_no_chapter: true`), lazy-synth 1 Chapter ảo lúc gặp Mục/Điều đầu
    # tiên để state machine có parent hợp lệ. Tắt mặc định ⇒ luật QH không đổi.
    allow_no_chapter: bool = False
    synth_chapter_title: str = ""


def _finalize_title(st: _State) -> None:
    if st.title_buffer is not None and st.title_owner is not None:
        st.title_owner["title"] = " ".join(t.strip() for t in st.title_buffer if t.strip()).strip()
    st.title_buffer = None
    st.title_owner = None


def _ensure_chapter(st: _State, law: str) -> None:
    """Lazy-synth Chương I khi `allow_no_chapter` bật và chưa thấy Chương nào.

    Văn bản dưới luật (NĐ/QĐ/TT) thường mở đầu trực tiếp bằng Mục/Điều — state
    machine chính cần một parent `Chapter` hợp lệ để chấp nhận chúng. Helper
    này được gọi ngay trước nhánh Section/Article; khi `allow_no_chapter=False`
    (mặc định), nó là no-op nên không ảnh hưởng các luật QH hiện hữu.
    """
    if st.current_chapter is not None or not st.allow_no_chapter:
        return
    ch = {
        "id": ids.chapter_id(law, 1),
        "law_code": law,
        "number": 1,
        "roman": "I",
        "title": st.synth_chapter_title,
        "sections": [],
        "articles": [],
    }
    st.chapters.append(ch)
    st.current_chapter = ch
    st.seen_first_chapter = True


def _update_quote_state(st: _State, text: str) -> None:
    """Track quoted replacement text in amendment articles.

    Vietnamese laws often amend another law by embedding quoted replacement
    articles. Those embedded paragraphs can start with `1.` or `a)`, but they
    are not clauses/points of the current law. Quote tracking keeps them as
    continuation text attached to the current top-level clause/point.
    """
    opens = text.count("“")
    closes = text.count("”")
    if opens > closes:
        st.in_nested_quote = True
    elif closes > opens:
        st.in_nested_quote = False


def _append_continuation(text: str, st: _State) -> None:
    if st.current_clause is not None:
        if st.current_clause["points"]:
            st.current_clause["points"][-1]["text"] += "\n" + text
        else:
            st.current_clause["text"] += "\n" + text
    elif st.current_article is not None:
        sep = "\n" if st.current_article["lead_text"] else ""
        st.current_article["lead_text"] += sep + text
    else:
        st.postamble.append(text)
    _update_quote_state(st, text)


def _handle_paragraph(text: str, st: _State, law: str) -> None:
    if st.in_postamble:
        st.postamble.append(text)
        return

    if RE_RATIFICATION.search(text):
        _finalize_title(st)
        st.in_postamble = True
        st.current_article = None
        st.current_clause = None
        st.postamble.append(text)
        return

    if st.title_buffer is not None:
        if _is_boundary(text):
            _finalize_title(st)
        else:
            st.title_buffer.append(text)
            return

    if st.in_nested_quote:
        _append_continuation(text, st)
        return

    m = RE_CHAPTER.match(text)
    if m:
        roman = m.group(1).upper()
        n = roman_to_int(roman)
        if n is not None:
            ch = {
                "id": ids.chapter_id(law, n),
                "law_code": law,
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

    m = RE_SECTION.match(text)
    if m and st.current_chapter is None:
        _ensure_chapter(st, law)
    if m and st.current_chapter is not None:
        sec_n = int(m.group(1))
        sec = {
            "id": ids.section_id(law, st.current_chapter["number"], sec_n),
            "law_code": law,
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

    m = RE_ARTICLE.match(text)
    if m and st.current_chapter is None:
        _ensure_chapter(st, law)
    if m and st.current_chapter is not None:
        art_n = int(m.group(1))
        art = {
            "id": ids.article_id(law, art_n),
            "law_code": law,
            "number": art_n,
            "title": m.group(2).strip(),
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

    if st.current_article is not None:
        m = RE_CLAUSE.match(text)
        if m:
            cl_n = int(m.group(1))
            cl = {
                "id": ids.clause_id(law, st.current_article["number"], cl_n),
                "law_code": law,
                "number": cl_n,
                "article_id": st.current_article["id"],
                "text": m.group(2).strip(),
                "points": [],
            }
            st.current_article["clauses"].append(cl)
            st.current_clause = cl
            _update_quote_state(st, text)
            return

    if st.current_clause is not None:
        m = RE_POINT.match(text)
        if m:
            letter = m.group(1).lower()
            pt = {
                "id": ids.point_id(
                    law,
                    st.current_article["number"],
                    st.current_clause["number"],
                    letter,
                ),
                "law_code": law,
                "letter": letter,
                "clause_id": st.current_clause["id"],
                "text": m.group(2).strip(),
            }
            st.current_clause["points"].append(pt)
            _update_quote_state(st, text)
            return

    if not st.seen_first_chapter:
        st.preamble.append(text)
        return

    if st.current_clause is not None:
        _append_continuation(text, st)
        return

    if st.current_article is not None:
        if st.current_article["clauses"]:
            _append_continuation(text, st)
            st.current_clause = st.current_article["clauses"][-1]
        else:
            _append_continuation(text, st)
        return

    st.postamble.append(text)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def _build_article_full_text(art: dict) -> str:
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


def _detect_source_metadata(st: _State) -> dict[str, Any]:
    out: dict[str, Any] = {}
    flat_pre_tbl = "\n".join(cell for tbl in st.preamble_tables for row in tbl for cell in row)
    if "QUỐC HỘI" in flat_pre_tbl.upper():
        out["issuer"] = "Quốc hội"
    if m := RE_LAW_CODE.search(flat_pre_tbl):
        out["full_id"] = m.group(1)

    for i, p in enumerate(st.preamble):
        if p.strip().upper() in {"LUẬT", "BỘ LUẬT"} and i + 1 < len(st.preamble):
            out["title"] = f"{p.strip().title()} {st.preamble[i + 1].strip()}"
            break

    flat_post = "\n".join(st.postamble)
    if m := RE_DATE.search(flat_post):
        out["issued_date"] = fmt_date(m.group(1), m.group(2), m.group(3))
    if m := RE_SESSION.search(flat_post):
        out["session"] = f"Khóa {m.group(1).upper()}, kỳ họp thứ {m.group(2)}"

    all_text = "\n".join(art["text"] for ch in st.chapters for art in ch["articles"])
    if m := RE_EFFECTIVE.search(all_text):
        out["effective_date"] = fmt_date(m.group(1), m.group(2), m.group(3))
    return out


def _extract_metadata(st: _State, meta: LawMetadata) -> dict[str, Any]:
    detected = _detect_source_metadata(st)
    out = meta.law_node
    for field_name in ("title", "issuer", "issued_date", "effective_date", "session", "full_id"):
        if not out.get(field_name) and detected.get(field_name):
            out[field_name] = detected[field_name]
    if detected.get("full_id") and detected["full_id"] != out.get("full_id"):
        out["detected_full_id"] = detected["full_id"]
    return out


def parse_docx(
    path: Path,
    law_code: str = "41/2024/QH15",
    metadata: LawMetadata | None = None,
) -> dict:
    meta = _resolve_metadata(law_code, metadata)
    doc = open_document(path)
    law = meta.id
    st = _State()
    st.allow_no_chapter = meta.allow_no_chapter
    st.synth_chapter_title = meta.canonical_title

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
                        "law_code": law,
                        "article_id": st.current_article["id"],
                        "rows": rows,
                    }
                )
            else:
                st.postamble_tables.append(rows)
            continue

        text = block.text.strip()
        if text:
            _handle_paragraph(text, st, law)

    _finalize_title(st)

    for ch in st.chapters:
        for art in ch["articles"]:
            art["text"] = _build_article_full_text(art)

    return {
        "law": _extract_metadata(st, meta),
        "preamble": st.preamble,
        "postamble": st.postamble,
        "chapters": st.chapters,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(
    result: dict,
    expect_chapters: int | None = None,
    expect_sections: int | None = None,
    expect_articles: int | None = None,
) -> None:
    chapters = result["chapters"]
    law_id = result.get("law", {}).get("id", "UNKNOWN")

    if expect_chapters is not None:
        assert len(chapters) == expect_chapters, (
            f"{law_id}: số Chương sai: mong đợi {expect_chapters}, thực tế {len(chapters)}"
        )

    n_sec = sum(len(ch["sections"]) for ch in chapters)
    if expect_sections is not None:
        assert n_sec == expect_sections, (
            f"{law_id}: số Mục sai: mong đợi {expect_sections}, thực tế {n_sec}"
        )

    n_art = sum(len(ch["articles"]) for ch in chapters)
    if expect_articles is not None:
        assert n_art == expect_articles, (
            f"{law_id}: số Điều sai: mong đợi {expect_articles}, thực tế {n_art}"
        )

    seen_ids: set[str] = set()
    article_numbers: list[int] = []
    for ch in chapters:
        assert ch["title"], f"{law_id}: Chương {ch['roman']} thiếu title"
        assert ch["id"] not in seen_ids, f"Trùng ID: {ch['id']}"
        seen_ids.add(ch["id"])
        for sec in ch["sections"]:
            assert sec["title"], f"{law_id}: Mục {sec['id']} thiếu title"
            assert sec["id"] not in seen_ids, f"Trùng ID: {sec['id']}"
            seen_ids.add(sec["id"])
        for art in ch["articles"]:
            assert art["id"] not in seen_ids, f"Trùng ID: {art['id']}"
            seen_ids.add(art["id"])
            article_numbers.append(art["number"])
            assert art["title"], f"{law_id}: Điều {art['number']} thiếu title"
            assert art["text"].strip(), f"{law_id}: Điều {art['number']} text rỗng"
            for cl in art["clauses"]:
                assert cl["id"] not in seen_ids, f"Trùng ID: {cl['id']}"
                seen_ids.add(cl["id"])
                assert cl["text"].strip(), f"{cl['id']} text rỗng"
                for pt in cl["points"]:
                    assert pt["id"] not in seen_ids, f"Trùng ID: {pt['id']}"
                    seen_ids.add(pt["id"])
                    assert pt["text"].strip(), f"{pt['id']} text rỗng"

    if article_numbers:
        expected_end = expect_articles or max(article_numbers)
        assert article_numbers == list(range(1, expected_end + 1)), (
            f"{law_id}: Article number không liên tiếp 1..{expected_end}"
        )


def validate_against_metadata(result: dict, meta: LawMetadata) -> None:
    validate(
        result,
        expect_chapters=meta.expected_chapters,
        expect_sections=meta.expected_sections,
        expect_articles=meta.expected_articles,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _stats(result: dict) -> dict[str, int]:
    return {
        "chapters": len(result["chapters"]),
        "sections": sum(len(ch["sections"]) for ch in result["chapters"]),
        "articles": sum(len(ch["articles"]) for ch in result["chapters"]),
        "clauses": sum(len(art["clauses"]) for ch in result["chapters"] for art in ch["articles"]),
        "points": sum(
            len(cl["points"])
            for ch in result["chapters"]
            for art in ch["articles"]
            for cl in art["clauses"]
        ),
        "tables": sum(
            len(art.get("tables", [])) for ch in result["chapters"] for art in ch["articles"]
        ),
    }


def _write_result(result: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_one(meta: LawMetadata, source: Path | None = None, out: Path | None = None) -> Path:
    source = source or meta.source_file
    out = out or Path(f"data/graph/interim/structured_law_{meta.id}.json")
    print(f"Đọc {source} cho {meta.id} ...")
    result = parse_docx(source, metadata=meta)
    validate_against_metadata(result, meta)
    _write_result(result, out)
    stats = _stats(result)
    print(
        "  OK "
        f"chapters={stats['chapters']} sections={stats['sections']} "
        f"articles={stats['articles']} clauses={stats['clauses']} points={stats['points']}"
    )
    print(f"  Saved: {out} ({out.stat().st_size / 1024:.1f} KB)")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse Vietnamese legal Word documents.")
    parser.add_argument("--metadata", default=str(Path("data/legal_metadata.yaml")))
    parser.add_argument("--law", default=None, help="Canonical law id/code/alias to parse.")
    parser.add_argument("--source", default=None, help="Override source file for --law.")
    parser.add_argument("--out", default=None, help="Override output JSON for --law.")
    parser.add_argument("--all", action="store_true", help="Parse all laws in metadata load_order.")
    parser.add_argument(
        "--write-legacy-l41",
        action="store_true",
        help="Also write data/graph/interim/structured_law.json when parsing L41_2024.",
    )
    args = parser.parse_args()

    laws = load_law_metadata(args.metadata)
    selected = load_order(args.metadata) if args.all else [args.law or "L41_2024"]

    for law_key in selected:
        try:
            meta = metadata_for(law_key, laws)
        except KeyError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        source = Path(args.source) if args.source else None
        out = Path(args.out) if args.out else None
        try:
            written = parse_one(meta, source=source, out=out)
            if args.write_legacy_l41 and meta.id == "L41_2024":
                legacy_out = Path("data/graph/interim/structured_law.json")
                shutil.copyfile(written, legacy_out)
                print(f"  Legacy copy: {legacy_out}")
        except Exception as exc:
            print(f"FAIL {meta.id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
