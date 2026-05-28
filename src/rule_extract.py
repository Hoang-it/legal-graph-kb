"""B2 — Trích viện dẫn / sửa đổi / định nghĩa bằng RULE (regex deterministic).

Đầu vào : data/interim/structured_law.json  (output của B1)
Đầu ra  :
    data/interim/internal_refs.json
    data/interim/external_refs.json
    data/interim/definitions.json
    data/interim/amendments.json

NGUYÊN TẮC PROVENANCE:
    Mọi extraction PHẢI có (source_clause, char_offset, span). Việc xác
    minh: text[char_offset : char_offset + len(span)] == span. Pipeline
    sẽ FAIL nếu vi phạm — không cho phép bịa.

Không gọi LLM. Không suy diễn. Chỉ regex.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from src import ids
from src.legal_metadata import fold_text, load_law_metadata, load_order, resolve_law_id

# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------

# External law: "Luật/Bộ luật/Nghị định/Nghị quyết [tên?] số XX/YYYY/CODE"
# Tên có thể chứa dấu phẩy (vd "Luật An toàn, vệ sinh lao động"), nhưng không
# kết thúc bằng "số" → dùng non-greedy + boundary " số "
_CODE = r"\d+/\d{4}/(?:QH\d+|NĐ-CP|NQ-CP|TT-[A-Z]+|CP|TTg)"
# Tên: tối đa 80 ký tự, ký tự cho phép = chữ + space + comma
_NAME = r"(?:[\wÀ-ỹĐđ][\wÀ-ỹĐđ,\s]{0,80}?)"
RE_EXT_TYPED = re.compile(
    r"(?P<doc>Luật|Bộ\s+luật|Nghị\s+định|Nghị\s+quyết)"
    r"(?:\s+(?P<name>" + _NAME + r"))?"
    r"\s+số\s+(?P<code>" + _CODE + r")"
)
# Bộ luật Lao động (không có số) — đặc biệt
RE_BO_LUAT_NAMED = re.compile(
    r"Bộ\s+luật\s+(Lao\s+động|Dân\s+sự|Hình\s+sự|Tố\s+tụng\s+[A-Za-zÀ-ỹ\s]+?)(?=\b|[,;.])"
)

# External với specific article: "khoản N (Điều M)? của Luật/Bộ luật ..."
RE_EXT_WITH_ART = re.compile(
    r"(?:(?:điểm\s+([a-zđ])(?:\s*(?:,\s*|\s+và\s+)\s*điểm\s+[a-zđ])*\s+)?"
    r"khoản\s+(\d+)\s+)?"
    r"Điều\s+(\d+)\s+của\s+"
    r"(?:(Luật|Bộ\s+luật|Nghị\s+định|Nghị\s+quyết)"
    r"(?:\s+" + _NAME + r")?\s+số\s+(" + _CODE + r"))"
)
# External với specific article, KHÔNG có số (vd "Điều 169 của Bộ luật Lao động")
RE_EXT_WITH_ART_NAMED = re.compile(
    r"(?:(?:điểm\s+([a-zđ])(?:\s*(?:,\s*|\s+và\s+)\s*điểm\s+[a-zđ])*\s+)?"
    r"khoản\s+(\d+)\s+)?"
    r"Điều\s+(\d+)\s+của\s+"
    r"(Bộ\s+luật\s+(?:Lao\s+động|Dân\s+sự|Hình\s+sự|Tố\s+tụng\s+[A-Za-zÀ-ỹ\s]+?))"
    r"(?=[\s,;.])"
)

# Internal patterns — chỉ match khi KHÔNG có "của Luật/Bộ luật" theo sau
# (để tránh nhận nhầm external)
RE_INT_POINT_CLAUSE_ART = re.compile(
    r"(?:các\s+|những\s+)?"
    r"điểm\s+([a-zđ])"
    r"(?:(?:\s*,\s*|\s+và\s+)(?:điểm\s+)?([a-zđ]))*"
    r"\s+khoản\s+(\d+)"
    r"\s+Điều\s+(\d+)"
    r"(?:\s+(?:của\s+)?Luật\s+này)?"
    r"(?!\s+của\s+(?:Luật|Bộ\s+luật|Nghị))"
)
RE_INT_CLAUSE_ART = re.compile(
    r"(?:các\s+|những\s+)?"
    r"khoản\s+(\d+)"
    r"(?:(?:\s*,\s*|\s+và\s+)(?:khoản\s+)?(\d+))*"
    r"\s+Điều\s+(\d+)"
    r"(?:\s+(?:của\s+)?Luật\s+này)?"
    r"(?!\s+của\s+(?:Luật|Bộ\s+luật|Nghị))"
)
RE_INT_ART = re.compile(
    r"(?:các\s+|những\s+)?"
    r"Điều\s+(\d+)"
    r"(?:(?:\s*,\s*|\s+và\s+)Điều\s+(\d+))*"
    r"(?:\s+(?:của\s+)?Luật\s+này)?"
    r"(?!\s+của\s+(?:Luật|Bộ\s+luật|Nghị))"
)

# Self-refs
RE_SELF_ART = re.compile(r"Điều\s+này")
RE_SELF_CLAUSE = re.compile(r"khoản\s+này")

# Amendments
RE_AMEND = re.compile(r"Sửa\s+đổi(?:,\s*bổ\s+sung)?")
RE_REPEAL = re.compile(r"Bãi\s+bỏ")

# Định nghĩa Điều 3: "Term là definition"
RE_DEF = re.compile(r"^(.+?)\s+là\s+(.+)$", re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _Span:
    start: int
    end: int


def _overlap(s: int, e: int, used: list[_Span]) -> bool:
    return any(not (e <= u.start or s >= u.end) for u in used)


def _iter_text_units(structured: dict) -> Iterable[tuple[str, str, str, int, str | None]]:
    """Yield (unit_id, source_clause_id, text, article_n, parent_letter).

    unit_id = Clause.id hoặc Point.id (nơi xuất hiện ref).
    source_clause_id = Clause.id chứa unit_id (= unit_id nếu là Clause).
    """
    for ch in structured["chapters"]:
        for art in ch["articles"]:
            art_n = art["number"]
            # Lead text → coi như Article-level reference (source_clause = clause đầu tiên nếu có,
            # nếu không có clause thì dùng article_id)
            if art["lead_text"]:
                # Lead text không có clause cha → source_clause = article_id (sẽ filter ở B6)
                yield art["id"], art["id"], art["lead_text"], art_n, None
            for cl in art["clauses"]:
                yield cl["id"], cl["id"], cl["text"], art_n, None
                for pt in cl["points"]:
                    yield pt["id"], cl["id"], pt["text"], art_n, pt["letter"]


# ---------------------------------------------------------------------------
# External refs
# ---------------------------------------------------------------------------


def extract_external_refs(
    text: str, source_id: str, source_clause: str
) -> tuple[list[dict], list[_Span]]:
    """Trả về (refs, used_spans). used_spans để skip ở pass internal."""
    refs: list[dict] = []
    used: list[_Span] = []

    # 1. Pattern có specific article + số: "Điều X của Luật Y số Z"
    # Groups: 1=letter 2=clause_n 3=article_n 4=doc_type 5=code
    for m in RE_EXT_WITH_ART.finditer(text):
        span = m.group(0)
        external_code = m.group(5)
        doc_type = m.group(4).strip()
        ext_id = ids.external_law_id(external_code)
        ref = {
            "src": source_id,
            "source_clause": source_clause,
            "external_code": external_code,
            "external_title": _normalize_doc_title(doc_type, external_code),
            "external_article": int(m.group(3)) if m.group(3) else None,
            "external_clause": int(m.group(2)) if m.group(2) else None,
            "external_point": m.group(1),
            "dst": ext_id,
            "kind": "CITES_EXTERNAL",
            "span": span,
            "char_offset": m.start(),
        }
        refs.append(ref)
        used.append(_Span(m.start(), m.end()))

    # 2. Pattern có specific article, không số: "Điều X của Bộ luật Lao động"
    for m in RE_EXT_WITH_ART_NAMED.finditer(text):
        if _overlap(m.start(), m.end(), used):
            continue
        span = m.group(0)
        title = m.group(4).strip()
        # Không có code → dùng title làm id (vd "ext:Bộ luật Lao động")
        ext_id = f"ext:{title}"
        ref = {
            "src": source_id,
            "source_clause": source_clause,
            "external_code": None,
            "external_title": title,
            "external_article": int(m.group(3)),
            "external_clause": int(m.group(2)) if m.group(2) else None,
            "external_point": m.group(1),
            "dst": ext_id,
            "kind": "CITES_EXTERNAL",
            "span": span,
            "char_offset": m.start(),
        }
        refs.append(ref)
        used.append(_Span(m.start(), m.end()))

    # 3. Pattern toàn luật có số: "Luật Y số Z" (không có Điều cụ thể)
    for m in RE_EXT_TYPED.finditer(text):
        if _overlap(m.start(), m.end(), used):
            continue
        span = m.group(0)
        code = m.group("code")
        doc_type = m.group("doc").strip()
        name_part = (m.group("name") or "").strip()
        title = f"{doc_type} {name_part}".strip() if name_part else doc_type
        ext_id = ids.external_law_id(code)
        refs.append(
            {
                "src": source_id,
                "source_clause": source_clause,
                "external_code": code,
                "external_title": title,
                "external_article": None,
                "external_clause": None,
                "external_point": None,
                "dst": ext_id,
                "kind": "CITES_EXTERNAL",
                "span": span,
                "char_offset": m.start(),
            }
        )
        used.append(_Span(m.start(), m.end()))

    # 4. Pattern Bộ luật named (no number, no article)
    for m in RE_BO_LUAT_NAMED.finditer(text):
        if _overlap(m.start(), m.end(), used):
            continue
        span = m.group(0)
        title = span.strip()
        ext_id = f"ext:{title}"
        refs.append(
            {
                "src": source_id,
                "source_clause": source_clause,
                "external_code": None,
                "external_title": title,
                "external_article": None,
                "external_clause": None,
                "external_point": None,
                "dst": ext_id,
                "kind": "CITES_EXTERNAL",
                "span": span,
                "char_offset": m.start(),
            }
        )
        used.append(_Span(m.start(), m.end()))

    return refs, used


def _normalize_doc_title(doc_type: str, code: str) -> str:
    """Phỏng đoán title từ doc_type (có thể chỉ là 'Luật'). Không bịa — chỉ
    trả về doc_type. Title đầy đủ sẽ được làm giàu ở B4 khi tổng hợp."""
    return doc_type.strip()


# ---------------------------------------------------------------------------
# Internal refs
# ---------------------------------------------------------------------------


def extract_internal_refs(
    text: str,
    source_id: str,
    source_clause: str,
    article_n: int,
    used: list[_Span],
    law: str,
    max_article: int,
) -> list[dict]:
    refs: list[dict] = []

    def _emit(dst: str, kind: str, span: str, start: int):
        refs.append(
            {
                "src": source_id,
                "source_clause": source_clause,
                "dst": dst,
                "kind": kind,
                "span": span,
                "char_offset": start,
                "is_self": False,
            }
        )

    # 1. điểm + khoản + Điều
    for m in RE_INT_POINT_CLAUSE_ART.finditer(text):
        if _overlap(m.start(), m.end(), used):
            continue
        span = m.group(0)
        clause_n = int(m.group(3))
        target_art_n = int(m.group(4))
        # Lấy TẤT CẢ letters trong span
        letters = re.findall(r"điểm\s+([a-zđ])", span)
        if not letters:
            letters = [m.group(1)]
        for letter in letters:
            _emit(
                ids.point_id(law, target_art_n, clause_n, letter),
                "REFERENCES_POINT",
                span,
                m.start(),
            )
        used.append(_Span(m.start(), m.end()))

    # 2. khoản + Điều
    for m in RE_INT_CLAUSE_ART.finditer(text):
        if _overlap(m.start(), m.end(), used):
            continue
        span = m.group(0)
        target_art_n = int(m.group(3))
        clause_nums = [int(c) for c in re.findall(r"khoản\s+(\d+)", span)]
        if not clause_nums:
            clause_nums = [int(m.group(1))]
        for cn in clause_nums:
            _emit(
                ids.clause_id(law, target_art_n, cn),
                "REFERENCES_CLAUSE",
                span,
                m.start(),
            )
        used.append(_Span(m.start(), m.end()))

    # 3. Điều đơn lẻ
    for m in RE_INT_ART.finditer(text):
        if _overlap(m.start(), m.end(), used):
            continue
        span = m.group(0)
        art_nums = [int(n) for n in re.findall(r"Điều\s+(\d+)", span)]
        for an in art_nums:
            # CHỈ giữ ref nếu Điều thuộc luật này (1..max_article). Nếu lớn hơn,
            # khả năng là ref tới luật khác mà regex không catch — bỏ qua
            # để không bịa target.
            if 1 <= an <= max_article:
                _emit(
                    ids.article_id(law, an),
                    "REFERENCES_ARTICLE",
                    span,
                    m.start(),
                )
        used.append(_Span(m.start(), m.end()))

    # 4. Self-refs
    for m in RE_SELF_ART.finditer(text):
        if _overlap(m.start(), m.end(), used):
            continue
        refs.append(
            {
                "src": source_id,
                "source_clause": source_clause,
                "dst": ids.article_id(law, article_n),
                "kind": "REFERENCES_ARTICLE",
                "span": m.group(0),
                "char_offset": m.start(),
                "is_self": True,
            }
        )
        used.append(_Span(m.start(), m.end()))

    for m in RE_SELF_CLAUSE.finditer(text):
        if _overlap(m.start(), m.end(), used):
            continue
        # "khoản này" → clause cha. Nếu source_id là Point, lấy clause_id;
        # nếu source_id là Clause, target = chính nó.
        refs.append(
            {
                "src": source_id,
                "source_clause": source_clause,
                "dst": source_clause,
                "kind": "REFERENCES_CLAUSE",
                "span": m.group(0),
                "char_offset": m.start(),
                "is_self": True,
            }
        )
        used.append(_Span(m.start(), m.end()))

    return refs


# ---------------------------------------------------------------------------
# Definitions (Điều 3)
# ---------------------------------------------------------------------------


def extract_definitions(structured: dict) -> list[dict]:
    art3 = next(
        (a for ch in structured["chapters"] for a in ch["articles"] if a["number"] == 3),
        None,
    )
    if art3 is None:
        return []

    defs: list[dict] = []
    for cl in art3["clauses"]:
        text = cl["text"]
        # Pattern: "<Term> là <definition>"
        m = RE_DEF.match(text)
        if not m:
            continue
        term = m.group(1).strip()
        definition = m.group(2).strip()
        # Loại bỏ trường hợp "Term" quá dài (>120) hoặc chứa chấm — coi là phi-định-nghĩa
        if len(term) > 120 or "." in term:
            continue
        defs.append(
            {
                "concept_id": ids.concept_id(term),
                "term": term,
                "definition": definition,
                "defined_in": cl["id"],
                "char_offset": 0,
                "span": text,
            }
        )
    return defs


# ---------------------------------------------------------------------------
# Amendments / Repeals (Điều 139, 140)
# ---------------------------------------------------------------------------


def extract_amendments(structured: dict, external_refs: list[dict]) -> list[dict]:
    """Find amendment/repeal/replacement clauses and link mentioned external laws."""
    amendments: list[dict] = []
    # Map source_clause → external refs trong clause đó
    by_clause: dict[str, list[dict]] = {}
    for r in external_refs:
        by_clause.setdefault(r["source_clause"], []).append(r)

    for ch in structured["chapters"]:
        for art in ch["articles"]:
            for cl in art["clauses"]:
                text = cl["text"]
                amend = bool(RE_AMEND.search(text))
                repeal = bool(RE_REPEAL.search(text))
                replaces = "hết hiệu lực" in text.lower()
                if not (amend or repeal or replaces):
                    continue
                action = "AMENDS" if amend else "REPEALS" if repeal else "REPLACES"
                for ext in by_clause.get(cl["id"], []):
                    amendments.append(
                        {
                            "src": cl["id"],
                            "source_clause": cl["id"],
                            "action": action,
                            "external_code": ext["external_code"],
                            "external_title": ext["external_title"],
                            "external_article": ext["external_article"],
                            "external_clause": ext["external_clause"],
                            "external_point": ext["external_point"],
                            "dst": ext["dst"],
                            "span_action": next(
                                (
                                    m.group(0)
                                    for m in [
                                        RE_AMEND.search(text) if amend else None,
                                        RE_REPEAL.search(text) if repeal else None,
                                    ]
                                    if m
                                ),
                                "REPLACES",
                            ),
                            "char_offset_action": (
                                (RE_AMEND.search(text) or RE_REPEAL.search(text)).start()
                                if (amend or repeal)
                                else 0
                            ),
                        }
                    )
    return amendments


# ---------------------------------------------------------------------------
# Main extraction + verification
# ---------------------------------------------------------------------------

def amendment_article_numbers(structured: dict) -> set[int]:
    """Articles that amend other laws should not emit internal refs from quoted text."""
    out: set[int] = set()
    for ch in structured["chapters"]:
        for art in ch["articles"]:
            title = (art.get("title") or "").lower()
            text = "\n".join(cl.get("text", "") for cl in art.get("clauses", []))
            folded = f"{title}\n{text}".lower()
            if (
                "sửa đổi" in folded
                or "bổ sung" in folded and "luật" in folded
                or "bãi bỏ" in folded and "luật" in folded
            ):
                out.add(int(art["number"]))
    return out


def _resolve_external_target(ref: dict, source_law: str) -> dict:
    laws = load_law_metadata()
    candidates = [
        ref.get("external_code"),
        ref.get("external_title"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            target_law = resolve_law_id(str(candidate), laws)
        except KeyError:
            continue
        ref["target_law"] = target_law
        if ref.get("external_article"):
            ref["target_article_id"] = ids.article_id(target_law, int(ref["external_article"]))
        if target_law != source_law:
            ref["is_loaded_cross_law"] = True
        return ref

    title_folded = fold_text(str(ref.get("external_title") or ""))
    if "bo luat lao dong" in title_folded and "L45_2019" in laws:
        ref["target_law"] = "L45_2019"
        if ref.get("external_article"):
            ref["target_article_id"] = ids.article_id("L45_2019", int(ref["external_article"]))
        ref["is_loaded_cross_law"] = source_law != "L45_2019"
    return ref


def extract_all(structured: dict, law: str | None = None) -> dict[str, list[dict]]:
    law = law or structured.get("law", {}).get("id") or ids.LAW_ID_DEFAULT
    article_numbers = [
        int(art["number"]) for ch in structured["chapters"] for art in ch["articles"]
    ]
    max_article = max(article_numbers) if article_numbers else 0
    skip_internal = amendment_article_numbers(structured)

    all_internal: list[dict] = []
    all_external: list[dict] = []

    unit_texts: dict[str, str] = {}

    for uid, sclause, text, art_n, _letter in _iter_text_units(structured):
        unit_texts[uid] = text
        ext_refs, used = extract_external_refs(text, uid, sclause)
        all_external.extend(ext_refs)
        if art_n not in skip_internal:
            int_refs = extract_internal_refs(text, uid, sclause, art_n, used, law, max_article)
            all_internal.extend(int_refs)

    all_external = [_resolve_external_target(r, law) for r in all_external]
    defs = extract_definitions(structured)
    amendments = extract_amendments(structured, all_external)

    # ----- Verify ngược: span phải khớp byte-for-byte với text gốc -----
    _verify_provenance(all_internal, unit_texts, "internal_refs")
    _verify_provenance(all_external, unit_texts, "external_refs")

    return {
        "internal_refs": all_internal,
        "external_refs": all_external,
        "definitions": defs,
        "amendments": amendments,
    }


def _verify_provenance(refs: list[dict], unit_texts: dict[str, str], label: str) -> None:
    """Đối chiếu byte-for-byte. FAIL nếu sai để chặn bịa."""
    errors: list[str] = []
    for r in refs:
        text = unit_texts.get(r["src"])
        if text is None:
            errors.append(f"{label}: src {r['src']} không có text")
            continue
        off = r["char_offset"]
        span = r["span"]
        actual = text[off : off + len(span)]
        if actual != span:
            errors.append(
                f"{label}: provenance sai tại {r['src']}@{off}: "
                f"expected={span!r} actual={actual!r}"
            )
            if len(errors) > 10:
                errors.append("... (truncated)")
                break
    if errors:
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        raise AssertionError(
            f"Provenance verification thất bại ({len(errors)} lỗi). " f"Không lưu output."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/interim/structured_law.json")
    p.add_argument("--out-dir", default="data/interim")
    p.add_argument("--law", default=None)
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    if args.all:
        status = 0
        for law_id in load_order():
            path = Path(f"data/interim/structured_law_{law_id}.json")
            if not path.exists():
                print(f"FAIL: không tìm thấy {path}. Chạy B1 trước.", file=sys.stderr)
                return 1
            status = max(status, _run_one(path, Path(args.out_dir), law_id))
        return status

    src = Path(args.input)
    out_dir = Path("data/interim")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    if not src.exists():
        print(f"FAIL: không tìm thấy {src}. Chạy B1 trước.", file=sys.stderr)
        return 1

    return _run_one(src, out_dir, args.law)


def _run_one(src: Path, out_dir: Path, law_override: str | None = None) -> int:
    with src.open(encoding="utf-8") as f:
        structured = json.load(f)

    law = law_override or structured.get("law", {}).get("id") or ids.LAW_ID_DEFAULT
    print(f"Đang trích {law}...")
    out = extract_all(structured, law=law)

    print("\n=== STATS ===")
    print(f"  Internal refs : {len(out['internal_refs']):>5}")
    print(f"  External refs : {len(out['external_refs']):>5}")
    print(f"  Definitions   : {len(out['definitions']):>5}")
    print(f"  Amendments    : {len(out['amendments']):>5}")

    # Breakdown
    from collections import Counter

    int_kinds = Counter(r["kind"] for r in out["internal_refs"])
    self_count = sum(1 for r in out["internal_refs"] if r["is_self"])
    print("\n  Internal breakdown:")
    for k, v in int_kinds.most_common():
        print(f"    {k:<20} {v:>5}")
    print(f"    (trong đó self-ref: {self_count})")

    ext_with_art = sum(1 for r in out["external_refs"] if r["external_article"])
    print(f"\n  External refs có Điều cụ thể: {ext_with_art}")
    ext_codes = Counter(r["external_code"] for r in out["external_refs"] if r["external_code"])
    print("  External codes:")
    for code, n in ext_codes.most_common():
        print(f"    {code:<20} {n:>3}")

    print("\n  Amendments:")
    for a in out["amendments"]:
        ext = f" Điều {a['external_article']}" if a.get("external_article") else " (toàn luật)"
        print(
            f"    [{a['src']}] {a['action']:<8} → {a['external_code'] or a['external_title']}{ext}"
        )

    print(f"\n  Definitions: {len(out['definitions'])} (Điều 3)")
    for d in out["definitions"][:5]:
        print(f"    {d['concept_id']:<45} ← {d['defined_in']}")
    if len(out["definitions"]) > 5:
        print(f"    ... và {len(out['definitions']) - 5} định nghĩa khác")

    print("\n=== VERIFY PROVENANCE ===")
    print("  OK — mọi extraction khớp byte-for-byte với text gốc.")

    suffix = "" if src.name == "structured_law.json" and law == "L41_2024" else f"_{law}"
    for name, items in out.items():
        path = out_dir / f"{name}{suffix}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {path} ({path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
