"""Quy ước sinh ID — single source of truth cho mọi bước trong pipeline.

Mọi node/edge trong Knowledge Graph phải dùng các hàm tại đây để sinh ID.
Định dạng cố định để mọi ID đều parse ngược được về (luật, chương, điều, khoản, điểm),
phục vụ yêu cầu provenance: từ bất kỳ node/edge nào cũng truy ra điều luật gốc.
"""

from __future__ import annotations

import re
import unicodedata

LAW_ID_DEFAULT = "L41_2024"


def slug(text: str) -> str:
    """Chuyển tiếng Việt có dấu sang kebab-case không dấu, ASCII-safe.

    Dùng cho ID của semantic node (Subject, Benefit, Concept, ...).
    Phải replace 'đ'/'Đ' trước normalize vì NFKD không tách dấu của đ.
    """
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return ascii_text


# ----- Structural IDs (1-1 với văn bản gốc) -----


def law_id(code: str = "41/2024/QH15") -> str:
    """Return canonical law ID.

    Chấp nhận 2 dạng input:

    1. **Canonical** — bất kỳ ID đã đăng ký trong ``data/legal_metadata.yaml``
       / ``data/legal_sources.yaml`` (Luật ``L41_2024``, Nghị định
       ``ND143_2018``, Quyết định ``QD366_BHXH``, Thông tư
       ``TT18_2022_BYT``, …). Trả lại nguyên ID.
    2. **Mã luật QH** dạng ``XX/YYYY/QH<n>`` (Luật/Bộ luật do Quốc hội ban
       hành) → convert sang canonical ``L<XX>_<YYYY>``.

    Mã NĐ/QĐ/TT/CV/… không có rule chung từ official code sang canonical
    (vd ``366/QĐ-BHXH`` không có năm cố định ở vị trí 2 nên không thể derive
    slug); caller phải truyền canonical ID đã đăng ký YAML.
    """
    code = code or ""
    # 1. Đã canonical — pass-through (giữ chính xác slug user đã đặt YAML)
    if re.match(r"^[A-Z][A-Z0-9_]*$", code):
        return code
    # 2. Mã QH XX/YYYY/QH<n> — convert được vì luôn có năm ở vị trí 2
    m = re.match(r"^(\d+)/(\d{4})/QH\d+$", code)
    if m:
        return f"L{m.group(1)}_{m.group(2)}"
    raise ValueError(f"Mã luật không hợp lệ: {code}")


def chapter_id(law: str, n: int) -> str:
    return f"{law}.C{n}"


def section_id(law: str, chapter_n: int, section_n: int) -> str:
    """Mục — đánh số lại từ 1 trong mỗi Chương."""
    return f"{law}.C{chapter_n}.M{section_n}"


def article_id(law: str, n: int) -> str:
    return f"{law}.A{n}"


def clause_id(law: str, article_n: int, clause_n: int) -> str:
    return f"{law}.A{article_n}.K{clause_n}"


def point_id(law: str, article_n: int, clause_n: int, letter: str) -> str:
    return f"{law}.A{article_n}.K{clause_n}.{letter.lower()}"


def table_id(law: str, article_n: int, idx: int) -> str:
    return f"{law}.A{article_n}.T{idx}"


# ----- Semantic IDs (đa tham chiếu) -----


def concept_id(term: str) -> str:
    return f"concept:{slug(term)}"


def subject_id(name: str) -> str:
    return f"subject:{slug(name)}"


def benefit_id(name: str) -> str:
    return f"benefit:{slug(name)}"


def organization_id(name: str) -> str:
    return f"org:{slug(name)}"


def role_id(name: str) -> str:
    return f"role:{slug(name)}"


def condition_id(text: str) -> str:
    # Điều kiện có thể dài → hash 12 ký tự đầu của slug
    return f"cond:{slug(text)[:80]}"


def obligation_id(text: str) -> str:
    return f"oblig:{slug(text)[:80]}"


def right_id(text: str) -> str:
    return f"right:{slug(text)[:80]}"


def prohibited_act_id(text: str) -> str:
    return f"prohib:{slug(text)[:80]}"


def fund_id(name: str) -> str:
    return f"fund:{slug(name)}"


def external_law_id(code: str) -> str:
    """ExternalLaw: dùng mã luật đầy đủ làm ID (vd 'ext:58/2014/QH13')."""
    return f"ext:{code.strip()}"


# ----- Parser ngược (provenance) -----

# Prefix luật chấp nhận mọi key canonical trong data/legal_sources.yaml:
# Luật QH (L41_2024), Nghị định (ND143_2018), Quyết định (QD366_BHXH),
# Thông tư (TT18_2022_BYT), Hiệp định (HIEPDINH_VN_KR_BHXH), Pháp lệnh
# (PHAPLENH_NCC), Bộ luật khác (BLDS_2015), … — đồng bộ shape với
# `citations._INTERNAL_ID_RE`. Semantic validity (có phải source đăng ký
# trong registry không) check ở tầng citations.py qua registry.
_ID_PATTERN = re.compile(
    r"^(?P<law>[A-Z][A-Z0-9_]*)"
    r"(?:\.C(?P<chapter>\d+))?"
    r"(?:\.A(?P<article>\d+))?"
    r"(?:\.K(?P<clause>\d+))?"
    r"(?:\.(?P<point>[a-zđ]))?"
    r"(?:\.T(?P<table>\d+))?$"
)


def parse_id(node_id: str) -> dict:
    """Tách ngược một structural ID về (law, chapter, article, clause, point).

    Dùng để truy nguồn (provenance) một node/edge bất kỳ về điều luật gốc.

    >>> parse_id('L41_2024.A64.K1.a')
    {'law': 'L41_2024', 'chapter': None, 'article': 64, 'clause': 1, 'point': 'a', 'table': None}
    """
    m = _ID_PATTERN.match(node_id)
    if not m:
        raise ValueError(f"Không phải structural ID hợp lệ: {node_id}")
    g = m.groupdict()
    return {
        "law": g["law"],
        "chapter": int(g["chapter"]) if g["chapter"] else None,
        "article": int(g["article"]) if g["article"] else None,
        "clause": int(g["clause"]) if g["clause"] else None,
        "point": g["point"],
        "table": int(g["table"]) if g["table"] else None,
    }


def citation_label(node_id: str) -> str:
    """Sinh nhãn citation tiếng Việt từ ID, ví dụ 'Điều 64 khoản 1 điểm a'.

    Dùng cho RAG để in citation trong câu trả lời.
    """
    p = parse_id(node_id)
    parts = []
    if p["article"]:
        parts.append(f"Điều {p['article']}")
    if p["clause"]:
        parts.append(f"khoản {p['clause']}")
    if p["point"]:
        parts.append(f"điểm {p['point']}")
    return " ".join(parts) if parts else node_id
