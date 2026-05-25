"""Test cho src/rule_extract.py — đảm bảo extraction không bịa.

Trụ cột:
- Mọi span PHẢI khớp byte-for-byte với text gốc tại char_offset.
- Mọi dst của internal ref PHẢI tồn tại trong structured_law.json.
- Spot-check các viện dẫn nổi tiếng (Bộ luật Lao động, Luật 58/2014, v.v.).
"""

import json

import pytest

from src import rule_extract


@pytest.fixture(scope="module")
def structured() -> dict:
    with open("data/interim/structured_law.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def extracted(structured) -> dict:
    return rule_extract.extract_all(structured)


@pytest.fixture(scope="module")
def unit_text(structured) -> dict[str, str]:
    """Map unit_id (Clause.id, Point.id, Article.id-cho-lead) → text gốc."""
    out: dict[str, str] = {}
    for ch in structured["chapters"]:
        for art in ch["articles"]:
            if art["lead_text"]:
                out[art["id"]] = art["lead_text"]
            for cl in art["clauses"]:
                out[cl["id"]] = cl["text"]
                for pt in cl["points"]:
                    out[pt["id"]] = pt["text"]
    return out


@pytest.fixture(scope="module")
def all_structural_ids(structured) -> set[str]:
    ids: set[str] = set()
    for ch in structured["chapters"]:
        ids.add(ch["id"])
        for sec in ch["sections"]:
            ids.add(sec["id"])
        for art in ch["articles"]:
            ids.add(art["id"])
            for cl in art["clauses"]:
                ids.add(cl["id"])
                for pt in cl["points"]:
                    ids.add(pt["id"])
    return ids


# ---------- TRỤ CỘT 1: provenance byte-for-byte ----------


def test_internal_refs_byte_for_byte(extracted, unit_text):
    for r in extracted["internal_refs"]:
        text = unit_text[r["src"]]
        actual = text[r["char_offset"] : r["char_offset"] + len(r["span"])]
        assert actual == r["span"], (
            f"src={r['src']} offset={r['char_offset']}\n"
            f"  expected: {r['span']!r}\n"
            f"  actual  : {actual!r}"
        )


def test_external_refs_byte_for_byte(extracted, unit_text):
    for r in extracted["external_refs"]:
        text = unit_text[r["src"]]
        actual = text[r["char_offset"] : r["char_offset"] + len(r["span"])]
        assert actual == r["span"], (
            f"src={r['src']} offset={r['char_offset']}\n"
            f"  expected: {r['span']!r}\n"
            f"  actual  : {actual!r}"
        )


def test_definitions_byte_for_byte(extracted, unit_text):
    for d in extracted["definitions"]:
        text = unit_text[d["defined_in"]]
        assert d["span"] == text, f"Định nghĩa {d['concept_id']} không khớp clause text gốc"
        assert d["term"] in text, f"Term {d['term']!r} không có trong text"


# ---------- TRỤ CỘT 2: dst phải tồn tại ----------


def test_internal_dst_phai_ton_tai(extracted, all_structural_ids):
    """Mọi dst của internal ref phải là 1 structural ID có thực."""
    invalid = [r for r in extracted["internal_refs"] if r["dst"] not in all_structural_ids]
    assert not invalid, (
        f"Có {len(invalid)} internal refs trỏ tới dst không tồn tại. "
        f"Mẫu đầu: {invalid[0] if invalid else None}"
    )


def test_definitions_defined_in_phai_la_clause_dieu_3(extracted):
    for d in extracted["definitions"]:
        assert d["defined_in"].startswith(
            "L41_2024.A3.K"
        ), f"Định nghĩa {d['concept_id']} có defined_in sai: {d['defined_in']}"


# ---------- TRỤ CỘT 3: spot-check viện dẫn nổi tiếng ----------


def test_co_ref_toi_bo_luat_lao_dong(extracted):
    """Bộ luật Lao động được nhắc nhiều lần trong luật BHXH."""
    bllao = [r for r in extracted["external_refs"] if "Lao động" in (r["external_title"] or "")]
    assert len(bllao) >= 10, f"Mong đợi >=10 refs tới Bộ luật Lao động, thực tế {len(bllao)}"


def test_co_ref_toi_luat_58_2014(extracted):
    """Luật 58/2014 (luật BHXH cũ) được nhắc trong Điều 140, 141."""
    refs = [r for r in extracted["external_refs"] if r["external_code"] == "58/2014/QH13"]
    assert len(refs) >= 3
    # Các ref đó phải nằm trong Điều 140 hoặc 141
    art_nums = {int(r["src"].split(".")[1][1:]) for r in refs}
    assert art_nums.issubset(
        {140, 141}
    ), f"Luật 58/2014 phải chỉ xuất hiện ở Điều 140/141, thực tế: {art_nums}"


def test_dieu_3_co_du_dinh_nghia(extracted):
    """Điều 3 có 12 khoản → kỳ vọng ≥10 định nghĩa được trích."""
    assert len(extracted["definitions"]) >= 10


def test_dinh_nghia_bao_hiem_xa_hoi(extracted):
    """Định nghĩa BHXH ở Điều 3 K1."""
    d = next(
        (d for d in extracted["definitions"] if d["term"].lower() == "bảo hiểm xã hội"),
        None,
    )
    assert d is not None
    assert d["defined_in"] == "L41_2024.A3.K1"
    assert "sự bảo đảm" in d["definition"]


# ---------- TRỤ CỘT 4: amendments ----------


def test_amendments_dien_ra_o_dieu_139_140(extracted):
    for a in extracted["amendments"]:
        art_n = int(a["src"].split(".")[1][1:])
        assert art_n in (139, 140), f"Amendment phải ở Điều 139/140, thực tế: {a['src']}"


def test_co_amends_luat_84_2015(extracted):
    """Điều 139 K1 sửa đổi Luật ATVSLĐ 84/2015."""
    a = next(
        (
            a
            for a in extracted["amendments"]
            if a["action"] == "AMENDS" and a["external_code"] == "84/2015/QH13"
        ),
        None,
    )
    assert a is not None


def test_co_repeals_luat_39_2009(extracted):
    """Điều 139 K3 bãi bỏ khoản 2 Điều 17 Luật Người cao tuổi 39/2009."""
    a = next(
        (
            a
            for a in extracted["amendments"]
            if a["action"] == "REPEALS" and a["external_code"] == "39/2009/QH12"
        ),
        None,
    )
    assert a is not None
    assert a["external_article"] == 17
    assert a["external_clause"] == 2


# ---------- TRỤ CỘT 5: cấu trúc src/source_clause ----------


def test_source_clause_la_id_clause_hoac_article(extracted):
    """source_clause phải là Clause.id (Lxx.Aa.Kb) hoặc Article.id (Lxx.Aa)
    cho trường hợp lead_text. Không được là Point.id."""
    import re

    pat = re.compile(r"^L\d+_\d{4}\.A\d+(\.K\d+)?$")
    for r in extracted["internal_refs"] + extracted["external_refs"]:
        assert pat.match(r["source_clause"]), f"source_clause sai format: {r['source_clause']}"


def test_self_ref_dieu_nay_tro_ve_article_cha(extracted):
    """'Điều này' phải trỏ về Article.id của nơi xuất hiện."""
    self_refs = [
        r for r in extracted["internal_refs"] if r.get("is_self") and "Điều này" in r["span"]
    ]
    assert len(self_refs) > 0
    for r in self_refs[:20]:
        # src dạng L41_2024.AX.KY hoặc L41_2024.AX.KY.z → article = AX
        src_parts = r["src"].split(".")
        expected_article = f"{src_parts[0]}.{src_parts[1]}"
        assert r["dst"] == expected_article, (
            f"Điều này tại {r['src']} phải trỏ về {expected_article}, " f"thực tế {r['dst']}"
        )


# ---------- Provenance roundtrip kết hợp ids.parse_id ----------


def test_parse_id_truy_nguoc_tu_dst(extracted):
    """Lấy 1 ref đến điểm cụ thể, parse dst ngược về (article, clause, point)."""
    from src import ids

    pt_refs = [r for r in extracted["internal_refs"] if r["kind"] == "REFERENCES_POINT"]
    assert len(pt_refs) > 0
    r = pt_refs[0]
    parsed = ids.parse_id(r["dst"])
    assert parsed["article"] is not None
    assert parsed["clause"] is not None
    assert parsed["point"] is not None
