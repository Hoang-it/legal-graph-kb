"""Test cho src/ids.py — ID convention & provenance round-trip."""

from src import ids

# ---------- slug ----------


def test_slug_tieng_viet_co_dau():
    assert ids.slug("Người lao động") == "nguoi-lao-dong"
    assert ids.slug("Bảo hiểm xã hội") == "bao-hiem-xa-hoi"
    assert ids.slug("Trợ cấp hưu trí") == "tro-cap-huu-tri"


def test_slug_xu_ly_dac_biet():
    assert ids.slug("Quỹ BHXH (bắt buộc)") == "quy-bhxh-bat-buoc"
    assert ids.slug("  Điều 64 — khoản 1  ") == "dieu-64-khoan-1"


# ---------- structural IDs ----------


def test_law_id_tu_code():
    assert ids.law_id("41/2024/QH15") == "L41_2024"
    assert ids.law_id("58/2014/QH13") == "L58_2014"


def test_structural_id_chain():
    law = ids.law_id()
    assert ids.chapter_id(law, 1) == "L41_2024.C1"
    assert ids.article_id(law, 64) == "L41_2024.A64"
    assert ids.clause_id(law, 64, 1) == "L41_2024.A64.K1"
    assert ids.point_id(law, 64, 1, "a") == "L41_2024.A64.K1.a"


# ---------- parse_id (provenance round-trip) ----------


def test_parse_article_id():
    p = ids.parse_id("L41_2024.A64")
    assert p["law"] == "L41_2024"
    assert p["article"] == 64
    assert p["clause"] is None
    assert p["point"] is None


def test_parse_point_id():
    p = ids.parse_id("L41_2024.A64.K1.a")
    assert p["article"] == 64
    assert p["clause"] == 1
    assert p["point"] == "a"


def test_citation_label():
    assert ids.citation_label("L41_2024.A64") == "Điều 64"
    assert ids.citation_label("L41_2024.A64.K1") == "Điều 64 khoản 1"
    assert ids.citation_label("L41_2024.A64.K1.a") == "Điều 64 khoản 1 điểm a"


# ---------- semantic IDs ----------


def test_semantic_ids_co_prefix():
    assert ids.subject_id("Người lao động") == "subject:nguoi-lao-dong"
    assert ids.benefit_id("Hưu trí") == "benefit:huu-tri"
    assert ids.organization_id("Bảo hiểm xã hội Việt Nam") == "org:bao-hiem-xa-hoi-viet-nam"
    assert ids.concept_id("Thời gian đóng BHXH") == "concept:thoi-gian-dong-bhxh"
