"""Test cho src/ids.py — ID convention & provenance round-trip."""

import pytest

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


# ============================================================================
# Phase 1 — parse_id chấp nhận mọi source registry (NĐ/QĐ/TT/CV/Hiệp định/…)
# ============================================================================


def test_parse_id_l41_van_giu_nguyen():
    """Backward compat — không có regression với prefix L41_2024."""
    p = ids.parse_id("L41_2024.A64.K1.a")
    assert p == {
        "law": "L41_2024",
        "chapter": None,
        "article": 64,
        "clause": 1,
        "point": "a",
        "table": None,
    }


def test_parse_id_nghi_dinh():
    p = ids.parse_id("ND143_2018.A5.K1.a")
    assert p["law"] == "ND143_2018"
    assert p["article"] == 5
    assert p["clause"] == 1
    assert p["point"] == "a"


def test_parse_id_quyet_dinh():
    p = ids.parse_id("QD366_BHXH.A1")
    assert p["law"] == "QD366_BHXH"
    assert p["article"] == 1
    assert p["clause"] is None


def test_parse_id_thong_tu():
    p = ids.parse_id("TT18_2022_BYT.A3.K2")
    assert p["law"] == "TT18_2022_BYT"
    assert p["article"] == 3
    assert p["clause"] == 2


def test_parse_id_hiep_dinh():
    p = ids.parse_id("HIEPDINH_VN_KR_BHXH.A4")
    assert p["law"] == "HIEPDINH_VN_KR_BHXH"
    assert p["article"] == 4


def test_parse_id_phap_lenh():
    p = ids.parse_id("PHAPLENH_NCC.A2.K3")
    assert p["law"] == "PHAPLENH_NCC"
    assert p["article"] == 2
    assert p["clause"] == 3


def test_parse_id_bo_luat_dan_su():
    p = ids.parse_id("BLDS_2015.A100.K1.a")
    assert p["law"] == "BLDS_2015"
    assert p["article"] == 100


def test_parse_id_cong_van():
    p = ids.parse_id("CV2068_BYT_BH.A1")
    assert p["law"] == "CV2068_BYT_BH"


def test_parse_id_reject_id_khong_hop_le():
    with pytest.raises(ValueError):
        ids.parse_id("123.A5")  # bắt đầu bằng số
    with pytest.raises(ValueError):
        ids.parse_id("lowercase.A5")  # bắt đầu bằng chữ thường
    with pytest.raises(ValueError):
        ids.parse_id("")  # rỗng
    with pytest.raises(ValueError):
        ids.parse_id("L41_2024.X9")  # marker không hợp lệ (X thay vì A/K/C/T)


def test_citation_label_tu_dong_lam_viec_cho_nghi_dinh():
    assert ids.citation_label("ND143_2018.A5.K1.a") == "Điều 5 khoản 1 điểm a"


def test_citation_label_tu_dong_lam_viec_cho_quyet_dinh():
    assert ids.citation_label("QD366_BHXH.A1") == "Điều 1"


def test_citation_label_tu_dong_lam_viec_cho_thong_tu():
    assert ids.citation_label("TT18_2022_BYT.A3.K2") == "Điều 3 khoản 2"


def test_parse_id_round_trip_voi_nghi_dinh():
    """Round-trip: sinh ID → parse → khớp."""
    nd = "ND143_2018"
    art = ids.article_id(nd, 7)
    cl = ids.clause_id(nd, 7, 2)
    pt = ids.point_id(nd, 7, 2, "b")
    assert ids.parse_id(art) == {
        "law": nd, "chapter": None, "article": 7,
        "clause": None, "point": None, "table": None,
    }
    assert ids.parse_id(cl)["clause"] == 2
    assert ids.parse_id(pt)["point"] == "b"


# ============================================================================
# Phase 2 — law_id chỉ convert mã QH, các ID đã canonical pass-through
# ============================================================================


def test_law_id_canonical_pass_through_qh():
    """Backward compat — L<n>_<yyyy> vẫn pass-through."""
    assert ids.law_id("L41_2024") == "L41_2024"
    assert ids.law_id("L58_2014") == "L58_2014"
    assert ids.law_id("L45_2019") == "L45_2019"


def test_law_id_canonical_pass_through_nghi_dinh():
    assert ids.law_id("ND143_2018") == "ND143_2018"
    assert ids.law_id("ND158_2025") == "ND158_2025"


def test_law_id_canonical_pass_through_quyet_dinh_thong_tu():
    assert ids.law_id("QD366_BHXH") == "QD366_BHXH"
    assert ids.law_id("TT18_2022_BYT") == "TT18_2022_BYT"
    assert ids.law_id("CV2068_BYT_BH") == "CV2068_BYT_BH"
    assert ids.law_id("HIEPDINH_VN_KR_BHXH") == "HIEPDINH_VN_KR_BHXH"
    assert ids.law_id("PHAPLENH_NCC") == "PHAPLENH_NCC"


def test_law_id_chi_convert_ma_qh():
    """Mã Quốc hội ``XX/YYYY/QH<n>`` → convert canonical L<XX>_<YYYY>."""
    assert ids.law_id("41/2024/QH15") == "L41_2024"
    assert ids.law_id("58/2014/QH13") == "L58_2014"
    assert ids.law_id("45/2019/QH14") == "L45_2019"


def test_law_id_khong_convert_nham_ma_nghi_dinh():
    """Trước đây ``143/2018/NĐ-CP`` ngầm convert sai thành ``L143_2018``
    (collision với hypothetical Luật 143/2018). Giờ phải reject — caller
    truyền canonical ID đã đăng ký YAML.
    """
    with pytest.raises(ValueError):
        ids.law_id("143/2018/NĐ-CP")
    with pytest.raises(ValueError):
        ids.law_id("146/2018/NĐ-CP")


def test_law_id_reject_ma_qd_khong_co_nam():
    """``366/QĐ-BHXH`` không có năm ở vị trí 2 — không thể derive canonical."""
    with pytest.raises(ValueError):
        ids.law_id("366/QĐ-BHXH")
    with pytest.raises(ValueError):
        ids.law_id("18/2022/TT-BYT")  # mã TT cũng không tự convert được


def test_law_id_reject_input_rong_va_rac():
    with pytest.raises(ValueError):
        ids.law_id("")
    with pytest.raises(ValueError):
        ids.law_id("not a code")
    with pytest.raises(ValueError):
        ids.law_id("lowercase_id")  # không match canonical regex (chữ thường)
