"""Test cho refactor multi-law của B3 (offline.llm_extract) và B4 fallback.

Kiểm tra:
- ``LawMetadata`` đọc đúng ``llm_skip_articles`` / ``llm_skip_reason`` từ YAML.
- Helper ``structured_path_for`` / ``out_dir_for`` derive đúng path theo law_id,
  với fallback legacy chỉ áp dụng cho L41_2024.
- ``merge_normalize._llm_files_for`` ưu tiên subdir per-law và fallback flat
  top-level chỉ cho L41_2024.
- ``extract_article`` skip theo metadata (data-driven), không còn hardcode 141.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from offline import llm_extract, merge_normalize
from src.legal_metadata import LawMetadata, metadata_for


# ---------------------------------------------------------------------------
# 1. LawMetadata đọc skip rule từ YAML
# ---------------------------------------------------------------------------


def test_l41_yaml_co_khai_bao_skip_141():
    meta = metadata_for("L41_2024")
    assert meta.llm_skip_articles == (141,)
    assert meta.llm_skip_reason  # non-empty


def test_l58_yaml_khong_co_skip_articles():
    meta = metadata_for("L58_2014")
    assert meta.llm_skip_articles == ()
    assert meta.llm_skip_reason == ""


def test_l45_yaml_khong_co_skip_articles():
    meta = metadata_for("L45_2019")
    assert meta.llm_skip_articles == ()
    assert meta.llm_skip_reason == ""


# ---------------------------------------------------------------------------
# 2. structured_path_for: ưu tiên per-law, fallback legacy chỉ cho L41
# ---------------------------------------------------------------------------


def test_structured_path_uu_tien_per_law(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_extract, "INTERIM_DIR", tmp_path)
    per_law = tmp_path / "structured_law_L41_2024.json"
    per_law.write_text("{}", encoding="utf-8")
    assert llm_extract.structured_path_for("L41_2024") == per_law


def test_structured_path_fallback_legacy_chi_cho_l41(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_extract, "INTERIM_DIR", tmp_path)
    legacy = tmp_path / "structured_law.json"
    legacy.write_text("{}", encoding="utf-8")
    # L41 fallback
    assert llm_extract.structured_path_for("L41_2024") == legacy
    # Luật khác KHÔNG fallback — trả về path per-law (không tồn tại)
    result = llm_extract.structured_path_for("ND143_2018")
    assert result == tmp_path / "structured_law_ND143_2018.json"
    assert not result.exists()


def test_structured_path_neither_files_l41_van_tra_canonical(tmp_path, monkeypatch):
    """Khi không có file nào tồn tại, caller phải nhận path canonical để báo lỗi rõ."""
    monkeypatch.setattr(llm_extract, "INTERIM_DIR", tmp_path)
    result = llm_extract.structured_path_for("L41_2024")
    assert result == tmp_path / "structured_law_L41_2024.json"
    assert not result.exists()


# ---------------------------------------------------------------------------
# 3. out_dir_for: per-law subdir, không pollute root
# ---------------------------------------------------------------------------


def test_out_dir_per_law(tmp_path, monkeypatch):
    root = tmp_path / "llm_extractions"
    monkeypatch.setattr(llm_extract, "LLM_ROOT", root)
    assert llm_extract.out_dir_for("L41_2024") == root / "L41_2024"
    assert llm_extract.out_dir_for("ND143_2018") == root / "ND143_2018"


# ---------------------------------------------------------------------------
# 4. B4 _llm_files_for: per-law subdir, fallback L41 only
# ---------------------------------------------------------------------------


def test_b4_uu_tien_subdir_per_law_khi_co(tmp_path, monkeypatch):
    monkeypatch.setattr(merge_normalize, "LLM_DIR", tmp_path)
    subdir = tmp_path / "L41_2024"
    subdir.mkdir()
    (subdir / "A1.json").write_text(json.dumps({"article_id": "L41_2024.A1"}), encoding="utf-8")
    (subdir / "A2.json").write_text(json.dumps({"article_id": "L41_2024.A2"}), encoding="utf-8")
    # Flat cũ — phải bị ignore khi subdir tồn tại (tránh duplicate)
    (tmp_path / "A1.json").write_text(json.dumps({"article_id": "OLD"}), encoding="utf-8")

    files = merge_normalize._llm_files_for("L41_2024")
    assert [f["article_id"] for f in files] == ["L41_2024.A1", "L41_2024.A2"]


def test_b4_fallback_flat_chi_cho_l41(tmp_path, monkeypatch):
    monkeypatch.setattr(merge_normalize, "LLM_DIR", tmp_path)
    (tmp_path / "A1.json").write_text(json.dumps({"article_id": "X"}), encoding="utf-8")
    # L41 fallback
    files_l41 = merge_normalize._llm_files_for("L41_2024")
    assert [f["article_id"] for f in files_l41] == ["X"]
    # Luật khác KHÔNG fallback flat (an toàn — không gán nhầm file cũ cho luật mới)
    assert merge_normalize._llm_files_for("ND143_2018") == []
    assert merge_normalize._llm_files_for("L58_2014") == []


def test_b4_subdir_per_law_cho_luat_khac_l41(tmp_path, monkeypatch):
    monkeypatch.setattr(merge_normalize, "LLM_DIR", tmp_path)
    subdir = tmp_path / "ND143_2018"
    subdir.mkdir()
    (subdir / "A1.json").write_text(json.dumps({"article_id": "ND143_2018.A1"}), encoding="utf-8")
    files = merge_normalize._llm_files_for("ND143_2018")
    assert [f["article_id"] for f in files] == ["ND143_2018.A1"]


def test_b4_khong_co_file_tra_ve_rong(tmp_path, monkeypatch):
    monkeypatch.setattr(merge_normalize, "LLM_DIR", tmp_path)
    assert merge_normalize._llm_files_for("L41_2024") == []
    assert merge_normalize._llm_files_for("ND143_2018") == []


def test_b4_isolation_giua_cac_luat(tmp_path, monkeypatch):
    """File của luật A không leak vào output của luật B (key collision test)."""
    monkeypatch.setattr(merge_normalize, "LLM_DIR", tmp_path)
    (tmp_path / "L41_2024").mkdir()
    (tmp_path / "L41_2024" / "A1.json").write_text(
        json.dumps({"article_id": "L41_2024.A1"}), encoding="utf-8"
    )
    (tmp_path / "ND143_2018").mkdir()
    (tmp_path / "ND143_2018" / "A1.json").write_text(
        json.dumps({"article_id": "ND143_2018.A1"}), encoding="utf-8"
    )

    files_l41 = merge_normalize._llm_files_for("L41_2024")
    files_nd = merge_normalize._llm_files_for("ND143_2018")
    assert [f["article_id"] for f in files_l41] == ["L41_2024.A1"]
    assert [f["article_id"] for f in files_nd] == ["ND143_2018.A1"]


# ---------------------------------------------------------------------------
# 5. extract_article: skip data-driven theo meta.llm_skip_articles
# ---------------------------------------------------------------------------


def _make_meta(law_id: str, skip_articles=(), skip_reason="") -> LawMetadata:
    return LawMetadata(
        id=law_id,
        code=law_id,
        full_id=law_id,
        title="",
        canonical_title=law_id,
        type="law",
        hierarchy_level="luật",
        priority=100,
        source_file=Path(""),
        llm_skip_articles=skip_articles,
        llm_skip_reason=skip_reason,
    )


def test_extract_article_skip_voi_reason_tu_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_extract, "LLM_ROOT", tmp_path)
    meta = _make_meta("L41_2024", skip_articles=(141,), skip_reason="custom reason from YAML")
    art = {
        "id": "L41_2024.A141",
        "number": 141,
        "clauses": [{"id": "L41_2024.A141.K1", "text": "x", "points": []}],
    }
    ch = {"roman": "XI", "title": "Điều khoản thi hành"}

    result = asyncio.run(
        llm_extract.extract_article(
            client=None, sem=None, art=art, chapter=ch, section=None, meta=meta
        )
    )
    assert result == {"article_id": "L41_2024.A141", "ok": True, "skipped": True}
    out_path = tmp_path / "L41_2024" / "A141.json"
    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["skipped_reason"] == "custom reason from YAML"
    assert data["article_id"] == "L41_2024.A141"


def test_extract_article_skip_reason_fallback_data_driven(tmp_path, monkeypatch):
    """Khi YAML không khai báo reason — fallback nhắc tới nguồn (YAML field) và Điều."""
    monkeypatch.setattr(llm_extract, "LLM_ROOT", tmp_path)
    meta = _make_meta("ND143_2018", skip_articles=(5,))  # llm_skip_reason rỗng
    art = {
        "id": "ND143_2018.A5",
        "number": 5,
        "clauses": [{"id": "ND143_2018.A5.K1", "text": "x", "points": []}],
    }
    ch = {"roman": "I", "title": ""}

    result = asyncio.run(
        llm_extract.extract_article(
            client=None, sem=None, art=art, chapter=ch, section=None, meta=meta
        )
    )
    assert result["skipped"] is True
    data = json.loads((tmp_path / "ND143_2018" / "A5.json").read_text(encoding="utf-8"))
    # Fallback reason phải nhắc tới (a) nguồn = YAML field, (b) số Điều bị skip
    assert "law_metadata" in data["skipped_reason"]
    assert "llm_skip_articles" in data["skipped_reason"]
    assert "5" in data["skipped_reason"]
    # Quan trọng: không hardcode "141"
    assert "141" not in data["skipped_reason"]


def test_extract_article_no_clauses_giu_nguyen_behavior_cu(tmp_path, monkeypatch):
    """Skip 'no_clauses (lead_text only)' phải giữ nguyên — không bị nuốt bởi refactor."""
    monkeypatch.setattr(llm_extract, "LLM_ROOT", tmp_path)
    meta = _make_meta("L41_2024")
    art = {"id": "L41_2024.A1", "number": 1, "clauses": []}
    ch = {"roman": "I", "title": ""}

    result = asyncio.run(
        llm_extract.extract_article(
            client=None, sem=None, art=art, chapter=ch, section=None, meta=meta
        )
    )
    assert result["skipped"] is True
    data = json.loads((tmp_path / "L41_2024" / "A1.json").read_text(encoding="utf-8"))
    assert data["skipped_reason"] == "no_clauses (lead_text only)"


def test_extract_article_khong_skip_neu_article_khong_trong_skip_list(tmp_path, monkeypatch):
    """Article ngoài skip_list nhưng đã có file cache → return None (skip do cache)."""
    monkeypatch.setattr(llm_extract, "LLM_ROOT", tmp_path)
    meta = _make_meta("L41_2024", skip_articles=(141,))
    out_dir = tmp_path / "L41_2024"
    out_dir.mkdir()
    cached = out_dir / "A5.json"
    cached.write_text("{}", encoding="utf-8")
    art = {
        "id": "L41_2024.A5",
        "number": 5,
        "clauses": [{"id": "L41_2024.A5.K1", "text": "x", "points": []}],
    }
    ch = {"roman": "I", "title": ""}

    result = asyncio.run(
        llm_extract.extract_article(
            client=None, sem=None, art=art, chapter=ch, section=None, meta=meta
        )
    )
    # skip_existing=True (default) → return None vì file cache đã tồn tại
    assert result is None


def test_extract_article_a141_khong_skip_khi_yaml_khac_luat(tmp_path, monkeypatch):
    """Điều 141 của 1 luật khác (không khai báo skip) PHẢI được xử lý bình thường.

    Đây là điểm chốt chứng minh không còn hardcode `art_n == 141`.
    """
    monkeypatch.setattr(llm_extract, "LLM_ROOT", tmp_path)
    meta = _make_meta("ND_TEST", skip_articles=())  # KHÔNG skip Điều 141
    out_dir = tmp_path / "ND_TEST"
    out_dir.mkdir()
    cached = out_dir / "A141.json"
    cached.write_text("{}", encoding="utf-8")
    art = {
        "id": "ND_TEST.A141",
        "number": 141,
        "clauses": [{"id": "ND_TEST.A141.K1", "text": "x", "points": []}],
    }
    ch = {"roman": "I", "title": ""}

    result = asyncio.run(
        llm_extract.extract_article(
            client=None, sem=None, art=art, chapter=ch, section=None, meta=meta
        )
    )
    # Không có skip rule → đi tới check cache → return None vì cache có.
    # Nếu hardcode 141 còn → sẽ ghi đè cache với skipped_reason. Test này phát hiện.
    assert result is None
    # File cache còn nguyên (không bị ghi đè bởi nhánh skip)
    assert cached.read_text(encoding="utf-8") == "{}"
