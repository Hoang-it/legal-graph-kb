"""Test cho offline/merge_normalize.py — đảm bảo graph thống nhất + truy nguyên."""

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def graph() -> dict:
    """Đọc lại merged_graph.json (chạy merge_normalize trước)."""
    path = Path("data/processed/merged_graph.json")
    if not path.exists():
        pytest.skip(
            "data/processed/merged_graph.json không tồn tại. Chạy `python -m offline.merge_normalize` trước."
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def all_ids(graph) -> set[str]:
    s: set[str] = set()
    for _label, nodes in graph["nodes"].items():
        for n in nodes:
            s.add(n["id"])
    return s


@pytest.fixture(scope="module")
def structural_ids(graph) -> set[str]:
    s: set[str] = set()
    for label in ("Clause", "Point", "Article"):
        for n in graph["nodes"].get(label, []):
            s.add(n["id"])
    return s


# ---------- 1. Structural sanity ----------


def test_du_so_node_co_ban(graph):
    assert len(graph["nodes"]["Law"]) == 3
    assert len(graph["nodes"]["Chapter"]) == 37
    assert len(graph["nodes"]["Section"]) == 46
    assert len(graph["nodes"]["Article"]) == 486
    assert len(graph["nodes"]["Clause"]) == 1585
    assert len(graph["nodes"]["Point"]) == 829


def test_structural_edges_dung_so(graph):
    e = graph["edges"]
    assert len(e["HAS_CHAPTER"]) == 37
    assert len(e["HAS_SECTION"]) == 46
    assert len(e["HAS_ARTICLE"]) == 486
    assert len(e["HAS_CLAUSE"]) == 1585
    assert len(e["HAS_POINT"]) == 829
    assert len(e["NEXT"]) == 483


# ---------- 2. Provenance: mọi semantic node có mentioned_in ----------


def test_semantic_node_co_mentioned_in(graph):
    semantic_labels = {
        "Subject",
        "Benefit",
        "Obligation",
        "Right",
        "Condition",
        "Organization",
        "Role",
        "Fund",
        "ProhibitedAct",
        "LegalConcept",
    }
    for label in semantic_labels:
        for n in graph["nodes"].get(label, []):
            assert "mentioned_in" in n, f"{label}/{n['id']} thiếu mentioned_in"
            assert n["mentioned_in"], f"{label}/{n['id']} mentioned_in rỗng"


def test_mentioned_in_la_clause_hoac_point_co_thuc(graph, structural_ids):
    """mentioned_in phải trỏ tới Clause/Point/Article có thực."""
    semantic_labels = {
        "Subject",
        "Benefit",
        "Obligation",
        "Right",
        "Condition",
        "Organization",
        "Role",
        "Fund",
        "ProhibitedAct",
        "LegalConcept",
    }
    bad = []
    for label in semantic_labels:
        for n in graph["nodes"].get(label, []):
            for m in n["mentioned_in"]:
                if m not in structural_ids:
                    bad.append(f"{label}/{n['id']}: mentioned_in {m}")
    assert not bad, f"{len(bad)} mentioned_in tham chiếu sai. Mẫu: {bad[:3]}"


# ---------- 3. Edge integrity ----------


def test_moi_edge_co_src_dst_ton_tai(graph, all_ids):
    bad = []
    for etype, edges in graph["edges"].items():
        for e in edges:
            if e["src"] not in all_ids:
                bad.append(f"{etype}: src {e['src']}")
            if e["dst"] not in all_ids:
                bad.append(f"{etype}: dst {e['dst']}")
            if len(bad) >= 20:
                break
    assert not bad, f"Có {len(bad)} edges với src/dst orphan. Mẫu: {bad[:5]}"


def test_moi_semantic_edge_co_source_clause(graph, structural_ids):
    """Edges của các quan hệ semantic phải có source_clause + source_text."""
    semantic_types = {
        "ENTITLED_TO",
        "HAS_OBLIGATION",
        "HAS_RIGHT",
        "APPLIES_TO",
        "REQUIRES",
        "PAID_FROM",
        "MANAGES",
        "RESPONSIBLE_FOR",
        "PROHIBITED_BY",
        "DEFINES",
    }
    for etype in semantic_types:
        for e in graph["edges"].get(etype, []):
            assert e.get("source_clause"), f"{etype} thiếu source_clause"
            assert (
                e["source_clause"] in structural_ids
            ), f"{etype}: source_clause {e['source_clause']} không có thực"


def test_moi_ref_edge_co_source_clause(graph, structural_ids):
    """REFERENCES / CITES_EXTERNAL cũng phải có source_clause."""
    law_ids = {n["id"] for n in graph["nodes"]["Law"]}
    for etype in ("REFERENCES", "CITES_EXTERNAL", "AMENDS", "REPEALS", "REPLACES"):
        for e in graph["edges"].get(etype, []):
            sc = e.get("source_clause")
            if not sc and etype == "REPEALS" and e["src"] in law_ids and e["dst"] in law_ids:
                continue
            assert sc, f"{etype} thiếu source_clause"
            assert sc in structural_ids, f"{etype}: source_clause {sc} không có thực"


# ---------- 4. Dedup hoạt động ----------


def test_id_node_la_duy_nhat_trong_moi_label(graph):
    for label, nodes in graph["nodes"].items():
        ids = [n["id"] for n in nodes]
        assert len(ids) == len(
            set(ids)
        ), f"{label} có ID trùng: {[x for x in ids if ids.count(x) > 1][:5]}"


def test_nguoi_lao_dong_dedup(graph):
    """'Người lao động' xuất hiện ở nhiều Article → vẫn chỉ 1 node với mentioned_in gộp."""
    nld = [s for s in graph["nodes"]["Subject"] if s["id"] == "subject:nguoi-lao-dong"]
    assert len(nld) == 1, "Phải có đúng 1 node Người lao động (dedup)"
    assert (
        len(nld[0]["mentioned_in"]) > 20
    ), f"Mentioned_in của NLĐ chỉ {len(nld[0]['mentioned_in'])} — dedup gộp không hoạt động?"


def test_bao_hiem_xa_hoi_concept_co_dinh_nghia(graph):
    """LegalConcept 'Bảo hiểm xã hội' phải có definition + defined_in = A3.K1."""
    bhxh = next(
        (c for c in graph["nodes"]["LegalConcept"] if c["id"] == "concept:bao-hiem-xa-hoi"),
        None,
    )
    assert bhxh is not None
    assert bhxh["defined_in"] in {"L58_2014.A3.K1", "L41_2024.A3.K1"}
    assert "L41_2024.A3.K1" in bhxh["mentioned_in"]
    assert "sự bảo đảm" in bhxh["definition"]


# ---------- 5. Reverse provenance — truy nguyên từ edge ----------


def test_truy_nguoc_tu_edge_ve_clause_text(graph):
    """Lấy 1 ENTITLED_TO edge, dùng source_clause để truy lại text Clause gốc."""
    clauses_by_id = {c["id"]: c for c in graph["nodes"]["Clause"]}
    points_by_id = {p["id"]: p for p in graph["nodes"]["Point"]}
    edges = graph["edges"].get("ENTITLED_TO", [])
    assert edges, "Không có ENTITLED_TO edge nào"
    e = edges[0]
    sc = e["source_clause"]
    text = clauses_by_id.get(sc, points_by_id.get(sc))
    assert text is not None, f"Không tìm thấy {sc} trong Clause/Point"
    # source_text phải là substring của text gốc
    assert e["source_text"] in text["text"], "source_text không khớp với text Clause gốc"


def test_external_law_bo_luat_lao_dong_ton_tai(graph):
    bllao = next(
        (e for e in graph["nodes"]["ExternalLaw"] if "Lao động" in e["title"]),
        None,
    )
    assert bllao is not None


def test_external_law_58_2014_ton_tai(graph):
    e = next(
        (x for x in graph["nodes"]["ExternalLaw"] if x.get("code") == "58/2014/QH13"),
        None,
    )
    assert e is not None


# ---------- 6. Containment graph thông suốt ----------


def test_co_the_di_tu_law_xuong_moi_clause(graph):
    """Mọi Clause phải có thể reach từ Law qua HAS_CHAPTER → HAS_ARTICLE → HAS_CLAUSE."""
    clauses_reachable = set()
    for law in graph["nodes"]["Law"]:
        law_id = law["id"]
        chapters = {e["dst"] for e in graph["edges"]["HAS_CHAPTER"] if e["src"] == law_id}
        articles = {e["dst"] for e in graph["edges"]["HAS_ARTICLE"] if e["src"] in chapters}
        clauses_reachable |= {e["dst"] for e in graph["edges"]["HAS_CLAUSE"] if e["src"] in articles}
    all_clauses = {c["id"] for c in graph["nodes"]["Clause"]}
    missing = all_clauses - clauses_reachable
    assert not missing, f"{len(missing)} Clause không reach được từ Law: {list(missing)[:5]}"
