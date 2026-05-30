"""Test cho B6 — verify Neo4j có data + provenance + vector search OK.

Skip nếu Neo4j không kết nối được hoặc DB trống (chưa chạy load).
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

try:
    from neo4j import GraphDatabase

    URI = os.getenv("NEO4J_URI")
    USER = os.getenv("NEO4J_USER")
    PWD = os.getenv("NEO4J_PASSWORD")
    DB = os.getenv("NEO4J_DATABASE", "neo4j")
    _driver = GraphDatabase.driver(URI, auth=(USER, PWD))
    _driver.verify_connectivity()
    with _driver.session(database=DB) as _s:
        _n = _s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    if _n == 0:
        pytest.skip("Neo4j trống — chạy `python -m offline.load_neo4j` trước.", allow_module_level=True)
except Exception as e:
    pytest.skip(f"Không kết nối được Neo4j: {e}", allow_module_level=True)


@pytest.fixture(scope="module")
def session():
    drv = GraphDatabase.driver(URI, auth=(USER, PWD))
    s = drv.session(database=DB)
    yield s
    s.close()
    drv.close()


# ---------- 1. Node counts ----------


@pytest.mark.parametrize(
    "label,expected",
    [
        # Baseline: 5 luật (L41/L58/L45/ND143/QD838) — xem docs/known_issues_kg_build.md.
        ("Law", 5),
        ("Chapter", 43),
        ("Section", 46),
        ("Article", 507),
        ("Clause", 1645),
        ("Point", 862),
    ],
)
def test_node_counts(session, label, expected):
    n = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
    assert n == expected, f"{label}: expected {expected}, got {n}"


# ---------- 2. Provenance enforcement (DB constraints) ----------


def test_moi_semantic_edge_co_source_clause(session):
    """Constraint *_src đảm bảo không thể tạo edge thiếu source_clause."""
    n = session.run("""
        MATCH ()-[r:ENTITLED_TO|HAS_OBLIGATION|HAS_RIGHT|REQUIRES|APPLIES_TO
                  |PAID_FROM|MANAGES|RESPONSIBLE_FOR|PROHIBITED_BY|DEFINES]-()
        WHERE r.source_clause IS NULL
          AND NOT startNode(r):Clause
        RETURN count(r) AS c
    """).single()["c"]
    assert n == 0


def test_moi_ref_edge_co_source_clause(session):
    n = session.run("""
        MATCH ()-[r:REFERENCES|CITES_EXTERNAL|AMENDS|REPEALS|REPLACES]-()
        WHERE r.source_clause IS NULL
          AND NOT startNode(r):Law
        RETURN count(r) AS c
    """).single()["c"]
    assert n == 0


def test_moi_article_clause_point_co_text(session):
    for label in ("Article", "Clause", "Point"):
        n = session.run(f"MATCH (n:{label}) WHERE n.text IS NULL RETURN count(n) AS c").single()[
            "c"
        ]
        assert n == 0, f"{label}: {n} node thiếu text"


def test_moi_semantic_node_co_mentioned_in(session):
    for label in (
        "Subject",
        "Benefit",
        "Obligation",
        "Right",
        "Condition",
        "Organization",
        "Fund",
        "ProhibitedAct",
    ):
        n = session.run(
            f"MATCH (n:{label}) WHERE n.mentioned_in IS NULL RETURN count(n) AS c"
        ).single()["c"]
        assert n == 0, f"{label}: {n} node thiếu mentioned_in"


# ---------- 3. Embeddings ----------


def test_embedding_coverage(session):
    # Baseline: 5 luật — xem docs/known_issues_kg_build.md.
    for label, expected in [("Article", 507), ("Clause", 1645), ("Point", 862)]:
        n = session.run(
            f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL RETURN count(n) AS c"
        ).single()["c"]
        assert n == expected, f"{label}: embedding {n}/{expected}"


def test_embedding_dim_1024(session):
    """Lấy 1 Article có embedding, kiểm tra size = 1024."""
    n = session.run(
        "MATCH (a:Article) WHERE a.embedding IS NOT NULL RETURN size(a.embedding) AS dim LIMIT 1"
    ).single()["dim"]
    assert n == 1024


# ---------- 4. Vector index ----------


def test_vector_index_ton_tai(session):
    rows = session.run("SHOW INDEXES YIELD name, type WHERE type = 'VECTOR' RETURN name").data()
    names = {r["name"] for r in rows}
    assert {"article_vec", "clause_vec", "point_vec"}.issubset(names)


def test_vector_search_a64_neighbors(session):
    """Top-5 láng giềng vector của A64 phải thuộc các Điều về hưu trí (60-72, 95-110)."""
    result = session.run("""
        MATCH (q:Article {id: 'L41_2024.A64'})
        CALL db.index.vector.queryNodes('article_vec', 6, q.embedding)
        YIELD node, score
        WHERE node.id <> q.id
        RETURN node.id AS id, score
        ORDER BY score DESC LIMIT 5
    """).data()
    assert len(result) == 5
    relevant_range = (
        {f"L41_2024.A{n}" for n in range(60, 80)}
        | {f"L41_2024.A{n}" for n in range(95, 110)}
        | {f"L58_2014.A{n}" for n in range(50, 76)}
        | {"L45_2019.A169"}
    )
    hits = sum(1 for r in result if r["id"] in relevant_range)
    assert hits >= 4, f"Top-5 chỉ {hits}/5 thuộc nhóm hưu trí: {[r['id'] for r in result]}"


# ---------- 5. Provenance roundtrip ----------


def test_subject_nld_truy_nguoc_ve_clause_text(session):
    """Subject NLĐ → ENTITLED_TO Benefit → source_clause → đọc text gốc."""
    result = session.run("""
        MATCH (s:Subject {id: 'subject:nguoi-lao-dong'})-[r:ENTITLED_TO]->(b:Benefit)
        MATCH (c:Clause {id: r.source_clause})
        RETURN b.name AS benefit, c.id AS cid, c.text AS clause_text, r.source_text AS evidence
        LIMIT 5
    """).data()
    assert result, "Không có ENTITLED_TO edge nào từ NLĐ"
    for r in result:
        # source_text phải là substring nguyên văn của clause_text
        assert (
            r["evidence"] in r["clause_text"]
        ), f"source_text không phải substring của clause text gốc tại {r['cid']}"


def test_definition_concept_truy_ve_dieu_3(session):
    """LegalConcept defined_in phải trỏ tới Clause của Điều 3."""
    rows = session.run("MATCH (c:LegalConcept) RETURN c.defined_in AS d").data()
    for r in rows:
        assert r["d"].split(".A", 1)[1].startswith("3.K"), f"Concept defined_in sai: {r['d']}"


# ---------- 6. Containment graph ----------


def test_moi_clause_reach_tu_law(session):
    n = session.run("""
        MATCH (l:Law)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_ARTICLE]->(:Article)-[:HAS_CLAUSE]->(c:Clause)
        RETURN count(DISTINCT c) AS c
    """).single()["c"]
    total = session.run("MATCH (c:Clause) RETURN count(c) AS c").single()["c"]
    assert n == total, f"Chỉ {n}/{total} Clause reach từ Law"


def test_article_thuoc_section_dung_chuong(session):
    """Article có IN_SECTION thì Section phải thuộc cùng Chapter cha của Article."""
    n = session.run("""
        MATCH (a:Article)-[:IN_SECTION]->(s:Section)
        MATCH (ch1:Chapter)-[:HAS_ARTICLE]->(a)
        MATCH (ch2:Chapter)-[:HAS_SECTION]->(s)
        WHERE ch1 <> ch2
        RETURN count(*) AS c
    """).single()["c"]
    assert n == 0, f"{n} Article có IN_SECTION tới Section của Chapter khác"
