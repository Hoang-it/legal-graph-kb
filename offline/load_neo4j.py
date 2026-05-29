"""B6 — Nạp merged graph + embeddings vào Neo4j 5.x.

Đầu vào:
    data/graph/processed/merged_graph.json    (output B4)
    data/graph/processed/embeddings.parquet   (output B5)
    schema/schema.cypher                (constraints + vector indexes)

Hoạt động:
1. Verify connectivity tới Neo4j (URI/USER/PWD từ .env).
2. (Tuỳ chọn) `--reset`: xoá toàn bộ nodes + edges (constraints/indexes giữ).
3. (Tuỳ chọn) `--apply-schema`: chạy schema.cypher (IF NOT EXISTS, idempotent).
4. UNWIND/MERGE batch-load tất cả nodes theo từng label.
5. UNWIND/MATCH/MERGE batch-load tất cả edges theo từng type.
6. Set vector property `embedding` cho Article/Clause/Point.
7. Chạy sanity Cypher queries (count, provenance roundtrip, vector search).

Idempotent: MERGE với khoá thích hợp → chạy lại không sinh trùng.
- Structural edges: khoá = (src, dst).
- REFERENCES / CITES_EXTERNAL: khoá = (src, dst, source_clause, char_offset).
- Các edge khác (AMENDS/REPEALS/REPLACES + semantic): khoá = (src, dst, source_clause).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")
DB = os.getenv("NEO4J_DATABASE", "neo4j")

GRAPH_PATH = Path("data/graph/processed/merged_graph.json")
EMBED_PATH = Path("data/graph/processed/embeddings.parquet")
SCHEMA_PATH = Path("schema/schema.cypher")

STRUCTURAL_EDGE_TYPES = {
    "HAS_CHAPTER",
    "HAS_SECTION",
    "HAS_ARTICLE",
    "BELONGS_TO",
    "IN_SECTION",
    "HAS_CLAUSE",
    "HAS_POINT",
    "HAS_TABLE",
    "NEXT",
}
EDGES_WITH_OFFSET = {"REFERENCES", "CITES_EXTERNAL", "REFERS_TO"}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def split_cypher(text: str) -> list[str]:
    """Tách file .cypher thành các câu lệnh — split theo `;` ngoài cùng,
    bỏ comment `//...`. Đủ cho schema.cypher (không có dấu `;` trong string)."""
    lines = []
    for line in text.split("\n"):
        # Strip comment // (nhưng không bỏ dấu // bên trong string — schema.cypher không có)
        if "//" in line:
            line = line[: line.index("//")]
        lines.append(line)
    cleaned = "\n".join(lines)
    stmts = [s.strip() for s in cleaned.split(";")]
    return [s for s in stmts if s]


def apply_schema(driver: Driver, db: str) -> None:
    print(f"\n=== APPLY SCHEMA ({SCHEMA_PATH}) ===")
    if not SCHEMA_PATH.exists():
        print(f"  WARN: thiếu {SCHEMA_PATH}, skip")
        return
    stmts = split_cypher(SCHEMA_PATH.read_text(encoding="utf-8"))
    print(f"  Áp {len(stmts)} câu lệnh...")
    ok, skipped = 0, 0
    with driver.session(database=db) as s:
        for i, stmt in enumerate(stmts):
            try:
                s.run(stmt)
                ok += 1
            except Exception as e:
                msg = str(e)
                if (
                    "already exists" in msg.lower()
                    or "EquivalentSchemaRuleAlreadyExistsException" in msg
                ):
                    skipped += 1
                else:
                    print(f"  ✗ [{i}] {stmt[:80]}...")
                    print(f"      {type(e).__name__}: {msg[:150]}")
    print(f"  OK: {ok}  (already-exists: {skipped})")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def load_nodes(driver: Driver, db: str, graph: dict) -> int:
    print("\n=== LOAD NODES ===")
    total = 0
    BATCH = 500
    with driver.session(database=db) as s:
        for label, nodes in graph["nodes"].items():
            if not nodes:
                continue
            n_label = 0
            for i in range(0, len(nodes), BATCH):
                chunk = nodes[i : i + BATCH]
                # Loại None values (Neo4j thích property KHÔNG có hơn là property=null)
                clean = [{k: v for k, v in n.items() if v is not None} for n in chunk]
                s.run(
                    f"UNWIND $batch AS n MERGE (x:{label} {{id: n.id}}) SET x += n",
                    batch=clean,
                )
                n_label += len(chunk)
            print(f"  {label:<18} {n_label:>5}")
            total += n_label
    print(f"  TOTAL nodes: {total}")
    return total


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def load_edges(driver: Driver, db: str, graph: dict) -> int:
    print("\n=== LOAD EDGES ===")
    total = 0
    with driver.session(database=db) as s:
        for etype, edges in graph["edges"].items():
            if not edges:
                continue
            n = _merge_edges(s, etype, edges)
            print(f"  {etype:<22} {n:>5}")
            total += n
    print(f"  TOTAL edges: {total}")
    return total


def _merge_edges(session, etype: str, edges: list[dict]) -> int:
    BATCH = 200

    if etype in STRUCTURAL_EDGE_TYPES:
        query = (
            "UNWIND $batch AS e "
            "MATCH (src {id: e.src}) "
            "MATCH (dst {id: e.dst}) "
            f"MERGE (src)-[:{etype}]->(dst)"
        )
        chunks_iter = (
            [{"src": e["src"], "dst": e["dst"]} for e in edges[i : i + BATCH]]
            for i in range(0, len(edges), BATCH)
        )
    elif etype in EDGES_WITH_OFFSET:
        query = (
            "UNWIND $batch AS e "
            "MATCH (src {id: e.src}) "
            "MATCH (dst {id: e.dst}) "
            f"MERGE (src)-[r:{etype} {{source_clause: e.source_clause, char_offset: e.char_offset}}]->(dst) "
            "SET r += e.props"
        )
        chunks_iter = (
            [_split_edge(e, ["src", "dst"]) for e in edges[i : i + BATCH]]
            for i in range(0, len(edges), BATCH)
        )
    else:
        with_source = [e for e in edges if e.get("source_clause") is not None]
        without_source = [e for e in edges if e.get("source_clause") is None]
        n = 0
        if with_source:
            n += _merge_edges_with_source_clause(session, etype, with_source, BATCH)
        if without_source:
            n += _merge_edges_without_source_clause(session, etype, without_source, BATCH)
        return n

    n = 0
    for batch in chunks_iter:
        session.run(query, batch=batch)
        n += len(batch)
    return n


def _merge_edges_with_source_clause(session, etype: str, edges: list[dict], batch_size: int) -> int:
        # Semantic + AMENDS/REPEALS/REPLACES — khoá source_clause là đủ
    query = (
        "UNWIND $batch AS e "
        "MATCH (src {id: e.src}) "
        "MATCH (dst {id: e.dst}) "
        f"MERGE (src)-[r:{etype} {{source_clause: e.source_clause}}]->(dst) "
        "SET r += e.props"
    )
    n = 0
    for i in range(0, len(edges), batch_size):
        batch = [_split_edge(e, ["src", "dst"]) for e in edges[i : i + batch_size]]
        session.run(query, batch=batch)
        n += len(batch)
    return n


def _merge_edges_without_source_clause(session, etype: str, edges: list[dict], batch_size: int) -> int:
    query = (
        "UNWIND $batch AS e "
        "MATCH (src {id: e.src}) "
        "MATCH (dst {id: e.dst}) "
        f"MERGE (src)-[r:{etype}]->(dst) "
        "SET r += e.props"
    )
    n = 0
    for i in range(0, len(edges), batch_size):
        batch = [_split_edge(e, ["src", "dst"]) for e in edges[i : i + batch_size]]
        session.run(query, batch=batch)
        n += len(batch)
    return n


def _split_edge(e: dict, keep_top: list[str]) -> dict:
    """Tách edge: giữ src/dst (+source_clause+char_offset) ở top-level,
    còn lại đưa vào props (để `SET r += e.props` không gán đè src/dst)."""
    out = {k: e[k] for k in keep_top if k in e}
    out["source_clause"] = e.get("source_clause")
    if "char_offset" in e:
        out["char_offset"] = e["char_offset"]
    props = {k: v for k, v in e.items() if k not in keep_top and v is not None}
    out["props"] = props
    return out


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def load_embeddings(driver: Driver, db: str) -> int:
    print("\n=== LOAD EMBEDDINGS ===")
    if not EMBED_PATH.exists():
        print(f"  WARN: thiếu {EMBED_PATH} — skip")
        return 0
    df = pd.read_parquet(EMBED_PATH)
    print(f"  Loaded {len(df)} embeddings từ parquet")

    BATCH = 100
    total = 0
    with driver.session(database=db) as s:
        for label in ("Article", "Clause", "Point"):
            subset = df[df["label"] == label]
            if subset.empty:
                continue
            n = 0
            for i in range(0, len(subset), BATCH):
                chunk = subset.iloc[i : i + BATCH]
                batch = [
                    {"id": r["id"], "vec": [float(x) for x in r["embedding"]]}
                    for _, r in chunk.iterrows()
                ]
                s.run(
                    f"""
                    UNWIND $batch AS row
                    MATCH (n:{label} {{id: row.id}})
                    CALL db.create.setNodeVectorProperty(n, 'embedding', row.vec)
                    """,
                    batch=batch,
                )
                n += len(chunk)
            print(f"  {label:<10} {n:>5} embeddings")
            total += n
    return total


# ---------------------------------------------------------------------------
# Sanity Cypher queries
# ---------------------------------------------------------------------------


def sanity(driver: Driver, db: str) -> None:
    print("\n=== SANITY QUERIES ===")
    with driver.session(database=db) as s:
        # 1. Counts
        print("  Node counts:")
        for label in (
            "Law",
            "Chapter",
            "Section",
            "Article",
            "Clause",
            "Point",
            "Subject",
            "Benefit",
            "Obligation",
            "LegalConcept",
            "ExternalLaw",
        ):
            n = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            print(f"    {label:<14} {n}")

        # 2. Mọi semantic edge có source_clause
        n = s.run("""
            MATCH ()-[r:ENTITLED_TO|HAS_OBLIGATION|HAS_RIGHT|REQUIRES|APPLIES_TO|PAID_FROM|MANAGES|RESPONSIBLE_FOR|PROHIBITED_BY|DEFINES]-()
            WHERE r.source_clause IS NULL
            RETURN count(r) AS c
        """).single()["c"]
        print(f"\n  Semantic edges KHÔNG có source_clause: {n}  (expect 0)")

        # 3. Embedding coverage
        for label in ("Article", "Clause", "Point"):
            n = s.run(
                f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL RETURN count(n) AS c"
            ).single()["c"]
            total = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            print(f"  {label} có embedding: {n}/{total}")

        # 4. Vector search top-5 láng giềng của A64
        print("\n  Vector search neighbors của A64 (hưu trí BB):")
        try:
            result = s.run("""
                MATCH (q:Article {id: 'L41_2024.A64'})
                CALL db.index.vector.queryNodes('article_vec', 6, q.embedding) YIELD node, score
                WHERE node.id <> q.id
                RETURN node.id AS id, node.title AS title, score
                ORDER BY score DESC LIMIT 5
            """).data()
            for r in result:
                print(f"    sim={r['score']:.3f}  {r['id']}: {r['title'][:70]}")
        except Exception as e:
            print(f"    ✗ Vector index không hoạt động: {e}")

        # 5. Provenance roundtrip: entity → clause text
        print("\n  Provenance: từ Subject 'NLĐ' truy ngược về Clause text gốc:")
        result = s.run("""
            MATCH (s:Subject {id: 'subject:nguoi-lao-dong'})-[r:ENTITLED_TO]->(b:Benefit)
            MATCH (c:Clause {id: r.source_clause})
            RETURN b.name AS benefit, c.id AS clause_id, left(c.text, 120) AS clause_text
            LIMIT 3
        """).data()
        for r in result:
            print(f"    NLĐ → {r['benefit']}")
            print(f"      ↑ {r['clause_id']}: \"{r['clause_text']}...\"")

        # 6. Graph reach: từ Law xuống tới mọi Clause?
        n = s.run("""
            MATCH (l:Law)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_ARTICLE]->(:Article)-[:HAS_CLAUSE]->(c:Clause)
            RETURN count(DISTINCT c) AS c
        """).single()["c"]
        total = s.run("MATCH (c:Clause) RETURN count(c) AS c").single()["c"]
        print(f"\n  Clause reachable từ Law: {n}/{total}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--reset",
        action="store_true",
        help="Xoá toàn bộ nodes + edges trước khi load (constraints giữ).",
    )
    p.add_argument(
        "--apply-schema",
        action="store_true",
        help="Chạy schema.cypher trước khi load (idempotent, IF NOT EXISTS).",
    )
    p.add_argument("--skip-embeddings", action="store_true", help="Không load embeddings.")
    p.add_argument(
        "--sanity-only", action="store_true", help="Chỉ chạy sanity queries, không load gì."
    )
    args = p.parse_args()

    if not all([URI, USER, PWD]):
        print("FAIL: thiếu NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD trong .env", file=sys.stderr)
        return 1

    print(f"Connecting to {URI} as {USER}, db={DB} ...")
    with GraphDatabase.driver(URI, auth=(USER, PWD)) as driver:
        driver.verify_connectivity()
        print("  ✓ Connected")

        if args.sanity_only:
            sanity(driver, DB)
            return 0

        if args.reset:
            print("\n[--reset] Xoá toàn bộ nodes + edges...")
            with driver.session(database=DB) as s:
                # Xoá theo batch để tránh OOM với lượng node lớn
                while True:
                    n = s.run("""
                        MATCH (n) WITH n LIMIT 1000
                        DETACH DELETE n
                        RETURN count(n) AS c
                    """).single()["c"]
                    if n == 0:
                        break
            print("  ✓ Done reset")

        if args.apply_schema:
            apply_schema(driver, DB)

        if not GRAPH_PATH.exists():
            print(f"FAIL: thiếu {GRAPH_PATH}. Chạy B4 trước.", file=sys.stderr)
            return 1
        print(f"\nLoading {GRAPH_PATH}...")
        with GRAPH_PATH.open(encoding="utf-8") as f:
            graph = json.load(f)

        t0 = time.time()
        load_nodes(driver, DB, graph)
        load_edges(driver, DB, graph)
        if not args.skip_embeddings:
            load_embeddings(driver, DB)
        print(f"\nLoad time: {time.time() - t0:.1f}s")

        sanity(driver, DB)

    print("\n=== DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
