"""Build a FULL semantic ontology snapshot from the live Neo4j KG.

Unlike :mod:`offline.build_ontology` — which builds an L41-only, 32-concept
ontology from the ``structured_law.json`` *text* corpus and never touches the
KG's extracted semantic layer — this exporter reads the **current Neo4j graph**
and emits every semantic node + edge across **all loaded laws**, with
clause/article provenance.

It writes a NEW artifact ``data/ontology/ontology_kg_full.json`` and DOES NOT
touch ``data/ontology/ontology_2024.json`` (the canonical file the
``logic_lm_ontology`` arm + the frozen exp01 baseline depend on). Zero
blast-radius on existing experiments.

Scope (a node/edge is "semantic" iff it is not purely structural):

    STRUCTURAL nodes (excluded): Law, Chapter, Section, Article, Clause,
                                 Point, Table
    STRUCTURAL edges (excluded): HAS_CHAPTER, HAS_SECTION, HAS_ARTICLE,
                                 IN_SECTION, HAS_CLAUSE, HAS_POINT, HAS_TABLE,
                                 BELONGS_TO, NEXT

Everything else is exported: LegalConcept / Subject / Benefit / Obligation /
Right / Fund / Organization / Role / ProhibitedAct / LegalTerm / LegalRule /
LegalCondition / NumericalThreshold / ProcedureStep / CanonicalPredicate /
ExternalLaw / LegalEntity nodes, and the DEFINES / ENTITLED_TO / HAS_OBLIGATION
/ HAS_RIGHT / APPLIES_TO / REQUIRES / PAID_FROM / MANAGES / RESPONSIBLE_FOR /
PROHIBITED_BY / EXTRACTED_FROM / INVOLVES_ENTITY / USES_PREDICATE /
DEFINES_PREDICATE / SAME_CONCEPT_AS / REFERENCES / REFERS_TO / CITES_EXTERNAL /
AMENDS / REPEALS / REPLACES / TRANSITIONS_FROM edges (with source_clause +
source_text provenance when present).

Read-only: opens the session with ``default_access_mode=READ_ACCESS``.

Run:
    python -m offline.build_ontology_kg
    python -m offline.build_ontology_kg --out data/ontology/ontology_kg_full.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

load_dotenv()
# Defensive: a blank OPENAI_BASE_URL would break unrelated SDKs; harmless here.
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")
DB = os.getenv("NEO4J_DATABASE", "neo4j")

OUT_DEFAULT = _REPO / "data" / "ontology" / "ontology_kg_full.json"

STRUCTURAL_NODE_LABELS = ["Law", "Chapter", "Section", "Article", "Clause", "Point", "Table"]
STRUCTURAL_EDGE_TYPES = [
    "HAS_CHAPTER", "HAS_SECTION", "HAS_ARTICLE", "IN_SECTION",
    "HAS_CLAUSE", "HAS_POINT", "HAS_TABLE", "BELONGS_TO", "NEXT",
]

# Property guards: never serialise embeddings or other large float vectors.
_DROP_PROP_KEYS = {"embedding", "embeddings", "vector", "vec"}


def _clean_props(props: dict) -> dict:
    out: dict = {}
    for k, v in (props or {}).items():
        if k in _DROP_PROP_KEYS:
            continue
        # Drop long numeric vectors that may have slipped onto a semantic node.
        if isinstance(v, list) and len(v) > 64 and all(isinstance(x, (int, float)) for x in v):
            out[k] = f"<{len(v)}-dim vector dropped>"
            continue
        out[k] = v
    return out


def _law_of(node_id) -> str | None:
    if isinstance(node_id, str) and "." in node_id:
        return node_id.split(".", 1)[0]
    return None


def build(driver) -> dict:
    from neo4j import READ_ACCESS

    with driver.session(database=DB, default_access_mode=READ_ACCESS) as s:
        # --- laws + structural counts (for the header + verification) ---
        law_nodes = [
            {"id": r["id"], "properties": _clean_props(r["props"])}
            for r in s.run("MATCH (l:Law) RETURN l.id AS id, properties(l) AS props ORDER BY l.id")
        ]
        art_by_law = {r["law"]: r["c"] for r in s.run(
            "MATCH (a:Article) RETURN split(a.id,'.')[0] AS law, count(*) AS c")}
        cla_by_law = {r["law"]: r["c"] for r in s.run(
            "MATCH (c:Clause) RETURN split(c.id,'.')[0] AS law, count(*) AS c")}
        laws = []
        for law in sorted(set(art_by_law) | set(cla_by_law)):
            props = next((ln["properties"] for ln in law_nodes if ln["id"] == law), {})
            laws.append({
                "id": law,
                "title": props.get("title") or props.get("name") or props.get("short_name") or "",
                "n_articles": art_by_law.get(law, 0),
                "n_clauses": cla_by_law.get(law, 0),
            })

        node_label_counts = {r["label"]: r["c"] for r in s.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY c DESC")}
        edge_type_counts = {r["t"]: r["c"] for r in s.run(
            "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC")}

        # --- clause -> article map (authoritative provenance lift) ---
        clause_article = {r["clause"]: r["article"] for r in s.run(
            "MATCH (a:Article)-[:HAS_CLAUSE]->(c:Clause) RETURN c.id AS clause, a.id AS article")}

        # --- semantic nodes (+ EXTRACTED_FROM / DEFINES provenance) ---
        node_rows = s.run(
            """
            MATCH (n) WHERE NOT any(l IN labels(n) WHERE l IN $structural)
            OPTIONAL MATCH (n)-[:EXTRACTED_FROM]->(ce:Clause)
            OPTIONAL MATCH (cd:Clause)-[:DEFINES]->(n)
            WITH n,
                 [x IN collect(DISTINCT ce.id) + collect(DISTINCT cd.id)
                    WHERE x IS NOT NULL] AS prov
            RETURN n.id AS id, labels(n) AS labels, properties(n) AS props, prov AS prov_clauses
            """,
            structural=STRUCTURAL_NODE_LABELS,
        ).data()

        # --- semantic edges (with provenance) ---
        edge_rows = s.run(
            """
            MATCH (a)-[r]->(b) WHERE NOT type(r) IN $struct_edges
            RETURN type(r) AS type, a.id AS src, labels(a) AS src_labels,
                   b.id AS dst, labels(b) AS dst_labels,
                   r.source_clause AS source_clause, r.source_text AS source_text
            """,
            struct_edges=STRUCTURAL_EDGE_TYPES,
        ).data()

    # provenance from incident edges' source_clause, keyed by node id
    edge_prov: dict[str, set] = defaultdict(set)
    edges: list[dict] = []
    for e in edge_rows:
        sc = e.get("source_clause")
        if sc:
            if e.get("src"):
                edge_prov[e["src"]].add(sc)
            if e.get("dst"):
                edge_prov[e["dst"]].add(sc)
        edges.append({
            "type": e["type"],
            "src": e["src"], "src_labels": e["src_labels"],
            "dst": e["dst"], "dst_labels": e["dst_labels"],
            "source_clause": sc,
            "source_text": e.get("source_text"),
        })

    nodes: list[dict] = []
    for r in node_rows:
        props = _clean_props(r["props"])
        prov = set(r["prov_clauses"] or [])
        prov |= edge_prov.get(r["id"], set())
        mi = props.get("mentioned_in")
        if isinstance(mi, list):
            prov |= {x for x in mi if isinstance(x, str)}
        clause_ids = sorted(c for c in prov if c)
        article_ids = sorted({clause_article[c] for c in clause_ids if c in clause_article})
        nodes.append({
            "id": r["id"],
            "labels": r["labels"],
            "properties": props,
            "mentioned_in_clauses": clause_ids,
            "article_ids": article_ids,
            "laws": sorted({_law_of(a) for a in article_ids if _law_of(a)}),
        })

    return {
        "type": "kg_semantic_ontology",
        "version": 1,
        "source": f"{URI}/{DB}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Full semantic layer exported from the live Neo4j KG across all "
            "loaded laws. Companion to (not a replacement for) ontology_2024.json."
        ),
        "laws": laws,
        "kg_counts": {"nodes": node_label_counts, "edges": edge_type_counts},
        "semantic_counts": {
            "nodes": len(nodes),
            "edges": len(edges),
            "node_labels": _count_labels(nodes),
            "edge_types": _count_edge_types(edges),
        },
        "nodes": nodes,
        "edges": edges,
    }


def _count_labels(nodes: list[dict]) -> dict:
    c: dict[str, int] = defaultdict(int)
    for n in nodes:
        for lab in n["labels"]:
            c[lab] += 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def _count_edge_types(edges: list[dict]) -> dict:
    c: dict[str, int] = defaultdict(int)
    for e in edges:
        c[e["type"]] += 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    args = p.parse_args()

    if not URI:
        print("FAIL: NEO4J_URI not set (check .env).", file=sys.stderr)
        return 1

    from neo4j import GraphDatabase

    print(f"Connecting Neo4j {URI} (db={DB}) ...", flush=True)
    driver = GraphDatabase.driver(URI, auth=(USER, PWD))
    try:
        driver.verify_connectivity()
        payload = build(driver)
    finally:
        driver.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    # default=str: stringify Neo4j temporal types (Date/DateTime/Duration) and
    # any other non-JSON-native value to its ISO/str form.
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    os.replace(tmp, out_path)

    sc = payload["semantic_counts"]
    print(f"\n  OK → {out_path}  ({out_path.stat().st_size/1_048_576:.2f} MB)")
    print(f"  laws ({len(payload['laws'])}): "
          + ", ".join(f"{l['id']}({l['n_articles']}A/{l['n_clauses']}C)" for l in payload['laws']))
    print(f"  semantic nodes = {sc['nodes']}   semantic edges = {sc['edges']}")
    print("  node labels :", json.dumps(sc["node_labels"], ensure_ascii=False))
    print("  edge types  :", json.dumps(sc["edge_types"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
