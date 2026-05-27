"""load_extracted_logic.py — Push extracted JSON facts vào Neo4j.

Reads data/eval/extracted_logic/*.json (output of extract_logic.py)
và creates new node types + edges per logic_extraction_schema.md §3.

Idempotent (MERGE) — re-run nào không duplicate.

Pre-conditions:
- Existing KG already loaded (Article + Clause nodes exist)
- Neo4j env vars set (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

CLI:
    python -m experiments.load_extracted_logic --dry-run    # preview Cypher
    python -m experiments.load_extracted_logic              # actually load
    python -m experiments.load_extracted_logic --clear      # delete all extracted nodes first
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

EXTRACT_DIR = Path("data/eval/extracted_logic")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

EXTRACTED_LABELS = ("LegalCondition", "NumericalThreshold", "LegalRule",
                    "LegalTerm", "ProcedureStep")
EXTRACTED_ENTITY_LABEL = "LegalEntity"


def make_id(prefix: str, clause_id: str, slot: str, idx: int) -> str:
    """Deterministic ID — idempotent across re-runs."""
    short = clause_id.replace("L41_2024.", "").replace(".", "_").lower()
    return f"{prefix}_{short}_{slot}_{idx}"


def safe_str(v) -> str:
    if v is None: return ""
    if isinstance(v, (list, dict)): return json.dumps(v, ensure_ascii=False)
    return str(v)


def build_cypher_for_clause(rec: dict) -> list[tuple[str, dict]]:
    """Returns list of (cypher, params) tuples cho 1 extracted clause."""
    clause_id = rec["clause_id"]
    extr = rec.get("extraction", {})
    if not extr or extr.get("_parse_error"):
        return []

    queries = []

    # Conditions
    cond_id_map = {}  # idx → node_id
    for i, c in enumerate(extr.get("conditions", [])):
        cid = make_id("cond", clause_id, c.get("predicate", "unk"), i)
        cond_id_map[i] = cid
        queries.append((
            """
            MATCH (cl:Clause {id: $clause_id})
            MERGE (c:LegalCondition {id: $id})
            SET c.predicate = $predicate,
                c.operator  = $operator,
                c.value     = $value,
                c.unit      = $unit,
                c.description_vi = $desc,
                c.source_clause_id = $clause_id
            MERGE (c)-[:EXTRACTED_FROM]->(cl)
            """,
            {
                "id": cid,
                "clause_id": clause_id,
                "predicate": c.get("predicate", ""),
                "operator": c.get("operator", "="),
                "value": safe_str(c.get("value")),
                "unit": c.get("unit", ""),
                "desc": c.get("description_vi", ""),
            },
        ))

    # Thresholds
    thr_id_map = {}
    for i, t in enumerate(extr.get("thresholds", [])):
        tid = make_id("thr", clause_id, t.get("context", "n")[:20], i)
        thr_id_map[i] = tid
        queries.append((
            """
            MATCH (cl:Clause {id: $clause_id})
            MERGE (t:NumericalThreshold {id: $id})
            SET t.value = $value,
                t.unit = $unit,
                t.direction = $direction,
                t.context = $context,
                t.description_vi = $desc,
                t.source_clause_id = $clause_id
            MERGE (t)-[:EXTRACTED_FROM]->(cl)
            """,
            {
                "id": tid,
                "clause_id": clause_id,
                "value": float(t["value"]) if isinstance(t.get("value"), (int, float)) else 0.0,
                "unit": t.get("unit", ""),
                "direction": t.get("direction", "exact"),
                "context": t.get("context", ""),
                "desc": t.get("description_vi", ""),
            },
        ))

    # Rules
    rule_id_map = {}
    for i, r in enumerate(extr.get("rules", [])):
        rid = make_id("rule", clause_id, r.get("then_predicate", "unk"), i)
        rule_id_map[i] = rid
        queries.append((
            """
            MATCH (cl:Clause {id: $clause_id})
            MERGE (r:LegalRule {id: $id})
            SET r.name = $name,
                r.conclusion = $conclusion,
                r.conclusion_value = $conclusion_value,
                r.conclusion_type = $conclusion_type,
                r.source_clause_id = $clause_id,
                r.confidence = $confidence,
                r.is_atomic = $is_atomic
            MERGE (r)-[:EXTRACTED_FROM]->(cl)
            """,
            {
                "id": rid,
                "clause_id": clause_id,
                "name": r.get("name", ""),
                "conclusion": r.get("then_predicate", ""),
                "conclusion_value": safe_str(r.get("then_value")),
                "conclusion_type": r.get("conclusion_type", "boolean"),
                "confidence": float(extr.get("extractor_confidence", 0.5)),
                "is_atomic": len(r.get("if_conditions_idx", [])) == 0,
            },
        ))
        # REQUIRES edges
        for cond_idx in r.get("if_conditions_idx", []):
            if cond_idx in cond_id_map:
                queries.append((
                    """
                    MATCH (r:LegalRule {id: $rid})
                    MATCH (c:LegalCondition {id: $cid})
                    MERGE (r)-[req:REQUIRES]->(c)
                    SET req.source_clause = r.source_clause_id
                    """,
                    {"rid": rid, "cid": cond_id_map[cond_idx]},
                ))
        # INVOLVES_ENTITY edges
        for ent in r.get("involves_entities", []):
            queries.append((
                """
                MATCH (r:LegalRule {id: $rid})
                MERGE (e:LegalEntity {abbreviation: $abbr})
                ON CREATE SET e.id = 'entity_' + $abbr, e.type = 'inferred'
                MERGE (r)-[:INVOLVES_ENTITY]->(e)
                """,
                {"rid": rid, "abbr": ent},
            ))

    # Exceptions
    for i, exc in enumerate(extr.get("exceptions", [])):
        of_rule_idx = exc.get("of_rule_idx")
        if of_rule_idx is None or of_rule_idx not in rule_id_map:
            continue
        exc_id = make_id("rule_exc", clause_id, "exc", i)
        queries.append((
            """
            MATCH (parent:LegalRule {id: $parent_id})
            MATCH (cl:Clause {id: $clause_id})
            MERGE (exc:LegalRule {id: $exc_id})
            SET exc.name = $name,
                exc.conclusion = 'exception',
                exc.modifies = $modifies,
                exc.source_clause_id = $clause_id,
                exc.is_atomic = true
            MERGE (exc)-[:EXTRACTED_FROM]->(cl)
            MERGE (exc)-[:EXCEPTION_OF]->(parent)
            """,
            {
                "exc_id": exc_id,
                "parent_id": rule_id_map[of_rule_idx],
                "clause_id": clause_id,
                "name": exc.get("condition_description", "")[:200],
                "modifies": exc.get("modifies", ""),
            },
        ))

    # References — create :REFERS_TO edges to target Articles
    for ref in extr.get("references", []):
        article_n = ref.get("article")
        if not article_n: continue
        law = ref.get("law", "L41_2024")
        if law != "L41_2024":
            # External law refs (e.g. BLLĐ) — skip for now (would need separate KG)
            continue
        target_id = f"L41_2024.A{article_n}"
        queries.append((
            """
            MATCH (cl:Clause {id: $clause_id})
            OPTIONAL MATCH (a:Article {id: $target_id})
            FOREACH (_ IN CASE WHEN a IS NOT NULL THEN [1] ELSE [] END |
                MERGE (cl)-[:REFERS_TO]->(a)
            )
            """,
            {"clause_id": clause_id, "target_id": target_id},
        ))

    # Defines
    for i, d in enumerate(extr.get("defines", [])):
        term_id = make_id("term", clause_id, "def", i)
        queries.append((
            """
            MATCH (cl:Clause {id: $clause_id})
            MERGE (t:LegalTerm {id: $tid})
            SET t.term_vi = $term_vi,
                t.definition = $definition,
                t.related_predicate = $rel_pred,
                t.source_clause_id = $clause_id
            MERGE (cl)-[:DEFINES]->(t)
            MERGE (t)-[:EXTRACTED_FROM]->(cl)
            """,
            {
                "tid": term_id,
                "clause_id": clause_id,
                "term_vi": d.get("term_vi", ""),
                "definition": d.get("definition", ""),
                "rel_pred": d.get("related_predicate") or "",
            },
        ))

    # Procedure steps — chain them
    prev_step_id = None
    for i, p in enumerate(extr.get("procedure_steps", [])):
        sid = make_id("step", clause_id, f"s{p.get('step_order',i)}", i)
        queries.append((
            """
            MATCH (cl:Clause {id: $clause_id})
            MERGE (s:ProcedureStep {id: $sid})
            SET s.step_order = $step_order,
                s.actor = $actor,
                s.action = $action,
                s.prerequisite = $prereq,
                s.source_clause_id = $clause_id
            MERGE (s)-[:EXTRACTED_FROM]->(cl)
            """,
            {
                "sid": sid,
                "clause_id": clause_id,
                "step_order": int(p.get("step_order", i)),
                "actor": p.get("actor", ""),
                "action": p.get("action", ""),
                "prereq": p.get("prerequisite") or "",
            },
        ))
        if prev_step_id:
            queries.append((
                "MATCH (a:ProcedureStep {id: $a}) MATCH (b:ProcedureStep {id: $b}) MERGE (a)-[:NEXT_STEP]->(b)",
                {"a": prev_step_id, "b": sid},
            ))
        prev_step_id = sid

    return queries


def clear_extracted(session):
    """Delete all extracted nodes (preserve original KG)."""
    for label in EXTRACTED_LABELS:
        session.run(f"MATCH (n:{label}) DETACH DELETE n")
    # Don't delete LegalEntity (rarely changes) unless force
    session.run("MATCH (n:LegalEntity) WHERE n.type = 'inferred' DETACH DELETE n")
    print("Cleared all extracted nodes")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--clear", action="store_true",
                   help="Delete existing extracted nodes first")
    args = p.parse_args()

    files = sorted(EXTRACT_DIR.glob("*.json"))
    if not files:
        print(f"No extracted JSON in {EXTRACT_DIR}", file=sys.stderr)
        return 1

    print(f"Loading {len(files)} extracted clauses...")

    if args.dry_run:
        # Just print Cypher count
        total = 0
        for fp in files[:3]:
            rec = json.loads(fp.read_text(encoding="utf-8"))
            qs = build_cypher_for_clause(rec)
            total += len(qs)
            print(f"  {rec['clause_id']}: {len(qs)} queries")
        avg = total / 3 if files else 0
        print(f"Average ~{avg:.0f} queries per clause × {len(files)} files = ~{int(avg*len(files))} total")
        print("\nDRY RUN — no DB changes")
        return 0

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session(database=NEO4J_DATABASE) as session:
        if args.clear:
            clear_extracted(session)

        total_q = 0
        n_done = 0
        for fp in files:
            try:
                rec = json.loads(fp.read_text(encoding="utf-8"))
                queries = build_cypher_for_clause(rec)
                for cypher, params in queries:
                    session.run(cypher, **params)
                total_q += len(queries)
                n_done += 1
                if n_done % 50 == 0:
                    print(f"  [{n_done}/{len(files)}] {total_q} queries executed")
            except Exception as e:
                print(f"  ✗ {fp.name}: {type(e).__name__}: {e}", file=sys.stderr)

        # Final counts
        print(f"\nDone: {n_done}/{len(files)} clauses, {total_q} queries\n")
        for label in EXTRACTED_LABELS + (EXTRACTED_ENTITY_LABEL,):
            count = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            print(f"  :{label}: {count}")

    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
