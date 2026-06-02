"""Load validated Phase 6 Prolog extractions into Neo4j."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src import ids
from src.legal_metadata import load_law_metadata
from src.prolog_utils import (
    canonical_predicate,
    extract_predicates,
    load_predicate_aliases,
    namespace_predicate_name,
)

load_dotenv()

ROOT = Path("data/eval/extracted_prolog")
LOGIC_LABELS = (
    "LegalRule",
    "LegalCondition",
    "NumericalThreshold",
    "LegalTerm",
    "ProcedureStep",
    "LegalEntity",
    "CanonicalPredicate",
)
LOGIC_RELS = (
    "EXTRACTED_FROM",
    "REQUIRES",
    "EXCEPTION_OF",
    "INVOLVES_ENTITY",
    "NEXT_STEP",
    "DEFINES_PREDICATE",
    "USES_PREDICATE",
    "SAME_CONCEPT_AS",
)


def iter_records(root: Path, law: str | None = None):
    dirs = [root / law] if law else sorted(p for p in root.iterdir() if p.is_dir())
    for d in dirs:
        if d.exists():
            for path in sorted(d.glob("*.json")):
                yield path, json.loads(path.read_text(encoding="utf-8"))


def make_rule_id(clause_id: str, predicate: str) -> str:
    return f"rule_{clause_id.replace('.', '_').lower()}_{ids.slug(predicate)}"


def make_node_id(prefix: str, clause_id: str, slot: str, idx: int) -> str:
    return f"{prefix}_{clause_id.replace('.', '_').lower()}_{ids.slug(slot)[:40]}_{idx}"


def clear_logic(session) -> None:
    rels = "|".join(LOGIC_RELS)
    labels = "|".join(LOGIC_LABELS)
    session.run(f"MATCH ()-[r:{rels}]-() DELETE r")
    session.run(f"MATCH (n:{labels}) DETACH DELETE n")


def load_record(session, record: dict[str, Any], laws: dict, alias_map: dict[str, str]) -> bool:
    if record.get("status") != "validated" or not record.get("validation", {}).get("ok"):
        return False

    clause_id = record["clause_id"]
    law_code = record["law_code"]
    meta = laws[law_code]
    extraction = record["extraction"]
    validation = record["validation"]
    prolog_full = validation.get("prolog_source_namespaced") or extraction.get("prolog_source") or ""
    legal_sources_pl = validation.get("legal_sources_pl") or extraction.get("legal_sources_pl") or ""
    if validation.get("no_prolog_rule") or not prolog_full.strip():
        load_decomposed_view(session, record, None, alias_map)
        return True

    main_base = canonical_predicate(str(extraction.get("main_predicate") or "legal_rule"), alias_map)
    main_arity = int(extraction.get("main_arity") or 0)
    main_namespaced = namespace_predicate_name(main_base, law_code, alias_map)
    rule_id = make_rule_id(clause_id, main_base)
    predicates = extract_predicates(prolog_full)

    session.run(
        """
        MATCH (cl:Clause {id: $clause_id})
        MERGE (r:LegalRule {id: $rule_id})
        SET r.name = $name,
            r.conclusion = $main_base,
            r.conclusion_value = 'true',
            r.conclusion_type = 'prolog',
            r.confidence = $confidence,
            r.is_atomic = false,
            r.law_code = $law_code,
            r.effective_from = CASE WHEN $effective_from IS NULL THEN NULL ELSE date($effective_from) END,
            r.effective_until = CASE WHEN $effective_until IS NULL THEN NULL ELSE date($effective_until) END,
            r.version = coalesce(r.version, 0) + 1,
            r.prolog_full = $prolog_full,
            r.prolog_head = $prolog_head,
            r.prolog_body = $prolog_body,
            r.legal_sources_pl = $legal_sources_pl,
            r.source_clause_id = $clause_id
        MERGE (r)-[:EXTRACTED_FROM]->(cl)
        MERGE (cp:CanonicalPredicate {id: $predicate_id})
        SET cp.namespaced_name = $predicate_id,
            cp.base_name = $main_base,
            cp.law_code = $law_code,
            cp.arity = $main_arity
        MERGE (r)-[:DEFINES_PREDICATE]->(cp)
        """,
        clause_id=clause_id,
        rule_id=rule_id,
        name=f"{main_base} from {clause_id}",
        main_base=main_base,
        main_arity=main_arity,
        confidence=float(extraction.get("extractor_confidence") or 0.0),
        law_code=law_code,
        effective_from=meta.effective_date,
        effective_until=meta.repealed_date,
        prolog_full=prolog_full,
        prolog_head=prolog_full.split(":-", 1)[0].strip()[:500],
        prolog_body=(prolog_full.split(":-", 1)[1].strip() if ":-" in prolog_full else "")[:2000],
        legal_sources_pl=legal_sources_pl,
        predicate_id=main_namespaced,
    )

    for pred in predicates:
        base = canonical_predicate(pred["name"].rsplit(f"_{law_code.lower()}", 1)[0], alias_map)
        namespaced = namespace_predicate_name(base, law_code, alias_map)
        session.run(
            """
            MATCH (r:LegalRule {id: $rule_id})
            MERGE (cp:CanonicalPredicate {id: $predicate_id})
            SET cp.namespaced_name = $predicate_id,
                cp.base_name = $base,
                cp.law_code = $law_code,
                cp.arity = $arity
            MERGE (r)-[:USES_PREDICATE]->(cp)
            WITH cp
            MATCH (other:CanonicalPredicate {base_name: $base})
            WHERE other.id <> cp.id
            MERGE (cp)-[:SAME_CONCEPT_AS]-(other)
            """,
            rule_id=rule_id,
            predicate_id=namespaced,
            base=base,
            law_code=law_code,
            arity=int(pred.get("arity") or 0),
        )

    load_decomposed_view(session, record, rule_id, alias_map)
    return True


def load_decomposed_view(
    session,
    record: dict[str, Any],
    rule_id: str | None,
    alias_map: dict[str, str],
) -> None:
    extraction = record["extraction"]
    clause_id = record["clause_id"]
    law_code = record["law_code"]
    view = extraction.get("decomposed_view") or {}

    for i, cond in enumerate(view.get("conditions") or []):
        if not isinstance(cond, dict):
            cond = {"predicate": "condition", "description_vi": str(cond)}
        predicate = canonical_predicate(str(cond.get("predicate") or "condition"), alias_map)
        cid = make_node_id("cond", clause_id, predicate, i)
        query = """
            MATCH (cl:Clause {id: $clause_id})
            MERGE (c:LegalCondition {id: $id})
            SET c.predicate = $predicate,
                c.operator = $operator,
                c.value = $value,
                c.unit = $unit,
                c.description_vi = $desc,
                c.law_code = $law_code,
                c.source_clause_id = $clause_id
            MERGE (c)-[:EXTRACTED_FROM]->(cl)
        """
        if rule_id:
            query += """
            WITH c
            MATCH (r:LegalRule {id: $rule_id})
            MERGE (r)-[req:REQUIRES]->(c)
            SET req.source_clause = $clause_id
            """
        session.run(
            query,
            clause_id=clause_id,
            rule_id=rule_id,
            id=cid,
            predicate=predicate,
            operator=str(cond.get("operator") or "="),
            value=json.dumps(cond.get("value"), ensure_ascii=False),
            unit=str(cond.get("unit") or ""),
            desc=str(cond.get("description_vi") or ""),
            law_code=law_code,
        )

    for i, threshold in enumerate(view.get("thresholds") or []):
        if not isinstance(threshold, dict):
            threshold = {"context": str(threshold), "value": 0}
        tid = make_node_id("thr", clause_id, str(threshold.get("context") or "threshold"), i)
        session.run(
            """
            MATCH (cl:Clause {id: $clause_id})
            MERGE (t:NumericalThreshold {id: $id})
            SET t.value = $value,
                t.unit = $unit,
                t.direction = $direction,
                t.context = $context,
                t.description_vi = $desc,
                t.law_code = $law_code,
                t.source_clause_id = $clause_id
            MERGE (t)-[:EXTRACTED_FROM]->(cl)
            """,
            clause_id=clause_id,
            id=tid,
            value=(
                json.dumps(threshold.get("value"), ensure_ascii=False)
                if isinstance(threshold.get("value"), (list, dict))
                else str(threshold.get("value") if threshold.get("value") is not None else "")
            ),
            unit=str(threshold.get("unit") or ""),
            direction=str(threshold.get("direction") or "exact"),
            context=str(threshold.get("context") or ""),
            desc=str(threshold.get("description_vi") or ""),
            law_code=law_code,
        )

    for i, definition in enumerate(view.get("defines") or []):
        if not isinstance(definition, dict):
            definition = {"term": str(definition), "definition": ""}
        term = str(definition.get("term_vi") or definition.get("term") or "term")
        tid = make_node_id("term", clause_id, term, i)
        session.run(
            """
            MATCH (cl:Clause {id: $clause_id})
            MERGE (t:LegalTerm {id: $id})
            SET t.term_vi = $term,
                t.definition = $definition,
                t.related_predicate = $related_predicate,
                t.law_code = $law_code,
                t.source_clause_id = $clause_id
            MERGE (cl)-[:DEFINES]->(t)
            MERGE (t)-[:EXTRACTED_FROM]->(cl)
            """,
            clause_id=clause_id,
            id=tid,
            term=term,
            definition=str(definition.get("definition") or ""),
            related_predicate=str(definition.get("related_predicate") or ""),
            law_code=law_code,
        )

    for i, step in enumerate(view.get("procedure_steps") or []):
        if not isinstance(step, dict):
            step = {"action": str(step), "step_order": i + 1}
        sid = make_node_id("step", clause_id, str(step.get("action") or "step"), i)
        session.run(
            """
            MATCH (cl:Clause {id: $clause_id})
            MERGE (s:ProcedureStep {id: $id})
            SET s.step_order = $order,
                s.actor = $actor,
                s.action = $action,
                s.prerequisite = $prerequisite,
                s.law_code = $law_code,
                s.source_clause_id = $clause_id
            MERGE (s)-[:EXTRACTED_FROM]->(cl)
            """,
            clause_id=clause_id,
            id=sid,
            order=int(step.get("step_order") or i + 1),
            actor=str(step.get("actor") or ""),
            action=str(step.get("action") or ""),
            prerequisite=str(step.get("prerequisite") or ""),
            law_code=law_code,
        )

    for ref in view.get("references") or []:
        if not isinstance(ref, dict):
            continue
        target_law = str(ref.get("law") or law_code)
        article = ref.get("article")
        if not article:
            continue
        try:
            article_n = int(article)
        except (TypeError, ValueError):
            continue
        target_id = ids.article_id(target_law, article_n)
        session.run(
            """
            MATCH (cl:Clause {id: $clause_id})
            OPTIONAL MATCH (a:Article {id: $target_id})
            FOREACH (_ IN CASE WHEN a IS NULL THEN [] ELSE [1] END |
                MERGE (cl)-[r:REFERS_TO {source_clause: $clause_id, law: $target_law}]->(a)
                SET r.external_article = $article
            )
            """,
            clause_id=clause_id,
            target_id=target_id,
            target_law=target_law,
            article=article_n,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--law", default=None)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    db = os.getenv("NEO4J_DATABASE", "neo4j")
    if not all([uri, user, password]):
        raise SystemExit("FAIL: missing NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD")

    from neo4j import GraphDatabase

    laws = load_law_metadata()
    alias_map = load_predicate_aliases()
    loaded = 0
    skipped = 0
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=db) as session:
            if args.clear:
                clear_logic(session)
            for _path, record in iter_records(Path(args.root), args.law):
                if load_record(session, record, laws, alias_map):
                    loaded += 1
                else:
                    skipped += 1
    print(json.dumps({"loaded": loaded, "skipped": skipped}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
