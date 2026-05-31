"""B7 — RAG query layer.

Pipeline:
1. Câu hỏi → embed bằng BGE-M3 (cùng model B5).
2. Vector search top-K Clause trong Neo4j (`clause_vec` index).
3. Expand graph quanh các Clause: Article cha + semantic edges
   (ENTITLED_TO/HAS_OBLIGATION/REQUIRES/...) anchor về cùng Clause,
   + REFERENCES nội bộ + CITES_EXTERNAL.
4. Build context (giữ Clause.id để LLM dùng làm citation).
5. Gọi GPT-4o-mini với SYSTEM_PROMPT: chỉ trả lời theo CONTEXT, kèm
   citation dạng [Điều X khoản Y].
6. Parse citation từ answer + map ngược về Clause/Article trong DB
   để verify (provenance roundtrip).

CLI:
    python -m runtime.rag_query --question "Khi nào được hưởng lương hưu?"
    python -m runtime.rag_query -q "..." -v --top-k 10
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from src.prompts import load_prompt

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Defensive: nếu OPENAI_BASE_URL được set rỗng, pop để SDK dùng default api.openai.com
# (nếu giữ empty string, SDK sẽ dùng nó làm URL → APIConnectionError)
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")
DB = os.getenv("NEO4J_DATABASE", "neo4j")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SEMANTIC_EDGE_TYPES = [
    "ENTITLED_TO",
    "HAS_OBLIGATION",
    "HAS_RIGHT",
    "REQUIRES",
    "APPLIES_TO",
    "PAID_FROM",
    "MANAGES",
    "RESPONSIBLE_FOR",
    "PROHIBITED_BY",
    "DEFINES",
]


SYSTEM_PROMPT = load_prompt("runtime/graphrag_system.md")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SearchHit:
    clause_id: str
    article_id: str
    article_n: int
    article_title: str
    clause_n: int
    text: str
    score: float


@dataclass
class RagAnswer:
    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    citation_ids: list[str] = field(default_factory=list)
    hits: list[SearchHit] = field(default_factory=list)
    n_semantic_edges: int = 0
    n_refs: int = 0
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Phase 3 — Logic-aware retrieval (logic_extraction integration)
# ---------------------------------------------------------------------------


@dataclass
class ExtractedFact:
    """Single piece of extracted logic attached to a Clause.

    kind ∈ {'condition', 'rule', 'threshold', 'definition', 'procedure_step'}.
    payload holds kind-specific fields preserved verbatim từ Neo4j node.
    """

    clause_id: str
    kind: str
    payload: dict[str, Any]


@dataclass
class HybridResult:
    """Combined semantic + logic retrieval output.

    - hits: vector-search clauses (raw text, current GraphRAG behavior)
    - facts: pre-extracted LegalRule/LegalCondition/NumericalThreshold/etc
      attached to those clauses (Phase 2 outputs)
    - referenced: multi-hop REFERS_TO expansion (Article-level neighbors)
    - n_facts_by_kind: counts cho quick inspection
    """

    hits: list[SearchHit] = field(default_factory=list)
    facts: list[ExtractedFact] = field(default_factory=list)
    referenced: list[dict] = field(default_factory=list)
    n_facts_by_kind: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class RagPipeline:
    def __init__(self):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(URI, auth=(USER, PWD))
        self.driver.verify_connectivity()
        self._embed_model = None
        self._openai = None

    @property
    def embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer

            print(f"Loading {EMBED_MODEL} on {EMBED_DEVICE}...", file=sys.stderr)
            t0 = time.time()
            self._embed_model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
            print(f"  loaded in {time.time() - t0:.1f}s", file=sys.stderr)
        return self._embed_model

    @property
    def openai(self):
        if self._openai is None:
            from openai import OpenAI

            self._openai = OpenAI()
        return self._openai

    # ------------------------------------------------------------------

    def vector_search(self, query: str, top_k: int = 8) -> list[SearchHit]:
        # "query: " prefix cho BGE-M3 asymmetric retrieval.
        # Empirical test: +0.013 cosine gap so với no-prefix trên cùng passage vectors.
        q_emb = self.embed_model.encode(
            ["query: " + query], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()
        return self.vector_search_by_vector(q_emb, top_k=top_k)

    def vector_search_by_vector(self, qvec, top_k: int = 8) -> list[SearchHit]:
        """Same clause_vec search as :meth:`vector_search` but with a
        precomputed query vector — lets callers swap the raw-question
        embedding for an alternative (e.g. a HyDE hypothetical-doc
        embedding) without changing the index path."""
        vec = qvec.tolist() if hasattr(qvec, "tolist") else list(qvec)
        with self.driver.session(database=DB) as s:
            rows = s.run(
                """
                CALL db.index.vector.queryNodes('clause_vec', $k, $q)
                YIELD node, score
                MATCH (a:Article)-[:HAS_CLAUSE]->(node)
                RETURN node.id AS clause_id, node.text AS text, score,
                       a.id AS article_id, a.title AS article_title,
                       a.number AS article_n, node.number AS clause_n
                ORDER BY score DESC
            """,
                k=top_k,
                q=vec,
            ).data()
        return [SearchHit(**r) for r in rows]

    def expand(self, clause_ids: list[str]) -> dict:
        """Lấy semantic edges + refs neo về các clause này."""
        with self.driver.session(database=DB) as s:
            edges = s.run(
                """
                MATCH (a)-[r]->(b)
                WHERE r.source_clause IN $cids
                  AND type(r) IN $types
                RETURN type(r) AS type,
                       coalesce(a.name, a.id) AS src_name,
                       labels(a)[0] AS src_label,
                       coalesce(b.name, b.description, b.id) AS dst_name,
                       labels(b)[0] AS dst_label,
                       r.source_clause AS source_clause,
                       r.source_text AS evidence
                LIMIT 30
            """,
                cids=clause_ids,
                types=SEMANTIC_EDGE_TYPES,
            ).data()

            refs = s.run(
                """
                MATCH (n)-[r:REFERENCES|CITES_EXTERNAL]->(m)
                WHERE r.source_clause IN $cids
                RETURN type(r) AS type, n.id AS src, m.id AS dst,
                       coalesce(m.title, m.id) AS dst_label, r.span AS span,
                       r.source_clause AS source_clause
                LIMIT 20
            """,
                cids=clause_ids,
            ).data()
        return {"edges": edges, "refs": refs}

    # ------------------------------------------------------------------
    # Phase 3 — Logic-aware methods (extracted facts + multi-hop)
    # ------------------------------------------------------------------

    def fetch_facts(self, clause_ids: list[str]) -> list[ExtractedFact]:
        """Fetch all Phase 2 extracted facts attached to given clauses.

        Returns conditions, rules (with their REQUIRES conditions inlined),
        thresholds, definitions, and procedure steps. Order: rules first
        (most informative), then conditions, thresholds, definitions, steps.
        """
        if not clause_ids:
            return []

        facts: list[ExtractedFact] = []
        with self.driver.session(database=DB) as s:
            # Rules with their requires-conditions inlined
            rule_rows = s.run(
                """
                MATCH (r:LegalRule)-[:EXTRACTED_FROM]->(cl:Clause)
                WHERE cl.id IN $cids
                OPTIONAL MATCH (r)-[:REQUIRES]->(c:LegalCondition)
                OPTIONAL MATCH (r)-[:INVOLVES_ENTITY]->(e:LegalEntity)
                WITH r, cl,
                     collect(DISTINCT {predicate: c.predicate, operator: c.operator,
                                       value: c.value, unit: c.unit,
                                       description_vi: c.description_vi}) AS reqs,
                     collect(DISTINCT e.abbreviation) AS entities
                RETURN cl.id AS clause_id,
                       r.id AS id, r.name AS name,
                       r.conclusion AS conclusion,
                       r.conclusion_value AS conclusion_value,
                       r.conclusion_type AS conclusion_type,
                       r.confidence AS confidence,
                       r.is_atomic AS is_atomic,
                       [x IN reqs WHERE x.predicate IS NOT NULL] AS requires,
                       [x IN entities WHERE x IS NOT NULL] AS involves_entities
                ORDER BY r.confidence DESC
                """,
                cids=clause_ids,
            ).data()
            for row in rule_rows:
                facts.append(ExtractedFact(
                    clause_id=row["clause_id"],
                    kind="rule",
                    payload={k: v for k, v in row.items() if k != "clause_id"},
                ))

            # Standalone conditions (also captured above when required by a rule,
            # but some conditions exist without rules — include all)
            cond_rows = s.run(
                """
                MATCH (c:LegalCondition)-[:EXTRACTED_FROM]->(cl:Clause)
                WHERE cl.id IN $cids
                RETURN cl.id AS clause_id, c.id AS id,
                       c.predicate AS predicate, c.operator AS operator,
                       c.value AS value, c.unit AS unit,
                       c.description_vi AS description_vi
                """,
                cids=clause_ids,
            ).data()
            for row in cond_rows:
                facts.append(ExtractedFact(
                    clause_id=row["clause_id"],
                    kind="condition",
                    payload={k: v for k, v in row.items() if k != "clause_id"},
                ))

            thr_rows = s.run(
                """
                MATCH (t:NumericalThreshold)-[:EXTRACTED_FROM]->(cl:Clause)
                WHERE cl.id IN $cids
                RETURN cl.id AS clause_id, t.id AS id,
                       t.value AS value, t.unit AS unit,
                       t.direction AS direction, t.context AS context,
                       t.description_vi AS description_vi
                """,
                cids=clause_ids,
            ).data()
            for row in thr_rows:
                facts.append(ExtractedFact(
                    clause_id=row["clause_id"],
                    kind="threshold",
                    payload={k: v for k, v in row.items() if k != "clause_id"},
                ))

            def_rows = s.run(
                """
                MATCH (cl:Clause)-[:DEFINES]->(t:LegalTerm)
                WHERE cl.id IN $cids
                RETURN cl.id AS clause_id, t.id AS id,
                       t.term_vi AS term_vi, t.definition AS definition,
                       t.related_predicate AS related_predicate
                """,
                cids=clause_ids,
            ).data()
            for row in def_rows:
                facts.append(ExtractedFact(
                    clause_id=row["clause_id"],
                    kind="definition",
                    payload={k: v for k, v in row.items() if k != "clause_id"},
                ))

            step_rows = s.run(
                """
                MATCH (p:ProcedureStep)-[:EXTRACTED_FROM]->(cl:Clause)
                WHERE cl.id IN $cids
                RETURN cl.id AS clause_id, p.id AS id,
                       p.step_order AS step_order, p.actor AS actor,
                       p.action AS action, p.prerequisite AS prerequisite
                ORDER BY p.step_order
                """,
                cids=clause_ids,
            ).data()
            for row in step_rows:
                facts.append(ExtractedFact(
                    clause_id=row["clause_id"],
                    kind="procedure_step",
                    payload={k: v for k, v in row.items() if k != "clause_id"},
                ))

        return facts

    def traverse(
        self,
        clause_ids: list[str],
        edge_type: str = "REFERS_TO",
        max_hops: int = 1,
    ) -> list[dict]:
        """Multi-hop expansion from clauses following a given edge type.

        Returns list of {source_clause_id, target_id, target_label, target_title,
        target_text, hop_distance} dicts. Edges traversed are :REFERS_TO from
        the extraction phase (clause-level cross-references between Articles).

        max_hops capped at 3 to avoid blow-up. Only positive integers.
        """
        if not clause_ids or max_hops < 1:
            return []
        max_hops = min(int(max_hops), 3)
        # Cypher does not allow parameterizing variable-length path bounds,
        # so format max_hops directly (validated to be int above).
        cypher = f"""
            UNWIND $cids AS cid
            MATCH (cl:Clause {{id: cid}})
            MATCH path = (cl)-[:{edge_type}*1..{max_hops}]->(target)
            WITH cid, target, length(path) AS hop
            RETURN DISTINCT
                cid AS source_clause_id,
                target.id AS target_id,
                labels(target)[0] AS target_label,
                coalesce(target.title, target.text, '') AS target_title,
                coalesce(target.text, '') AS target_text,
                min(hop) AS hop_distance
            ORDER BY hop_distance ASC
            LIMIT 40
        """
        with self.driver.session(database=DB) as s:
            rows = s.run(cypher, cids=clause_ids).data()
        return rows

    def logic_search(
        self,
        predicate: str | None = None,
        topic: str | None = None,
        limit: int = 20,
    ) -> list[ExtractedFact]:
        """Structured query — find rules/conditions filtered by predicate or topic.

        Example: logic_search(predicate='years_contributed') returns all
        conditions referencing that predicate, plus rules that require them.
        topic (free-text) is matched against clause text via simple CONTAINS.
        """
        clauses_q = ""
        params: dict[str, Any] = {"limit": limit}
        filters = []
        if predicate:
            filters.append("c.predicate = $predicate")
            params["predicate"] = predicate
        if topic:
            filters.append("toLower(cl.text) CONTAINS toLower($topic)")
            params["topic"] = topic
        where = "WHERE " + " AND ".join(filters) if filters else ""

        cypher = f"""
            MATCH (c:LegalCondition)-[:EXTRACTED_FROM]->(cl:Clause)
            {where}
            OPTIONAL MATCH (r:LegalRule)-[:REQUIRES]->(c)
            RETURN cl.id AS clause_id, c.id AS cond_id,
                   c.predicate AS predicate, c.operator AS operator,
                   c.value AS value, c.unit AS unit,
                   c.description_vi AS description_vi,
                   collect(DISTINCT {{rule_id: r.id, rule_name: r.name,
                                       conclusion: r.conclusion}}) AS rules
            LIMIT $limit
        """
        with self.driver.session(database=DB) as s:
            rows = s.run(cypher, **params).data()

        out: list[ExtractedFact] = []
        for row in rows:
            payload = {k: v for k, v in row.items() if k != "clause_id"}
            # Strip empty rule placeholders
            payload["rules"] = [r for r in payload.get("rules", [])
                                if r.get("rule_id")]
            out.append(ExtractedFact(
                clause_id=row["clause_id"],
                kind="condition",
                payload=payload,
            ))
        return out

    def hybrid_search(
        self,
        question: str,
        top_k: int = 8,
        max_hops: int = 1,
    ) -> HybridResult:
        """Combine semantic vector search + extracted facts + multi-hop refs.

        Layer A (multi-hop):  follow :REFERS_TO from vector hits → referenced Articles
        Layer B (symbolic):   fetch LegalRule/LegalCondition/Threshold/Term/Step
                              attached to vector-hit clauses
        Layer C (semantic):   the original vector_search hits

        Returns HybridResult — caller decides how to format for the LLM.
        """
        hits = self.vector_search(question, top_k=top_k)
        clause_ids = [h.clause_id for h in hits]
        facts = self.fetch_facts(clause_ids)
        referenced = self.traverse(clause_ids, edge_type="REFERS_TO", max_hops=max_hops)

        by_kind: dict[str, int] = {}
        for f in facts:
            by_kind[f.kind] = by_kind.get(f.kind, 0) + 1

        return HybridResult(
            hits=hits,
            facts=facts,
            referenced=referenced,
            n_facts_by_kind=by_kind,
        )

    @staticmethod
    def format_facts_for_prompt(result: HybridResult, max_chars: int = 4000) -> str:
        """Render HybridResult.facts + referenced as human-readable lines for LLM.

        Layout:
            ## FACTS extracted from retrieved clauses
            - Rule [L41_2024.A64.K1]: Điều kiện hưởng lương hưu
              conclusion: eligible_pension = True
              requires:
                * years_contributed >= 15 (year) — Đóng BHXH ≥ 15 năm
                * age >= retirement_age (year) — Đủ tuổi nghỉ hưu
            - Threshold [L41_2024.A26.K1]: 75% (exact) — 75% lương đóng BHXH
            - Definition [L41_2024.A4.K15]: "Mức bình quân tiền lương đóng BHXH" = ...

            ## REFERENCES (cross-clause)
            - L41_2024.A64.K1 → L41_2024.A28 (Điều 28 — ...)
        """
        lines: list[str] = []
        if result.facts:
            lines.append("# FACTS trích xuất từ các Điều khoản liên quan\n")
            for f in result.facts:
                p = f.payload
                if f.kind == "rule":
                    reqs = p.get("requires", [])
                    name = p.get("name", "")
                    concl = p.get("conclusion", "")
                    cval = p.get("conclusion_value", "")
                    ent = p.get("involves_entities", []) or []
                    head = (f"- Rule [{f.clause_id}] {name}: "
                            f"=> {concl}={cval}")
                    if ent:
                        head += f" (actors: {','.join(ent)})"
                    lines.append(head)
                    for r in reqs:
                        op = r.get("operator", "=")
                        val = r.get("value", "")
                        unit = r.get("unit", "") or ""
                        desc = r.get("description_vi", "") or ""
                        lines.append(
                            f"    * {r.get('predicate')} {op} {val} {unit}"
                            + (f" — {desc}" if desc else "")
                        )
                elif f.kind == "condition":
                    op = p.get("operator", "=")
                    val = p.get("value", "")
                    unit = p.get("unit", "") or ""
                    desc = p.get("description_vi", "") or ""
                    lines.append(
                        f"- Condition [{f.clause_id}]: "
                        f"{p.get('predicate')} {op} {val} {unit}"
                        + (f" — {desc}" if desc else "")
                    )
                elif f.kind == "threshold":
                    lines.append(
                        f"- Threshold [{f.clause_id}]: "
                        f"{p.get('value')} {p.get('unit')} ({p.get('direction')}) "
                        f"context={p.get('context')} — {p.get('description_vi', '')}"
                    )
                elif f.kind == "definition":
                    lines.append(
                        f"- Định nghĩa [{f.clause_id}]: \"{p.get('term_vi')}\" = "
                        f"{p.get('definition')}"
                        + (f" (predicate={p.get('related_predicate')})"
                           if p.get('related_predicate') else "")
                    )
                elif f.kind == "procedure_step":
                    lines.append(
                        f"- Bước {p.get('step_order')} [{f.clause_id}] "
                        f"({p.get('actor')}): {p.get('action')}"
                        + (f" (sau: {p.get('prerequisite')})"
                           if p.get("prerequisite") else "")
                    )

        if result.referenced:
            lines.append("\n# REFERENCES (Điều khoản được viện dẫn chéo)\n")
            for r in result.referenced:
                tgt_title = (r.get("target_title") or "")[:80]
                lines.append(
                    f"- {r['source_clause_id']} → {r['target_id']} "
                    f"(hop={r['hop_distance']}, {r['target_label']}: {tgt_title})"
                )

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n... (facts truncated)"
        return text

    # ------------------------------------------------------------------

    def build_context(self, hits: list[SearchHit], expansion: dict, max_chars: int = 7000) -> str:
        parts: list[str] = []
        parts.append("# CÁC ĐOẠN ĐIỀU LUẬT LIÊN QUAN (sắp theo relevance giảm dần)\n")
        for h in hits:
            parts.append(
                f"\n## [{h.clause_id}] Điều {h.article_n}. {h.article_title} - Khoản {h.clause_n} "
                f"(score={h.score:.3f})"
            )
            parts.append(h.text)

        if expansion["edges"]:
            parts.append("\n\n# QUAN HỆ NGỮ NGHĨA TRÍCH XUẤT")
            for e in expansion["edges"][:20]:
                parts.append(
                    f"- [{e['source_clause']}] "
                    f"{e['src_name']}({e['src_label']}) "
                    f"--{e['type']}--> "
                    f"{e['dst_name']}({e['dst_label']})"
                )
                if e.get("evidence"):
                    snip = e["evidence"][:200]
                    parts.append(f'    Bằng chứng: "{snip}"')

        if expansion["refs"]:
            parts.append("\n\n# VIỆN DẪN LIÊN QUAN")
            for r in expansion["refs"]:
                parts.append(f"- Từ {r['source_clause']}: \"{r['span']}\" → {r['dst']}")

        ctx = "\n".join(parts)
        if len(ctx) > max_chars:
            ctx = ctx[:max_chars] + "\n\n... (context truncated)"
        return ctx

    def parse_citations(self, answer: str) -> tuple[list[str], list[str]]:
        """Trả về (citations_str, citation_ids).

        Citation_ids = ID dạng L41_2024.A<X>.K<Y>[.<z>] từ "[Điều X khoản Y]".
        """
        # Pattern khớp [Điều 64], [Điều 64 khoản 1], [Điều 64 khoản 1 điểm a]
        pat = re.compile(r"\[Điều\s+(\d+)(?:\s+khoản\s+(\d+))?(?:\s+điểm\s+([a-zđ]))?\]")
        citations: list[str] = []
        ids: list[str] = []
        for m in pat.finditer(answer):
            citations.append(m.group(0))
            art, cl, pt = m.group(1), m.group(2), m.group(3)
            cid = f"L41_2024.A{art}"
            if cl:
                cid += f".K{cl}"
                if pt:
                    cid += f".{pt}"
            ids.append(cid)
        # Dedup giữ thứ tự
        return list(dict.fromkeys(citations)), list(dict.fromkeys(ids))

    def verify_citations(self, ids: list[str]) -> dict[str, bool]:
        """Kiểm tra mỗi citation ID có tồn tại thực trong Neo4j không."""
        if not ids:
            return {}
        with self.driver.session(database=DB) as s:
            rows = s.run(
                """
                UNWIND $ids AS id
                OPTIONAL MATCH (n) WHERE n.id = id AND (n:Article OR n:Clause OR n:Point)
                RETURN id, n IS NOT NULL AS exists
            """,
                ids=ids,
            ).data()
        return {r["id"]: r["exists"] for r in rows}

    # ------------------------------------------------------------------

    def ask(self, question: str, top_k: int = 8, verbose: bool = False) -> RagAnswer:
        t0 = time.time()

        hits = self.vector_search(question, top_k=top_k)
        if verbose:
            print(f"\n[vector_search] {len(hits)} hits:", file=sys.stderr)
            for h in hits[:5]:
                print(f"  {h.score:.3f}  {h.clause_id}: {h.text[:80]}", file=sys.stderr)

        clause_ids = [h.clause_id for h in hits]
        expansion = self.expand(clause_ids)
        if verbose:
            print(
                f"[expand] {len(expansion['edges'])} semantic edges, "
                f"{len(expansion['refs'])} refs",
                file=sys.stderr,
            )

        context = self.build_context(hits, expansion)
        if verbose:
            print(f"[context] {len(context)} chars", file=sys.stderr)

        resp = self.openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\n---\n\nCÂU HỎI: {question}"},
            ],
            temperature=0,
        )
        answer_text = resp.choices[0].message.content or ""

        citations, citation_ids = self.parse_citations(answer_text)

        return RagAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            citation_ids=citation_ids,
            hits=hits,
            n_semantic_edges=len(expansion["edges"]),
            n_refs=len(expansion["refs"]),
            elapsed_s=round(time.time() - t0, 2),
        )

    def close(self):
        self.driver.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-q", "--question", required=True)
    p.add_argument("-k", "--top-k", type=int, default=8)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--verify", action="store_true", help="Kiểm tra citations có thật trong DB không."
    )
    args = p.parse_args()

    if not all([URI, USER, PWD]):
        print("FAIL: thiếu NEO4J_* env vars", file=sys.stderr)
        return 1

    pipeline = RagPipeline()
    try:
        result = pipeline.ask(args.question, top_k=args.top_k, verbose=args.verbose)

        print(f"\n{'=' * 70}")
        print(f"CÂU HỎI: {result.question}")
        print(f"{'=' * 70}")
        print(result.answer)
        print(f"\n{'-' * 70}")
        print(f"Citations: {result.citations}")
        print(f"Citation IDs: {result.citation_ids}")
        print(
            f"Vector hits: {len(result.hits)}, "
            f"semantic edges expanded: {result.n_semantic_edges}, "
            f"refs: {result.n_refs}"
        )
        print(f"Elapsed: {result.elapsed_s}s")

        if args.verify and result.citation_ids:
            verified = pipeline.verify_citations(result.citation_ids)
            print("\nCitation verification:")
            for cid, ok in verified.items():
                mark = "✓" if ok else "✗"
                print(f"  {mark} {cid}")
    finally:
        pipeline.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
