"""GraphRAG-Cypher arm — Hybrid (vector seed → LLM-generated Cypher → graph walk).

Pipeline per question:
1. Vector search over Clause.text (reuse RagPipeline.vector_search) → top-K seed clauses.
2. LLM (gpt-4o-mini by default) writes a Cypher query that traverses the schema
   starting from the seed clauses. Output is JSON {cypher, rationale}.
3. Validator (regex + whitelist) rejects writes, non-whitelisted labels/edges,
   and queries that do not reference ``$seed_ids``.
4. Execute the Cypher with ``session.execute_read`` (server-enforced read mode).
5. If the result is empty OR validation/execution failed: send error feedback to
   LLM and retry, up to MAX_REPAIR_ROUNDS. If still no rows: FALLBACK to vanilla
   GraphRAG context (vector hits + ``RagPipeline.expand``).
6. Render the final answer with a system prompt that has both GRAPH FACTS and
   CLAUSE TEXTS sections.

The record emitted by the arm runner is honest about which path was taken:
``cypher_used`` is True iff a Cypher query returned ≥1 row. ``fallback_used``
records the dual signal.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from runtime.rag_query import RagPipeline, SearchHit
from src.prompts import load_prompt

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CYPHER_GEN_SYSTEM = load_prompt("runtime/graphrag_cypher/cypher_gen.md")
ANSWER_RENDER_SYSTEM = load_prompt("runtime/graphrag_cypher/answer_render.md")

# --- Whitelists (derived from schema/schema.cypher + merged_graph.json reality) ---

NODE_LABELS = frozenset({
    "Law", "Chapter", "Section", "Article", "Clause", "Point", "Table",
    "Subject", "Organization", "Role", "Benefit", "Condition", "Obligation",
    "Right", "ProhibitedAct", "Fund", "LegalConcept", "ExternalLaw",
    # Phase-3 extraction labels — present in schema, may exist in DB
    "LegalRule", "LegalCondition", "NumericalThreshold", "LegalTerm",
    "ProcedureStep", "LegalEntity", "CanonicalPredicate",
})

EDGE_TYPES = frozenset({
    # Structural
    "HAS_CHAPTER", "HAS_SECTION", "HAS_ARTICLE", "IN_SECTION",
    "HAS_CLAUSE", "HAS_POINT", "HAS_TABLE", "BELONGS_TO", "NEXT",
    # Reference / citation
    "REFERENCES", "REFERS_TO", "CITES_EXTERNAL",
    "AMENDS", "REPEALS", "REPLACES", "TRANSITIONS_FROM",
    # Semantic
    "DEFINES", "ENTITLED_TO", "HAS_OBLIGATION", "HAS_RIGHT", "APPLIES_TO",
    "REQUIRES", "PAID_FROM", "MANAGES", "RESPONSIBLE_FOR", "PROHIBITED_BY",
    # Phase-3 extraction
    "EXTRACTED_FROM", "INVOLVES_ENTITY",
})

# Keywords whose presence anywhere in the (uppercased, tokenised) Cypher
# means the query is write-capable or unsafe — reject pre-execution.
FORBIDDEN_KEYWORDS = (
    "CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP", "LOAD",
    "FOREACH", "CALL",
)

MAX_REPAIR_ROUNDS = 2
CYPHER_TX_TIMEOUT_S = 15
ROWS_HARD_CAP = 30


@dataclass
class CypherAttempt:
    round: int
    cypher: str
    rationale: str
    valid: bool
    validation_error: str
    executed: bool
    execution_error: str
    n_rows: int
    rows_preview: list[dict] = field(default_factory=list)  # up to 3 rows


@dataclass
class GraphRagCypherAnswer:
    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    citation_ids: list[str] = field(default_factory=list)
    vector_hits: list[SearchHit] = field(default_factory=list)
    cypher_attempts: list[CypherAttempt] = field(default_factory=list)
    cypher_used: bool = False                # ≥1 row from a validated Cypher
    cypher_rows: list[dict] = field(default_factory=list)
    cypher_clause_ids_added: list[str] = field(default_factory=list)
    fallback_used: bool = False
    elapsed_s: float = 0.0
    elapsed_breakdown: dict[str, float] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---------------------------------------------------------------------------
# Cypher validation
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def parse_llm_json(raw: str) -> dict:
    """Tolerate ```json fences or extra prose. Returns dict or raises ValueError."""
    cleaned = _strip_code_fences(raw)
    # Find the first {...} block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"No JSON object found in LLM output: {raw[:200]}")
    blob = cleaned[start : end + 1]
    return json.loads(blob)


_NODE_PAREN_RE = re.compile(r"\(([^()]*)\)")
_REL_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
# Inside a node paren, labels look like `:Label`. Strip property maps `{...}`
# first so JSON-style values don't get scanned.
_PROPMAP_RE = re.compile(r"\{[^{}]*\}")
_INNER_LABEL_RE = re.compile(r":\s*([A-Z][A-Za-z0-9_]*)")
# Inside a relationship bracket, relationship type appears after the variable
# (e.g. `r:TYPE`) or right after the opening (`:TYPE`). Multiple types are
# OR-joined with `|`. Property maps may follow.
_INNER_REL_RE = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]*)?\s*:\s*([A-Z_][A-Z0-9_]*(?:\s*\|\s*:?\s*[A-Z_][A-Z0-9_]*)*)"
)
_KW_RE_TEMPLATE = r"(?:^|\s|;)({})\b"


def validate_cypher(cypher: str) -> tuple[bool, str]:
    """Return (ok, reason). Reason is '' if ok.

    Checks (in order):
    1. Non-empty.
    2. No forbidden write/CALL keywords (word-boundary, case-insensitive).
    3. Must reference ``$seed_ids`` somewhere.
    4. Every ``:Label`` token must be in NODE_LABELS whitelist.
    5. Every ``[r:TYPE]`` or ``[:TYPE]`` (including ``A|B|C`` lists) must be
       in EDGE_TYPES whitelist.
    6. Must contain ``RETURN`` and the keyword ``clause_id`` (the contract on
       result shape — relaxed: case-insensitive, allows aliasing).
    7. Must contain ``LIMIT`` clause with value ≤ ROWS_HARD_CAP.
    """
    if not cypher or not cypher.strip():
        return False, "empty cypher"

    up = cypher.upper()
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(_KW_RE_TEMPLATE.format(kw), up):
            return False, f"forbidden keyword: {kw}"

    if "$SEED_IDS" not in up:
        return False, "must reference $seed_ids parameter"

    # Node labels — only inside `(...)` parens. Strip property maps first so
    # `{...}` literals (which may contain ':') don't get scanned.
    for paren_m in _NODE_PAREN_RE.finditer(cypher):
        inner = _PROPMAP_RE.sub("", paren_m.group(1))
        for m in _INNER_LABEL_RE.finditer(inner):
            lbl = m.group(1)
            if lbl not in NODE_LABELS:
                return False, f"unknown node label: {lbl}"

    # Relationship types — only inside `[...]` brackets. Support `[r:TYPE]`,
    # `[:TYPE]`, `[r:A|B|C]`, `[r:A|:B]`. Property maps removed first.
    for br_m in _REL_BRACKET_RE.finditer(cypher):
        inner = _PROPMAP_RE.sub("", br_m.group(1))
        # Skip variable-length specs like `*1..3` — they cannot carry a type
        # alone but may co-exist (e.g. `[r:REFERS_TO*1..2]`); the type still
        # matches the regex below, so this is fine.
        for m in _INNER_REL_RE.finditer(inner):
            for tok in re.split(r"\s*\|\s*:?\s*", m.group(1)):
                tok = tok.strip()
                if not tok:
                    continue
                if tok not in EDGE_TYPES:
                    return False, f"unknown edge type: {tok}"

    if not re.search(r"\bRETURN\b", up):
        return False, "missing RETURN"

    # Be lenient: 'clause_id' may appear as an alias, parameter, or property
    if "clause_id" not in cypher.lower():
        return False, "RETURN must include a column named clause_id"

    m_lim = re.search(r"\bLIMIT\s+(\d+)", up)
    if not m_lim:
        return False, "missing LIMIT (≤ 30 required)"
    try:
        if int(m_lim.group(1)) > ROWS_HARD_CAP:
            return False, f"LIMIT > {ROWS_HARD_CAP}"
    except ValueError:
        return False, "LIMIT value not an integer"

    return True, ""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class GraphRagCypherPipeline:
    """Hybrid vector-seed + LLM-Cypher + fallback GraphRAG."""

    def __init__(self, top_k: int = 8, cypher_model: str | None = None,
                 answer_model: str | None = None):
        self.top_k = top_k
        self.cypher_model = cypher_model or OPENAI_MODEL
        self.answer_model = answer_model or OPENAI_MODEL
        self.rag = RagPipeline()
        self._openai = None

    @property
    def embed_model(self):
        return self.rag.embed_model

    @property
    def openai(self):
        if self._openai is None:
            from openai import OpenAI

            self._openai = OpenAI()
        return self._openai

    def close(self) -> None:
        self.rag.close()

    # ------------------------------------------------------------------
    # Cypher generation + repair loop
    # ------------------------------------------------------------------

    def _generate_cypher(
        self,
        question: str,
        seed_ids: list[str],
        prev_attempt: CypherAttempt | None,
    ) -> tuple[str, str, int, int]:
        """Call LLM. Returns (cypher, rationale, prompt_tokens, completion_tokens)."""
        seed_block = json.dumps(seed_ids, ensure_ascii=False)
        user_parts = [
            f"CÂU HỎI: {question}",
            f"SEED Clause.id: {seed_block}",
        ]
        if prev_attempt is not None:
            err = prev_attempt.validation_error or prev_attempt.execution_error or (
                "kết quả 0 rows — câu Cypher đi sai hướng, hãy đổi đường traversal."
            )
            user_parts.append(
                "LẦN TRƯỚC bạn đã viết:\n```cypher\n"
                + prev_attempt.cypher.strip()
                + f"\n```\nLỖI: {err}\nHãy viết lại câu Cypher khác."
            )
        user_msg = "\n\n".join(user_parts)

        resp = self.openai.chat.completions.create(
            model=self.cypher_model,
            messages=[
                {"role": "system", "content": CYPHER_GEN_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        try:
            blob = parse_llm_json(raw)
            cypher = str(blob.get("cypher") or "").strip()
            rationale = str(blob.get("rationale") or "").strip()
        except (json.JSONDecodeError, ValueError) as exc:
            cypher = ""
            rationale = f"LLM trả về JSON không hợp lệ: {exc}"
        return (
            cypher,
            rationale,
            resp.usage.prompt_tokens if resp.usage else 0,
            resp.usage.completion_tokens if resp.usage else 0,
        )

    def _execute_cypher(self, cypher: str, seed_ids: list[str]) -> list[dict]:
        """Run in an explicit READ transaction with a server-side timeout.

        Read access is enforced via ``default_access_mode=READ_ACCESS`` — the
        server rejects any write-side mutation regardless of what the LLM
        emitted, even if our regex validator missed something. Timeout is
        enforced server-side via ``begin_transaction(timeout=...)``.
        """
        from neo4j import READ_ACCESS

        db = os.getenv("NEO4J_DATABASE", "neo4j")
        with self.rag.driver.session(
            database=db, default_access_mode=READ_ACCESS
        ) as s:
            with s.begin_transaction(timeout=CYPHER_TX_TIMEOUT_S) as tx:
                rows = tx.run(cypher, seed_ids=seed_ids).data()
        return rows[:ROWS_HARD_CAP]

    def _cypher_loop(
        self,
        question: str,
        seed_ids: list[str],
    ) -> tuple[list[CypherAttempt], list[dict], int, int]:
        attempts: list[CypherAttempt] = []
        accepted_rows: list[dict] = []
        total_prompt = 0
        total_completion = 0
        prev: CypherAttempt | None = None

        for rnd in range(MAX_REPAIR_ROUNDS + 1):
            cypher, rationale, pt, ct = self._generate_cypher(question, seed_ids, prev)
            total_prompt += pt
            total_completion += ct

            attempt = CypherAttempt(
                round=rnd,
                cypher=cypher,
                rationale=rationale,
                valid=False,
                validation_error="",
                executed=False,
                execution_error="",
                n_rows=0,
            )

            if not cypher:
                attempt.validation_error = "LLM returned empty cypher"
                attempts.append(attempt)
                prev = attempt
                continue

            ok, reason = validate_cypher(cypher)
            attempt.valid = ok
            attempt.validation_error = reason
            if not ok:
                attempts.append(attempt)
                prev = attempt
                continue

            try:
                rows = self._execute_cypher(cypher, seed_ids)
                attempt.executed = True
                attempt.n_rows = len(rows)
                attempt.rows_preview = rows[:3]
                if rows:
                    accepted_rows = rows
                    attempts.append(attempt)
                    return attempts, accepted_rows, total_prompt, total_completion
            except Exception as exc:
                attempt.execution_error = f"{type(exc).__name__}: {str(exc)[:300]}"

            attempts.append(attempt)
            prev = attempt

        return attempts, accepted_rows, total_prompt, total_completion

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    @staticmethod
    def _format_graph_facts(rows: list[dict]) -> str:
        if not rows:
            return "(không có)"
        lines = []
        for r in rows:
            cid = r.get("clause_id") or "?"
            rel = r.get("relation_type") or "?"
            tgt_lbl = r.get("target_label") or ""
            tgt_txt = r.get("target_text") or r.get("target_id") or ""
            src_ent = r.get("source_entity") or ""
            evidence = (r.get("evidence") or "")
            if isinstance(tgt_txt, list):
                tgt_txt = ", ".join(str(x) for x in tgt_txt)
            line = f"- [{cid}] "
            if src_ent:
                line += f"{src_ent} --{rel}--> "
            else:
                line += f"--{rel}--> "
            line += f"{tgt_txt} ({tgt_lbl})"
            if evidence:
                snip = str(evidence)[:200]
                line += f'\n    Bằng chứng: "{snip}"'
            lines.append(line)
        return "\n".join(lines)

    def _build_context(
        self,
        hits: list[SearchHit],
        cypher_rows: list[dict],
        extra_clause_ids: list[str],
        max_chars: int = 8000,
    ) -> str:
        parts: list[str] = ["# GRAPH FACTS (từ câu Cypher đi trên đồ thị)"]
        parts.append(self._format_graph_facts(cypher_rows))

        # Pull text for clauses that the Cypher surfaced but vector did not
        extra_text_block = ""
        if extra_clause_ids:
            extra = self._fetch_clause_texts(extra_clause_ids)
            if extra:
                extra_text_block = (
                    "\n\n# CLAUSE TEXTS (Khoản được Cypher chỉ về, ngoài vector hits)\n"
                )
                for r in extra:
                    extra_text_block += (
                        f"\n## [{r['clause_id']}] Điều {r['article_n']}. "
                        f"{r['article_title']} - Khoản {r['clause_n']}\n"
                    )
                    extra_text_block += r["text"]

        parts.append("\n\n# CLAUSE TEXTS (từ vector search, sắp theo relevance)")
        for h in hits:
            parts.append(
                f"\n## [{h.clause_id}] Điều {h.article_n}. {h.article_title} - "
                f"Khoản {h.clause_n} (score={h.score:.3f})"
            )
            parts.append(h.text)
        if extra_text_block:
            parts.append(extra_text_block)

        ctx = "\n".join(parts)
        if len(ctx) > max_chars:
            ctx = ctx[:max_chars] + "\n\n... (context truncated)"
        return ctx

    def _build_fallback_context(self, hits: list[SearchHit], expansion: dict) -> str:
        return self.rag.build_context(hits, expansion)

    def _fetch_clause_texts(self, clause_ids: list[str]) -> list[dict]:
        if not clause_ids:
            return []
        with self.rag.driver.session(
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        ) as s:
            rows = s.run(
                """
                UNWIND $ids AS cid
                MATCH (a:Article)-[:HAS_CLAUSE]->(c:Clause {id: cid})
                RETURN c.id AS clause_id, c.text AS text, c.number AS clause_n,
                       a.id AS article_id, a.title AS article_title,
                       a.number AS article_n
                """,
                ids=clause_ids,
            ).data()
        return rows

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _generate_answer(
        self, question: str, context: str
    ) -> tuple[str, int, int]:
        resp = self.openai.chat.completions.create(
            model=self.answer_model,
            messages=[
                {"role": "system", "content": ANSWER_RENDER_SYSTEM},
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context}\n\n---\n\nCÂU HỎI: {question}",
                },
            ],
            temperature=0,
        )
        ans = resp.choices[0].message.content or ""
        pt = resp.usage.prompt_tokens if resp.usage else 0
        ct = resp.usage.completion_tokens if resp.usage else 0
        return ans, pt, ct

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def ask(self, question: str) -> GraphRagCypherAnswer:
        t0 = time.time()
        breakdown: dict[str, float] = {}

        # 1) vector seed
        t = time.time()
        hits = self.rag.vector_search(question, top_k=self.top_k)
        breakdown["vector_s"] = round(time.time() - t, 3)
        seed_ids = [h.clause_id for h in hits]

        # 2) Cypher loop
        t = time.time()
        attempts, cypher_rows, pt_c, ct_c = self._cypher_loop(question, seed_ids)
        breakdown["cypher_s"] = round(time.time() - t, 3)

        cypher_used = bool(cypher_rows)
        extra_clause_ids: list[str] = []
        if cypher_used:
            seed_set = set(seed_ids)
            for r in cypher_rows:
                cid = r.get("clause_id")
                if isinstance(cid, str) and cid not in seed_set:
                    extra_clause_ids.append(cid)
            # dedupe while preserving order
            extra_clause_ids = list(dict.fromkeys(extra_clause_ids))

        # 3) Build context — Cypher-led or fallback
        fallback_used = False
        if cypher_used:
            t = time.time()
            context = self._build_context(hits, cypher_rows, extra_clause_ids)
            breakdown["context_s"] = round(time.time() - t, 3)
        else:
            fallback_used = True
            t = time.time()
            expansion = self.rag.expand(seed_ids)
            context = self._build_fallback_context(hits, expansion)
            breakdown["context_s"] = round(time.time() - t, 3)

        # 4) Render answer
        t = time.time()
        answer_text, pt_a, ct_a = self._generate_answer(question, context)
        breakdown["render_s"] = round(time.time() - t, 3)

        citations, citation_ids = self.rag.parse_citations(answer_text)

        return GraphRagCypherAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            citation_ids=citation_ids,
            vector_hits=hits,
            cypher_attempts=attempts,
            cypher_used=cypher_used,
            cypher_rows=cypher_rows,
            cypher_clause_ids_added=extra_clause_ids,
            fallback_used=fallback_used,
            elapsed_s=round(time.time() - t0, 2),
            elapsed_breakdown=breakdown,
            prompt_tokens=pt_c + pt_a,
            completion_tokens=ct_c + ct_a,
        )


# ---------------------------------------------------------------------------
# CLI — single question, for ad-hoc testing
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-q", "--question", required=True)
    p.add_argument("-k", "--top-k", type=int, default=8)
    args = p.parse_args()

    pipe = GraphRagCypherPipeline(top_k=args.top_k)
    try:
        a = pipe.ask(args.question)
        print(f"\n=== QUESTION ===\n{a.question}")
        print(f"\n=== CYPHER ATTEMPTS ({len(a.cypher_attempts)}) ===")
        for att in a.cypher_attempts:
            print(f"\n[round {att.round}] valid={att.valid} executed={att.executed} n_rows={att.n_rows}")
            print(f"rationale: {att.rationale}")
            print(att.cypher[:500])
            if att.validation_error:
                print(f"VAL ERR: {att.validation_error}")
            if att.execution_error:
                print(f"EXE ERR: {att.execution_error}")
        print(f"\ncypher_used={a.cypher_used}  fallback_used={a.fallback_used}")
        print(f"extra_clause_ids: {a.cypher_clause_ids_added}")
        print(f"\n=== ANSWER ===\n{a.answer}")
        print(f"\nCitations: {a.citations}")
        print(f"Elapsed: {a.elapsed_s}s  breakdown: {a.elapsed_breakdown}")
        print(f"Tokens: prompt={a.prompt_tokens} completion={a.completion_tokens}")
    finally:
        pipe.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
