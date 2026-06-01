"""``CypherWalkRetriever`` — a retrieval-layer component.

Position in the architecture: it is a *peer* of
``RagPipeline.vector_search`` / ``V5RetrievalPipeline`` —
it takes a question and returns a ranked set of clauses + provenance.

    NO LLM render. NO citation parsing. NO answer generation.

Pipeline:

    [1] vector_search(question, top_k_seed)                → seed clauses
    [2] LLM Cypher gen (constrained, with repair)          → outward walk
          MUST seed from node identity (`<node>.id IN $seed_ids`)
          MUST traverse OUTWARD and RETURN target_clause_id / target_article_id
          MUST NOT pin on `r.source_clause IN $seed_ids` (that anchors
          every row to the seed set by construction)
    [3] execute on Neo4j (READ_ACCESS, timeout 15s)
          success = >=1 row whose clause is NOT in the seed set
          (else repair, up to max_repair_rounds)
    [4] if all Cypher rounds yield 0 NEW clauses → fallback_expand =
          RagPipeline.expand(seed_clause_ids); derive NEW clause candidates
          from its REFERENCES / CITES_EXTERNAL refs (vanilla graphrag's own
          retrieval-side behaviour — honest degrade to baseline, not to nothing)
    [5] fuse (RRF, k=60) seed ∥ cypher-new ∥ fallback → top_k_final clauses

The component reuses ``RagPipeline.vector_search`` for the dense seed and
``RagPipeline.driver`` for Neo4j access, but it does not depend on the v5
retrieval stack — that keeps the exp 11 audit comparable to vanilla
graphrag without pulling in v5 changes.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from runtime.rag_query import RagPipeline, SearchHit
from src.ids import parse_id
from src.prompts import load_prompt

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

DB = os.getenv("NEO4J_DATABASE", "neo4j")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CYPHER_GEN_SYSTEM = load_prompt("runtime/cypher_walk/cypher_gen.md")

# --- Whitelists (derived from schema/schema.cypher + merged_graph.json reality) ---

NODE_LABELS = frozenset({
    "Law", "Chapter", "Section", "Article", "Clause", "Point", "Table",
    "Subject", "Organization", "Role", "Benefit", "Condition", "Obligation",
    "Right", "ProhibitedAct", "Fund", "LegalConcept", "ExternalLaw",
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

FORBIDDEN_KEYWORDS = (
    "CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP", "LOAD",
    "FOREACH", "CALL",
)

DEFAULT_MAX_REPAIR_ROUNDS = 2
CYPHER_TX_TIMEOUT_S = 15
ROWS_HARD_CAP = 30
DEFAULT_RRF_K = 60


# ---------------------------------------------------------------------------
# Data classes (plan §3)
# ---------------------------------------------------------------------------


@dataclass
class RetrievedClause:
    clause_id: str
    article_id: str
    article_number: int | None
    article_title: str
    clause_number: int | None
    text: str
    score: float                # final fused (RRF) rank score
    source: str                 # "vector" | "cypher" | "fallback_expand"


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
    n_new_clauses: int          # rows whose clause is NOT in the seed set
    rows_preview: list[dict] = field(default_factory=list)  # up to 3 rows


@dataclass
class CypherWalkResult:
    question: str
    hits: list[RetrievedClause] = field(default_factory=list)   # top-K final
    seed_clause_ids: list[str] = field(default_factory=list)
    cypher_new_clause_ids: list[str] = field(default_factory=list)  # NEW (beyond seed)
    fallback_clause_ids: list[str] = field(default_factory=list)    # NEW via vanilla expand
    n_seed: int = 0                          # vector seed count
    n_cypher_new: int = 0                    # NEW clauses surfaced by Cypher (KEY signal)
    n_fallback_added: int = 0                # clauses added by vanilla expand
    cypher_used: bool = False                # >=1 NEW clause from Cypher
    fallback_used: bool = False              # vanilla expand kicked in
    cypher_attempts: list[CypherAttempt] = field(default_factory=list)
    elapsed_s: float = 0.0
    elapsed_breakdown: dict[str, float] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# JSON / Cypher parsing helpers
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
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"No JSON object found in LLM output: {raw[:200]}")
    return json.loads(cleaned[start : end + 1])


# ---------------------------------------------------------------------------
# Cypher validation (plan §4.3 + §4.4)
# ---------------------------------------------------------------------------

_NODE_PAREN_RE = re.compile(r"\(([^()]*)\)")
_REL_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
_PROPMAP_RE = re.compile(r"\{[^{}]*\}")
_INNER_LABEL_RE = re.compile(r":\s*([A-Z][A-Za-z0-9_]*)")
_INNER_REL_RE = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]*)?\s*:\s*([A-Z_][A-Z0-9_]*(?:\s*\|\s*:?\s*[A-Z_][A-Z0-9_]*)*)"
)
_KW_RE_TEMPLATE = r"(?:^|\s|;)({})\b"

# Outward-traversal contract regexes
_TARGET_COL_RE = re.compile(r"\btarget_(?:clause|article)_id\b", re.IGNORECASE)
_NODE_SEED_RE = re.compile(r"\.\s*id\s+IN\s+\$seed_ids", re.IGNORECASE)
_PIN_RE = re.compile(r"\.\s*source_clause\s+IN\s+\$seed_ids", re.IGNORECASE)
_SRC_AS_TARGET_RE = re.compile(
    r"source_clause\s+AS\s+target_(?:clause|article)_id", re.IGNORECASE
)


def validate_cypher(cypher: str) -> tuple[bool, str]:
    """Return ``(ok, reason)``. ``reason`` is '' if ok.

    Keeps all hygiene checks from the previous attempt (read-only keyword
    denylist, whitelisted node labels + edge types, ``$seed_ids`` reference,
    ``LIMIT <= 30``) AND adds the no-pin / outward-traversal contract:

    - RETURN must expose ``target_clause_id`` OR ``target_article_id``.
    - The query must seed from **node identity** (``<node>.id IN $seed_ids``)
      so it starts at the seed nodes and can traverse OUTWARD to other nodes.
    - The seed-pin pattern ``<rel>.source_clause IN $seed_ids`` is rejected:
      it anchors every row to a seed clause by construction and therefore
      cannot surface a new clause (this is the exact degeneracy the redo
      corrects — see plan §1 / §4.3). A query that cannot surface a new
      clause is not a graph walk.
    - ``source_clause AS target_*_id`` is rejected (the target must come from
      a traversed node, not from the seed anchor).
    """
    if not cypher or not cypher.strip():
        return False, "empty cypher"

    up = cypher.upper()
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(_KW_RE_TEMPLATE.format(kw), up):
            return False, f"forbidden keyword: {kw}"

    if "$SEED_IDS" not in up:
        return False, "must reference $seed_ids parameter"

    for paren_m in _NODE_PAREN_RE.finditer(cypher):
        inner = _PROPMAP_RE.sub("", paren_m.group(1))
        for m in _INNER_LABEL_RE.finditer(inner):
            lbl = m.group(1)
            if lbl not in NODE_LABELS:
                return False, f"unknown node label: {lbl}"

    for br_m in _REL_BRACKET_RE.finditer(cypher):
        inner = _PROPMAP_RE.sub("", br_m.group(1))
        for m in _INNER_REL_RE.finditer(inner):
            for tok in re.split(r"\s*\|\s*:?\s*", m.group(1)):
                tok = tok.strip()
                if not tok:
                    continue
                if tok not in EDGE_TYPES:
                    return False, f"unknown edge type: {tok}"

    if not re.search(r"\bRETURN\b", up):
        return False, "missing RETURN"

    # --- outward-traversal contract -------------------------------------
    if not _TARGET_COL_RE.search(cypher):
        return False, "RETURN must expose target_clause_id or target_article_id"

    if _PIN_RE.search(cypher):
        return False, (
            "seed-pin pattern `<rel>.source_clause IN $seed_ids` is forbidden — "
            "seed from node identity (`<node>.id IN $seed_ids`) and traverse OUTWARD"
        )

    if not _NODE_SEED_RE.search(cypher):
        return False, (
            "must seed from node identity (`<node>.id IN $seed_ids`) and traverse "
            "OUTWARD to other clauses/articles"
        )

    if _SRC_AS_TARGET_RE.search(cypher):
        return False, "target_*_id must come from a traversed node, not source_clause"

    m_lim = re.search(r"\bLIMIT\s+(\d+)", up)
    if not m_lim:
        return False, f"missing LIMIT (<= {ROWS_HARD_CAP} required)"
    try:
        if int(m_lim.group(1)) > ROWS_HARD_CAP:
            return False, f"LIMIT > {ROWS_HARD_CAP}"
    except ValueError:
        return False, "LIMIT value not an integer"

    return True, ""


# ---------------------------------------------------------------------------
# id helpers
# ---------------------------------------------------------------------------


def _is_clause_id(node_id: str) -> bool:
    try:
        p = parse_id(node_id)
    except ValueError:
        return False
    return p.get("article") is not None and p.get("clause") is not None


def _is_article_id(node_id: str) -> bool:
    try:
        p = parse_id(node_id)
    except ValueError:
        return False
    return p.get("article") is not None and p.get("clause") is None


def _dedupe(seq) -> list[str]:
    return list(dict.fromkeys(x for x in seq if x))


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class CypherWalkRetriever:
    """Vector seed + LLM-authored outward Cypher walk + fallback + RRF fusion.

    Pure retrieval — returns a :class:`CypherWalkResult`. No answer, no
    citations.
    """

    def __init__(
        self,
        rag: RagPipeline,
        top_k_seed: int = 8,
        top_k_final: int = 12,
        max_repair_rounds: int = DEFAULT_MAX_REPAIR_ROUNDS,
        cypher_model: str | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        seed_query_encoder=None,
    ):
        self.rag = rag
        self.top_k_seed = top_k_seed
        self.top_k_final = top_k_final
        self.max_repair_rounds = max_repair_rounds
        self.cypher_model = cypher_model or OPENAI_MODEL
        self.rrf_k = rrf_k
        # Optional ``question -> query embedding`` callable. When set, the
        # vector seed is drawn with this embedding (e.g. a HyDE hypothetical-
        # doc embedding) instead of the raw-question embedding. None →
        # vanilla raw-question seed (the exp 11 behaviour, unchanged).
        self.seed_query_encoder = seed_query_encoder
        self._openai = None

    @property
    def seed_mode(self) -> str:
        return "hyde" if self.seed_query_encoder is not None else "raw"

    def _vector_seed(self, question: str) -> list[SearchHit]:
        if self.seed_query_encoder is None:
            return self.rag.vector_search(question, top_k=self.top_k_seed)
        qvec = self.seed_query_encoder(question)
        return self.rag.vector_search_by_vector(qvec, top_k=self.top_k_seed)

    @property
    def openai(self):
        if self._openai is None:
            from openai import OpenAI

            self._openai = OpenAI()
        return self._openai

    # ------------------------------------------------------------------
    # Cypher generation + repair loop
    # ------------------------------------------------------------------

    def _generate_cypher(
        self,
        question: str,
        seed_ids: list[str],
        prev_attempt: CypherAttempt | None,
    ) -> tuple[str, str, int, int]:
        seed_block = json.dumps(seed_ids, ensure_ascii=False)
        user_parts = [
            f"CÂU HỎI: {question}",
            f"SEED Clause.id (đi RA TỪ các node này): {seed_block}",
        ]
        if prev_attempt is not None:
            err = (
                prev_attempt.validation_error
                or prev_attempt.execution_error
                or (
                    "câu Cypher hợp lệ nhưng KHÔNG surface Khoản/Điều MỚI nào "
                    "ngoài seed — hãy đổi đường traversal để chạm tới node khác."
                )
            )
            user_parts.append(
                "LẦN TRƯỚC bạn đã viết:\n```cypher\n"
                + prev_attempt.cypher.strip()
                + f"\n```\nLỖI: {err}\nHãy viết lại câu Cypher KHÁC đi ra ngoài seed."
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
        server rejects any write regardless of what the LLM emitted, even if
        the regex validator missed it.
        """
        from neo4j import READ_ACCESS

        with self.rag.driver.session(
            database=DB, default_access_mode=READ_ACCESS
        ) as s:
            with s.begin_transaction(timeout=CYPHER_TX_TIMEOUT_S) as tx:
                rows = tx.run(cypher, seed_ids=seed_ids).data()
        return rows[:ROWS_HARD_CAP]

    def _new_clause_ids_from_rows(
        self, rows: list[dict], seed_set: set[str]
    ) -> list[str]:
        """Map Cypher rows → ordered list of NEW clause ids (not in seed).

        ``target_clause_id`` is used directly when it parses as a Clause id.
        ``target_article_id`` (and any ``target_clause_id`` that is really an
        Article id) is expanded to the article's first Clause downstream
        (plan §4.2). Targets that are neither (entities, ExternalLaw) are
        dropped — they cannot surface a new clause.
        """
        article_targets: list[str] = []
        for r in rows:
            for key in ("target_clause_id", "target_article_id"):
                tid = r.get(key)
                if isinstance(tid, str) and _is_article_id(tid):
                    article_targets.append(tid)
        first_clause = self._first_clause_of_articles(_dedupe(article_targets))

        ordered: list[str] = []
        for r in rows:
            cid = r.get("target_clause_id")
            if isinstance(cid, str) and _is_clause_id(cid):
                ordered.append(cid)
            for key in ("target_clause_id", "target_article_id"):
                tid = r.get(key)
                if isinstance(tid, str) and _is_article_id(tid):
                    resolved = first_clause.get(tid, {}).get("clause_id")
                    if resolved:
                        ordered.append(resolved)
        return [c for c in _dedupe(ordered) if c not in seed_set]

    def _cypher_loop(
        self, question: str, seed_ids: list[str]
    ) -> tuple[list[CypherAttempt], list[str], int, int]:
        """Returns (attempts, new_clause_ids_from_first_success, prompt_tok, completion_tok).

        Success = a validated query that surfaces >=1 NEW clause beyond seed.
        """
        seed_set = set(seed_ids)
        attempts: list[CypherAttempt] = []
        new_clause_ids: list[str] = []
        total_prompt = total_completion = 0
        prev: CypherAttempt | None = None

        for rnd in range(self.max_repair_rounds + 1):
            cypher, rationale, pt, ct = self._generate_cypher(question, seed_ids, prev)
            total_prompt += pt
            total_completion += ct

            attempt = CypherAttempt(
                round=rnd, cypher=cypher, rationale=rationale,
                valid=False, validation_error="", executed=False,
                execution_error="", n_rows=0, n_new_clauses=0,
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
                surfaced = self._new_clause_ids_from_rows(rows, seed_set)
                attempt.n_new_clauses = len(surfaced)
                if surfaced:
                    new_clause_ids = surfaced
                    attempts.append(attempt)
                    return attempts, new_clause_ids, total_prompt, total_completion
            except Exception as exc:  # noqa: BLE001
                attempt.execution_error = f"{type(exc).__name__}: {str(exc)[:300]}"

            attempts.append(attempt)
            prev = attempt

        return attempts, new_clause_ids, total_prompt, total_completion

    # ------------------------------------------------------------------
    # Neo4j fetch helpers
    # ------------------------------------------------------------------

    def _fetch_clause_rows(self, clause_ids: list[str]) -> dict[str, dict]:
        if not clause_ids:
            return {}
        with self.rag.driver.session(database=DB) as s:
            rows = s.run(
                """
                UNWIND $ids AS cid
                MATCH (a:Article)-[:HAS_CLAUSE]->(c:Clause {id: cid})
                RETURN c.id AS clause_id, c.text AS text, c.number AS clause_number,
                       a.id AS article_id, a.title AS article_title,
                       a.number AS article_number
                """,
                ids=clause_ids,
            ).data()
        return {r["clause_id"]: r for r in rows}

    def _first_clause_of_articles(self, article_ids: list[str]) -> dict[str, dict]:
        if not article_ids:
            return {}
        with self.rag.driver.session(database=DB) as s:
            rows = s.run(
                """
                UNWIND $ids AS aid
                MATCH (a:Article {id: aid})-[:HAS_CLAUSE]->(c:Clause)
                WITH a, c ORDER BY c.number ASC
                WITH a, head(collect(c)) AS c
                WHERE c IS NOT NULL
                RETURN a.id AS article_id, c.id AS clause_id,
                       c.number AS clause_number, c.text AS text,
                       a.title AS article_title, a.number AS article_number
                """,
                ids=article_ids,
            ).data()
        return {r["article_id"]: r for r in rows}

    def _fallback_expand(
        self, seed_ids: list[str], seed_set: set[str]
    ) -> list[str]:
        """Vanilla graphrag's retrieval-side neighbour behaviour.

        ``RagPipeline.expand`` returns semantic edges (entity→entity, which
        carry no clause/article id and cannot add a candidate clause) plus
        REFERENCES / CITES_EXTERNAL refs. We derive NEW clause candidates
        from the refs only: a ref whose target is a Clause id is taken
        directly; a ref whose target is an Article id is expanded to that
        article's first clause. ExternalLaw / entity targets are dropped.
        Order follows the refs order. This mirrors exactly what vanilla
        graphrag surfaces on the retrieval side — no REFERS_TO, no v5
        additions.
        """
        expansion = self.rag.expand(seed_ids)
        ref_clause_targets: list[str] = []
        ref_article_targets: list[str] = []
        for ref in expansion.get("refs", []):
            dst = ref.get("dst")
            if not isinstance(dst, str):
                continue
            if _is_clause_id(dst):
                ref_clause_targets.append(dst)
            elif _is_article_id(dst):
                ref_article_targets.append(dst)
        first_clause = self._first_clause_of_articles(_dedupe(ref_article_targets))

        ordered: list[str] = []
        for ref in expansion.get("refs", []):
            dst = ref.get("dst")
            if not isinstance(dst, str):
                continue
            if _is_clause_id(dst):
                ordered.append(dst)
            elif _is_article_id(dst):
                resolved = first_clause.get(dst, {}).get("clause_id")
                if resolved:
                    ordered.append(resolved)
        return [c for c in _dedupe(ordered) if c not in seed_set]

    # ------------------------------------------------------------------
    # Fusion (plan §4.5)
    # ------------------------------------------------------------------

    def _rrf(self, ranked_lists: list[list[str]]) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        for lst in ranked_lists:
            for rank, key in enumerate(lst, start=1):
                scores[key] += 1.0 / (self.rrf_k + rank)
        return dict(scores)

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> CypherWalkResult:
        t0 = time.time()
        breakdown: dict[str, float] = {}

        # [1] vector seed (raw-question embedding, or HyDE doc embedding
        #     when a seed_query_encoder is wired)
        t = time.time()
        seed_hits: list[SearchHit] = self._vector_seed(question)
        breakdown["vector_s"] = round(time.time() - t, 3)
        seed_ids = [h.clause_id for h in seed_hits]
        seed_set = set(seed_ids)

        # [2-3] Cypher loop
        t = time.time()
        attempts, cypher_new_ids, pt, ct = self._cypher_loop(question, seed_ids)
        breakdown["cypher_s"] = round(time.time() - t, 3)
        cypher_used = bool(cypher_new_ids)

        # [4] fallback only when Cypher surfaced 0 new clauses
        fallback_ids: list[str] = []
        fallback_used = False
        if not cypher_used:
            t = time.time()
            fallback_ids = self._fallback_expand(seed_ids, seed_set)
            breakdown["fallback_s"] = round(time.time() - t, 3)
            fallback_used = True  # the path was taken (it may add 0 — honest)

        # [5] fuse (RRF) + materialise top_k_final
        t = time.time()
        extra_ids = _dedupe(cypher_new_ids + fallback_ids)
        extra_rows = self._fetch_clause_rows(extra_ids)

        ranked_lists = [seed_ids]
        if cypher_new_ids:
            ranked_lists.append(cypher_new_ids)
        if fallback_ids:
            ranked_lists.append(fallback_ids)
        fused = self._rrf(ranked_lists)

        # source tag per clause: seed → vector, cypher-new → cypher, else fallback
        source_of: dict[str, str] = {}
        for cid in seed_ids:
            source_of[cid] = "vector"
        for cid in cypher_new_ids:
            source_of.setdefault(cid, "cypher")
        for cid in fallback_ids:
            source_of.setdefault(cid, "fallback_expand")

        seed_meta = {h.clause_id: h for h in seed_hits}
        first_seen = {cid: i for i, cid in enumerate(
            seed_ids + cypher_new_ids + fallback_ids
        )}
        ordered_keys = sorted(
            fused.keys(), key=lambda c: (-fused[c], first_seen.get(c, 1 << 30))
        )[: self.top_k_final]

        hits: list[RetrievedClause] = []
        for cid in ordered_keys:
            if cid in seed_meta:
                h = seed_meta[cid]
                hits.append(RetrievedClause(
                    clause_id=cid, article_id=h.article_id,
                    article_number=h.article_n, article_title=h.article_title,
                    clause_number=h.clause_n, text=h.text,
                    score=round(fused[cid], 6), source=source_of.get(cid, "vector"),
                ))
            elif cid in extra_rows:
                r = extra_rows[cid]
                hits.append(RetrievedClause(
                    clause_id=cid, article_id=r["article_id"],
                    article_number=r.get("article_number"),
                    article_title=r.get("article_title") or "",
                    clause_number=r.get("clause_number"),
                    text=r.get("text") or "",
                    score=round(fused[cid], 6),
                    source=source_of.get(cid, "cypher"),
                ))
            # else: a fused key we could not materialise (clause vanished) — drop
        breakdown["fuse_s"] = round(time.time() - t, 3)

        return CypherWalkResult(
            question=question,
            hits=hits,
            seed_clause_ids=seed_ids,
            cypher_new_clause_ids=cypher_new_ids,
            fallback_clause_ids=fallback_ids,
            n_seed=len(seed_ids),
            n_cypher_new=len(cypher_new_ids),
            n_fallback_added=len(fallback_ids),
            cypher_used=cypher_used,
            fallback_used=fallback_used,
            cypher_attempts=attempts,
            elapsed_s=round(time.time() - t0, 3),
            elapsed_breakdown=breakdown,
            prompt_tokens=pt,
            completion_tokens=ct,
            config={
                "top_k_seed": self.top_k_seed,
                "top_k_final": self.top_k_final,
                "max_repair_rounds": self.max_repair_rounds,
                "cypher_model": self.cypher_model,
                "rrf_k": self.rrf_k,
                "seed_mode": self.seed_mode,
            },
        )


# ---------------------------------------------------------------------------
# CLI — single question, for ad-hoc inspection (NOT an experiment run)
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-q", "--question", required=True)
    p.add_argument("--top-k-seed", type=int, default=8)
    p.add_argument("--top-k-final", type=int, default=12)
    p.add_argument("--max-repair-rounds", type=int, default=DEFAULT_MAX_REPAIR_ROUNDS)
    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    rag = RagPipeline()
    retr = CypherWalkRetriever(
        rag,
        top_k_seed=args.top_k_seed,
        top_k_final=args.top_k_final,
        max_repair_rounds=args.max_repair_rounds,
    )
    try:
        res = retr.retrieve(args.question)
        print(f"\n=== QUESTION ===\n{res.question}")
        print(f"\nseed={res.n_seed}  cypher_new={res.n_cypher_new}  "
              f"fallback_added={res.n_fallback_added}  "
              f"cypher_used={res.cypher_used}  fallback_used={res.fallback_used}")
        print(f"\n=== CYPHER ATTEMPTS ({len(res.cypher_attempts)}) ===")
        for a in res.cypher_attempts:
            print(f"\n[round {a.round}] valid={a.valid} executed={a.executed} "
                  f"n_rows={a.n_rows} n_new={a.n_new_clauses}")
            print(f"rationale: {a.rationale}")
            print((a.cypher or "")[:500])
            if a.validation_error:
                print(f"VAL ERR: {a.validation_error}")
            if a.execution_error:
                print(f"EXE ERR: {a.execution_error}")
        print(f"\n=== TOP {len(res.hits)} HITS ===")
        for i, h in enumerate(res.hits, 1):
            print(f"{i:>2}. [{h.source:<15}] {h.clause_id}  score={h.score:.5f}  "
                  f"{(h.text or '')[:70]}")
        print(f"\nElapsed: {res.elapsed_s}s  breakdown: {res.elapsed_breakdown}")
        print(f"Tokens: prompt={res.prompt_tokens} completion={res.completion_tokens}")
    finally:
        rag.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
