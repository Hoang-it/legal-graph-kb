"""M4 — Hybrid retrieval.

Combines three independent signal sources, then fuses by Reciprocal Rank Fusion:

1. **Dense**  — BGE-M3 1024-d cosine via Neo4j ``clause_vec`` vector index.
2. **Sparse** — Lucene BM25 via Neo4j FULLTEXT index ``clause_fulltext``.
3. **Temporal filter** — drop clauses whose owning Law was not in force at
   the event_date detected from the question (or *today* if none).

Output: ``list[Candidate]`` of length up to ``top_after_fusion`` (default 50),
sorted by RRF score descending. Each Candidate carries provenance (clause_id,
article_id, article_n, article_title, clause_n, text, law_id) and the per-list
ranks + RRF score for downstream auditing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    clause_id: str
    article_id: str
    article_n: int
    article_title: str
    clause_n: int
    law_id: str
    text: str
    dense_rank: int | None = None      # 1-based; None = not present in dense list
    sparse_rank: int | None = None
    dense_score: float | None = None   # cosine sim
    sparse_score: float | None = None  # bm25
    rrf_score: float = 0.0


# ---------------------------------------------------------------------------
# Event date detection — data-driven, no hardcoded law mapping
# ---------------------------------------------------------------------------

_RE_DATE_DMY = re.compile(r"\b(\d{1,2})[\/\.\-](\d{1,2})[\/\.\-](\d{4})\b")
_RE_YEAR = re.compile(r"\b(19[5-9]\d|20\d{2})\b")


def detect_event_date(question: str, default: date | None = None) -> date:
    """Detect the latest legally-relevant date mentioned in the question.

    Priority: explicit DD/MM/YYYY > standalone YYYY. Picks the *latest* hit so
    that a question referencing both an old event and a current law-version
    resolves to the current state.
    """
    candidates: list[date] = []
    for m in _RE_DATE_DMY.finditer(question):
        try:
            candidates.append(date(int(m.group(3)), int(m.group(2)), int(m.group(1))))
        except ValueError:
            pass
    for m in _RE_YEAR.finditer(question):
        try:
            candidates.append(date(int(m.group(1)), 12, 31))
        except ValueError:
            pass
    if candidates:
        return max(candidates)
    return default or date.today()


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: Sequence[list[str]],
    k: int = 60,
) -> dict[str, float]:
    """Cormack et al. 2009. ``score(d) = Σ_i 1/(k + rank_i(d))``.

    Only rank position matters — score scales of the input lists are ignored.
    Returns dict from doc id → fused score, sorted by score-desc when iterated.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------


@dataclass
class RetrievalAudit:
    """Per-query diagnostics — surfaced in records for Sprint 1 audit."""
    event_date: str = ""
    n_dense: int = 0
    n_sparse: int = 0
    n_after_temporal: int = 0
    n_after_fusion: int = 0
    n_dropped_by_temporal: int = 0


class HybridRetriever:
    """Combines dense + sparse + temporal filter + RRF.

    Single source of truth for Cypher queries: kept here so swapping sparse
    backends or changing the index name happens in one place.

    The dense / sparse index names are instance attributes so a tuned
    BGE-M3 corpus loaded under ``clause_vec_tuned`` (Sprint 2 Phase 3) can
    coexist with the vanilla ``clause_vec`` for A/B comparison.
    """

    DEFAULT_DENSE_INDEX = "clause_vec"
    DEFAULT_SPARSE_INDEX = "clause_fulltext"

    def __init__(
        self,
        driver,
        db: str,
        embed_model,
        dense_k: int = 30,
        sparse_k: int = 30,
        top_after_fusion: int = 50,
        rrf_k: int = 60,
        dense_index: str | None = None,
        sparse_index: str | None = None,
    ):
        self._driver = driver
        self._db = db
        self._embed_model = embed_model
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.top_after_fusion = top_after_fusion
        self.rrf_k = rrf_k
        self.dense_index = dense_index or self.DEFAULT_DENSE_INDEX
        self.sparse_index = sparse_index or self.DEFAULT_SPARSE_INDEX

    # ------------------------------------------------------------------
    # Per-signal queries
    # ------------------------------------------------------------------

    def _dense_search(self, query: str, top_k: int) -> list[dict]:
        q_emb = self._embed_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()
        with self._driver.session(database=self._db) as s:
            rows = s.run(
                f"""
                CALL db.index.vector.queryNodes('{self.dense_index}', $k, $q)
                YIELD node, score
                MATCH (a:Article)-[:HAS_CLAUSE]->(node)
                MATCH (a)<-[:HAS_ARTICLE]-(:Chapter)<-[:HAS_CHAPTER]-(law:Law)
                RETURN node.id AS clause_id, node.text AS text, score AS score,
                       a.id AS article_id, a.title AS article_title,
                       a.number AS article_n, node.number AS clause_n,
                       law.id AS law_id
                ORDER BY score DESC
                """,
                k=top_k,
                q=q_emb,
            ).data()
        return rows

    def _sparse_search(self, query: str, top_k: int) -> list[dict]:
        # Neo4j FULLTEXT requires Lucene-safe query: escape syntax chars.
        cleaned = _lucene_escape(query)
        with self._driver.session(database=self._db) as s:
            rows = s.run(
                f"""
                CALL db.index.fulltext.queryNodes('{self.sparse_index}', $q)
                YIELD node, score
                WHERE node:Clause
                WITH node, score ORDER BY score DESC LIMIT $k
                MATCH (a:Article)-[:HAS_CLAUSE]->(node)
                MATCH (a)<-[:HAS_ARTICLE]-(:Chapter)<-[:HAS_CHAPTER]-(law:Law)
                RETURN node.id AS clause_id, node.text AS text, score AS score,
                       a.id AS article_id, a.title AS article_title,
                       a.number AS article_n, node.number AS clause_n,
                       law.id AS law_id
                """,
                q=cleaned,
                k=top_k,
            ).data()
        return rows

    # ------------------------------------------------------------------
    # Temporal filter — uses Law.effective_date + REPEALS edges
    # ------------------------------------------------------------------

    def _in_force_law_ids(self, event_date: date) -> set[str]:
        """Return Law ids in force at event_date.

        A law is in force iff:
        - ``effective_date`` ≤ event_date (or effective_date is null — treat as
          always-in-force, conservative).
        - no Law with REPEALS edge to it has ``effective_date`` ≤ event_date.
        """
        with self._driver.session(database=self._db) as s:
            rows = s.run(
                """
                MATCH (law:Law)
                OPTIONAL MATCH (newer:Law)-[:REPEALS]->(law)
                WITH law,
                     collect(newer.effective_date) AS repealers
                RETURN law.id AS law_id,
                       law.effective_date AS effective_from,
                       repealers
                """
            ).data()
        in_force: set[str] = set()
        for r in rows:
            eff_from = r.get("effective_from")
            # Conservative: missing effective_date → treat as in force.
            from_ok = (eff_from is None) or (_to_date(eff_from) <= event_date)
            repealed = False
            for rep in r.get("repealers") or []:
                if rep is None:
                    continue
                if _to_date(rep) <= event_date:
                    repealed = True
                    break
            if from_ok and not repealed:
                in_force.add(r["law_id"])
        return in_force

    # ------------------------------------------------------------------
    # Pipeline entry
    # ------------------------------------------------------------------

    def retrieve(self, query: str, event_date: date | None = None) -> tuple[list[Candidate], RetrievalAudit]:
        ev_date = event_date or detect_event_date(query)
        audit = RetrievalAudit(event_date=ev_date.isoformat())

        dense_rows = self._dense_search(query, self.dense_k)
        sparse_rows = self._sparse_search(query, self.sparse_k)
        audit.n_dense = len(dense_rows)
        audit.n_sparse = len(sparse_rows)

        # Build unified candidate map first (before temporal filter so we can
        # report dropped count honestly).
        by_id: dict[str, Candidate] = {}
        dense_ranked: list[str] = []
        for rank, r in enumerate(dense_rows, start=1):
            cid = r["clause_id"]
            by_id.setdefault(cid, _row_to_candidate(r))
            by_id[cid].dense_rank = rank
            by_id[cid].dense_score = float(r["score"])
            dense_ranked.append(cid)

        sparse_ranked: list[str] = []
        for rank, r in enumerate(sparse_rows, start=1):
            cid = r["clause_id"]
            by_id.setdefault(cid, _row_to_candidate(r))
            by_id[cid].sparse_rank = rank
            by_id[cid].sparse_score = float(r["score"])
            sparse_ranked.append(cid)

        # Temporal filter — drop candidates whose law was not in force
        in_force = self._in_force_law_ids(ev_date)
        before = len(by_id)
        kept_ids = {cid for cid, c in by_id.items() if c.law_id in in_force}
        audit.n_dropped_by_temporal = before - len(kept_ids)
        audit.n_after_temporal = len(kept_ids)

        # RRF — fuse the two rank lists (after restricting to kept ids)
        dense_kept = [d for d in dense_ranked if d in kept_ids]
        sparse_kept = [d for d in sparse_ranked if d in kept_ids]
        fused = reciprocal_rank_fusion([dense_kept, sparse_kept], k=self.rrf_k)

        ordered: list[Candidate] = []
        for cid in fused.keys():
            c = by_id[cid]
            c.rrf_score = fused[cid]
            ordered.append(c)
            if len(ordered) >= self.top_after_fusion:
                break

        audit.n_after_fusion = len(ordered)
        return ordered, audit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Lucene syntax characters that must be escaped in user queries.
_LUCENE_SPECIALS = r'+-&|!(){}[]^"~*?:\\/'


def _lucene_escape(q: str) -> str:
    """Escape Lucene query-syntax characters so user text is treated as terms.

    Neo4j FULLTEXT passes the string straight to Lucene's QueryParser, which
    breaks on unescaped ``? : / "`` etc. that appear in Vietnamese questions.
    """
    out = []
    for ch in q:
        if ch in _LUCENE_SPECIALS:
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _row_to_candidate(r: dict) -> Candidate:
    return Candidate(
        clause_id=r["clause_id"],
        article_id=r["article_id"],
        article_n=int(r["article_n"]),
        article_title=str(r.get("article_title") or ""),
        clause_n=int(r["clause_n"]),
        law_id=str(r.get("law_id") or ""),
        text=str(r.get("text") or ""),
    )


def _to_date(value) -> date:
    """Coerce Neo4j date / ISO string / datetime to ``date``.

    Neo4j driver returns ``neo4j.time.Date`` for date-typed properties; if the
    schema stored a plain string we fall back to ISO parsing.
    """
    if hasattr(value, "to_native"):  # neo4j.time.Date
        try:
            return value.to_native()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return date(value.year, value.month, value.day)
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise ValueError(f"Cannot coerce {value!r} ({type(value).__name__}) to date")
