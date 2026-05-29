"""M5 — Graph-based expansion via ``REFERS_TO`` edges.

For each seed clause, walk REFERS_TO 1..max_hops in Neo4j and bring referenced
*Articles* into the candidate pool. Articles (not clauses) are the natural
expansion unit because REFERS_TO points to Articles in the current schema
(see ``offline/merge_normalize.build_cross_law_refs``).

Plan §6 caveat: REFERS_TO coverage is thin in the current graph
(audit: 25 edges, 2 unique target articles). This module is implemented as
designed; whether it contributes is an empirical question for the Sprint 1
audit, not a design assumption.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Neighbor:
    """An article reachable from a seed clause via REFERS_TO."""
    target_id: str
    target_label: str  # 'Article' or 'Clause' depending on edge target
    target_title: str
    target_text: str
    seed_clause_id: str
    hop_distance: int


class GraphExpander:
    """Walks REFERS_TO from seed clauses; returns deduplicated Articles."""

    def __init__(self, driver, db: str, max_hops: int = 3, per_seed_limit: int = 10):
        self._driver = driver
        self._db = db
        self.max_hops = max(1, min(int(max_hops), 3))   # cap at 3 per plan §6
        self.per_seed_limit = per_seed_limit

    def expand(self, seed_clause_ids: list[str]) -> list[Neighbor]:
        if not seed_clause_ids:
            return []
        # Cypher does not parameterize variable-length path bounds.
        # max_hops is int-validated above.
        cypher = f"""
        UNWIND $cids AS cid
        MATCH (cl:Clause {{id: cid}})
        MATCH path = (cl)-[:REFERS_TO*1..{self.max_hops}]->(target)
        WITH cid, target, length(path) AS hop
        WITH cid, target.id AS target_id,
             labels(target)[0] AS target_label,
             coalesce(target.title, '') AS target_title,
             coalesce(target.text, '') AS target_text,
             min(hop) AS hop_distance
        RETURN cid AS seed_clause_id, target_id, target_label,
               target_title, target_text, hop_distance
        ORDER BY hop_distance ASC
        LIMIT $limit
        """
        with self._driver.session(database=self._db) as s:
            rows = s.run(
                cypher,
                cids=seed_clause_ids,
                limit=self.per_seed_limit * max(1, len(seed_clause_ids)),
            ).data()

        # Dedupe by target_id (keep smallest hop_distance per target)
        seen: dict[str, Neighbor] = {}
        for r in rows:
            tid = r["target_id"]
            if tid in seen and seen[tid].hop_distance <= int(r["hop_distance"]):
                continue
            seen[tid] = Neighbor(
                target_id=tid,
                target_label=str(r.get("target_label") or ""),
                target_title=str(r.get("target_title") or ""),
                target_text=str(r.get("target_text") or ""),
                seed_clause_id=str(r["seed_clause_id"]),
                hop_distance=int(r["hop_distance"]),
            )
        return list(seen.values())
