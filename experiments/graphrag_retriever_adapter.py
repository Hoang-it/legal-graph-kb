"""Adapter: dùng main GraphRAG (`src/rag_query.py`) làm retriever cho elite.

Elite pipeline expect `context` object có attribute `.chunks: List[
RetrievedKnowledgeChunk]` và `.scores: Dict[str, float]`. GraphRAG trả
về `List[SearchHit]` (clause_id, score, text, article_n, clause_n, ...).
Adapter convert.

Mục đích: cho phép arm `elite_graphrag` dùng Neo4j vector search thay vì
elite's ontology/hybrid retrieval, để compare retrieval quality.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make elite importable trước khi import RetrievedKnowledge*
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ELITE_ROOT = _REPO_ROOT / "elite"
for _p in (_REPO_ROOT, _ELITE_ROOT, _ELITE_ROOT / "src"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from knowledge.hybrid_retrieval import (  # type: ignore
    RetrievedKnowledgeChunk,
    RetrievedKnowledgeContext,
)

DOCUMENT_LABEL = "Luật BHXH 2024 (41/2024/QH15)"


class GraphRAGAsEliteRetriever:
    """Wrap `RagPipeline` từ src/rag_query.py để dùng làm retriever cho elite.

    Khác biệt với native elite retrievers:
    - Score = cosine similarity từ Neo4j vector index (range 0..1)
    - Chunk text = Clause.text gốc trong KG (đã có embedding)
    - ID format = "L41_2024.A<n>.K<m>" thay vì "c<seq>"
    - Document fixed = "Luật BHXH 2024 (41/2024/QH15)"
    - Point luôn None (vì vector index ở Clause level, không Point level)
    """

    def __init__(self, rag_pipeline):
        # rag_pipeline = instance của src.rag_query.RagPipeline
        # Đã pre-load model + connect Neo4j
        self._rag = rag_pipeline

    def retrieve(self, query: str, top_k: int = 8) -> RetrievedKnowledgeContext:
        if not query or top_k <= 0:
            return RetrievedKnowledgeContext(chunks=[], scores={})

        hits = self._rag.vector_search(query, top_k=top_k)
        chunks = [
            RetrievedKnowledgeChunk(
                id=h.clause_id,
                text=h.text,
                document=DOCUMENT_LABEL,
                article=str(h.article_n),
                clause=str(h.clause_n),
                point=None,
            )
            for h in hits
        ]
        scores = {h.clause_id: float(h.score) for h in hits}
        return RetrievedKnowledgeContext(chunks=chunks, scores=scores)


class GraphRAGLogicAsEliteRetriever(GraphRAGAsEliteRetriever):
    """Adapter cho elite_graphrag_logic arm — vector hits PLUS pre-extracted facts.

    Khác `GraphRAGAsEliteRetriever`:
    - Mỗi chunk.text = raw clause text + `=== FACTS ===` block kèm rules/conditions
      /thresholds đã extract sẵn từ Phase 2.
    - Thêm 1 chunk "synthetic" cuối cùng chứa block `# REFERENCES` (multi-hop
      REFERS_TO) nếu có refs.

    LLM được instruct (qua prompt `elite_graphrag_logic.md`) để xài FACTS thay
    vì hallucinate predicate names / threshold values từ raw text.
    """

    def __init__(self, rag_pipeline, max_hops: int = 1, include_references: bool = True):
        super().__init__(rag_pipeline)
        self.max_hops = max_hops
        self.include_references = include_references

    def retrieve(self, query: str, top_k: int = 8) -> RetrievedKnowledgeContext:
        if not query or top_k <= 0:
            return RetrievedKnowledgeContext(chunks=[], scores={})

        # Use full hybrid_search — leverages Phase 3 API
        result = self._rag.hybrid_search(query, top_k=top_k, max_hops=self.max_hops)
        # Group facts by source clause for inlining
        from collections import defaultdict
        facts_by_clause: dict[str, list] = defaultdict(list)
        for f in result.facts:
            facts_by_clause[f.clause_id].append(f)

        chunks = []
        for h in result.hits:
            text = h.text
            cfacts = facts_by_clause.get(h.clause_id, [])
            if cfacts:
                # Render facts inline using same formatter
                from src.rag_query import HybridResult
                sub = HybridResult(hits=[h], facts=cfacts, referenced=[])
                fact_block = self._rag.format_facts_for_prompt(sub, max_chars=2000)
                text = f"{h.text}\n\n=== FACTS ===\n{fact_block}"
            chunks.append(
                RetrievedKnowledgeChunk(
                    id=h.clause_id,
                    text=text,
                    document=DOCUMENT_LABEL,
                    article=str(h.article_n),
                    clause=str(h.clause_n),
                    point=None,
                )
            )

        # Append synthetic chunk for cross-references (multi-hop)
        if self.include_references and result.referenced:
            ref_lines = ["# REFERENCES — Điều khoản viện dẫn (multi-hop từ retrieved clauses)"]
            for r in result.referenced[:15]:
                ref_lines.append(
                    f"- {r['source_clause_id']} → {r['target_id']} "
                    f"(hop={r['hop_distance']}, {r['target_label']}: "
                    f"{(r.get('target_title') or '')[:80]})"
                )
            chunks.append(
                RetrievedKnowledgeChunk(
                    id="__refs__",
                    text="\n".join(ref_lines),
                    document=DOCUMENT_LABEL,
                    article=None,
                    clause=None,
                    point=None,
                )
            )

        scores = {h.clause_id: float(h.score) for h in result.hits}
        return RetrievedKnowledgeContext(chunks=chunks, scores=scores)


if __name__ == "__main__":
    # Smoke test
    import os
    from dotenv import load_dotenv
    load_dotenv()
    if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
        os.environ.pop("OPENAI_BASE_URL", None)

    from src.rag_query import RagPipeline  # noqa

    rag = RagPipeline()
    try:
        _ = rag.embed_model  # warm up
        adapter = GraphRAGAsEliteRetriever(rag)
        ctx = adapter.retrieve("Bảo hiểm xã hội là gì?", top_k=5)
        print(f"[semantic] Retrieved {len(ctx.chunks)} chunks:")
        for c in ctx.chunks:
            print(f"  [{c.id}] art={c.article} cl={c.clause}  "
                  f"score={ctx.scores.get(c.id, 0):.3f}  text={c.text[:80]}")

        adapter2 = GraphRAGLogicAsEliteRetriever(rag, max_hops=1)
        ctx2 = adapter2.retrieve("Điều kiện hưởng lương hưu?", top_k=5)
        print(f"\n[logic] Retrieved {len(ctx2.chunks)} chunks:")
        for c in ctx2.chunks[:3]:
            print(f"  [{c.id}] text (250ch): {c.text[:250]}")
            print()
    finally:
        rag.close()
