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
        print(f"Retrieved {len(ctx.chunks)} chunks:")
        for c in ctx.chunks:
            print(f"  [{c.id}] art={c.article} cl={c.clause}  "
                  f"score={ctx.scores.get(c.id, 0):.3f}  text={c.text[:80]}")
    finally:
        rag.close()
