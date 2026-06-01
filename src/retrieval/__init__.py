"""Plan v5 Sprint 1 — vanilla hybrid retrieval pipeline.

Public entry point: :class:`V5RetrievalPipeline`.

Module boundary:
- ``hybrid_retriever`` — M4: BGE-M3 dense + Lucene BM25 + temporal filter + RRF fusion
- ``reranker``         — Cross-encoder wrapper (BAAI/bge-reranker-v2-m3)
- ``graph_expansion``  — M5: REFERS_TO 2-3 hop traversal in Neo4j
- ``pipeline``         — V5RetrievalPipeline.ask() → V5Answer dataclass
"""
from src.retrieval.pipeline import V5Answer, V5RetrievalPipeline

__all__ = ["V5RetrievalPipeline", "V5Answer"]
