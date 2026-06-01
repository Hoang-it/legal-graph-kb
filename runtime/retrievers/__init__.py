"""Retrieval-layer components.

This package houses *retrieval-layer* components — things that take a
question and return a ranked set of clauses + provenance, with **no** LLM
render, **no** citation parsing, and **no** answer generation. They are
peers of ``RagPipeline.vector_search`` and ``V5RetrievalPipeline`` and are
meant to be drop-in replacements for any code path that today calls
``vector_search``.

``RagPipeline`` itself stays in ``runtime/rag_query.py`` because it is an
*arm* (it renders an answer), not a pure retriever.

Current members:

- :class:`CypherWalkRetriever` — vector seed → LLM-authored outward Cypher
  walk → fallback expand → RRF fusion.
"""

from runtime.retrievers.cypher_walk import (
    CypherAttempt,
    CypherWalkResult,
    CypherWalkRetriever,
    RetrievedClause,
    validate_cypher,
)

__all__ = [
    "CypherWalkRetriever",
    "CypherWalkResult",
    "RetrievedClause",
    "CypherAttempt",
    "validate_cypher",
]
