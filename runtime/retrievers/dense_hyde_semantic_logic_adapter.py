"""Adapter: dùng `dense_hyde_semantic` (Plan v5 retrieval) làm retriever cho logic-lm.

Khác `graphrag_retriever_adapter.py` (vector-search trên raw question), arm này:

1. build concept-frame BHXH — `runtime.retrievers.semantic_context.build_semantic_context`,
2. sinh 1 đoạn **hypothesis** grounded trên frame — `OpenAISemanticHydeGenerator`,
3. dense-search bằng embedding của hypothesis (BGE-M3 mean-pool),

rồi expose CẢ `chunks` (để logic-lm cite) LẪN `last_hypothesis` (để bơm vào bước sinh
Prolog). Hypothesis là sản phẩm phụ tất yếu của dense_hyde_semantic — control arm dùng
cùng adapter này nhưng KHÔNG bơm hypothesis vào rule-gen, nên cô lập đúng một biến.

Không sửa code trong `runtime/logic_lm/`. `last_*` được set sau mỗi `retrieve()` để
pipeline đọc đồng bộ (mỗi câu hỏi xử lý tuần tự).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Make repo root importable for absolute `runtime.*` / `src.*` paths.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runtime.logic_lm.knowledge.hybrid_retrieval import (
    RetrievedKnowledgeChunk,
    RetrievedKnowledgeContext,
)
from runtime.retrievers.semantic_context import (
    DEFAULT_ONTOLOGY_PATH,
    SemanticContext,
    build_semantic_context,
)
from src.retrieval.hyde_semantic import OpenAISemanticHydeGenerator
from src.retrieval.pipeline import V5RetrievalPipeline

DEFAULT_CACHE_DIR = "artifacts/logic_lm_hyde_semantic/hyde_semantic"


class DenseHydeSemanticAsLogicLMRetriever:
    """Wrap dense_hyde_semantic retrieval as a logic-lm retriever.

    Exposes the generated hypothesis passage via ``last_hypothesis`` and the
    matched concept frame via ``last_semantic_context`` so the logic-lm pipeline
    (and a UI) can surface the full reasoning chain.
    """

    def __init__(
        self,
        pipeline: Optional[V5RetrievalPipeline] = None,
        ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH,
        hyde_model: Optional[str] = None,
        hyde_n: int = 1,
        hyde_max_tokens: int = 700,
        temperature: float = 0.0,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ) -> None:
        self.ontology_path = str(ontology_path)
        self.last_hypothesis: str = ""
        self.last_semantic_context: Optional[SemanticContext] = None
        if pipeline is None:
            generator = OpenAISemanticHydeGenerator(
                model=hyde_model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
                n=hyde_n,
                cache_dir=cache_dir,
                max_tokens=hyde_max_tokens,
                temperature=temperature,
            )
            pipeline = V5RetrievalPipeline(hyde_semantic=generator)
        self._pipe = pipeline
        _ = self._pipe.embed_model  # warm BGE-M3 so per-question timings are clean

    def retrieve(self, query: str, top_k: int = 8) -> RetrievedKnowledgeContext:
        self.last_hypothesis = ""
        self.last_semantic_context = None
        if not query or top_k <= 0:
            return RetrievedKnowledgeContext(chunks=[], scores={})

        ctx = build_semantic_context(query, ontology_path=self.ontology_path)
        rows, docs = self._pipe.dense_hyde_semantic_rows(
            query,
            frame_text=ctx.frame_text,
            context_key_ids=ctx.context_key_ids,
            top_k=top_k,
        )
        self.last_semantic_context = ctx
        self.last_hypothesis = docs[0] if docs else ""

        chunks: list[RetrievedKnowledgeChunk] = []
        scores: dict[str, float] = {}
        for r in rows:
            clause_id = r["clause_id"]
            chunks.append(
                RetrievedKnowledgeChunk(
                    id=clause_id,
                    text=str(r.get("text") or ""),
                    document=self._pipe._law_display(str(r.get("law_id") or "")),
                    article=str(r.get("article_n") or ""),
                    clause=str(r.get("clause_n") or ""),
                    point=None,
                )
            )
            scores[clause_id] = float(r.get("score") or 0.0)
        return RetrievedKnowledgeContext(chunks=chunks, scores=scores)

    def verify_citations(self, ids: list[str]) -> dict[str, bool]:
        """Delegate citation verification to the underlying V5 pipeline (Neo4j)."""
        return self._pipe.verify_citations(ids)

    def close(self) -> None:
        try:
            self._pipe.close()
        except Exception:
            pass
