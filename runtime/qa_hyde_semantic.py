"""QA arm `qa_hyde_semantic`: dense_hyde_semantic retrieval → direct generation.

Same retrieval as the logic-LM-hyde-semantic arms (concept-frame →
HyDE-semantic → dense BGE-M3), but the answer comes straight from the GraphRAG
generator (`V5RetrievalPipeline.ask_dense_hyde_semantic`) instead of a Prolog
program — this is the "hyde only" arm that isolates the contribution of the
logic-LM layer. Shares the HyDE cache dir with the logic-LM arms so the
hypothesis generation is $0 on repeated questions.

Returns the same `V5Answer` dataclass as the `graphrag_v5` arm, so the inference
runner pattern in `eval_core/inference.py` plugs in without bespoke handling.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Make repo root importable for absolute `runtime.*` / `src.*` paths.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runtime.retrievers.semantic_context import (
    DEFAULT_ONTOLOGY_PATH,
    build_semantic_context,
)
from src.retrieval.hyde_semantic import OpenAISemanticHydeGenerator
from src.retrieval.pipeline import V5Answer, V5RetrievalPipeline

DEFAULT_CACHE_DIR = "artifacts/logic_lm_hyde_semantic/hyde_semantic"


class QAHydeSemanticPipeline:
    """dense_hyde_semantic retrieval + direct GraphRAG-style generation (no logic-LM)."""

    arm_name = "qa_hyde_semantic"

    def __init__(
        self,
        pipeline: Optional[V5RetrievalPipeline] = None,
        ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH,
        model: Optional[str] = None,
        hyde_model: Optional[str] = None,
        hyde_n: int = 1,
        hyde_max_tokens: int = 700,
        temperature: float = 0.0,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        top_k: int = 8,
    ) -> None:
        self.ontology_path = str(ontology_path)
        self.top_k = top_k
        if pipeline is None:
            generator = OpenAISemanticHydeGenerator(
                model=hyde_model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
                n=hyde_n,
                cache_dir=cache_dir,
                max_tokens=hyde_max_tokens,
                temperature=temperature,
            )
            pipeline = V5RetrievalPipeline(hyde_semantic=generator, model=model)
        self._pipe = pipeline
        _ = self._pipe.embed_model  # warm BGE-M3 so per-question timings are clean

    def ask(self, question: str) -> V5Answer:
        ctx = build_semantic_context(question, ontology_path=self.ontology_path)
        return self._pipe.ask_dense_hyde_semantic(
            question,
            frame_text=ctx.frame_text,
            context_key_ids=ctx.context_key_ids,
            top_k=self.top_k,
        )

    def verify_citations(self, ids: list[str]) -> dict[str, bool]:
        return self._pipe.verify_citations(ids)

    def close(self) -> None:
        try:
            self._pipe.close()
        except Exception:
            pass
