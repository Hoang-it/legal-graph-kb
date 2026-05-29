"""Cross-encoder reranker wrapper.

Default model: ``BAAI/bge-reranker-v2-m3`` — same family as BGE-M3 retriever,
multilingual (incl. Vietnamese), ~568M params. Runs locally on the same GPU.

The model emits a *raw logit*; passing ``apply_sigmoid=True`` returns the
probability-style score in [0, 1] which is easier to threshold and report.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Sequence

# Defer heavy imports
_DEFAULT_MODEL = os.getenv("V5_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
_DEFAULT_DEVICE = os.getenv("V5_RERANKER_DEVICE") or os.getenv("EMBED_DEVICE", "cuda")
_DEFAULT_BATCH = int(os.getenv("V5_RERANKER_BATCH", "4"))


class CrossEncoderReranker:
    """Lazy-loaded cross-encoder.

    Score interpretation:
        score = sigmoid(model(query, candidate))  ∈ [0, 1]
    Higher = more relevant. Sort descending → top-k.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = _DEFAULT_DEVICE,
        batch_size: int = _DEFAULT_BATCH,
    ):
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            print(
                f"Loading reranker {self._model_name} on {self._device}...",
                file=sys.stderr,
            )
            t0 = time.time()
            self._model = CrossEncoder(self._model_name, device=self._device)
            print(f"  loaded in {time.time() - t0:.1f}s", file=sys.stderr)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Return ``(original_index, score)`` pairs sorted by score-desc.

        ``original_index`` lets the caller map back to its own dataclass list
        without zipping candidates back.
        """
        if not candidates:
            return []
        import torch

        pairs = [(query, text) for text in candidates]
        scores = self.model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            activation_fct=torch.nn.Sigmoid(),
        )
        # Release transient activations so VRAM fragmentation doesn't accumulate
        # across questions on small GPUs (e.g. RTX 3050 4 GB).
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ranked = sorted(
            enumerate(float(s) for s in scores),
            key=lambda kv: kv[1],
            reverse=True,
        )
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked
