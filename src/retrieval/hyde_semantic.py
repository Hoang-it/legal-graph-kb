"""HyDE-Semantic — concept-frame-grounded HyDE generator (exp 13).

Like :class:`src.retrieval.hyde2.OpenAIGroundedHydeGenerator`, but the
grounding context is a **BHXH concept frame** built by
``runtime.retrievers.semantic_context.build_semantic_context`` (query →
concepts/subjects/benefits, NO dense clause seed) instead of clause text
from a dense seed pass.

Differences vs the base generators:

- Default ``prompt_path`` → ``runtime/hyde_generate_semantic.md``.
- Default ``cache_dir`` → ``artifacts/hyde_semantic`` (own namespace; never
  collides with ``artifacts/hyde`` / ``artifacts/hyde2``).
- :meth:`generate` takes ``(question, frame_text, context_key_ids)`` — the
  frame string goes straight into ``{context}`` (no ``[i]`` wrapping), and the
  cache key is extended with a hash of ``sorted(context_key_ids)`` (the matched
  concept + KG-entity ids). Since ``build_semantic_context`` is deterministic,
  the key is stable → idempotent, $0 on re-run; it changes only when the
  ontology / concept set changes.
- Soft fallback: an empty frame (no concept matched) is allowed; the prompt
  writes a general BHXH passage (≈ HyDE1). The sentinel key id
  ``__no_concept__`` keeps the cache key well-defined.

Cache/cost/retry/async machinery is reused verbatim from the base classes.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from openai import AsyncOpenAI

from src.retrieval.hyde import (
    _CachePayload,
    _call_async_with_retry,
    _call_sync_with_retry,
    _estimate_cost_usd,
    _strip_openai_base_url_if_blank,
)
from src.retrieval.hyde2 import OpenAIGroundedHydeGenerator

_NO_CONCEPT_KEY = "__no_concept__"


def _key_ids(context_key_ids: Sequence[str] | None) -> list[str]:
    ids = [str(x) for x in (context_key_ids or []) if str(x)]
    return ids or [_NO_CONCEPT_KEY]


class OpenAISemanticHydeGenerator(OpenAIGroundedHydeGenerator):
    """HyDE generator conditioned on a BHXH concept frame (no dense seed)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        n: int = 1,
        cache_dir: str | Path = "artifacts/hyde_semantic",
        prompt_path: str = "runtime/hyde_generate_semantic.md",
        max_tokens: int = 700,
        temperature: float = 0.0,
        concurrency: int = 5,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        super().__init__(
            model=model,
            n=n,
            cache_dir=cache_dir,
            prompt_path=prompt_path,
            max_tokens=max_tokens,
            temperature=temperature,
            concurrency=concurrency,
            api_key_env=api_key_env,
        )

    # ------------------------------------------------------------------
    # Sync single-question generate (cache-aware)
    # ------------------------------------------------------------------

    def generate(  # type: ignore[override]
        self,
        question: str,
        frame_text: str | None = None,
        context_key_ids: Sequence[str] | None = None,
    ) -> list[str]:
        if frame_text is None or context_key_ids is None:
            raise TypeError(
                "OpenAISemanticHydeGenerator.generate requires frame_text and "
                "context_key_ids (use build_semantic_context())."
            )
        key_ids = _key_ids(context_key_ids)
        key = self._cache_key_grounded(question, key_ids)
        cached = self._cache_get(key)
        if cached is not None:
            self._record_cache_hit()
            return list(cached["generated_docs"])

        user_filled = self._user_tmpl.replace("{context}", frame_text).replace(
            "{question}", question
        )
        client = self.sync_client
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        docs: list[str] = []
        model_returned = self.model
        for _ in range(self.n):
            doc, usage, m_ret = _call_sync_with_retry(
                client=client,
                model=self.model,
                system=self._system,
                user=user_filled,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            docs.append(doc)
            for kk in usage_total:
                usage_total[kk] += int(usage.get(kk) or 0)
            model_returned = m_ret

        self._write_payload(question, key, key_ids, frame_text, docs, usage_total, model_returned)
        self._record_call(usage_total, _estimate_cost_usd(self.model, usage_total))
        return docs

    # ------------------------------------------------------------------
    # Batched async generate
    # ------------------------------------------------------------------

    def generate_batch(  # type: ignore[override]
        self,
        triples: Sequence[tuple[str, str, Sequence[str]]],
    ) -> list[list[str]]:
        """Each element is ``(question, frame_text, context_key_ids)``."""
        results: list[list[str] | None] = [None] * len(triples)
        miss: list[int] = []
        for i, (q, _frame, key_ids) in enumerate(triples):
            key = self._cache_key_grounded(q, _key_ids(key_ids))
            cached = self._cache_get(key)
            if cached is not None:
                results[i] = list(cached["generated_docs"])
                self._record_cache_hit()
            else:
                miss.append(i)

        if not miss:
            return [r for r in results if r is not None]

        async def _run() -> dict[int, list[str]]:
            _strip_openai_base_url_if_blank()
            async with AsyncOpenAI(api_key=self._ensure_api_key()) as aclient:
                sem = asyncio.Semaphore(self.concurrency)

                async def _one(idx: int) -> tuple[int, list[str]]:
                    q, frame_text, key_ids_raw = triples[idx]
                    key_ids = _key_ids(key_ids_raw)
                    user_filled = self._user_tmpl.replace(
                        "{context}", frame_text
                    ).replace("{question}", q)
                    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
                    docs: list[str] = []
                    model_returned = self.model
                    async with sem:
                        for _ in range(self.n):
                            doc, usage, m_ret = await _call_async_with_retry(
                                aclient=aclient,
                                model=self.model,
                                system=self._system,
                                user=user_filled,
                                max_tokens=self.max_tokens,
                                temperature=self.temperature,
                            )
                            docs.append(doc)
                            for kk in usage_total:
                                usage_total[kk] += int(usage.get(kk) or 0)
                            model_returned = m_ret
                    key = self._cache_key_grounded(q, key_ids)
                    self._write_payload(
                        q, key, key_ids, frame_text, docs, usage_total, model_returned
                    )
                    self._record_call(usage_total, _estimate_cost_usd(self.model, usage_total))
                    return idx, docs

                return dict(await asyncio.gather(*[_one(i) for i in miss]))

        for i, docs in asyncio.run(_run()).items():
            results[i] = docs
        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # Cache write (shared by sync + async)
    # ------------------------------------------------------------------

    def _write_payload(
        self,
        question: str,
        key: str,
        key_ids: list[str],
        frame_text: str,
        docs: list[str],
        usage_total: dict[str, int],
        model_returned: str,
    ) -> None:
        payload = _CachePayload(
            question=question,
            model_id=self.model,
            model_returned=model_returned,
            prompt_sha=self._prompt_sha,
            n=self.n,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            generated_at=datetime.now(timezone.utc).isoformat(),
            generated_docs=docs,
            usage=usage_total,
            cost_usd=_estimate_cost_usd(self.model, usage_total),
        )
        d = payload.to_dict()
        d["context_key_ids"] = list(key_ids)
        d["frame_text"] = frame_text
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path(key).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self._cache_path(key))


__all__ = ["OpenAISemanticHydeGenerator"]
