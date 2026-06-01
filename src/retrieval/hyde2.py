"""HyDE2 — retrieval-grounded iterative HyDE generator.

Extends standard HyDE (:class:`src.retrieval.hyde.OpenAIHydeGenerator`)
with a two-pass retrieval-and-generate flow:

1. **Seed pass** (caller-side, deterministic): top-K BGE-M3 dense
   retrieval on the raw question → K clause passages.
2. **Grounded generation** (this class): feed the K clause passages
   into the LLM prompt as ``{context}``, generate one hypothetical
   document conditioned on the real BHXH vocabulary present in
   those passages.
3. **Final pass** (caller-side): embed the grounded hypothetical doc
   → second dense retrieval.

The grounded generator is a *subclass* of :class:`OpenAIHydeGenerator`
so it inherits the disk cache, atomic writes, async batching,
tenacity retry, and cost accounting. What changes:

- Default ``prompt_path`` → ``runtime/hyde_generate_grounded.md``.
- Default ``cache_dir`` → ``artifacts/hyde2`` (separate namespace so
  HyDE1 / HyDE2 caches never collide even at the same prompt-sha).
- :meth:`_cache_key` is extended with a hash of
  ``sorted(seed_clause_ids)`` — when the seed pass returns a
  different set (because the LoRA model / dense index changed), the
  cache key changes and the LLM is re-called automatically. No
  manual purge needed at the cost of one extra hash field.
- :meth:`generate` takes ``context_passages`` + ``seed_clause_ids``
  as required positional args.
- :meth:`generate_batch` takes pre-computed
  ``list[(question, context_passages, seed_clause_ids)]`` triples.
- No ``embed_query_callable`` — wiring HyDE2 through HybridRetriever's
  ``query_encoder`` slot would create a recursion (the seed pass
  needs the *raw-question* dense encoder, not a HyDE2-augmented
  one). The orchestration lives in
  :meth:`V5RetrievalPipeline.retrieve_dense_only_hyde2` instead.
"""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from openai import AsyncOpenAI

from src.retrieval.hyde import (
    OpenAIHydeGenerator,
    _CachePayload,
    _call_async_with_retry,
    _call_sync_with_retry,
    _estimate_cost_usd,
    _strip_openai_base_url_if_blank,
)


# Default context section template — each clause rendered as a numbered
# block. Caller passes raw clause text; this module is responsible for
# formatting so the cache key only depends on the actual content fed to
# the LLM (not on caller formatting choices).
def _format_context(context_passages: Sequence[str]) -> str:
    parts: list[str] = []
    for i, txt in enumerate(context_passages, start=1):
        parts.append(f"[{i}] {txt.strip()}")
    return "\n\n".join(parts)


def _seed_clause_ids_hash(seed_clause_ids: Sequence[str]) -> str:
    """Stable hash over the seed set — sort first so order doesn't matter."""
    h = hashlib.sha256()
    h.update(",".join(sorted(seed_clause_ids)).encode("utf-8"))
    return h.hexdigest()


class OpenAIGroundedHydeGenerator(OpenAIHydeGenerator):
    """HyDE generator that conditions on retrieved clause context.

    Construction is identical to :class:`OpenAIHydeGenerator` except
    the prompt + cache-dir defaults. The grounded prompt file uses
    both ``{context}`` and ``{question}`` placeholders — the inherited
    ``_split_prompt`` accepts this because its check is "contains
    ``{question}``", not "exactly one placeholder".
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        n: int = 1,
        cache_dir: str | Path = "artifacts/hyde2",
        prompt_path: str = "runtime/hyde_generate_grounded.md",
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
        if "{context}" not in self._user_tmpl:
            raise ValueError(
                "HyDE2 user template missing `{context}` placeholder — "
                "check prompts/runtime/hyde_generate_grounded.md."
            )

    # ------------------------------------------------------------------
    # Cache key — extended with sorted seed_clause_ids hash
    # ------------------------------------------------------------------

    def _cache_key_grounded(
        self, question: str, seed_clause_ids: Sequence[str]
    ) -> str:
        h = hashlib.sha256()
        h.update(question.encode("utf-8"))
        h.update(b"|sha=")
        h.update(self._prompt_sha.encode("utf-8"))
        h.update(b"|n=")
        h.update(str(self.n).encode("utf-8"))
        h.update(b"|model=")
        h.update(self.model.encode("utf-8"))
        h.update(b"|mt=")
        h.update(str(self.max_tokens).encode("utf-8"))
        h.update(b"|temp=")
        h.update(f"{self.temperature:.4f}".encode("utf-8"))
        h.update(b"|seeds=")
        h.update(_seed_clause_ids_hash(seed_clause_ids).encode("utf-8"))
        return h.hexdigest()

    # Block the inherited single-arg cache_key — HyDE2 callers must use
    # the extended signature. Defensive: keeps a stray bare super().generate
    # call from silently writing under the wrong key namespace.
    def _cache_key(self, question: str) -> str:  # type: ignore[override]
        raise RuntimeError(
            "OpenAIGroundedHydeGenerator._cache_key requires seed_clause_ids — "
            "use _cache_key_grounded(question, seed_clause_ids) instead."
        )

    # ------------------------------------------------------------------
    # Sync single-question generate (cache-aware)
    # ------------------------------------------------------------------

    def generate(  # type: ignore[override]
        self,
        question: str,
        context_passages: Sequence[str] | None = None,
        seed_clause_ids: Sequence[str] | None = None,
    ) -> list[str]:
        """Generate N grounded hypothetical-doc passages.

        ``context_passages`` and ``seed_clause_ids`` are required. The
        Sequence|None defaults exist only so the signature is technically
        valid as an override; passing None raises immediately.
        """
        if context_passages is None or seed_clause_ids is None:
            raise TypeError(
                "OpenAIGroundedHydeGenerator.generate requires "
                "context_passages and seed_clause_ids."
            )
        if not seed_clause_ids:
            raise ValueError("seed_clause_ids must be non-empty")
        if not context_passages:
            raise ValueError("context_passages must be non-empty")

        key = self._cache_key_grounded(question, seed_clause_ids)
        cached = self._cache_get(key)
        if cached is not None:
            self._record_cache_hit()
            return list(cached["generated_docs"])

        context_block = _format_context(context_passages)
        user_filled = self._user_tmpl.replace("{context}", context_block).replace(
            "{question}", question
        )

        client = self.sync_client
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        docs: list[str] = []
        model_returned: str = self.model
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

        cost = _estimate_cost_usd(self.model, usage_total)
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
            cost_usd=cost,
        )
        # Augment payload with grounding-specific fields. _CachePayload
        # is a dataclass; tack the extras onto the dict at write time so
        # the on-disk schema is auditable.
        payload_dict = payload.to_dict()
        payload_dict["seed_clause_ids"] = list(seed_clause_ids)
        payload_dict["seed_clause_ids_hash"] = _seed_clause_ids_hash(seed_clause_ids)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        import json
        import os
        tmp = self._cache_path(key).with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self._cache_path(key))
        self._record_call(usage_total, cost)
        return docs

    # ------------------------------------------------------------------
    # Batched async generate
    # ------------------------------------------------------------------

    def generate_batch(  # type: ignore[override]
        self,
        triples: Sequence[tuple[str, Sequence[str], Sequence[str]]],
    ) -> list[list[str]]:
        """Batched async generation.

        Each element of ``triples`` is ``(question, context_passages,
        seed_clause_ids)``. Returns docs parallel to triples.
        """
        results: list[list[str] | None] = [None] * len(triples)
        miss_indices: list[int] = []
        for i, (q, _ctx, seeds) in enumerate(triples):
            key = self._cache_key_grounded(q, seeds)
            cached = self._cache_get(key)
            if cached is not None:
                results[i] = list(cached["generated_docs"])
                self._record_cache_hit()
            else:
                miss_indices.append(i)

        if not miss_indices:
            out: list[list[str]] = []
            for r in results:
                assert r is not None
                out.append(r)
            return out

        async def _run() -> dict[int, list[str]]:
            _strip_openai_base_url_if_blank()
            async with AsyncOpenAI(api_key=self._ensure_api_key()) as aclient:
                sem = asyncio.Semaphore(self.concurrency)

                async def _one(idx: int) -> tuple[int, list[str]]:
                    q, ctx, seeds = triples[idx]
                    context_block = _format_context(ctx)
                    user_filled = self._user_tmpl.replace(
                        "{context}", context_block
                    ).replace("{question}", q)
                    docs: list[str] = []
                    usage_total = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cached_tokens": 0,
                    }
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
                    cost = _estimate_cost_usd(self.model, usage_total)
                    key = self._cache_key_grounded(q, seeds)
                    payload = _CachePayload(
                        question=q,
                        model_id=self.model,
                        model_returned=model_returned,
                        prompt_sha=self._prompt_sha,
                        n=self.n,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        generated_at=datetime.now(timezone.utc).isoformat(),
                        generated_docs=docs,
                        usage=usage_total,
                        cost_usd=cost,
                    )
                    payload_dict = payload.to_dict()
                    payload_dict["seed_clause_ids"] = list(seeds)
                    payload_dict["seed_clause_ids_hash"] = _seed_clause_ids_hash(seeds)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    import json
                    import os
                    tmp = self._cache_path(key).with_suffix(".json.tmp")
                    tmp.write_text(
                        json.dumps(payload_dict, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    os.replace(tmp, self._cache_path(key))
                    self._record_call(usage_total, cost)
                    return idx, docs

                coros = [_one(i) for i in miss_indices]
                out_pairs = await asyncio.gather(*coros)
                return dict(out_pairs)

        miss_results = asyncio.run(_run())
        for i, docs in miss_results.items():
            results[i] = docs

        out: list[list[str]] = []
        for r in results:
            assert r is not None
            out.append(r)
        return out


__all__ = [
    "OpenAIGroundedHydeGenerator",
    "_format_context",
    "_seed_clause_ids_hash",
]
