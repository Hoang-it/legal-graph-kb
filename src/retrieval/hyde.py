"""HyDE (Hypothetical Document Embeddings) generator using OpenAI gpt-4o-mini.

Implements Gao et al. 2022 (https://arxiv.org/abs/2212.10496) on top of the
BGE-M3 dense retrieval channel of :class:`src.retrieval.V5RetrievalPipeline`.

The generator calls OpenAI's `gpt-4o-mini` (constructor-overridable) and
caches every response to disk so re-runs of the same (question, prompt,
n, model, max_tokens, temperature) combo cost $0.

Per-question flow:

1. ``generate(question)`` → produces N hypothetical legal-document
   passages via a system + user chat prompt loaded from
   ``prompts/runtime/hyde_generate.md``. Cache-aware. Sync — used by the
   ``embed_query_callable`` closure called inline by
   :meth:`src.retrieval.HybridRetriever._dense_search`.

2. ``generate_batch(questions)`` → batched, concurrent generation for the
   exp 08 runner's pre-warm phase. Uses an internal asyncio loop with
   ``Semaphore(5)`` (matches the project's ``OPENAI_CONCURRENCY=5``
   default in ``offline/llm_extract.py``) + tenacity retry. Cache writes
   are atomic per question so a Ctrl+C / network blip never corrupts
   partial state.

3. ``embed_query_callable(embed_model)`` returns a closure
   ``question → np.ndarray`` that the :class:`HybridRetriever` plugs
   into ``_dense_search`` as the query encoder. With N>1 the
   embeddings of the N docs are mean-pooled then re-normalised so the
   resulting vector remains unit-length for cosine search.

Cost tracking
-------------
Every API call records ``prompt_tokens``, ``completion_tokens``,
``cached_tokens`` (OpenAI prompt-cache hits, not our disk cache) and a
``cost_usd`` field into the cache payload. The cost formula is reused
verbatim from :mod:`offline.llm_extract` so all gpt-4o-mini cost
arithmetic in this repo has a single source of truth.

The generator instance also exposes ``total_cost_usd`` /
``total_api_calls`` / ``total_cache_hits`` counters so the runner can
print a per-run cost summary at the end of a batch.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from openai import APIError, AsyncOpenAI, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.prompts import load_prompt, resolve_prompt_path


# ---------------------------------------------------------------------------
# Prompt parsing — single-file `system + ===== USER =====` split
# ---------------------------------------------------------------------------

_USER_SENTINEL = "===== USER ====="


def _split_prompt(raw: str) -> tuple[str, str]:
    """Split the loaded prompt into (system, user_template).

    The user side may contain ``{question}`` which is interpolated at call
    time. Splitting on the sentinel keeps both halves in one source file
    so prompt sha covers them jointly.
    """
    if _USER_SENTINEL not in raw:
        raise ValueError(
            f"HyDE prompt missing user sentinel {_USER_SENTINEL!r}. "
            "Check prompts/runtime/hyde_generate.md."
        )
    system_part, user_part = raw.split(_USER_SENTINEL, 1)
    system = system_part.strip()
    user_tmpl = user_part.strip()
    if "{question}" not in user_tmpl:
        raise ValueError(
            "HyDE user template missing `{question}` placeholder — "
            "the runner cannot interpolate the question."
        )
    return system, user_tmpl


# ---------------------------------------------------------------------------
# gpt-4o-mini pricing — kept in sync with offline/llm_extract.py:637-644
# ---------------------------------------------------------------------------

# gpt-4o-mini: input $0.15/M, cached-input $0.075/M, output $0.60/M.
# Any model override at construction time must use the same pricing OR
# the caller must extend _MODEL_PRICING below. Keeping one constant
# avoids drifting prices across modules.
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {
        "input_per_1m": 0.15,
        "cached_input_per_1m": 0.075,
        "output_per_1m": 0.60,
    },
}


def _estimate_cost_usd(model_id: str, usage: dict[str, int]) -> float:
    """Compute USD cost from usage dict. Mirrors offline/llm_extract.py."""
    pricing = _MODEL_PRICING.get(model_id)
    if pricing is None:
        # Unknown model — return 0 so cost summary doesn't lie. The cache
        # still stores the raw usage so a future price-aware audit can
        # recompute retroactively.
        return 0.0
    prompt_t = int(usage.get("prompt_tokens") or 0)
    cached_t = int(usage.get("cached_tokens") or 0)
    completion_t = int(usage.get("completion_tokens") or 0)
    return (
        max(0, prompt_t - cached_t) * pricing["input_per_1m"] / 1e6
        + cached_t * pricing["cached_input_per_1m"] / 1e6
        + completion_t * pricing["output_per_1m"] / 1e6
    )


# ---------------------------------------------------------------------------
# Cache payload schema
# ---------------------------------------------------------------------------


@dataclass
class _CachePayload:
    question: str
    model_id: str
    model_returned: str  # snapshot id from resp.model — proxy for "revision"
    prompt_sha: str
    n: int
    max_tokens: int
    temperature: float
    generated_at: str
    generated_docs: list[str]
    usage: dict[str, int]
    cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "model_id": self.model_id,
            "model_returned": self.model_returned,
            "prompt_sha": self.prompt_sha,
            "n": self.n,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "generated_at": self.generated_at,
            "generated_docs": list(self.generated_docs),
            "usage": dict(self.usage),
            "cost_usd": round(self.cost_usd, 8),
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _model_safe(model_id: str) -> str:
    """Filesystem-safe directory fragment for a model id (``gpt-4o-mini``)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


def _strip_openai_base_url_if_blank() -> None:
    """Mirror the defensive pop used elsewhere in the project.

    The OpenAI SDK treats ``OPENAI_BASE_URL=""`` as the literal empty URL
    and raises APIConnectionError. Every runtime entry-point in this repo
    pops a blank value early; we do the same so the generator can be
    imported safely from any context.
    """
    if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
        os.environ.pop("OPENAI_BASE_URL", None)


class OpenAIHydeGenerator:
    """OpenAI-backed HyDE doc generator with on-disk persistent cache.

    Construction is cheap: prompt is read + sha'd eagerly so a malformed
    prompt fails at __init__, but the OpenAI client is lazy — first call
    to ``generate`` / ``generate_batch`` opens the connection. This lets
    unit tests / import smoke run without an API key.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        n: int = 1,
        cache_dir: str | Path = "artifacts/hyde",
        prompt_path: str = "runtime/hyde_generate.md",
        max_tokens: int = 700,
        temperature: float = 0.0,
        concurrency: int = 5,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        if not (0.0 <= temperature <= 2.0):
            raise ValueError(f"temperature must be in [0, 2], got {temperature}")

        self.model = model
        self.n = n
        self.cache_dir = Path(cache_dir) / f"openai__{_model_safe(model)}"
        self.prompt_path = prompt_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency
        self.api_key_env = api_key_env

        # Load prompt eagerly — fails fast on bad prompt.
        raw_prompt = load_prompt(prompt_path)
        self._prompt_sha = hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest()
        self._system, self._user_tmpl = _split_prompt(raw_prompt)

        # Cost accounting counters (instance-level; runner reads at the end).
        self.total_cost_usd: float = 0.0
        self.total_api_calls: int = 0
        self.total_cache_hits: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cached_tokens: int = 0

        # Counter mutex — generate_batch fires concurrent coroutines on a
        # background thread; the sync generate() and batched updates must
        # not race on the totals.
        self._counter_lock = threading.Lock()

        # Lazy clients.
        self._sync_client: OpenAI | None = None

    # ------------------------------------------------------------------
    # Clients
    # ------------------------------------------------------------------

    def _ensure_api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise EnvironmentError(
                f"Missing API key — set ${self.api_key_env} in env (.env / Colab Secret)."
            )
        return key

    @property
    def sync_client(self) -> OpenAI:
        if self._sync_client is None:
            _strip_openai_base_url_if_blank()
            self._sync_client = OpenAI(api_key=self._ensure_api_key())
        return self._sync_client

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_key(self, question: str) -> str:
        """sha256 over every input that influences the response."""
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
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        docs = data.get("generated_docs")
        if not isinstance(docs, list) or len(docs) != self.n:
            return None
        if any(not isinstance(d, str) for d in docs):
            return None
        return data

    def _cache_put(self, key: str, payload: _CachePayload) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Atomic write — tmp → rename so a crash mid-write never leaves a
        # half-JSON file in cache.
        tmp = self._cache_path(key).with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self._cache_path(key))

    # ------------------------------------------------------------------
    # Counter updates
    # ------------------------------------------------------------------

    def _record_call(self, usage: dict[str, int], cost_usd: float) -> None:
        with self._counter_lock:
            self.total_api_calls += 1
            self.total_cost_usd += cost_usd
            self.total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.total_completion_tokens += int(usage.get("completion_tokens") or 0)
            self.total_cached_tokens += int(usage.get("cached_tokens") or 0)

    def _record_cache_hit(self) -> None:
        with self._counter_lock:
            self.total_cache_hits += 1

    # ------------------------------------------------------------------
    # Public: sync single-question generate (cache-aware)
    # ------------------------------------------------------------------

    def generate(self, question: str) -> list[str]:
        """Return N hypothetical legal-document passages for ``question``.

        Cache hit → returns cached docs without touching OpenAI.
        Cache miss → issues N synchronous API calls, persists payload,
        returns docs. ``cost_usd`` is added to ``self.total_cost_usd``.
        """
        key = self._cache_key(question)
        cached = self._cache_get(key)
        if cached is not None:
            self._record_cache_hit()
            return list(cached["generated_docs"])

        client = self.sync_client
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        docs: list[str] = []
        model_returned: str = self.model
        for _ in range(self.n):
            doc, usage, m_ret = _call_sync_with_retry(
                client=client,
                model=self.model,
                system=self._system,
                user=self._user_tmpl.replace("{question}", question),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            docs.append(doc)
            for k in usage_total:
                usage_total[k] += int(usage.get(k) or 0)
            model_returned = m_ret  # last wins — OpenAI returns the same id within a call

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
        self._cache_put(key, payload)
        self._record_call(usage_total, cost)
        return docs

    # ------------------------------------------------------------------
    # Public: batched async generate (cache-aware, concurrent)
    # ------------------------------------------------------------------

    def generate_batch(self, questions: list[str]) -> list[list[str]]:
        """Batched, concurrent generation. Cache hits return instantly.

        Internally runs an asyncio loop with ``Semaphore(self.concurrency)``
        + tenacity retry per call. Returned list is parallel to ``questions``.
        """
        results: list[list[str] | None] = [None] * len(questions)
        miss_indices: list[int] = []
        for i, q in enumerate(questions):
            key = self._cache_key(q)
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

        # Run async batch for cache misses.
        async def _run() -> dict[int, list[str]]:
            _strip_openai_base_url_if_blank()
            async with AsyncOpenAI(api_key=self._ensure_api_key()) as aclient:
                sem = asyncio.Semaphore(self.concurrency)

                async def _one(idx: int) -> tuple[int, list[str]]:
                    q = questions[idx]
                    docs: list[str] = []
                    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
                    model_returned = self.model
                    async with sem:
                        for _ in range(self.n):
                            doc, usage, m_ret = await _call_async_with_retry(
                                aclient=aclient,
                                model=self.model,
                                system=self._system,
                                user=self._user_tmpl.replace("{question}", q),
                                max_tokens=self.max_tokens,
                                temperature=self.temperature,
                            )
                            docs.append(doc)
                            for k in usage_total:
                                usage_total[k] += int(usage.get(k) or 0)
                            model_returned = m_ret
                    cost = _estimate_cost_usd(self.model, usage_total)
                    key = self._cache_key(q)
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
                    self._cache_put(key, payload)
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

    # ------------------------------------------------------------------
    # Embedding callable for HybridRetriever
    # ------------------------------------------------------------------

    def embed_query_callable(self, embed_model) -> Callable[[str], np.ndarray]:
        """Return ``question → np.ndarray`` for ``HybridRetriever.query_encoder``.

        For N=1 the returned vector is the L2-normalised embedding of the
        single HyDE doc. For N>1 the N normalised embeddings are
        mean-pooled then re-normalised so cosine search remains valid.
        """

        def encode(question: str) -> np.ndarray:
            docs = self.generate(question)
            if not docs:
                raise RuntimeError(
                    f"HyDE generator returned 0 docs for question {question!r}"
                )
            embs = embed_model.encode(
                docs,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            arr = np.asarray(embs, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            mean = arr.mean(axis=0)
            norm = np.linalg.norm(mean)
            if norm > 0:
                mean = mean / norm
            return mean

        return encode

    # ------------------------------------------------------------------
    # Diagnostics / accessors
    # ------------------------------------------------------------------

    @property
    def prompt_sha(self) -> str:
        return self._prompt_sha

    @property
    def prompt_source_path(self) -> Path:
        return resolve_prompt_path(self.prompt_path)

    def cost_summary(self) -> dict[str, Any]:
        """Snapshot of counters — runner prints this at the end of a batch."""
        with self._counter_lock:
            return {
                "model_id": self.model,
                "api_calls": self.total_api_calls,
                "cache_hits": self.total_cache_hits,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "cached_tokens": self.total_cached_tokens,
            }


# ---------------------------------------------------------------------------
# OpenAI call wrappers — tenacity-retried, sync + async variants
# ---------------------------------------------------------------------------

# Retry policy matches offline/llm_extract.py:289-294 verbatim so a
# transient RateLimitError / APIError gets the same handling everywhere.
_RETRY_KW: dict[str, Any] = dict(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIError)),
    reraise=True,
)


@retry(**_RETRY_KW)
def _call_sync_with_retry(
    *,
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, int], str]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return _extract_doc_usage(resp)


@retry(**_RETRY_KW)
async def _call_async_with_retry(
    *,
    aclient: AsyncOpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, int], str]:
    resp = await aclient.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return _extract_doc_usage(resp)


def _extract_doc_usage(resp) -> tuple[str, dict[str, int], str]:
    """Pull (doc_text, usage_dict, model_returned) from a chat completion."""
    if not resp.choices:
        raise RuntimeError("OpenAI returned 0 choices")
    content = resp.choices[0].message.content or ""
    doc = content.strip()
    if not doc:
        raise RuntimeError("OpenAI returned empty content")
    cached = 0
    if resp.usage is not None and getattr(resp.usage, "prompt_tokens_details", None) is not None:
        cached = getattr(resp.usage.prompt_tokens_details, "cached_tokens", 0) or 0
    usage = {
        "prompt_tokens": int(resp.usage.prompt_tokens) if resp.usage else 0,
        "completion_tokens": int(resp.usage.completion_tokens) if resp.usage else 0,
        "cached_tokens": int(cached),
    }
    model_returned = str(resp.model or "")
    return doc, usage, model_returned


__all__ = ["OpenAIHydeGenerator"]
