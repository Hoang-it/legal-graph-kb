"""HyDE (Hypothetical Document Embeddings) generator using Qwen 2.5 3B Instruct.

Implements Gao et al. 2022 (https://arxiv.org/abs/2212.10496) on top of the
BGE-M3 dense retrieval channel of :class:`src.retrieval.V5RetrievalPipeline`.

The generator runs locally — designed for Colab Free T4 (16 GB VRAM,
fp16 by default; 4-bit via bitsandbytes as OOM fallback). See
``docs/plans/hyde_qwen_colab.md`` for the design contract.

Per-question flow:

1. ``generate(question)`` → produces N hypothetical legal-document
   passages via the Qwen chat template + the prompt at
   ``prompts/runtime/hyde_generate.md``. Cache-aware: cache key is
   sha256 of question + prompt sha + n + max_new_tokens.
2. ``embed_query_callable(embed_model)`` returns a closure
   ``question → np.ndarray`` that the :class:`HybridRetriever` plugs
   into ``_dense_search`` as the query encoder. With N>1 the
   embeddings of the N docs are mean-pooled then re-normalised so the
   resulting vector remains unit-length for cosine search.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

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
# Cache payload schema
# ---------------------------------------------------------------------------


@dataclass
class _CachePayload:
    question: str
    model_id: str
    model_revision: str
    prompt_sha: str
    n: int
    max_new_tokens: int
    generated_at: str
    generated_docs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "prompt_sha": self.prompt_sha,
            "n": self.n,
            "max_new_tokens": self.max_new_tokens,
            "generated_at": self.generated_at,
            "generated_docs": list(self.generated_docs),
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _model_id_safe(model_id: str) -> str:
    """Filesystem-safe directory name for a HF model id (``Qwen/Qwen2.5-3B-Instruct``)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


class QwenHydeGenerator:
    """Local Qwen-based HyDE doc generator.

    Constructor is cheap: model + tokenizer load lazily on first
    ``generate``/``generate_batch`` call so an instance can be constructed
    in a non-GPU environment for syntactic checks. Cache writes happen
    after every (single or batched) generation so a re-run picks up
    every doc that was produced before a crash.
    """

    DTYPES = {"fp16", "bf16", "4bit"}

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        n: int = 1,
        cache_dir: str | Path = "artifacts/hyde",
        prompt_path: str = "runtime/hyde_generate.md",
        dtype: str = "fp16",
        batch_size: int = 4,
        max_new_tokens: int = 400,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        if dtype not in self.DTYPES:
            raise ValueError(f"dtype must be one of {sorted(self.DTYPES)}, got {dtype!r}")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.model_id = model_id
        self.n = n
        self.cache_dir = Path(cache_dir) / _model_id_safe(model_id)
        self.prompt_path = prompt_path
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.seed = seed

        # Load prompt eagerly so a missing prompt fails at construction time
        # (cheap, no GPU). _system / _user_tmpl are immutable after init.
        raw_prompt = load_prompt(prompt_path)
        self._prompt_sha = hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest()
        self._system, self._user_tmpl = _split_prompt(raw_prompt)

        # Lazy components
        self._model = None
        self._tokenizer = None
        self._model_revision: str | None = None

    # ------------------------------------------------------------------
    # Model load
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Capture the resolved HF revision (commit sha) for audit. Best-effort:
        # if the hub is unreachable we record "unknown" rather than failing —
        # the generator can still run from a local cache.
        try:
            from huggingface_hub import HfApi  # type: ignore

            self._model_revision = HfApi().model_info(self.model_id).sha or "unknown"
        except Exception:  # noqa: BLE001
            self._model_revision = "unknown"

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # Qwen 2.5 ships with pad_token=None; left-padding required for
        # batched causal generation so all sequences end aligned.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {}
        if self.dtype == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        elif self.dtype == "bf16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif self.dtype == "4bit":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self.device,
            **model_kwargs,
        )
        model.eval()

        # Deterministic seeding for reproducibility — Qwen uses sampling by
        # default. We use deterministic greedy below so seed is mainly
        # belt-and-braces for any future sampling experiments.
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self._tokenizer = tokenizer
        self._model = model

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    def _apply_chat_template(self, question: str) -> str:
        """Render the ChatML prompt string for one question."""
        assert self._tokenizer is not None, "_load_model must be called first"
        user_msg = self._user_tmpl.replace("{question}", question)
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": user_msg},
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_key(self, question: str) -> str:
        """sha256(question + prompt_sha + n + max_new_tokens) — model_id is
        already part of the cache_dir layout so it isn't repeated in the key."""
        h = hashlib.sha256()
        h.update(question.encode("utf-8"))
        h.update(b"|")
        h.update(self._prompt_sha.encode("utf-8"))
        h.update(b"|n=")
        h.update(str(self.n).encode("utf-8"))
        h.update(b"|mnt=")
        h.update(str(self.max_new_tokens).encode("utf-8"))
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> list[str] | None:
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
        return docs

    def _cache_put(self, key: str, question: str, docs: list[str]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = _CachePayload(
            question=question,
            model_id=self.model_id,
            model_revision=self._model_revision or "unknown",
            prompt_sha=self._prompt_sha,
            n=self.n,
            max_new_tokens=self.max_new_tokens,
            generated_at=datetime.now(timezone.utc).isoformat(),
            generated_docs=docs,
        )
        # Atomic write — write to .tmp then rename so a crash mid-write
        # never leaves a partial JSON in cache.
        tmp = self._cache_path(key).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self._cache_path(key))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_raw(self, prompts: list[str]) -> list[str]:
        """Run the model on a list of fully-rendered chat prompts."""
        assert self._model is not None and self._tokenizer is not None

        import torch

        enc = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,             # deterministic greedy
                temperature=1.0,             # ignored when do_sample=False
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        # out: [batch, prompt_len + gen_len] — slice the gen portion only.
        gen_only = out[:, enc["input_ids"].shape[1]:]
        decoded = self._tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        return [d.strip() for d in decoded]

    def generate(self, question: str) -> list[str]:
        """Return N hypothetical legal-document passages. Cache-aware."""
        key = self._cache_key(question)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        self._load_model()
        prompt = self._apply_chat_template(question)
        # When N>1 we feed the same prompt N times in one batch. Since
        # do_sample=False is deterministic the N outputs would be identical
        # — for the plan's D2 (N=1) this is fine. If a future ablation
        # raises N, swap to sampling here (and re-seed per call).
        prompts = [prompt] * self.n
        docs = self._generate_raw(prompts)
        self._cache_put(key, question, docs)
        return docs

    def generate_batch(self, questions: list[str]) -> list[list[str]]:
        """Batched single-pass generation across multiple questions.

        Cache hits return immediately and never trigger a model load. Only
        the cache-miss subset is run through the model, in chunks of
        ``batch_size``. Returned list is parallel to ``questions``.
        """
        results: list[list[str] | None] = [None] * len(questions)
        miss_indices: list[int] = []
        miss_keys: list[str] = []
        for i, q in enumerate(questions):
            key = self._cache_key(q)
            cached = self._cache_get(key)
            if cached is not None:
                results[i] = cached
            else:
                miss_indices.append(i)
                miss_keys.append(key)

        if miss_indices:
            self._load_model()
            for chunk_start in range(0, len(miss_indices), self.batch_size):
                chunk_idx = miss_indices[chunk_start : chunk_start + self.batch_size]
                chunk_keys = miss_keys[chunk_start : chunk_start + self.batch_size]
                # Each question expands to N copies of its prompt; flatten.
                expanded_prompts: list[str] = []
                expanded_owner: list[int] = []  # index back into chunk_idx
                for local_i, q_idx in enumerate(chunk_idx):
                    prompt = self._apply_chat_template(questions[q_idx])
                    for _ in range(self.n):
                        expanded_prompts.append(prompt)
                        expanded_owner.append(local_i)

                decoded = self._generate_raw(expanded_prompts)

                # Re-bucket back to per-question N-tuples and persist cache.
                per_q_docs: dict[int, list[str]] = {i: [] for i in range(len(chunk_idx))}
                for owner, doc in zip(expanded_owner, decoded):
                    per_q_docs[owner].append(doc)
                for local_i, q_idx in enumerate(chunk_idx):
                    docs = per_q_docs[local_i]
                    self._cache_put(chunk_keys[local_i], questions[q_idx], docs)
                    results[q_idx] = docs

        # All slots must be filled now.
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
    # Diagnostics
    # ------------------------------------------------------------------

    def cuda_memory_mb(self) -> dict[str, float]:
        """Return current/peak CUDA memory in MB for the active device.

        Used by the Phase 3 dry-run to report VRAM headroom before / after
        model load. Returns ``{}`` when CUDA is not available.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return {}
            dev = torch.cuda.current_device()
            return {
                "allocated_mb": round(torch.cuda.memory_allocated(dev) / 1024**2, 1),
                "reserved_mb": round(torch.cuda.memory_reserved(dev) / 1024**2, 1),
                "max_allocated_mb": round(
                    torch.cuda.max_memory_allocated(dev) / 1024**2, 1
                ),
            }
        except Exception:  # noqa: BLE001
            return {}

    @property
    def prompt_sha(self) -> str:
        """sha256 of the loaded prompt file — used for cache keys + audit."""
        return self._prompt_sha

    @property
    def prompt_source_path(self) -> Path:
        """Resolved on-disk path of the prompt file (honours override env)."""
        return resolve_prompt_path(self.prompt_path)


__all__ = ["QwenHydeGenerator"]
