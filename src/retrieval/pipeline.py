"""V5 Sprint 1 vanilla pipeline — wires M4 + M5 + generator.

Per-query flow:

1. Hybrid retrieve top-50 (BGE-M3 dense ∥ Lucene BM25 → temporal filter → RRF).
2. Cross-encoder rerank top-15 (= "seed set" for graph expansion).
3. Graph-expand seeds along REFERS_TO (1..max_hops) → neighbor Articles.
4. Re-rerank (seeds + neighbors) → top-12 (= final LLM context).
5. GPT-4o-mini generator with strict v5 prompt → answer + canonical citations.
6. Parse + verify citations against KG.

Returns a :class:`V5Answer` dataclass mirroring ``RagAnswer`` fields so the
existing inference runner pattern in ``eval_core/inference.py`` plugs in
without bespoke handling, plus a ``retrieval_audit`` block holding per-stage
counts for Sprint 1 diagnostic analysis.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from dotenv import load_dotenv

from src.bge_m3_loader import adapter_path_from_env, load_bge_m3
from src.citations import (
    DEFAULT_REGISTRY_PATH,
    format_citation,
    load_registry,
    parse_displayed_citations,
)
from src.legal_metadata import load_law_metadata
from src.prompts import load_prompt
from src.retrieval.graph_expansion import GraphExpander, Neighbor
from src.retrieval.hybrid_retriever import (
    Candidate,
    HybridRetriever,
    RetrievalAudit,
    _dedupe_articles_in_order,
)
from src.retrieval.hyde import OpenAIHydeGenerator
from src.retrieval.hyde2 import OpenAIGroundedHydeGenerator
from src.retrieval.hyde_semantic import OpenAISemanticHydeGenerator
from src.retrieval.reranker import CrossEncoderReranker

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")
DB = os.getenv("NEO4J_DATABASE", "neo4j")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda")
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT_REL = "runtime/graphrag_v5_system.md"
MAX_CONTEXT_CHARS = 7000


# ---------------------------------------------------------------------------
# Answer dataclass
# ---------------------------------------------------------------------------


@dataclass
class V5Answer:
    question: str
    answer: str
    citations: list[str] = field(default_factory=list)        # canonical display strings
    citation_ids: list[str] = field(default_factory=list)     # item_ids (matches gold_articles)
    hits: list[dict[str, Any]] = field(default_factory=list)  # final top-K used in context
    # Audit (for Sprint 1 diagnostic — surfaced into record JSON)
    retrieval_audit: dict[str, Any] = field(default_factory=dict)
    n_seeds: int = 0                # post first rerank
    n_neighbors_added: int = 0      # net new from graph expansion
    n_final: int = 0                # post second rerank
    n_semantic_edges: int = 0       # always 0 for v5 (kept for record-shape parity)
    n_refs: int = 0                 # always 0 for v5
    elapsed_s: float = 0.0
    elapsed_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalOnlyAnswer:
    """Output of ``V5RetrievalPipeline.retrieve_only`` — no LLM call.

    Captures the article ids that survived each retrieval stage so the Week 1
    audit can pinpoint where gold drops off. Field order mirrors the stage
    order in :meth:`V5RetrievalPipeline.retrieve_only`.
    """
    question: str
    # Stage pools — article-level, ordered, deduped (first occurrence wins)
    dense_article_ids: list[str] = field(default_factory=list)
    sparse_article_ids: list[str] = field(default_factory=list)
    post_temporal_article_ids: list[str] = field(default_factory=list)
    fused_article_ids: list[str] = field(default_factory=list)
    rerank1_article_ids: list[str] = field(default_factory=list)
    expanded_article_ids: list[str] = field(default_factory=list)  # seeds + neighbours (pre rerank2)
    final_article_ids: list[str] = field(default_factory=list)     # top-K after rerank2
    # Metadata
    retrieval_audit: dict[str, Any] = field(default_factory=dict)
    n_seeds: int = 0
    n_neighbors_added: int = 0
    n_final: int = 0
    elapsed_s: float = 0.0
    elapsed_breakdown: dict[str, float] = field(default_factory=dict)
    # Knob snapshot — so audit records carry the config they were produced under
    config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class V5RetrievalPipeline:
    """Plan v5 Sprint 1 vanilla pipeline.

    Construction is lazy: Neo4j driver and OpenAI client connect at __init__,
    but BGE-M3 + cross-encoder are loaded on first use to keep import cheap.
    """

    def __init__(
        self,
        model: str | None = None,
        dense_k: int = 30,
        sparse_k: int = 30,
        top_after_fusion: int = 50,
        rerank1_top_k: int = 15,
        rerank2_top_k: int = 12,
        max_hops: int = 3,
        rrf_k: int = 60,
        per_seed_neighbors: int = 10,
        adapter_path: str | None = None,
        dense_index: str | None = None,
        reranker_model: str | None = None,
        temporal_mode: str = "strict_today_default",
        hyde: OpenAIHydeGenerator | None = None,
        hyde2: OpenAIGroundedHydeGenerator | None = None,
        hyde2_seed_k: int = 5,
        hyde_semantic: OpenAISemanticHydeGenerator | None = None,
    ):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(URI, auth=(USER, PWD))
        self.driver.verify_connectivity()
        self.db = DB
        self.openai_model = model or OPENAI_MODEL_DEFAULT

        # Hyperparameters from plan §4
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.top_after_fusion = top_after_fusion
        self.rerank1_top_k = rerank1_top_k
        self.rerank2_top_k = rerank2_top_k
        self.max_hops = max_hops
        self.rrf_k = rrf_k
        self.per_seed_neighbors = per_seed_neighbors

        # Sprint 2 swap points (env-driven by default, constructor-override available)
        env_adapter = adapter_path_from_env()
        self.adapter_path = (
            adapter_path if adapter_path is not None else (str(env_adapter) if env_adapter else None)
        )
        env_dense = (os.environ.get("V5_DENSE_INDEX") or "").strip() or None
        self.dense_index = dense_index or env_dense
        self.reranker_model = reranker_model  # None → reranker defaults from env / module const
        self.temporal_mode = temporal_mode
        self.hyde = hyde
        self.hyde2 = hyde2
        self.hyde2_seed_k = hyde2_seed_k
        self.hyde_semantic = hyde_semantic

        # Static deps
        self._registry = load_registry(DEFAULT_REGISTRY_PATH)
        self._law_meta = load_law_metadata()
        self._system_prompt = load_prompt(SYSTEM_PROMPT_REL)

        # Lazy components
        self._embed_model = None
        self._retriever = None
        self._reranker = None
        self._expander = None
        self._openai = None

    # ------------------------------------------------------------------
    # Lazy components
    # ------------------------------------------------------------------

    @property
    def embed_model(self):
        if self._embed_model is None:
            self._embed_model = load_bge_m3(adapter_path=self.adapter_path, device=EMBED_DEVICE)
        return self._embed_model

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            # HyDE wiring: when self.hyde is set, build a query encoder
            # closure that swaps the raw question for the embedding of N
            # HyDE-generated docs. Sparse channel keeps the raw question
            # (plan §D3 — isolates HyDE contribution to dense only).
            query_encoder = (
                self.hyde.embed_query_callable(self.embed_model)
                if self.hyde is not None
                else None
            )
            self._retriever = HybridRetriever(
                driver=self.driver,
                db=self.db,
                embed_model=self.embed_model,
                dense_k=self.dense_k,
                sparse_k=self.sparse_k,
                top_after_fusion=self.top_after_fusion,
                rrf_k=self.rrf_k,
                dense_index=self.dense_index,
                temporal_mode=self.temporal_mode,
                query_encoder=query_encoder,
            )
        return self._retriever

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                model_name=self.reranker_model or os.getenv(
                    "V5_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
                ),
            )
        return self._reranker

    @property
    def expander(self) -> GraphExpander:
        if self._expander is None:
            self._expander = GraphExpander(
                driver=self.driver,
                db=self.db,
                max_hops=self.max_hops,
                per_seed_limit=self.per_seed_neighbors,
            )
        return self._expander

    @property
    def openai(self):
        if self._openai is None:
            from openai import OpenAI

            self._openai = OpenAI()
        return self._openai

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def retrieve_dense_only(self, question: str, top_k: int | None = None) -> RetrievalOnlyAnswer:
        """Pure BGE-M3 dense retrieval — no sparse, no temporal, no RRF, no
        rerank, no graph expand.

        Used as the "dense" arm in experiment 06. ``top_k`` defaults to
        ``self.dense_k``. Returns a :class:`RetrievalOnlyAnswer` where
        ``dense_article_ids`` and ``final_article_ids`` both carry the same
        article-deduped, rank-preserving list so downstream metric code can
        treat the arm as a final-stage output without special-casing.
        """
        t0 = time.time()
        timings: dict[str, float] = {}

        k = top_k if top_k is not None else self.dense_k
        t = time.time()
        rows = self.retriever._dense_search(question, k)
        timings["dense"] = round(time.time() - t, 3)

        article_ids = _dedupe_articles_in_order(r["article_id"] for r in rows)

        config_snapshot = {
            "mode": "dense_only",
            "dense_k": k,
            "adapter_path": self.adapter_path,
            "dense_index": self.dense_index,
        }

        return RetrievalOnlyAnswer(
            question=question,
            dense_article_ids=article_ids,
            final_article_ids=article_ids,
            n_final=len(article_ids),
            elapsed_s=round(time.time() - t0, 3),
            elapsed_breakdown=timings,
            config=config_snapshot,
        )

    def retrieve_dense_only_hyde(
        self, question: str, top_k: int | None = None
    ) -> RetrievalOnlyAnswer:
        """Pure dense retrieval using HyDE embedding. Requires ``hyde`` to be set.

        Mirrors :meth:`retrieve_dense_only` but the dense query is the
        mean-pooled embedding of N HyDE-generated hypothetical documents
        rather than the raw question. No sparse, no temporal, no RRF, no
        rerank, no graph expand — used as the ``dense_hyde`` arm in
        experiment 08.

        Embedding is computed via the same closure the HybridRetriever
        uses when wired through ``self.retriever``, so the cached HyDE
        doc + the cached embedding are shared across arms within the
        same pipeline instance.
        """
        if self.hyde is None:
            raise RuntimeError(
                "retrieve_dense_only_hyde requires a OpenAIHydeGenerator passed "
                "via V5RetrievalPipeline(hyde=...)."
            )

        t0 = time.time()
        timings: dict[str, float] = {}

        k = top_k if top_k is not None else self.dense_k

        # Reuse the retriever's dense path so a single code path handles
        # both vanilla and HyDE encoding. ``self.retriever`` already has
        # query_encoder=self.hyde.embed_query_callable(self.embed_model)
        # because self.hyde is non-None at construction.
        t = time.time()
        rows = self.retriever._dense_search(question, k)
        timings["dense"] = round(time.time() - t, 3)

        article_ids = _dedupe_articles_in_order(r["article_id"] for r in rows)

        config_snapshot = {
            "mode": "dense_only_hyde",
            "dense_k": k,
            "adapter_path": self.adapter_path,
            "dense_index": self.dense_index,
            "hyde": {
                "model_id": self.hyde.model,
                "n": self.hyde.n,
                "max_tokens": self.hyde.max_tokens,
                "temperature": self.hyde.temperature,
                "prompt_sha": self.hyde.prompt_sha,
            },
        }

        return RetrievalOnlyAnswer(
            question=question,
            dense_article_ids=article_ids,
            final_article_ids=article_ids,
            n_final=len(article_ids),
            elapsed_s=round(time.time() - t0, 3),
            elapsed_breakdown=timings,
            config=config_snapshot,
        )

    def retrieve_dense_only_hyde2(
        self, question: str, top_k: int | None = None
    ) -> RetrievalOnlyAnswer:
        """Retrieval-grounded iterative HyDE (HyDE2) — pure dense, two-pass.

        Pass 1: BGE-M3+LoRA dense top-``self.hyde2_seed_k`` on the raw
        question → seed clause set.
        Pass 2 (LLM): :class:`OpenAIGroundedHydeGenerator` produces N
        hypothetical-doc passages conditioned on the seed clause texts.
        Pass 3 (dense): mean-pool the BGE-M3 embeddings of the N HyDE2
        docs, normalize, then dense top-``top_k`` against the same index.

        Returns a :class:`RetrievalOnlyAnswer` whose ``final_article_ids``
        is the article-deduped, rank-preserving result of pass 3 — same
        contract as :meth:`retrieve_dense_only_hyde` so the metrics
        engine treats this arm identically. ``config`` snapshot includes
        ``seed_clause_ids`` + ``seed_clause_ids_hash`` so the cache key
        for this record is reconstructable from the audit alone.

        Used by the ``dense_hyde2`` arm in experiment 09.
        """
        if self.hyde2 is None:
            raise RuntimeError(
                "retrieve_dense_only_hyde2 requires an OpenAIGroundedHydeGenerator "
                "passed via V5RetrievalPipeline(hyde2=...)."
            )

        import hashlib
        import numpy as np

        t0 = time.time()
        timings: dict[str, float] = {}

        seed_k = self.hyde2_seed_k
        k = top_k if top_k is not None else self.dense_k

        # Pass 1 — seed retrieval on RAW question. We call _dense_search
        # directly; for a pipe constructed with hyde2 but no hyde, the
        # retriever's query_encoder is None and the call uses raw BGE-M3
        # encoding of the question, which is what we want for pass 1.
        t = time.time()
        seed_rows = self.retriever._dense_search(question, seed_k)
        timings["seed_retrieve"] = round(time.time() - t, 3)

        if len(seed_rows) < seed_k:
            # Index too small? Still try with whatever we got — but error if 0.
            if not seed_rows:
                raise RuntimeError(
                    f"HyDE2 pass-1 returned 0 seeds for question {question!r} — "
                    f"dense index '{self.dense_index}' is empty or unreachable."
                )

        seed_clause_ids = [r["clause_id"] for r in seed_rows]
        seed_texts = [r["text"] for r in seed_rows]

        # Pass 2 — grounded LLM generation. Cache-aware inside generate().
        t = time.time()
        docs = self.hyde2.generate(
            question,
            context_passages=seed_texts,
            seed_clause_ids=seed_clause_ids,
        )
        timings["hyde_generate"] = round(time.time() - t, 3)

        # Pass 3 — encode docs, mean-pool, normalize, search.
        t = time.time()
        embs = self.embed_model.encode(
            docs, normalize_embeddings=True, show_progress_bar=False
        )
        arr = np.asarray(embs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        mean_vec = arr.mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0:
            mean_vec = mean_vec / norm
        rows = self.retriever._dense_search_by_vector(mean_vec.tolist(), k)
        timings["final_retrieve"] = round(time.time() - t, 3)

        article_ids = _dedupe_articles_in_order(r["article_id"] for r in rows)

        seed_ids_hash = hashlib.sha256(
            ",".join(sorted(seed_clause_ids)).encode("utf-8")
        ).hexdigest()

        config_snapshot = {
            "mode": "dense_only_hyde2",
            "dense_k": k,
            "seed_k": seed_k,
            "adapter_path": self.adapter_path,
            "dense_index": self.dense_index,
            "hyde2": {
                "model_id": self.hyde2.model,
                "n": self.hyde2.n,
                "max_tokens": self.hyde2.max_tokens,
                "temperature": self.hyde2.temperature,
                "prompt_sha": self.hyde2.prompt_sha,
                "seed_clause_ids": seed_clause_ids,
                "seed_clause_ids_hash": seed_ids_hash,
            },
        }

        return RetrievalOnlyAnswer(
            question=question,
            dense_article_ids=article_ids,
            final_article_ids=article_ids,
            n_final=len(article_ids),
            elapsed_s=round(time.time() - t0, 3),
            elapsed_breakdown=timings,
            config=config_snapshot,
        )

    def retrieve_dense_only_hyde_semantic(
        self,
        question: str,
        frame_text: str,
        context_key_ids: list[str],
        top_k: int | None = None,
    ) -> RetrievalOnlyAnswer:
        """Semantic-grounded HyDE (exp 13) — pure dense, NO seed pass.

        The hypothetical doc is grounded on a precomputed BHXH **concept
        frame** (built upstream by
        ``runtime.retrievers.semantic_context.build_semantic_context`` —
        query→concepts, no dense clause seed → attacks exp 09's domain-noisy
        seed). We encode the doc with BGE-M3(+LoRA), mean-pool + normalize,
        then dense top-``top_k`` against the same index. Same
        :class:`RetrievalOnlyAnswer` contract as the other dense arms.

        ``frame_text`` may be empty (no concept matched) — the generator
        soft-falls-back to a HyDE1-style general passage; ``context_key_ids``
        still keys the cache deterministically.
        """
        if self.hyde_semantic is None:
            raise RuntimeError(
                "retrieve_dense_only_hyde_semantic requires an "
                "OpenAISemanticHydeGenerator passed via "
                "V5RetrievalPipeline(hyde_semantic=...)."
            )
        import numpy as np

        t0 = time.time()
        timings: dict[str, float] = {}
        k = top_k if top_k is not None else self.dense_k

        # Pass A — grounded LLM generation (cache-aware inside generate()).
        t = time.time()
        docs = self.hyde_semantic.generate(question, frame_text, context_key_ids)
        timings["hyde_generate"] = round(time.time() - t, 3)

        # Pass B — encode docs, mean-pool, normalize, dense search by vector.
        t = time.time()
        embs = self.embed_model.encode(
            docs, normalize_embeddings=True, show_progress_bar=False
        )
        arr = np.asarray(embs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        mean_vec = arr.mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0:
            mean_vec = mean_vec / norm
        rows = self.retriever._dense_search_by_vector(mean_vec.tolist(), k)
        timings["final_retrieve"] = round(time.time() - t, 3)

        article_ids = _dedupe_articles_in_order(r["article_id"] for r in rows)

        config_snapshot = {
            "mode": "dense_only_hyde_semantic",
            "dense_k": k,
            "adapter_path": self.adapter_path,
            "dense_index": self.dense_index,
            "hyde_semantic": {
                "model_id": self.hyde_semantic.model,
                "n": self.hyde_semantic.n,
                "max_tokens": self.hyde_semantic.max_tokens,
                "temperature": self.hyde_semantic.temperature,
                "prompt_sha": self.hyde_semantic.prompt_sha,
                "n_context_key_ids": len(context_key_ids),
                "concept_match": bool(frame_text),
            },
        }

        return RetrievalOnlyAnswer(
            question=question,
            dense_article_ids=article_ids,
            final_article_ids=article_ids,
            n_final=len(article_ids),
            elapsed_s=round(time.time() - t0, 3),
            elapsed_breakdown=timings,
            config=config_snapshot,
        )

    def dense_hyde_semantic_rows(
        self,
        question: str,
        frame_text: str,
        context_key_ids: list[str],
        top_k: int | None = None,
        on_step: Optional[Callable[[str], None]] = None,
    ) -> tuple[list[dict], list[str]]:
        """HyDE-semantic dense search returning raw clause rows + hypothesis docs.

        Same retrieval mechanics as :meth:`retrieve_dense_only_hyde_semantic`
        (concept-frame-grounded HyDE → BGE-M3 mean-pool → dense search), but
        returns the per-clause rows (``clause_id`` / ``article_*`` / ``law_id`` /
        ``text`` / ``score``) instead of an article-deduped id list, plus the raw
        hypothesis ``docs``. Lets a logic-LM retriever adapter reuse the exact
        Tier-1 dense_hyde_semantic path for BOTH the context chunks and the
        hypothesis passage. Does not modify
        :meth:`retrieve_dense_only_hyde_semantic`.
        """
        if self.hyde_semantic is None:
            raise RuntimeError(
                "dense_hyde_semantic_rows requires an OpenAISemanticHydeGenerator "
                "passed via V5RetrievalPipeline(hyde_semantic=...)."
            )
        import numpy as np

        k = top_k if top_k is not None else self.dense_k
        if on_step is not None:
            on_step("hypothesis")
        docs = self.hyde_semantic.generate(question, frame_text, context_key_ids)
        if on_step is not None:
            on_step("search")
        embs = self.embed_model.encode(
            docs, normalize_embeddings=True, show_progress_bar=False
        )
        arr = np.asarray(embs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        mean_vec = arr.mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0:
            mean_vec = mean_vec / norm
        rows = self.retriever._dense_search_by_vector(mean_vec.tolist(), k)
        return rows, docs

    def ask_dense_hyde_semantic(
        self,
        question: str,
        frame_text: str,
        context_key_ids: list[str],
        top_k: int | None = None,
    ) -> V5Answer:
        """Direct QA over dense_hyde_semantic retrieval — NO logic-LM, NO rerank/expand.

        Retrieves clauses exactly like :meth:`dense_hyde_semantic_rows`
        (concept-frame-grounded HyDE → BGE-M3 mean-pool → dense top-k), then
        reuses the SAME generator as :meth:`ask` (identical system prompt,
        context format via :meth:`_build_context`, citation parsing) to produce
        a prose answer. This is the "hyde only" QA arm: same retrieval as the
        logic-LM-hyde-semantic arms, but the answer comes straight from the
        generator instead of a Prolog program — isolating the contribution of
        the logic-LM layer. Does not touch :meth:`ask`.
        """
        t0 = time.time()
        timings: dict[str, float] = {}

        t = time.time()
        rows, _docs = self.dense_hyde_semantic_rows(
            question, frame_text=frame_text, context_key_ids=context_key_ids, top_k=top_k
        )
        timings["retrieve"] = round(time.time() - t, 3)

        if not rows:
            return V5Answer(
                question=question,
                answer="Theo các điều luật được cung cấp, tôi không có đủ thông tin để trả lời chính xác câu hỏi này.",
                elapsed_s=round(time.time() - t0, 3),
                elapsed_breakdown=timings,
            )

        # Map dense rows → the (meta, text, score) shape _build_context expects
        # for a "seed" chunk, so the LLM sees the exact same header/citation
        # format as the graphrag_v5 arm.
        final: list[tuple[dict[str, Any], str, float]] = []
        for r in rows:
            meta = {
                "kind": "seed",
                "clause_id": r["clause_id"],
                "article_id": r.get("article_id"),
                "law_id": str(r.get("law_id") or ""),
                "article_n": r.get("article_n"),
                "article_title": r.get("article_title"),
                "clause_n": r.get("clause_n"),
            }
            final.append((meta, str(r.get("text") or ""), float(r.get("score") or 0.0)))

        t = time.time()
        context_str = self._build_context(final)
        resp = self.openai.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": f"CONTEXT:\n{context_str}\n\n---\n\nCÂU HỎI: {question}"},
            ],
            temperature=0,
        )
        answer_text = resp.choices[0].message.content or ""
        timings["llm"] = round(time.time() - t, 3)

        citations, citation_ids = self._parse_citations(answer_text)
        return V5Answer(
            question=question,
            answer=answer_text,
            citations=citations,
            citation_ids=citation_ids,
            hits=[
                {"rank": i + 1, "score": round(float(score), 4), **meta}
                for i, (meta, _txt, score) in enumerate(final)
            ],
            n_final=len(final),
            elapsed_s=round(time.time() - t0, 3),
            elapsed_breakdown=timings,
        )

    def retrieve_only(self, question: str) -> RetrievalOnlyAnswer:
        """Run dense + sparse + temporal + RRF + rerank1 + expand + rerank2.

        Skips the LLM generator step entirely — no OpenAI call, no citation
        parsing. Returns the article id pool at every stage so audit scripts
        can compute ``recall@K_stage`` and pinpoint gold drop-off.
        """
        t0 = time.time()
        timings: dict[str, float] = {}

        t = time.time()
        candidates, audit = self.retriever.retrieve(question)
        timings["retrieve"] = round(time.time() - t, 3)

        config_snapshot = {
            "dense_k": self.dense_k,
            "sparse_k": self.sparse_k,
            "top_after_fusion": self.top_after_fusion,
            "rerank1_top_k": self.rerank1_top_k,
            "rerank2_top_k": self.rerank2_top_k,
            "max_hops": self.max_hops,
            "rrf_k": self.rrf_k,
            "per_seed_neighbors": self.per_seed_neighbors,
            "adapter_path": self.adapter_path,
            "dense_index": self.dense_index,
            "reranker_model": self.reranker_model,
            "temporal_mode": self.temporal_mode,
            "hyde": (
                None
                if self.hyde is None
                else {
                    "model_id": self.hyde.model,
                    "n": self.hyde.n,
                    "max_tokens": self.hyde.max_tokens,
                    "temperature": self.hyde.temperature,
                    "prompt_sha": self.hyde.prompt_sha,
                }
            ),
        }

        if not candidates:
            return RetrievalOnlyAnswer(
                question=question,
                retrieval_audit=audit.__dict__,
                elapsed_s=round(time.time() - t0, 3),
                elapsed_breakdown=timings,
                config=config_snapshot,
            )

        # Rerank pass 1 — seed set
        t = time.time()
        rerank1 = self.reranker.rerank(
            question,
            [c.text for c in candidates],
            top_k=self.rerank1_top_k,
        )
        seed_indices = [i for i, _ in rerank1]
        seeds = [candidates[i] for i in seed_indices]
        timings["rerank1"] = round(time.time() - t, 3)
        rerank1_article_ids = _dedupe_articles_in_order(c.article_id for c in seeds)

        # Graph expansion
        t = time.time()
        neighbors = self.expander.expand([c.clause_id for c in seeds])
        timings["expand"] = round(time.time() - t, 3)

        # Pool = seeds + dedup-ed neighbours
        pool_texts: list[str] = []
        pool_article_ids: list[str] = []
        seen_target_ids: set[str] = set()
        for c in seeds:
            pool_texts.append(c.text)
            pool_article_ids.append(c.article_id)
            seen_target_ids.add(c.article_id)
        n_neighbors_added = 0
        for nb in neighbors:
            if nb.target_id in seen_target_ids or not nb.target_text:
                continue
            pool_texts.append(nb.target_text)
            pool_article_ids.append(nb.target_id)
            seen_target_ids.add(nb.target_id)
            n_neighbors_added += 1

        # Rerank pass 2 — final top-K
        t = time.time()
        rerank2 = self.reranker.rerank(question, pool_texts, top_k=self.rerank2_top_k)
        timings["rerank2"] = round(time.time() - t, 3)
        final_article_ids = _dedupe_articles_in_order(
            pool_article_ids[i] for i, _ in rerank2
        )

        return RetrievalOnlyAnswer(
            question=question,
            dense_article_ids=list(audit.dense_article_ids),
            sparse_article_ids=list(audit.sparse_article_ids),
            post_temporal_article_ids=list(audit.post_temporal_article_ids),
            fused_article_ids=list(audit.fused_article_ids),
            rerank1_article_ids=rerank1_article_ids,
            expanded_article_ids=pool_article_ids,
            final_article_ids=final_article_ids,
            retrieval_audit=audit.__dict__,
            n_seeds=len(seeds),
            n_neighbors_added=n_neighbors_added,
            n_final=len(final_article_ids),
            elapsed_s=round(time.time() - t0, 3),
            elapsed_breakdown=timings,
            config=config_snapshot,
        )

    def ask(self, question: str) -> V5Answer:
        t0 = time.time()
        timings: dict[str, float] = {}

        # 1. Hybrid retrieve (top-50)
        t = time.time()
        candidates, audit = self.retriever.retrieve(question)
        timings["retrieve"] = round(time.time() - t, 3)

        if not candidates:
            return V5Answer(
                question=question,
                answer="Theo các điều luật được cung cấp, tôi không có đủ thông tin để trả lời chính xác câu hỏi này.",
                retrieval_audit=audit.__dict__,
                elapsed_s=round(time.time() - t0, 3),
                elapsed_breakdown=timings,
            )

        # 2. Rerank pass 1 → top-15 seed set
        t = time.time()
        rerank1 = self.reranker.rerank(
            question,
            [c.text for c in candidates],
            top_k=self.rerank1_top_k,
        )
        seed_indices = [i for i, _ in rerank1]
        seeds = [candidates[i] for i in seed_indices]
        seed_scores = {i: s for i, s in rerank1}
        timings["rerank1"] = round(time.time() - t, 3)

        # 3. Graph-expand seeds
        t = time.time()
        neighbors = self.expander.expand([c.clause_id for c in seeds])
        timings["expand"] = round(time.time() - t, 3)

        # 4. Build the final pool (seeds + dedup-ed neighbors)
        pool_texts: list[str] = []
        pool_meta: list[dict[str, Any]] = []
        seen_target_ids: set[str] = set()

        for i, c in enumerate(seeds):
            pool_texts.append(c.text)
            pool_meta.append({
                "kind": "seed",
                "clause_id": c.clause_id,
                "article_id": c.article_id,
                "law_id": c.law_id,
                "article_n": c.article_n,
                "article_title": c.article_title,
                "clause_n": c.clause_n,
                "rerank1_score": seed_scores.get(seed_indices[i], 0.0),
            })
            seen_target_ids.add(c.article_id)

        n_neighbors_added = 0
        for nb in neighbors:
            if nb.target_id in seen_target_ids:
                continue
            if not nb.target_text:
                continue
            pool_texts.append(nb.target_text)
            pool_meta.append({
                "kind": "neighbor",
                "target_id": nb.target_id,
                "target_label": nb.target_label,
                "target_title": nb.target_title,
                "seed_clause_id": nb.seed_clause_id,
                "hop_distance": nb.hop_distance,
            })
            seen_target_ids.add(nb.target_id)
            n_neighbors_added += 1

        # 5. Rerank pass 2 → top-12 final context
        t = time.time()
        rerank2 = self.reranker.rerank(
            question,
            pool_texts,
            top_k=self.rerank2_top_k,
        )
        final = [(pool_meta[i], pool_texts[i], score) for i, score in rerank2]
        timings["rerank2"] = round(time.time() - t, 3)

        # 6. Build context + LLM call
        t = time.time()
        context_str = self._build_context(final)
        resp = self.openai.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": f"CONTEXT:\n{context_str}\n\n---\n\nCÂU HỎI: {question}"},
            ],
            temperature=0,
        )
        answer_text = resp.choices[0].message.content or ""
        timings["llm"] = round(time.time() - t, 3)

        # 7. Parse citations (registry-backed)
        citations, citation_ids = self._parse_citations(answer_text)

        return V5Answer(
            question=question,
            answer=answer_text,
            citations=citations,
            citation_ids=citation_ids,
            hits=[
                {
                    "rank": i + 1,
                    "rerank2_score": round(float(score), 4),
                    **meta,
                }
                for i, (meta, _txt, score) in enumerate(final)
            ],
            retrieval_audit=audit.__dict__,
            n_seeds=len(seeds),
            n_neighbors_added=n_neighbors_added,
            n_final=len(final),
            elapsed_s=round(time.time() - t0, 3),
            elapsed_breakdown=timings,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_context(self, final: list[tuple[dict, str, float]]) -> str:
        """Render top-K with headers that include the law display title.

        The header format mirrors the canonical citation produced by
        ``src.citations.format_citation`` so the LLM has the exact string to
        copy into its answer.
        """
        parts: list[str] = []
        for meta, text, score in final:
            if meta["kind"] == "seed":
                law_id = meta["law_id"]
                display = self._law_display(law_id)
                header = (
                    f"## [{meta['clause_id']}] {display}, "
                    f"Điều {meta['article_n']}. {meta['article_title']} — "
                    f"Khoản {meta['clause_n']}  "
                    f"(rerank2={score:.3f})"
                )
            else:  # neighbor (Article via REFERS_TO)
                target_id = meta["target_id"]
                law_id = target_id.split(".")[0]
                display = self._law_display(law_id)
                header = (
                    f"## [{target_id}] {display}, {meta['target_label']}: {meta['target_title']}  "
                    f"(via REFERS_TO hop={meta['hop_distance']} from {meta['seed_clause_id']}; "
                    f"rerank2={score:.3f})"
                )
            parts.append(header)
            parts.append(text)
            parts.append("")

        ctx = "\n".join(parts)
        if len(ctx) > MAX_CONTEXT_CHARS:
            ctx = ctx[:MAX_CONTEXT_CHARS] + "\n\n... (context truncated)"
        return ctx

    def _law_display(self, law_id: str) -> str:
        meta = self._law_meta.get(law_id)
        if meta and meta.canonical_title:
            return meta.canonical_title
        return law_id  # fallback — registry alias resolution would still work

    def _parse_citations(self, answer_text: str) -> tuple[list[str], list[str]]:
        refs = parse_displayed_citations(answer_text, self._registry)
        seen: set[str] = set()
        citations: list[str] = []
        ids: list[str] = []
        for ref in refs:
            cid = ref.item_id
            if cid in seen:
                continue
            seen.add(cid)
            citations.append(format_citation(ref))
            ids.append(cid)
        return citations, ids

    # ------------------------------------------------------------------
    # Same verify helper as RagPipeline for record parity
    # ------------------------------------------------------------------

    def verify_citations(self, ids: list[str]) -> dict[str, bool]:
        if not ids:
            return {}
        with self.driver.session(database=self.db) as s:
            rows = s.run(
                """
                UNWIND $ids AS id
                OPTIONAL MATCH (n) WHERE n.id = id AND (n:Article OR n:Clause OR n:Point)
                RETURN id, n IS NOT NULL AS exists
                """,
                ids=ids,
            ).data()
        return {r["id"]: r["exists"] for r in rows}

    def close(self):
        try:
            self.driver.close()
        except Exception:
            pass
