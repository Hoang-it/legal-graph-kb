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
from typing import Any

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
