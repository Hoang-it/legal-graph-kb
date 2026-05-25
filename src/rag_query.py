"""B7 — RAG query layer.

Pipeline:
1. Câu hỏi → embed bằng BGE-M3 (cùng model B5).
2. Vector search top-K Clause trong Neo4j (`clause_vec` index).
3. Expand graph quanh các Clause: Article cha + semantic edges
   (ENTITLED_TO/HAS_OBLIGATION/REQUIRES/...) anchor về cùng Clause,
   + REFERENCES nội bộ + CITES_EXTERNAL.
4. Build context (giữ Clause.id để LLM dùng làm citation).
5. Gọi GPT-4o-mini với SYSTEM_PROMPT: chỉ trả lời theo CONTEXT, kèm
   citation dạng [Điều X khoản Y].
6. Parse citation từ answer + map ngược về Clause/Article trong DB
   để verify (provenance roundtrip).

CLI:
    python -m src.rag_query --question "Khi nào được hưởng lương hưu?"
    python -m src.rag_query -q "..." -v --top-k 10
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Defensive: nếu OPENAI_BASE_URL được set rỗng, pop để SDK dùng default api.openai.com
# (nếu giữ empty string, SDK sẽ dùng nó làm URL → APIConnectionError)
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")
DB = os.getenv("NEO4J_DATABASE", "neo4j")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SEMANTIC_EDGE_TYPES = [
    "ENTITLED_TO",
    "HAS_OBLIGATION",
    "HAS_RIGHT",
    "REQUIRES",
    "APPLIES_TO",
    "PAID_FROM",
    "MANAGES",
    "RESPONSIBLE_FOR",
    "PROHIBITED_BY",
    "DEFINES",
]


SYSTEM_PROMPT = """Bạn là trợ lý pháp lý chuyên về Luật Bảo hiểm xã hội số 41/2024/QH15 của Việt Nam.

QUY TẮC TUYỆT ĐỐI:
1. CHỈ trả lời dựa trên các đoạn điều luật trong CONTEXT phía dưới. KHÔNG dùng kiến thức ngoài.
2. Nếu CONTEXT không đủ thông tin để trả lời, hãy nói rõ "Theo các điều luật được cung cấp, tôi không có đủ thông tin để trả lời chính xác câu hỏi này" và liệt kê các Điều liên quan có trong CONTEXT (nếu có).
3. MỌI khẳng định pháp lý PHẢI kèm citation dạng `[Điều X khoản Y]` hoặc `[Điều X khoản Y điểm z]`. Citation lấy từ ID `L41_2024.A<X>.K<Y>.<z>` trong CONTEXT — convert sang format dễ đọc.
4. Trả lời bằng tiếng Việt, ngắn gọn nhưng đầy đủ. Cấu trúc: trả lời chính → giải thích → liệt kê citation.
5. KHÔNG bịa số liệu, ngày tháng, mức tiền nếu CONTEXT không nói rõ.
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SearchHit:
    clause_id: str
    article_id: str
    article_n: int
    article_title: str
    clause_n: int
    text: str
    score: float


@dataclass
class RagAnswer:
    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    citation_ids: list[str] = field(default_factory=list)
    hits: list[SearchHit] = field(default_factory=list)
    n_semantic_edges: int = 0
    n_refs: int = 0
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class RagPipeline:
    def __init__(self):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(URI, auth=(USER, PWD))
        self.driver.verify_connectivity()
        self._embed_model = None
        self._openai = None

    @property
    def embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer

            print(f"Loading {EMBED_MODEL} on {EMBED_DEVICE}...", file=sys.stderr)
            t0 = time.time()
            self._embed_model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
            print(f"  loaded in {time.time() - t0:.1f}s", file=sys.stderr)
        return self._embed_model

    @property
    def openai(self):
        if self._openai is None:
            from openai import OpenAI

            self._openai = OpenAI()
        return self._openai

    # ------------------------------------------------------------------

    def vector_search(self, query: str, top_k: int = 8) -> list[SearchHit]:
        q_emb = self.embed_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()

        with self.driver.session(database=DB) as s:
            rows = s.run(
                """
                CALL db.index.vector.queryNodes('clause_vec', $k, $q)
                YIELD node, score
                MATCH (a:Article)-[:HAS_CLAUSE]->(node)
                RETURN node.id AS clause_id, node.text AS text, score,
                       a.id AS article_id, a.title AS article_title,
                       a.number AS article_n, node.number AS clause_n
                ORDER BY score DESC
            """,
                k=top_k,
                q=q_emb,
            ).data()
        return [SearchHit(**r) for r in rows]

    def expand(self, clause_ids: list[str]) -> dict:
        """Lấy semantic edges + refs neo về các clause này."""
        with self.driver.session(database=DB) as s:
            edges = s.run(
                """
                MATCH (a)-[r]->(b)
                WHERE r.source_clause IN $cids
                  AND type(r) IN $types
                RETURN type(r) AS type,
                       coalesce(a.name, a.id) AS src_name,
                       labels(a)[0] AS src_label,
                       coalesce(b.name, b.description, b.id) AS dst_name,
                       labels(b)[0] AS dst_label,
                       r.source_clause AS source_clause,
                       r.source_text AS evidence
                LIMIT 30
            """,
                cids=clause_ids,
                types=SEMANTIC_EDGE_TYPES,
            ).data()

            refs = s.run(
                """
                MATCH (n)-[r:REFERENCES|CITES_EXTERNAL]->(m)
                WHERE r.source_clause IN $cids
                RETURN type(r) AS type, n.id AS src, m.id AS dst,
                       coalesce(m.title, m.id) AS dst_label, r.span AS span,
                       r.source_clause AS source_clause
                LIMIT 20
            """,
                cids=clause_ids,
            ).data()
        return {"edges": edges, "refs": refs}

    def build_context(self, hits: list[SearchHit], expansion: dict, max_chars: int = 7000) -> str:
        parts: list[str] = []
        parts.append("# CÁC ĐOẠN ĐIỀU LUẬT LIÊN QUAN (sắp theo relevance giảm dần)\n")
        for h in hits:
            parts.append(
                f"\n## [{h.clause_id}] Điều {h.article_n}. {h.article_title} - Khoản {h.clause_n} "
                f"(score={h.score:.3f})"
            )
            parts.append(h.text)

        if expansion["edges"]:
            parts.append("\n\n# QUAN HỆ NGỮ NGHĨA TRÍCH XUẤT")
            for e in expansion["edges"][:20]:
                parts.append(
                    f"- [{e['source_clause']}] "
                    f"{e['src_name']}({e['src_label']}) "
                    f"--{e['type']}--> "
                    f"{e['dst_name']}({e['dst_label']})"
                )
                if e.get("evidence"):
                    snip = e["evidence"][:200]
                    parts.append(f'    Bằng chứng: "{snip}"')

        if expansion["refs"]:
            parts.append("\n\n# VIỆN DẪN LIÊN QUAN")
            for r in expansion["refs"]:
                parts.append(f"- Từ {r['source_clause']}: \"{r['span']}\" → {r['dst']}")

        ctx = "\n".join(parts)
        if len(ctx) > max_chars:
            ctx = ctx[:max_chars] + "\n\n... (context truncated)"
        return ctx

    def parse_citations(self, answer: str) -> tuple[list[str], list[str]]:
        """Trả về (citations_str, citation_ids).

        Citation_ids = ID dạng L41_2024.A<X>.K<Y>[.<z>] từ "[Điều X khoản Y]".
        """
        # Pattern khớp [Điều 64], [Điều 64 khoản 1], [Điều 64 khoản 1 điểm a]
        pat = re.compile(r"\[Điều\s+(\d+)(?:\s+khoản\s+(\d+))?(?:\s+điểm\s+([a-zđ]))?\]")
        citations: list[str] = []
        ids: list[str] = []
        for m in pat.finditer(answer):
            citations.append(m.group(0))
            art, cl, pt = m.group(1), m.group(2), m.group(3)
            cid = f"L41_2024.A{art}"
            if cl:
                cid += f".K{cl}"
                if pt:
                    cid += f".{pt}"
            ids.append(cid)
        # Dedup giữ thứ tự
        return list(dict.fromkeys(citations)), list(dict.fromkeys(ids))

    def verify_citations(self, ids: list[str]) -> dict[str, bool]:
        """Kiểm tra mỗi citation ID có tồn tại thực trong Neo4j không."""
        if not ids:
            return {}
        with self.driver.session(database=DB) as s:
            rows = s.run(
                """
                UNWIND $ids AS id
                OPTIONAL MATCH (n) WHERE n.id = id AND (n:Article OR n:Clause OR n:Point)
                RETURN id, n IS NOT NULL AS exists
            """,
                ids=ids,
            ).data()
        return {r["id"]: r["exists"] for r in rows}

    # ------------------------------------------------------------------

    def ask(self, question: str, top_k: int = 8, verbose: bool = False) -> RagAnswer:
        t0 = time.time()

        hits = self.vector_search(question, top_k=top_k)
        if verbose:
            print(f"\n[vector_search] {len(hits)} hits:", file=sys.stderr)
            for h in hits[:5]:
                print(f"  {h.score:.3f}  {h.clause_id}: {h.text[:80]}", file=sys.stderr)

        clause_ids = [h.clause_id for h in hits]
        expansion = self.expand(clause_ids)
        if verbose:
            print(
                f"[expand] {len(expansion['edges'])} semantic edges, "
                f"{len(expansion['refs'])} refs",
                file=sys.stderr,
            )

        context = self.build_context(hits, expansion)
        if verbose:
            print(f"[context] {len(context)} chars", file=sys.stderr)

        resp = self.openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\n---\n\nCÂU HỎI: {question}"},
            ],
            temperature=0,
        )
        answer_text = resp.choices[0].message.content or ""

        citations, citation_ids = self.parse_citations(answer_text)

        return RagAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            citation_ids=citation_ids,
            hits=hits,
            n_semantic_edges=len(expansion["edges"]),
            n_refs=len(expansion["refs"]),
            elapsed_s=round(time.time() - t0, 2),
        )

    def close(self):
        self.driver.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-q", "--question", required=True)
    p.add_argument("-k", "--top-k", type=int, default=8)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--verify", action="store_true", help="Kiểm tra citations có thật trong DB không."
    )
    args = p.parse_args()

    if not all([URI, USER, PWD]):
        print("FAIL: thiếu NEO4J_* env vars", file=sys.stderr)
        return 1

    pipeline = RagPipeline()
    try:
        result = pipeline.ask(args.question, top_k=args.top_k, verbose=args.verbose)

        print(f"\n{'=' * 70}")
        print(f"CÂU HỎI: {result.question}")
        print(f"{'=' * 70}")
        print(result.answer)
        print(f"\n{'-' * 70}")
        print(f"Citations: {result.citations}")
        print(f"Citation IDs: {result.citation_ids}")
        print(
            f"Vector hits: {len(result.hits)}, "
            f"semantic edges expanded: {result.n_semantic_edges}, "
            f"refs: {result.n_refs}"
        )
        print(f"Elapsed: {result.elapsed_s}s")

        if args.verify and result.citation_ids:
            verified = pipeline.verify_citations(result.citation_ids)
            print("\nCitation verification:")
            for cid, ok in verified.items():
                mark = "✓" if ok else "✗"
                print(f"  {mark} {cid}")
    finally:
        pipeline.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
