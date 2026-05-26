"""Build elite-compatible corpus + ontology from Luật BHXH 41/2024/QH15.

Convert `data/interim/structured_law.json` (B1 parse output) thành JSONL
format mà `elite/src/knowledge/bhxh_ontology.py:build_bhxh_ontology` expect:

    {"id": "c<seq>", "text": "<clause text>",
     "document": "Luật BHXH 2024 (41/2024/QH15)",
     "article": "<number>", "clause": "<number>"}

Mỗi Clause → 1 chunk. Article không có Clause (chỉ lead_text) → 1 chunk
với article-level info. Point → KHÔNG tách riêng (text đã embed trong
Clause.full_text từ B1).

Sau khi sinh JSONL, gọi `build_ontology_file` từ elite để build ontology.

Output:
    data/eval/elite_corpus_2024.jsonl
    data/eval/elite_ontology_2024.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make elite importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ELITE_ROOT = _REPO_ROOT / "elite"
for _p in (_REPO_ROOT, _ELITE_ROOT, _ELITE_ROOT / "src"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


STRUCTURED_PATH = Path("data/interim/structured_law.json")
OUT_CORPUS = Path("data/eval/elite_corpus_2024.jsonl")
OUT_ONTOLOGY = Path("data/eval/elite_ontology_2024.json")

DOCUMENT_LABEL = "Luật BHXH 2024 (41/2024/QH15)"


def build_corpus_jsonl() -> int:
    """Trả về số chunks generated."""
    if not STRUCTURED_PATH.exists():
        print(
            f"FAIL: thiếu {STRUCTURED_PATH}. Chạy `python -m src.parse_docx` trước.",
            file=sys.stderr,
        )
        return 0

    with STRUCTURED_PATH.open(encoding="utf-8") as f:
        structured = json.load(f)

    chunks: list[dict] = []
    seq = 0
    skipped_lead_only = 0

    for chapter in structured["chapters"]:
        for article in chapter["articles"]:
            art_n = article["number"]

            # Article có Clauses → 1 chunk per Clause
            if article["clauses"]:
                for clause in article["clauses"]:
                    seq += 1
                    # full_text bao gồm "1. ..." + các điểm (a) (b) ...
                    text = clause.get("full_text") or clause["text"]
                    chunks.append(
                        {
                            "id": f"c{seq:04d}",
                            "text": text,
                            "document": DOCUMENT_LABEL,
                            "article": str(art_n),
                            "clause": str(clause["number"]),
                        }
                    )
            elif article.get("lead_text"):
                # Article chỉ có lead_text (no numbered clauses) → 1 chunk
                # với article-level metadata, clause=None
                seq += 1
                # Bao gồm tiêu đề Điều cho rõ ràng
                text_with_header = (
                    f"Điều {art_n}. {article['title']}\n{article['lead_text']}"
                )
                chunks.append(
                    {
                        "id": f"c{seq:04d}",
                        "text": text_with_header,
                        "document": DOCUMENT_LABEL,
                        "article": str(art_n),
                        # clause = None để elite hiểu là article-level (no clause)
                    }
                )
            else:
                skipped_lead_only += 1

    OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CORPUS.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False))
            f.write("\n")

    print(f"  ✓ {len(chunks)} chunks → {OUT_CORPUS}")
    if skipped_lead_only:
        print(f"    (skipped {skipped_lead_only} Article không có clause + lead_text)")

    # Stats
    by_article = {}
    for c in chunks:
        by_article.setdefault(c["article"], 0)
        by_article[c["article"]] += 1
    n_articles = len(by_article)
    print(f"    cover {n_articles}/141 Articles")
    return len(chunks)


def build_ontology() -> dict:
    """Gọi elite's build_ontology_file để gen ontology JSON.

    Note: import dạng `knowledge.bhxh_ontology` (không có `src.`) vì
    elite/src đã được insert vào sys.path. Tránh conflict với main `src/`
    package của project (không có `src.knowledge` namespace).
    """
    from knowledge.bhxh_ontology import build_ontology_file  # type: ignore

    ontology = build_ontology_file(OUT_CORPUS, OUT_ONTOLOGY, pretty=True)
    n_nodes = ontology.get("node_count", 0)
    n_edges = ontology.get("edge_count", 0)
    n_chunks = len(ontology.get("chunks") or [])
    print(f"  ✓ ontology built → {OUT_ONTOLOGY}")
    print(f"    nodes={n_nodes}, edges={n_edges}, chunks={n_chunks}")
    return ontology


def main() -> int:
    print(f"=== STEP 1: build corpus JSONL từ {STRUCTURED_PATH} ===")
    n = build_corpus_jsonl()
    if n == 0:
        return 1

    print(f"\n=== STEP 2: build ontology JSON ===")
    try:
        build_ontology()
    except Exception as e:
        print(f"FAIL build ontology: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2

    print(f"\n=== DONE ===")
    print(f"  corpus  : {OUT_CORPUS}")
    print(f"  ontology: {OUT_ONTOLOGY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
