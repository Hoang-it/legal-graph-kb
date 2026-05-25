"""Eval RAG trên 10 câu hỏi gold.

Metrics:
- citation_recall    : ít nhất 1 expected_article xuất hiện trong citations
- citation_validity  : tất cả citation IDs PHẢI tồn tại trong DB (provenance)
- keyword_coverage   : keywords mong đợi xuất hiện trong answer (loose)

Test này tốn LLM cost (~$0.01-0.02 cho 10 câu) — chỉ chạy khi --run-eval.
Mặc định skip để pytest nhanh.
"""

import os
import re
from pathlib import Path

import pytest
import yaml
from dotenv import load_dotenv

load_dotenv()

EVAL_FILE = Path("tests/eval_questions.yaml")

# Skip mặc định trừ khi env RUN_RAG_EVAL=1
SKIP_EVAL = os.getenv("RUN_RAG_EVAL", "0") != "1"


pytestmark = pytest.mark.skipif(
    SKIP_EVAL,
    reason="Skip RAG eval (tốn LLM cost). Bật bằng: RUN_RAG_EVAL=1 pytest tests/test_rag_eval.py",
)


@pytest.fixture(scope="module")
def pipeline():
    from src.rag_query import RagPipeline

    p = RagPipeline()
    yield p
    p.close()


@pytest.fixture(scope="module")
def eval_set() -> list[dict]:
    with EVAL_FILE.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def results(pipeline, eval_set) -> list[dict]:
    """Chạy RAG 1 lần cho cả module, cache results."""
    out = []
    for item in eval_set:
        r = pipeline.ask(item["q"], top_k=8)
        verified = pipeline.verify_citations(r.citation_ids)
        out.append(
            {
                "q": item["q"],
                "expected_article": item["expected_article"],
                "expected_keywords": item.get("expected_keywords", []),
                "answer": r.answer,
                "citations": r.citations,
                "citation_ids": r.citation_ids,
                "verified": verified,
                "elapsed_s": r.elapsed_s,
            }
        )
    return out


def test_citation_recall(results):
    """Ít nhất 1 expected_article xuất hiện trong citations cho mỗi câu."""
    misses = []
    for r in results:
        expected = r["expected_article"]
        expected_pattern = re.compile(rf"\[Điều\s+{expected}\b")
        hit = any(expected_pattern.search(c) for c in r["citations"])
        if not hit:
            misses.append(f"{r['q']!r} → expected A{expected}, got {r['citations']}")
    recall = (len(results) - len(misses)) / len(results)
    print(f"\nCitation recall: {recall:.0%} ({len(results) - len(misses)}/{len(results)})")
    for m in misses:
        print(f"  MISS: {m}")
    assert recall >= 0.7, f"Recall {recall:.0%} < 70%"


def test_citation_validity(results):
    """Mọi citation ID phải tồn tại trong DB (không bịa)."""
    all_invalid = []
    for r in results:
        for cid, ok in r["verified"].items():
            if not ok:
                all_invalid.append(f"{r['q'][:40]!r}: {cid}")
    assert not all_invalid, (
        f"Có {len(all_invalid)} citation BỊA (không tồn tại trong DB):\n  "
        + "\n  ".join(all_invalid[:10])
    )


def test_keyword_coverage(results):
    """Loose: ≥50% từ khoá expected có trong answer (case-insensitive)."""
    failed = []
    for r in results:
        kws = r["expected_keywords"]
        if not kws:
            continue
        answer_lower = r["answer"].lower()
        hits = sum(1 for kw in kws if kw.lower() in answer_lower)
        coverage = hits / len(kws)
        if coverage < 0.5:
            failed.append(f"{r['q'][:40]!r}: hit {hits}/{len(kws)} keywords {kws}")
    assert not failed, "Một số câu thiếu keyword:\n  " + "\n  ".join(failed)


def test_answer_co_it_nhat_1_citation(results):
    """Mọi câu phải có ít nhất 1 citation (cấm trả lời không nguồn)."""
    no_cite = [r["q"] for r in results if not r["citation_ids"]]
    assert not no_cite, f"{len(no_cite)} câu không có citation: {no_cite}"
