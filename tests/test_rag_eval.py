"""Live RAG smoke checks over a small gold set.

These tests are not part of the headline experiment metrics.  They are skipped
by default because they call the LLM and Neo4j.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import yaml
from dotenv import load_dotenv

load_dotenv()

EVAL_FILE = Path("tests/eval_questions.yaml")
SKIP_EVAL = os.getenv("RUN_RAG_EVAL", "0") != "1"

pytestmark = pytest.mark.skipif(
    SKIP_EVAL,
    reason="Skip live RAG eval. Enable with: RUN_RAG_EVAL=1 pytest tests/test_rag_eval.py",
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
    out = []
    for item in eval_set:
        r = pipeline.ask(item["q"], top_k=8)
        out.append(
            {
                "q": item["q"],
                "expected_article": item["expected_article"],
                "expected_keywords": item.get("expected_keywords", []),
                "answer": r.answer,
                "citations": r.citations,
                "citation_ids": r.citation_ids,
                "verified": pipeline.verify_citations(r.citation_ids),
                "elapsed_s": r.elapsed_s,
            }
        )
    return out


def test_expected_article_is_cited(results):
    misses = []
    for r in results:
        expected = r["expected_article"]
        expected_pattern = re.compile(rf"\bĐiều\s+{expected}\b")
        hit = any(expected_pattern.search(c) for c in r["citations"])
        if not hit:
            misses.append(f"{r['q']!r} -> expected article {expected}, got {r['citations']}")
    recall = (len(results) - len(misses)) / len(results)
    assert recall >= 0.7, f"Expected-article hit rate {recall:.0%} < 70%: {misses[:10]}"


def test_citation_ids_resolve_in_db(results):
    unresolved = []
    for r in results:
        for cid, ok in r["verified"].items():
            if not ok:
                unresolved.append(f"{r['q'][:40]!r}: {cid}")
    assert not unresolved, "Unresolved citation IDs:\n  " + "\n  ".join(unresolved[:10])


def test_keyword_coverage(results):
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
    assert not failed, "Some answers miss expected keywords:\n  " + "\n  ".join(failed)


def test_answer_has_at_least_one_citation(results):
    no_cite = [r["q"] for r in results if not r["citation_ids"]]
    assert not no_cite, f"{len(no_cite)} answers have no citation: {no_cite}"
