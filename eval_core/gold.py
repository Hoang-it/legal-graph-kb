"""Validate dataset gold citations before academic metrics.

`gold_citations_raw` remains the source of truth. This script only parses it
into a generated audit artifact and fails hard if any record is unusable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from src.citations import (
    DEFAULT_REGISTRY_PATH,
    load_registry,
    parse_gold_citations_raw,
)

DEFAULT_QUESTIONS = Path("data/eval/questions_200.json")
DEFAULT_OUT_DIR = Path("metrics/academic")
NORMALIZED_OUT = "gold_citations_normalized.json"
ERRORS_OUT = "gold_citation_validation_errors.csv"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_gold_citations(
    questions_path: Path = DEFAULT_QUESTIONS,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> tuple[bool, dict[str, Any]]:
    registry = load_registry(registry_path)
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized: dict[str, Any] = {
        "questions_path": str(questions_path),
        "questions_sha256": _sha256(questions_path),
        "registry_path": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "source": "gold_citations_raw",
        # `granularity` = "tuple" since 2026-05-31: each record carries
        # `gold_items` as the full (law_id, article, clause, point) tuple
        # used by the strict-tuple-equal E2E metric (v5 plan §5).
        # `gold_articles` is kept as a derived field (article-level dedupe)
        # for retrieval-only diagnostic experiments that compare against
        # the dense article pool — see plan §5 scope distinction.
        "granularity": "tuple",
        "records": {},
    }
    errors: list[dict[str, Any]] = []

    for q in questions:
        stt = q.get("stt")
        raw = q.get("gold_citations_raw")
        result = parse_gold_citations_raw(raw, registry)
        # Full tuple per gold citation. For the current 200-question dataset
        # 0/200 gold cites have khoản/điểm so `gold_items == gold_articles`;
        # the field exists so that a future dataset extension carrying
        # khoản-level gold is handled by the same code path.
        gold_items = sorted({r.item_id for r in result.refs})
        gold_articles = sorted({r.article_id for r in result.refs})
        if result.errors:
            for err in result.errors:
                errors.append(
                    {
                        "stt": stt,
                        "question": q.get("question", ""),
                        "gold_citations_raw": raw or "",
                        "error_type": err.error_type,
                        "segment": err.text,
                        "detail": err.detail,
                        "suggested_fix": "Update gold_citations_raw with explicit authority and article.",
                    }
                )
        normalized["records"][str(stt)] = {
            "gold_items": gold_items,
            "gold_articles": gold_articles,
            "gold_citations_raw": raw,
        }

    errors_path = out_dir / ERRORS_OUT
    if errors:
        with errors_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "stt",
                    "question",
                    "gold_citations_raw",
                    "error_type",
                    "segment",
                    "detail",
                    "suggested_fix",
                ],
            )
            writer.writeheader()
            writer.writerows(errors)
        return False, {
            "n_questions": len(questions),
            "n_errors": len(errors),
            "errors_path": str(errors_path),
        }

    normalized_path = out_dir / NORMALIZED_OUT
    normalized_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if errors_path.exists():
        errors_path.unlink()
    return True, {
        "n_questions": len(questions),
        "n_errors": 0,
        "normalized_path": str(normalized_path),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Validate gold_citations_raw for academic metrics.")
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    ok, summary = validate_gold_citations(args.questions, args.registry, args.out_dir)
    if ok:
        print(
            f"OK: validated {summary['n_questions']} records. "
            f"Artifact: {summary['normalized_path']}"
        )
        return 0

    print(
        f"FAIL: gold citation validation found {summary['n_errors']} errors. "
        f"Fix dataset before computing metrics. See: {summary['errors_path']}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
