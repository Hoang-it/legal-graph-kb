"""Validate Phase 6 extracted Prolog records with real SWI-Prolog consults."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.prolog_utils import validate_and_namespace_record

DEFAULT_ROOT = Path("data/eval/extracted_prolog")
DEFAULT_REPORT = Path("reports/prolog_validation_summary.json")


def validate_record(record: dict[str, Any], timeout_s: int = 5) -> dict[str, Any]:
    extraction = record.get("extraction") or record
    law_code = record.get("law_code") or str(record.get("clause_id", "")).split(".")[0]
    result = validate_and_namespace_record(extraction, law_code, timeout_s=timeout_s)
    return result


def iter_records(root: Path, law: str | None = None):
    dirs = [root / law] if law else sorted(p for p in root.iterdir() if p.is_dir())
    for d in dirs:
        if not d.exists():
            continue
        for path in sorted(d.glob("*.json")):
            yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--law", default=None)
    parser.add_argument("--write-back", action="store_true")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--timeout", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root)
    paths = list(iter_records(root, args.law))
    summary = {
        "root": str(root),
        "law": args.law,
        "total": len(paths),
        "passed": 0,
        "failed": 0,
        "by_law": defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0}),
        "failure_stages": Counter(),
        "failures": [],
    }
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        result = validate_record(record, timeout_s=args.timeout)
        law_code = record.get("law_code") or str(record.get("clause_id", "")).split(".")[0]
        summary["by_law"][law_code]["total"] += 1
        if result.get("ok"):
            summary["passed"] += 1
            summary["by_law"][law_code]["passed"] += 1
        else:
            summary["failed"] += 1
            summary["by_law"][law_code]["failed"] += 1
            summary["failure_stages"][result.get("stage", "unknown")] += 1
            summary["failures"].append(
                {
                    "path": str(path),
                    "clause_id": record.get("clause_id"),
                    "stage": result.get("stage"),
                    "diagnostic": result.get("diagnostic"),
                }
            )
        if args.write_back:
            record["validation"] = result
            record["status"] = "validated" if result.get("ok") else "invalid"
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    serializable = {
        **summary,
        "by_law": dict(summary["by_law"]),
        "failure_stages": dict(summary["failure_stages"]),
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(serializable, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
