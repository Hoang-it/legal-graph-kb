"""Experiment-owned loader for academic metrics.

This module knows the experiment result-folder layout and arm selection rules.
It loads records, attaches validated gold articles, groups records by experiment
arm, then calls :func:`eval_core.metrics.compute_academic_metrics` for each arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval_core.arms import MAIN_EXPERIMENT_ARMS, parse_metrics_arms
from eval_core.gold import (
    DEFAULT_QUESTIONS,
    NORMALIZED_OUT,
    validate_gold_citations,
)
from eval_core.metrics import (
    DEFAULT_OUTPUT_DIR,
    METRIC_VERSION,
    compute_academic_metrics,
)
from eval_core.report import write_grouped_academic_metrics_outputs
from src.citations import DEFAULT_REGISTRY_PATH, load_registry

DEFAULT_RESULTS_ROOT = Path("data/eval/results")


def _load_gold_map(path: Path) -> dict[int, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, list[str]] = {}
    for stt, rec in (data.get("records") or {}).items():
        out[int(stt)] = list(rec.get("gold_articles") or [])
    return out


def _load_question_map(path: Path) -> dict[int, dict[str, Any]]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    return {int(q["stt"]): q for q in questions}


def load_result_records(
    results_root: Path,
    arms: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load experiment result records from ``{results_root}/{arm}/A*.json``."""
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")
    if arms:
        arm_dirs = [results_root / arm for arm in arms]
        missing = [str(path) for path in arm_dirs if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing result arm directories: {missing}")
    else:
        arm_dirs = sorted(p for p in results_root.iterdir() if p.is_dir())

    records: list[dict[str, Any]] = []
    for arm_dir in arm_dirs:
        for path in sorted(arm_dir.glob("A*.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON record: {path}") from exc
            rec.setdefault("arm", arm_dir.name)
            rec["_record_path"] = str(path)
            records.append(rec)
    if not records:
        raise ValueError(f"No result records found under {results_root}")
    return records


def build_metric_record_groups(
    records: list[dict[str, Any]],
    gold_map: dict[int, list[str]],
    question_map: dict[int, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Attach validated gold fields and group records by experiment arm."""
    out: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        arm = str(rec["arm"])
        stt = int(rec["stt"])
        if stt not in gold_map:
            raise RuntimeError(f"No validated gold citations for stt={stt}")
        q = question_map.get(stt, {})
        enriched = {
            key: value
            for key, value in rec.items()
            if key not in {"arm", "stt"}
        }
        enriched["record_id"] = str(stt)
        enriched["gold_articles"] = gold_map[stt]
        if not enriched.get("gold_answer"):
            enriched["gold_answer"] = q.get("gold_answer")
        if not enriched.get("gold_citations_raw"):
            enriched["gold_citations_raw"] = q.get("gold_citations_raw")
        out.setdefault(arm, []).append(enriched)
    return out


def load_academic_metric_records(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    questions_path: Path = DEFAULT_QUESTIONS,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    academic_dir: Path = DEFAULT_OUTPUT_DIR / "academic",
    arms: list[str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    ok, summary = validate_gold_citations(questions_path, registry_path, academic_dir)
    if not ok:
        raise RuntimeError(
            "gold citation validation failed; fix dataset before computing metrics. "
            f"See {summary['errors_path']}"
        )

    raw_records = load_result_records(results_root, arms=arms)
    question_map = _load_question_map(questions_path)
    gold_map = _load_gold_map(academic_dir / NORMALIZED_OUT)
    record_groups = build_metric_record_groups(raw_records, gold_map, question_map)
    metadata = {
        "results_root": str(results_root),
        "questions_path": str(questions_path),
        "gold_artifact": str(academic_dir / NORMALIZED_OUT),
        "arms_filter": arms,
    }
    return record_groups, metadata


def compute_grouped_academic_metrics(
    record_groups: dict[str, list[dict[str, Any]]],
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    result_records: dict[str, list[dict[str, Any]]] = {}
    aggregates: dict[str, dict[str, Any]] = {}
    bertscore_metadata: dict[str, dict[str, Any]] = {}
    n_input_records = 0

    for arm, records in record_groups.items():
        arm_result = compute_academic_metrics(
            records=records,
            registry=registry,
            registry_path=registry_path,
            metadata=metadata,
        )
        result_records[arm] = arm_result["records"]
        aggregates[arm] = arm_result["aggregate"]
        bertscore_metadata[arm] = arm_result["bertscore_metadata"]
        n_input_records += arm_result["n_input_records"]

    return {
        "metric_version": METRIC_VERSION,
        "n_input_records": n_input_records,
        "registry_path": str(registry_path),
        "gold_source": "record.gold_articles",
        "metadata": metadata or {},
        "bertscore_metadata": bertscore_metadata,
        "records": result_records,
        "aggregates": aggregates,
    }


def compute_experiment_academic_metrics(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    questions_path: Path = DEFAULT_QUESTIONS,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    academic_dir: Path | None = None,
    metrics_out: Path | None = None,
    csv_out: Path | None = None,
    report_out: Path | None = None,
    arms: list[str] | None = None,
) -> dict[str, Any]:
    academic_dir = academic_dir or output_dir / "academic"
    record_groups, metadata = load_academic_metric_records(
        results_root=results_root,
        questions_path=questions_path,
        registry_path=registry_path,
        academic_dir=academic_dir,
        arms=arms,
    )
    result = compute_grouped_academic_metrics(
        record_groups=record_groups,
        registry_path=registry_path,
        metadata=metadata,
    )
    return write_grouped_academic_metrics_outputs(
        result,
        output_dir=output_dir,
        metrics_out=metrics_out,
        csv_out=csv_out,
        report_out=report_out,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Load experiment records and compute metrics.")
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--academic-dir", type=Path, default=None)
    p.add_argument("--metrics-out", type=Path, default=None)
    p.add_argument("--csv-out", type=Path, default=None)
    p.add_argument("--report-out", type=Path, default=None)
    p.add_argument(
        "--arms",
        type=str,
        default="main",
        help=(
            "Arm selection: 'main' (default), 'all' to discover every result dir, "
            "or comma-separated arms. Main = "
            + ",".join(MAIN_EXPERIMENT_ARMS)
        ),
    )
    args = p.parse_args()
    try:
        arms = parse_metrics_arms(args.arms)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    try:
        result = compute_experiment_academic_metrics(
            results_root=args.results_root,
            questions_path=args.questions,
            registry_path=args.registry,
            output_dir=args.output_dir,
            academic_dir=args.academic_dir,
            metrics_out=args.metrics_out,
            csv_out=args.csv_out,
            report_out=args.report_out,
            arms=arms,
        )
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(
        f"OK: wrote {result['metrics_out']}, {result['csv_out']}, {result['report_out']} "
        f"for {len(result['records'])} arms."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
