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

from eval_core import paths
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


# ---------------------------------------------------------------------------
# Experiment-aware entry point
# ---------------------------------------------------------------------------


def _enrich_records_with_gold(
    records: list[dict[str, Any]],
    gold_map: dict[int, list[str]],
    question_map: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Variant of build_metric_record_groups but returns flat list with arm kept.

    The Experiment-based flow keeps `arm` on each record because records are
    pulled from arm-named subfolders (or inherited from a parent experiment).
    Records carry the arm tag so downstream grouping is by record["arm"].
    """
    out: list[dict[str, Any]] = []
    for rec in records:
        stt = int(rec["stt"])
        if stt not in gold_map:
            raise RuntimeError(f"No validated gold citations for stt={stt}")
        q = question_map.get(stt, {})
        enriched = dict(rec)
        enriched["record_id"] = str(stt)
        enriched["gold_articles"] = gold_map[stt]
        if not enriched.get("gold_answer"):
            enriched["gold_answer"] = q.get("gold_answer")
        if not enriched.get("gold_citations_raw"):
            enriched["gold_citations_raw"] = q.get("gold_citations_raw")
        out.append(enriched)
    return out


def _load_multimodel_records(experiment) -> dict[str, list[dict[str, Any]]]:
    """Walk ``results/multimodel/<combo>/`` and return {combo_name: [records]}.

    Combo names are taken verbatim from the subdirectory names
    (e.g. ``logic_lm_graphrag__gpt-4_1``).
    """
    mm_dir = experiment.results_dir / paths.MULTIMODEL_SUBDIR
    if not mm_dir.is_dir():
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for combo_dir in sorted(p for p in mm_dir.iterdir() if p.is_dir()):
        combo_records: list[dict[str, Any]] = []
        for rec_path in sorted(combo_dir.glob("A*.json")):
            try:
                rec = json.loads(rec_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON record: {rec_path}") from exc
            # Force the combo to be the arm so grouping uses combo name.
            rec["arm"] = combo_dir.name
            rec["_record_path"] = str(rec_path)
            combo_records.append(rec)
        if combo_records:
            out[combo_dir.name] = combo_records
    return out


def compute_metrics_for_experiment(
    experiment,
    arms_filter: list[str] | None = None,
    include_multimodel: bool = True,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Run the full metric pipeline for an experiment.

    1. Validate ``gold_citations_raw`` against the registry; write
       ``gold_citations_normalized.json`` into ``experiment.metrics_dir``.
    2. Pull records for every arm in ``experiment.arms`` (single-model;
       inherited via ``records_for_arm``).
    3. If ``include_multimodel``, also pull records for every combo under
       ``experiment.results_dir / 'multimodel'``.
    4. Compute academic metrics per arm/combo.
    5. Write ``academic_metrics.json``, ``academic_metrics.csv`` to
       ``experiment.metrics_dir`` and ``academic_report.md`` to
       ``experiment.report_dir``.

    Returns the result dict (also written to disk).
    """
    from eval_core.experiment import Experiment

    if not isinstance(experiment, Experiment):
        raise TypeError(f"expected Experiment, got {type(experiment).__name__}")

    experiment.validate()

    metrics_dir = experiment.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_dir = experiment.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gold validation — writes normalized + errors into metrics/
    questions_path = experiment.dataset.questions
    ok, summary = validate_gold_citations(
        questions_path,
        registry_path,
        metrics_dir,
    )
    if not ok:
        raise RuntimeError(
            "gold citation validation failed; fix dataset before computing metrics. "
            f"See {summary['errors_path']}"
        )

    gold_map = _load_gold_map(metrics_dir / NORMALIZED_OUT)
    question_map = _load_question_map(questions_path)

    # 2. Single-model arms — respects inheritance via Experiment.records_for_arm
    declared_arms = list(experiment.arms.keys())
    if arms_filter is not None:
        unknown = [a for a in arms_filter if a not in declared_arms]
        if unknown:
            raise ValueError(
                f"arms_filter contains unknown arms: {unknown}. "
                f"Declared: {declared_arms}"
            )
        arms_to_use = list(arms_filter)
    else:
        arms_to_use = declared_arms

    all_records: list[dict[str, Any]] = []
    inheritance_trace: dict[str, str] = {}
    for arm in arms_to_use:
        records = experiment.records_for_arm(arm)
        for rec in records:
            rec["arm"] = arm  # ensure arm tag matches the logical arm name
        all_records.extend(records)
        source = experiment.records_source(arm)
        inheritance_trace[arm] = source.name

    enriched = _enrich_records_with_gold(all_records, gold_map, question_map)

    # Group by arm
    record_groups: dict[str, list[dict[str, Any]]] = {}
    for rec in enriched:
        record_groups.setdefault(str(rec["arm"]), []).append(rec)

    # 3. Multimodel combos
    if include_multimodel:
        mm_groups = _load_multimodel_records(experiment)
        for combo, recs in mm_groups.items():
            enriched_combo = _enrich_records_with_gold(recs, gold_map, question_map)
            record_groups[combo] = enriched_combo
            inheritance_trace[combo] = experiment.name  # combos always own their records

    metadata = {
        "experiment_path": str(experiment.path),
        "experiment_name": experiment.name,
        "questions_path": str(questions_path),
        "gold_artifact": str(metrics_dir / NORMALIZED_OUT),
        "arms_filter": arms_filter,
        "records_source": inheritance_trace,
    }
    result = compute_grouped_academic_metrics(
        record_groups=record_groups,
        registry_path=registry_path,
        metadata=metadata,
    )
    return write_grouped_academic_metrics_outputs(
        result,
        output_dir=metrics_dir,
        metrics_out=paths.metrics_json_path(experiment.path),
        csv_out=paths.metrics_csv_path(experiment.path),
        report_out=paths.report_md_path(experiment.path),
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
