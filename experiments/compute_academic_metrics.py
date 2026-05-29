"""Experiment-owned loader for academic metrics.

This module knows the experiment result-folder layout and arm selection rules.
It loads records, attaches validated gold articles, groups records by experiment
arm, then calls ``evaluation.compute_academic_metrics`` for each arm.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from evaluation.compute_academic_metrics import (
    DEFAULT_OUTPUT_DIR,
    METRIC_VERSION,
    compute_academic_metrics,
)
from evaluation.validate_gold_citations import (
    DEFAULT_QUESTIONS,
    NORMALIZED_OUT,
    validate_gold_citations,
)
from experiments.arms import MAIN_EXPERIMENT_ARMS, parse_metrics_arms
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


def _fmt(v: Any) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


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


def _write_experiment_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "arm",
        "stt",
        "citation_recall",
        "citation_precision",
        "citation_f1",
        "citation_display_rate",
        "bertscore_p",
        "bertscore_r",
        "bertscore_f1",
        "latency_s",
        "prolog_first_try_solution",
        "repair_invoked",
        "repair_success",
        "pred_citation_parse_errors",
        "record_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "arm": r["arm"],
                    "stt": r.get("record_id", r["record_index"]),
                    "citation_recall": r["citation"]["citation_recall"],
                    "citation_precision": r["citation"]["citation_precision"],
                    "citation_f1": r["citation"]["citation_f1"],
                    "citation_display_rate": r["citation"]["citation_display"][
                        "citation_display_rate"
                    ],
                    "bertscore_p": r.get("bertscore", {}).get("bertscore_p"),
                    "bertscore_r": r.get("bertscore", {}).get("bertscore_r"),
                    "bertscore_f1": r.get("bertscore", {}).get("bertscore_f1"),
                    "latency_s": r["latency"]["latency_s"],
                    "prolog_first_try_solution": r["prolog"]["prolog_first_try_solution"],
                    "repair_invoked": r["prolog"]["repair_invoked"],
                    "repair_success": r["prolog"]["repair_success"],
                    "pred_citation_parse_errors": "|".join(
                        r["citation"]["pred_parse_errors"]
                    ),
                    "record_path": r.get("_record_path", ""),
                }
            )


def _write_experiment_report(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Academic Metrics Report",
        "",
        f"- metric_version: `{result['metric_version']}`",
        f"- n_input_records: `{result['n_input_records']}`",
        f"- gold source: `{result['gold_source']}`",
        "- judge metrics: not included",
        "",
        "## Headline Macro Metrics",
        "",
        "| Arm | n | citation_recall | citation_precision | citation_f1 | citation_display_rate | bertscore_f1 | latency_s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, agg in result["aggregates"].items():
        macro = agg["macro"]
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    str(agg["n_records"]),
                    _fmt(macro["citation_recall"]),
                    _fmt(macro["citation_precision"]),
                    _fmt(macro["citation_f1"]),
                    _fmt(macro["citation_display_rate"]),
                    _fmt(macro["bertscore_f1"]),
                    _fmt(macro["latency_s"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Citation Micro Metrics",
            "",
            "| Arm | recall | precision | display_rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for arm, agg in result["aggregates"].items():
        micro = agg["micro"]
        lines.append(
            f"| {arm} | {_fmt(micro['citation_recall'])} "
            f"(sum={micro['recall_num']}/{micro['recall_denom']}) | "
            f"{_fmt(micro['citation_precision'])} "
            f"(sum={micro['precision_num']}/{micro['precision_denom']}) | "
            f"{_fmt(micro['citation_display_rate'])} "
            f"(sum={micro['display_num']}/{micro['display_denom']}) |"
        )

    lines.extend(
        [
            "",
            "## Prolog Metrics",
            "",
            "| Arm | n_prolog | first_try_solution | repair_invoked | repair_success |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm, agg in result["aggregates"].items():
        pr = agg["prolog"]
        if pr["n_prolog_records"] == 0:
            continue
        lines.append(
            f"| {arm} | {pr['n_prolog_records']} | "
            f"{_fmt(pr['prolog_first_try_solution_rate'])} | "
            f"{_fmt(pr['repair_invoked_rate'])} | "
            f"{_fmt(pr['repair_success_rate'])} "
            f"(sum={pr['repair_success_num']}/{pr['repair_success_denom']}) |"
        )

    lines.extend(
        [
            "",
            "## BERTScore Status",
            "",
            "```json",
            json.dumps(result["bertscore_metadata"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Error Counts",
            "",
            "| Arm | pred_citation_parse_errors | records_with_no_pred_citations |",
            "|---|---:|---:|",
        ]
    )
    for arm, agg in result["aggregates"].items():
        e = agg["error_counts"]
        lines.append(
            f"| {arm} | {e['pred_citation_parse_errors']} | "
            f"{e['records_with_no_pred_citations']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_grouped_academic_metrics_outputs(
    result: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    metrics_out: Path | None = None,
    csv_out: Path | None = None,
    report_out: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    metrics_out = metrics_out or output_dir / "academic_metrics.json"
    csv_out = csv_out or output_dir / "academic_metrics.csv"
    report_out = report_out or output_dir / "academic_report.md"

    result = dict(result)
    result["output_dir"] = str(output_dir)
    result["metrics_out"] = str(metrics_out)
    result["csv_out"] = str(csv_out)
    result["report_out"] = str(report_out)

    flat_rows: list[dict[str, Any]] = []
    for arm, arm_records in result["records"].items():
        for record in arm_records:
            row = dict(record)
            row["arm"] = arm
            flat_rows.append(row)

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_experiment_csv(flat_rows, csv_out)
    _write_experiment_report(result, report_out)
    return result


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
