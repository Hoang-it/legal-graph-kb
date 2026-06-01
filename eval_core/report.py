"""CSV + Markdown report writers for academic metrics.

This module is the I/O layer on top of :mod:`eval_core.metrics`. It receives
the dict returned by :func:`eval_core.metrics.compute_academic_metrics` (or
the multi-arm result from :mod:`eval_core.runners`) and writes the standard
output trio: ``academic_metrics.json``, ``academic_metrics.csv``,
``academic_report.md``.

Two flavours of writer:

- Single-arm (``write_academic_metrics_outputs``,
  ``compute_and_write_academic_metrics``) — used by the direct CLI
  ``python -m eval_core.metrics``.
- Multi-arm (``write_grouped_academic_metrics_outputs``,
  ``_write_experiment_csv``, ``_write_experiment_report``) — used by the
  experiment-level loader in :mod:`eval_core.runners`.

The CSV / Markdown layouts intentionally differ because the multi-arm
report compares arms side-by-side, while the single-arm one is flat.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from eval_core.metrics import (
    DEFAULT_OUTPUT_DIR,
    compute_academic_metrics,
)
from src.citations import DEFAULT_REGISTRY_PATH


def _fmt(v: Any) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ---------------------------------------------------------------------------
# Single-arm writers
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "record_index",
        "record_id",
        "citation_recall",
        "citation_precision",
        "citation_f1",
        "citation_display_rate",
        "bertscore_p",
        "bertscore_r",
        "bertscore_f1",
        "rouge1",
        "rouge2",
        "rougeL",
        "bleu",
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
                    "record_index": r["record_index"],
                    "record_id": r.get("record_id", ""),
                    "citation_recall": r["citation"]["citation_recall"],
                    "citation_precision": r["citation"]["citation_precision"],
                    "citation_f1": r["citation"]["citation_f1"],
                    "citation_display_rate": r["citation"]["citation_display"][
                        "citation_display_rate"
                    ],
                    "bertscore_p": r.get("bertscore", {}).get("bertscore_p"),
                    "bertscore_r": r.get("bertscore", {}).get("bertscore_r"),
                    "bertscore_f1": r.get("bertscore", {}).get("bertscore_f1"),
                    "rouge1": r.get("text_overlap", {}).get("rouge1"),
                    "rouge2": r.get("text_overlap", {}).get("rouge2"),
                    "rougeL": r.get("text_overlap", {}).get("rougeL"),
                    "bleu": r.get("text_overlap", {}).get("bleu"),
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


def _write_report(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    agg = result["aggregate"]
    macro = agg["macro"]
    micro = agg["micro"]
    pr = agg["prolog"]
    errors = agg["error_counts"]
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
        "| n | citation_recall | citation_precision | citation_f1 | citation_display_rate | bertscore_f1 | latency_s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            [
                str(agg["n_records"]),
                _fmt(macro["citation_recall"]),
                _fmt(macro["citation_precision"]),
                _fmt(macro["citation_f1"]),
                _fmt(macro["citation_display_rate"]),
                _fmt(macro["bertscore_f1"]),
                _fmt(macro["latency_s"]),
            ]
        )
        + " |",
        "",
        "## Text-overlap Macro Metrics (ROUGE / BLEU)",
        "",
        "| rouge1 | rouge2 | rougeL | bleu |",
        "|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            [
                _fmt(macro.get("rouge1")),
                _fmt(macro.get("rouge2")),
                _fmt(macro.get("rougeL")),
                _fmt(macro.get("bleu")),
            ]
        )
        + " |",
        "",
        "## Citation Micro Metrics",
        "",
        "| recall | precision | display_rate |",
        "|---:|---:|---:|",
        f"| {_fmt(micro['citation_recall'])} "
        f"(sum={micro['recall_num']}/{micro['recall_denom']}) | "
        f"{_fmt(micro['citation_precision'])} "
        f"(sum={micro['precision_num']}/{micro['precision_denom']}) | "
        f"{_fmt(micro['citation_display_rate'])} "
        f"(sum={micro['display_num']}/{micro['display_denom']}) |",
        "",
        "## Prolog Metrics",
        "",
        "| n_prolog | first_try_solution | repair_invoked | repair_success |",
        "|---:|---:|---:|---:|",
        f"| {pr['n_prolog_records']} | "
        f"{_fmt(pr['prolog_first_try_solution_rate'])} | "
        f"{_fmt(pr['repair_invoked_rate'])} | "
        f"{_fmt(pr['repair_success_rate'])} "
        f"(sum={pr.get('repair_success_num', 0)}/{pr.get('repair_success_denom', 0)}) |",
        "",
        "## BERTScore Status",
        "",
        "```json",
        json.dumps(result["bertscore_metadata"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Text-overlap Status (ROUGE / BLEU)",
        "",
        "```json",
        json.dumps(result.get("text_overlap_metadata", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Error Counts",
        "",
        "| pred_citation_parse_errors | records_with_no_pred_citations |",
        "|---:|---:|",
        f"| {errors['pred_citation_parse_errors']} | "
        f"{errors['records_with_no_pred_citations']} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_academic_metrics_outputs(
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

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(result["records"], csv_out)
    _write_report(result, report_out)
    return result


def compute_and_write_academic_metrics(
    records: list[dict[str, Any]],
    registry=None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    metadata: dict[str, Any] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    metrics_out: Path | None = None,
    csv_out: Path | None = None,
    report_out: Path | None = None,
) -> dict[str, Any]:
    result = compute_academic_metrics(
        records=records,
        registry=registry,
        registry_path=registry_path,
        metadata=metadata,
    )
    return write_academic_metrics_outputs(
        result,
        output_dir=output_dir,
        metrics_out=metrics_out,
        csv_out=csv_out,
        report_out=report_out,
    )


# ---------------------------------------------------------------------------
# Multi-arm writers (used by eval_core.runners)
# ---------------------------------------------------------------------------


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
        "rouge1",
        "rouge2",
        "rougeL",
        "bleu",
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
                    "rouge1": r.get("text_overlap", {}).get("rouge1"),
                    "rouge2": r.get("text_overlap", {}).get("rouge2"),
                    "rougeL": r.get("text_overlap", {}).get("rougeL"),
                    "bleu": r.get("text_overlap", {}).get("bleu"),
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
            "## Text-overlap Macro Metrics (ROUGE / BLEU)",
            "",
            "| Arm | n | rouge1 | rouge2 | rougeL | bleu |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm, agg in result["aggregates"].items():
        macro = agg["macro"]
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    str(agg["n_records"]),
                    _fmt(macro.get("rouge1")),
                    _fmt(macro.get("rouge2")),
                    _fmt(macro.get("rougeL")),
                    _fmt(macro.get("bleu")),
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
            "## Text-overlap Status (ROUGE / BLEU)",
            "",
            "```json",
            json.dumps(result.get("text_overlap_metadata", {}), ensure_ascii=False, indent=2),
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
