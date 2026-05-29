"""Standard on-disk layout of an experiment folder.

Every experiment under ``experiments/`` follows this layout. The constants
here are the single source of truth — never hardcode these names elsewhere.

::

    experiments/<NN_name>/
    ├── config.yaml
    ├── README.md
    ├── results/
    │   ├── <arm>/A<stt>.json                              # single-model arms
    │   └── multimodel/<arm>__<model_safe>/A<stt>.json     # multi-model combos
    ├── metrics/
    │   ├── academic_metrics.json
    │   ├── academic_metrics.csv
    │   └── gold_citations_normalized.json
    ├── report/
    │   └── academic_report.md
    └── prompts_override/      # optional, points LEGAL_KG_PROMPTS_DIR
"""

from __future__ import annotations

from pathlib import Path

CONFIG_FILENAME = "config.yaml"
RESULTS_DIRNAME = "results"
METRICS_DIRNAME = "metrics"
REPORT_DIRNAME = "report"
PROMPTS_OVERRIDE_DIRNAME = "prompts_override"

# Subdirectory for multimodel combos within results/
MULTIMODEL_SUBDIR = "multimodel"

# Files inside metrics/
METRICS_JSON = "academic_metrics.json"
METRICS_CSV = "academic_metrics.csv"
GOLD_NORMALIZED_JSON = "gold_citations_normalized.json"
GOLD_ERRORS_CSV = "gold_citation_validation_errors.csv"

# Files inside report/
REPORT_MD = "academic_report.md"


def results_dir(exp_path: Path) -> Path:
    return exp_path / RESULTS_DIRNAME


def metrics_dir(exp_path: Path) -> Path:
    return exp_path / METRICS_DIRNAME


def report_dir(exp_path: Path) -> Path:
    return exp_path / REPORT_DIRNAME


def arm_results_dir(exp_path: Path, arm: str) -> Path:
    return results_dir(exp_path) / arm


def multimodel_combo_dir(exp_path: Path, arm: str, model_safe: str) -> Path:
    return results_dir(exp_path) / MULTIMODEL_SUBDIR / f"{arm}__{model_safe}"


def metrics_json_path(exp_path: Path) -> Path:
    return metrics_dir(exp_path) / METRICS_JSON


def metrics_csv_path(exp_path: Path) -> Path:
    return metrics_dir(exp_path) / METRICS_CSV


def gold_normalized_path(exp_path: Path) -> Path:
    return metrics_dir(exp_path) / GOLD_NORMALIZED_JSON


def report_md_path(exp_path: Path) -> Path:
    return report_dir(exp_path) / REPORT_MD


def prompts_override_dir(exp_path: Path) -> Path:
    return exp_path / PROMPTS_OVERRIDE_DIRNAME
