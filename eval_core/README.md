# eval_core

Shared infrastructure that every experiment under [`experiments/`](../experiments)
calls into. The two halves:

1. **Experiment model** — what an experiment IS (a folder with `config.yaml`
   and the standard layout) and how to read its records (with inheritance
   from a parent experiment).
2. **Pipeline functions** — inference orchestration, gold validation,
   metric computation, report generation.

Plus a thin CLI (`python -m eval_core <cmd> <experiment_path>`) that wires
them together.

## Module layout

| Module | Role |
|---|---|
| [`paths.py`](paths.py) | Standard on-disk layout of an experiment folder. Single source of truth for `results/`, `metrics/`, `report/`, file names. |
| [`experiment.py`](experiment.py) | `Experiment` class. Loads `config.yaml`, resolves the parent chain lazily, returns inheritance-aware records via `records_for_arm()`. Cycle detection. |
| [`arms.py`](arms.py) | `ALL_ARMS`, `MAIN_EXPERIMENT_ARMS`, plus `parse_run_arms` / `parse_metrics_arms` for CLI selection. |
| [`inference.py`](inference.py) | `run_experiment(experiment, arms=None, force, verbose)`. Per-arm runners that wrap `runtime.rag_query`, `runtime.llm_only`, `runtime.logic_lm_pipelines.*`. Writes records to `experiment.results_dir / <arm>`. |
| [`multimodel.py`](multimodel.py) | `run_experiment_multimodel(experiment, arms=None, models=None, ...)`. Same shape as `inference.py` but iterates the `multimodel:` matrix from config. Writes to `results/multimodel/<arm>__<model_safe>/`. |
| [`gold.py`](gold.py) | Strict parse of `gold_citations_raw` against the citation registry. Writes `gold_citations_normalized.json` + `gold_citation_validation_errors.csv`. Fail-hard. |
| [`metrics.py`](metrics.py) | Deterministic metric engine (pure-computational). Reads records with `gold_articles` attached, returns the metric dict. No I/O for reports. |
| [`report.py`](report.py) | CSV + Markdown writers. Single-arm and multi-arm flavours. |
| [`runners.py`](runners.py) | Multi-arm loader: pulls records (own + inherited + multimodel combos), calls gold + metrics + report, writes outputs to the experiment folder. |
| [`rerender.py`](rerender.py) | Backfill `plain_answer` on legacy logic-LM records of an experiment. Walks `results/` for `logic_lm*` directories. |
| [`text_normalize.py`](text_normalize.py) | IRAC → prose helper used to keep BERTScore comparable between IRAC and prose arms. |
| [`judge.py`](judge.py) | Fail-closed placeholder. Judge metrics are intentionally outside the main flow. |
| [`cli.py`](cli.py) + [`__main__.py`](__main__.py) | Unified subcommand CLI: `run`, `multimodel`, `metrics`, `all`. |

## CLI

Every command takes an experiment path as the first positional argument:

```powershell
python -m eval_core run        experiments/<NN_name> [--arms ...] [--force] [--verbose]
python -m eval_core multimodel experiments/<NN_name> [--arms ...] [--models ...] [--force]
python -m eval_core metrics    experiments/<NN_name> [--arms ...] [--no-multimodel]
python -m eval_core all        experiments/<NN_name> [--force] [--verbose]
```

`all` runs `run` + `multimodel` (if configured) + `metrics` in sequence —
the typical end-to-end driver for a fresh experiment.

When the experiment's `config.yaml` sets `prompts_override_dir: <rel>`,
every command exports `LEGAL_KG_PROMPTS_DIR=<exp_path>/<rel>` for the
duration of the run. The prompt loader in `src.prompts` checks the
override first and falls back to the canonical `prompts/` tree per file.

## Inheritance

An arm with `mode: inherit` reads its records from the parent
experiment's `results/<arm>/`, recursively. The metric step still groups
by arm name, so reports look identical whether records were generated
here or inherited.

Guards:

- The parent chain is resolved lazily.
- `Experiment.validate()` walks every `mode: inherit` arm before any
  expensive operation so a misconfigured chain fails fast.
- Cycles are detected via a visited set on `records_for_arm`.
- Depth is capped at 10 levels.

Multimodel combos do not inherit today — they always live in the
experiment that owns them. This keeps the design honest: a combo is
"this specific arm × this specific model on this dataset," which is
rarely a 1:1 match across experiments.

## Tests

[`tests/test_experiment.py`](../tests/test_experiment.py) covers config
parsing, inheritance, cycle detection, validation, and standard paths.
[`tests/test_evaluation_sample_metrics.py`](../tests/test_evaluation_sample_metrics.py)
exercises the metric engine end-to-end on the [`samples/`](samples/)
fixture.
