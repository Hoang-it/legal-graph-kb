"""Shared evaluation infrastructure for all experiments.

This package consolidates:

- arms        Arm definitions + CLI parsing.
- inference   Single-model batch inference orchestrator.
- multimodel  Multi-model (arm × model) batch inference orchestrator.
- rerender    Backfill plain_answer on legacy logic-LM records.
- metrics     Deterministic academic-metric engine (per-record computation).
- report      CSV + Markdown report writers.
- gold        Gold citation validator.
- judge       Judge-metric placeholder (fail-closed by design).
- text_normalize  IRAC → prose helper used by BERTScore fairness.
- runners     Multi-arm metric loader (groups records by arm, calls metrics).
- experiment  ``Experiment`` class — encapsulates a single experiment folder,
              including inheritance from a parent experiment.
- paths       Standardized output-file names within an experiment folder.
- cli         ``python -m eval_core <cmd>`` entry point.
"""
