# Colab Notebooks - Legal KB Experiment

Active notebooks are scoped to inference plus deterministic academic metrics.
Judge-based metrics are not part of the main experiment workflow.

## Active Flow

| Notebook | Purpose |
|---|---|
| `01_colab_setup.ipynb` | Install dependencies and configure Colab secrets. |
| `02_colab_inference.ipynb` | Run experiment inference and write `data/eval/results/{arm}/A{stt}.json`. |
| `03_colab_metrics_report.ipynb` | Validate gold citations and compute academic metrics. |

## Metrics

The academic workflow uses:

- `citation_recall`, `citation_precision`, `citation_f1`
- `citation_display_rate`
- `latency_s`
- `bertscore_p`, `bertscore_r`, `bertscore_f1`
- `prolog_first_try_solution_rate`, `repair_invoked_rate`, `repair_success_rate`

Outputs:

- `data/eval/academic_metrics.json`
- `data/eval/academic_metrics.csv`
- `reports/academic_report.md`

## Commands

```bash
python -m experiments.run_inference --arms main --n 200
python -m experiments.validate_gold_citations
python -m experiments.compute_academic_metrics
```

For optional multimodel result folders, use the same academic metric engine with
an explicit results root and output names:

```bash
python -m experiments.compute_academic_metrics \
  --results-root data/eval/multimodel/results \
  --arms all \
  --metrics-out data/eval/multimodel/academic_metrics.json \
  --csv-out data/eval/multimodel/academic_metrics.csv \
  --report-out reports/multimodel_academic_report.md
```

## Removed Legacy Flow

The old non-academic metric notebooks and reports were removed from the active system.
`experiments.compute_judge_metrics` remains only as a fail-closed optional
placeholder until a separate judge rubric is designed.
