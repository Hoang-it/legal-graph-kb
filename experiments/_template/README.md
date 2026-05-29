# Experiment template

This is a skeleton. To create a new experiment:

```powershell
Copy-Item -Recurse experiments/_template experiments/NN_your_short_name
# Edit config.yaml — fill in name, description, date, arms.
# Edit this README — replace it with your WHAT/WHY.
```

Then drive it through `eval_core`:

```powershell
python -m eval_core run     experiments/NN_your_short_name
python -m eval_core metrics experiments/NN_your_short_name
# OR all-in-one:
python -m eval_core all     experiments/NN_your_short_name
```

## Required sections to fill in this README

### What
1–3 sentences: the question this experiment answers.

### Why
What previous result motivated this experiment? Reference the parent
experiment by folder name and report file when applicable.

### Setup
- Arms run vs inherited (matches `config.yaml`).
- Dataset / N.
- Prompt overrides, if any.
- Models used (single + multimodel).

### Expected outcome
Before running, write down what you predict and what threshold would make
you change conclusion. This protects against post-hoc rationalization.

### Result summary
Filled in after the run. Link `metrics/academic_metrics.json` and
`report/academic_report.md`. State which arm won + by how much + p-value
when applicable.
