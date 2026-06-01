# Experiment template

A skeleton conforming to the shared experiment contract (see
[`../../CONTRACT.md`](../../CONTRACT.md)). To create a new experiment:

```powershell
Copy-Item -Recurse experiments/_template experiments/NN_your_short_name
# Edit config.yaml — set name, description, date, FAMILY (qa|retrieval), recompute.
# Edit this README — replace it with your WHAT/WHY.
```

Then generate its data. **Which path depends on `family`:**

### family: qa  (generic, config-only)

```powershell
python -m eval_core run     experiments/NN_your_short_name   # Tier-1: results/  (online)
python -m eval_core metrics experiments/NN_your_short_name   # Tier-2: metrics/  (offline)
# OR all-in-one:
python -m eval_core all     experiments/NN_your_short_name
```

### family: retrieval  (config-driven metrics via eval_core)

```powershell
# Tier-1 (online): write an experiment-local producer, usually:
#   experiments/NN_your_short_name/run_retrieval.py
# It writes results/<arm>/A*.json
#   (Neo4j + embeddings + LLM), with ranked ids at retrieval_only.final_article_ids.
#   Declare the arms + K cutoffs in config.yaml's `retrieval:` block.
python -m eval_core metrics experiments/NN_your_short_name   # Tier-2: metrics/  (offline)
```

Do not create a one-off `scripts/exp<NN>_*.py` for a retrieval experiment.
`scripts/` is for reusable repo-level utilities; experiment-specific Tier-1
producers stay with the experiment folder so the folder is self-contained.

Keep `recompute: eval_core` (the default) so the offline Tier-2 step is
discoverable by `expkit --recompute` in the experiments repo — the generic
`eval_core.retrieval_metrics` engine reads the `retrieval:` block.

### Validate before comparing / copying over

```powershell
python -m experiment_contract validate experiments/NN_your_short_name
```

A folder is **comparable** once it has a valid `metrics/academic_metrics.json`
and a resolvable `family`. Copy the folder into the experiments repo and it is
auto-discovered into the leaderboard; offline recompute there uses the consumer's
own `eval_core` (both families) — nothing else to copy. See [`../../CONTRACT.md`](../../CONTRACT.md).

## Required sections to fill in this README

### What
1–3 sentences: the question this experiment answers.

### Why
What previous result motivated this experiment? Reference the parent
experiment by folder name and report file when applicable.

### Setup
- Family (qa | retrieval) and how records are produced.
- Arms run vs inherited (matches `config.yaml`).
- Dataset / N.
- Prompt overrides, if any.
- Models used (single + multimodel).

### Success criterion & cost (pre-registered — no result prediction)
Before running, write down only: (a) the **objective success bar** decided in
advance — e.g. "beat the prior best on `in_corpus` by ≥ X" — and what you'll
conclude for each outcome; and (b) the **cost estimate** (API $ / tokens / time).
Do **not** predict the result numbers or a % improvement, and do not anchor on
the current system or any prior method's conclusion. This is the anti-post-hoc
safeguard — see Rule 5 in the `legal-kg-logic-extraction` skill.

### Result summary
Filled in after the run. Link `metrics/academic_metrics.json` and
`report/academic_report.md`. State which arm won + by how much + p-value
when applicable.
