# experiments/

One folder per experiment. The folder owns *everything* that defines the
experiment: config, inputs metadata, generated records, computed metrics,
report. Shared logic lives in [`eval_core/`](../eval_core/) — no orchestration
code belongs here.

## Layout

```
experiments/
├── README.md                 ← this file
├── _template/                ← copy this to start a new experiment
│   ├── config.yaml
│   ├── README.md
│   └── .gitignore            ← results/ ignored by default
└── <NN_short_name>/
    ├── config.yaml           ← arms, dataset, parent, models, prompt overrides
    ├── README.md             ← WHAT/WHY of this experiment
    ├── .gitignore            ← override here if you want to commit results/
    ├── results/              ← raw inference records
    │   ├── <arm>/A<stt>.json
    │   └── multimodel/<arm>__<model_safe>/A<stt>.json
    ├── metrics/              ← academic_metrics.json + .csv + gold_normalized
    ├── report/               ← academic_report.md
    └── prompts_override/     ← optional per-experiment prompt overrides
```

The standard layout is encoded in [`eval_core/paths.py`](../eval_core/paths.py).
Don't write to other locations — downstream tools look here.

## Naming

`NN_short_name/` — two-digit prefix for ordering + a short
descriptive name. Date goes in `config.yaml`. Example:

```
01_initial_eval/
02_logic_decomposition/
03_multilaw_phase1/
```

## Creating a new experiment

```powershell
Copy-Item -Recurse experiments/_template experiments/03_my_idea
# Edit experiments/03_my_idea/config.yaml
# Edit experiments/03_my_idea/README.md (WHAT + WHY + expected outcome)

python -m eval_core all experiments/03_my_idea
```

`all` runs `run` (inference for every `mode: run` arm) + `multimodel`
(if configured) + `metrics`. Use the individual subcommands when you
want to step through the lifecycle.

## Inheritance

Frozen experiments can hand their records down. Declare the parent and
mark inherited arms in `config.yaml`:

```yaml
parent: 01_initial_eval
arms:
  graphrag:              { mode: inherit }
  llm_only:              { mode: inherit }
  logic_lm_no_retrieval: { mode: inherit }
  logic_lm_ontology:     { mode: inherit }
  logic_lm_graphrag:     { mode: inherit }
  my_new_arm:            { mode: run, model: gpt-4o-mini }
```

`eval_core.metrics_for_experiment` reads the inherited arms from the
parent folder without re-running inference. Reports include a
`records_source` map so the provenance is visible.

Cycle detection guards the parent chain (depth ≤ 10).

## Git policy

The repo's root `.gitignore` ignores `experiments/*/results/` by default.
A frozen baseline that wants to share its records (so others can inherit
them) adds an exception in its own `.gitignore` — see
[`01_initial_eval/.gitignore`](01_initial_eval/.gitignore).

`metrics/` and `report/` are always tracked: they're small, auditable,
and they're what the experiment claims.

## Headline metric discipline

- Citation recall / precision / F1 / display rate come from the canonical
  [registry](../data/legal_sources.yaml) parsed by
  [`src.citations`](../src/citations.py). No per-experiment authority
  hardcoding.
- BERTScore runs fail-soft (skips if dep / model missing); citation
  metrics never silently fail — gold validation fails-hard.
- `eval_core.judge` is fail-closed by design. Judge metrics are
  intentionally outside the main flow.
