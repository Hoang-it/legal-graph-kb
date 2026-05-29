# 01 — Initial evaluation (R1 + R2)

First full evaluation of the project. Establishes baseline numbers that
future experiments will be compared against (and optionally inherit from).

## What

Two studies bundled into one experiment folder because they share the
same dataset (200 BHXH questions) and gold citations:

- **R1 (single-model)**: 5 arms × 200 questions × gpt-4o-mini.
- **R2 (multi-model)**: 2 logic-LM arms × {gpt-4.1, gpt-4o, gpt-5-mini} × 200 questions.

## Why

Establish honest baselines under a single registry-based citation parser
and a fail-soft BERTScore. Prior reports used per-script citation logic
and a judge-based metric that conflated several concerns; those have
been removed from the main flow (see `eval_core.judge`).

## Setup

- Dataset: `data/eval/questions_200.json`, n=200.
- Arms: every arm with `mode: run` (see `config.yaml`).
- Prompts: canonical `prompts/` tree (no override).
- Models: `gpt-4o-mini` for R1; `gpt-4.1 / gpt-4o / gpt-5-mini` for R2.

## How to reproduce

The R1 and R2 records are committed in `results/` so the metric
pipeline can be re-run without any LLM calls:

```powershell
# Just recompute metrics + report from the committed records
python -m eval_core metrics experiments/01_initial_eval

# Or re-run inference from scratch (~$$ in API calls)
python -m eval_core run         experiments/01_initial_eval --force
python -m eval_core multimodel  experiments/01_initial_eval --force
python -m eval_core metrics     experiments/01_initial_eval
```

## Result summary

See:
- `metrics/academic_metrics.json`
- `metrics/academic_metrics.csv`
- `report/academic_report.md`

The defensible claims that survived four audit rounds are documented in
the project skill (`.claude/skills/legal-kg-logic-extraction/SKILL.md`)
and in `docs/plans/v5_general_retrieval.md`. The headline finding —
`llm_only` beats `graphrag` pairwise — is what motivates the v5
retrieval plan.

## Inheritance

Future experiments can do:

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

…and the metric pipeline pulls the five baseline arms' records from
this folder without re-running them.
