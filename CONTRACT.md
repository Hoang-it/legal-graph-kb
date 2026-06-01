# Experiment contract

The single spec that the **producer** repo (`legal-graph-kb`, which *generates*
experiments) and the **consumer** repo (`experiments-repo`, which *compares*
them via `expkit`) both honour. The machine-readable form lives in
[`experiment_contract.py`](experiment_contract.py) — a dependency-light
(stdlib + PyYAML) module that **both repos ship byte-identical**. If you change
one, copy it to the other.

> Why a shared file instead of two implementations: a folder produced here must
> be loadable + comparable there without guesswork. One module = one definition
> of "what a valid experiment is", used by both the generator and the leaderboard.

## Folder shape

```
experiments/<NN>_<slug>/
├── config.yaml                      # metadata (see below)              [required]
├── metrics/academic_metrics.json    # the comparable metrics            [required]
├── results/<arm>/A<stt>.json        # raw per-record outputs            [Tier-1 inputs]
├── report/                          # human-readable reports            [optional]
└── README.md                        # WHAT / WHY                        [optional]
```

- `<NN>` is the experiment number (zero-padded), used to order + to derive the
  default retrieval recompute module.
- The folder is **comparable** as soon as `metrics/academic_metrics.json` exists
  and a `family` can be resolved. Everything else is optional for comparison.

## `config.yaml`

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | Human-readable title. |
| `description` | yes | Hypothesis / what's new vs parent. |
| `date` | yes | ISO date. |
| `family` | yes (new) | `qa` or `retrieval`. Legacy folders may omit it → family is inferred from the metrics shape; new experiments must set it. |
| `recompute` | optional | How to regenerate `metrics/` offline (Tier-2). Defaults per family. |
| `dataset` | qa | `{ questions, n }`. |
| `parent` | optional | Sibling folder name for arm inheritance. |
| `arms` / `multimodel` | qa | Drives `eval_core`. Retrieval leaves `arms: {}`. |

`recompute` accepts:

```yaml
recompute: eval_core                 # python -m eval_core metrics <exp>  (both families)
recompute: some.metrics_module       # python -m some.metrics_module      (custom override)
recompute: { module: some.metrics_module }
recompute: { command: [python, -m, some.metrics_module, --full] }
```

Omit it and the family default applies — **both families recompute via
`eval_core`**. The CLI dispatches on `family`: `qa` → the arm runners,
`retrieval` → the config-driven engine `eval_core.retrieval_metrics`.

## Two families

| Family | Experiments | `metrics/academic_metrics.json` shape | Producer |
|---|---|---|---|
| **qa** | 01–04 | `aggregates[arm].macro` (+ `.prolog`) — citation R/P/F1, display rate, BERTScore, latency, Prolog rates | generic `eval_core` (config-only) |
| **retrieval** | 06–14 | `overall_macro[arm]` + `stratified[arm][stratum]` + `Ks` — recall@k, precision@k, r_precision, mrr, ndcg@k | Tier-1: your online retrieval script → `results/<arm>/A*.json`; Tier-2: generic `eval_core` (reads the `retrieval:` config block) |

The consumer detects the family from `config.family` first, then from these JSON
keys (so the 14 pre-existing folders keep working without edits).

## Reproducibility tiers

| Tier | Artifact | Produced by | Offline? | Available in |
|---|---|---|---|---|
| 1 | `results/` | qa: `eval_core run`; retrieval: your online retrieval script (Neo4j + embeddings + LLM) | ❌ | producer only |
| 2 | `metrics/` | `eval_core metrics` from `results/` (both families) | ✅ | both repos |
| 3 | leaderboard | `expkit` from `metrics/` | ✅ | both repos |

The consumer repo deliberately has **no Tier-1 code** (no inference). It commits
`results/` so Tier-2 recompute stays reproducible offline, and reads `metrics/`
for Tier-3 comparison.

## Workflow: add an experiment and compare it

1. **Create** in the producer: `Copy-Item -Recurse experiments/_template
   experiments/NN_slug`, set `family` + `recompute`, write WHAT/WHY.
2. **Generate data** (Tier-1 → Tier-2) per family (see the template README).
3. **Validate**: `python -m experiment_contract validate experiments/NN_slug`
   (must report `OK` / comparable).
4. **Copy the folder** into the consumer at `experiments/NN_slug/`. Offline
   recompute (both families) uses the consumer's own `eval_core` — no
   per-experiment metrics script to copy.
5. **Compare**: in the consumer, `python -m expkit leaderboard --all` —
   the new folder is auto-discovered and ranked against every prior experiment.
   `python -m expkit validate --all` checks every folder against this contract.

Because the leaderboard ranks by a single metric on one scale, a new experiment
is directly comparable to all previous ones in its family the moment its folder
is dropped in.

## Guarantees a folder must meet to be comparable

- `metrics/academic_metrics.json` exists and is valid JSON.
- A `family` resolves (explicit `config.family`, or an inferable metrics shape).
- If `config.family` is set, it agrees with the metrics shape.

`validate_experiment` (and the `validate` CLI in both `experiment_contract` and
`expkit`) enforce exactly these, reporting hard **errors** (not comparable) vs
**warnings** (comparable, but tighten this).
