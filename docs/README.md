# Documentation

Single source of truth for project documentation. **All** project-level
docs live here (or in `docs/plans/` for design plans). The repo root
holds only the README stub + `LICENSE`; package directories hold code,
not docs.

Per-experiment write-ups and the experiment starter template
([`experiments/_template/README.md`](../experiments/_template/README.md))
stay co-located because they're part of the experiment artifact, not
project-level docs.

## Map

### Getting started
- [Quickstart](quickstart.md) — install, env vars, first inference run.
- [Architecture](architecture.md) — pipeline diagram (B1–B7), what each
  top-level package does, the prompt loader, the Experiment model.
- [Neo4j setup](neo4j-setup.md) — Desktop install + APOC + vector index.

### Reference
- [eval_core](eval_core.md) — module-by-module guide to the shared
  evaluation infrastructure (Experiment class, CLI, metrics, report).
- [experiments](experiments.md) — folder layout, naming, inheritance,
  git policy.

### Plans
- [`plans/`](plans/) — design plans for upcoming work (one file per plan).

### Meta
- [Changelog](changelog.md)
- [Contributing](contributing.md)
- [Code of conduct](code-of-conduct.md)

## House rules

- Add new project-level docs **here**, not in package folders.
- Cross-link with relative paths (e.g. `[eval_core](eval_core.md)`).
- Per-experiment READMEs and the prompt files under `prompts/` are not
  docs — leave them where they are.
- The auto-generated `data/graph/processed/extraction_summary.md` is a build
  artifact of `offline.merge_normalize`, not a doc.
