"""Unified CLI for experiment lifecycle.

::

    python -m eval_core run     <experiment_path> [--arms ...] [--force] [--verbose]
    python -m eval_core multimodel <experiment_path> [--arms ...] [--models ...] [--force] [--verbose]
    python -m eval_core metrics <experiment_path> [--arms ...] [--no-multimodel]
    python -m eval_core all     <experiment_path> [--force] [--verbose]

Each subcommand operates on a single experiment folder. The path can be
relative (e.g. ``experiments/01_my_experiment``) or absolute. The CLI lives
in this single module so adding a new command means editing one file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval_core.experiment import Experiment


def _load(path: Path) -> Experiment:
    try:
        return Experiment.from_path(path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _is_retrieval(experiment: Experiment) -> bool:
    """A retrieval-family experiment: declared ``family: retrieval`` or carrying
    a ``retrieval:`` config block. Its metrics come from
    :mod:`eval_core.retrieval_metrics`, not the qa arm runners."""
    cfg = experiment.config or {}
    fam = str(cfg.get("family") or "").strip().lower()
    if fam == "retrieval":
        return True
    if fam in ("qa", "qa_citation"):
        return False
    return bool(cfg.get("retrieval"))


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    experiment = _load(args.experiment)
    if _is_retrieval(experiment):
        print(
            "NOTE: retrieval-family Tier-1 inference runs outside eval_core "
            "(its own online producer). eval_core only owns the offline metrics "
            f"step — run `python -m eval_core metrics {args.experiment}` once "
            "results/<arm>/A*.json exist.",
            file=sys.stderr,
        )
        return 0

    from eval_core.inference import run_experiment

    arms = [a.strip() for a in args.arms.split(",") if a.strip()] if args.arms else None
    run_experiment(experiment, arms=arms, force=args.force, verbose=args.verbose)
    return 0


def cmd_multimodel(args: argparse.Namespace) -> int:
    from eval_core.multimodel import run_experiment_multimodel

    experiment = _load(args.experiment)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()] if args.arms else None
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    run_experiment_multimodel(
        experiment, arms=arms, models=models, force=args.force, verbose=args.verbose,
    )
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    experiment = _load(args.experiment)
    arms_filter = (
        [a.strip() for a in args.arms.split(",") if a.strip()] if args.arms else None
    )

    if _is_retrieval(experiment):
        from eval_core.retrieval_metrics import compute_retrieval_metrics

        result = compute_retrieval_metrics(
            experiment, arms_filter=arms_filter, force_full=args.full,
        )
        print(
            f"OK: wrote {result['metrics_out']}, {result['csv_out']}, {result['report_out']} "
            f"for {len(result['records'])} arms (retrieval, n={result['n_scored']})."
        )
        return 0

    from eval_core.runners import compute_metrics_for_experiment

    result = compute_metrics_for_experiment(
        experiment,
        arms_filter=arms_filter,
        include_multimodel=not args.no_multimodel,
    )
    print(
        f"OK: wrote {result['metrics_out']}, {result['csv_out']}, {result['report_out']} "
        f"for {len(result['records'])} arms."
    )
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    experiment = _load(args.experiment)
    if _is_retrieval(experiment):
        from eval_core.retrieval_metrics import compute_retrieval_metrics

        print(
            "NOTE: retrieval-family Tier-1 inference runs outside eval_core; "
            "computing offline metrics only.",
            file=sys.stderr,
        )
        result = compute_retrieval_metrics(experiment)
        print(
            f"OK: wrote {result['metrics_out']}, {result['csv_out']}, {result['report_out']} "
            f"for {len(result['records'])} arms (retrieval, n={result['n_scored']})."
        )
        return 0

    from eval_core.inference import run_experiment
    from eval_core.multimodel import run_experiment_multimodel
    from eval_core.runners import compute_metrics_for_experiment

    run_experiment(experiment, force=args.force, verbose=args.verbose)
    if experiment.multimodel:
        run_experiment_multimodel(experiment, force=args.force, verbose=args.verbose)
    result = compute_metrics_for_experiment(experiment)
    print(
        f"OK: wrote {result['metrics_out']}, {result['csv_out']}, {result['report_out']} "
        f"for {len(result['records'])} arms."
    )
    return 0


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval_core",
        description="Experiment lifecycle commands.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run inference for mode=run arms.")
    run_p.add_argument("experiment", type=Path)
    run_p.add_argument("--arms", type=str, default=None)
    run_p.add_argument("--force", action="store_true")
    run_p.add_argument("--verbose", action="store_true")
    run_p.set_defaults(func=cmd_run)

    mm_p = sub.add_parser("multimodel", help="Run the multimodel matrix.")
    mm_p.add_argument("experiment", type=Path)
    mm_p.add_argument("--arms", type=str, default=None)
    mm_p.add_argument("--models", type=str, default=None)
    mm_p.add_argument("--force", action="store_true")
    mm_p.add_argument("--verbose", action="store_true")
    mm_p.set_defaults(func=cmd_multimodel)

    met_p = sub.add_parser("metrics", help="Compute academic metrics.")
    met_p.add_argument("experiment", type=Path)
    met_p.add_argument("--arms", type=str, default=None)
    met_p.add_argument(
        "--no-multimodel",
        action="store_true",
        help="Skip multimodel combos under results/multimodel/.",
    )
    met_p.add_argument(
        "--full",
        action="store_true",
        help="Retrieval family: ignore retrieval.pilot_subset and score every record on disk.",
    )
    met_p.set_defaults(func=cmd_metrics)

    all_p = sub.add_parser("all", help="run + multimodel (if configured) + metrics.")
    all_p.add_argument("experiment", type=Path)
    all_p.add_argument("--force", action="store_true")
    all_p.add_argument("--verbose", action="store_true")
    all_p.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
