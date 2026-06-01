"""Experiment contract — the single shared spec for an experiment folder.

This module is the ONE source of truth shared by:

- the **producer** repo (``legal-graph-kb``) that *generates* experiments
  (``eval_core`` for the qa family, ``scripts/exp<NN>_*`` for the retrieval
  family), and
- the **consumer** repo (``experiments-repo``) that *compares* them via
  ``expkit``.

Both repos ship a byte-identical copy of this file (it is intentionally
dependency-light — stdlib + optional PyYAML — so the consumer's ``expkit`` can
import it without pulling the producer's inference stack). A folder produced
under this contract is therefore guaranteed loadable + comparable by the
consumer, and ``validate_experiment`` lets either side check a folder before
trusting/comparing it.

A valid experiment folder ``experiments/<NN>_<slug>/``::

    config.yaml                      # metadata (name, date, family, recompute, …)
    metrics/academic_metrics.json    # the comparable metrics (one of two shapes)
    results/<arm>/A<stt>.json        # raw per-record outputs (Tier-1 inputs)
    report/                          # human-readable reports (optional)
    README.md                        # WHAT / WHY (optional)

Two families. The family is taken from ``config.family`` when present
(explicit), else inferred from the metrics JSON shape (legacy folders that
pre-date this field)::

    qa         metrics has top-level ``aggregates[arm].macro`` (+ ``.prolog``)
    retrieval  metrics has ``overall_macro`` / ``stratified`` / ``Ks``

Reproducibility tiers (what code is needed to regenerate each artifact)::

    Tier 1  results/  ← producer run (Neo4j / OpenAI / GPU): NOT offline,
                        producer repo only (``eval_core run`` | ``exp<NN>_run``).
    Tier 2  metrics/  ← recompute from results/: offline, either repo
                        (see ``recompute_spec``).
    Tier 3  leaderboard ← metrics/: offline, either repo (``expkit``).

CLI (runnable from either repo's root)::

    python -m experiment_contract validate experiments/15_my_experiment
    python -m experiment_contract validate experiments/*      # shell-expanded
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # PyYAML is optional; without it config.yaml simply can't be read.
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

# --------------------------------------------------------------------------- #
# Folder layout — MUST stay in sync with eval_core/paths.py (the producer).
# A test asserts the two agree so they cannot drift.
# --------------------------------------------------------------------------- #
CONFIG_FILENAME = "config.yaml"
RESULTS_DIRNAME = "results"
METRICS_DIRNAME = "metrics"
REPORT_DIRNAME = "report"
METRICS_JSON = "academic_metrics.json"
METRICS_REL = f"{METRICS_DIRNAME}/{METRICS_JSON}"

# An experiment directory's name starts with its number: ``13_hyde_semantic``.
EXP_DIR_RE = re.compile(r"^\d+[_-]")

# --------------------------------------------------------------------------- #
# Families
# --------------------------------------------------------------------------- #
QA = "qa"
RETRIEVAL = "retrieval"
UNKNOWN = "unknown"
FAMILIES = (QA, RETRIEVAL)


def config_path(exp_path: Path | str) -> Path:
    return Path(exp_path) / CONFIG_FILENAME


def metrics_json_path(exp_path: Path | str) -> Path:
    return Path(exp_path) / METRICS_DIRNAME / METRICS_JSON


def results_dir(exp_path: Path | str) -> Path:
    return Path(exp_path) / RESULTS_DIRNAME


def is_experiment_dir(path: Path | str) -> bool:
    p = Path(path)
    return p.is_dir() and bool(EXP_DIR_RE.match(p.name))


def experiment_number(slug: str) -> int | None:
    """``13_hyde_semantic`` -> 13;  a non-numeric slug -> ``None``."""
    m = re.match(r"(\d+)", slug)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Loading helpers (tolerant; never raise on bad config)
# --------------------------------------------------------------------------- #
def load_config(exp_path: Path | str) -> dict[str, Any]:
    cp = config_path(exp_path)
    if not cp.is_file() or yaml is None:
        return {}
    try:
        data = yaml.safe_load(cp.read_text(encoding="utf-8")) or {}
    except Exception:  # pragma: no cover - malformed yaml shouldn't be fatal
        return {}
    return data if isinstance(data, dict) else {}


def load_metrics(exp_path: Path | str) -> dict[str, Any] | None:
    """Return parsed metrics JSON, or ``None`` if absent. Raises on bad JSON."""
    mp = metrics_json_path(exp_path)
    if not mp.is_file():
        return None
    return json.loads(mp.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Family detection — explicit config.family wins; else infer from metrics shape.
# --------------------------------------------------------------------------- #
def _metrics_shape_family(metrics: Any) -> str:
    if not isinstance(metrics, dict):
        return UNKNOWN
    if "aggregates" in metrics:
        return QA
    if any(k in metrics for k in ("overall_macro", "stratified", "Ks")):
        return RETRIEVAL
    return UNKNOWN


def normalize_family(value: Any) -> str | None:
    """Return a canonical family string for a config value, or ``None``."""
    if isinstance(value, str) and value.strip().lower() in FAMILIES:
        return value.strip().lower()
    return None


def detect_family(metrics: Any = None, config: Any = None) -> str:
    """Determine an experiment's family.

    ``config['family']`` (explicit, the contract going forward) wins; otherwise
    the family is inferred from the metrics JSON shape (legacy folders). Returns
    one of ``qa`` / ``retrieval`` / ``unknown``.

    ``metrics`` is accepted positionally so legacy one-arg calls
    ``detect_family(metrics_dict)`` keep working.
    """
    if isinstance(config, dict):
        fam = normalize_family(config.get("family"))
        if fam is not None:
            return fam
    return _metrics_shape_family(metrics)


# --------------------------------------------------------------------------- #
# Recompute spec — how to regenerate metrics/ from results/ (Tier-2, offline).
# --------------------------------------------------------------------------- #
@dataclass
class RecomputeSpec:
    """Declares *what* to run to recompute an experiment's metrics offline.

    The contract only names the entry point; the runner (``expkit.recompute``)
    turns it into a concrete subprocess (interpreter, cwd, experiment path).
    """

    runner: str  # "module" | "eval_core_metrics" | "command"
    module: str | None = None  # runner="module": e.g. "scripts.exp13_metrics"
    command: list[str] | None = None  # runner="command": explicit argv (no python)
    source: str = ""  # "config" | "default" — where the spec came from


def recompute_spec(
    slug: str,
    config: dict[str, Any] | None = None,
    family: str | None = None,
) -> RecomputeSpec | None:
    """Resolve how to recompute this experiment's metrics, offline.

    ``config['recompute']`` (explicit) wins; accepted forms::

        recompute: eval_core                  # qa: run `eval_core metrics <exp>`
        recompute: scripts.exp15_metrics      # retrieval: run that module
        recompute: { module: scripts.exp15_metrics }
        recompute: { command: [python, -m, scripts.exp15_metrics, --full] }

    Otherwise a family default is used::

        retrieval -> module  scripts.exp<NN>_metrics   (NN from the slug)
        qa        -> eval_core_metrics

    Returns ``None`` when the family is unknown / no entry point can be derived.
    """
    fam = family or detect_family(config=config)
    rc = config.get("recompute") if isinstance(config, dict) else None

    if isinstance(rc, str) and rc.strip():
        token = rc.strip()
        if token in ("eval_core", "eval_core_metrics"):
            return RecomputeSpec(runner="eval_core_metrics", source="config")
        return RecomputeSpec(runner="module", module=token, source="config")
    if isinstance(rc, dict):
        if rc.get("module"):
            return RecomputeSpec(runner="module", module=str(rc["module"]), source="config")
        if rc.get("command"):
            return RecomputeSpec(
                runner="command",
                command=[str(x) for x in rc["command"]],
                source="config",
            )
        if str(rc.get("runner", "")).startswith("eval_core"):
            return RecomputeSpec(runner="eval_core_metrics", source="config")

    if fam == RETRIEVAL:
        nn = experiment_number(slug)
        if nn is None:
            return None
        return RecomputeSpec(runner="module", module=f"scripts.exp{nn:02d}_metrics", source="default")
    if fam == QA:
        return RecomputeSpec(runner="eval_core_metrics", source="default")
    return None


# --------------------------------------------------------------------------- #
# Validation — does a folder honour the contract (and is it comparable)?
# --------------------------------------------------------------------------- #
@dataclass
class ValidationReport:
    slug: str
    path: Path
    family: str = UNKNOWN
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the folder is comparable (no hard errors)."""
        return not self.errors


def validate_experiment(path: Path | str) -> ValidationReport:
    """Check one experiment folder against the contract.

    Hard ``errors`` mean *not comparable* (the leaderboard can't use it).
    ``warnings`` are non-fatal (still comparable, but something is off — e.g.
    family only inferred, or Tier-2 recompute unavailable).
    """
    path = Path(path)
    rep = ValidationReport(slug=path.name, path=path)

    if not path.is_dir():
        rep.errors.append("not a directory")
        return rep
    if not EXP_DIR_RE.match(path.name):
        rep.warnings.append(
            "folder name should start with its number, e.g. 15_my_experiment"
        )

    # --- config.yaml (metadata) -------------------------------------------
    cfg: dict[str, Any] = {}
    cp = config_path(path)
    if not cp.is_file():
        rep.warnings.append(f"missing {CONFIG_FILENAME} (no name/date/family metadata)")
    elif yaml is None:
        rep.warnings.append("PyYAML not installed — cannot read config.yaml")
    else:
        cfg = load_config(path)
        if not cfg:
            rep.warnings.append(f"{CONFIG_FILENAME} is empty or not valid YAML")

    # --- metrics JSON (the comparability gate) ----------------------------
    metrics: Any = None
    mp = metrics_json_path(path)
    if not mp.is_file():
        rep.errors.append(f"missing {METRICS_REL} (not comparable)")
    else:
        try:
            metrics = json.loads(mp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            rep.errors.append(f"{METRICS_REL} is invalid JSON: {exc}")

    # --- family -----------------------------------------------------------
    shape_fam = _metrics_shape_family(metrics) if metrics is not None else UNKNOWN
    fam = detect_family(metrics, cfg)
    rep.family = fam
    rep.info["family"] = fam
    rep.info["family_inferred"] = normalize_family(cfg.get("family")) is None

    declared = cfg.get("family")
    if declared is not None and normalize_family(declared) is None:
        rep.errors.append(f"config.family={declared!r} is not one of {FAMILIES}")
    elif normalize_family(declared) and metrics is not None and shape_fam != UNKNOWN:
        if normalize_family(declared) != shape_fam:
            rep.errors.append(
                f"config.family={declared!r} but metrics shape looks like {shape_fam!r}"
            )
    elif declared is None and metrics is not None:
        if shape_fam == UNKNOWN:
            rep.errors.append(
                "cannot determine family: no config.family and unrecognized metrics shape"
            )
        else:
            rep.warnings.append(
                f"family inferred as {shape_fam!r} from metrics shape; add "
                f"`family: {shape_fam}` to {CONFIG_FILENAME} to make it explicit"
            )

    # --- recompute entry point (Tier-2 readiness) -------------------------
    spec = recompute_spec(path.name, cfg, fam)
    if spec is None:
        rep.warnings.append("no recompute entry point (offline Tier-2 regen unavailable)")
    else:
        label = spec.runner + (f":{spec.module}" if spec.module else "")
        rep.info["recompute"] = f"{label} ({spec.source})"
        if not results_dir(path).is_dir():
            rep.warnings.append(
                f"no {RESULTS_DIRNAME}/ (Tier-2 recompute would have no inputs)"
            )

    return rep


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def print_report(rep: ValidationReport) -> None:
    """Print a single validation report (shared by both repos' ``validate`` CLIs)."""
    status = "OK  " if rep.ok else "FAIL"
    extra = rep.info.get("recompute", "")
    print(f"[{status}] {rep.slug}  family={rep.family}" + (f"  recompute={extra}" if extra else ""))
    for e in rep.errors:
        print(f"    error:   {e}")
    for w in rep.warnings:
        print(f"    warning: {w}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="experiment_contract",
        description="Validate experiment folders against the shared contract.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="Validate one or more experiment folders.")
    v.add_argument("paths", nargs="+", type=Path, help="Experiment folder(s).")
    args = parser.parse_args(argv)

    if args.cmd == "validate":
        rc = 0
        n_ok = 0
        for p in args.paths:
            rep = validate_experiment(p)
            print_report(rep)
            if rep.ok:
                n_ok += 1
            else:
                rc = 1
        print(f"\n{n_ok}/{len(args.paths)} folder(s) comparable.")
        return rc
    return 2


if __name__ == "__main__":
    sys.exit(main())
