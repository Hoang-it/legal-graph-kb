"""Experiment abstraction with parent-inheritance.

An experiment is a directory under ``experiments/`` containing a
``config.yaml`` plus the standard subfolders defined in :mod:`eval_core.paths`.
The :class:`Experiment` class lazily loads the config and resolves the
parent chain on demand.

Config schema (YAML)::

    name: "Human-readable name"
    description: |
      What hypothesis this experiment tests.
    date: "YYYY-MM-DD"

    dataset:
      questions: data/eval/questions_200.json   # path relative to repo root
      n: 200                                    # null = full

    parent: 01_baseline                         # sibling folder under experiments/
                                                # or null when no parent

    prompts_override_dir: prompts_override      # relative to experiment folder,
                                                # or null to use the canonical prompts/

    arms:
      graphrag:              { mode: run, model: gpt-4o-mini }
      llm_only:              { mode: inherit }
      logic_lm_decomposed:   { mode: run, model: gpt-4o-mini }

    multimodel:                                 # optional
      arms: [logic_lm_graphrag]
      models: ["gpt-4.1", "gpt-4o"]

Arm modes
---------

- ``run``     — inference writes records to ``results/<arm>/`` in this
                experiment.
- ``inherit`` — records come from the parent experiment's ``results/<arm>/``.
                The parent is read-only. Resolved recursively up the chain
                with cycle detection.

Path semantics
--------------

All paths in ``config.yaml`` that point *into the repo* (e.g. ``dataset.questions``)
are resolved relative to the repo root. Paths that point *into the experiment
folder* (e.g. ``prompts_override_dir``) are resolved relative to the
experiment directory. This split keeps configs portable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from eval_core import paths

_MAX_PARENT_DEPTH = 10


@dataclass(frozen=True)
class ArmConfig:
    name: str
    mode: str  # "run" | "inherit"
    model: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("run", "inherit"):
            raise ValueError(
                f"arm {self.name!r}: mode must be 'run' or 'inherit', got {self.mode!r}"
            )


@dataclass(frozen=True)
class DatasetConfig:
    questions: Path
    n: int | None = None


@dataclass(frozen=True)
class MultimodelConfig:
    arms: tuple[str, ...] = ()
    models: tuple[str, ...] = ()


class Experiment:
    """One experiment, anchored to its directory."""

    def __init__(self, path: Path, config: dict[str, Any]) -> None:
        self.path: Path = path.resolve()
        self.config: dict[str, Any] = config
        # Cached lazy-loaded parent
        self._parent: Experiment | None = None
        self._parent_resolved: bool = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, path: Path | str) -> Experiment:
        p = Path(path)
        if not p.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {p}")
        cfg_path = p / paths.CONFIG_FILENAME
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing experiment config: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"Top-level YAML in {cfg_path} must be a mapping")
        return cls(p, cfg)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return str(self.config.get("name") or self.path.name)

    @property
    def description(self) -> str:
        return str(self.config.get("description") or "")

    @property
    def dataset(self) -> DatasetConfig:
        d = self.config.get("dataset") or {}
        if "questions" not in d:
            raise ValueError(
                f"{self.name}: config.yaml missing dataset.questions"
            )
        return DatasetConfig(
            questions=Path(d["questions"]),
            n=d.get("n"),
        )

    @property
    def arms(self) -> dict[str, ArmConfig]:
        raw = self.config.get("arms") or {}
        return {
            name: ArmConfig(
                name=name,
                mode=spec.get("mode", "run"),
                model=spec.get("model"),
            )
            for name, spec in raw.items()
        }

    @property
    def multimodel(self) -> MultimodelConfig | None:
        m = self.config.get("multimodel")
        if not m:
            return None
        return MultimodelConfig(
            arms=tuple(m.get("arms") or ()),
            models=tuple(m.get("models") or ()),
        )

    @property
    def parent(self) -> Experiment | None:
        if self._parent_resolved:
            return self._parent
        self._parent_resolved = True
        parent_name = self.config.get("parent")
        if not parent_name:
            return None
        parent_path = self.path.parent / parent_name
        self._parent = Experiment.from_path(parent_path)
        return self._parent

    @property
    def prompts_override_dir(self) -> Path | None:
        rel = self.config.get("prompts_override_dir")
        if not rel:
            return None
        return (self.path / rel).resolve()

    # ------------------------------------------------------------------
    # Standard paths
    # ------------------------------------------------------------------

    @property
    def results_dir(self) -> Path:
        return paths.results_dir(self.path)

    @property
    def metrics_dir(self) -> Path:
        return paths.metrics_dir(self.path)

    @property
    def report_dir(self) -> Path:
        return paths.report_dir(self.path)

    def arm_results_dir(self, arm: str) -> Path:
        return paths.arm_results_dir(self.path, arm)

    def multimodel_combo_dir(self, arm: str, model_safe: str) -> Path:
        return paths.multimodel_combo_dir(self.path, arm, model_safe)

    # ------------------------------------------------------------------
    # Inheritance — records lookup
    # ------------------------------------------------------------------

    def records_for_arm(
        self,
        arm: str,
        _visited: set[Path] | None = None,
    ) -> list[dict[str, Any]]:
        """Return inference records for ``arm`` — own or inherited from parent."""
        if arm not in self.arms:
            raise KeyError(f"Arm {arm!r} not declared in experiment {self.name!r}")
        spec = self.arms[arm]
        if spec.mode == "run":
            return list(self._iter_own_records(arm))

        # mode == "inherit"
        visited = set(_visited) if _visited else set()
        if self.path in visited:
            raise RuntimeError(
                f"Parent cycle detected at {self.path} while resolving arm {arm!r}"
            )
        if len(visited) >= _MAX_PARENT_DEPTH:
            raise RuntimeError(
                f"Parent chain too deep (>{_MAX_PARENT_DEPTH}) at {self.path}"
            )
        visited.add(self.path)
        if self.parent is None:
            raise RuntimeError(
                f"Arm {arm!r} in {self.name!r} has mode=inherit but no parent declared"
            )
        return self.parent.records_for_arm(arm, _visited=visited)

    def records_source(self, arm: str) -> Experiment:
        """Return the experiment that actually OWNS records for ``arm``.

        Walks the parent chain. Useful for provenance reporting.
        """
        if arm not in self.arms:
            raise KeyError(f"Arm {arm!r} not declared in experiment {self.name!r}")
        if self.arms[arm].mode == "run":
            return self
        if self.parent is None:
            raise RuntimeError(
                f"Arm {arm!r} in {self.name!r} has mode=inherit but no parent declared"
            )
        return self.parent.records_source(arm)

    def _iter_own_records(self, arm: str) -> Iterable[dict[str, Any]]:
        arm_dir = self.arm_results_dir(arm)
        if not arm_dir.is_dir():
            raise FileNotFoundError(
                f"Arm {arm!r} mode=run but results dir missing: {arm_dir}"
            )
        record_paths = sorted(arm_dir.glob("A*.json"))
        if not record_paths:
            raise FileNotFoundError(
                f"Arm {arm!r} mode=run but no A*.json records in {arm_dir}"
            )
        for p in record_paths:
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON record: {p}") from exc
            rec.setdefault("arm", arm)
            rec["_record_path"] = str(p)
            yield rec

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Cheap structural validation before running anything expensive.

        Checks:
        - Dataset file exists.
        - Every arm with ``mode=inherit`` resolves end-to-end without cycle.
        - Parent declared but missing → fail (handled by from_path).
        """
        ds = self.dataset
        if not ds.questions.exists():
            raise FileNotFoundError(
                f"Dataset questions file not found: {ds.questions}"
            )
        for arm_name, spec in self.arms.items():
            if spec.mode == "inherit":
                # Resolve to surface missing parent / cycle errors.
                src = self.records_source(arm_name)
                # Also check the source dir has at least one record.
                src_dir = src.arm_results_dir(arm_name)
                if not src_dir.is_dir() or not any(src_dir.glob("A*.json")):
                    raise FileNotFoundError(
                        f"Arm {arm_name!r} inherited from {src.name!r} but "
                        f"no records found in {src_dir}"
                    )
