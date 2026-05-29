"""Tests for :mod:`eval_core.experiment` — Experiment class + inheritance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from eval_core.experiment import (
    ArmConfig,
    DatasetConfig,
    Experiment,
    MultimodelConfig,
)
from eval_core import paths


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _write_record(arm_dir: Path, stt: int, arm: str) -> None:
    arm_dir.mkdir(parents=True, exist_ok=True)
    rec = {"arm": arm, "stt": stt, "answer": f"answer {stt}", "citation_ids": []}
    (arm_dir / f"A{stt}.json").write_text(
        json.dumps(rec, ensure_ascii=False), encoding="utf-8"
    )


@pytest.fixture
def make_experiment(tmp_path: Path):
    """Returns a builder that creates an experiment folder under tmp_path."""

    def _build(
        name: str,
        config: dict,
        records: dict[str, list[int]] | None = None,
    ) -> Path:
        exp_dir = tmp_path / name
        _write_yaml(exp_dir / paths.CONFIG_FILENAME, config)
        if records:
            for arm, stts in records.items():
                arm_dir = paths.arm_results_dir(exp_dir, arm)
                for stt in stts:
                    _write_record(arm_dir, stt, arm)
        return exp_dir

    return _build


@pytest.fixture
def dataset_file(tmp_path: Path) -> Path:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps([{"stt": 1, "question": "Q1"}, {"stt": 2, "question": "Q2"}]),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# from_path + config parsing
# ---------------------------------------------------------------------------


def test_from_path_loads_config(make_experiment, dataset_file):
    exp_dir = make_experiment(
        "01_a",
        {
            "name": "First",
            "dataset": {"questions": str(dataset_file), "n": 2},
            "arms": {
                "graphrag": {"mode": "run", "model": "gpt-4o-mini"},
                "llm_only": {"mode": "run"},
            },
        },
    )
    exp = Experiment.from_path(exp_dir)
    assert exp.name == "First"
    assert exp.dataset == DatasetConfig(questions=Path(str(dataset_file)), n=2)
    assert exp.arms == {
        "graphrag": ArmConfig(name="graphrag", mode="run", model="gpt-4o-mini"),
        "llm_only": ArmConfig(name="llm_only", mode="run", model=None),
    }
    assert exp.parent is None
    assert exp.multimodel is None


def test_from_path_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        Experiment.from_path(tmp_path / "nope")


def test_from_path_missing_config(tmp_path: Path):
    exp_dir = tmp_path / "01_a"
    exp_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        Experiment.from_path(exp_dir)


def test_arm_mode_validation(make_experiment, dataset_file):
    exp_dir = make_experiment(
        "01_bad",
        {
            "dataset": {"questions": str(dataset_file)},
            "arms": {"graphrag": {"mode": "bogus"}},
        },
    )
    with pytest.raises(ValueError, match="mode must be 'run' or 'inherit'"):
        _ = Experiment.from_path(exp_dir).arms


def test_multimodel_parsing(make_experiment, dataset_file):
    exp_dir = make_experiment(
        "01_mm",
        {
            "dataset": {"questions": str(dataset_file)},
            "arms": {},
            "multimodel": {
                "arms": ["logic_lm_graphrag"],
                "models": ["gpt-4.1", "gpt-4o"],
            },
        },
    )
    mm = Experiment.from_path(exp_dir).multimodel
    assert mm == MultimodelConfig(
        arms=("logic_lm_graphrag",),
        models=("gpt-4.1", "gpt-4o"),
    )


# ---------------------------------------------------------------------------
# records_for_arm — own and inherited
# ---------------------------------------------------------------------------


def test_records_for_arm_own(make_experiment, dataset_file):
    exp_dir = make_experiment(
        "01_own",
        {
            "dataset": {"questions": str(dataset_file)},
            "arms": {"graphrag": {"mode": "run"}},
        },
        records={"graphrag": [1, 2]},
    )
    exp = Experiment.from_path(exp_dir)
    records = exp.records_for_arm("graphrag")
    assert len(records) == 2
    assert {r["stt"] for r in records} == {1, 2}
    assert all(r["arm"] == "graphrag" for r in records)
    assert all(r["_record_path"].endswith(".json") for r in records)


def test_records_for_arm_inherit_from_parent(make_experiment, dataset_file):
    make_experiment(
        "01_parent",
        {
            "dataset": {"questions": str(dataset_file)},
            "arms": {"graphrag": {"mode": "run"}},
        },
        records={"graphrag": [1, 2]},
    )
    child_dir = make_experiment(
        "02_child",
        {
            "dataset": {"questions": str(dataset_file)},
            "parent": "01_parent",
            "arms": {"graphrag": {"mode": "inherit"}},
        },
    )
    child = Experiment.from_path(child_dir)
    records = child.records_for_arm("graphrag")
    assert len(records) == 2
    # records_source reports the parent as the owner
    assert child.records_source("graphrag").name == "01_parent"


def test_records_for_arm_unknown_arm(make_experiment, dataset_file):
    exp_dir = make_experiment(
        "01_only_g",
        {
            "dataset": {"questions": str(dataset_file)},
            "arms": {"graphrag": {"mode": "run"}},
        },
        records={"graphrag": [1]},
    )
    exp = Experiment.from_path(exp_dir)
    with pytest.raises(KeyError, match="llm_only"):
        exp.records_for_arm("llm_only")


def test_records_inherit_missing_parent_records(make_experiment, dataset_file):
    make_experiment(
        "01_empty_parent",
        {
            "dataset": {"questions": str(dataset_file)},
            "arms": {"graphrag": {"mode": "run"}},
        },
        records=None,  # no records on disk
    )
    child_dir = make_experiment(
        "02_child",
        {
            "dataset": {"questions": str(dataset_file)},
            "parent": "01_empty_parent",
            "arms": {"graphrag": {"mode": "inherit"}},
        },
    )
    with pytest.raises(FileNotFoundError):
        Experiment.from_path(child_dir).records_for_arm("graphrag")


def test_inherit_without_parent_declared(make_experiment, dataset_file):
    exp_dir = make_experiment(
        "01_orphan",
        {
            "dataset": {"questions": str(dataset_file)},
            "arms": {"graphrag": {"mode": "inherit"}},
        },
    )
    exp = Experiment.from_path(exp_dir)
    with pytest.raises(RuntimeError, match="mode=inherit but no parent"):
        exp.records_for_arm("graphrag")


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


def test_cycle_detection_two_node(make_experiment, dataset_file):
    # A references B as parent; B references A. Both inherit arm 'x'.
    a_dir = make_experiment(
        "exp_a",
        {
            "dataset": {"questions": str(dataset_file)},
            "parent": "exp_b",
            "arms": {"graphrag": {"mode": "inherit"}},
        },
    )
    make_experiment(
        "exp_b",
        {
            "dataset": {"questions": str(dataset_file)},
            "parent": "exp_a",
            "arms": {"graphrag": {"mode": "inherit"}},
        },
    )
    exp_a = Experiment.from_path(a_dir)
    with pytest.raises(RuntimeError, match="cycle"):
        exp_a.records_for_arm("graphrag")


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def test_validate_passes_for_clean_experiment(make_experiment, dataset_file):
    exp_dir = make_experiment(
        "01_clean",
        {
            "dataset": {"questions": str(dataset_file)},
            "arms": {"graphrag": {"mode": "run"}},
        },
        records={"graphrag": [1]},
    )
    Experiment.from_path(exp_dir).validate()


def test_validate_missing_dataset(make_experiment, tmp_path: Path):
    exp_dir = make_experiment(
        "01_no_data",
        {
            "dataset": {"questions": str(tmp_path / "missing.json")},
            "arms": {"graphrag": {"mode": "run"}},
        },
    )
    with pytest.raises(FileNotFoundError, match="Dataset"):
        Experiment.from_path(exp_dir).validate()


def test_validate_inheritance_missing_parent_arm(make_experiment, dataset_file):
    make_experiment(
        "01_parent_no_recs",
        {
            "dataset": {"questions": str(dataset_file)},
            "arms": {"graphrag": {"mode": "run"}},  # declared but no records on disk
        },
    )
    child_dir = make_experiment(
        "02_child",
        {
            "dataset": {"questions": str(dataset_file)},
            "parent": "01_parent_no_recs",
            "arms": {"graphrag": {"mode": "inherit"}},
        },
    )
    with pytest.raises(FileNotFoundError):
        Experiment.from_path(child_dir).validate()


# ---------------------------------------------------------------------------
# Standard paths
# ---------------------------------------------------------------------------


def test_standard_paths(make_experiment, dataset_file):
    exp_dir = make_experiment(
        "01_paths",
        {
            "dataset": {"questions": str(dataset_file)},
            "arms": {"graphrag": {"mode": "run"}},
        },
    )
    exp = Experiment.from_path(exp_dir)
    assert exp.results_dir == exp_dir.resolve() / "results"
    assert exp.metrics_dir == exp_dir.resolve() / "metrics"
    assert exp.report_dir == exp_dir.resolve() / "report"
    assert exp.arm_results_dir("graphrag") == exp_dir.resolve() / "results" / "graphrag"
    assert exp.multimodel_combo_dir("logic_lm_graphrag", "gpt-4_1") == (
        exp_dir.resolve() / "results" / "multimodel" / "logic_lm_graphrag__gpt-4_1"
    )


def test_prompts_override_dir_relative_resolves(make_experiment, dataset_file):
    exp_dir = make_experiment(
        "01_po",
        {
            "dataset": {"questions": str(dataset_file)},
            "arms": {},
            "prompts_override_dir": "my_prompts",
        },
    )
    exp = Experiment.from_path(exp_dir)
    assert exp.prompts_override_dir == (exp_dir / "my_prompts").resolve()


def test_prompts_override_dir_null_returns_none(make_experiment, dataset_file):
    exp_dir = make_experiment(
        "01_no_po",
        {
            "dataset": {"questions": str(dataset_file)},
            "arms": {},
        },
    )
    exp = Experiment.from_path(exp_dir)
    assert exp.prompts_override_dir is None
