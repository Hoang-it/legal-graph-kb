"""Producer-side binding for the shared experiment contract.

``experiment_contract.py`` is shipped byte-identical to the consumer
(experiments) repo, so the producer must keep agreeing with it: the folder
layout it declares has to match ``eval_core.paths`` (the engine that writes those
folders), the ``_template`` it ships must satisfy the contract, and the real
experiment folders must validate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import experiment_contract as ec
from eval_core import paths

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"

# Metrics-JSON shapes the contract uses to detect family (see _metrics_shape_family).
_SHAPE = {
    ec.QA: {"aggregates": {"graphrag": {"macro": {"citation_f1": 0.0}}}},
    ec.RETRIEVAL: {"overall_macro": {}, "stratified": {}, "Ks": [12, "all"]},
}


def _make_experiment(tmp_path: Path, slug: str, family: str, *, with_metrics: bool = True) -> Path:
    """Build a minimal contract-valid experiment folder under ``tmp_path``."""
    exp = tmp_path / slug
    (exp / paths.METRICS_DIRNAME).mkdir(parents=True)
    (exp / paths.CONFIG_FILENAME).write_text(
        yaml.safe_dump({"name": slug, "family": family, "recompute": "eval_core"}),
        encoding="utf-8",
    )
    if with_metrics:
        payload = dict(_SHAPE[family])
        payload["family"] = family
        ec.metrics_json_path(exp).write_text(json.dumps(payload), encoding="utf-8")
    return exp


def test_contract_layout_matches_eval_core_paths():
    """The contract's filenames must equal the producer engine's (no drift)."""
    assert ec.CONFIG_FILENAME == paths.CONFIG_FILENAME
    assert ec.METRICS_DIRNAME == paths.METRICS_DIRNAME
    assert ec.RESULTS_DIRNAME == paths.RESULTS_DIRNAME
    assert ec.REPORT_DIRNAME == paths.REPORT_DIRNAME
    assert ec.METRICS_JSON == paths.METRICS_JSON
    # The contract's metrics path helper must point at the same file.
    exp = EXPERIMENTS / "15_example"
    assert ec.metrics_json_path(exp) == paths.metrics_json_path(exp)


def test_template_declares_family_and_recompute():
    """New experiments are created from _template, which must be contract-ready."""
    cfg = yaml.safe_load((EXPERIMENTS / "_template" / "config.yaml").read_text(encoding="utf-8"))
    assert ec.normalize_family(cfg.get("family")) in (ec.QA, ec.RETRIEVAL)
    assert "recompute" in cfg


@pytest.mark.parametrize("family", [ec.QA, ec.RETRIEVAL])
def test_default_recompute_is_eval_core(family):
    """Both families recompute offline through eval_core (no per-experiment script)."""
    spec = ec.recompute_spec(f"15_some_{family}", {}, family)
    assert spec is not None and spec.runner == "eval_core_metrics"


@pytest.mark.parametrize("family", [ec.QA, ec.RETRIEVAL])
def test_synthetic_experiments_validate(tmp_path, family):
    exp = _make_experiment(tmp_path, f"15_synth_{family}", family)
    rep = ec.validate_experiment(exp)
    assert rep.ok, f"{family} synthetic not comparable: {rep.errors}"
    assert rep.family == family


def test_experiment_without_metrics_is_not_comparable(tmp_path):
    # A folder with config but no academic_metrics.json (e.g. an aborted run).
    exp = _make_experiment(tmp_path, "15_no_metrics", ec.RETRIEVAL, with_metrics=False)
    rep = ec.validate_experiment(exp)
    assert not rep.ok
