"""Producer-side binding for the shared experiment contract.

``experiment_contract.py`` is shipped byte-identical to the consumer
(experiments) repo, so the producer must keep agreeing with it: the folder
layout it declares has to match ``eval_core.paths`` (the engine that writes those
folders), the ``_template`` it ships must satisfy the contract, and the real
experiment folders must validate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import experiment_contract as ec
from eval_core import paths

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"


def test_contract_layout_matches_eval_core_paths():
    """The contract's filenames must equal the producer engine's (no drift)."""
    assert ec.CONFIG_FILENAME == paths.CONFIG_FILENAME
    assert ec.METRICS_DIRNAME == paths.METRICS_DIRNAME
    assert ec.RESULTS_DIRNAME == paths.RESULTS_DIRNAME
    assert ec.REPORT_DIRNAME == paths.REPORT_DIRNAME
    assert ec.METRICS_JSON == paths.METRICS_JSON
    # The contract's metrics path helper must point at the same file.
    exp = EXPERIMENTS / "01_initial_eval"
    assert ec.metrics_json_path(exp) == paths.metrics_json_path(exp)


def test_template_declares_family_and_recompute():
    """New experiments are created from _template, which must be contract-ready."""
    cfg = yaml.safe_load((EXPERIMENTS / "_template" / "config.yaml").read_text(encoding="utf-8"))
    assert ec.normalize_family(cfg.get("family")) in (ec.QA, ec.RETRIEVAL)
    assert "recompute" in cfg


def test_retrieval_default_recompute_module_exists():
    """The family-default retrieval entry point must resolve to a real script."""
    spec = ec.recompute_spec("13_hyde_semantic", {}, ec.RETRIEVAL)
    assert spec is not None and spec.runner == "module"
    rel = Path(*spec.module.split(".")).with_suffix(".py")
    assert (REPO / rel).is_file(), f"missing producer script {rel}"


def test_qa_default_recompute_is_eval_core():
    spec = ec.recompute_spec("01_initial_eval", {}, ec.QA)
    assert spec is not None and spec.runner == "eval_core_metrics"


@pytest.mark.parametrize(
    "slug, family",
    [("01_initial_eval", ec.QA), ("13_hyde_semantic", ec.RETRIEVAL)],
)
def test_real_experiments_validate(slug, family):
    rep = ec.validate_experiment(EXPERIMENTS / slug)
    assert rep.ok, f"{slug} not comparable: {rep.errors}"
    assert rep.family == family


def test_experiment_without_metrics_is_not_comparable():
    # 05 was an aborted audit with no academic_metrics.json.
    rep = ec.validate_experiment(EXPERIMENTS / "05_v5_retrieval_audit")
    assert not rep.ok
