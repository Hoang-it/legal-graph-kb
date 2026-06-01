"""Unit + end-to-end tests for the retrieval metric engine.

The metric *definitions* in :mod:`eval_core.retrieval_metrics` are the
contract (skill Rule 2). These tests pin them to hand-computed values so a
later refactor can't silently change a number, and exercise
``compute_retrieval_metrics`` end-to-end (config -> scoring -> JSON shape)
with the external gold / law-metadata helpers stubbed out so the test needs
no Neo4j / project data.
"""
from __future__ import annotations

import json
import math
import types
from pathlib import Path

import pytest
import yaml

import eval_core.retrieval_metrics as rm
from eval_core.experiment import Experiment


# --------------------------------------------------------------------------- #
# Primitives — hand-computed expected values.
# --------------------------------------------------------------------------- #
def test_recall_precision_f1():
    gold = {"A1", "A2"}
    assert rm.recall(["A1", "A3"], gold) == 0.5
    assert rm.recall(["A1", "A2"], gold) == 1.0
    assert rm.recall(["A1"], set()) is None          # no gold -> undefined
    assert rm.precision(["A1", "A3"], gold) == 0.5
    assert rm.precision([], gold) == 0.0             # retrieved nothing but gold exists
    assert rm.precision([], set()) is None
    assert rm.f1_score(0.5, 0.5) == 0.5
    assert rm.f1_score(None, 0.5) is None
    assert rm.f1_score(0.0, 0.0) == 0.0


def test_r_precision_and_mrr():
    gold = {"A1", "A2"}
    # top-|gold|=top-2 of [A1,A3,A2,A4] = [A1,A3] -> 1 hit / 2
    assert rm.r_precision(["A1", "A3", "A2", "A4"], gold) == 0.5
    assert rm.r_precision(["A1", "A2"], gold) == 1.0
    assert rm.r_precision(["A1"], set()) is None
    assert rm.reciprocal_rank(["A3", "A1"], {"A1"}) == 0.5
    assert rm.reciprocal_rank(["A1"], {"A1"}) == 1.0
    assert rm.reciprocal_rank(["A3"], {"A1"}) == 0.0
    assert rm.reciprocal_rank([], {"A1"}) == 0.0
    assert rm.reciprocal_rank(["A1"], set()) is None


def test_ndcg():
    gold = {"A1", "A2"}
    retrieved = ["A1", "A3", "A2", "A4"]
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    # @all: A1@1 (1/log2 2) + A2@3 (1/log2 4)
    assert rm.ndcg_at_k(retrieved, gold, 4) == pytest.approx((1.0 + 0.5) / idcg)
    # @2: only A1@1 counts
    assert rm.ndcg_at_k(retrieved, gold, 2) == pytest.approx(1.0 / idcg)
    assert rm.ndcg_at_k(retrieved, set(), 4) is None


def test_categorize():
    codes = {"41/2024/QH15"}
    assert rm.categorize("41/2024/QH15", codes) == "in_corpus"
    assert rm.categorize("99/2099/QH15", codes) == "ooc"
    assert rm.categorize("41/2024/QH15\n99/2099/QH15", codes) == "mixed"
    assert rm.categorize("", codes) == "unparseable"
    assert rm.categorize("no legal code here", codes) == "unparseable"
    assert rm.categorize(["41/2024/QH15"], codes) == "in_corpus"   # list input joined


def test_parse_ks_and_dig():
    assert rm._parse_ks([12, "all"]) == (12, None)
    assert rm._parse_ks(["ALL", 5]) == (None, 5)
    assert rm._parse_ks(None) == rm._DEFAULT_KS
    assert rm._parse_ks([]) == rm._DEFAULT_KS
    assert rm._dig({"a": {"b": [1, 2]}}, "a.b") == [1, 2]
    assert rm._dig({"a": 1}, "a.b") is None
    assert rm._dig({}, "a.b") is None


def test_score_and_aggregate():
    gold_map = {1: ["A1", "A2"]}
    questions = {1: {"gold_citations_raw": "41/2024/QH15"}}
    records = {1: {"stt": 1, "retrieval_only": {"final_article_ids": ["A1", "A3", "A2", "A4"], "elapsed_s": 0.2}}}
    ks = (2, None)
    rows = rm.score_arm(
        "good", records, gold_map, questions, {"41/2024/QH15"}, ks,
        "retrieval_only.final_article_ids", "retrieval_only.elapsed_s",
    )
    assert rows[0]["category"] == "in_corpus"
    macro = rm.aggregate_macro(rows, ks)
    assert macro["n"] == 1
    assert macro["recall@2"] == 0.5
    assert macro["recall@all"] == 1.0
    assert macro["precision@all"] == 0.5
    assert macro["r_precision"] == 0.5
    assert macro["mrr"] == 1.0
    assert macro["avg_n_retrieved@2"] == 2
    strat = rm.aggregate_stratified(rows, ks)
    assert strat["in_corpus"]["n"] == 1
    assert strat["ooc"] == {"n": 0}


# --------------------------------------------------------------------------- #
# End-to-end: config -> compute_retrieval_metrics -> academic_metrics.json,
# with gold + law-metadata helpers stubbed (no project data needed).
# --------------------------------------------------------------------------- #
def test_compute_retrieval_metrics_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rm, "load_law_metadata",
        lambda: {"bhxh": types.SimpleNamespace(full_id="41/2024/QH15")},
    )

    def fake_validate(questions_path, registry_path, out_dir):
        norm = Path(out_dir) / "gold_citations_normalized.json"
        norm.write_text(
            json.dumps({"records": {"1": {"gold_articles": ["A1", "A2"]}}}),
            encoding="utf-8",
        )
        return True, {"normalized_path": str(norm)}

    monkeypatch.setattr(rm, "validate_gold_citations", fake_validate)

    exp_dir = tmp_path / "15_synth_retrieval"
    (exp_dir / "results" / "good").mkdir(parents=True)
    (exp_dir / "metrics").mkdir(parents=True)
    (exp_dir / "results" / "good" / "A1.json").write_text(
        json.dumps({"stt": 1, "retrieval_only": {
            "final_article_ids": ["A1", "A3", "A2", "A4"], "elapsed_s": 0.1}}),
        encoding="utf-8",
    )
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps([{"stt": 1, "gold_citations_raw": "41/2024/QH15"}]), encoding="utf-8",
    )
    (exp_dir / "config.yaml").write_text(
        yaml.safe_dump({
            "name": "synthetic retrieval",
            "family": "retrieval",
            "dataset": {"questions": str(questions)},
            "retrieval": {"arms": ["good"], "ks": [2, "all"]},
        }),
        encoding="utf-8",
    )

    exp = Experiment.from_path(exp_dir)
    result = rm.compute_retrieval_metrics(exp)

    data = json.loads(Path(result["metrics_out"]).read_text(encoding="utf-8"))
    assert data["family"] == "retrieval"
    assert data["arms"] == ["good"]
    assert data["Ks"] == [2, "all"]
    macro = data["overall_macro"]["good"]
    assert macro["recall@2"] == 0.5
    assert macro["recall@all"] == 1.0
    assert macro["r_precision"] == 0.5
    assert macro["mrr"] == 1.0
    assert data["stratified"]["good"]["in_corpus"]["n"] == 1
    # report + csv written
    assert Path(result["report_out"]).is_file()
    assert Path(result["csv_out"]).is_file()
