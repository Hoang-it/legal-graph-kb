from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import eval_core.metrics as metrics_module
from eval_core.report import compute_and_write_academic_metrics
from eval_core.runners import (
    build_metric_record_groups,
    compute_experiment_academic_metrics,
)

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "eval_core" / "samples"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "academic_metrics"
SAMPLE_ARMS = ["graphrag", "logic_lm_graphrag"]
GOLD_RAW_BY_STT = {
    1: "41/2024/QH15 Dieu 64",
    2: "41/2024/QH15 Dieu 50",
}


def _assert_expected_subset(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            _assert_expected_subset(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def test_sample_academic_metrics_end_to_end(tmp_path: Path) -> None:
    expected = json.loads(
        (FIXTURE_DIR / "expected_summary.json").read_text(encoding="utf-8")
    )
    metrics_out = tmp_path / "academic_metrics.json"
    csv_out = tmp_path / "academic_metrics.csv"
    report_out = tmp_path / "academic_report.md"
    records = json.loads((SAMPLE_DIR / "records.json").read_text(encoding="utf-8"))

    result = compute_and_write_academic_metrics(
        records=records,
        output_dir=tmp_path / "metrics",
        metrics_out=metrics_out,
        csv_out=csv_out,
        report_out=report_out,
    )

    actual_by_index = {str(r["record_index"]): r for r in result["records"]}
    for record_index, expected_record in expected["records"].items():
        actual_citation = actual_by_index[record_index]["citation"]
        actual_record = {
            "citation_recall": actual_citation["citation_recall"],
            "citation_precision": actual_citation["citation_precision"],
            "citation_f1": actual_citation["citation_f1"],
            "citation_display_rate": actual_citation["citation_display"][
                "citation_display_rate"
            ],
            "precision_denom": actual_citation["precision_denom"],
            "pred_parse_errors": actual_citation["pred_parse_errors"],
        }
        assert actual_record == expected_record

    _assert_expected_subset(result["aggregate"], expected["aggregate"])

    assert result["bertscore_metadata"]["status"] == "no_records_with_gold_answer"

    written_result = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert written_result["aggregate"] == result["aggregate"]

    rows = list(csv.DictReader(csv_out.open(encoding="utf-8", newline="")))
    assert len(rows) == 4
    assert "arm" not in rows[0]
    assert "stt" not in rows[0]

    report_text = report_out.read_text(encoding="utf-8")
    assert "# Academic Metrics Report" in report_text
    assert "Headline Macro Metrics" in report_text


def test_sample_academic_metrics_aggregates_bertscore_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_compute_bertscore(records: list[dict[str, Any]]) -> tuple[dict, dict]:
        assert len(records) == 4
        assert all("arm" not in record and "stt" not in record for record in records)
        return (
            {
                "1": {
                    "bertscore_p": 0.8,
                    "bertscore_r": 0.7,
                    "bertscore_f1": 0.6,
                    "candidate_source": "cleaned_answer",
                    "status": "ok",
                },
                "2": {
                    "bertscore_p": 0.6,
                    "bertscore_r": 0.5,
                    "bertscore_f1": 0.4,
                    "candidate_source": "cleaned_answer",
                    "status": "ok",
                },
                "3": {
                    "bertscore_p": 0.9,
                    "bertscore_r": 0.8,
                    "bertscore_f1": 0.7,
                    "candidate_source": "cleaned_answer",
                    "status": "ok",
                },
                "4": {
                    "bertscore_p": 0.5,
                    "bertscore_r": 0.4,
                    "bertscore_f1": 0.3,
                    "candidate_source": "cleaned_answer",
                    "status": "ok",
                },
            },
            {"status": "stubbed_for_sample_test"},
        )

    monkeypatch.setattr(metrics_module, "compute_bertscore", fake_compute_bertscore)
    csv_out = tmp_path / "academic_metrics.csv"
    records = json.loads((SAMPLE_DIR / "records.json").read_text(encoding="utf-8"))

    result = compute_and_write_academic_metrics(
        records=records,
        output_dir=tmp_path / "metrics",
        csv_out=csv_out,
    )

    assert result["bertscore_metadata"]["status"] == "stubbed_for_sample_test"
    assert result["aggregate"]["macro"]["bertscore_p"] == 0.7
    assert result["aggregate"]["macro"]["bertscore_r"] == 0.6
    assert result["aggregate"]["macro"]["bertscore_f1"] == 0.5

    rows = {
        row["record_index"]: row
        for row in csv.DictReader(csv_out.open(encoding="utf-8", newline=""))
    }
    assert rows["1"]["bertscore_f1"] == "0.6"
    assert rows["4"]["bertscore_f1"] == "0.3"


def test_experiment_loader_builds_metric_records_from_sample_layout(tmp_path: Path) -> None:
    records = json.loads((SAMPLE_DIR / "records.json").read_text(encoding="utf-8"))
    questions_path = tmp_path / "questions.json"
    results_root = tmp_path / "results"

    questions = []
    for stt, gold_raw in GOLD_RAW_BY_STT.items():
        questions.append(
            {
                "stt": stt,
                "question": f"Experiment loader sample question {stt}",
                "group": "sample",
                "gold_citations_raw": gold_raw,
            }
        )
    questions_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")

    grouped_records = {
        SAMPLE_ARMS[0]: records[:2],
        SAMPLE_ARMS[1]: records[2:],
    }
    for group, group_records in grouped_records.items():
        arm_dir = results_root / group
        arm_dir.mkdir(parents=True, exist_ok=True)
        for stt, rec in enumerate(group_records, start=1):
            result_record = {
                key: value
                for key, value in rec.items()
                if key != "gold_articles"
            }
            result_record["arm"] = group
            result_record["stt"] = stt
            (arm_dir / f"A{stt}.json").write_text(
                json.dumps(result_record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    result = compute_experiment_academic_metrics(
        results_root=results_root,
        questions_path=questions_path,
        output_dir=tmp_path / "metrics",
        arms=SAMPLE_ARMS,
    )

    assert result["metadata"]["results_root"] == str(results_root)
    assert result["metadata"]["arms_filter"] == SAMPLE_ARMS
    # Under academic_v2 (strict tuple-equal, 2026-05-31):
    #   graphrag arm = records #1 (cite L41_2024.A64.K1 vs gold L41_2024.A64)
    #                  and #2 (cite L41_2024.A51 vs gold L41_2024.A50)
    #   → record #1 MISS (over-specified khoản), #2 MISS (wrong article)
    #   → macro recall 0.0 (was 0.5 under v1 which counted #1 as HIT
    #     via article-only intersection).
    assert result["aggregates"]["graphrag"]["macro"]["citation_recall"] == 0.0
    # logic_lm_graphrag arm = records #3 (cite L41_2024.A64 vs gold L41_2024.A64)
    # and #4 (empty citations) → 1 HIT + 1 MISS → recall 0.5 (unchanged).
    assert result["aggregates"]["logic_lm_graphrag"]["macro"]["citation_recall"] == 0.5
    assert result["aggregates"]["logic_lm_graphrag"]["prolog"][
        "prolog_first_try_solution_rate"
    ] == 0.5
    assert (tmp_path / "metrics" / "academic" / "gold_citations_normalized.json").exists()


def test_experiment_grouping_strips_experiment_identity_before_evaluation() -> None:
    record_groups = build_metric_record_groups(
        records=[
            {
                "arm": "graphrag",
                "stt": 7,
                "answer": "sample",
                "citation_ids": [],
            }
        ],
        gold_map={7: ["L41_2024.A64"]},
        question_map={7: {"gold_answer": "gold"}},
    )

    metric_record = record_groups["graphrag"][0]
    assert "arm" not in metric_record
    assert "stt" not in metric_record
    assert metric_record["record_id"] == "7"
    assert metric_record["gold_articles"] == ["L41_2024.A64"]
