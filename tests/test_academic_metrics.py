from experiments.arms import MAIN_EXPERIMENT_ARMS, parse_metrics_arms, parse_run_arms
from experiments.compute_academic_metrics import (
    _aggregate_arm,
    _clean_answer_for_semantic,
    compute_citation_metrics,
    compute_prolog_fields,
)
from src.citations import (
    format_citation,
    load_registry,
    normalize_citation_id,
    parse_displayed_citations,
    parse_gold_citations_raw,
)


def test_main_arm_preset_is_shared_between_runner_and_metrics():
    expected = [
        "graphrag",
        "llm_only",
        "logic_lm_no_retrieval",
        "logic_lm_ontology",
        "logic_lm_graphrag",
    ]

    assert list(MAIN_EXPERIMENT_ARMS) == expected
    assert parse_run_arms("main") == expected
    assert parse_metrics_arms("main") == expected
    assert parse_metrics_arms(None) == expected
    assert parse_metrics_arms("all") is None


def test_internal_citation_normalizes_old_point_format():
    assert normalize_citation_id("L41_2024.A64.K1.a") == "L41_2024.A64.K1.Da"


def test_format_citation_uses_registry_canonical_title():
    assert (
        format_citation("L41_2024.A50.K4")
        == "[Luật BHXH 2024 (41/2024/QH15), Điều 50 khoản 4]"
    )


def test_display_parser_requires_explicit_authority():
    registry = load_registry()
    strict = parse_displayed_citations(
        "[Luật BHXH 2024 (41/2024/QH15), Điều 50 khoản 4]", registry
    )
    ambiguous = parse_displayed_citations("[Điều 50 khoản 4]", registry)

    assert [r.item_id for r in strict] == ["L41_2024.A50.K4"]
    assert ambiguous == []


def test_gold_parser_extracts_article_level_authorities():
    registry = load_registry()
    raw = (
        "Khoản 4 Điều 28 Nghị định số 115/2015/NĐ-CP\n"
        "Tại Khoản 2 Điều 53 Nghị định số 152/2006/NĐ-CP"
    )
    result = parse_gold_citations_raw(raw, registry)

    assert not result.errors
    assert {r.article_id for r in result.refs} == {"ND115_2015.A28", "ND152_2006.A53"}


def test_gold_parser_fails_when_article_missing():
    registry = load_registry()
    result = parse_gold_citations_raw("Luật BHXH 2024", registry)

    assert result.errors
    assert result.errors[0].error_type == "article_missing"


def test_citation_metrics_use_gold_article_level_and_strict_authority():
    registry = load_registry()
    record = {
        "answer": "[Luật BHXH 2024 (41/2024/QH15), Điều 50 khoản 4]",
        "citation_ids": ["L41_2024.A50.K4", "L41_2024.A50.K5", "BAD_ID"],
    }
    metrics = compute_citation_metrics(
        record,
        {"L41_2024.A50", "ND115_2015.A28"},
        registry,
    )

    assert metrics["citation_recall"] == 0.5
    assert metrics["citation_precision"] == 0.5
    assert metrics["citation_f1"] == 0.5
    assert metrics["precision_denom"] == 2
    assert metrics["pred_parse_errors"] == ["BAD_ID"]


def test_citation_display_rate_is_item_level_with_article_to_clause_match():
    registry = load_registry()
    record = {
        "answer": (
            "[Luật BHXH 2024 (41/2024/QH15), Điều 50 khoản 1] "
            "[Luật BHXH 2024 (41/2024/QH15), Điều 51 khoản 9]"
        ),
        "citation_ids": ["L41_2024.A50.K1", "L41_2024.A50.K2", "L41_2024.A51"],
    }
    metrics = compute_citation_metrics(record, {"L41_2024.A50"}, registry)

    display = metrics["citation_display"]
    assert display["display_num"] == 2
    assert display["display_denom"] == 3
    assert display["citation_display_rate"] == 0.6667


def test_clean_answer_for_semantic_prefers_plain_answer():
    text, source = _clean_answer_for_semantic(
        {"answer": "RAW [Điều 1]", "plain_answer": "Plain answer"}
    )

    assert text == "Plain answer"
    assert source == "plain_answer"


def test_prolog_aggregate_formulas_count_failures():
    records = []
    for rec in [
        {"arm": "logic_lm_graphrag", "prolog_success": True, "n_repair_rounds": 0},
        {"arm": "logic_lm_graphrag", "prolog_success": True, "n_repair_rounds": 1},
        {"arm": "logic_lm_graphrag", "prolog_success": False, "n_repair_rounds": 2},
    ]:
        records.append(
            {
                "citation": {
                    "recall_num": 0,
                    "recall_denom": 1,
                    "precision_num": 0,
                    "precision_denom": 0,
                    "citation_recall": 0.0,
                    "citation_precision": 0.0,
                    "citation_f1": 0.0,
                    "citation_display": {
                        "display_num": 0,
                        "display_denom": 0,
                        "citation_display_rate": None,
                    },
                    "pred_parse_errors": [],
                },
                "latency": {"latency_s": 1.0},
                "bertscore": {},
                "prolog": compute_prolog_fields(rec),
            }
        )

    agg = _aggregate_arm(records)["prolog"]
    assert agg["prolog_first_try_solution_rate"] == 0.3333
    assert agg["repair_invoked_rate"] == 0.6667
    assert agg["repair_success_rate"] == 0.5


def test_experiment_parsers_use_strict_displayed_citations():
    from experiments.llm_only import _parse_citations as parse_llm_citations

    text = "Đủ điều kiện theo [Luật BHXH 2024 (41/2024/QH15), Điều 64 khoản 1]."
    expected_citation = "[Luật BHXH 2024 (41/2024/QH15), Điều 64 khoản 1]"

    assert parse_llm_citations(text) == ([expected_citation], ["L41_2024.A64.K1"])
    assert parse_llm_citations("[Điều 64 khoản 1]") == ([], [])


def test_elite_parser_uses_registry_and_legal_source_law_id():
    from experiments.logic_lm_pipelines import (
        _parse_citations_from_irac,
        _parse_citations_from_legal_sources,
    )

    text = "Rule: [Luật BHXH 2024 (41/2024/QH15), Điều 64 khoản 1]."
    assert _parse_citations_from_irac(text) == (
        ["[Luật BHXH 2024 (41/2024/QH15), Điều 64 khoản 1]"],
        ["L41_2024.A64.K1"],
    )
    assert _parse_citations_from_irac("[Điều 64 khoản 1]") == ([], [])
    assert _parse_citations_from_legal_sources(
        [
            "legal_source(source_a64_k1, law_bhxh_2024, article_64, clause_1, none, 'text')."
        ]
    ) == ["L41_2024.A64.K1"]
