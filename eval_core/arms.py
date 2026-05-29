"""Shared experiment arm definitions.

`main` is the approved comparison set for the academic-metrics phase. It is
intentionally narrower than `all`: experimental arms can exist in the repo
without entering the headline evaluation.
"""

from __future__ import annotations

ALL_ARMS = (
    "graphrag",
    "llm_only",
    "logic_lm_no_retrieval",
    "logic_lm_ontology",
    "logic_lm_graphrag",
    "graphrag_v5",
)

MAIN_EXPERIMENT_ARMS = (
    "graphrag",
    "llm_only",
    "logic_lm_no_retrieval",
    "logic_lm_ontology",
    "logic_lm_graphrag",
    "graphrag_v5",
)


def parse_run_arms(raw: str) -> list[str]:
    """Parse arm selection for inference runs."""
    if raw == "main":
        return list(MAIN_EXPERIMENT_ARMS)
    if raw == "all":
        return list(ALL_ARMS)
    arms = [arm.strip() for arm in raw.split(",") if arm.strip()]
    invalid = [arm for arm in arms if arm not in ALL_ARMS]
    if invalid:
        valid = ", ".join(ALL_ARMS)
        raise ValueError(f"Unknown arm(s): {invalid}. Valid: {valid}, 'main', or 'all'")
    return arms


def parse_metrics_arms(raw: str | None) -> list[str] | None:
    """Parse arm selection for academic metrics.

    Return None for `all`, which means "discover every result directory".
    """
    if raw is None or not raw.strip() or raw == "main":
        return list(MAIN_EXPERIMENT_ARMS)
    if raw == "all":
        return None
    return parse_run_arms(raw)

