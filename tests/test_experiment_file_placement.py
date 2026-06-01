from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_experiment_specific_producers_do_not_live_under_scripts():
    offenders = sorted((REPO_ROOT / "scripts").glob("exp[0-9]*.py"))
    assert not offenders, (
        "Experiment-specific producers must live under experiments/<NN>/ "
        f"(for example run_retrieval.py), not scripts/: {offenders}"
    )
