"""Optional judge-metric entrypoint.

Judge-based metrics are intentionally outside the main experiment.  This module
exists as a fail-closed placeholder so old judge formulas cannot be invoked by
accident under a new name.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "Judge metrics are optional and are not implemented in the current "
        "academic-metrics phase. Main evaluation is `python -m "
        "evaluation.compute_academic_metrics`.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
