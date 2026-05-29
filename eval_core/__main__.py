"""Entry point so ``python -m eval_core <cmd>`` dispatches to :mod:`eval_core.cli`."""

from eval_core.cli import main

raise SystemExit(main())
