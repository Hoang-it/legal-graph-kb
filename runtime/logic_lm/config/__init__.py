"""Re-export the aggregate settings namespace.

Code can do either:
    from runtime.logic_lm.config import settings
or:
    from runtime.logic_lm.config.settings import SOME_CONST
"""

from . import settings  # noqa: F401
