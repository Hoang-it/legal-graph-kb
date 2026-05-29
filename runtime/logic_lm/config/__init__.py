"""Re-export the aggregate settings namespace.

Code can do either:
    from src.logic_lm.config import settings
or:
    from src.logic_lm.config.settings import SOME_CONST
"""

from . import settings  # noqa: F401
