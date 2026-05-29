"""Single source of truth for all system prompts.

Default location: the ``prompts/`` directory at the repo root. Each consumer
loads its prompt by a stable relative path under that directory, e.g.::

    from src.prompts import load_prompt
    SYSTEM_PROMPT = load_prompt("runtime/graphrag_system.md")

Canonical layout::

    prompts/
        offline/
            llm_extract.md                 # B3 LLM extraction (offline)
        runtime/
            graphrag_system.md             # GraphRAG generator system prompt
            llm_only_system.md             # LLM-only baseline system prompt
            logic_lm/
                rule_gen.md                # default Prolog generator
                rule_gen_no_retrieval.md   # no-retrieval ablation variant
                irac_render.md             # default IRAC renderer
                irac_with_plain.md         # IRAC + plain_answer renderer

External override for experiments
---------------------------------

Set the environment variable ``LEGAL_KG_PROMPTS_DIR`` to an alternate
directory. For each ``load_prompt(rel)`` call, the override directory is
checked first; if ``<override>/<rel>`` exists it is used, otherwise the
default ``prompts/<rel>`` is used. This lets experiments swap individual
prompts without touching the canonical files.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

DEFAULT_PROMPTS_DIR = Path("prompts")
OVERRIDE_DIR_ENV = "LEGAL_KG_PROMPTS_DIR"


def _override_dir() -> Path | None:
    raw = (os.environ.get(OVERRIDE_DIR_ENV) or "").strip()
    return Path(raw) if raw else None


def resolve_prompt_path(rel_path: str) -> Path:
    """Resolve the on-disk path for ``rel_path`` under prompts/.

    Override dir wins when the file exists there; otherwise the default
    ``prompts/`` directory is used. The returned path is not guaranteed to
    exist when neither location has the file — call ``load_prompt`` to get
    a hard failure with a clear message instead.
    """
    override = _override_dir()
    if override is not None:
        candidate = override / rel_path
        if candidate.exists():
            return candidate
    return DEFAULT_PROMPTS_DIR / rel_path


@lru_cache(maxsize=None)
def load_prompt(rel_path: str) -> str:
    """Read and return the prompt at ``prompts/<rel_path>`` (or its override)."""
    path = resolve_prompt_path(rel_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt not found: {rel_path}. "
            f"Looked in override dir ({_override_dir() or 'unset'}) and "
            f"default {DEFAULT_PROMPTS_DIR}/."
        )
    return path.read_text(encoding="utf-8")
