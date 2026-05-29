"""Default prompt texts for the logic-LM runtime.

Texts are loaded from the canonical ``prompts/runtime/logic_lm/`` directory
via the shared loader in :mod:`src.prompts`, which honours the
``LEGAL_KG_PROMPTS_DIR`` env var for external overrides. The names exported
here are kept for backwards compatibility with code that imports them
through ``runtime.logic_lm.config.settings``.
"""

from src.prompts import load_prompt

IRAC_RENDER_PROMPT = load_prompt("runtime/logic_lm/irac_render.md")

LOGIC_LLM_RULE_GEN_PROMPT = load_prompt("runtime/logic_lm/rule_gen.md")
