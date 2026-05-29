import os

from runtime.logic_lm.config import settings
from runtime.logic_lm.llm.client import LLMClient, OpenAILLMClient


def create_default_llm_client() -> LLMClient:
    model = os.environ.get(
        settings.OPENAI_MODEL_ENV_VAR,
        settings.DEFAULT_OPENAI_MODEL,
    )
    return OpenAILLMClient(model=model)
