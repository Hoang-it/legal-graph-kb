import os

from config import settings
from llm.client import LLMClient, OpenAILLMClient


def create_default_llm_client() -> LLMClient:
    model = os.environ.get(
        settings.OPENAI_MODEL_ENV_VAR,
        settings.DEFAULT_OPENAI_MODEL,
    )
    return OpenAILLMClient(model=model)
