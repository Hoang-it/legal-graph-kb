import json
import re
from typing import Any, Dict, List

from runtime.logic_lm.config import settings


class LLMClient:
    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError()


class OpenAILLMClient(LLMClient):
    def __init__(
        self,
        model: str = settings.DEFAULT_OPENAI_MODEL,
        temperature: float = settings.DEFAULT_OPENAI_TEMPERATURE,
    ) -> None:
        import os
        from openai import OpenAI

        api_key = os.environ.get(settings.OPENAI_API_KEY_ENV_VAR)
        if not api_key:
            raise EnvironmentError(
                settings.ERROR_OPENAI_API_KEY_MISSING_TEMPLATE.format(
                    env_var=settings.OPENAI_API_KEY_ENV_VAR
                )
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        task = str(payload.get(settings.TASK_KEY) or settings.EMPTY_STRING)
        if task == settings.TASK_LOGIC_RULE_GEN:
            return self._logic_llm_rule_gen(payload)
        if task == settings.TASK_IRAC_RENDER:
            return self._irac_render(payload)
        raise ValueError(
            settings.ERROR_UNSUPPORTED_LLM_TASK_TEMPLATE.format(task=task)
        )

    def _irac_render(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        trace = payload.get(settings.PAYLOAD_TRACE_KEY) or {}
        user_msg = json.dumps(trace, ensure_ascii=False, indent=2, default=str)
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=settings.DETERMINISTIC_LLM_TEMPERATURE,
            messages=[
                {
                    settings.OPENAI_ROLE_KEY: settings.OPENAI_SYSTEM_ROLE,
                    settings.OPENAI_CONTENT_KEY: settings.IRAC_RENDER_PROMPT,
                },
                {
                    settings.OPENAI_ROLE_KEY: settings.OPENAI_USER_ROLE,
                    settings.OPENAI_CONTENT_KEY: user_msg,
                },
            ],
        )
        text = response.choices[0].message.content or settings.EMPTY_STRING
        return {settings.RESPONSE_TEXT_KEY: text.strip()}

    def _logic_llm_rule_gen(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        training_question = str(
            payload.get(settings.PAYLOAD_TRAINING_QUESTION_KEY)
            or settings.EMPTY_STRING
        )
        chunks = payload.get(settings.PAYLOAD_RETRIEVED_CHUNKS_KEY) or []
        previous_error = str(
            payload.get(settings.PAYLOAD_PREVIOUS_ERROR_KEY)
            or settings.EMPTY_STRING
        )
        previous_output = payload.get(settings.PAYLOAD_PREVIOUS_OUTPUT_KEY)
        previous_program = str(
            payload.get(settings.PAYLOAD_PREVIOUS_PROGRAM_KEY)
            or settings.EMPTY_STRING
        )
        feedback = str(
            payload.get(settings.PAYLOAD_FEEDBACK_KEY) or settings.EMPTY_STRING
        )

        user_parts = [
            settings.LLM_TRAINING_QUESTION_LINE_TEMPLATE.format(
                training_question=training_question
            ),
            settings.LLM_USER_RETRIEVED_CHUNKS_HEADER,
            settings.NEWLINE.join(_chunk_lines(chunks))
            if chunks
            else settings.LLM_USER_EMPTY_CHUNKS,
        ]
        if previous_error:
            user_parts.append(settings.EMPTY_STRING)
            user_parts.append(settings.LLM_PREVIOUS_ATTEMPT_REPAIR_MESSAGE)
            user_parts.append(
                settings.LLM_PREVIOUS_ERROR_LINE_TEMPLATE.format(
                    previous_error=previous_error
                )
            )
            user_parts.append(settings.LLM_REPAIR_REQUIREMENTS)
            if previous_program:
                user_parts.append(
                    settings.LLM_PREVIOUS_PROGRAM_LINE_TEMPLATE.format(
                        previous_program=previous_program
                    )
                )
            if previous_output is not None:
                user_parts.append(
                    settings.LLM_PREVIOUS_OUTPUT_LINE_TEMPLATE.format(
                        previous_output=json.dumps(
                            previous_output,
                            ensure_ascii=False,
                        )
                    )
                )
        elif previous_output is not None or feedback:
            user_parts.append(settings.EMPTY_STRING)
            user_parts.append(settings.LLM_EXISTING_PROGRAM_REVIEW_MESSAGE)
            if feedback:
                user_parts.append(
                    settings.LLM_HUMAN_FEEDBACK_LINE_TEMPLATE.format(
                        feedback=feedback
                    )
                )
            if previous_output is not None:
                user_parts.append(
                    settings.LLM_PREVIOUS_OUTPUT_LINE_TEMPLATE.format(
                        previous_output=json.dumps(
                            previous_output,
                            ensure_ascii=False,
                        )
                    )
                )

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=settings.DETERMINISTIC_LLM_TEMPERATURE,
            messages=[
                {
                    settings.OPENAI_ROLE_KEY: settings.OPENAI_SYSTEM_ROLE,
                    settings.OPENAI_CONTENT_KEY: settings.LOGIC_LLM_RULE_GEN_PROMPT,
                },
                {
                    settings.OPENAI_ROLE_KEY: settings.OPENAI_USER_ROLE,
                    settings.OPENAI_CONTENT_KEY: settings.NEWLINE.join(user_parts),
                },
            ],
        )
        raw = response.choices[0].message.content or settings.JSON_EMPTY_OBJECT
        return _parse_json_object(_strip_markdown(raw).strip())


def _chunk_lines(chunks: Any) -> List[str]:
    lines: List[str] = []
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            doc = chunk.get(settings.FIELD_DOCUMENT) or settings.EMPTY_STRING
            article = chunk.get(settings.FIELD_ARTICLE) or settings.EMPTY_STRING
            clause = chunk.get(settings.FIELD_CLAUSE) or settings.EMPTY_STRING
            point = chunk.get(settings.FIELD_POINT) or settings.EMPTY_STRING
            text = chunk.get(settings.FIELD_TEXT) or settings.EMPTY_STRING
            ref = settings.SPACE.join(
                p for p in (doc, article, clause, point) if p
            )
            lines.append(
                settings.LLM_CHUNK_LINE_TEMPLATE.format(
                    index=i,
                    ref=ref,
                    text=text,
                )
            )
        else:
            lines.append(
                settings.LLM_FALLBACK_CHUNK_LINE_TEMPLATE.format(
                    index=i,
                    chunk=chunk,
                )
            )
    return lines


def _strip_markdown(text: str) -> str:
    text = re.sub(
        settings.REGEX_MARKDOWN_FENCE,
        settings.MARKDOWN_FENCE_REPLACEMENT,
        text,
    )
    return text.replace(
        settings.MARKDOWN_CODE_FENCE,
        settings.MARKDOWN_FENCE_REPLACEMENT,
    )


def _parse_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(settings.REGEX_JSON_OBJECT, text, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}
