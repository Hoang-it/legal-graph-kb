"""Wrap elite/ pipeline thành 3 arm-aware classes có cùng interface với
`src/rag_query.py:RagPipeline` (`ask()` returns dataclass) để
`experiments/run_inference.py` route đồng nhất.

3 arms:
- EliteNoRetrievalPipeline   — Arm C — empty context + relaxed prompt
- EliteOntologyPipeline      — Arm D — OntologyRetrieval (elite_ontology_2024.json)
- EliteGraphRAGPipeline      — Arm E — wrap RagPipeline qua adapter

Provenance/tracking:
- n_repair_rounds: số lần `_attempt` được gọi sau lần đầu (0 = first-try success)
- prolog_status: success | derived_false | syntax_error | unable_to_conclude | ...
- prolog_success: True nếu execution_result.success VÀ trace có thực
- prompt_tokens / completion_tokens: tổng cộng cho mọi LLM call (gen ×
  (1 + n_repair_rounds) + render)

IRAC parser:
- Tách answer text thành 4 section Issue / Rule / Application / Conclusion
- Extract citations dạng [Điều X khoản Y] hoặc "Điều X, khoản Y" (fallback)
- Map → ID format `L41_2024.A<X>.K<Y>[.<z>]`

KHÔNG sửa code trong `elite/` — chỉ import + wrap.
"""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from dotenv import load_dotenv

load_dotenv()
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

# Make elite importable BEFORE importing any elite module
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ELITE_ROOT = _REPO_ROOT / "elite"
for _p in (_REPO_ROOT, _ELITE_ROOT, _ELITE_ROOT / "src"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from config import settings as elite_settings  # type: ignore
from knowledge.hybrid_retrieval import (  # type: ignore
    RetrievedKnowledgeContext,
)
from knowledge.ontology_retrieval import OntologyRetrieval  # type: ignore
from llm.client import LLMClient, OpenAILLMClient  # type: ignore
from pipelines.program_pipeline import (  # type: ignore
    ProgramEnvelope,
    ExecutionResult,
    _absorb_response,
    _attempt,
    _chunks_for_llm,
    _validate_predicate_inputs,
    _validate_query_no_literals,
    _verify,
)
from solvers.prolog_solver import PrologSolver  # type: ignore  # noqa: F401


ELITE_ONTOLOGY_PATH = Path("data/eval/elite_ontology_2024.json")
NO_RETRIEVAL_PROMPT_PATH = Path("experiments/prompts/elite_no_retrieval.md")
IRAC_WITH_PLAIN_PROMPT_PATH = Path("experiments/prompts/irac_with_plain.md")


# ---------------------------------------------------------------------------
# Output dataclass — tương đương RagAnswer cho elite arms
# ---------------------------------------------------------------------------

@dataclass
class EliteAnswer:
    question: str
    answer: str = ""                # IRAC rendered text
    plain_answer: str = ""          # NEW: prose-form answer cho fair compare với prose arms
    citations: list[str] = field(default_factory=list)       # parsed [Điều X khoản Y]
    citation_ids: list[str] = field(default_factory=list)     # L41_2024.A<n>.K<m> format
    citation_indices: list[int] = field(default_factory=list)  # raw from envelope (chunk indices)
    prolog_status: str = ""
    prolog_success: bool = False
    n_repair_rounds: int = 0
    prolog_trace: str = ""
    irac_sections: dict[str, str] = field(default_factory=dict)
    elapsed_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None     # if pipeline raised


# ---------------------------------------------------------------------------
# Citation parser — IRAC text → L41_2024 IDs
# ---------------------------------------------------------------------------

# Pattern 1: [Điều X khoản Y điểm z]  (bracketed như main src/)
_CIT_BRACKET = re.compile(
    r"\[Điều\s+(\d+)(?:\s+khoản\s+(\d+))?(?:\s+điểm\s+([a-zđ]))?\]"
)
# Pattern 2: "Điều X khoản Y" without brackets (IRAC text natural Vietnamese)
_CIT_INLINE = re.compile(
    r"Điều\s+(\d+)(?:[,\s]+[Kk]ho[ảa]n\s+(\d+))?(?:[,\s]+[ĐđDd]i[ểe]m\s+([a-zđ]))?"
)


def _parse_citations_from_irac(text: str) -> tuple[list[str], list[str]]:
    """Return (citation_strings, citation_ids).

    Try bracketed first; fall back to inline pattern (deduped against
    bracketed positions to avoid double-counting).
    """
    if not text:
        return [], []

    cites: list[tuple[str, str]] = []  # (display, id)
    consumed_spans: list[tuple[int, int]] = []

    for m in _CIT_BRACKET.finditer(text):
        art, cl, pt = m.group(1), m.group(2), m.group(3)
        cid = f"L41_2024.A{art}"
        if cl:
            cid += f".K{cl}"
            if pt:
                cid += f".{pt}"
        cites.append((m.group(0), cid))
        consumed_spans.append((m.start(), m.end()))

    for m in _CIT_INLINE.finditer(text):
        # Skip if overlaps with a bracketed match
        if any(not (m.end() <= s or m.start() >= e) for s, e in consumed_spans):
            continue
        art, cl, pt = m.group(1), m.group(2), m.group(3)
        cid = f"L41_2024.A{art}"
        if cl:
            cid += f".K{cl}"
            if pt:
                cid += f".{pt}"
        display = f"[Điều {art}" + (f" khoản {cl}" if cl else "") + (f" điểm {pt}" if pt else "") + "]"
        cites.append((display, cid))

    # Dedup giữ thứ tự
    seen_ids = set()
    cs, ids = [], []
    for display, cid in cites:
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        cs.append(display)
        ids.append(cid)
    return cs, ids


def _parse_citations_from_legal_sources(legal_sources: list[str]) -> list[str]:
    """Fallback: extract article/clause from Prolog legal_source(...) facts.

    Pattern: legal_source(_, _, article_<N>, clause_<M>, ...)
    """
    pat = re.compile(
        r"article_(\d+)(?:[\s,]+clause_(\d+))?",
        re.IGNORECASE,
    )
    ids = []
    seen = set()
    for src in legal_sources:
        for m in pat.finditer(src):
            art = m.group(1)
            cl = m.group(2)
            cid = f"L41_2024.A{art}"
            if cl:
                cid += f".K{cl}"
            if cid not in seen:
                seen.add(cid)
                ids.append(cid)
    return ids


# ---------------------------------------------------------------------------
# IRAC section parser
# ---------------------------------------------------------------------------

_IRAC_LABELS = ("Issue", "Rule", "Application", "Conclusion")


def _parse_irac_sections(text: str) -> dict[str, str]:
    if not text:
        return {}
    # Find positions of "Issue:", "Rule:", "Application:", "Conclusion:" at line starts
    positions = []
    for label in _IRAC_LABELS:
        pat = re.compile(rf"^\s*{label}\s*:", re.MULTILINE)
        m = pat.search(text)
        if m:
            positions.append((label, m.start(), m.end()))
    positions.sort(key=lambda x: x[1])

    out = {}
    for i, (label, _start, end) in enumerate(positions):
        next_start = positions[i + 1][1] if i + 1 < len(positions) else len(text)
        out[label.lower()] = text[end:next_start].strip()
    return out


# ---------------------------------------------------------------------------
# Token accounting LLM wrapper
# ---------------------------------------------------------------------------

class _TokenTrackingLLMClient(LLMClient):
    """Wrap OpenAILLMClient để cộng dồn token usage qua nhiều call.

    Cũng cho phép override system prompt cho:
    - logic_llm_rule_gen task (Arm C / no-retrieval custom prompt)
    - irac_render task (new: với prompt sinh ra cả IRAC + plain_answer)

    Reasoning-model fallback:
    - Một số model (gpt-5*, o1*, o3*, o4*) reject temperature != 1.
      Khi OpenAI trả BadRequestError mentioning "temperature", retry không truyền temperature.
      Áp dụng per-instance — nhớ flag để tránh fail lần sau.
    """

    def __init__(
        self,
        base_client: OpenAILLMClient,
        override_logic_prompt: Optional[str] = None,
        override_irac_prompt: Optional[str] = None,
    ):
        self._base = base_client
        self._override = override_logic_prompt
        self._override_irac = override_irac_prompt
        self.prompt_tokens = 0
        self.completion_tokens = 0
        # Khi True → bỏ temperature khỏi mọi call sau đó
        self._drop_temperature = False
        # Một số reasoning models cũng reject response_format json_object
        self._drop_response_format = False

    def generate(self, payload: dict) -> dict:
        # Để track tokens, gọi trực tiếp OpenAI API qua base client
        # nhưng intercept response để đếm.
        #
        # Cách đơn giản: monkey-patch task-specific prompt nếu cần,
        # rồi gọi base.generate, sau đó cộng dồn (cần response object).
        # Vì base.generate trả về dict đã parsed → mất usage info.
        # → Override _logic_llm_rule_gen + _irac_render để track.
        task = str(payload.get(elite_settings.TASK_KEY) or "")
        if task == elite_settings.TASK_LOGIC_RULE_GEN:
            return self._logic_with_tracking(payload)
        if task == elite_settings.TASK_IRAC_RENDER:
            return self._irac_with_tracking(payload)
        return self._base.generate(payload)

    def _logic_with_tracking(self, payload: dict) -> dict:
        import json
        # Tái tạo logic call từ base.OpenAILLMClient._logic_llm_rule_gen
        training_question = str(
            payload.get(elite_settings.PAYLOAD_TRAINING_QUESTION_KEY) or ""
        )
        chunks = payload.get(elite_settings.PAYLOAD_RETRIEVED_CHUNKS_KEY) or []
        previous_error = str(
            payload.get(elite_settings.PAYLOAD_PREVIOUS_ERROR_KEY) or ""
        )
        previous_output = payload.get(elite_settings.PAYLOAD_PREVIOUS_OUTPUT_KEY)
        previous_program = str(
            payload.get(elite_settings.PAYLOAD_PREVIOUS_PROGRAM_KEY) or ""
        )
        feedback = str(payload.get(elite_settings.PAYLOAD_FEEDBACK_KEY) or "")

        # Reuse base's chunk formatter via import
        from llm.client import _chunk_lines  # type: ignore

        user_parts = [
            elite_settings.LLM_TRAINING_QUESTION_LINE_TEMPLATE.format(
                training_question=training_question
            ),
            elite_settings.LLM_USER_RETRIEVED_CHUNKS_HEADER,
            elite_settings.NEWLINE.join(_chunk_lines(chunks))
            if chunks
            else elite_settings.LLM_USER_EMPTY_CHUNKS,
        ]
        if previous_error:
            user_parts.append("")
            user_parts.append(elite_settings.LLM_PREVIOUS_ATTEMPT_REPAIR_MESSAGE)
            user_parts.append(
                elite_settings.LLM_PREVIOUS_ERROR_LINE_TEMPLATE.format(
                    previous_error=previous_error
                )
            )
            user_parts.append(elite_settings.LLM_REPAIR_REQUIREMENTS)
            if previous_program:
                user_parts.append(
                    elite_settings.LLM_PREVIOUS_PROGRAM_LINE_TEMPLATE.format(
                        previous_program=previous_program
                    )
                )
            if previous_output is not None:
                user_parts.append(
                    elite_settings.LLM_PREVIOUS_OUTPUT_LINE_TEMPLATE.format(
                        previous_output=json.dumps(
                            previous_output, ensure_ascii=False
                        )
                    )
                )

        system_prompt = self._override or elite_settings.LOGIC_LLM_RULE_GEN_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": elite_settings.NEWLINE.join(user_parts)},
        ]
        resp = self._chat_with_fallback(
            messages,
            use_response_format=True,
        )
        self.prompt_tokens += resp.usage.prompt_tokens
        self.completion_tokens += resp.usage.completion_tokens
        raw = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _irac_with_tracking(self, payload: dict) -> dict:
        """Render IRAC. If `override_irac_prompt` provided, also extract
        `plain_answer` field from JSON response.

        Output dict keys:
            - text:         IRAC text (4-section format) — backward compat
            - plain_answer: NEW prose-form answer (only when override active)
        """
        import json as _json
        trace = payload.get(elite_settings.PAYLOAD_TRACE_KEY) or {}
        user_msg = _json.dumps(trace, ensure_ascii=False, indent=2, default=str)
        system_prompt = self._override_irac or elite_settings.IRAC_RENDER_PROMPT
        # If overriden prompt → expect JSON output with irac + plain_answer
        use_json = bool(self._override_irac)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        resp = self._chat_with_fallback(messages, use_response_format=use_json)
        self.prompt_tokens += resp.usage.prompt_tokens
        self.completion_tokens += resp.usage.completion_tokens
        text = (resp.choices[0].message.content or "").strip()
        if not use_json:
            return {elite_settings.RESPONSE_TEXT_KEY: text}
        # Parse JSON: tolerant fallback nếu LLM trả markdown-fenced JSON
        clean = text
        if clean.startswith("```"):
            # strip markdown fence
            clean = re.sub(r"^```(?:json)?\s*", "", clean)
            clean = re.sub(r"\s*```\s*$", "", clean)
        try:
            parsed = _json.loads(clean)
        except _json.JSONDecodeError:
            # Fallback: try regex extract for "plain_answer" + "irac"
            m_plain = re.search(r'"plain_answer"\s*:\s*"((?:[^"\\]|\\.)*)"', clean, re.DOTALL)
            m_irac = re.search(r'"irac"\s*:\s*"((?:[^"\\]|\\.)*)"', clean, re.DOTALL)
            parsed = {
                "irac": m_irac.group(1).encode().decode("unicode_escape") if m_irac else text,
                "plain_answer": m_plain.group(1).encode().decode("unicode_escape") if m_plain else "",
            }
        if not isinstance(parsed, dict):
            parsed = {"irac": text, "plain_answer": ""}
        return {
            elite_settings.RESPONSE_TEXT_KEY: (parsed.get("irac") or "").strip(),
            "plain_answer": (parsed.get("plain_answer") or "").strip(),
        }

    # -----------------------------------------------------------------
    # Real-API call with adaptive param fallback (reasoning models)
    # -----------------------------------------------------------------
    def _chat_with_fallback(self, messages: list, use_response_format: bool):
        """Gọi OpenAI chat completion. Nếu API reject `temperature` (gpt-5*,
        o-series reasoning models enforce temperature=1.0), bỏ param và retry.
        Tương tự với response_format=json_object.

        Đây KHÔNG phải workaround mock — chỉ là tôn trọng constraint thật của
        từng model. Vẫn gọi API thật, vẫn lấy response thật.
        """
        from openai import BadRequestError

        def _do_call(drop_temp: bool, drop_rf: bool):
            kwargs: dict = {
                "model": self._base._model,
                "messages": messages,
            }
            if not drop_temp:
                kwargs["temperature"] = elite_settings.DETERMINISTIC_LLM_TEMPERATURE
            if use_response_format and not drop_rf:
                kwargs["response_format"] = {"type": "json_object"}
            return self._base._client.chat.completions.create(**kwargs)

        try:
            return _do_call(
                drop_temp=self._drop_temperature,
                drop_rf=self._drop_response_format,
            )
        except BadRequestError as e:
            msg = str(e).lower()
            changed = False
            if "temperature" in msg and not self._drop_temperature:
                self._drop_temperature = True
                changed = True
            if (
                use_response_format
                and ("response_format" in msg or "json_object" in msg)
                and not self._drop_response_format
            ):
                self._drop_response_format = True
                changed = True
            if not changed:
                raise
            # Retry once với param đã loại bỏ
            return _do_call(
                drop_temp=self._drop_temperature,
                drop_rf=self._drop_response_format,
            )


# ---------------------------------------------------------------------------
# Repair counter
# ---------------------------------------------------------------------------

def _generate_and_execute_with_counter(
    llm: LLMClient,
    question: str,
    context: Any,
    max_repair_rounds: int = 2,
) -> tuple[ProgramEnvelope, ExecutionResult, int]:
    """Tái triển khai `generate_and_execute` để expose số repair rounds.

    Trả về (envelope, result, n_repair_rounds). n_repair_rounds = 0 nếu
    success ngay first try.
    """
    chunks = _chunks_for_llm(context)
    envelope, result = _attempt(
        llm,
        question,
        chunks,
        feedback="",
        previous_output=None,
        previous_program="",
    )
    rounds = 0
    while not result.success and rounds < max_repair_rounds:
        rounds += 1
        envelope, result = _attempt(
            llm,
            question,
            chunks,
            previous_error=result.error,
            feedback="",
            previous_output=envelope.raw_llm_output,
            previous_program=envelope.raw_program_text,
        )
    return envelope, result, rounds


# ---------------------------------------------------------------------------
# Variant cho Arm C — bypass citation_indices check (no retrieval = no chunks)
# ---------------------------------------------------------------------------

def _attempt_no_citation_check(
    llm: LLMClient,
    question: str,
    chunks: list,
    *,
    previous_error: str = "",
    feedback: str = "",
    previous_output: Optional[dict] = None,
    previous_program: str = "",
) -> tuple[ProgramEnvelope, ExecutionResult]:
    """Phiên bản `_attempt` cho Arm C — SKIP citation_indices check.

    Logic giống `pipelines.program_pipeline._attempt` (elite) nhưng bỏ
    qua step `STATUS_CITATION_REQUIRED` vì Arm C cố tình không có chunks.
    """
    payload: dict = {
        elite_settings.TASK_KEY: elite_settings.TASK_LOGIC_RULE_GEN,
        elite_settings.PAYLOAD_TRAINING_QUESTION_KEY: question,
        elite_settings.PAYLOAD_RETRIEVED_CHUNKS_KEY: chunks,
    }
    if previous_error:
        payload[elite_settings.PAYLOAD_PREVIOUS_ERROR_KEY] = previous_error
    if feedback:
        payload[elite_settings.PAYLOAD_FEEDBACK_KEY] = feedback
    if previous_output is not None:
        payload[elite_settings.PAYLOAD_PREVIOUS_OUTPUT_KEY] = previous_output
    if previous_program:
        payload[elite_settings.PAYLOAD_PREVIOUS_PROGRAM_KEY] = previous_program

    try:
        response = llm.generate(payload)
    except Exception as exc:
        return ProgramEnvelope(question=question), ExecutionResult(
            success=False,
            status=elite_settings.STATUS_UNABLE_TO_CONCLUDE,
            error=str(exc),
        )

    if not isinstance(response, Mapping):
        return ProgramEnvelope(question=question), ExecutionResult(
            success=False,
            status=elite_settings.STATUS_UNABLE_TO_CONCLUDE,
            error=elite_settings.ERROR_LLM_RESPONSE_NOT_JSON,
        )

    envelope = _absorb_response(response, question)
    if not envelope.rules or not envelope.query:
        return envelope, ExecutionResult(
            success=False,
            status=elite_settings.STATUS_UNABLE_TO_CONCLUDE,
            error=elite_settings.ERROR_LLM_NO_RULES_OR_QUERY,
        )

    # ↓↓↓ SKIP citation_indices check (Arm C-specific) ↓↓↓

    predicate_input_error = _validate_predicate_inputs(envelope)
    if predicate_input_error:
        return envelope, ExecutionResult(
            success=False,
            status=elite_settings.STATUS_INVALID_PROGRAM,
            error=predicate_input_error,
        )

    query_error = _validate_query_no_literals(envelope.query)
    if query_error:
        return envelope, ExecutionResult(
            success=False,
            status=elite_settings.STATUS_INVALID_QUERY,
            error=query_error,
        )

    return envelope, _verify(envelope)


def _generate_and_execute_no_citation_check(
    llm: LLMClient,
    question: str,
    context: Any,
    max_repair_rounds: int = 2,
) -> tuple[ProgramEnvelope, ExecutionResult, int]:
    """Same loop pattern nhưng dùng `_attempt_no_citation_check`."""
    chunks = _chunks_for_llm(context)
    envelope, result = _attempt_no_citation_check(
        llm, question, chunks,
        feedback="", previous_output=None, previous_program="",
    )
    rounds = 0
    while not result.success and rounds < max_repair_rounds:
        rounds += 1
        envelope, result = _attempt_no_citation_check(
            llm, question, chunks,
            previous_error=result.error,
            feedback="",
            previous_output=envelope.raw_llm_output,
            previous_program=envelope.raw_program_text,
        )
    return envelope, result, rounds


# ---------------------------------------------------------------------------
# Pipeline base + concrete classes
# ---------------------------------------------------------------------------

class _EliteBasePipeline:
    """Base for 3 elite arms. Subclass provides `retriever` and optional
    `prompt_override` (Arm C only)."""

    arm_name: str = "elite"
    # Set True trong subclass (Arm C) để bypass citation_indices check
    skip_citation_check: bool = False

    def __init__(
        self,
        retriever: Optional[Any] = None,
        prompt_override: Optional[str] = None,
        irac_prompt_override: Optional[str] = None,
        max_repair_rounds: int = 2,
        top_k: int = 8,
        model: Optional[str] = None,
        enable_plain_answer: bool = True,
    ):
        self.retriever = retriever
        self.prompt_override = prompt_override
        # IRAC prompt override: nếu None và enable_plain_answer=True, dùng
        # IRAC_WITH_PLAIN_PROMPT_PATH (output JSON với irac + plain_answer)
        if irac_prompt_override is None and enable_plain_answer:
            if IRAC_WITH_PLAIN_PROMPT_PATH.exists():
                irac_prompt_override = IRAC_WITH_PLAIN_PROMPT_PATH.read_text(encoding="utf-8")
        self.irac_prompt_override = irac_prompt_override
        self.max_repair_rounds = max_repair_rounds
        self.top_k = top_k
        # None → dùng default của elite (gpt-4o-mini)
        self.model = model

    def _empty_context(self) -> RetrievedKnowledgeContext:
        return RetrievedKnowledgeContext(chunks=[], scores={})

    def _make_llm(self) -> _TokenTrackingLLMClient:
        if self.model:
            base = OpenAILLMClient(model=self.model)
        else:
            base = OpenAILLMClient()
        return _TokenTrackingLLMClient(
            base_client=base,
            override_logic_prompt=self.prompt_override,
            override_irac_prompt=self.irac_prompt_override,
        )

    def ask(self, question: str) -> EliteAnswer:
        t0 = time.time()
        try:
            # 1. Retrieve
            if self.retriever is not None:
                context = self.retriever.retrieve(question, top_k=self.top_k)
            else:
                context = self._empty_context()

            # 2. LLM gen Prolog + execute (with repair)
            llm = self._make_llm()
            executor = (
                _generate_and_execute_no_citation_check
                if self.skip_citation_check
                else _generate_and_execute_with_counter
            )
            envelope, result, n_rounds = executor(
                llm, question, context, max_repair_rounds=self.max_repair_rounds
            )

            prolog_status = result.status or ""
            prolog_success = bool(result.success)

            # 3. Find Trace binding
            trace_value = self._extract_trace(result.solutions)

            irac_text = ""
            plain_answer = ""  # NEW field
            irac_sections: dict[str, str] = {}

            # 4. Render IRAC (only if prolog success + trace exists)
            if prolog_success and trace_value is not None:
                render_payload = self._build_render_payload(
                    question, context, envelope, result
                )
                render_resp = llm.generate(
                    {
                        elite_settings.TASK_KEY: elite_settings.TASK_IRAC_RENDER,
                        elite_settings.PAYLOAD_TRACE_KEY: render_payload,
                    }
                )
                irac_text = str(
                    render_resp.get(elite_settings.RESPONSE_TEXT_KEY)
                    or render_resp.get(elite_settings.RESPONSE_ANSWER_KEY)
                    or ""
                ).strip()
                plain_answer = str(render_resp.get("plain_answer") or "").strip()
                irac_sections = _parse_irac_sections(irac_text)

            # 5. Citations: parse từ IRAC text + fallback từ legal_sources
            citations, citation_ids = _parse_citations_from_irac(irac_text)
            if not citation_ids:
                # Fallback: extract từ legal_sources Prolog facts
                fallback_ids = _parse_citations_from_legal_sources(
                    list(envelope.legal_sources)
                )
                citation_ids = fallback_ids
                # Generate display strings
                for cid in fallback_ids:
                    m = re.match(r"L41_2024\.A(\d+)(?:\.K(\d+))?", cid)
                    if m:
                        art, cl = m.group(1), m.group(2)
                        display = f"[Điều {art}" + (f" khoản {cl}" if cl else "") + "]"
                        if display not in citations:
                            citations.append(display)

            elapsed = time.time() - t0
            return EliteAnswer(
                question=question,
                answer=irac_text or self._failure_message(result, n_rounds),
                plain_answer=plain_answer,
                citations=citations,
                citation_ids=citation_ids,
                citation_indices=list(envelope.citation_indices),
                prolog_status=prolog_status,
                prolog_success=prolog_success,
                n_repair_rounds=n_rounds,
                prolog_trace=str(trace_value) if trace_value is not None else "",
                irac_sections=irac_sections,
                elapsed_s=round(elapsed, 3),
                prompt_tokens=llm.prompt_tokens,
                completion_tokens=llm.completion_tokens,
            )

        except Exception as e:
            elapsed = time.time() - t0
            return EliteAnswer(
                question=question,
                answer="",
                elapsed_s=round(elapsed, 3),
                error=f"{type(e).__name__}: {e}",
                prolog_status="exception",
            )

    @staticmethod
    def _extract_trace(solutions: list[Mapping[str, Any]]) -> Any:
        for sol in solutions or []:
            for k, v in sol.items():
                if str(k).lower() == elite_settings.TRACE_VARIABLE_NAME.lower():
                    return v
        return None

    @staticmethod
    def _failure_message(result: ExecutionResult, n_rounds: int) -> str:
        return (
            f"[Pipeline không trả về kết luận. "
            f"prolog_status={result.status}, "
            f"repair_rounds={n_rounds}, "
            f"error={result.error[:200] if result.error else 'N/A'}]"
        )

    def _build_render_payload(
        self,
        question: str,
        context: Any,
        envelope: ProgramEnvelope,
        result: ExecutionResult,
    ) -> dict:
        """Subset của _render_payload trong elite/cli/answer_with_program.py:427.

        Trả về dict đủ field IRAC_RENDER_PROMPT cần. Chỉ giữ field thiết
        yếu cho rendering — full payload không cần thiết.
        """
        chunks = list(getattr(context, "chunks", []) or [])
        citations = []
        for i in (envelope.citation_indices or [])[:5]:
            if 0 <= i < len(chunks):
                c = chunks[i]
                citations.append(
                    {
                        "index": i,
                        "chunk_id": getattr(c, "id", ""),
                        "document": getattr(c, "document", None),
                        "article": getattr(c, "article", None),
                        "clause": getattr(c, "clause", None),
                        "point": getattr(c, "point", None),
                        "raw_text": (getattr(c, "text", "") or "")[:400],
                    }
                )
        return {
            "normalized_question": question,
            "legal_issue": question,
            "domain_context": elite_settings.DOMAIN_CONTEXT,
            "selected_function": {
                "name": elite_settings.SELECTED_FUNCTION_NAME,
                "description": elite_settings.SELECTED_FUNCTION_DESCRIPTION,
            },
            "slot_bindings": {},
            "verify_facts": list(envelope.verify_facts or []),
            "citations": citations,
            "execution_result": list(result.solutions or []),
            "prolog_trace": self._extract_trace(result.solutions or []),
            "generated_program": {
                "legal_sources": list(envelope.legal_sources or []),
                "rules": list(envelope.rules or []),
                "query": envelope.query,
                "answer_var": envelope.answer_var,
                "answer_type": envelope.answer_type,
            },
        }

    def close(self):
        pass  # nothing to close at this layer


class EliteNoRetrievalPipeline(_EliteBasePipeline):
    """Arm C: empty context + relaxed prompt + bypass citation_indices check
    (vì không có chunks nên LLM trả citations=[] hợp lệ)."""

    arm_name = "elite_no_retrieval"
    skip_citation_check = True

    def __init__(self, **kwargs):
        prompt = NO_RETRIEVAL_PROMPT_PATH.read_text(encoding="utf-8")
        super().__init__(
            retriever=None,
            prompt_override=prompt,
            **kwargs,
        )


class EliteOntologyPipeline(_EliteBasePipeline):
    """Arm D: elite's OntologyRetrieval với Luật 2024 ontology."""

    arm_name = "elite_ontology"

    def __init__(self, **kwargs):
        if not ELITE_ONTOLOGY_PATH.exists():
            raise FileNotFoundError(
                f"Ontology not found: {ELITE_ONTOLOGY_PATH}. "
                f"Run `python -m experiments.build_elite_corpus_2024` first."
            )
        retriever = OntologyRetrieval()
        retriever.index_ontology(ELITE_ONTOLOGY_PATH)
        super().__init__(retriever=retriever, **kwargs)


class EliteGraphRAGPipeline(_EliteBasePipeline):
    """Arm E: GraphRAG (Neo4j vector) làm retriever cho elite."""

    arm_name = "elite_graphrag"

    def __init__(self, rag_pipeline=None, **kwargs):
        from experiments.graphrag_retriever_adapter import GraphRAGAsEliteRetriever
        if rag_pipeline is None:
            from src.rag_query import RagPipeline
            rag_pipeline = RagPipeline()
            _ = rag_pipeline.embed_model  # warm up
        self._owned_rag = rag_pipeline  # so close() can dispose
        adapter = GraphRAGAsEliteRetriever(rag_pipeline)
        super().__init__(retriever=adapter, **kwargs)

    def close(self):
        try:
            self._owned_rag.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Smoke test: chạy 1 câu hỏi qua mỗi arm
    test_question = "Người sử dụng lao động có trách nhiệm gì về bảo hiểm xã hội?"
    print(f"Q: {test_question}\n")

    for cls in (EliteNoRetrievalPipeline, EliteOntologyPipeline, EliteGraphRAGPipeline):
        print(f"--- {cls.arm_name} ---")
        try:
            p = cls()
            ans = p.ask(test_question)
            print(f"  prolog_success: {ans.prolog_success}")
            print(f"  n_repair_rounds: {ans.n_repair_rounds}")
            print(f"  prolog_status: {ans.prolog_status}")
            print(f"  citations: {ans.citations}")
            print(f"  citation_ids: {ans.citation_ids}")
            print(f"  elapsed_s: {ans.elapsed_s}")
            print(f"  tokens: {ans.prompt_tokens}+{ans.completion_tokens}")
            print(f"  answer (200ch): {ans.answer[:200]}")
            if ans.error:
                print(f"  ERROR: {ans.error}")
            p.close()
        except Exception as e:
            print(f"  EXCEPTION: {type(e).__name__}: {e}")
        print()
