import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from runtime.logic_lm.config import settings
from runtime.logic_lm.llm.client import LLMClient
from runtime.logic_lm.llm.factory import create_default_llm_client
from runtime.logic_lm.solvers.prolog_solver import PrologSolver


@dataclass
class ProgramEnvelope:
    legal_sources: List[str] = field(default_factory=list)
    rules: List[str] = field(default_factory=list)
    verify_facts: List[str] = field(default_factory=list)
    query: str = settings.EMPTY_STRING
    answer_var: str = settings.EMPTY_STRING
    answer_type: str = settings.ANSWER_TYPE_SCALAR
    predicate_inputs: Dict[str, Any] = field(default_factory=dict)
    citation_indices: List[int] = field(default_factory=list)
    question: str = settings.EMPTY_STRING
    raw_llm_output: Dict[str, Any] = field(default_factory=dict)

    @property
    def program_text(self) -> str:
        return settings.NEWLINE.join(
            [s for s in self.legal_sources if s.strip()]
            + [r for r in self.rules if r.strip()]
        )

    @property
    def raw_program_text(self) -> str:
        return settings.NEWLINE.join(
            [f for f in self.verify_facts if f.strip()]
            + [s for s in self.legal_sources if s.strip()]
            + [r for r in self.rules if r.strip()]
        )


@dataclass
class ExecutionResult:
    success: bool
    result: Any = None
    status: str = settings.EMPTY_STRING
    error: str = settings.EMPTY_STRING
    solutions: List[Dict[str, Any]] = field(default_factory=list)


def generate_and_execute(
    question: str,
    context: Any,
    llm: Optional[LLMClient] = None,
    max_repair_rounds: int = settings.DEFAULT_PIPELINE_MAX_REPAIR_ROUNDS,
    feedback: str = settings.EMPTY_STRING,
    previous_output: Optional[Mapping[str, Any]] = None,
    previous_program: str = settings.EMPTY_STRING,
) -> Tuple[ProgramEnvelope, ExecutionResult]:
    if llm is None:
        llm = create_default_llm_client()

    chunks = _chunks_for_llm(context)
    envelope, result = _attempt(
        llm,
        question,
        chunks,
        feedback=feedback,
        previous_output=dict(previous_output) if previous_output else None,
        previous_program=previous_program,
    )
    rounds = 0
    while not result.success and rounds < max_repair_rounds:
        rounds += 1
        envelope, result = _attempt(
            llm,
            question,
            chunks,
            previous_error=result.error,
            feedback=feedback,
            previous_output=envelope.raw_llm_output,
            previous_program=envelope.raw_program_text,
        )
    return envelope, result


def _attempt(
    llm: LLMClient,
    question: str,
    chunks: List[Dict[str, Any]],
    *,
    previous_error: str = settings.EMPTY_STRING,
    feedback: str = settings.EMPTY_STRING,
    previous_output: Optional[Dict[str, Any]] = None,
    previous_program: str = settings.EMPTY_STRING,
) -> Tuple[ProgramEnvelope, ExecutionResult]:
    payload: Dict[str, Any] = {
        settings.TASK_KEY: settings.TASK_LOGIC_RULE_GEN,
        settings.PAYLOAD_TRAINING_QUESTION_KEY: question,
        settings.PAYLOAD_RETRIEVED_CHUNKS_KEY: chunks,
    }
    if previous_error:
        payload[settings.PAYLOAD_PREVIOUS_ERROR_KEY] = previous_error
    if feedback:
        payload[settings.PAYLOAD_FEEDBACK_KEY] = feedback
    if previous_output is not None:
        payload[settings.PAYLOAD_PREVIOUS_OUTPUT_KEY] = previous_output
    if previous_program:
        payload[settings.PAYLOAD_PREVIOUS_PROGRAM_KEY] = previous_program

    try:
        response = llm.generate(payload)
    except Exception as exc:
        return ProgramEnvelope(question=question), ExecutionResult(
            success=False,
            status=settings.STATUS_UNABLE_TO_CONCLUDE,
            error=str(exc),
        )

    if not isinstance(response, Mapping):
        return ProgramEnvelope(question=question), ExecutionResult(
            success=False,
            status=settings.STATUS_UNABLE_TO_CONCLUDE,
            error=settings.ERROR_LLM_RESPONSE_NOT_JSON,
        )

    envelope = _absorb_response(response, question)
    if not envelope.rules or not envelope.query:
        return envelope, ExecutionResult(
            success=False,
            status=settings.STATUS_UNABLE_TO_CONCLUDE,
            error=settings.ERROR_LLM_NO_RULES_OR_QUERY,
        )
    if not envelope.citation_indices:
        return envelope, ExecutionResult(
            success=False,
            status=settings.STATUS_CITATION_REQUIRED,
            error=settings.ERROR_LLM_NO_CITATIONS,
        )

    predicate_input_error = _validate_predicate_inputs(envelope)
    if predicate_input_error:
        return envelope, ExecutionResult(
            success=False,
            status=settings.STATUS_INVALID_PROGRAM,
            error=predicate_input_error,
        )

    query_error = _validate_query_no_literals(envelope.query)
    if query_error:
        return envelope, ExecutionResult(
            success=False,
            status=settings.STATUS_INVALID_QUERY,
            error=query_error,
        )

    return envelope, _verify(envelope)


def _absorb_response(response: Mapping[str, Any], question: str) -> ProgramEnvelope:
    legal_sources = _normalise_clause_list(
        response.get(settings.RESPONSE_LEGAL_SOURCES_KEY) or []
    )
    rules = _normalise_clause_list(response.get(settings.RESPONSE_RULES_KEY) or [])
    verify_facts = _normalise_clause_list(
        response.get(settings.RESPONSE_VERIFY_FACTS_KEY) or []
    )
    predicate_inputs = _normalise_predicate_inputs(
        response.get(settings.RESPONSE_PREDICATE_INPUTS_KEY) or {}
    )
    citations_raw = response.get(settings.RESPONSE_CITATIONS_KEY) or []
    citation_indices = (
        [int(i) for i in citations_raw if isinstance(i, (int, float)) and int(i) >= 0]
        if isinstance(citations_raw, list)
        else []
    )
    return ProgramEnvelope(
        legal_sources=legal_sources,
        rules=rules,
        verify_facts=verify_facts,
        query=_normalise_query(
            str(response.get(settings.RESPONSE_QUERY_KEY) or settings.EMPTY_STRING)
        ),
        answer_var=str(
            response.get(settings.RESPONSE_ANSWER_VAR_KEY) or settings.EMPTY_STRING
        ).strip(),
        answer_type=str(
            response.get(settings.RESPONSE_ANSWER_TYPE_KEY)
            or settings.ANSWER_TYPE_SCALAR
        ).lower().strip()
        or settings.ANSWER_TYPE_SCALAR,
        predicate_inputs=predicate_inputs,
        citation_indices=citation_indices,
        question=question,
        raw_llm_output=dict(response),
    )


def _normalise_predicate_inputs(raw: Any) -> Dict[str, Any]:
    predicate_inputs: Dict[str, Any] = {}
    if not isinstance(raw, Mapping):
        return predicate_inputs
    for key, value in raw.items():
        slot_id = str(key).strip()
        if not slot_id:
            continue
        predicate_inputs[slot_id] = (
            dict(value) if isinstance(value, Mapping) else str(value).strip()
        )
    return predicate_inputs


def _normalise_clause_list(raw: Any) -> List[str]:
    if isinstance(raw, str):
        items = [raw]
    else:
        items = [str(item) for item in raw if str(item).strip()]
    text = settings.NEWLINE.join(item.strip() for item in items if item.strip())
    clauses = _split_prolog_clauses(text)
    if clauses:
        return clauses
    stripped = text.strip()
    if not stripped:
        return []
    return [
        stripped
        if stripped.endswith(settings.PERIOD)
        else stripped + settings.PERIOD
    ]


def _split_prolog_clauses(text: str) -> List[str]:
    clauses: List[str] = []
    start = 0
    quote = settings.EMPTY_STRING
    escaped = False
    for index, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == settings.PROLOG_ESCAPE_CHAR:
                escaped = True
            elif char == quote:
                quote = settings.EMPTY_STRING
            continue
        if char in settings.PROLOG_QUOTE_CHARS:
            quote = char
            continue
        if char != settings.PERIOD:
            continue
        next_char = (
            text[index + 1]
            if index + 1 < len(text)
            else settings.EMPTY_STRING
        )
        prev_char = text[index - 1] if index > 0 else settings.EMPTY_STRING
        if prev_char.isdigit() and next_char.isdigit():
            continue
        if next_char and not next_char.isspace():
            continue
        clause = text[start : index + 1].strip()
        if clause:
            clauses.append(clause)
        start = index + 1
    tail = text[start:].strip()
    if tail:
        clauses.append(
            tail
            if tail.endswith(settings.PERIOD)
            else tail + settings.PERIOD
        )
    return clauses


_FACT_HEAD = re.compile(settings.REGEX_FACT_HEAD, re.UNICODE)


def _validate_predicate_inputs(envelope: ProgramEnvelope) -> str:
    if not envelope.verify_facts:
        if envelope.predicate_inputs:
            return settings.ERROR_PREDICATE_INPUTS_WITHOUT_VERIFY_FACTS
        return settings.EMPTY_STRING
    if not envelope.predicate_inputs:
        return settings.ERROR_VERIFY_FACTS_WITHOUT_PREDICATE_INPUTS

    fact_predicates = _fact_predicate_names(envelope.verify_facts)
    mapped_predicates = _mapped_predicate_names(envelope.predicate_inputs)
    missing = sorted(fact_predicates - mapped_predicates)
    if missing:
        return settings.ERROR_PREDICATE_INPUTS_MISSING_TEMPLATE.format(
            names=settings.COMMA_SPACE.join(missing)
        )
    return settings.EMPTY_STRING


def _fact_predicate_names(facts: List[str]) -> set[str]:
    names: set[str] = set()
    for fact in facts:
        match = _FACT_HEAD.match(fact)
        if match:
            names.add(match.group(1))
    return names


def _mapped_predicate_names(predicate_inputs: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for slot_id, spec in predicate_inputs.items():
        default_name = str(slot_id).strip()
        name = (
            str(spec.get(settings.RESPONSE_PREDICATE_KEY) or default_name).strip()
            if isinstance(spec, Mapping)
            else str(spec or default_name).strip()
        )
        if name:
            names.add(name)
    return names


def _verify(envelope: ProgramEnvelope) -> ExecutionResult:
    program_text = envelope.raw_program_text
    if not program_text:
        return ExecutionResult(
            success=False,
            status=settings.STATUS_UNABLE_TO_CONCLUDE,
            error=settings.ERROR_EMPTY_PROLOG_SOURCE,
        )

    try:
        solutions = PrologSolver.run(program_text, envelope.query)
    except SyntaxError as exc:
        return ExecutionResult(
            success=False,
            status=settings.STATUS_SYNTAX_ERROR,
            error=str(exc),
        )
    except Exception as exc:
        msg = str(exc).lower()
        status = (
            settings.STATUS_SYNTAX_ERROR
            if settings.PROLOG_SYNTAX_ERROR_ATOM in msg
            or settings.PROLOG_SYNTAX_ERROR_TEXT.lower() in msg
            or settings.PROLOG_PARSE_ERROR_TEXT in msg
            else settings.STATUS_UNABLE_TO_CONCLUDE
        )
        return ExecutionResult(success=False, status=status, error=str(exc))

    if not solutions:
        return ExecutionResult(
            success=False,
            status=settings.STATUS_DERIVED_FALSE,
            result=False,
            error=settings.ERROR_PROLOG_NO_SOLUTIONS,
            solutions=[],
        )

    result_value = _extract_result(solutions, envelope.answer_var)
    return ExecutionResult(
        success=True,
        status=settings.STATUS_SUCCESS,
        result=result_value,
        solutions=solutions,
    )


def _extract_result(
    solutions: List[Dict[str, Any]],
    answer_var: str = settings.EMPTY_STRING,
) -> Any:
    first = solutions[0] if solutions else {}
    if not first:
        return True
    if answer_var and answer_var in first:
        return first[answer_var]
    if len(first) == 1:
        return next(iter(first.values()))
    return dict(first)


_QUERY_LITERAL_NUMBER = re.compile(settings.REGEX_QUERY_LITERAL_NUMBER)


def _validate_query_no_literals(query: str) -> str:
    body = query.strip()
    if body.startswith(settings.PROLOG_QUERY_PREFIX):
        body = body[2:].strip()
    if body.endswith(settings.PERIOD):
        body = body[:-1].strip()
    if not body:
        return settings.ERROR_EMPTY_QUERY
    if _QUERY_LITERAL_NUMBER.search(body):
        return settings.ERROR_QUERY_CONTAINS_NUMBER_TEMPLATE.format(query=query)
    if settings.SINGLE_QUOTE in body or settings.DOUBLE_QUOTE in body:
        return settings.ERROR_QUERY_CONTAINS_QUOTED_LITERAL_TEMPLATE.format(
            query=query
        )

    if settings.LEFT_PAREN in body and body.endswith(settings.RIGHT_PAREN):
        _, _, rest = body.partition(settings.LEFT_PAREN)
        args = rest[:-1].split(settings.COMMA_SPACE.strip())
        for arg in args:
            arg = arg.strip()
            if not arg:
                return settings.ERROR_EMPTY_ARGUMENT_IN_QUERY_TEMPLATE.format(
                    query=query
                )
            if (
                arg[0].isupper()
                or arg[0] == settings.PROLOG_ANONYMOUS_VARIABLE_PREFIX
                or arg == settings.USER_ATOM
            ):
                continue
            return settings.ERROR_QUERY_ARGUMENT_TEMPLATE.format(arg=arg)
    return settings.EMPTY_STRING


def _normalise_query(query: str) -> str:
    q = query.strip()
    if not q:
        return q
    if not q.startswith(settings.PROLOG_QUERY_PREFIX):
        q = settings.PROLOG_QUERY_TEMPLATE.format(
            prefix=settings.PROLOG_QUERY_PREFIX,
            query=q,
        )
    if not q.endswith(settings.PERIOD):
        q = settings.PROLOG_DOTTED_QUERY_TEMPLATE.format(
            query=q,
            period=settings.PERIOD,
        )
    return q


def _chunks_for_llm(context: Any) -> List[Dict[str, Any]]:
    raw = getattr(context, settings.FIELD_CHUNKS, None) if context is not None else None
    if not raw:
        return []
    return [
        {
            settings.FIELD_ID: getattr(
                chunk,
                settings.FIELD_ID,
                settings.EMPTY_STRING,
            ),
            settings.FIELD_DOCUMENT: getattr(
                chunk,
                settings.FIELD_DOCUMENT,
                settings.EMPTY_STRING,
            ),
            settings.FIELD_ARTICLE: getattr(
                chunk,
                settings.FIELD_ARTICLE,
                settings.EMPTY_STRING,
            ),
            settings.FIELD_CLAUSE: getattr(
                chunk,
                settings.FIELD_CLAUSE,
                settings.EMPTY_STRING,
            ),
            settings.FIELD_POINT: getattr(
                chunk,
                settings.FIELD_POINT,
                settings.EMPTY_STRING,
            ),
            settings.FIELD_TEXT: (
                getattr(chunk, settings.FIELD_TEXT, settings.EMPTY_STRING)
                or settings.EMPTY_STRING
            )[: settings.DEFAULT_LLM_CHUNK_TEXT_LIMIT],
        }
        for chunk in raw[: settings.DEFAULT_LLM_CHUNK_LIMIT]
    ]


__all__ = settings.PROGRAM_PIPELINE_PUBLIC_API
