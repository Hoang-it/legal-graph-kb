import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.logic_lm.config import settings

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, settings.STREAM_RECONFIGURE_METHOD):
        try:
            _stream.reconfigure(encoding=settings.PATH_ENCODING)
        except Exception:
            pass


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=settings.CLI_DESCRIPTION
    )
    parser.add_argument(
        settings.CLI_ARGUMENT_QUESTION_PARTS,
        nargs=settings.CLI_NARGS_ANY,
        help=settings.CLI_HELP_QUESTION_PARTS,
    )
    parser.add_argument(
        settings.CLI_ARGUMENT_QUESTION,
        default=settings.EMPTY_STRING,
        help=settings.CLI_HELP_QUESTION,
    )
    parser.add_argument(
        settings.CLI_ARGUMENT_CORPUS,
        default=str(
            REPO_ROOT / settings.DATA_DIR_NAME / settings.LOGIC_LM_DATA_SUBDIR / settings.DEFAULT_CORPUS_FILENAME
        ),
        help=settings.CLI_HELP_CORPUS,
    )
    parser.add_argument(
        settings.CLI_ARGUMENT_ONTOLOGY,
        default=str(
            REPO_ROOT / settings.DATA_DIR_NAME / settings.LOGIC_LM_DATA_SUBDIR / settings.DEFAULT_ONTOLOGY_FILENAME
        ),
        help=settings.CLI_HELP_ONTOLOGY,
    )
    parser.add_argument(
        settings.CLI_ARGUMENT_RETRIEVAL_MODE,
        choices=(settings.RETRIEVAL_MODE_ONTOLOGY, settings.RETRIEVAL_MODE_HYBRID),
        default=settings.DEFAULT_RETRIEVAL_MODE,
        help=settings.CLI_HELP_RETRIEVAL_MODE,
    )
    parser.add_argument(
        settings.CLI_ARGUMENT_PROGRAM_DIR,
        default=str(
            REPO_ROOT
            / settings.DATA_DIR_NAME
            / settings.LOGIC_LM_DATA_SUBDIR
            / settings.GENERATED_DIR_NAME
            / settings.PROGRAM_ARTIFACTS_DIR_NAME
        ),
        help=settings.CLI_HELP_PROGRAM_DIR,
    )
    parser.add_argument(
        settings.CLI_ARGUMENT_TOP_K,
        type=int,
        default=settings.DEFAULT_RETRIEVAL_TOP_K,
        help=settings.CLI_HELP_TOP_K,
    )
    parser.add_argument(
        settings.CLI_ARGUMENT_MAX_REPAIR_ROUNDS,
        type=int,
        default=settings.DEFAULT_MAX_REPAIR_ROUNDS,
        help=settings.CLI_HELP_MAX_REPAIR_ROUNDS,
    )
    parser.add_argument(
        settings.CLI_ARGUMENT_TRACE_REPAIR_ROUNDS,
        type=int,
        default=settings.DEFAULT_TRACE_REPAIR_ROUNDS,
        help=settings.CLI_HELP_TRACE_REPAIR_ROUNDS,
    )
    parser.add_argument(
        settings.CLI_ARGUMENT_JSON,
        action=settings.CLI_ACTION_STORE_TRUE,
        help=settings.CLI_HELP_JSON,
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    from src.logic_lm.pipelines.program_pipeline import generate_and_execute
    from src.logic_lm.llm.factory import create_default_llm_client

    args = make_parser().parse_args(list(argv) if argv is not None else None)
    question = _resolve_question(args)
    if not question:
        print(settings.CLI_ERROR_QUESTION_REQUIRED, file=sys.stderr)
        return 2

    corpus_path = Path(args.corpus)
    ontology_path = Path(args.ontology)
    if args.retrieval_mode == settings.RETRIEVAL_MODE_HYBRID and not corpus_path.exists():
        print(
            settings.CLI_ERROR_CORPUS_NOT_FOUND_TEMPLATE.format(
                corpus_path=corpus_path
            ),
            file=sys.stderr,
        )
        return 2
    if (
        args.retrieval_mode == settings.RETRIEVAL_MODE_ONTOLOGY
        and not ontology_path.exists()
        and not corpus_path.exists()
    ):
        print(
            settings.CLI_ERROR_CORPUS_NOT_FOUND_TEMPLATE.format(
                corpus_path=corpus_path
            ),
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.program_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    retriever = _make_retriever(args.retrieval_mode, corpus_path, ontology_path)
    context = retriever.retrieve(question, top_k=max(1, int(args.top_k)))

    llm = create_default_llm_client()

    envelope, result = generate_and_execute(
        question,
        context,
        llm=llm,
        max_repair_rounds=max(0, int(args.max_repair_rounds)),
    )

    trace_rounds = max(0, int(args.trace_repair_rounds))
    for _ in range(trace_rounds):
        if not result.success:
            break
        if _trace_binding(result.solutions) is not None:
            break
        feedback = settings.CLI_TRACE_REPAIR_FEEDBACK
        envelope, result = generate_and_execute(
            question,
            context,
            llm=llm,
            max_repair_rounds=max(0, int(args.max_repair_rounds)),
            feedback=feedback,
            previous_output=envelope.raw_llm_output,
        )

    artifact_base = _artifact_base(out_dir, question)
    if not result.success:
        failure_artifact = _make_artifact(
            question=question,
            context=context,
            envelope=envelope,
            result=result,
            render_payload={},
            rendered_answer=settings.EMPTY_STRING,
        )
        failure_path = artifact_base.with_suffix(settings.FILE_SUFFIX_FAILED_JSON)
        _write_json(failure_path, failure_artifact)
        print(
            settings.CLI_ERROR_PROGRAM_GENERATION_FAILED_TEMPLATE.format(
                status=result.status,
                error=result.error,
            ),
            file=sys.stderr,
        )
        print(
            settings.CLI_DEBUG_ARTIFACT_TEMPLATE.format(
                artifact_path=failure_path
            ),
            file=sys.stderr,
        )
        return 1

    trace_value = _trace_binding(result.solutions)
    if trace_value is None:
        failure_artifact = _make_artifact(
            question=question,
            context=context,
            envelope=envelope,
            result=result,
            render_payload={},
            rendered_answer=settings.EMPTY_STRING,
        )
        failure_path = artifact_base.with_suffix(settings.FILE_SUFFIX_FAILED_JSON)
        _write_json(failure_path, failure_artifact)
        print(settings.CLI_ERROR_VERIFIED_PROGRAM_MISSING_TRACE, file=sys.stderr)
        print(
            settings.CLI_DEBUG_ARTIFACT_TEMPLATE.format(
                artifact_path=failure_path
            ),
            file=sys.stderr,
        )
        return 1

    render_payload = _render_payload(question, context, envelope, result)
    render_response = llm.generate(
        {
            settings.TASK_KEY: settings.TASK_IRAC_RENDER,
            settings.PAYLOAD_TRACE_KEY: render_payload,
        }
    )
    rendered_answer = str(
        render_response.get(settings.RESPONSE_TEXT_KEY)
        or render_response.get(settings.RESPONSE_ANSWER_KEY)
        or settings.EMPTY_STRING
    ).strip()
    if not rendered_answer:
        failure_artifact = _make_artifact(
            question=question,
            context=context,
            envelope=envelope,
            result=result,
            render_payload=render_payload,
            rendered_answer=settings.EMPTY_STRING,
        )
        failure_path = artifact_base.with_suffix(settings.FILE_SUFFIX_FAILED_JSON)
        _write_json(failure_path, failure_artifact)
        print(settings.CLI_ERROR_EMPTY_RENDERED_ANSWER, file=sys.stderr)
        print(
            settings.CLI_DEBUG_ARTIFACT_TEMPLATE.format(
                artifact_path=failure_path
            ),
            file=sys.stderr,
        )
        return 1

    program_path, json_path = _save_success_artifacts(
        artifact_base,
        question=question,
        context=context,
        envelope=envelope,
        result=result,
        render_payload=render_payload,
        rendered_answer=rendered_answer,
    )

    if args.json:
        print(
            json.dumps(
                json.loads(json_path.read_text(encoding=settings.PATH_ENCODING)),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(rendered_answer)
        print()
        print(
            settings.CLI_OUTPUT_LINE_TEMPLATE.format(
                label=settings.CLI_PROGRAM_OUTPUT_LABEL,
                path=program_path,
            )
        )
        print(
            settings.CLI_OUTPUT_LINE_TEMPLATE.format(
                label=settings.CLI_ARTIFACT_OUTPUT_LABEL,
                path=json_path,
            )
        )
    return 0


def _make_retriever(retrieval_mode: str, corpus_path: Path, ontology_path: Path) -> Any:
    if retrieval_mode == settings.RETRIEVAL_MODE_HYBRID:
        from src.logic_lm.knowledge.hybrid_retrieval import M1Retrieval

        retriever = M1Retrieval()
        retriever.index_corpus(corpus_path)
        return retriever

    from src.logic_lm.knowledge.bhxh_ontology import build_ontology_file
    from src.logic_lm.knowledge.ontology_retrieval import OntologyRetrieval

    if not ontology_path.exists():
        build_ontology_file(corpus_path, ontology_path)

    retriever = OntologyRetrieval()
    retriever.index_ontology(ontology_path)
    return retriever


def _resolve_question(args: argparse.Namespace) -> str:
    if args.question and args.question.strip():
        return args.question.strip()
    if args.question_parts:
        return settings.SPACE.join(args.question_parts).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return settings.EMPTY_STRING


def _artifact_base(out_dir: Path, question: str) -> Path:
    timestamp = datetime.now().strftime(settings.ARTIFACT_TIMESTAMP_FORMAT)
    digest = hashlib.sha256(question.encode(settings.PATH_ENCODING)).hexdigest()
    digest = digest[: settings.ARTIFACT_DIGEST_LENGTH]
    return out_dir / settings.ARTIFACT_FILENAME_TEMPLATE.format(
        timestamp=timestamp,
        digest=digest,
    )


def _trace_binding(solutions: List[Mapping[str, Any]]) -> Any:
    for solution in solutions or []:
        for key, value in solution.items():
            if str(key).lower() == settings.TRACE_VARIABLE_NAME.lower():
                return value
    return None


def _selected_citations(context: Any, indices: List[int]) -> List[Dict[str, Any]]:
    chunks = list(getattr(context, settings.FIELD_CHUNKS, []) or [])
    citations: List[Dict[str, Any]] = []
    for index in indices:
        if index < 0 or index >= len(chunks):
            continue
        chunk = chunks[index]
        citations.append(
            {
                settings.FIELD_INDEX: index,
                settings.FIELD_CHUNK_ID: getattr(
                    chunk,
                    settings.FIELD_ID,
                    settings.EMPTY_STRING,
                ),
                settings.FIELD_DOCUMENT: getattr(
                    chunk,
                    settings.FIELD_DOCUMENT,
                    None,
                ),
                settings.FIELD_ARTICLE: getattr(
                    chunk,
                    settings.FIELD_ARTICLE,
                    None,
                ),
                settings.FIELD_CLAUSE: getattr(
                    chunk,
                    settings.FIELD_CLAUSE,
                    None,
                ),
                settings.FIELD_POINT: getattr(
                    chunk,
                    settings.FIELD_POINT,
                    None,
                ),
                settings.FIELD_RAW_TEXT: getattr(
                    chunk,
                    settings.FIELD_TEXT,
                    settings.EMPTY_STRING,
                )
                or settings.EMPTY_STRING,
            }
        )
    return citations


def _retrieved_chunks(context: Any) -> List[Dict[str, Any]]:
    chunks = list(getattr(context, settings.FIELD_CHUNKS, []) or [])
    return [
        {
            settings.FIELD_CHUNK_ID: getattr(
                chunk,
                settings.FIELD_ID,
                settings.EMPTY_STRING,
            ),
            settings.FIELD_DOCUMENT: getattr(chunk, settings.FIELD_DOCUMENT, None),
            settings.FIELD_ARTICLE: getattr(chunk, settings.FIELD_ARTICLE, None),
            settings.FIELD_CLAUSE: getattr(chunk, settings.FIELD_CLAUSE, None),
            settings.FIELD_POINT: getattr(chunk, settings.FIELD_POINT, None),
            settings.FIELD_TEXT: getattr(
                chunk,
                settings.FIELD_TEXT,
                settings.EMPTY_STRING,
            )
            or settings.EMPTY_STRING,
        }
        for chunk in chunks
    ]


def _predicate_name(slot_id: str, spec: Any) -> str:
    if isinstance(spec, Mapping):
        return str(spec.get(settings.RESPONSE_PREDICATE_KEY) or slot_id)
    return str(spec or slot_id)


def _slot_bindings(envelope: Any) -> Dict[str, Dict[str, Any]]:
    bindings: Dict[str, Dict[str, Any]] = {}
    facts = list(getattr(envelope, settings.RESPONSE_VERIFY_FACTS_KEY, []) or [])
    predicate_inputs = dict(
        getattr(envelope, settings.RESPONSE_PREDICATE_INPUTS_KEY, {}) or {}
    )
    for slot_id, spec in predicate_inputs.items():
        predicate = _predicate_name(str(slot_id), spec)
        matched_facts = [
            fact
            for fact in facts
            if str(fact)
            .lstrip()
            .startswith(
                settings.PREDICATE_FACT_PREFIX_TEMPLATE.format(
                    predicate=predicate
                )
            )
        ]
        bindings[str(slot_id)] = {
            settings.RESPONSE_PREDICATE_KEY: predicate,
            settings.FIELD_FACTS: matched_facts,
        }
    return bindings


def _render_payload(question: str, context: Any, envelope: Any, result: Any) -> Dict[str, Any]:
    citations = _selected_citations(
        context,
        list(getattr(envelope, settings.CITATION_INDICES_ATTRIBUTE, []) or []),
    )
    return {
        settings.FIELD_NORMALIZED_QUESTION: question,
        settings.FIELD_LEGAL_ISSUE: question,
        settings.FIELD_DOMAIN_CONTEXT: settings.DOMAIN_CONTEXT,
        settings.FIELD_SELECTED_FUNCTION: {
            settings.FIELD_NAME: settings.SELECTED_FUNCTION_NAME,
            settings.FIELD_DESCRIPTION: settings.SELECTED_FUNCTION_DESCRIPTION,
        },
        settings.FIELD_SLOT_BINDINGS: _slot_bindings(envelope),
        settings.RESPONSE_VERIFY_FACTS_KEY: list(
            getattr(envelope, settings.RESPONSE_VERIFY_FACTS_KEY, []) or []
        ),
        settings.RESPONSE_CITATIONS_KEY: citations,
        settings.FIELD_EXECUTION_RESULT: list(
            getattr(result, settings.FIELD_SOLUTIONS, []) or []
        ),
        settings.FIELD_PROLOG_TRACE: _trace_binding(
            getattr(result, settings.FIELD_SOLUTIONS, []) or []
        ),
        settings.FIELD_GENERATED_PROGRAM: {
            settings.RESPONSE_LEGAL_SOURCES_KEY: list(
                getattr(envelope, settings.RESPONSE_LEGAL_SOURCES_KEY, []) or []
            ),
            settings.RESPONSE_RULES_KEY: list(
                getattr(envelope, settings.RESPONSE_RULES_KEY, []) or []
            ),
            settings.RESPONSE_QUERY_KEY: getattr(
                envelope,
                settings.RESPONSE_QUERY_KEY,
                settings.EMPTY_STRING,
            ),
            settings.RESPONSE_ANSWER_VAR_KEY: getattr(
                envelope,
                settings.RESPONSE_ANSWER_VAR_KEY,
                settings.EMPTY_STRING,
            ),
            settings.RESPONSE_ANSWER_TYPE_KEY: getattr(
                envelope,
                settings.RESPONSE_ANSWER_TYPE_KEY,
                settings.EMPTY_STRING,
            ),
        },
    }


def _make_artifact(
    *,
    question: str,
    context: Any,
    envelope: Any,
    result: Any,
    render_payload: Mapping[str, Any],
    rendered_answer: str,
) -> Dict[str, Any]:
    return {
        settings.FIELD_QUESTION: question,
        settings.FIELD_RETRIEVED_CHUNKS: _retrieved_chunks(context),
        settings.FIELD_PROGRAM: {
            settings.RESPONSE_LEGAL_SOURCES_KEY: list(
                getattr(envelope, settings.RESPONSE_LEGAL_SOURCES_KEY, []) or []
            ),
            settings.RESPONSE_RULES_KEY: list(
                getattr(envelope, settings.RESPONSE_RULES_KEY, []) or []
            ),
            settings.RESPONSE_VERIFY_FACTS_KEY: list(
                getattr(envelope, settings.RESPONSE_VERIFY_FACTS_KEY, []) or []
            ),
            settings.RESPONSE_QUERY_KEY: getattr(
                envelope,
                settings.RESPONSE_QUERY_KEY,
                settings.EMPTY_STRING,
            ),
            settings.RESPONSE_ANSWER_VAR_KEY: getattr(
                envelope,
                settings.RESPONSE_ANSWER_VAR_KEY,
                settings.EMPTY_STRING,
            ),
            settings.RESPONSE_ANSWER_TYPE_KEY: getattr(
                envelope,
                settings.RESPONSE_ANSWER_TYPE_KEY,
                settings.EMPTY_STRING,
            ),
            settings.RESPONSE_PREDICATE_INPUTS_KEY: dict(
                getattr(envelope, settings.RESPONSE_PREDICATE_INPUTS_KEY, {}) or {}
            ),
            settings.RESPONSE_RAW_LLM_OUTPUT_KEY: dict(
                getattr(envelope, settings.RESPONSE_RAW_LLM_OUTPUT_KEY, {}) or {}
            ),
            settings.FIELD_PROLOG_SOURCE: getattr(
                envelope,
                settings.RAW_PROGRAM_TEXT_ATTRIBUTE,
                settings.EMPTY_STRING,
            ),
        },
        settings.FIELD_VERIFICATION: {
            settings.FIELD_SUCCESS: bool(getattr(result, settings.FIELD_SUCCESS, False)),
            settings.FIELD_STATUS: getattr(
                result,
                settings.FIELD_STATUS,
                settings.EMPTY_STRING,
            ),
            settings.FIELD_ERROR: getattr(
                result,
                settings.FIELD_ERROR,
                settings.EMPTY_STRING,
            ),
            settings.FIELD_RESULT: getattr(result, settings.FIELD_RESULT, None),
            settings.FIELD_SOLUTIONS: list(
                getattr(result, settings.FIELD_SOLUTIONS, []) or []
            ),
        },
        settings.FIELD_RENDER_PAYLOAD: dict(render_payload),
        settings.FIELD_RENDERED_ANSWER: rendered_answer,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        + settings.NEWLINE,
        encoding=settings.PATH_ENCODING,
    )


def _save_success_artifacts(
    base_path: Path,
    *,
    question: str,
    context: Any,
    envelope: Any,
    result: Any,
    render_payload: Mapping[str, Any],
    rendered_answer: str,
) -> Tuple[Path, Path]:
    program_path = base_path.with_suffix(settings.FILE_SUFFIX_PROLOG)
    json_path = base_path.with_suffix(settings.FILE_SUFFIX_JSON)
    program_source = str(
        getattr(
            envelope,
            settings.RAW_PROGRAM_TEXT_ATTRIBUTE,
            settings.EMPTY_STRING,
        )
        or settings.EMPTY_STRING
    ).strip()
    program_path.write_text(
        program_source + settings.NEWLINE,
        encoding=settings.PATH_ENCODING,
    )
    artifact = _make_artifact(
        question=question,
        context=context,
        envelope=envelope,
        result=result,
        render_payload=render_payload,
        rendered_answer=rendered_answer,
    )
    _write_json(json_path, artifact)
    return program_path, json_path


if __name__ == settings.MAIN_MODULE_NAME:
    raise SystemExit(main())
