import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.logic_lm.config import settings

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, settings.STREAM_RECONFIGURE_METHOD):
        try:
            _stream.reconfigure(encoding=settings.PATH_ENCODING)
        except Exception:
            pass


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve BHXH legal context from the ontology graph."
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
        default=str(REPO_ROOT / settings.DATA_DIR_NAME / settings.LOGIC_LM_DATA_SUBDIR / settings.DEFAULT_CORPUS_FILENAME),
        help=settings.CLI_HELP_CORPUS,
    )
    parser.add_argument(
        settings.CLI_ARGUMENT_ONTOLOGY,
        default=str(REPO_ROOT / settings.DATA_DIR_NAME / settings.LOGIC_LM_DATA_SUBDIR / settings.DEFAULT_ONTOLOGY_FILENAME),
        help=settings.CLI_HELP_ONTOLOGY,
    )
    parser.add_argument(
        settings.CLI_ARGUMENT_TOP_K,
        type=int,
        default=settings.DEFAULT_RETRIEVAL_TOP_K,
        help=settings.CLI_HELP_TOP_K,
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    from runtime.logic_lm.knowledge.bhxh_ontology import build_ontology_file
    from runtime.logic_lm.knowledge.ontology_retrieval import OntologyRetrieval

    args = make_parser().parse_args(list(argv) if argv is not None else None)
    question = _resolve_question(args)
    if not question:
        print(settings.CLI_ERROR_QUESTION_REQUIRED, file=sys.stderr)
        return 2

    ontology_path = Path(args.ontology)
    corpus_path = Path(args.corpus)
    if not ontology_path.exists():
        if not corpus_path.exists():
            print(
                settings.CLI_ERROR_CORPUS_NOT_FOUND_TEMPLATE.format(
                    corpus_path=corpus_path,
                ),
                file=sys.stderr,
            )
            return 2
        build_ontology_file(corpus_path, ontology_path)

    retriever = OntologyRetrieval()
    retriever.index_ontology(ontology_path)
    context = retriever.retrieve(question, top_k=max(1, int(args.top_k)))
    print(
        json.dumps(
            {
                settings.FIELD_QUESTION: question,
                settings.FIELD_RETRIEVED_CHUNKS: [
                    {
                        "score": context.scores.get(chunk.id),
                        settings.FIELD_ID: chunk.id,
                        settings.FIELD_DOCUMENT: chunk.document,
                        settings.FIELD_ARTICLE: chunk.article,
                        settings.FIELD_CLAUSE: chunk.clause,
                        settings.FIELD_POINT: chunk.point,
                        settings.FIELD_TEXT: chunk.text,
                    }
                    for chunk in context.chunks
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _resolve_question(args: argparse.Namespace) -> str:
    if args.question and args.question.strip():
        return args.question.strip()
    if args.question_parts:
        return settings.SPACE.join(args.question_parts).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return settings.EMPTY_STRING


if __name__ == settings.MAIN_MODULE_NAME:
    raise SystemExit(main())
