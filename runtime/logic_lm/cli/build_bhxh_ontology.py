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
        description="Build a BHXH ontology JSON file from a corpus JSONL."
    )
    parser.add_argument(
        settings.CLI_ARGUMENT_CORPUS,
        default=str(REPO_ROOT / settings.DATA_DIR_NAME / settings.ONTOLOGY_DIR_NAME / settings.DEFAULT_CORPUS_FILENAME),
        help=settings.CLI_HELP_CORPUS,
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / settings.DATA_DIR_NAME / settings.ONTOLOGY_DIR_NAME / settings.DEFAULT_ONTOLOGY_FILENAME),
        help="Output ontology JSON path.",
    )
    parser.add_argument(
        "--compact",
        action=settings.CLI_ACTION_STORE_TRUE,
        help="Write compact JSON instead of pretty JSON.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    from runtime.logic_lm.knowledge.bhxh_ontology import build_ontology_file

    args = make_parser().parse_args(list(argv) if argv is not None else None)
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(
            settings.CLI_ERROR_CORPUS_NOT_FOUND_TEMPLATE.format(
                corpus_path=corpus_path,
            ),
            file=sys.stderr,
        )
        return 2

    ontology = build_ontology_file(corpus_path, args.out, pretty=not args.compact)
    print(
        json.dumps(
            {
                "path": str(Path(args.out)),
                "node_count": ontology.get("node_count"),
                "edge_count": ontology.get("edge_count"),
                "chunk_count": len(ontology.get("chunks") or []),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == settings.MAIN_MODULE_NAME:
    raise SystemExit(main())
