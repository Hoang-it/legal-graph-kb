import os
import re
import tempfile
from typing import Any, Dict, List

from runtime.logic_lm.config import settings






class PrologSolver:
    @staticmethod
    def run(program: str, query: str) -> List[Dict[str, Any]]:

        try:
            return _run_swipl(program, query)
        except SyntaxError:
            raise
        except FileNotFoundError as exc:
            raise RuntimeError(
                settings.PROLOG_EXECUTABLE_NOT_FOUND_TEMPLATE.format(
                    executable=settings.PROLOG_EXECUTABLE
                )
            ) from exc






def _normalise_query(query: str) -> str:
    q = query.strip()
    if q.startswith(settings.PROLOG_QUERY_PREFIX):
        q = q[2:].strip()
    if q.endswith(settings.PERIOD):
        q = q[:-1].strip()
    return q


def _extract_vars(query: str) -> List[str]:

    return list(dict.fromkeys(
        v for v in re.findall(settings.REGEX_PROLOG_VARIABLE, query)
    ))


def _compose_prolog_source(program: str, query: str) -> tuple[str, List[str]]:

    q = _normalise_query(query)
    query_vars = _extract_vars(q)

    if query_vars:

        write_parts = settings.COMMA_SPACE.join(
            settings.PROLOG_VAR_WRITE_TEMPLATE.format(var=v)
            for v in query_vars
        )
        output_goal = settings.PROLOG_OUTPUT_GOAL_TEMPLATE.format(
            begin_marker=settings.PROLOG_RESULT_BEGIN,
            write_parts=write_parts,
            end_marker=settings.PROLOG_RESULT_END,
        )
        main_goal = settings.PROLOG_MAIN_GOAL_WITH_OUTPUT_TEMPLATE.format(
            query=q,
            output_goal=output_goal,
            no_solution_marker=settings.PROLOG_NO_SOLUTION,
        )
    else:
        main_goal = settings.PROLOG_MAIN_GOAL_TRUE_TEMPLATE.format(
            query=q,
            true_marker=settings.PROLOG_RESULT_TRUE,
            no_solution_marker=settings.PROLOG_NO_SOLUTION,
        )

    source = settings.PROLOG_SOURCE_TEMPLATE.format(
        program=program,
        main_goal=main_goal,
        error_marker=settings.PROLOG_ERROR,
    )
    return source, query_vars


def _parse_swipl_output(stdout: str, query_vars: List[str]) -> List[Dict[str, Any]]:
    stdout = stdout.strip()

    if (
        not stdout
        or stdout == settings.PROLOG_NO_SOLUTION
        or stdout == settings.PROLOG_ERROR
    ):
        return []

    if stdout == settings.PROLOG_RESULT_TRUE:
        return [{}]

    if (
        settings.PROLOG_RESULT_BEGIN in stdout
        and settings.PROLOG_RESULT_END in stdout
    ):
        result: Dict[str, Any] = {}
        for line in stdout.splitlines():
            line = line.strip()
            if settings.EQUALS in line and line not in (
                settings.PROLOG_RESULT_BEGIN,
                settings.PROLOG_RESULT_END,
            ):
                var, _, val = line.partition(settings.EQUALS)
                result[var.strip()] = _coerce(val.strip())
        return [result] if result else []

    return []


def _coerce(s: str) -> Any:

    s = s.strip().strip(settings.SINGLE_QUOTE + settings.DOUBLE_QUOTE)
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _run_swipl(program: str, query: str) -> List[Dict[str, Any]]:

    import subprocess

    source, query_vars = _compose_prolog_source(program, query)

    with tempfile.NamedTemporaryFile(
        settings.PROLOG_TEMP_FILE_MODE,
        suffix=settings.PROLOG_TEMP_FILE_SUFFIX,
        delete=False,
        encoding=settings.PATH_ENCODING,
    ) as fh:
        fh.write(source)
        fname = fh.name

    try:
        proc = subprocess.run(
            [settings.PROLOG_EXECUTABLE, settings.PROLOG_QUIET_FLAG, fname],
            capture_output=True,
            text=True,
            timeout=settings.PROLOG_TIMEOUT_SECONDS,
            encoding=settings.PATH_ENCODING,
        )
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()


        if (
            settings.PROLOG_SYNTAX_ERROR_TEXT in stderr
            or settings.PROLOG_SYNTAX_ERROR_ATOM in stderr
        ):
            raise SyntaxError(stderr)


        if proc.returncode != 0 and not stdout:
            raise RuntimeError(
                settings.PROLOG_EXIT_ERROR_TEMPLATE.format(
                    return_code=proc.returncode,
                    stderr=stderr,
                )
            )

        return _parse_swipl_output(stdout, query_vars)

    except subprocess.TimeoutExpired:
        return []
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass
