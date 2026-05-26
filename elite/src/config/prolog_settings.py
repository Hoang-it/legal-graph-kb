from .base_settings import DOUBLE_QUOTE, SINGLE_QUOTE


PROLOG_QUERY_PREFIX = "?-"
PROLOG_QUERY_TEMPLATE = "{prefix} {query}"
PROLOG_DOTTED_QUERY_TEMPLATE = "{query}{period}"
PROLOG_TEMP_FILE_MODE = "w"
PROLOG_TEMP_FILE_SUFFIX = ".pl"
PROLOG_ESCAPE_CHAR = "\\"
PROLOG_QUOTE_CHARS = (SINGLE_QUOTE, DOUBLE_QUOTE)
PROLOG_ANONYMOUS_VARIABLE_PREFIX = "_"
PROLOG_EXECUTABLE = "swipl"
PROLOG_QUIET_FLAG = "-q"
PROLOG_RESULT_BEGIN = "RESULT_BEGIN"
PROLOG_RESULT_END = "RESULT_END"
PROLOG_RESULT_TRUE = "RESULT_TRUE"
PROLOG_NO_SOLUTION = "NO_SOLUTION"
PROLOG_ERROR = "PROLOG_ERROR"
PROLOG_SYNTAX_ERROR_TEXT = "Syntax error"
PROLOG_SYNTAX_ERROR_ATOM = "syntax_error"
PROLOG_PARSE_ERROR_TEXT = "parse"
PROLOG_VAR_WRITE_TEMPLATE = "(write('{var}='), write({var}), nl)"
PROLOG_OUTPUT_GOAL_TEMPLATE = (
    "(writeln('{begin_marker}'), {write_parts}, writeln('{end_marker}'))"
)
PROLOG_MAIN_GOAL_WITH_OUTPUT_TEMPLATE = (
    "( once(({query})) -> {output_goal} ; writeln('{no_solution_marker}') )"
)
PROLOG_MAIN_GOAL_TRUE_TEMPLATE = (
    "( once(({query})) -> writeln('{true_marker}') ; "
    "writeln('{no_solution_marker}') )"
)
PROLOG_SOURCE_TEMPLATE = (
    "{program}\n"
    ":- catch(({main_goal}), _Err, writeln('{error_marker}')), halt.\n"
)
PROLOG_EXIT_ERROR_TEMPLATE = "swipl exited {return_code}: {stderr}"
PROLOG_EXECUTABLE_NOT_FOUND_TEMPLATE = (
    "SWI-Prolog executable '{executable}' was not found in PATH"
)
PROLOG_TIMEOUT_SECONDS = 15

REGEX_PROLOG_VARIABLE = r"\b([A-Z][A-Za-z0-9_]*)\b"
