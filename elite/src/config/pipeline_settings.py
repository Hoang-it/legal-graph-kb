DEFAULT_PIPELINE_MAX_REPAIR_ROUNDS = 2
DEFAULT_LLM_CHUNK_LIMIT = 8
DEFAULT_LLM_CHUNK_TEXT_LIMIT = 600

ERROR_LLM_RESPONSE_NOT_JSON = "LLM response was not a JSON object"
ERROR_LLM_NO_RULES_OR_QUERY = "LLM returned no rules or no query"
ERROR_LLM_NO_CITATIONS = "LLM returned no citations for generated legal rules"
ERROR_PREDICATE_INPUTS_WITHOUT_VERIFY_FACTS = (
    "LLM returned predicate_inputs but no verify_facts"
)
ERROR_VERIFY_FACTS_WITHOUT_PREDICATE_INPUTS = (
    "LLM returned verify_facts but no predicate_inputs"
)
ERROR_PREDICATE_INPUTS_MISSING_TEMPLATE = "predicate_inputs missing specs for: {names}"
ERROR_EMPTY_PROLOG_SOURCE = "empty Prolog source"
ERROR_PROLOG_NO_SOLUTIONS = "Prolog produced no solutions"
ERROR_EMPTY_QUERY = "empty query"
ERROR_QUERY_CONTAINS_NUMBER_TEMPLATE = "query contains literal number(s): {query!r}"
ERROR_QUERY_CONTAINS_QUOTED_LITERAL_TEMPLATE = (
    "query contains quoted literal(s): {query!r}"
)
ERROR_EMPTY_ARGUMENT_IN_QUERY_TEMPLATE = "empty argument in query: {query!r}"
ERROR_QUERY_ARGUMENT_TEMPLATE = "query argument must be variable or user, got {arg!r}"

REGEX_FACT_HEAD = r"^\s*([a-z_][\w]*)\s*\("
REGEX_QUERY_LITERAL_NUMBER = r"(?<![A-Za-z_])-?\d+(?:\.\d+)?"

PROGRAM_PIPELINE_PUBLIC_API = (
    "ProgramEnvelope",
    "ExecutionResult",
    "generate_and_execute",
)
