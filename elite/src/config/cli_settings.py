CLI_DESCRIPTION = (
    "Answer one question by generating a trace-rich Prolog program, "
    "verifying it with SWI-Prolog, saving the program artifact, and "
    "rendering a user-facing answer with a second LLM call."
)

CLI_ARGUMENT_QUESTION_PARTS = "question_parts"
CLI_ARGUMENT_QUESTION = "--question"
CLI_ARGUMENT_CORPUS = "--corpus"
CLI_ARGUMENT_ONTOLOGY = "--ontology"
CLI_ARGUMENT_RETRIEVAL_MODE = "--retrieval-mode"
CLI_ARGUMENT_PROGRAM_DIR = "--program-dir"
CLI_ARGUMENT_TOP_K = "--top-k"
CLI_ARGUMENT_MAX_REPAIR_ROUNDS = "--max-repair-rounds"
CLI_ARGUMENT_TRACE_REPAIR_ROUNDS = "--trace-repair-rounds"
CLI_ARGUMENT_JSON = "--json"
CLI_NARGS_ANY = "*"
CLI_ACTION_STORE_TRUE = "store_true"

CLI_HELP_QUESTION_PARTS = "Question text. If omitted, the command reads stdin."
CLI_HELP_QUESTION = "Question text. Overrides positional text when provided."
CLI_HELP_CORPUS = "Legal corpus JSONL used to build the ontology or hybrid index."
CLI_HELP_ONTOLOGY = "BHXH ontology JSON used for ontology retrieval."
CLI_HELP_RETRIEVAL_MODE = "Retrieval backend: ontology graph or legacy hybrid corpus search."
CLI_HELP_PROGRAM_DIR = "Directory where generated .pl and .json artifacts are written."
CLI_HELP_TOP_K = "Number of retrieved legal chunks supplied to the LLM."
CLI_HELP_MAX_REPAIR_ROUNDS = (
    "Maximum LLM repair attempts after Prolog/schema verification fails."
)
CLI_HELP_TRACE_REPAIR_ROUNDS = (
    "Additional repair attempts when the verified program lacks Trace."
)
CLI_HELP_JSON = "Print the full artifact JSON instead of only the rendered answer."

CLI_ERROR_QUESTION_REQUIRED = "question is required"
CLI_ERROR_CORPUS_NOT_FOUND_TEMPLATE = "corpus not found: {corpus_path}"
CLI_ERROR_PROGRAM_GENERATION_FAILED_TEMPLATE = (
    "program generation failed: status={status} error={error}"
)
CLI_ERROR_VERIFIED_PROGRAM_MISSING_TRACE = "verified program does not expose a Trace binding"
CLI_ERROR_EMPTY_RENDERED_ANSWER = "answer rendering LLM returned empty text"
CLI_DEBUG_ARTIFACT_TEMPLATE = "debug artifact: {artifact_path}"
CLI_TRACE_REPAIR_FEEDBACK = (
    "The Prolog program verified, but the query did not return a Trace "
    "variable. Regenerate the program so every rule constructs a "
    "step(..., based_on(...)) trace and the query returns Trace."
)

CLI_PROGRAM_OUTPUT_LABEL = "program"
CLI_ARTIFACT_OUTPUT_LABEL = "artifact"
CLI_OUTPUT_LINE_TEMPLATE = "{label}: {path}"

DEFAULT_RETRIEVAL_TOP_K = 15
DEFAULT_MAX_REPAIR_ROUNDS = 3
DEFAULT_TRACE_REPAIR_ROUNDS = 2
