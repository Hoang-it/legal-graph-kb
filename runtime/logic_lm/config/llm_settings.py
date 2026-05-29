ENCODER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
OPENAI_MODEL_ENV_VAR = "OPENAI_MODEL"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_TEMPERATURE = 0.1
DETERMINISTIC_LLM_TEMPERATURE = 0.0

OPENAI_ROLE_KEY = "role"
OPENAI_CONTENT_KEY = "content"
OPENAI_SYSTEM_ROLE = "system"
OPENAI_USER_ROLE = "user"

ERROR_OPENAI_API_KEY_MISSING_TEMPLATE = (
    "{env_var} environment variable is not set."
)
ERROR_UNSUPPORTED_LLM_TASK_TEMPLATE = "Unsupported LLM task for answer_with_program: {task}"

LLM_USER_RETRIEVED_CHUNKS_HEADER = "retrieved_chunks:"
LLM_USER_EMPTY_CHUNKS = "  (empty)"
LLM_PREVIOUS_ATTEMPT_REPAIR_MESSAGE = (
    "Previous attempt failed verification. Repair the output."
)
LLM_REPAIR_REQUIREMENTS = (
    "repair_requirements: return complete Prolog clauses; each rules[] "
    "item must be one full clause ending with '.', never a clause line "
    "fragment ending with ':-', ',', or ';'. If the question asks for "
    "legal conditions generally, create a no-input rule returning a "
    "Conditions or Answer term plus Trace instead of requiring missing "
    "person-specific facts."
)
LLM_EXISTING_PROGRAM_REVIEW_MESSAGE = (
    "Existing persisted program is being reviewed. Generate a corrected "
    "replacement only if the feedback identifies a real issue."
)
LLM_TRAINING_QUESTION_LINE_TEMPLATE = "training_question: {training_question}"
LLM_PREVIOUS_ERROR_LINE_TEMPLATE = "previous_error: {previous_error}"
LLM_PREVIOUS_PROGRAM_LINE_TEMPLATE = "previous_prolog_source:\n{previous_program}"
LLM_PREVIOUS_OUTPUT_LINE_TEMPLATE = "previous_output: {previous_output}"
LLM_HUMAN_FEEDBACK_LINE_TEMPLATE = "human_feedback: {feedback}"
LLM_CHUNK_LINE_TEMPLATE = "  [{index}] {ref}: {text}"
LLM_FALLBACK_CHUNK_LINE_TEMPLATE = "  [{index}] {chunk}"
