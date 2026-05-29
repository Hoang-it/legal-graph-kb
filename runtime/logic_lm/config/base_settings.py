EMPTY_STRING = ""
SPACE = " "
NEWLINE = "\n"
COMMA_SPACE = ", "
PERIOD = "."
SINGLE_QUOTE = "'"
DOUBLE_QUOTE = '"'
EQUALS = "="
LEFT_PAREN = "("
RIGHT_PAREN = ")"
PATH_ENCODING = "utf-8"
STREAM_RECONFIGURE_METHOD = "reconfigure"
FILE_MODE_READ = "r"
MAIN_MODULE_NAME = "__main__"

SRC_DIR_NAME = "src"
DATA_DIR_NAME = "data"
# Sub-directory under data/ where the logic-LM knowledge base (corpus +
# ontology) lives. `data/` is read-only KG input — only the KB sits here.
ONTOLOGY_DIR_NAME = "ontology"
DEFAULT_CORPUS_FILENAME = "corpus_2024.jsonl"
DEFAULT_ONTOLOGY_FILENAME = "ontology_2024.json"
# Runtime-generated Prolog programs land at the repo root under
# `artifacts/logic_lm/programs/` — NOT under `data/` (data is input-only).
ARTIFACTS_DIR_NAME = "artifacts"
LOGIC_LM_ARTIFACTS_SUBDIR = "logic_lm"
PROGRAM_ARTIFACTS_DIR_NAME = "programs"

FILE_SUFFIX_FAILED_JSON = ".failed.json"
FILE_SUFFIX_PROLOG = ".pl"
FILE_SUFFIX_JSON = ".json"
ARTIFACT_FILENAME_TEMPLATE = "program_{timestamp}_{digest}"
ARTIFACT_DIGEST_LENGTH = 10
ARTIFACT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

MARKDOWN_FENCE_REPLACEMENT = ""
MARKDOWN_CODE_FENCE = "```"
JSON_EMPTY_OBJECT = "{}"

REGEX_WORD = r"\w+"
REGEX_MARKDOWN_FENCE = r"```[a-zA-Z]*\n?"
REGEX_JSON_OBJECT = r"\{.*\}"
