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
# Sub-directory under data/ where logic-lm corpus & ontology live after
# the v5 refactor (former `elite/data/` is now `data/logic_lm/`).
LOGIC_LM_DATA_SUBDIR = "logic_lm"
DEFAULT_CORPUS_FILENAME = "corpus_bhxh.jsonl"
DEFAULT_ONTOLOGY_FILENAME = "ontology_bhxh.json"
GENERATED_DIR_NAME = "generated"
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
