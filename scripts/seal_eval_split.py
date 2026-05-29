"""Hash-seal a stratified 150-test / 50-dev split of the eval question set.

Plan v5 §5 contract: the test/dev split must be frozen once and never touched
again — otherwise downstream metric comparisons are meaningless. This script:

1. Categorizes each of the 200 questions in ``data/eval/questions_200.json``
   by *gold_citations_raw corpus type*:
     - ``in_corpus``   — every parseable law code in gold ∈ KG (3 luật).
     - ``ooc``         — no parseable law code in gold is in KG.
     - ``mixed``       — at least one in-corpus + at least one OOC code.
     - ``unparseable`` — no machine-readable law code at all
       (e.g. Vietnamese-title-only or pure-narrative gold).
2. Stratified-samples 150 test / 50 dev (3 : 1 ratio) **per category** using a
   fixed seed.
3. Writes:
     - ``data/eval/questions_150_test.json``
     - ``data/eval/questions_50_dev.json``
     - ``data/eval/eval_split_hashes.json`` (SHA256 of both files + script
       version metadata).

After running once, **commit the three output files**. Re-running the script
verifies the existing hashes match — exits non-zero if anything drifts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.legal_metadata import load_law_metadata

SOURCE = Path("data/eval/questions_200.json")
TEST_OUT = Path("data/eval/questions_150_test.json")
DEV_OUT = Path("data/eval/questions_50_dev.json")
HASHES_OUT = Path("data/eval/eval_split_hashes.json")

SCRIPT_VERSION = "v5-sprint2-phase0b-1"
SEED = 42

_RE_LAW_CODE = re.compile(r"\d+/\d{4}/(?:QH\d+|N[ĐD]-CP|NQ-CP|TT-[A-Z]+|CP|TTg)")


def categorize(raw_gold: str, in_corpus_codes: set[str]) -> str:
    if not raw_gold:
        return "unparseable"
    if isinstance(raw_gold, list):
        raw_gold = "\n".join(str(x) for x in raw_gold)
    hits = _RE_LAW_CODE.findall(raw_gold)
    if not hits:
        return "unparseable"
    in_kg = sum(1 for h in hits if h in in_corpus_codes)
    if in_kg == len(hits):
        return "in_corpus"
    if in_kg == 0:
        return "ooc"
    return "mixed"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def stratified_split(
    questions: list[dict],
    categories: dict[int, str],
    test_ratio: float = 0.75,  # 150/200
    seed: int = SEED,
) -> tuple[list[dict], list[dict], dict[str, dict[str, int]]]:
    """Per-category random split. Returns (test, dev, stats)."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_cat[categories[q["stt"]]].append(q)

    rng = random.Random(seed)
    test: list[dict] = []
    dev: list[dict] = []
    stats: dict[str, dict[str, int]] = {}

    for cat in sorted(by_cat.keys()):
        group = list(by_cat[cat])
        rng.shuffle(group)
        # ceil split so categories with odd counts lean to test
        n_test = round(len(group) * test_ratio)
        n_test = max(0, min(n_test, len(group)))
        test.extend(group[:n_test])
        dev.extend(group[n_test:])
        stats[cat] = {
            "total": len(group),
            "test": n_test,
            "dev": len(group) - n_test,
        }

    # Sort each split by stt for determinism on disk.
    test.sort(key=lambda q: q["stt"])
    dev.sort(key=lambda q: q["stt"])
    return test, dev, stats


def write_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def verify_lock(actual_test_sha: str, actual_dev_sha: str) -> tuple[bool, dict]:
    if not HASHES_OUT.exists():
        return False, {}
    locked = json.loads(HASHES_OUT.read_text(encoding="utf-8"))
    test_match = locked.get("test_sha256") == actual_test_sha
    dev_match = locked.get("dev_sha256") == actual_dev_sha
    return test_match and dev_match, locked


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--write",
        action="store_true",
        help="Write the split files (and hash lock) to disk. Without this flag,"
        " the script only computes the categorization and prints the stats.",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing files against the locked hashes (CI mode).",
    )
    args = p.parse_args()

    if not SOURCE.exists():
        print(f"FAIL: {SOURCE} not found", file=sys.stderr)
        return 2

    questions = json.loads(SOURCE.read_text(encoding="utf-8"))
    laws = load_law_metadata()
    in_corpus_codes = {m.full_id for m in laws.values()}
    print(f"In-corpus law codes (KG-loaded): {sorted(in_corpus_codes)}")

    categories: dict[int, str] = {}
    for q in questions:
        categories[q["stt"]] = categorize(q.get("gold_citations_raw"), in_corpus_codes)

    cat_totals: dict[str, int] = defaultdict(int)
    for c in categories.values():
        cat_totals[c] += 1
    print(f"\nCategorization (n={len(questions)}):")
    for c in sorted(cat_totals):
        print(f"  {c:<14} {cat_totals[c]:>4}")

    test, dev, stats = stratified_split(questions, categories)
    assert len(test) + len(dev) == len(questions)

    print(f"\nStratified split (seed={SEED}):")
    print(f"  {'category':<14}{'total':>6}{'test':>6}{'dev':>6}")
    for cat, s in sorted(stats.items()):
        print(f"  {cat:<14}{s['total']:>6}{s['test']:>6}{s['dev']:>6}")
    print(f"  {'TOTAL':<14}{len(questions):>6}{len(test):>6}{len(dev):>6}")

    if not args.write and not args.verify:
        print("\n(Dry run — pass --write to commit, --verify to check lock.)")
        return 0

    # Always compute the split + hashes; either write or verify.
    write_json(TEST_OUT, test)
    write_json(DEV_OUT, dev)
    test_sha = sha256_of(TEST_OUT)
    dev_sha = sha256_of(DEV_OUT)

    if args.verify:
        ok, locked = verify_lock(test_sha, dev_sha)
        if not ok:
            print("\nFAIL: split file hashes do not match the lock.")
            print(f"  current test  sha256: {test_sha}")
            print(f"  current dev   sha256: {dev_sha}")
            print(f"  locked        record: {locked}")
            return 3
        print(f"\nOK — split files match the lock ({HASHES_OUT}).")
        return 0

    # --write path: write lock file
    lock = {
        "script_version": SCRIPT_VERSION,
        "seed": SEED,
        "source_sha256": sha256_of(SOURCE),
        "test_path": str(TEST_OUT),
        "test_sha256": test_sha,
        "test_n": len(test),
        "dev_path": str(DEV_OUT),
        "dev_sha256": dev_sha,
        "dev_n": len(dev),
        "categories_in_test": {
            c: sum(1 for q in test if categories[q["stt"]] == c)
            for c in sorted(cat_totals)
        },
        "categories_in_dev": {
            c: sum(1 for q in dev if categories[q["stt"]] == c)
            for c in sorted(cat_totals)
        },
        "in_corpus_codes": sorted(in_corpus_codes),
    }
    HASHES_OUT.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote:")
    print(f"  {TEST_OUT}  sha256={test_sha[:16]}...  n={len(test)}")
    print(f"  {DEV_OUT}  sha256={dev_sha[:16]}...  n={len(dev)}")
    print(f"  {HASHES_OUT}")
    print("\nCommit all three files. After this, re-running with --verify must succeed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
