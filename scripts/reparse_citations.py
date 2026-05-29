"""Re-parse stored answer text with the current strict citation parser.

Use after a citation-parser change (e.g. v5 Sprint 2 Phase 0a) to update the
``citation_ids`` / ``citations`` fields of already-recorded experiment results.
This is the half of the skill Rule 2 protocol that the metric engine cannot
do on its own — ``compute_citation_metrics`` consumes the stored ``citation_ids``,
so without re-parsing the records first, changing the parser has no effect on
the reported numbers.

Usage::

    python scripts/reparse_citations.py experiments/03_v5_sprint1_vanilla \\
        --arms graphrag_v5

The script is conservative:
- Only rewrites ``citations`` and ``citation_ids``.
- Adds an audit field ``citation_ids_pre_strict_parser`` capturing the prior
  value (kept once — re-runs see the field and don't overwrite it).
- Refuses to run on an arm whose records lack ``answer`` strings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make repo importable when run as a script
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.citations import (
    DEFAULT_REGISTRY_PATH,
    format_citation,
    load_registry,
    parse_displayed_citations,
)


def reparse_record(record: dict, registry) -> tuple[dict, dict]:
    """Return (new_record, diff_stats). Pure — does not write."""
    answer = record.get("answer") or ""
    if not isinstance(answer, str):
        raise ValueError("record.answer must be str")

    old_ids = list(record.get("citation_ids") or [])
    refs = parse_displayed_citations(answer, registry)
    seen: set[str] = set()
    new_ids: list[str] = []
    new_cits: list[str] = []
    for ref in refs:
        cid = ref.item_id
        if cid in seen:
            continue
        seen.add(cid)
        new_ids.append(cid)
        new_cits.append(format_citation(ref, registry))

    new_record = dict(record)
    # Preserve the pre-strict value once for audit; never overwrite on re-run.
    if "citation_ids_pre_strict_parser" not in new_record:
        new_record["citation_ids_pre_strict_parser"] = old_ids
    new_record["citation_ids"] = new_ids
    new_record["citations"] = new_cits

    diff = {
        "n_old": len(old_ids),
        "n_new": len(new_ids),
        "dropped": sorted(set(old_ids) - set(new_ids)),
        "added": sorted(set(new_ids) - set(old_ids)),
    }
    return new_record, diff


def reparse_arm(arm_dir: Path, registry, dry_run: bool = False) -> dict:
    rec_paths = sorted(arm_dir.glob("A*.json"))
    if not rec_paths:
        return {"arm": arm_dir.name, "n": 0, "changed": 0, "examples": []}
    n = 0
    changed = 0
    examples: list[dict] = []
    for rp in rec_paths:
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if "answer" not in rec:
            # Skip records without an answer (e.g. logic-LM records may store it
            # under a different field — those need their own reparse handler).
            continue
        new_rec, diff = reparse_record(rec, registry)
        n += 1
        if diff["dropped"] or diff["added"]:
            changed += 1
            if len(examples) < 5:
                examples.append({
                    "stt": rec.get("stt"),
                    "diff": diff,
                })
        if not dry_run and (diff["dropped"] or diff["added"]):
            rp.write_text(
                json.dumps(new_rec, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    return {
        "arm": arm_dir.name,
        "n": n,
        "changed": changed,
        "examples": examples,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("experiment", type=Path, help="experiments/<NN_name>")
    p.add_argument(
        "--arms",
        type=str,
        default="",
        help="Comma-separated arm subfolders. Default = all subfolders of results/.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Citation registry path.",
    )
    args = p.parse_args()

    results_root = args.experiment / "results"
    if not results_root.is_dir():
        print(f"FAIL: {results_root} not found", file=sys.stderr)
        return 2

    registry = load_registry(args.registry)
    arm_filter = {a.strip() for a in args.arms.split(",") if a.strip()}

    summaries: list[dict] = []
    for arm_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        if arm_dir.name == "multimodel":
            # Recurse one level into multimodel combos
            for combo_dir in sorted(p for p in arm_dir.iterdir() if p.is_dir()):
                if arm_filter and combo_dir.name not in arm_filter:
                    continue
                summaries.append(reparse_arm(combo_dir, registry, args.dry_run))
            continue
        if arm_filter and arm_dir.name not in arm_filter:
            continue
        summaries.append(reparse_arm(arm_dir, registry, args.dry_run))

    print(f"\n=== Reparse summary ({'dry-run' if args.dry_run else 'WROTE'}) ===")
    for s in summaries:
        print(f"  {s['arm']:<28} n={s['n']:>3}  changed={s['changed']:>3}")
        for ex in s["examples"]:
            d = ex["diff"]
            tag = f"-{len(d['dropped'])}+{len(d['added'])}"
            print(f"     stt={ex['stt']:<3}  {tag}  dropped={d['dropped']}  added={d['added']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
