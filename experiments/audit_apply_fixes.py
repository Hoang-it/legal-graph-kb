"""audit_apply_fixes.py — Re-aggregate metrics.json files với fixed pairwise logic.

**No new API calls.** Reads cached judge raw outputs (`judge_cache.jsonl`) và
overwrites `pairwise_vs_baseline` / `pairwise_vs_no_retrieval` fields trong
metrics.json với corrected `_vote` mapping.

Backups created:
    data/eval/metrics.json.bak_pre_pairwise_fix
    data/eval/multimodel/metrics.json.bak_pre_pairwise_fix

Usage:
    python -m experiments.audit_apply_fixes
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

# Cache key format: pair_<stt>_<arm_a>_vs_<arm_b>_<swap_id>
_KEY_PAT = re.compile(r"^pair_(\d+)_(.+?)_vs_(.+?)_(ab|ba)$")


def _vote_fixed(w: str, arm_a: str, arm_b: str) -> str:
    """Corrected mapping — see compute_metrics.py:m_pairwise FIX comment."""
    w = (w or "").strip().lower()
    if w == "tie":
        return "tie"
    if w == "a":
        return arm_a
    if w == "b":
        return arm_b
    return f"unknown:{w!r}"


def load_pairwise_cache(cache_path: Path) -> dict[tuple, dict]:
    """Returns (stt, arm_a, arm_b) → {"ab": {data:...}, "ba": {data:...}}."""
    if not cache_path.exists():
        return {}
    out: dict[tuple, dict] = {}
    with cache_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = _KEY_PAT.match(entry.get("key", ""))
            if not m:
                continue
            stt, arm_a, arm_b, swap = m.groups()
            key = (int(stt), arm_a, arm_b)
            if key not in out:
                out[key] = {}
            out[key][swap] = entry.get("result", {})
    return out


def rebuild_pairwise(record_a_arm: str, record_b_arm: str, ab_result: dict,
                     ba_result: dict) -> dict:
    """Replicate m_pairwise output structure but with fixed _vote."""
    w_ab = (ab_result.get("data", {}) or {}).get("winner", "") if isinstance(ab_result.get("data"), dict) else ""
    w_ba = (ba_result.get("data", {}) or {}).get("winner", "") if isinstance(ba_result.get("data"), dict) else ""
    vote_ab = _vote_fixed(w_ab, record_a_arm, record_b_arm)
    vote_ba = _vote_fixed(w_ba, record_a_arm, record_b_arm)
    return {
        "vote_ab": vote_ab,
        "vote_ba": vote_ba,
        "consensus": vote_ab if vote_ab == vote_ba else "split",
        "raw": {"ab": ab_result.get("data"), "ba": ba_result.get("data")},
        "_judge_usage": {
            "prompt_tokens": (ab_result.get("usage", {}).get("prompt_tokens", 0)
                              + ba_result.get("usage", {}).get("prompt_tokens", 0)),
            "completion_tokens": (ab_result.get("usage", {}).get("completion_tokens", 0)
                                   + ba_result.get("usage", {}).get("completion_tokens", 0)),
        },
    }


def fix_r1(metrics_path: Path, cache_path: Path) -> tuple[int, int, int]:
    """Returns (n_records_total, n_records_updated, n_records_unchanged)."""
    if not metrics_path.exists():
        print(f"SKIP: {metrics_path} not found", file=sys.stderr)
        return (0, 0, 0)

    print(f"\n=== R1: {metrics_path} ===")
    bak = metrics_path.with_suffix(metrics_path.suffix + ".bak_pre_pairwise_fix")
    if not bak.exists():
        shutil.copy(metrics_path, bak)
        print(f"  Backup → {bak}")
    else:
        print(f"  Backup already exists: {bak} (not overwriting)")

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    cache = load_pairwise_cache(cache_path)
    print(f"  Cache pairwise entries (grouped by stt+arm_a+arm_b): {len(cache)}")

    # R1: baseline = "graphrag" → pairs are (graphrag, non_baseline_arm)
    PAIRWISE_BASELINE = "graphrag"
    n_total = n_updated = n_unchanged = 0
    for arm, recs in data.items():
        if arm == PAIRWISE_BASELINE:
            continue
        for rec in recs:
            n_total += 1
            stt = rec["stt"]
            old_pw = rec.get("pairwise_vs_baseline")
            ab_ba = cache.get((stt, PAIRWISE_BASELINE, arm))
            if not ab_ba or "ab" not in ab_ba or "ba" not in ab_ba:
                continue
            new_pw = rebuild_pairwise(PAIRWISE_BASELINE, arm, ab_ba["ab"], ab_ba["ba"])
            # Preserve old pairwise prompt_tokens/completion_tokens if existed
            rec["pairwise_vs_baseline"] = new_pw
            if old_pw is None or old_pw.get("vote_ba") != new_pw["vote_ba"]:
                n_updated += 1
            else:
                n_unchanged += 1

    metrics_path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"  Records: {n_total} total, {n_updated} updated, {n_unchanged} unchanged")
    return (n_total, n_updated, n_unchanged)


def fix_r2(metrics_path: Path, cache_path: Path) -> tuple[int, int, int]:
    """R2: pairwise per-model, baseline = elite_no_retrieval__{model}."""
    if not metrics_path.exists():
        print(f"SKIP: {metrics_path} not found", file=sys.stderr)
        return (0, 0, 0)

    print(f"\n=== R2: {metrics_path} ===")
    bak = metrics_path.with_suffix(metrics_path.suffix + ".bak_pre_pairwise_fix")
    if not bak.exists():
        shutil.copy(metrics_path, bak)
        print(f"  Backup → {bak}")
    else:
        print(f"  Backup already exists: {bak}")

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    cache = load_pairwise_cache(cache_path)
    print(f"  Cache pairwise entries (grouped by stt+arm_a+arm_b): {len(cache)}")

    # R2: pairwise is per-model: NR vs GR within same model
    # Find each (model, arm=elite_graphrag, baseline=elite_no_retrieval) combo
    n_total = n_updated = n_unchanged = 0
    for combo, recs in data.items():
        # Only elite_graphrag combos have pairwise_vs_no_retrieval
        if not combo.startswith("elite_graphrag__"):
            continue
        model_safe = combo[len("elite_graphrag__"):]
        nr_combo = f"elite_no_retrieval__{model_safe}"
        for rec in recs:
            n_total += 1
            stt = rec["stt"]
            old_pw = rec.get("pairwise_vs_no_retrieval")
            ab_ba = cache.get((stt, nr_combo, combo))
            if not ab_ba or "ab" not in ab_ba or "ba" not in ab_ba:
                continue
            new_pw = rebuild_pairwise(nr_combo, combo, ab_ba["ab"], ab_ba["ba"])
            rec["pairwise_vs_no_retrieval"] = new_pw
            if old_pw is None or old_pw.get("vote_ba") != new_pw["vote_ba"]:
                n_updated += 1
            else:
                n_unchanged += 1

    metrics_path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"  Records: {n_total} total, {n_updated} updated, {n_unchanged} unchanged")
    return (n_total, n_updated, n_unchanged)


def main():
    print("=" * 70)
    print(" audit_apply_fixes — re-aggregate pairwise from cache (NO API CALLS)")
    print("=" * 70)

    r1 = fix_r1(Path("data/eval/metrics.json"),
                Path("data/eval/judge_cache.jsonl"))
    r2 = fix_r2(Path("data/eval/multimodel/metrics.json"),
                Path("data/eval/multimodel/judge_cache.jsonl"))

    total = r1[0] + r2[0]
    updated = r1[1] + r2[1]
    print(f"\n=== SUMMARY ===")
    print(f"  Total non-baseline records: {total}")
    print(f"  Updated (pairwise_vote_ba flipped): {updated}")
    print(f"  Run `python -m experiments.generate_report` and `generate_multimodel_report` to render new tables")


if __name__ == "__main__":
    main()
