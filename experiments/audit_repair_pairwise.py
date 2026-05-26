"""audit_repair_pairwise.py — re-parse existing pairwise judge cache với fixed _vote.

**No new API calls.** Đọc lại raw judge outputs đã cache:
    data/eval/judge_cache.jsonl                  (R1)
    data/eval/multimodel/judge_cache.jsonl       (R2)

Apply CORRECTED `_vote` logic: label A always refers to ans_a, label B always refers to ans_b
(regardless of display order — see audit doc cho lý do).

Compare new vs old aggregation, output:
    reports/pairwise_repaired.md
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Cache key format: pair_<stt>_<arm_a>_vs_<arm_b>_<swap_id>
# ---------------------------------------------------------------------------
_KEY_PAT = re.compile(r"^pair_(\d+)_(.+?)_vs_(.+?)_(ab|ba)$")


def _vote_fixed(w: str, record_a_arm: str, record_b_arm: str) -> str:
    """FIXED vote: label is stable; A→ans_a, B→ans_b, regardless of a_first.

    Old buggy code inverted vote_ba.  Correct mapping:
      w='a' → record_a's arm
      w='b' → record_b's arm
      w='tie' → 'tie'
    """
    w = (w or "").lower().strip()
    if w == "tie":
        return "tie"
    if w == "a":
        return record_a_arm
    if w == "b":
        return record_b_arm
    return f"unknown_winner:{w!r}"


def _vote_old(w: str, a_first: bool, record_a_arm: str, record_b_arm: str) -> str:
    """OLD buggy logic — for reproducing original tables."""
    w = (w or "").lower().strip()
    if w == "tie":
        return "tie"
    if (w == "a" and a_first) or (w == "b" and not a_first):
        return record_a_arm
    return record_b_arm


def load_cache(path: Path) -> dict[tuple, dict]:
    """Trả về dict (stt, arm_a, arm_b, swap_id) → judge_output."""
    out = {}
    if not path.exists():
        print(f"Missing: {path}")
        return out
    with path.open(encoding="utf-8") as f:
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
            data = entry.get("result", {}).get("data") or {}
            winner = data.get("winner", "")
            out[(int(stt), arm_a, arm_b, swap)] = winner
    return out


def aggregate(cache_entries: dict, vote_fn):
    """Group by (arm_a, arm_b) pair, return consensus + per-direction tallies.
    vote_fn(w, a_first, arm_a, arm_b) → vote string.
    """
    # Group ab+ba per (stt, arm_a, arm_b)
    pairs = defaultdict(dict)  # (stt, arm_a, arm_b) → {'ab': w, 'ba': w}
    for (stt, arm_a, arm_b, swap), w in cache_entries.items():
        pairs[(stt, arm_a, arm_b)][swap] = w

    # For each (arm_a, arm_b) comparison, tally
    by_pair = defaultdict(lambda: {
        "consensus": Counter(),
        "vote_ab": Counter(),
        "vote_ba": Counter(),
        "n_complete": 0,
        "n_total": 0,
    })

    for (stt, arm_a, arm_b), votes in pairs.items():
        key = (arm_a, arm_b)
        by_pair[key]["n_total"] += 1
        if "ab" not in votes or "ba" not in votes:
            continue
        by_pair[key]["n_complete"] += 1
        v_ab = vote_fn(votes["ab"], True, arm_a, arm_b)
        v_ba = vote_fn(votes["ba"], False, arm_a, arm_b)
        by_pair[key]["vote_ab"][v_ab] += 1
        by_pair[key]["vote_ba"][v_ba] += 1
        consensus = v_ab if v_ab == v_ba else "split"
        by_pair[key]["consensus"][consensus] += 1
    return dict(by_pair)


def format_consensus_table(agg_old, agg_new, source_label):
    lines = [f"## Source: {source_label}", ""]
    pairs = sorted(set(agg_old.keys()) | set(agg_new.keys()))
    for pair in pairs:
        arm_a, arm_b = pair
        old = agg_old.get(pair)
        new = agg_new.get(pair)
        if not new:
            continue
        n_comp = new["n_complete"]
        lines.append(f"### `{arm_a}` vs `{arm_b}` (n_complete={n_comp})")
        lines.append("")
        # Consensus comparison
        lines.append("**Consensus (OLD buggy vs FIXED):**")
        lines.append("")
        lines.append("| Consensus | OLD count | OLD % | FIXED count | FIXED % |")
        lines.append("|---|---:|---:|---:|---:|")
        all_keys = sorted(set(old["consensus"].keys()) | set(new["consensus"].keys()))
        for k in all_keys:
            o = old["consensus"].get(k, 0)
            n = new["consensus"].get(k, 0)
            op = o / n_comp * 100 if n_comp else 0
            np_ = n / n_comp * 100 if n_comp else 0
            label = "**" + k + "**" if k in (arm_a, arm_b) else k
            lines.append(f"| {label} | {o} | {op:.1f}% | {n} | {np_:.1f}% |")
        lines.append("")

        # Position swap detail (FIXED)
        lines.append("**Position-swap detail (FIXED interpretation):**")
        lines.append("")
        lines.append("| Vote | A=arm_a B=arm_b | A=arm_b B=arm_a |")
        lines.append("|---|---:|---:|")
        all_v = sorted(set(new["vote_ab"].keys()) | set(new["vote_ba"].keys()))
        for v in all_v:
            ab = new["vote_ab"].get(v, 0)
            ba = new["vote_ba"].get(v, 0)
            lines.append(f"| {v} | {ab} | {ba} |")
        lines.append("")
    return lines


def main():
    out = []
    out.append("# Pairwise Re-aggregation (Bug Repaired)")
    out.append("")
    out.append("**Bug**: `_vote(w, a_first=False)` inverted vote_ba assignment.")
    out.append("Original code assumed labels were swapped when `a_first=False`, but")
    out.append("`_ask()` only swaps display order — label-content pairing stays stable.")
    out.append("Result: any consistent agreement got recorded as 'split'.")
    out.append("")
    out.append("**Fix**: `_vote(w)` always returns `record_a` for w='a', `record_b` for w='b'.")
    out.append("No new judge calls — re-parse cached raw outputs.")
    out.append("")

    # R1
    print("Loading R1 cache...")
    cache_r1 = load_cache(Path("data/eval/judge_cache.jsonl"))
    print(f"  R1 cache pair entries: {len(cache_r1)}")
    if cache_r1:
        agg_old_r1 = aggregate(cache_r1, _vote_old)
        agg_new_r1 = aggregate(cache_r1, lambda w, af, a, b: _vote_fixed(w, a, b))
        out.extend(format_consensus_table(agg_old_r1, agg_new_r1, "R1 (5-arm, gpt-4o-mini)"))

    # R2
    print("Loading R2 cache...")
    cache_r2 = load_cache(Path("data/eval/multimodel/judge_cache.jsonl"))
    print(f"  R2 cache pair entries: {len(cache_r2)}")
    if cache_r2:
        agg_old_r2 = aggregate(cache_r2, _vote_old)
        agg_new_r2 = aggregate(cache_r2, lambda w, af, a, b: _vote_fixed(w, a, b))
        out.extend(format_consensus_table(agg_old_r2, agg_new_r2, "R2 (multi-model)"))

    # Summary diff
    out.append("\n## Summary delta (fixed − old) consensus counts")
    out.append("")
    out.append("| Source | Pair | Δ split | Δ winner=record_a | Δ winner=record_b | Δ tie |")
    out.append("|---|---|---:|---:|---:|---:|")
    for src_label, agg_old, agg_new in [
        ("R1", agg_old_r1 if cache_r1 else {}, agg_new_r1 if cache_r1 else {}),
        ("R2", agg_old_r2 if cache_r2 else {}, agg_new_r2 if cache_r2 else {}),
    ]:
        for pair in sorted(agg_new.keys()):
            arm_a, arm_b = pair
            old_c = agg_old.get(pair, {}).get("consensus", Counter())
            new_c = agg_new[pair]["consensus"]
            d_split = new_c.get("split", 0) - old_c.get("split", 0)
            d_a = new_c.get(arm_a, 0) - old_c.get(arm_a, 0)
            d_b = new_c.get(arm_b, 0) - old_c.get(arm_b, 0)
            d_tie = new_c.get("tie", 0) - old_c.get("tie", 0)
            out.append(f"| {src_label} | `{arm_a}` vs `{arm_b}` | "
                       f"{d_split:+d} | {d_a:+d} (a) | {d_b:+d} (b) | {d_tie:+d} |")
    out.append("")

    out_path = Path("reports/pairwise_repaired.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"\nSaved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
