"""audit_apply_fixes_v2.py — Post-process metrics.json:

1. Split hallucination_rate into:
   - content_hallucination_rate = (misstate + unsup) / max(1, n_claims)
   - invented_citation_rate     = n_invented / max(1, n_total_citations)

2. Tag API-error records (prompt_tokens==0 AND completion_tokens==0
   on non-graphrag arms — graphrag arm doesn't track tokens) so report
   generators can filter them.

3. Recompute from the existing nested fields (n_misstate, n_unsupported,
   n_invented_citations, n_claims, n_total_citations) — NO new judge calls.

Backups created:
    data/eval/metrics.json.bak_pre_v2
    data/eval/multimodel/metrics.json.bak_pre_v2

Usage:
    python -m experiments.audit_apply_fixes_v2
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def _to_int(v) -> int:
    if v is None:
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _recompute_halu_split(halu_block: dict) -> dict:
    """Add content_hallucination_rate + invented_citation_rate fields."""
    out = dict(halu_block)
    n_claims = _to_int(halu_block.get("n_claims"))
    n_misstate = _to_int(halu_block.get("n_misstate"))
    n_unsup = _to_int(halu_block.get("n_unsupported"))
    n_invented = _to_int(halu_block.get("n_invented_citations"))
    n_total_cits = halu_block.get("n_total_citations")
    if n_total_cits is None:
        # Older records: estimate from n_invented + assume valid pool exists
        # In skip_reason='no_valid_citations' case, n_invented == n_total
        if halu_block.get("_skip_reason") == "no_valid_citations":
            n_total_cits = n_invented
        else:
            n_total_cits = None
    n_total_cits_int = _to_int(n_total_cits) if n_total_cits is not None else 0

    # content_hallucination_rate — requires judge claims (n_claims > 0)
    if n_claims > 0:
        out["content_hallucination_rate"] = round((n_misstate + n_unsup) / n_claims, 4)
    else:
        out["content_hallucination_rate"] = None

    # invented_citation_rate — deterministic
    if n_total_cits_int > 0:
        out["invented_citation_rate"] = round(n_invented / n_total_cits_int, 4)
    elif halu_block.get("hallucination_rate") is None:
        out["invented_citation_rate"] = None
    else:
        out["invented_citation_rate"] = 0.0

    if "n_total_citations" not in out and n_total_cits is not None:
        out["n_total_citations"] = n_total_cits_int
    return out


def _is_api_error(rec_record: dict) -> bool:
    """Detect silent API failures: prompt_tokens==0 AND completion_tokens==0.
    Only meaningful for arms that DO track tokens (elite_* và llm_only).
    Graphrag arm uses estimated tokens — won't be 0,0."""
    cost = rec_record.get("cost") or {}
    pt = cost.get("prompt_tokens")
    ct = cost.get("completion_tokens")
    estimated = cost.get("estimated", False)
    if estimated:
        return False  # graphrag estimated tokens, never 0,0
    return pt == 0 and ct == 0


def fix_file(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        print(f"SKIP: {metrics_path}")
        return {}

    bak = metrics_path.with_suffix(metrics_path.suffix + ".bak_pre_v2")
    if not bak.exists():
        shutil.copy(metrics_path, bak)
        print(f"  Backup → {bak}")

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    stats = {}
    for arm, recs in data.items():
        n_total = len(recs)
        n_api_err = 0
        n_halu_split_added = 0
        for r in recs:
            # Tag API error
            if _is_api_error(r):
                r["api_error"] = True
                n_api_err += 1
            else:
                r["api_error"] = False
            # Split halu
            if isinstance(r.get("hallucination"), dict):
                r["hallucination"] = _recompute_halu_split(r["hallucination"])
                n_halu_split_added += 1
        stats[arm] = {
            "n_total": n_total,
            "n_api_error": n_api_err,
            "n_halu_split": n_halu_split_added,
        }

    metrics_path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return stats


def main():
    print("=" * 70)
    print(" audit_apply_fixes_v2 — split hallucination + tag API errors")
    print("=" * 70)

    for label, path in [
        ("R1", Path("data/eval/metrics.json")),
        ("R2", Path("data/eval/multimodel/metrics.json")),
    ]:
        print(f"\n=== {label}: {path} ===")
        stats = fix_file(path)
        for arm, s in stats.items():
            api_tag = f"  [{s['n_api_error']} api_error]" if s['n_api_error'] else ""
            print(f"  {arm:<40} n_total={s['n_total']}, halu_split=+{s['n_halu_split']}{api_tag}")


if __name__ == "__main__":
    main()
