"""compute_significance.py — Significance tests cho top claims trong paper.

Tests:
- McNemar test (paired binary outcomes, e.g., Prolog success)
- Bootstrap 95% CI (10k resamples) cho continuous metrics
- Bonferroni correction tại α = 0.05 / n_claims

Top 5 claims (paper-defensible — chọn theo reports hiện tại sau audit):

R1 (5-arm):
  C1: llm_only **beats** graphrag on pairwise judge (~86% on consistent subset)
      → Test: McNemar trên paired (graphrag vs llm_only)
  C2: graphrag **beats** elite_no_retrieval on faithfulness (Es 2024 RAGAS)
      → Test: bootstrap CI(diff) trên faithfulness
  C3: elite_no_retrieval has highest prolog_success_rate among elite arms
      → Test: McNemar (NR vs Ontology, NR vs GraphRAG)

R2 (multi-model):
  C4: For gpt-5-mini, NR **beats** GR on pairwise (52.5% vs 30.5%)
      → Test: McNemar trên paired (NR vs GR within gpt-5-mini)
  C5: For gpt-5-mini, GR does NOT improve prolog_success vs NR
      → Test: McNemar trên Prolog success (after API-error filter)

Output: reports/significance.md
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from collections import Counter

random.seed(42)

R1_METRICS = Path("data/eval/metrics.json")
R2_METRICS = Path("data/eval/multimodel/metrics.json")
OUT = Path("reports/significance.md")

ALPHA = 0.05
# N_CLAIMS dynamic: 5 paper-baseline claims + 2 logic-extraction claims (chỉ tested
# khi `elite_graphrag_logic` xuất hiện trong R1 metrics). BONFERRONI_ALPHA được
# update trong main() trước khi gọi claim_*() để mỗi claim có flag đúng.
N_CLAIMS = 5  # default; main() sẽ overwrite
BONFERRONI_ALPHA = ALPHA / N_CLAIMS
N_BOOTSTRAP = 10_000


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def mcnemar_pvalue(b: int, c: int) -> float:
    """McNemar exact test (binomial). b, c are discordant pairs.
    H0: equal performance → expected b == c.
    Returns 2-sided p-value via exact binomial test against B(b+c, 0.5).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # 2 * P(X <= k | X ~ Binomial(n, 0.5))
    cum = 0.0
    for i in range(k + 1):
        cum += math.comb(n, i) * (0.5 ** n)
    p = 2 * cum
    return min(p, 1.0)


def bootstrap_ci(diff_vals: list[float], n_resample: int = N_BOOTSTRAP,
                 ci: float = 0.95) -> tuple[float, float, float]:
    """Returns (mean_diff, lo, hi) for the mean of diff_vals (paired)."""
    if not diff_vals:
        return (0.0, 0.0, 0.0)
    n = len(diff_vals)
    means = []
    for _ in range(n_resample):
        sample = [diff_vals[random.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((1 - ci) / 2 * n_resample)
    hi_idx = int((1 + ci) / 2 * n_resample) - 1
    mean_diff = sum(diff_vals) / n
    return (mean_diff, means[lo_idx], means[hi_idx])


def _pair_by_stt(recs_a: list[dict], recs_b: list[dict]):
    """Return list of (rec_a, rec_b) paired by stt."""
    by_stt_b = {r["stt"]: r for r in recs_b}
    paired = []
    for ra in recs_a:
        rb = by_stt_b.get(ra["stt"])
        if rb is not None:
            paired.append((ra, rb))
    return paired


def _value_chain(rec, chain):
    v = rec
    for k in chain:
        if not isinstance(v, dict):
            return None
        v = v.get(k)
    return v


def _bool_value(rec, chain):
    v = _value_chain(rec, chain)
    if v is None:
        return None
    return bool(v)


def _is_api_error(rec):
    return rec.get("api_error", False)


# ---------------------------------------------------------------------------
# Claim runners
# ---------------------------------------------------------------------------

def claim_pairwise_winner(recs_a: list[dict], recs_b: list[dict],
                          arm_a: str, arm_b: str, pw_field: str,
                          label: str) -> dict:
    """McNemar on pairwise consensus: count which arm wins consistently.
    pw_field = field name in non-baseline records ('pairwise_vs_baseline'
    for R1 or 'pairwise_vs_no_retrieval' for R2).
    The non-baseline records hold the pw data; recs_b is the non-baseline.
    """
    # Pairwise stored in arm_b's records (non-baseline)
    n_a_wins = n_b_wins = n_tie = n_split = 0
    for r in recs_b:
        if _is_api_error(r):
            continue
        pw = r.get(pw_field)
        if not pw:
            continue
        c = pw.get("consensus", "")
        if c == arm_a:
            n_a_wins += 1
        elif c == arm_b:
            n_b_wins += 1
        elif c == "tie":
            n_tie += 1
        else:
            n_split += 1
    # McNemar on discordant pairs: only consider consistent verdicts
    p = mcnemar_pvalue(n_a_wins, n_b_wins)
    n_consistent = n_a_wins + n_b_wins + n_tie
    n_total = n_a_wins + n_b_wins + n_tie + n_split
    return {
        "label": label,
        "test": "McNemar (pairwise consensus)",
        "n_total": n_total,
        "n_consistent": n_consistent,
        "n_a_wins": n_a_wins,
        "n_b_wins": n_b_wins,
        "n_tie": n_tie,
        "n_split": n_split,
        "p_value": p,
        "significant_bonferroni": p < BONFERRONI_ALPHA,
        "winner_if_sig": (arm_a if n_a_wins > n_b_wins else arm_b) if p < BONFERRONI_ALPHA else None,
    }


def claim_paired_continuous(recs_a: list[dict], recs_b: list[dict],
                            chain: list[str], arm_a: str, arm_b: str,
                            label: str) -> dict:
    """Bootstrap CI cho mean diff (recs_a − recs_b) trên paired records."""
    paired = _pair_by_stt(recs_a, recs_b)
    diffs = []
    for ra, rb in paired:
        if _is_api_error(ra) or _is_api_error(rb):
            continue
        va = _value_chain(ra, chain)
        vb = _value_chain(rb, chain)
        if va is None or vb is None:
            continue
        diffs.append(va - vb)
    if not diffs:
        return {"label": label, "n": 0, "_skip": "no_paired_valid"}
    mean_d, lo, hi = bootstrap_ci(diffs)
    sig = (lo > 0) or (hi < 0)  # CI excludes 0
    return {
        "label": label,
        "test": f"Bootstrap 95% CI on mean({arm_a} − {arm_b})",
        "n_paired": len(diffs),
        "mean_diff": mean_d,
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_excludes_zero": sig,
        "winner": arm_a if mean_d > 0 else arm_b,
    }


def claim_paired_binary(recs_a: list[dict], recs_b: list[dict],
                        chain: list[str], arm_a: str, arm_b: str,
                        label: str) -> dict:
    """McNemar trên paired binary outcomes (e.g., prolog_success)."""
    paired = _pair_by_stt(recs_a, recs_b)
    b = c = 0  # a=succ b=fail (b), a=fail b=succ (c)
    n_concordant_yes = n_concordant_no = 0
    for ra, rb in paired:
        if _is_api_error(ra) or _is_api_error(rb):
            continue
        va = _bool_value(ra, chain)
        vb = _bool_value(rb, chain)
        if va is None or vb is None:
            continue
        if va and not vb:
            b += 1
        elif not va and vb:
            c += 1
        elif va and vb:
            n_concordant_yes += 1
        else:
            n_concordant_no += 1
    p = mcnemar_pvalue(b, c)
    return {
        "label": label,
        "test": "McNemar (paired binary)",
        "n_paired_clean": b + c + n_concordant_yes + n_concordant_no,
        "discord_a_only": b,  # a=succ, b=fail
        "discord_b_only": c,  # b=succ, a=fail
        "concord_both_yes": n_concordant_yes,
        "concord_both_no": n_concordant_no,
        "p_value": p,
        "significant_bonferroni": p < BONFERRONI_ALPHA,
        "winner_if_sig": (arm_a if b > c else arm_b) if p < BONFERRONI_ALPHA else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not R1_METRICS.exists() or not R2_METRICS.exists():
        print("Missing metrics files", file=sys.stderr)
        return 1

    r1 = json.loads(R1_METRICS.read_text(encoding="utf-8"))
    r2 = json.loads(R2_METRICS.read_text(encoding="utf-8"))

    # Recompute Bonferroni dynamically based on which arms are present
    global N_CLAIMS, BONFERRONI_ALPHA
    extra_logic_claims = 2 if "elite_graphrag_logic" in r1 else 0
    N_CLAIMS = 5 + extra_logic_claims
    BONFERRONI_ALPHA = ALPHA / N_CLAIMS

    results = []

    # C1: llm_only beats graphrag (R1 pairwise)
    results.append(claim_pairwise_winner(
        recs_a=r1["graphrag"], recs_b=r1["llm_only"],
        arm_a="graphrag", arm_b="llm_only",
        pw_field="pairwise_vs_baseline",
        label="C1: llm_only beats graphrag (R1 pairwise)",
    ))

    # C2: graphrag beats elite_no_retrieval on faithfulness (R1)
    results.append(claim_paired_continuous(
        recs_a=r1["graphrag"], recs_b=r1["elite_no_retrieval"],
        chain=["faithfulness", "faithfulness"],
        arm_a="graphrag", arm_b="elite_no_retrieval",
        label="C2: graphrag faithfulness > elite_no_retrieval (R1)",
    ))

    # C3: elite_no_retrieval prolog_success > elite_ontology (R1 paired)
    results.append(claim_paired_binary(
        recs_a=r1["elite_no_retrieval"], recs_b=r1["elite_ontology"],
        chain=["prolog_rollback", "prolog_success"],
        arm_a="elite_no_retrieval", arm_b="elite_ontology",
        label="C3a: NR prolog_success > Ontology (R1)",
    ))
    results.append(claim_paired_binary(
        recs_a=r1["elite_no_retrieval"], recs_b=r1["elite_graphrag"],
        chain=["prolog_rollback", "prolog_success"],
        arm_a="elite_no_retrieval", arm_b="elite_graphrag",
        label="C3b: NR prolog_success > GraphRAG (R1)",
    ))

    # C4: NR beats GR for gpt-5-mini (R2 pairwise)
    results.append(claim_pairwise_winner(
        recs_a=r2["elite_no_retrieval__gpt-5-mini"],
        recs_b=r2["elite_graphrag__gpt-5-mini"],
        arm_a="elite_no_retrieval__gpt-5-mini",
        arm_b="elite_graphrag__gpt-5-mini",
        pw_field="pairwise_vs_no_retrieval",
        label="C4: NR beats GR for gpt-5-mini (R2 pairwise)",
    ))

    # C5: GR vs NR prolog_success for gpt-5-mini (paired binary, after API filter)
    results.append(claim_paired_binary(
        recs_a=r2["elite_no_retrieval__gpt-5-mini"],
        recs_b=r2["elite_graphrag__gpt-5-mini"],
        chain=["prolog_rollback", "prolog_success"],
        arm_a="elite_no_retrieval__gpt-5-mini",
        arm_b="elite_graphrag__gpt-5-mini",
        label="C5: GR vs NR prolog_success for gpt-5-mini (after API-error exclude)",
    ))

    # Phase 5 logic-extraction claims — only ran nếu arm `elite_graphrag_logic`
    # đã có metrics. Pre-registered hypothesis từ plan_logic_extraction.md §7:
    #   prolog_success(logic) ≥ prolog_success(semantic) + 5pp
    if "elite_graphrag_logic" in r1:
        # C6: elite_graphrag_logic prolog_success > elite_graphrag (main hypothesis)
        results.append(claim_paired_binary(
            recs_a=r1["elite_graphrag_logic"], recs_b=r1["elite_graphrag"],
            chain=["prolog_rollback", "prolog_success"],
            arm_a="elite_graphrag_logic", arm_b="elite_graphrag",
            label="C6: elite_graphrag_logic prolog_success > elite_graphrag (R1, main hypothesis)",
        ))
        # C7: faithfulness diff (secondary, bootstrap CI)
        results.append(claim_paired_continuous(
            recs_a=r1["elite_graphrag_logic"], recs_b=r1["elite_graphrag"],
            chain=["faithfulness", "faithfulness"],
            arm_a="elite_graphrag_logic", arm_b="elite_graphrag",
            label="C7: elite_graphrag_logic faithfulness > elite_graphrag (R1, secondary)",
        ))

    # Write report
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# Significance Tests — Top {N_CLAIMS} Paper Claims")
    lines.append("")
    lines.append(f"- α = {ALPHA}")
    lines.append(f"- Bonferroni correction for {N_CLAIMS} claims → α_bonf = {BONFERRONI_ALPHA:.4f}")
    lines.append(f"- Bootstrap resamples: {N_BOOTSTRAP:,}")
    lines.append(f"- API-error records excluded from all tests")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    for r in results:
        lines.append(f"### {r['label']}")
        lines.append("")
        lines.append(f"- Test: `{r.get('test', '?')}`")
        for k, v in r.items():
            if k in ("label", "test"):
                continue
            if isinstance(v, float):
                lines.append(f"- {k}: {v:.4f}")
            else:
                lines.append(f"- {k}: {v}")
        lines.append("")

    # Summary
    lines.append("## Summary — Defensible at α_bonf = 0.01")
    lines.append("")
    lines.append("| Claim | Test | Stat key | Value | Defensible? |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        p = r.get("p_value")
        ci_excl = r.get("ci_excludes_zero")
        if p is not None:
            stat_str = f"p = {p:.4f}"
            defensible = bool(r.get("significant_bonferroni"))
        elif ci_excl is not None:
            stat_str = f"CI = [{r.get('ci_lo', 0):.4f}, {r.get('ci_hi', 0):.4f}]"
            defensible = bool(ci_excl)
        else:
            stat_str = "—"
            defensible = False
        emoji = "✓ DEFENSIBLE" if defensible else "✗ DROP"
        lines.append(f"| {r['label']} | {r.get('test','?')} | — | {stat_str} | {emoji} |")
    lines.append("")
    lines.append("**Reminder**: 'DROP' claims must NOT appear as headline findings in paper "
                 "(insufficient evidence at multiple-comparison-corrected α).")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
