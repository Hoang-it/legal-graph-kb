"""validate_extraction.py — Validate extracted logic JSON files.

5 checks per clause:
  1. JSON schema validity
  2. Predicate vocabulary check (canonical or flagged)
  3. Numerical value cross-check vs regex_facts
  4. Reference resolution (do refs exist in KG schema?)
  5. Structural completeness (every rule has if+then, every condition has predicate+operator+value)

Output: per-clause score + overall accuracy + gate verdict.

Per Phase 2 of plan_logic_extraction.md, gate is:
  - Predicate accuracy ≥ 85%
  - Numerical value accuracy ≥ 95%
  - Overall structural pass ≥ 80%

CLI:
    python -m experiments.validate_extraction               # validate all extracted
    python -m experiments.validate_extraction --pilot      # only pilot subset
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

EXTRACT_DIR = Path("data/eval/extracted_logic")
REPORT_OUT = Path("reports/logic_extraction_validation.md")

# Canonical predicate vocabulary (from logic_extraction_schema.md §2)
CANONICAL_PREDICATES = {
    # Person facts
    "age", "gender", "years_contributed", "months_contributed",
    "years_before_2014", "years_after_2014", "years_after_2025",
    "so_con", "disability_percentage", "work_condition",
    # Money facts
    "average_salary", "base_salary", "monthly_amount", "contribution_rate",
    "pension_rate_pct", "support_percentage", "standard_amount",
    # Time facts
    "so_thang", "so_ngay", "time_period_months", "time_period_years",
    # Insurance/benefit concepts (from elite CONCEPT_SPECS)
    "social_insurance", "mandatory_social_insurance", "voluntary_social_insurance",
    "unemployment_insurance", "health_insurance", "pension", "early_retirement",
    "pension_rate", "one_time_social_insurance", "maternity", "sick_leave",
    "work_accident", "survivorship", "contribution", "contribution_salary",
    "employer_contribution", "employee_contribution", "retirement_age",
    "disability", "hazardous_work", "labor_contract", "legal_dossier",
    "social_insurance_book", "complaint", "state_support", "reservation",
    "pension_adjustment", "prohibited_acts", "foreign_worker", "employee",
    "employer", "one_time_social_insurance_dossier",
    # Compound conclusions allowed
    "eligible_pension", "eligible_maternity", "eligible_sick_leave",
    "eligible_one_time", "eligible_survivorship",
    # ─── Phase 2 vocab expansion (after pilot 30 review) ───
    # Responsibility / obligation predicates (legitimate gap found in pilot)
    "responsibility", "obligation", "right", "prohibited",
    "BHXH_agency_responsibility", "state_agency_responsibility",
    "employer_responsibility", "employee_responsibility",
    # Payment synonyms
    "mandatory_payment", "payment_method", "payment_deadline",
    # Procedural responses
    "BHXH_agency_response", "agency_decision", "appeal_outcome",
}

VALID_OPERATORS = {">=", "<=", ">", "<", "=", "in_range", "in", "formula"}
VALID_UNITS = {"year", "month", "day", "percent", "vnd", "vnd_per_month",
                "child", "times", "enum", None}
VALID_ENTITIES = {"NLĐ", "NSDLĐ", "BHXH_agency", "Nhà_nước", "Tòa_án",
                   "Cơ_quan_thanh_tra", "Đại_diện_NLĐ", "Hộ_kinh_doanh",
                   "Hộ_gia_đình", "Doanh_nghiệp"}


def check_schema(extr: dict) -> list[str]:
    """Return list of schema violation messages. Empty = pass."""
    issues = []
    required_keys = {"conditions", "thresholds", "rules", "exceptions",
                     "references", "defines", "actors", "procedure_steps",
                     "extractor_confidence", "non_canonical_flags"}
    missing = required_keys - set(extr.keys())
    if missing:
        issues.append(f"missing keys: {missing}")
    if extr.get("_parse_error"):
        issues.append("JSON parse error")
    return issues


def check_predicates(extr: dict) -> tuple[int, int, list[str]]:
    """Return (n_canonical, n_total, list of non-canonical predicates)."""
    n_total = n_canonical = 0
    non_canonical = []
    for c in extr.get("conditions", []):
        pred = c.get("predicate")
        if not pred: continue
        n_total += 1
        if pred in CANONICAL_PREDICATES:
            n_canonical += 1
        else:
            non_canonical.append(pred)
    for r in extr.get("rules", []):
        pred = r.get("then_predicate")
        if not pred: continue
        n_total += 1
        if pred in CANONICAL_PREDICATES:
            n_canonical += 1
        else:
            non_canonical.append(pred)
    return n_canonical, n_total, non_canonical


def check_operators(extr: dict) -> tuple[int, int]:
    """Return (n_valid_ops, n_total_ops)."""
    n_total = n_valid = 0
    for c in extr.get("conditions", []):
        op = c.get("operator")
        if not op: continue
        n_total += 1
        if op in VALID_OPERATORS:
            n_valid += 1
    return n_valid, n_total


def check_numerical_cross_validation(extr: dict, regex_facts: dict) -> tuple[int, int]:
    """Cross-validate numbers trong extraction vs regex facts.
    Returns (n_matched, n_total_in_extraction)."""
    # Collect numbers from extraction
    extracted_nums = set()
    for c in extr.get("conditions", []):
        v = c.get("value")
        if isinstance(v, (int, float)):
            extracted_nums.add(float(v))
    for t in extr.get("thresholds", []):
        v = t.get("value")
        if isinstance(v, (int, float)):
            extracted_nums.add(float(v))

    # Regex facts: combined number set
    regex_nums = set(map(float,
        regex_facts.get("percentages", [])
        + regex_facts.get("years", [])
        + regex_facts.get("months", [])
        + regex_facts.get("days", [])
        + regex_facts.get("vnd_amounts", [])))

    if not extracted_nums:
        return 0, 0
    matched = sum(1 for n in extracted_nums if n in regex_nums)
    return matched, len(extracted_nums)


def check_references(extr: dict, regex_facts: dict) -> tuple[int, int]:
    """Verify extracted references vs regex-extracted Điều X."""
    extracted_refs = set()
    for r in extr.get("references", []):
        art = r.get("article")
        if art is not None:
            extracted_refs.add(int(art))
    regex_refs = set(a for a, _ in regex_facts.get("references", []))
    if not extracted_refs:
        return 0, 0
    matched = sum(1 for a in extracted_refs if a in regex_refs)
    return matched, len(extracted_refs)


def check_structural_completeness(extr: dict) -> tuple[int, int]:
    """Returns (n_complete, n_total) checking rule + condition shape."""
    n_total = n_complete = 0
    for c in extr.get("conditions", []):
        n_total += 1
        if c.get("predicate") and c.get("operator") and c.get("value") is not None:
            n_complete += 1
    for r in extr.get("rules", []):
        n_total += 1
        if (r.get("name")
            and "if_conditions_idx" in r
            and r.get("then_predicate")
            and r.get("conclusion_type")):
            n_complete += 1
    return n_complete, n_total


def validate_clause(rec: dict) -> dict:
    extr = rec.get("extraction", {})
    regex = rec.get("regex_facts", {})

    schema_issues = check_schema(extr)
    n_can, n_tot_pred, non_canonical = check_predicates(extr)
    n_valid_op, n_tot_op = check_operators(extr)
    n_match_num, n_tot_num = check_numerical_cross_validation(extr, regex)
    n_match_ref, n_tot_ref = check_references(extr, regex)
    n_complete, n_tot_struct = check_structural_completeness(extr)

    confidence = extr.get("extractor_confidence", 0.0)

    return {
        "clause_id": rec.get("clause_id"),
        "schema_issues": schema_issues,
        "n_canonical_predicates": (n_can, n_tot_pred),
        "non_canonical_predicates": non_canonical,
        "n_valid_operators": (n_valid_op, n_tot_op),
        "n_numeric_matched_regex": (n_match_num, n_tot_num),
        "n_ref_matched_regex": (n_match_ref, n_tot_ref),
        "n_structurally_complete": (n_complete, n_tot_struct),
        "confidence": confidence,
        "n_conditions": len(extr.get("conditions", [])),
        "n_rules": len(extr.get("rules", [])),
        "n_thresholds": len(extr.get("thresholds", [])),
        "n_references": len(extr.get("references", [])),
        "n_defines": len(extr.get("defines", [])),
    }


def aggregate(results: list[dict]) -> dict:
    """Compute overall accuracy metrics from per-clause results."""
    def sum_pair(key):
        s_match = sum(r[key][0] for r in results)
        s_total = sum(r[key][1] for r in results)
        return s_match, s_total, (s_match / s_total) if s_total > 0 else None

    pred_match, pred_tot, pred_acc = sum_pair("n_canonical_predicates")
    op_match, op_tot, op_acc = sum_pair("n_valid_operators")
    num_match, num_tot, num_acc = sum_pair("n_numeric_matched_regex")
    ref_match, ref_tot, ref_acc = sum_pair("n_ref_matched_regex")
    struct_match, struct_tot, struct_acc = sum_pair("n_structurally_complete")

    n_with_issues = sum(1 for r in results if r["schema_issues"])
    avg_confidence = sum(r["confidence"] for r in results) / max(1, len(results))

    # Aggregate non-canonical predicates
    all_non_can = []
    for r in results:
        all_non_can.extend(r["non_canonical_predicates"])
    non_can_freq = Counter(all_non_can)

    return {
        "n_clauses": len(results),
        "schema_pass_rate": (len(results) - n_with_issues) / len(results),
        "predicate_accuracy": pred_acc,
        "predicate_counts": (pred_match, pred_tot),
        "operator_accuracy": op_acc,
        "operator_counts": (op_match, op_tot),
        "numerical_accuracy_vs_regex": num_acc,
        "numerical_counts": (num_match, num_tot),
        "reference_accuracy_vs_regex": ref_acc,
        "reference_counts": (ref_match, ref_tot),
        "structural_completeness": struct_acc,
        "structural_counts": (struct_match, struct_tot),
        "avg_confidence": avg_confidence,
        "top_non_canonical_predicates": non_can_freq.most_common(10),
    }


def check_gate(agg: dict) -> tuple[bool, list[str]]:
    """Phase 2 gate per plan_logic_extraction.md §8."""
    fails = []
    if (agg["predicate_accuracy"] or 0) < 0.85:
        fails.append(f"Predicate accuracy {agg['predicate_accuracy']:.2%} < 85% target")
    if (agg["numerical_accuracy_vs_regex"] or 0) < 0.95:
        fails.append(f"Numerical accuracy {agg['numerical_accuracy_vs_regex']:.2%} < 95% target")
    if (agg["reference_accuracy_vs_regex"] or 0) < 0.90:
        fails.append(f"Reference accuracy {agg['reference_accuracy_vs_regex']:.2%} < 90% target")
    if (agg["structural_completeness"] or 0) < 0.80:
        fails.append(f"Structural completeness {agg['structural_completeness']:.2%} < 80% target")
    return len(fails) == 0, fails


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verbose", action="store_true", help="Print per-clause details")
    args = p.parse_args()

    if not EXTRACT_DIR.exists():
        print(f"No extracted files in {EXTRACT_DIR}", file=sys.stderr)
        return 1

    files = sorted(EXTRACT_DIR.glob("*.json"))
    if not files:
        print(f"No extracted JSON files found in {EXTRACT_DIR}", file=sys.stderr)
        return 1

    results = []
    for fp in files:
        try:
            rec = json.loads(fp.read_text(encoding="utf-8"))
            results.append(validate_clause(rec))
        except Exception as e:
            print(f"  ✗ {fp.name}: {e}", file=sys.stderr)

    agg = aggregate(results)
    passed, fails = check_gate(agg)

    # ─── Report ───
    lines = ["# Logic Extraction — Validation Report", ""]
    lines.append(f"Files validated: **{agg['n_clauses']}**")
    lines.append(f"Date: {Path('.').stat().st_mtime}")
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Metric | Value | Target | Status |")
    lines.append("|---|---:|---:|:---:|")

    def row(label, val, target, suffix=""):
        if val is None:
            return f"| {label} | N/A | {target} | ⚠ |"
        emoji = "✓" if val >= target else "✗"
        return f"| {label} | {val:.2%}{suffix} | {target:.0%} | {emoji} |"

    lines.append(row("Predicate accuracy", agg["predicate_accuracy"], 0.85,
                     f" ({agg['predicate_counts'][0]}/{agg['predicate_counts'][1]})"))
    lines.append(row("Operator accuracy", agg["operator_accuracy"], 0.95,
                     f" ({agg['operator_counts'][0]}/{agg['operator_counts'][1]})"))
    lines.append(row("Numerical accuracy vs regex", agg["numerical_accuracy_vs_regex"], 0.95,
                     f" ({agg['numerical_counts'][0]}/{agg['numerical_counts'][1]})"))
    lines.append(row("Reference accuracy vs regex", agg["reference_accuracy_vs_regex"], 0.90,
                     f" ({agg['reference_counts'][0]}/{agg['reference_counts'][1]})"))
    lines.append(row("Structural completeness", agg["structural_completeness"], 0.80,
                     f" ({agg['structural_counts'][0]}/{agg['structural_counts'][1]})"))
    lines.append(row("Schema pass rate", agg["schema_pass_rate"], 0.95))
    lines.append(f"| Avg confidence | {agg['avg_confidence']:.3f} | ≥ 0.70 | "
                 f"{'✓' if agg['avg_confidence'] >= 0.70 else '✗'} |")
    lines.append("")

    lines.append("## Phase 2 gate verdict")
    lines.append("")
    if passed:
        lines.append("✓ **PASS** — extractor đủ chất lượng để scale full 543 clauses.")
    else:
        lines.append("✗ **FAIL** — cần iterate prompt + few-shot examples trước khi scale.")
        lines.append("")
        for f in fails:
            lines.append(f"- {f}")
    lines.append("")

    # Per-clause breakdown
    lines.append("## Per-clause stats")
    lines.append("")
    lines.append("| Clause | n_cond | n_rules | n_thr | n_ref | n_def | confidence | issues |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in results:
        issues_str = "; ".join(r["schema_issues"]) if r["schema_issues"] else "ok"
        lines.append(f"| `{r['clause_id']}` | {r['n_conditions']} | {r['n_rules']} | "
                     f"{r['n_thresholds']} | {r['n_references']} | {r['n_defines']} | "
                     f"{r['confidence']:.2f} | {issues_str} |")
    lines.append("")

    # Non-canonical predicates summary
    if agg["top_non_canonical_predicates"]:
        lines.append("## Non-canonical predicates encountered")
        lines.append("")
        lines.append("These predicates không có trong vocabulary — manual review needed:")
        lines.append("")
        lines.append("| Predicate | Frequency |")
        lines.append("|---|---:|")
        for pred, freq in agg["top_non_canonical_predicates"]:
            lines.append(f"| `{pred}` | {freq} |")
        lines.append("")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{'✓ PASS' if passed else '✗ FAIL'} validation gate")
    print(f"Saved: {REPORT_OUT}")

    if args.verbose:
        for r in results:
            print(f"\n{r['clause_id']}: conf={r['confidence']:.2f}, "
                  f"pred {r['n_canonical_predicates']}, "
                  f"struct {r['n_structurally_complete']}, "
                  f"non_can: {r['non_canonical_predicates']}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
