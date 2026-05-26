"""text_normalize.py — Normalize IRAC text cho BERTScore + AR fair compare.

Mode "app_conclusion" (recommended): Strip IRAC headers, drop Issue & Rule
sections (Issue = question paraphrase → AR bias; Rule = quoted law text →
BERTScore bias). Keep Application + Conclusion = elite's actual reasoning
+ final answer, comparable to prose.

Usage:
    from experiments.text_normalize import normalize_for_prose_metric
    normed = normalize_for_prose_metric(answer, irac_sections, mode="app_conclusion")
"""

from __future__ import annotations

import re

_HEADER_PAT = re.compile(r"^\s*(Issue|Rule|Application|Conclusion)\s*:\s*",
                          re.MULTILINE)


def normalize_for_prose_metric(
    answer_text: str,
    irac_sections: dict | None = None,
    mode: str = "app_conclusion",
) -> str:
    """Returns text suitable cho prose-based metric (BERTScore, AR).

    Args:
        answer_text: Raw answer (may be IRAC or free prose).
        irac_sections: Pre-parsed dict {'issue':..., 'rule':..., 'application':...,
                       'conclusion':...} từ elite_pipelines._parse_irac_sections.
                       None → text được xem là free prose (no normalize).
        mode: 'app_conclusion' (default) — keep Application + Conclusion only.
              'conclusion_only' — only Conclusion.
              'strip_headers' — keep all text, just remove labels.
              'no_op' — return as-is.

    Returns:
        Normalized string.
    """
    if mode == "no_op" or not irac_sections:
        return answer_text or ""

    if mode == "conclusion_only":
        return (irac_sections.get("conclusion") or "").strip() or answer_text

    if mode == "app_conclusion":
        parts = []
        app = irac_sections.get("application", "").strip()
        con = irac_sections.get("conclusion", "").strip()
        if app:
            parts.append(app)
        if con:
            parts.append(con)
        return "\n\n".join(parts) if parts else (answer_text or "")

    if mode == "strip_headers":
        return _HEADER_PAT.sub("", answer_text or "").strip()

    raise ValueError(f"Unknown mode: {mode}")


def is_irac(record: dict) -> bool:
    """True nếu record là từ elite arm (có irac_sections non-empty)."""
    sec = record.get("irac_sections") or {}
    return bool(sec) and any(sec.values())
