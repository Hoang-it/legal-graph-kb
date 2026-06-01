"""``build_semantic_context`` — query → BHXH concept frame (no dense seed).

This is the "get semantic relevance" step of exp 13 (semantic-grounded HyDE,
``docs/plans/exp13_hyde_semantic.md`` §4). It maps a user question to a
**concept frame** drawn from:

1. the 32 curated BHXH concepts (``runtime.logic_lm.knowledge.bhxh_ontology``
   — rich aliases, robust to informal phrasing), and
2. the live KG's multi-law semantic entities exported to
   ``data/ontology/ontology_kg_full.json`` (Subject / Benefit / Obligation /
   Right / Fund / Organization / ProhibitedAct — human-readable Vietnamese
   names with clause/article provenance).

It deliberately does **NOT** run a dense clause search — that is the exact
domain-noisy seed that sank exp 09 (HyDE2). The frame is concept-only: it
carries NO Điều/Khoản numbers (the HyDE prompt forbids citing them), so
``frame_text`` is safe to drop straight into ``{context}``.

Pure + deterministic (no LLM, no network) → the downstream HyDE cache key is
stable. Article ids are collected for diagnostics only; they are NOT used to
retrieve (retrieval stays pure HyDE-dense for a clean comparison vs HyDE1).

CLI (ad-hoc inspection):
    python -m runtime.retrievers.semantic_context -q "Khi nào được hưởng lương hưu?"
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from runtime.logic_lm.knowledge.bhxh_ontology import (
    concept_specs_by_id,
    match_query_concept_ids,
    normalize_for_matching,
)

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_ONTOLOGY_PATH = _REPO / "data" / "ontology" / "ontology_kg_full.json"

# KG semantic labels that carry human-readable Vietnamese names worth putting
# in the concept frame, with their Vietnamese category header. LegalConcept is
# intentionally excluded — its nodes are id-slug-only (no name) and duplicate
# the Subject entries.
_FRAME_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("Subject", "Đối tượng áp dụng"),
    ("Benefit", "Chế độ / quyền lợi"),
    ("Right", "Quyền"),
    ("Obligation", "Nghĩa vụ"),
    ("Fund", "Quỹ"),
    ("Organization", "Tổ chức liên quan"),
    ("ProhibitedAct", "Hành vi bị nghiêm cấm"),
)
_FRAME_LABELS = {lab for lab, _ in _FRAME_CATEGORIES}

_MAX_CONCEPTS = 12
_MAX_PER_CATEGORY = 6
_MAX_KG_ENTITIES = 18


@dataclass
class SemanticContext:
    question: str
    frame_text: str
    concept_ids: list[str] = field(default_factory=list)   # matched 32-concept ids
    kg_entity_ids: list[str] = field(default_factory=list)  # matched KG node ids
    context_key_ids: list[str] = field(default_factory=list)  # → HyDE cache key
    article_ids: list[str] = field(default_factory=list)   # diagnostics only (NOT for retrieval)
    laws: list[str] = field(default_factory=list)          # diagnostics
    concept_match: bool = False
    n_concepts: int = 0
    n_kg_entities: int = 0


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def _phrase_in(haystack_norm: str, phrase: str) -> bool:
    """True if normalised ``phrase`` occurs as a token-bounded substring."""
    pn = normalize_for_matching(phrase)
    if not pn:
        return False
    if len(pn) <= 4:  # short tokens → require word boundaries to avoid noise
        import re
        return re.search(rf"(?<!\w){re.escape(pn)}(?!\w)", haystack_norm) is not None
    return pn in haystack_norm


def _entity_display_name(node: dict) -> str:
    props = node.get("properties") or {}
    for key in ("name", "label", "term_vi", "description"):
        v = props.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # fall back to de-slugging the id ("subject:nguoi-lao-dong" → "nguoi lao dong")
    nid = str(node.get("id") or "")
    tail = nid.split(":", 1)[-1]
    return tail.replace("-", " ").replace("_", " ").strip() or nid


@lru_cache(maxsize=4)
def _load_frame_entities(ontology_path: str) -> tuple[dict, ...]:
    """Load + index the concept-like KG entities once per path.

    Returns a tuple of ``{id, label, name, name_norm, article_ids, laws}``
    dicts for the labels in ``_FRAME_LABELS`` that have a usable name.
    """
    p = Path(ontology_path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — build it with `python -m offline.build_ontology_kg`."
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    out: list[dict] = []
    for n in data.get("nodes") or []:
        labels = n.get("labels") or []
        label = next((l for l in labels if l in _FRAME_LABELS), None)
        if label is None:
            continue
        name = _entity_display_name(n)
        name_norm = normalize_for_matching(name)
        if not name_norm:
            continue
        out.append({
            "id": n.get("id"),
            "label": label,
            "name": name,
            "name_norm": name_norm,
            "article_ids": list(n.get("article_ids") or []),
            "laws": list(n.get("laws") or []),
        })
    # Longer (more specific) names first so per-category truncation keeps signal.
    out.sort(key=lambda e: -len(e["name_norm"]))
    return tuple(out)


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------


def _concept_frame_lines(concept_ids: Iterable[str]) -> list[str]:
    specs = concept_specs_by_id()
    ids = list(concept_ids)
    if not ids:
        return []
    # invert parent → children for sibling/child context
    children: dict[str, list[str]] = {}
    for cid, spec in specs.items():
        for parent in spec.get("parents", []) or []:
            children.setdefault(parent, []).append(cid)

    lines: list[str] = []
    seen: set[str] = set()
    for cid in ids[:_MAX_CONCEPTS]:
        spec = specs.get(cid)
        if spec is None or cid in seen:
            continue
        seen.add(cid)
        label = spec.get("label", cid)
        parents = [specs.get(p, {}).get("label", p) for p in spec.get("parents", []) or []]
        kids = [specs.get(k, {}).get("label", k) for k in children.get(cid, [])][:4]
        suffix = ""
        if parents:
            suffix += f"  (thuộc: {', '.join(parents)})"
        if kids:
            suffix += f"  → {', '.join(kids)}"
        lines.append(f"- {label}{suffix}")
    return lines


def _kg_frame_blocks(matched: list[dict]) -> list[str]:
    by_label: dict[str, list[str]] = {}
    for e in matched:
        by_label.setdefault(e["label"], [])
        if len(by_label[e["label"]]) < _MAX_PER_CATEGORY:
            by_label[e["label"]].append(e["name"])
    blocks: list[str] = []
    for label, header in _FRAME_CATEGORIES:
        names = by_label.get(label)
        if names:
            blocks.append(f"{header}: " + "; ".join(names))
    return blocks


def _render_frame(concept_lines: list[str], kg_blocks: list[str]) -> str:
    parts: list[str] = []
    if concept_lines:
        parts.append("Khái niệm BHXH liên quan:")
        parts.extend(concept_lines)
    if kg_blocks:
        if parts:
            parts.append("")
        parts.extend(kg_blocks)
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def build_semantic_context(
    question: str,
    ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH,
) -> SemanticContext:
    """Map ``question`` → a BHXH concept frame. No dense clause seed."""
    q_norm = normalize_for_matching(question)

    concept_ids = sorted(match_query_concept_ids(question))

    entities = _load_frame_entities(str(ontology_path))
    matched: list[dict] = []
    seen_norm: set[str] = set()
    for e in entities:  # pre-sorted longest-name-first → keep the most specific variant
        if e["name_norm"] in seen_norm:
            continue
        if _phrase_in(q_norm, e["name_norm"]):
            seen_norm.add(e["name_norm"])
            matched.append(e)
    matched = matched[:_MAX_KG_ENTITIES]

    concept_lines = _concept_frame_lines(concept_ids)
    kg_blocks = _kg_frame_blocks(matched)
    frame_text = _render_frame(concept_lines, kg_blocks)

    kg_entity_ids = [e["id"] for e in matched if e.get("id")]
    article_ids = sorted({a for e in matched for a in e["article_ids"]})
    laws = sorted({l for e in matched for l in e["laws"]})
    context_key_ids = sorted(concept_ids) + sorted(kg_entity_ids)

    return SemanticContext(
        question=question,
        frame_text=frame_text,
        concept_ids=concept_ids,
        kg_entity_ids=kg_entity_ids,
        context_key_ids=context_key_ids,
        article_ids=article_ids,
        laws=laws,
        concept_match=bool(frame_text),
        n_concepts=len(concept_ids),
        n_kg_entities=len(matched),
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-q", "--question", required=True)
    ap.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY_PATH))
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    ctx = build_semantic_context(args.question, ontology_path=args.ontology)
    print(f"=== QUESTION ===\n{ctx.question}\n")
    print(f"concept_match={ctx.concept_match}  n_concepts={ctx.n_concepts}  "
          f"n_kg_entities={ctx.n_kg_entities}  laws={ctx.laws}")
    print(f"concept_ids={ctx.concept_ids}")
    print(f"\n=== FRAME ({len(ctx.frame_text)} chars) ===\n{ctx.frame_text}")
    print(f"\ncontext_key_ids ({len(ctx.context_key_ids)}): {ctx.context_key_ids[:20]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
