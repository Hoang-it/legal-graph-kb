"""Metadata-driven Prolog source terms for legal citations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from src.legal_metadata import LawMetadata, load_law_metadata

_CITATION_RE = re.compile(
    r"^(?P<law>[A-Za-z0-9_]+)\.A(?P<article>\d+[a-z]?)"
    r"(?:\.K(?P<clause>\d+[a-z]?))?"
    r"(?:\.(?:D)?(?P<point>[a-z]+))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LegalSourceTerms:
    citation_id: str
    source_atom: str
    law_code: str
    law_atom: str
    article_atom: str
    clause_atom: str
    point_atom: str


def prolog_atom(value: str, *, prefix: str | None = None) -> str:
    atom = re.sub(r"[^a-z0-9_]+", "_", (value or "").lower()).strip("_")
    if prefix and not atom.startswith(f"{prefix}_"):
        atom = f"{prefix}_{atom}"
    if not atom or not re.match(r"^[a-z]", atom):
        atom = f"{prefix or 'atom'}_{atom}".strip("_")
    return atom


def source_atom_for_citation_id(citation_id: str) -> str:
    return prolog_atom(citation_id, prefix="source")


def law_atom_for_code(
    law_code: str,
    laws: dict[str, LawMetadata] | None = None,
) -> str:
    laws = laws or load_law_metadata()
    meta = laws.get(law_code)
    if meta and meta.prolog_law_ids:
        return prolog_atom(meta.prolog_law_ids[0])
    return prolog_atom(law_code, prefix="law")


@lru_cache(maxsize=8)
def prolog_law_atom_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for law_code, meta in load_law_metadata().items():
        for atom in meta.prolog_law_ids:
            index[prolog_atom(atom)] = law_code
        index[prolog_atom(law_code)] = law_code
        index[prolog_atom(law_code, prefix="law")] = law_code
    return index


def law_code_from_atom(law_atom: str) -> str | None:
    return prolog_law_atom_index().get(prolog_atom(law_atom))


def parse_citation_id(citation_id: str) -> dict[str, str | None]:
    match = _CITATION_RE.match((citation_id or "").strip())
    if not match:
        raise ValueError(f"Invalid legal citation id: {citation_id!r}")
    law = match.group("law").upper()
    if re.match(r"^L\d+_\d{4}$", law):
        law = law[0] + law[1:]
    return {
        "law_code": law,
        "article": match.group("article").lower(),
        "clause": (match.group("clause") or "").lower() or None,
        "point": (match.group("point") or "").lower() or None,
    }


def citation_id_from_prolog_terms(
    law_atom: str,
    article_atom: str,
    clause_atom: str | None,
    point_atom: str | None,
) -> str | None:
    law_code = law_code_from_atom(law_atom)
    if not law_code:
        return None

    article_match = re.fullmatch(r"article_(\d+[a-z]?)", (article_atom or "").lower())
    if not article_match:
        return None

    parts = [law_code, f"A{article_match.group(1)}"]
    clause_atom = (clause_atom or "none").lower()
    if clause_atom != "none":
        clause_match = re.fullmatch(r"clause_(\d+[a-z]?)", clause_atom)
        if not clause_match:
            return None
        parts.append(f"K{clause_match.group(1)}")

    point_atom = (point_atom or "none").lower()
    if point_atom != "none":
        point_match = re.fullmatch(r"point_([a-z]+)", point_atom)
        if not point_match:
            return None
        parts.append(point_match.group(1))
    return ".".join(parts)


def terms_for_citation_id(
    citation_id: str,
    law_code: str | None = None,
    laws: dict[str, LawMetadata] | None = None,
) -> LegalSourceTerms:
    parsed = parse_citation_id(citation_id)
    resolved_law = law_code or str(parsed["law_code"])
    article = str(parsed["article"])
    clause = parsed["clause"]
    point = parsed["point"]
    return LegalSourceTerms(
        citation_id=citation_id,
        source_atom=source_atom_for_citation_id(citation_id),
        law_code=resolved_law,
        law_atom=law_atom_for_code(resolved_law, laws),
        article_atom=f"article_{article}",
        clause_atom=f"clause_{clause}" if clause else "none",
        point_atom=f"point_{point}" if point else "none",
    )


def terms_for_context_item(
    item: dict[str, Any],
    laws: dict[str, LawMetadata] | None = None,
) -> LegalSourceTerms:
    citation_id = str(item.get("citation_id") or "")
    law_code = str(item.get("law_code") or "") or None
    return terms_for_citation_id(citation_id, law_code=law_code, laws=laws)


def quote_prolog_text(text: str, limit: int = 900) -> str:
    clipped = (text or "")[:limit]
    return "'" + clipped.replace("\\", "\\\\").replace("'", "\\'") + "'"


def legal_source_fact_for_context_item(
    item: dict[str, Any],
    laws: dict[str, LawMetadata] | None = None,
) -> str:
    terms = terms_for_context_item(item, laws=laws)
    text = quote_prolog_text(str(item.get("text") or ""))
    return (
        "legal_source("
        f"{terms.source_atom}, "
        f"{terms.law_atom}, "
        f"{terms.article_atom}, "
        f"{terms.clause_atom}, "
        f"{terms.point_atom}, "
        f"{text})."
    )
