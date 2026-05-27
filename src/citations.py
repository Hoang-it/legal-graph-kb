"""Shared citation parsing and formatting for evaluation and reporting.

The authority metadata comes only from ``data/legal_sources.yaml``. This module
does not infer unknown legal sources or silently map ambiguous references.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_REGISTRY_PATH = Path("data/legal_sources.yaml")


@dataclass(frozen=True)
class SourceInfo:
    key: str
    canonical_title: str
    type: str
    aliases: tuple[str, ...]
    folded_aliases: tuple[str, ...]
    prolog_law_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CitationRef:
    source: str
    article: str
    clause: str | None = None
    point: str | None = None

    @property
    def article_id(self) -> str:
        return f"{self.source}.A{self.article}"

    @property
    def item_id(self) -> str:
        out = self.article_id
        if self.clause:
            out += f".K{self.clause}"
        if self.point:
            out += f".D{self.point}"
        return out


@dataclass(frozen=True)
class CitationParseError:
    error_type: str
    text: str
    detail: str


@dataclass(frozen=True)
class GoldCitationParseResult:
    refs: tuple[CitationRef, ...]
    errors: tuple[CitationParseError, ...]


_INTERNAL_ID_RE = re.compile(
    r"^(?P<source>[A-Z0-9_]+)\.A(?P<article>\d+[a-z]?)"
    r"(?:\.K(?P<clause>\d+[a-z]?))?"
    r"(?:\.(?:D)?(?P<point>[a-z]))?$",
    re.IGNORECASE,
)
_ARTICLE_RE = re.compile(
    r"\bdieu\s+(?P<spec>\d+[a-z]?(?:\s*(?:-|,|va)\s*\d+[a-z]?)*)(?=\b)"
)


def fold_text(text: str) -> str:
    """Lowercase, remove Vietnamese accents, and normalize spacing."""
    text = (text or "").replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9/_-]+", " ", ascii_text)
    return re.sub(r"\s+", " ", ascii_text).strip()


def _load_registry_uncached(path: Path) -> dict[str, SourceInfo]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sources = raw.get("sources") or {}
    if not isinstance(sources, dict):
        raise ValueError(f"Invalid legal source registry: {path}")

    out: dict[str, SourceInfo] = {}
    for key, value in sources.items():
        if not isinstance(value, dict):
            raise ValueError(f"Invalid source entry for {key}")
        canonical = str(value.get("canonical_title") or "").strip()
        source_type = str(value.get("type") or "").strip()
        aliases = tuple(str(a).strip() for a in (value.get("aliases") or []) if str(a).strip())
        if not canonical or not aliases:
            raise ValueError(f"Source {key} must define canonical_title and aliases")
        folded_aliases = tuple(
            sorted({fold_text(canonical), *(fold_text(a) for a in aliases)}, key=len, reverse=True)
        )
        prolog_law_ids = tuple(
            str(a).strip().lower()
            for a in (value.get("prolog_law_ids") or [])
            if str(a).strip()
        )
        out[str(key)] = SourceInfo(
            key=str(key),
            canonical_title=canonical,
            type=source_type,
            aliases=aliases,
            folded_aliases=folded_aliases,
            prolog_law_ids=prolog_law_ids,
        )
    return out


@lru_cache(maxsize=8)
def load_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, SourceInfo]:
    return _load_registry_uncached(Path(path))


def parse_internal_citation_id(
    citation_id: str,
    registry: dict[str, SourceInfo] | None = None,
) -> CitationRef:
    registry = registry or load_registry()
    m = _INTERNAL_ID_RE.match((citation_id or "").strip())
    if not m:
        raise ValueError(f"Invalid citation id: {citation_id!r}")
    source = m.group("source").upper()
    if source not in registry:
        raise ValueError(f"Unknown citation source: {source}")
    return CitationRef(
        source=source,
        article=m.group("article").lower(),
        clause=(m.group("clause") or "").lower() or None,
        point=(m.group("point") or "").lower() or None,
    )


def normalize_citation_id(citation_id: str, registry: dict[str, SourceInfo] | None = None) -> str:
    return parse_internal_citation_id(citation_id, registry).item_id


def article_id_from_citation_id(
    citation_id: str,
    registry: dict[str, SourceInfo] | None = None,
) -> str:
    return parse_internal_citation_id(citation_id, registry).article_id


def format_citation(ref_or_id: CitationRef | str, registry: dict[str, SourceInfo] | None = None) -> str:
    registry = registry or load_registry()
    ref = parse_internal_citation_id(ref_or_id, registry) if isinstance(ref_or_id, str) else ref_or_id
    source = registry[ref.source]
    parts = [f"Điều {ref.article}"]
    if ref.clause:
        parts.append(f"khoản {ref.clause}")
    if ref.point:
        parts.append(f"điểm {ref.point}")
    return f"[{source.canonical_title}, {' '.join(parts)}]"


def source_from_prolog_law_id(
    law_id: str,
    registry: dict[str, SourceInfo] | None = None,
) -> str:
    registry = registry or load_registry()
    normalized = (law_id or "").strip().lower()
    for key, info in registry.items():
        if normalized in info.prolog_law_ids:
            return key
    raise ValueError(f"Unknown Prolog LawId: {law_id!r}")


def ref_from_prolog_terms(
    law_id: str,
    article_atom: str,
    clause_atom: str | None = None,
    point_atom: str | None = None,
    registry: dict[str, SourceInfo] | None = None,
) -> CitationRef:
    source = source_from_prolog_law_id(law_id, registry)

    def _strip(prefix: str, value: str | None) -> str | None:
        if not value or value == "none":
            return None
        value = value.strip().lower()
        return value.removeprefix(prefix)

    article = _strip("article_", article_atom)
    if not article:
        raise ValueError(f"Missing Prolog article atom: {article_atom!r}")
    return CitationRef(
        source=source,
        article=article,
        clause=_strip("clause_", clause_atom),
        point=_strip("point_", point_atom),
    )


def _source_occurrences(
    folded_text: str,
    registry: dict[str, SourceInfo],
) -> list[tuple[int, int, str, str]]:
    matches: list[tuple[int, int, str, str]] = []
    for key, source in registry.items():
        for alias in source.folded_aliases:
            if not alias:
                continue
            for m in re.finditer(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", folded_text):
                matches.append((m.start(), m.end(), key, alias))

    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    selected: list[tuple[int, int, str, str]] = []
    occupied: list[tuple[int, int]] = []
    for start, end, key, alias in matches:
        if any(not (end <= occ_start or start >= occ_end) for occ_start, occ_end in occupied):
            continue
        selected.append((start, end, key, alias))
        occupied.append((start, end))
    return sorted(selected, key=lambda x: x[0])


def _expand_article_spec(spec: str) -> list[str]:
    out: list[str] = []
    for part in re.split(r"\s*,\s*|\s+va\s+", spec.strip()):
        if not part:
            continue
        if "-" in part:
            left, right = (p.strip() for p in part.split("-", 1))
            if left.isdigit() and right.isdigit():
                start, end = int(left), int(right)
                if start <= end and end - start <= 50:
                    out.extend(str(n) for n in range(start, end + 1))
                continue
        out.append(part.lower())
    return out


def _extract_articles(folded_text: str) -> list[str]:
    articles: list[str] = []
    for m in _ARTICLE_RE.finditer(folded_text):
        articles.extend(_expand_article_spec(m.group("spec")))
    return articles


def parse_gold_citations_raw(
    raw_text: str | None,
    registry: dict[str, SourceInfo] | None = None,
) -> GoldCitationParseResult:
    """Parse gold citation text into article-level refs.

    The parser is intentionally strict: every segment must contain a known
    authority and at least one article reference.
    """
    registry = registry or load_registry()
    if not raw_text or not raw_text.strip():
        return GoldCitationParseResult(
            refs=(),
            errors=(
                CitationParseError(
                    "missing_gold_citations_raw",
                    "",
                    "gold_citations_raw is empty",
                ),
            ),
        )

    refs: list[CitationRef] = []
    errors: list[CitationParseError] = []
    segments = [s.strip() for s in re.split(r"[\n;]+", raw_text) if s.strip()]
    for segment in segments:
        folded = fold_text(segment)
        source_occ = _source_occurrences(folded, registry)
        if not source_occ:
            errors.append(
                CitationParseError(
                    "authority_unresolved",
                    segment,
                    "No known authority alias from data/legal_sources.yaml was found",
                )
            )
            continue

        for i, (start, end, source_key, _alias) in enumerate(source_occ):
            prev_end = source_occ[i - 1][1] if i > 0 else 0
            next_start = source_occ[i + 1][0] if i + 1 < len(source_occ) else len(folded)
            left = folded[prev_end:start]
            right = folded[end:next_start]
            before_articles = _extract_articles(left)
            after_articles = _extract_articles(right)
            articles = before_articles or after_articles
            if not articles:
                errors.append(
                    CitationParseError(
                        "article_missing",
                        segment,
                        f"Authority {source_key} has no explicit article in its segment",
                    )
                )
                continue
            refs.extend(CitationRef(source=source_key, article=a) for a in articles)

    unique: dict[str, CitationRef] = {}
    for ref in refs:
        unique[ref.article_id] = CitationRef(source=ref.source, article=ref.article)
    return GoldCitationParseResult(
        refs=tuple(unique.values()),
        errors=tuple(errors),
    )


def parse_displayed_citations(
    answer_text: str,
    registry: dict[str, SourceInfo] | None = None,
) -> list[CitationRef]:
    """Parse strict displayed citations with explicit authority and article."""
    registry = registry or load_registry()
    folded = fold_text(answer_text or "")
    refs: dict[str, CitationRef] = {}
    for start, end, source_key, _alias in _source_occurrences(folded, registry):
        right = folded[end : min(len(folded), end + 180)]
        left = folded[max(0, start - 120) : start]
        article_matches = list(_iter_article_clause_point(right))
        if not article_matches:
            article_matches = list(_iter_article_clause_point(left))
        for article, clause, point in article_matches:
            ref = CitationRef(source=source_key, article=article, clause=clause, point=point)
            refs[ref.item_id] = ref
    return list(refs.values())


def _iter_article_clause_point(text: str):
    pattern = re.compile(
        r"\bdieu\s+(?P<article>\d+[a-z]?)"
        r"(?:\s+khoan\s+(?P<clause>\d+[a-z]?))?"
        r"(?:\s+diem\s+(?P<point>[a-z]))?"
    )
    for m in pattern.finditer(text):
        yield (
            m.group("article").lower(),
            (m.group("clause") or "").lower() or None,
            (m.group("point") or "").lower() or None,
        )


def displayed_matches_pipeline(pipeline_ref: CitationRef, displayed_ref: CitationRef) -> bool:
    if pipeline_ref.source != displayed_ref.source or pipeline_ref.article != displayed_ref.article:
        return False
    if pipeline_ref.point:
        return pipeline_ref.clause == displayed_ref.clause and pipeline_ref.point == displayed_ref.point
    if pipeline_ref.clause:
        return pipeline_ref.clause == displayed_ref.clause
    return True
