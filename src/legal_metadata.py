"""Deterministic metadata registry for multi-law processing."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_METADATA_PATH = Path("data/legal_metadata.yaml")


def fold_text(text: str) -> str:
    text = (text or "").replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9/_-]+", " ", ascii_text)
    return re.sub(r"\s+", " ", ascii_text).strip()


@dataclass(frozen=True)
class LawMetadata:
    id: str
    code: str
    full_id: str
    title: str
    canonical_title: str
    type: str
    hierarchy_level: str
    priority: int
    source_file: Path
    issuer: str | None = None
    issued_date: str | None = None
    effective_date: str | None = None
    repealed_date: str | None = None
    expected_chapters: int | None = None
    expected_sections: int | None = None
    expected_articles: int | None = None
    aliases: tuple[str, ...] = ()
    prolog_law_ids: tuple[str, ...] = ()
    repeals: tuple[str, ...] = ()
    allow_no_chapter: bool = False
    llm_skip_articles: tuple[int, ...] = ()
    llm_skip_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def law_node(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "code": self.code,
            "full_id": self.full_id,
            "title": self.title,
            "canonical_title": self.canonical_title,
            "type": self.type,
            "issuer": self.issuer,
            "issued_date": self.issued_date,
            "effective_date": self.effective_date,
            "repealed_date": self.repealed_date,
            "hierarchy_level": self.hierarchy_level,
            "priority": self.priority,
            "aliases": list(self.aliases),
            "prolog_law_ids": list(self.prolog_law_ids),
        }


def _as_date_string(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _law_from_mapping(key: str, value: dict[str, Any]) -> LawMetadata:
    expected = value.get("expected") or {}
    return LawMetadata(
        id=str(value.get("id") or key),
        code=str(value.get("code") or value.get("id") or key),
        full_id=str(value["full_id"]),
        title=str(value.get("title") or value.get("canonical_title") or key),
        canonical_title=str(value.get("canonical_title") or value.get("title") or key),
        type=str(value.get("type") or "law"),
        hierarchy_level=str(value.get("hierarchy_level") or "luật"),
        priority=int(value.get("priority") or 100),
        issuer=value.get("issuer"),
        issued_date=_as_date_string(value.get("issued_date")),
        effective_date=_as_date_string(value.get("effective_date")),
        repealed_date=_as_date_string(value.get("repealed_date")),
        source_file=Path(str(value["source_file"])),
        expected_chapters=expected.get("chapters"),
        expected_sections=expected.get("sections"),
        expected_articles=expected.get("articles"),
        aliases=tuple(str(a) for a in (value.get("aliases") or [])),
        prolog_law_ids=tuple(str(a) for a in (value.get("prolog_law_ids") or [])),
        repeals=tuple(str(a) for a in (value.get("repeals") or [])),
        allow_no_chapter=bool(value.get("allow_no_chapter") or False),
        llm_skip_articles=tuple(int(x) for x in (value.get("llm_skip_articles") or [])),
        llm_skip_reason=str(value.get("llm_skip_reason") or ""),
        raw=dict(value),
    )


def load_law_metadata(path: Path | str = DEFAULT_METADATA_PATH) -> dict[str, LawMetadata]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    laws = raw.get("laws") or {}
    if not isinstance(laws, dict):
        raise ValueError(f"Invalid legal metadata file: {path}")
    return {str(key): _law_from_mapping(str(key), value) for key, value in laws.items()}


def load_order(path: Path | str = DEFAULT_METADATA_PATH) -> list[str]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    order = [str(x) for x in (raw.get("load_order") or [])]
    laws = load_law_metadata(path)
    return [law_id for law_id in order if law_id in laws] or list(laws)


def alias_index(laws: dict[str, LawMetadata]) -> dict[str, str]:
    out: dict[str, str] = {}
    for law_id, meta in laws.items():
        candidates = {
            law_id,
            meta.id,
            meta.code,
            meta.full_id,
            meta.title,
            meta.canonical_title,
            *meta.aliases,
            *meta.prolog_law_ids,
        }
        for candidate in candidates:
            folded = fold_text(candidate)
            if folded:
                out[folded] = law_id
    return out


def resolve_law_id(value: str, laws: dict[str, LawMetadata] | None = None) -> str:
    laws = laws or load_law_metadata()
    if value in laws:
        return value
    folded = fold_text(value)
    idx = alias_index(laws)
    if folded in idx:
        return idx[folded]
    m = re.match(r"(\d+)/(\d{4})/", value or "")
    if m:
        candidate = f"L{m.group(1)}_{m.group(2)}"
        if candidate in laws:
            return candidate
    raise KeyError(f"Unknown law id/code/alias: {value!r}")


def metadata_for(value: str, laws: dict[str, LawMetadata] | None = None) -> LawMetadata:
    laws = laws or load_law_metadata()
    return laws[resolve_law_id(value, laws)]
