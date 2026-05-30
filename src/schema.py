"""Schema dữ liệu thống nhất cho toàn bộ pipeline (Pydantic v2).

Đây là single source of truth cho:
- B3 (LLM extraction): models này dùng làm JSON schema cho structured output.
- B4 (merge/normalize): validate dữ liệu trước khi load Neo4j.
- B6 (load Neo4j): mapping 1-1 sang node/edge.

NGUYÊN TẮC PROVENANCE (yêu cầu cứng của dự án):
- Structural node (Article/Clause/Point) lưu nguyên text gốc → bản thân là nguồn.
- Mọi SEMANTIC node phải có `mentioned_in: list[str]` — danh sách Clause.id nơi
  thực thể được đề cập. Không được rỗng.
- Mọi SEMANTIC edge phải có `source_clause: str` (Clause.id) + `source_text: str`
  (snippet ≤ 300 ký tự từ Clause gốc làm bằng chứng).
- Từ bất kỳ node/edge nào, có thể truy về Điều/Khoản/Điểm gốc bằng
  `ids.parse_id()` hoặc thuộc tính `mentioned_in` / `source_clause`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# =====================================================================
# STRUCTURAL NODES — tự thân là nguồn (1-1 với văn bản gốc)
# =====================================================================


class LawNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    code: str  # '41/2024/QH15'
    title: str
    issued_date: str | None = None  # YYYY-MM-DD
    effective_date: str | None = None
    issuer: str | None = None  # 'Quốc hội'


class ChapterNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    number: int
    roman: str  # 'I', 'II', ...
    title: str


class SectionNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    number: int  # đánh số lại từ 1 trong mỗi Chương
    title: str
    chapter_id: str


class ArticleNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    number: int
    title: str
    text: str  # toàn văn Điều (gộp các Khoản)
    chapter_id: str
    section_id: str | None = None  # nếu Article thuộc Mục
    embedding: list[float] | None = None


class ClauseNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    number: int
    text: str  # nguyên văn Khoản
    article_id: str
    embedding: list[float] | None = None


class PointNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    letter: str  # 'a', 'b', ...
    text: str
    clause_id: str
    embedding: list[float] | None = None


class TableNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    article_id: str
    caption: str | None = None
    rows_json: str  # JSON-serialized 2D array


# =====================================================================
# SEMANTIC NODES — luôn có mentioned_in[] để truy nguồn
# =====================================================================


class _SemanticBase(BaseModel):
    """Base cho mọi semantic node — bắt buộc có mentioned_in.

    `mentioned_in` là danh sách Clause.id (cũng có thể chứa Point.id) nơi
    thực thể này xuất hiện. Phục vụ truy ngược: từ entity → tất cả điều luật
    nhắc tới nó.
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    mentioned_in: list[str] = Field(..., min_length=1)

    @field_validator("mentioned_in")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("mentioned_in không được rỗng — vi phạm provenance")
        return v


class LegalConcept(_SemanticBase):
    term: str
    definition: str
    defined_in: str  # Clause.id của định nghĩa (thường ở Điều 3)


class Subject(_SemanticBase):
    name: str
    type: Literal["individual", "organization", "role", "group", "other"] = "other"


class Organization(_SemanticBase):
    name: str
    type: Literal[
        "state_agency", "ministry", "social_insurance_agency", "employer", "fund_manager", "other"
    ] = "other"


class Role(_SemanticBase):
    name: str


class Benefit(_SemanticBase):
    name: str
    category: Literal[
        "huu_tri",
        "om_dau",
        "thai_san",
        "tu_tuat",
        "tnld_bnn",
        "tro_cap_huu_tri_xa_hoi",
        "bhxh_tu_nguyen",
        "bhht_bo_sung",
        "khac",
    ] = "khac"


class Condition(_SemanticBase):
    description: str


class Obligation(_SemanticBase):
    description: str


class Right(_SemanticBase):
    description: str


class ProhibitedAct(_SemanticBase):
    description: str


class Fund(_SemanticBase):
    name: str


class ExternalLaw(BaseModel):
    """ExternalLaw không cần mentioned_in vì luôn được tạo từ một viện dẫn cụ thể."""

    model_config = ConfigDict(extra="forbid")
    id: str
    code: str | None = None  # vd '58/2014/QH13'
    title: str


# =====================================================================
# EDGES
# =====================================================================


class StructuralEdge(BaseModel):
    """Cạnh chứa (Law→Chapter→[Section]→Article→Clause→Point) hoặc NEXT."""

    model_config = ConfigDict(extra="forbid")
    type: Literal[
        "HAS_CHAPTER",
        "HAS_SECTION",
        "HAS_ARTICLE",
        "IN_SECTION",
        "HAS_CLAUSE",
        "HAS_POINT",
        "HAS_TABLE",
        "NEXT",
    ]
    src: str
    dst: str


class ReferenceEdge(BaseModel):
    """Viện dẫn nội bộ (rule-based): Clause/Article → Clause/Article."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["REFERENCES"] = "REFERENCES"
    src: str
    dst: str
    span: str  # cụm chữ gốc 'khoản 1 Điều 64'
    source_clause: str  # Clause.id chứa cụm viện dẫn


class ExternalCitationEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal[
        "CITES_EXTERNAL",
        "AMENDS",
        "REPEALS",
        "REPLACES",
        "TRANSITIONS_FROM",
    ]
    src: str  # Article.id (hoặc Clause.id)
    dst: str  # ExternalLaw.id
    span: str
    source_clause: str
    old_article: str | None = None  # cho AMENDS: điều bị sửa của luật cũ


class SemanticEdge(BaseModel):
    """Cạnh ngữ nghĩa từ LLM — luôn anchor về Clause gốc.

    `source_clause` = Clause.id, `source_text` = snippet bằng chứng (≤ 300 ký tự).
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal[
        "DEFINES",
        "ENTITLED_TO",
        "HAS_OBLIGATION",
        "HAS_RIGHT",
        "APPLIES_TO",
        "REQUIRES",
        "PAID_FROM",
        "MANAGES",
        "RESPONSIBLE_FOR",
        "PROHIBITED_BY",
    ]
    src: str
    dst: str
    source_clause: str = Field(..., min_length=1)
    source_text: str = Field(..., min_length=1, max_length=300)

    @field_validator("source_clause")
    @classmethod
    def _must_be_clause_id(cls, v: str) -> str:
        # Đồng bộ shape với `ids._ID_PATTERN` + `citations._INTERNAL_ID_RE`:
        # prefix luật canonical bất kỳ trong registry (L41_2024, ND143_2018,
        # QD366_BHXH, TT18_2022_BYT, …). Letter point gồm a-z + đ (bảng chữ
        # cái VN dùng trong luật).
        import re

        if not re.match(r"^[A-Z][A-Z0-9_]*\.A\d+\.K\d+(\.[a-zđ])?$", v):
            raise ValueError(f"source_clause phải là Clause.id hoặc Point.id, nhận: {v}")
        return v


# =====================================================================
# WRAPPER cho LLM output (B3)
# =====================================================================


class LLMArticleExtraction(BaseModel):
    """Schema cho output của LLM khi trích từ MỘT Article.

    LLM phải gắn mentioned_in / source_clause = các Clause thuộc Article này.
    """

    model_config = ConfigDict(extra="forbid")
    article_id: str
    concepts: list[LegalConcept] = []
    subjects: list[Subject] = []
    organizations: list[Organization] = []
    roles: list[Role] = []
    benefits: list[Benefit] = []
    conditions: list[Condition] = []
    obligations: list[Obligation] = []
    rights: list[Right] = []
    prohibited_acts: list[ProhibitedAct] = []
    funds: list[Fund] = []
    semantic_edges: list[SemanticEdge] = []


# =====================================================================
# Tiện ích: liệt kê tất cả label & edge type
# =====================================================================

ALL_NODE_LABELS = [
    "Law",
    "Chapter",
    "Section",
    "Article",
    "Clause",
    "Point",
    "Table",
    "LegalConcept",
    "Subject",
    "Organization",
    "Role",
    "Benefit",
    "Condition",
    "Obligation",
    "Right",
    "ProhibitedAct",
    "Fund",
    "ExternalLaw",
]

STRUCTURAL_EDGE_TYPES = [
    "HAS_CHAPTER",
    "HAS_SECTION",
    "HAS_ARTICLE",
    "IN_SECTION",
    "HAS_CLAUSE",
    "HAS_POINT",
    "HAS_TABLE",
    "NEXT",
]

REFERENCE_EDGE_TYPES = [
    "REFERENCES",
    "CITES_EXTERNAL",
    "AMENDS",
    "REPEALS",
    "REPLACES",
    "TRANSITIONS_FROM",
]

SEMANTIC_EDGE_TYPES = [
    "DEFINES",
    "ENTITLED_TO",
    "HAS_OBLIGATION",
    "HAS_RIGHT",
    "APPLIES_TO",
    "REQUIRES",
    "PAID_FROM",
    "MANAGES",
    "RESPONSIBLE_FOR",
    "PROHIBITED_BY",
]
