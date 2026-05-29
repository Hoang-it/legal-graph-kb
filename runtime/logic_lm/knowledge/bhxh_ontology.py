import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from runtime.logic_lm.config import settings


ConceptSpec = Dict[str, Any]


CONCEPT_SPECS: Sequence[ConceptSpec] = (
    {
        "id": "social_insurance",
        "label": "Bảo hiểm xã hội",
        "aliases": ["bảo hiểm xã hội", "bhxh"],
    },
    {
        "id": "mandatory_social_insurance",
        "label": "BHXH bắt buộc",
        "aliases": ["bhxh bắt buộc", "bảo hiểm xã hội bắt buộc"],
        "parents": ["social_insurance"],
    },
    {
        "id": "voluntary_social_insurance",
        "label": "BHXH tự nguyện",
        "aliases": ["bhxh tự nguyện", "bảo hiểm xã hội tự nguyện"],
        "parents": ["social_insurance"],
    },
    {
        "id": "unemployment_insurance",
        "label": "Bảo hiểm thất nghiệp",
        "aliases": ["bảo hiểm thất nghiệp", "bhtn", "trợ cấp thất nghiệp"],
    },
    {
        "id": "health_insurance",
        "label": "Bảo hiểm y tế",
        "aliases": ["bảo hiểm y tế", "bhyt"],
    },
    {
        "id": "pension",
        "label": "Lương hưu",
        "aliases": ["lương hưu", "hưu trí", "nghỉ hưu", "hưởng lương hưu"],
        "parents": ["social_insurance"],
    },
    {
        "id": "retirement_age",
        "label": "Tuổi nghỉ hưu",
        "aliases": ["tuổi nghỉ hưu", "tuổi hưu", "lộ trình tuổi nghỉ hưu"],
        "parents": ["pension"],
    },
    {
        "id": "early_retirement",
        "label": "Nghỉ hưu trước tuổi",
        "aliases": ["nghỉ hưu trước tuổi", "tuổi thấp hơn", "trước tuổi"],
        "parents": ["pension"],
    },
    {
        "id": "pension_rate",
        "label": "Tỷ lệ lương hưu",
        "aliases": [
            "tỷ lệ lương hưu",
            "mức lương hưu",
            "lương hưu hàng tháng",
            "tối đa 75%",
        ],
        "parents": ["pension"],
    },
    {
        "id": "one_time_social_insurance",
        "label": "BHXH một lần",
        "aliases": ["bhxh một lần", "bảo hiểm xã hội một lần", "hưởng một lần"],
        "parents": ["social_insurance"],
    },
    {
        "id": "contribution",
        "label": "Đóng BHXH",
        "aliases": ["đóng bhxh", "mức đóng", "thời gian đóng", "năm đóng", "tháng đóng"],
        "parents": ["social_insurance"],
    },
    {
        "id": "contribution_salary",
        "label": "Tiền lương tháng đóng BHXH",
        "aliases": [
            "tiền lương tháng đóng",
            "mức bình quân tiền lương",
            "mức tiền lương tháng đóng",
            "quỹ tiền lương",
        ],
        "parents": ["contribution"],
    },
    {
        "id": "employer_contribution",
        "label": "Mức đóng của người sử dụng lao động",
        "aliases": [
            "mức đóng bhxh bắt buộc của người sử dụng lao động",
            "mức đóng của người sử dụng lao động",
            "đơn vị hàng tháng đóng",
        ],
        "parents": ["contribution"],
    },
    {
        "id": "employee_contribution",
        "label": "Mức đóng của người lao động",
        "aliases": ["người lao động đóng", "khấu trừ từ tiền lương"],
        "parents": ["contribution"],
    },
    {
        "id": "maternity",
        "label": "Chế độ thai sản",
        "aliases": [
            "thai sản",
            "sinh con",
            "mang thai",
            "khám thai",
            "sảy thai",
            "nhận nuôi con nuôi",
            "mang thai hộ",
            "tránh thai",
            "triệt sản",
        ],
        "parents": ["social_insurance"],
    },
    {
        "id": "sick_leave",
        "label": "Chế độ ốm đau",
        "aliases": ["ốm đau", "nghỉ ốm", "chữa bệnh", "điều trị"],
        "parents": ["social_insurance"],
    },
    {
        "id": "work_accident",
        "label": "Tai nạn lao động, bệnh nghề nghiệp",
        "aliases": ["tai nạn lao động", "bệnh nghề nghiệp", "tnlđ", "bnn"],
        "parents": ["social_insurance"],
    },
    {
        "id": "survivorship",
        "label": "Chế độ tử tuất",
        "aliases": ["tử tuất", "trợ cấp tuất", "mai táng phí", "thân nhân"],
        "parents": ["social_insurance"],
    },
    {
        "id": "employee",
        "label": "Người lao động",
        "aliases": ["người lao động", "lao động nam", "lao động nữ"],
    },
    {
        "id": "employer",
        "label": "Người sử dụng lao động",
        "aliases": ["người sử dụng lao động", "doanh nghiệp", "đơn vị"],
    },
    {
        "id": "labor_contract",
        "label": "Hợp đồng lao động",
        "aliases": ["hợp đồng lao động", "chấm dứt hợp đồng", "hợp đồng"],
    },
    {
        "id": "foreign_worker",
        "label": "Người lao động nước ngoài",
        "aliases": ["người lao động nước ngoài", "lao động nước ngoài"],
        "parents": ["employee"],
    },
    {
        "id": "disability",
        "label": "Suy giảm khả năng lao động",
        "aliases": [
            "suy giảm khả năng lao động",
            "giám định y khoa",
            "không còn khả năng lao động",
        ],
    },
    {
        "id": "hazardous_work",
        "label": "Công việc nặng nhọc, độc hại",
        "aliases": [
            "nặng nhọc",
            "độc hại",
            "nguy hiểm",
            "đặc biệt khó khăn",
            "vùng có điều kiện kinh tế",
        ],
    },
    {
        "id": "legal_dossier",
        "label": "Hồ sơ, giấy tờ",
        "aliases": ["hồ sơ", "giấy tờ", "đơn đề nghị", "giấy khai sinh", "giấy chứng sinh"],
    },
    {
        "id": "one_time_social_insurance_dossier",
        "label": "Hồ sơ hưởng BHXH một lần",
        "aliases": [
            "hồ sơ hưởng bhxh một lần",
            "hồ sơ hưởng bảo hiểm xã hội một lần",
            "đơn đề nghị hưởng bhxh một lần",
        ],
        "parents": ["one_time_social_insurance", "legal_dossier"],
    },
    {
        "id": "social_insurance_book",
        "label": "Sổ BHXH",
        "aliases": ["sổ bhxh", "sổ bảo hiểm xã hội"],
        "parents": ["legal_dossier"],
    },
    {
        "id": "complaint",
        "label": "Khiếu nại về BHXH",
        "aliases": ["khiếu nại", "tố cáo", "khởi kiện"],
    },
    {
        "id": "prohibited_acts",
        "label": "Hành vi bị nghiêm cấm",
        "aliases": ["trốn đóng", "chậm đóng", "gian lận", "giả mạo hồ sơ", "chiếm dụng"],
    },
    {
        "id": "state_support",
        "label": "Hỗ trợ của Nhà nước",
        "aliases": ["hỗ trợ", "hộ nghèo", "cận nghèo", "nhà nước hỗ trợ"],
        "parents": ["voluntary_social_insurance"],
    },
    {
        "id": "reservation",
        "label": "Bảo lưu thời gian đóng",
        "aliases": ["bảo lưu thời gian đóng", "bảo lưu"],
        "parents": ["contribution"],
    },
    {
        "id": "pension_adjustment",
        "label": "Điều chỉnh lương hưu",
        "aliases": ["điều chỉnh lương hưu", "chỉ số giá tiêu dùng", "tăng trưởng kinh tế"],
        "parents": ["pension"],
    },
)


_THRESHOLD_PATTERN = re.compile(
    r"\b\d+(?:[,.]\d+)?\s*(?:%|tuổi|tháng|năm|ngày|lần)\b",
    re.IGNORECASE | re.UNICODE,
)
_DATE_PATTERN = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


def build_ontology_file(
    corpus_path: Union[str, Path],
    ontology_path: Union[str, Path],
    *,
    pretty: bool = True,
) -> Dict[str, Any]:
    ontology = build_bhxh_ontology(corpus_path)
    path = Path(ontology_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ontology, ensure_ascii=False, indent=2 if pretty else None)
        + settings.NEWLINE,
        encoding=settings.PATH_ENCODING,
    )
    return ontology


def build_bhxh_ontology(corpus_path: Union[str, Path]) -> Dict[str, Any]:
    records = _load_corpus_records(corpus_path)
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Set[Tuple[str, str, str]] = set()
    chunks: List[Dict[str, Any]] = []

    for spec in CONCEPT_SPECS:
        concept_id = _concept_node_id(str(spec["id"]))
        nodes[concept_id] = {
            "id": concept_id,
            "type": "concept",
            "label": str(spec["label"]),
            "aliases": list(spec.get("aliases", [])),
            "parents": list(spec.get("parents", []) or []),
        }
        for parent in spec.get("parents", []) or []:
            edges.add((concept_id, "is_a", _concept_node_id(str(parent))))

    for record in records:
        chunk_id = str(record[settings.FIELD_ID])
        chunk_node_id = _chunk_node_id(chunk_id)
        document = _optional_str(record.get(settings.FIELD_DOCUMENT))
        article = _optional_str(record.get(settings.FIELD_ARTICLE))
        clause = _optional_str(record.get(settings.FIELD_CLAUSE))
        point = _optional_str(record.get(settings.FIELD_POINT))
        text = str(record[settings.FIELD_TEXT])

        nodes[chunk_node_id] = {
            "id": chunk_node_id,
            "type": "legal_rule",
            "label": _rule_label(record),
            "chunk_id": chunk_id,
            "text": text,
            "document": document,
            "article": article,
            "clause": clause,
            "point": point,
        }

        document_node_id = _document_node_id(document) if document else None
        article_node_id = (
            _article_node_id(document, article) if document and article else None
        )
        clause_node_id = (
            _clause_node_id(document, article, clause)
            if document and article and clause
            else None
        )

        if document_node_id:
            nodes.setdefault(
                document_node_id,
                {"id": document_node_id, "type": "document", "label": document},
            )
            edges.add((chunk_node_id, "cites_document", document_node_id))

        if article_node_id:
            nodes.setdefault(
                article_node_id,
                {
                    "id": article_node_id,
                    "type": "article",
                    "label": f"Điều {article}",
                    "document": document,
                    "article": article,
                },
            )
            edges.add((chunk_node_id, "cites_article", article_node_id))
            if document_node_id:
                edges.add((article_node_id, "part_of", document_node_id))

        if clause_node_id:
            nodes.setdefault(
                clause_node_id,
                {
                    "id": clause_node_id,
                    "type": "clause",
                    "label": f"Điều {article} khoản {clause}",
                    "document": document,
                    "article": article,
                    "clause": clause,
                },
            )
            edges.add((chunk_node_id, "cites_clause", clause_node_id))
            if article_node_id:
                edges.add((clause_node_id, "part_of", article_node_id))

        matched_concepts = sorted(_match_concept_ids(text))
        for concept_id in matched_concepts:
            edges.add((chunk_node_id, "mentions", _concept_node_id(concept_id)))

        thresholds = sorted(_extract_threshold_labels(text), key=_threshold_sort_key)
        threshold_ids: List[str] = []
        for label in thresholds:
            threshold_node_id = _threshold_node_id(label)
            threshold_ids.append(threshold_node_id)
            nodes.setdefault(
                threshold_node_id,
                {
                    "id": threshold_node_id,
                    "type": "threshold",
                    "label": label,
                    "normalized": _normalize_text(label),
                },
            )
            edges.add((chunk_node_id, "has_threshold", threshold_node_id))

        chunks.append(
            {
                settings.FIELD_ID: chunk_id,
                settings.FIELD_TEXT: text,
                settings.FIELD_DOCUMENT: document,
                settings.FIELD_ARTICLE: article,
                settings.FIELD_CLAUSE: clause,
                settings.FIELD_POINT: point,
                "node_id": chunk_node_id,
                "concept_ids": matched_concepts,
                "threshold_ids": threshold_ids,
            }
        )

    return {
        "type": "bhxh_ontology",
        "version": 1,
        "source": str(Path(corpus_path)),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": sorted(nodes.values(), key=lambda item: item["id"]),
        "edges": [
            {"source": source, "relation": relation, "target": target}
            for source, relation, target in sorted(edges)
        ],
        "chunks": chunks,
    }


def concept_specs_by_id() -> Dict[str, ConceptSpec]:
    return {str(spec["id"]): dict(spec) for spec in CONCEPT_SPECS}


def normalize_for_matching(text: str) -> str:
    return _normalize_text(text)


def extract_query_threshold_ids(text: str) -> Set[str]:
    return {_threshold_node_id(label) for label in _extract_threshold_labels(text)}


def match_query_concept_ids(text: str) -> Set[str]:
    return _match_concept_ids(text)


def _load_corpus_records(corpus_path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(corpus_path)
    records: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    with path.open(settings.FILE_MODE_READ, encoding=settings.PATH_ENCODING) as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    settings.ERROR_INVALID_JSONL_TEMPLATE.format(
                        line_number=line_number,
                        error=exc,
                    )
                ) from exc
            _validate_record(record, line_number)
            chunk_id = str(record[settings.FIELD_ID])
            if chunk_id in seen_ids:
                raise ValueError(
                    settings.ERROR_DUPLICATE_CORPUS_ID_TEMPLATE.format(
                        line_number=line_number,
                        chunk_id=chunk_id,
                    )
                )
            seen_ids.add(chunk_id)
            records.append(dict(record))
    return records


def _validate_record(record: Mapping[str, Any], line_number: int) -> None:
    for field_name in (settings.FIELD_ID, settings.FIELD_TEXT):
        if field_name not in record:
            raise ValueError(
                settings.ERROR_MISSING_CORPUS_FIELD_TEMPLATE.format(
                    field=field_name,
                    line_number=line_number,
                )
            )
        if not str(record[field_name]):
            raise ValueError(
                settings.ERROR_EMPTY_CORPUS_FIELD_TEMPLATE.format(
                    field=field_name,
                    line_number=line_number,
                )
            )


def _match_concept_ids(text: str) -> Set[str]:
    normalized_text = _normalize_text(text)
    matches: Set[str] = set()
    for spec in CONCEPT_SPECS:
        aliases = list(spec.get("aliases", [])) + [str(spec["label"])]
        for alias in aliases:
            normalized_alias = _normalize_text(str(alias))
            if not normalized_alias:
                continue
            if _contains_normalized_phrase(normalized_text, normalized_alias):
                matches.add(str(spec["id"]))
                break
    return matches


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    if len(phrase) <= 4:
        return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None
    return phrase in text


def _extract_threshold_labels(text: str) -> Set[str]:
    labels = {_clean_threshold(match.group(0)) for match in _THRESHOLD_PATTERN.finditer(text)}
    labels.update(match.group(0) for match in _DATE_PATTERN.finditer(text))
    labels.update(match.group(0) for match in _YEAR_PATTERN.finditer(text))
    return {label for label in labels if label}


def _clean_threshold(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _threshold_sort_key(label: str) -> Tuple[int, str]:
    return (0 if re.search(r"\d", label) else 1, _normalize_text(label))


def _rule_label(record: Mapping[str, Any]) -> str:
    document = _optional_str(record.get(settings.FIELD_DOCUMENT))
    article = _optional_str(record.get(settings.FIELD_ARTICLE))
    clause = _optional_str(record.get(settings.FIELD_CLAUSE))
    text = str(record.get(settings.FIELD_TEXT) or settings.EMPTY_STRING)
    ref_parts = [part for part in (document, f"Điều {article}" if article else None, f"Khoản {clause}" if clause else None) if part]
    if ref_parts:
        return settings.COMMA_SPACE.join(ref_parts)
    sentence = text.split(settings.PERIOD, 1)[0].strip()
    return sentence[:120] if sentence else str(record[settings.FIELD_ID])


def _optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text or settings.EMPTY_STRING)
    without_marks = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    without_marks = without_marks.replace("đ", "d").replace("Đ", "d")
    lowered = without_marks.lower()
    return re.sub(r"\s+", " ", lowered).strip()


def _slug(value: str) -> str:
    normalized = _normalize_text(value)
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return slug or "unknown"


def _concept_node_id(concept_id: str) -> str:
    return f"concept:{_slug(concept_id)}"


def _chunk_node_id(chunk_id: str) -> str:
    return f"chunk:{_slug(chunk_id)}"


def _document_node_id(document: str) -> str:
    return f"document:{_slug(document)}"


def _article_node_id(document: str, article: str) -> str:
    return f"article:{_slug(document)}:{_slug(article)}"


def _clause_node_id(document: str, article: str, clause: str) -> str:
    return f"clause:{_slug(document)}:{_slug(article)}:{_slug(clause)}"


def _threshold_node_id(label: str) -> str:
    return f"threshold:{_slug(label)}"
