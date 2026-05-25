"""Test cho src/schema.py — Pydantic models + provenance enforcement."""

import pytest
from pydantic import ValidationError

from src import schema as S

# ---------- Provenance: semantic node BẮT BUỘC có mentioned_in ----------


def test_subject_phai_co_mentioned_in():
    with pytest.raises(ValidationError):
        S.Subject(
            id="subject:nguoi-lao-dong", name="Người lao động", type="individual", mentioned_in=[]
        )


def test_subject_hop_le():
    s = S.Subject(
        id="subject:nguoi-lao-dong",
        name="Người lao động",
        type="individual",
        mentioned_in=["L41_2024.A2.K1"],
    )
    assert s.mentioned_in == ["L41_2024.A2.K1"]


# ---------- Provenance: semantic edge BẮT BUỘC có source_clause + source_text ----------


def test_semantic_edge_thieu_source_clause():
    with pytest.raises(ValidationError):
        S.SemanticEdge(
            type="ENTITLED_TO",
            src="subject:nguoi-lao-dong",
            dst="benefit:huu-tri",
            source_text="text",
            # thiếu source_clause
        )


def test_semantic_edge_source_clause_sai_format():
    with pytest.raises(ValidationError):
        S.SemanticEdge(
            type="ENTITLED_TO",
            src="subject:nguoi-lao-dong",
            dst="benefit:huu-tri",
            source_clause="not-a-valid-clause-id",
            source_text="text",
        )


def test_semantic_edge_hop_le_clause_id():
    e = S.SemanticEdge(
        type="ENTITLED_TO",
        src="subject:nguoi-lao-dong",
        dst="benefit:huu-tri",
        source_clause="L41_2024.A64.K1",
        source_text="Người lao động được hưởng lương hưu khi đủ điều kiện...",
    )
    assert e.source_clause == "L41_2024.A64.K1"


def test_semantic_edge_hop_le_point_id():
    e = S.SemanticEdge(
        type="REQUIRES",
        src="benefit:huu-tri",
        dst="cond:du-tuoi-nghi-huu",
        source_clause="L41_2024.A64.K1.a",
        source_text="...đủ tuổi nghỉ hưu theo quy định...",
    )
    assert e.source_clause == "L41_2024.A64.K1.a"


def test_semantic_edge_source_text_qua_dai():
    with pytest.raises(ValidationError):
        S.SemanticEdge(
            type="ENTITLED_TO",
            src="subject:nguoi-lao-dong",
            dst="benefit:huu-tri",
            source_clause="L41_2024.A64.K1",
            source_text="x" * 301,  # > 300
        )


# ---------- Structural ----------


def test_clause_hop_le():
    c = S.ClauseNode(
        id="L41_2024.A64.K1",
        number=1,
        text="Người lao động được hưởng lương hưu khi...",
        article_id="L41_2024.A64",
    )
    assert c.text != ""


# ---------- Tiện ích liệt kê ----------


def test_co_du_label_va_edge_types():
    assert "Article" in S.ALL_NODE_LABELS
    assert "Subject" in S.ALL_NODE_LABELS
    assert "ENTITLED_TO" in S.SEMANTIC_EDGE_TYPES
    assert "REFERENCES" in S.REFERENCE_EDGE_TYPES
    assert "HAS_CHAPTER" in S.STRUCTURAL_EDGE_TYPES
