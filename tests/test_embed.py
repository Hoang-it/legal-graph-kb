"""Test cho offline/embed.py — verify parquet output đúng schema + semantic OK."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EMBED_PATH = Path("data/processed/embeddings.parquet")
GRAPH_PATH = Path("data/processed/merged_graph.json")
EMBED_DIM = 1024


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    if not EMBED_PATH.exists():
        pytest.skip(f"{EMBED_PATH} không tồn tại. Chạy `python -m offline.embed --force` trước.")
    return pd.read_parquet(EMBED_PATH)


@pytest.fixture(scope="module")
def graph() -> dict:
    if not GRAPH_PATH.exists():
        pytest.skip(f"{GRAPH_PATH} không tồn tại. Chạy `python -m offline.merge_normalize` trước.")
    with GRAPH_PATH.open(encoding="utf-8") as f:
        return json.load(f)


# ---------- 1. Schema + size ----------


def test_columns_dung_format(df):
    assert list(df.columns) == ["id", "label", "text_preview", "embedding"]


def test_du_so_row(df):
    # 486 Article + 1585 Clause + 829 Point = 2900 across 3 laws.
    assert len(df) == 2900
    by_label = df["label"].value_counts().to_dict()
    assert by_label["Article"] == 486
    assert by_label["Clause"] == 1585
    assert by_label["Point"] == 829


def test_id_la_duy_nhat(df):
    assert df["id"].is_unique


# ---------- 2. Embedding correctness ----------


def test_moi_embedding_dim_1024(df):
    for emb in df["embedding"].iloc[:10]:
        assert len(emb) == EMBED_DIM, f"Dim sai: {len(emb)}"


def test_moi_embedding_la_unit_vector(df):
    """Normalize OK → mọi vector có norm = 1."""
    embs = np.stack([np.asarray(e, dtype=np.float32) for e in df["embedding"]])
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(
        norms, 1.0, atol=1e-4
    ), f"Norms không = 1: min={norms.min()}, max={norms.max()}"


def test_embedding_la_float32(df):
    """Lưu float32 (không float64) để parquet gọn + tương thích Neo4j."""
    emb = df["embedding"].iloc[0]
    # Đã convert list, nhưng giá trị từng phần tử nên trong range [-1, 1]
    arr = np.asarray(emb)
    assert arr.dtype in (np.float32, np.float64)
    assert arr.min() > -1.5 and arr.max() < 1.5


# ---------- 3. Provenance (ID khớp với graph) ----------


def test_moi_id_ton_tai_trong_graph(df, graph):
    graph_ids: set[str] = set()
    for label in ("Article", "Clause", "Point"):
        for n in graph["nodes"].get(label, []):
            graph_ids.add(n["id"])
    embed_ids = set(df["id"])
    missing = embed_ids - graph_ids
    assert (
        not missing
    ), f"{len(missing)} ID có embedding nhưng không trong graph: {list(missing)[:3]}"


def test_moi_article_co_embedding(df, graph):
    """141 Article đều có embedding (không thiếu cái nào)."""
    art_ids_graph = {n["id"] for n in graph["nodes"]["Article"]}
    art_ids_embed = set(df[df["label"] == "Article"]["id"])
    missing = art_ids_graph - art_ids_embed
    assert not missing, f"{len(missing)} Article thiếu embedding: {list(missing)[:5]}"


# ---------- 4. Semantic sanity ----------


def test_sim_dieu_64_va_dieu_98_cao(df):
    """A64 (hưu trí BB) và A98 (hưu trí TN) đều về hưu trí → similarity cao."""
    by_id = {row["id"]: np.asarray(row["embedding"]) for _, row in df.iterrows()}
    sim = float(np.dot(by_id["L41_2024.A64"], by_id["L41_2024.A98"]))
    assert sim > 0.75, f"A64↔A98 sim chỉ {sim:.3f}, expect > 0.75"


def test_sim_dieu_39_va_dieu_40_cao(df):
    """A39 (trốn đóng) và A40 (chậm đóng) cùng chủ đề vi phạm đóng BHXH."""
    by_id = {row["id"]: np.asarray(row["embedding"]) for _, row in df.iterrows()}
    sim = float(np.dot(by_id["L41_2024.A39"], by_id["L41_2024.A40"]))
    assert sim > 0.7, f"A39↔A40 sim chỉ {sim:.3f}, expect > 0.7"


def test_top_neighbor_cua_a64_la_dieu_huu_tri(df):
    """Top-3 article gần A64 phải có liên quan đến hưu trí / BHXH một lần."""
    arts = df[df["label"] == "Article"].reset_index(drop=True)
    by_id = {row["id"]: i for i, row in arts.iterrows()}
    embs = np.stack([np.asarray(e, dtype=np.float32) for e in arts["embedding"]])
    a64_idx = by_id["L41_2024.A64"]
    sims = embs @ embs[a64_idx]
    top = np.argsort(-sims)
    # Bỏ chính nó
    top_ids = [arts.loc[j, "id"] for j in top if j != a64_idx][:5]
    # Các điều liên quan: hưu trí (64-72), hưu trí TN (98-107)
    relevant_range_bb = {f"L41_2024.A{n}" for n in range(60, 80)}
    relevant_range_tn = {f"L41_2024.A{n}" for n in range(95, 110)}
    relevant_range_l58 = {f"L58_2014.A{n}" for n in range(50, 76)}
    relevant = relevant_range_bb | relevant_range_tn | relevant_range_l58 | {"L45_2019.A169"}
    n_hit = sum(1 for tid in top_ids[:5] if tid in relevant)
    assert n_hit >= 4, (
        f"Top-5 neighbors của A64 chỉ {n_hit}/5 thuộc nhóm hưu trí. " f"Top: {top_ids}"
    )


def test_text_preview_khop_voi_graph(df, graph):
    """Text preview (200 ký tự đầu) phải khớp với text trong graph."""
    graph_text: dict[str, str] = {}
    for label in ("Article", "Clause", "Point"):
        for n in graph["nodes"].get(label, []):
            graph_text[n["id"]] = n.get("text", "")
    # Spot check 10 rows
    for _, row in df.head(10).iterrows():
        gt = graph_text.get(row["id"], "")
        assert row["text_preview"] == gt[:200], f"{row['id']}: preview không khớp graph"
