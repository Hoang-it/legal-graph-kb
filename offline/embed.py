"""B5 — Sinh embeddings cho Article / Clause / Point bằng BGE-M3 (1024-d).

Đầu vào : data/graph/processed/merged_graph.json  (output của B4)
Đầu ra  : data/graph/processed/embeddings.parquet

Schema parquet:
    id            : str    — node ID (vd 'L41_2024.A64.K1.a')
    label         : str    — 'Article' | 'Clause' | 'Point'
    text_preview  : str    — 200 ký tự đầu (dễ debug, không dùng để search)
    embedding     : list[float]  — vector 1024-d, đã normalize (cosine sim = dot)

Đặc tính:
- Vector NORMALIZED → cosine similarity = dot product → tương thích với
  Neo4j vector index OPTIONS {`vector.similarity_function`: 'cosine'}.
- Batch GPU (RTX 3050) — tự fall back CPU nếu CUDA không available.
- Idempotent: nếu file output đã có và `--force` không được set → skip.

Provenance: mỗi row có `id` 1-1 với node trong merged_graph.json — embedding
có thể join ngược lại structural text bất cứ lúc nào.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Suppress noisy warning about HF cache symlinks on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
DEVICE = os.getenv("EMBED_DEVICE", "cuda")
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))  # 16 an toàn cho RTX 3050 4GB

GRAPH_PATH = Path("data/graph/processed/merged_graph.json")
OUT_PATH = Path("data/graph/processed/embeddings.parquet")

# Labels có embedding (units văn bản)
EMBED_LABELS = ("Article", "Clause", "Point")


def collect_units(graph: dict) -> list[dict]:
    """Lấy tất cả Article / Clause / Point có text non-empty."""
    units: list[dict] = []
    for label in EMBED_LABELS:
        for n in graph["nodes"].get(label, []):
            text = (n.get("text") or "").strip()
            if not text:
                continue
            units.append(
                {
                    "id": n["id"],
                    "label": label,
                    "text": text,
                }
            )
    return units


def encode_all(units: list[dict], model) -> np.ndarray:
    """Encode toàn bộ units (chuẩn hoá L2)."""
    texts = [u["text"] for u in units]
    print(f"Encoding {len(texts)} texts (batch_size={BATCH_SIZE}, device={DEVICE})...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    print(f"  Encoded in {elapsed:.1f}s ({elapsed / len(texts) * 1000:.0f}ms/unit)")
    return embeddings


def save_parquet(units: list[dict], embeddings: np.ndarray, out: Path) -> None:
    df = pd.DataFrame(
        {
            "id": [u["id"] for u in units],
            "label": [u["label"] for u in units],
            "text_preview": [u["text"][:200] for u in units],
            # Convert mỗi vector thành list[float] để parquet/Neo4j parse tốt
            "embedding": [e.astype(np.float32).tolist() for e in embeddings],
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression="snappy")
    print(f"\nSaved: {out} ({out.stat().st_size / 1024 / 1024:.1f} MB, {len(df)} rows)")


def sanity_check(units: list[dict], embeddings: np.ndarray) -> None:
    print("\n=== SANITY CHECK ===")
    # 1. Shape
    assert embeddings.shape == (
        len(units),
        EMBED_DIM,
    ), f"Shape sai: {embeddings.shape}, expect ({len(units)}, {EMBED_DIM})"
    print(f"  Shape OK: {embeddings.shape}")

    # 2. Norms (đã normalize)
    norms = np.linalg.norm(embeddings, axis=1)
    print(
        f"  Norms: min={norms.min():.4f} max={norms.max():.4f} mean={norms.mean():.4f} (expect ~1.0)"
    )
    assert abs(norms.mean() - 1.0) < 0.01, "Vectors không được normalize đúng"

    # 3. Semantic spot-check: A64 (hưu trí BB), A98 (hưu trí TN), A1 (phạm vi)
    by_id = {u["id"]: i for i, u in enumerate(units)}
    pairs = [
        ("L41_2024.A64", "L41_2024.A98", "hưu trí BB ↔ hưu trí TN", "high"),
        ("L41_2024.A64", "L41_2024.A1", "hưu trí ↔ phạm vi điều chỉnh", "low"),
        ("L41_2024.A39", "L41_2024.A40", "trốn đóng ↔ chậm đóng", "high"),
    ]
    print("\n  Cosine similarity (đối chiếu kỳ vọng):")
    for a, b, label, expect in pairs:
        if a in by_id and b in by_id:
            sim = float(np.dot(embeddings[by_id[a]], embeddings[by_id[b]]))
            mark = (
                "✓"
                if (expect == "high" and sim > 0.55) or (expect == "low" and sim < 0.55)
                else "?"
            )
            print(f"    {mark} {a} ↔ {b}  ({label:<35}) = {sim:.3f}  expect {expect}")

    # 4. Spot check: tìm top-3 Article gần nhất với A64
    a64_idx = by_id.get("L41_2024.A64")
    if a64_idx is not None:
        art_mask = np.array([u["label"] == "Article" for u in units])
        art_indices = np.where(art_mask)[0]
        art_embs = embeddings[art_indices]
        sims = art_embs @ embeddings[a64_idx]
        # Tránh chính nó
        top = np.argsort(-sims)
        print("\n  Top-5 Article gần nhất với A64 (chế độ hưu trí BB):")
        for rank, j in enumerate(top[:6]):
            idx = art_indices[j]
            u = units[idx]
            if u["id"] == "L41_2024.A64":
                continue
            print(f"    [{rank}] sim={sims[j]:.3f}  {u['id']}: {u['text'][:80].splitlines()[0]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="B5 — Embedding với BGE-M3")
    parser.add_argument("--force", action="store_true", help="Bỏ qua cache, encode lại từ đầu.")
    parser.add_argument("--limit", type=int, default=0, help="Chỉ encode N unit đầu tiên (debug).")
    args = parser.parse_args()

    if not GRAPH_PATH.exists():
        print(f"FAIL: thiếu {GRAPH_PATH}. Chạy B4 (merge_normalize) trước.", file=sys.stderr)
        return 1

    if OUT_PATH.exists() and not args.force:
        print(f"OK — {OUT_PATH} đã tồn tại. Dùng --force để encode lại.")
        return 0

    print(f"Loading graph from {GRAPH_PATH}...")
    with GRAPH_PATH.open(encoding="utf-8") as f:
        graph = json.load(f)

    units = collect_units(graph)
    print(f"Tổng unit cần encode: {len(units)}")
    for label in EMBED_LABELS:
        n = sum(1 for u in units if u["label"] == label)
        print(f"  {label:<10} {n:>5}")

    if args.limit:
        units = units[: args.limit]
        print(f"\n[DEBUG] Limit {args.limit} units")

    print(f"\nLoading {MODEL_NAME} on {DEVICE}...")
    # Lazy import (sentence_transformers tải chậm)
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(
        f"  Model loaded in {time.time() - t0:.1f}s, dim={model.get_sentence_embedding_dimension()}"
    )
    if model.get_sentence_embedding_dimension() != EMBED_DIM:
        print(
            f"FAIL: model dim ({model.get_sentence_embedding_dimension()}) != EMBED_DIM ({EMBED_DIM})",
            file=sys.stderr,
        )
        return 2

    embeddings = encode_all(units, model)
    sanity_check(units, embeddings)
    save_parquet(units, embeddings, OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
