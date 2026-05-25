"""Verify môi trường cho Bước 5 (embedding).

Kiểm tra:
1. import được torch, sentence_transformers, pyarrow, pandas
2. CUDA available không + tên GPU
3. Load BGE-M3 + check dim = 1024
4. Encode 1 câu thử + đo thời gian

Exit code:
  0 — mọi thứ OK
  1 — thiếu package
  2 — model load fail
  3 — encode fail / dim sai
"""

from __future__ import annotations

import sys
import time


def _try_import(name: str) -> tuple[bool, str]:
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "?")
        return True, ver
    except ImportError as e:
        return False, str(e)


def main() -> int:
    print("=== VERIFY B5 ===\n")

    # 1. Imports
    print("[1/4] Kiểm tra packages")
    required = ["torch", "sentence_transformers", "pyarrow", "pandas", "tqdm"]
    missing = []
    for pkg in required:
        ok, info = _try_import(pkg)
        mark = "✓" if ok else "✗"
        print(f"  {mark} {pkg:<25} {info}")
        if not ok:
            missing.append(pkg)
    if missing:
        print(f"\n  THIẾU: {missing}")
        print("  Chạy: .\\scripts\\install_b5.ps1")
        return 1

    # 2. CUDA
    print("\n[2/4] Kiểm tra device")
    import torch

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        print(f"  ✓ CUDA available: {n} GPU(s)")
        for i, name in enumerate(names):
            print(f"    [{i}] {name}")
        print(f"  CUDA version (build): {torch.version.cuda}")
        device_hint = "cuda"
    else:
        print(f"  ✓ CPU mode (torch {torch.__version__})")
        device_hint = "cpu"
    print(f"  Khuyến nghị set EMBED_DEVICE={device_hint} trong .env")

    # 3. Load model
    print("\n[3/4] Load BAAI/bge-m3")
    t0 = time.time()
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("BAAI/bge-m3", device=device_hint)
    except Exception as e:
        print(f"  ✗ Load fail: {type(e).__name__}: {e}")
        return 2
    elapsed = time.time() - t0
    dim = model.get_sentence_embedding_dimension()
    print(f"  ✓ Loaded in {elapsed:.1f}s, dim = {dim}")
    if dim != 1024:
        print(f"  ✗ Dim sai (expected 1024, got {dim})")
        return 3

    # 4. Encode thử
    print("\n[4/4] Encode 3 câu tiếng Việt thử")
    samples = [
        "Người lao động được hưởng lương hưu khi đủ tuổi nghỉ hưu.",
        "Bảo hiểm xã hội bắt buộc là loại hình do Nhà nước tổ chức.",
        "Cơ quan bảo hiểm xã hội có trách nhiệm cấp sổ bảo hiểm xã hội.",
    ]
    t0 = time.time()
    try:
        embs = model.encode(samples, show_progress_bar=False, normalize_embeddings=True)
    except Exception as e:
        print(f"  ✗ Encode fail: {type(e).__name__}: {e}")
        return 3
    elapsed = time.time() - t0
    print(
        f"  ✓ Encoded {len(samples)} câu trong {elapsed:.2f}s ({elapsed/len(samples)*1000:.0f}ms/câu)"
    )
    print(f"  Shape: {embs.shape}")
    # Cosine similarity giữa 2 câu BHXH (1, 2) so với câu 3 (cơ quan)
    import numpy as np

    sim12 = float(np.dot(embs[0], embs[1]))
    sim13 = float(np.dot(embs[0], embs[2]))
    sim23 = float(np.dot(embs[1], embs[2]))
    print("  Cosine sim mẫu:")
    print(f"    câu1 vs câu2 (cùng chủ đề BHXH+hưu trí): {sim12:.3f}")
    print(f"    câu1 vs câu3 (BHXH vs cơ quan)          : {sim13:.3f}")
    print(f"    câu2 vs câu3 (BHXH vs cơ quan)          : {sim23:.3f}")

    # Ước lượng thời gian chạy full
    n_units = 543 + 359 + 141  # Clause + Point + Article
    est_total = elapsed / len(samples) * n_units
    print(f"\n  Ước lượng encode toàn bộ {n_units} unit: ~{est_total:.0f}s")

    print("\n=== OK — sẵn sàng cho Bước 5 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
