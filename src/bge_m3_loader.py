"""Single source of truth for loading BGE-M3 — with or without a LoRA adapter.

Both ``offline/embed.py`` (corpus encoding) and ``src/retrieval/pipeline.py``
(per-query encoding) must encode with the *same* model state to produce
compatible vectors. Centralising the load logic prevents the silent failure
mode where one side loads vanilla BGE-M3 and the other loads a fine-tuned
adapter — cosine similarities silently degrade with no error.

Adapter format: HuggingFace PEFT (``adapter_config.json`` +
``adapter_model.safetensors``), produced by the Colab notebook
``notebooks/finetune_bge_m3.ipynb``.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

DEFAULT_BASE = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
DEFAULT_DEVICE = os.getenv("EMBED_DEVICE", "cuda")


def load_bge_m3(
    adapter_path: str | Path | None = None,
    base_model: str = DEFAULT_BASE,
    device: str = DEFAULT_DEVICE,
):
    """Return a SentenceTransformer ready for ``.encode(...)``.

    - ``adapter_path=None`` → vanilla BGE-M3 (the Sprint 1 baseline).
    - ``adapter_path=<dir>``→ BGE-M3 + LoRA adapter merged at the transformer
      layer so the SentenceTransformer interface remains identical.

    The loader is intentionally strict: if ``adapter_path`` is given but the
    directory does not contain ``adapter_config.json`` it raises immediately
    rather than silently falling back to vanilla. Silent fallback would
    corrupt the index↔query symmetry described above.
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading {base_model} on {device}...", file=sys.stderr)
    t0 = time.time()
    model = SentenceTransformer(base_model, device=device)
    print(f"  base loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    if adapter_path is None:
        return model

    adapter_path = Path(adapter_path).expanduser().resolve()
    config_file = adapter_path / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"adapter_path {adapter_path} does not contain adapter_config.json — "
            "this is not a PEFT/LoRA adapter directory."
        )

    print(f"  attaching LoRA adapter from {adapter_path}...", file=sys.stderr)
    t0 = time.time()
    try:
        from peft import PeftModel
    except ImportError as e:
        raise ImportError(
            "Install peft to load a LoRA adapter: pip install peft"
        ) from e

    # SentenceTransformer wraps the HF transformer in module index 0.
    transformer_module = model._modules["0"]
    base_hf = transformer_module.auto_model
    adapted = PeftModel.from_pretrained(base_hf, str(adapter_path))
    # Merge so inference path is regular SentenceTransformer flow.
    adapted = adapted.merge_and_unload()
    transformer_module.auto_model = adapted
    if device.startswith("cuda"):
        transformer_module.auto_model = transformer_module.auto_model.to(device)
    print(f"  adapter merged in {time.time() - t0:.1f}s", file=sys.stderr)
    return model


def adapter_path_from_env() -> Path | None:
    """Return the adapter path declared by ``BGE_M3_ADAPTER_PATH`` env var.

    Used by runtime modules so a single env switch flips the whole runtime
    between vanilla and tuned encoding. Returns None if unset/blank.
    """
    raw = (os.environ.get("BGE_M3_ADAPTER_PATH") or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"BGE_M3_ADAPTER_PATH={raw} but directory does not exist."
        )
    return p
