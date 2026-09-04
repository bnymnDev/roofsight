"""PyTorch → ONNX, opset 17, static ``1×3×640×640``. Uses rfdetr's own exporter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

OPSET = 17


def export_onnx(model: Any, out_dir: Path, resolution: int = 640) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.export(output_dir=str(out_dir), opset_version=OPSET, simplify=True, resolution=resolution)
    candidates = sorted(out_dir.glob("*.onnx"))
    if not candidates:
        raise RuntimeError(f"rfdetr export wrote no .onnx into {out_dir}")
    return candidates[0]


def onnx_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
