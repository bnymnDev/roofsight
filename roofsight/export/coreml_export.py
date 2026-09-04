"""ONNX → Core ML ML Program (fp16, Neural Engine), plus the model card next to it."""

from __future__ import annotations

from pathlib import Path


def export_coreml(onnx_path: Path, out: Path, resolution: int = 640) -> Path:
    import coremltools as ct  # ≥ 8

    model = ct.converters.onnx.convert(
        model=str(onnx_path),
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        inputs=[ct.ImageType(name="image", shape=(1, 3, resolution, resolution), scale=1 / 255.0)],
    )
    model.short_description = "RoofSight roof-plane and obstacle segmentation (RF-DETR-Seg)"
    model.license = "Apache-2.0"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    return out


def model_card(
    model_name: str, run_id: str, dataset_version: str, metrics: dict[str, object], sha256: str
) -> str:
    lines = [
        f"# Model card: {model_name}",
        "",
        f"- Run: `{run_id}`",
        f"- Dataset: RoofSight {dataset_version}",
        "- Input: RGB 640×640, scale 1/255",
        "- Output: per-instance class logits, boxes, masks (RF-DETR-Seg head)",
        "- Precision: fp16 ML Program, Neural Engine",
        f"- SHA256: `{sha256}`",
        "",
        "## Metrics (test split, `roofsight eval`)",
        "",
    ]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.4f}")
    lines += [
        "",
        "## Intended use",
        "",
        "First on-site PV layout from a single ground-level photo. Not a substitute for a",
        "structural survey. Trained on residential roofs in DE/NL/AT; expect degraded results",
        "on flat commercial roofs and non-European roof styles.",
        "",
        "## License",
        "",
        "Model weights: Apache-2.0. Training data: RoofSight dataset, CC-BY-SA 4.0.",
    ]
    return "\n".join(lines) + "\n"
