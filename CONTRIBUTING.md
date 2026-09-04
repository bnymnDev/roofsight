# Contributing

Ground-view roof and rooftop-obstacle segmentation for on-site PV planning. Dataset,
benchmark, models, Core ML export, and a Swift package for roof geometry. Public monorepo.
Read [SPEC.md](SPEC.md) before implementing anything.

The Voltaris iOS app consumes this repo but lives elsewhere. Nothing app-specific goes in here.

## Stack

- Python 3.12, `uv` for everything (no pip, no conda). `uv run` for all scripts.
- PyTorch 2.x, `torchvision`, `rfdetr` (Roboflow, Apache-2.0) as primary trainer, `supervision`
  for mask/box utils
- SAM 3 via the official `sam3` package: **labeling only**, never shipped, never imported
  outside `roofsight/labeling/`
- `pycocotools` (COCO JSON is the canonical annotation format), `fiftyone` for dataset review
- `onnx`, `onnxruntime`, `coremltools` ≥ 8 for export
- `typer` for CLIs, `pydantic` for configs, `ruff` + `mypy --strict` for lint/types
- `pytest`; tests must run on CPU with tiny fixtures
- Swift 5.10+, Swift Package Manager, ARKit/RealityKit, tests via `swift test` (geometry only,
  no device tests in CI)
- Docs: MkDocs Material

## Layout

```
roofsight/                 # python package
  labeling/                # SAM 3 auto-labeling, prompt configs, review export
  data/                    # dataset build: mapillary download, dedupe, split, COCO writer
  train/                   # rfdetr fine-tune, configs in configs/
  eval/                    # metrics, leaderboard generation
  export/                  # onnx -> coreml, validation against torch
  cli.py                   # `roofsight <cmd>`
configs/
datasets/                  # .gitignored; DVC pointers only
swift/RoofGeometry/        # Swift package: pitch/azimuth from ARKit pose + mask
docs/
paper/                     # LaTeX, figures, results tables (generated)
```

## Rules

- Datasets and weights are never committed. Use DVC with an S3/R2 remote; only `.dvc` files
  are in git.
- License hygiene: every image record carries `source`, `license`, `attribution`. Only
  CC-BY-SA (Mapillary) and own photos. A build step fails if a record lacks a license.
- No faces, no license plates: run the anonymizer in `data/` before any image enters the
  dataset; the check is part of `roofsight data validate`.
- All annotations are COCO JSON with our fixed category list (SPEC.md). Category ids are
  stable; never renumber.
- Reproducibility: every train/eval run writes `run.json` (git sha, config hash, dataset
  version, seed, hardware). Results without `run.json` are not accepted into the leaderboard.
- Metrics are computed by `roofsight eval` only. No hand-computed numbers in docs or paper.
- Export: a Core ML model is only "done" when `roofsight export verify` shows mask IoU ≥ 0.98
  against the PyTorch model on the verify set.
- Swift package: pure functions, `Sendable`, no UIKit. ARKit types are converted to simple
  structs at the boundary so geometry is testable without a device.

## Commands

```sh
uv sync --group dev
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest

uv run roofsight data build --config configs/data/v0.1.yaml
uv run roofsight label --prompts configs/labeling/prompts.yaml --in datasets/v0.1 --out datasets/v0.1 --checkpoint weights/sam3.pt
uv run roofsight train --config configs/train/rfdetr-s.yaml
uv run roofsight eval --run runs/<id> --split test
uv run roofsight export coreml --run runs/<id>
uv run roofsight leaderboard          # regenerates docs/leaderboard.md and paper/results.tex

cd swift/RoofGeometry && swift test
```

## Working style

- Small commits, conventional commits. Work on `develop`; `main` only moves by merging
  `develop`.
- New category or prompt change → dataset version bump (`v0.x`) and changelog entry in
  `docs/dataset.md`.
- Any change to metrics → golden test in `tests/test_metrics.py` with the tiny fixture set.
- Keep `docs/leaderboard.md` generated, never edited by hand.
