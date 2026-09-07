"""``roofsight <cmd>``. Thin typer layer; the logic lives in the subpackages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from roofsight import __version__
from roofsight.coco import SPLITS, Split, read_coco

app = typer.Typer(help="RoofSight: ground-view roof segmentation dataset, benchmark and models.")
data_app = typer.Typer(help="Dataset build and validation.")
export_app = typer.Typer(help="ONNX and Core ML export.")
app.add_typer(data_app, name="data")
app.add_typer(export_app, name="export")
console = Console()
err = Console(stderr=True)


def _split(value: str) -> Split:
    for s in SPLITS:
        if s == value:
            return s
    raise typer.BadParameter(f"split must be one of {SPLITS}")


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="Print version and exit.")] = False,
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


# ---------------------------------------------------------------------------------- data


@data_app.command("build")
def data_build(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    no_roof_filter: Annotated[
        bool, typer.Option(help="Skip the SAM 3 roof presence filter.")
    ] = False,
    sam3_checkpoint: Annotated[
        str, typer.Option(help="facebook/sam3, a local HF dir, or a Meta *.pt")
    ] = "facebook/sam3",
) -> None:
    """Download, dedupe, anonymize, filter, split. Writes <out>/<version>/images.json."""
    import logging

    from roofsight.data.build import build
    from roofsight.data.config import DataConfig

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = DataConfig.load(config)
    roof_filter = None
    if not no_roof_filter:
        from roofsight.labeling.pipeline import load_image
        from roofsight.labeling.sam3 import load_segmenter, roof_fraction

        seg = load_segmenter(sam3_checkpoint)
        roof_filter = lambda p: roof_fraction(seg, load_image(p), cfg.roof_filter.prompt)  # noqa: E731
    ds = build(cfg, roof_filter)
    console.print(f"[green]built[/] {cfg.version}: {len(ds.images)} images")


@data_app.command("fetch")
def data_fetch(
    dataset: Annotated[Path, typer.Argument(help="Dataset directory with images.json")],
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    prune_missing: Annotated[
        bool, typer.Option(help="Remove images Mapillary no longer serves from the manifest")
    ] = False,
) -> None:
    """Re-download the images listed in images.json by Mapillary id, then anonymize them.

    images.json is the manifest in git; this restores the pixels on any machine. Images that
    are gone upstream are reported; they are removed from the manifest only with
    --prune-missing.
    """
    import logging

    from roofsight.data.build import fetch_manifest
    from roofsight.data.config import DataConfig

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from roofsight.data.mapillary import MapillaryAuthError

    ds = read_coco(dataset / "images.json")
    try:
        n, missing = fetch_manifest(ds, dataset / "images", DataConfig.load(config))
    except MapillaryAuthError as e:
        err.print(f"[red]{e}[/]")
        raise typer.Exit(2) from e
    if missing:
        names = ", ".join(r.file_name for r in missing[:5])
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        if prune_missing:
            ids = {r.id for r in missing}
            ds = ds.model_copy(update={"images": [r for r in ds.images if r.id not in ids]})
            ds.write(dataset / "images.json")
            err.print(f"[yellow]{len(missing)} images gone upstream, removed: {names}{more}[/]")
        else:
            err.print(
                f"[yellow]{len(missing)} images gone upstream, kept in the manifest: "
                f"{names}{more}. Rerun with --prune-missing to drop them.[/]"
            )
    present = sum((dataset / "images" / r.file_name).exists() for r in ds.images)
    console.print(
        f"[green]fetched {n}[/]; {present}/{len(ds.images)} images present in {dataset / 'images'}"
    )


@data_app.command("filter")
def data_filter(
    dataset: Annotated[Path, typer.Argument(help="Dataset directory with images.json")],
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    backend: Annotated[str | None, typer.Option(help="sam3 | file; default from config")] = None,
    threshold: Annotated[float | None, typer.Option(help="Override the config threshold")] = None,
    dry_run: Annotated[bool, typer.Option(help="Score and report, delete nothing")] = False,
    sam3_checkpoint: Annotated[
        str, typer.Option(help="facebook/sam3, a local HF dir, or a Meta *.pt")
    ] = "facebook/sam3",
    device: Annotated[str | None, typer.Option(help="cuda | cpu; default: auto")] = None,
    scores: Annotated[Path | None, typer.Option(help='file backend: {"name.jpg": score}')] = None,
) -> None:
    """Score roof presence per image, drop images below the threshold, re-assign splits.

    Writes images.json with roof_score on every record and dropped.json for review.
    """
    import logging

    from roofsight.data.config import DataConfig
    from roofsight.data.roof_filter import Scorer, apply_filter, file_scorer, score_all

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = DataConfig.load(config)
    be = backend or cfg.roof_filter.backend
    ds = read_coco(dataset / "images.json")
    unscored = [r for r in ds.images if r.roof_score is None]
    if unscored:
        scorer: Scorer
        if be == "file":
            if scores is None:
                raise typer.BadParameter("--scores is required for the file backend")
            scorer = file_scorer(scores)
        elif be == "sam3":
            from roofsight.labeling.pipeline import load_image
            from roofsight.labeling.sam3 import load_segmenter, roof_fraction

            seg = load_segmenter(sam3_checkpoint, device)
            scorer = lambda p: roof_fraction(seg, load_image(p), cfg.roof_filter.prompt)  # noqa: E731
        else:
            raise typer.BadParameter("backend must be sam3 or file")
        # scores are persisted every 25 images and before any deletion; a restart resumes
        score_all(
            unscored, dataset / "images", scorer, lambda _n: ds.write(dataset / "images.json")
        )
    thr = threshold if threshold is not None else cfg.roof_filter.min_roof_fraction
    frozen: list[int] = []
    if cfg.split.frozen_test and cfg.split.frozen_test.exists():
        frozen = json.loads(cfg.split.frozen_test.read_text())
    filtered, dropped = apply_filter(
        ds, dataset / "images", thr, cfg.split, frozen, delete_files=not dry_run
    )
    (dataset / "dropped.json").write_text(
        json.dumps([r.model_dump(exclude_none=True) for r in dropped], ensure_ascii=False),
        encoding="utf-8",
    )
    if not dry_run:
        filtered.write(dataset / "images.json")
    console.print(
        f"{'would keep' if dry_run else 'kept'} {len(filtered.images)}, "
        f"dropped {len(dropped)} below {thr} ({be})"
    )


@data_app.command("validate")
def data_validate(
    dataset: Annotated[Path, typer.Argument(help="Dataset directory, e.g. datasets/v0.1")],
    annotations: Annotated[str, typer.Option()] = "annotations.json",
) -> None:
    """License, attribution, anonymization, category and file checks. Exits 1 on any error."""
    from roofsight.data.validate import validate

    ds = read_coco(dataset / annotations)
    report = validate(ds, dataset / "images")
    for w in report.warnings:
        err.print(f"[yellow]warning[/] {w}")
    for e in report.errors:
        err.print(f"[red]error[/] {e}")
    console.print(
        f"{report.n_images} images, {report.n_annotations} annotations, "
        f"{len(report.errors)} errors, {len(report.warnings)} warnings"
    )
    if not report.ok:
        raise typer.Exit(1)


@data_app.command("split")
def data_split(dataset: Path, annotations: str = "annotations.json") -> None:
    """Print split sizes."""
    from roofsight.data.split import split_counts

    ds = read_coco(dataset / annotations)
    counts = split_counts({im.id: im.split for im in ds.images if im.split is not None})
    t = Table("split", "images")
    for k, v in counts.items():
        t.add_row(k, str(v))
    console.print(t)


# --------------------------------------------------------------------------------- label


@app.command("label")
def label(
    prompts: Annotated[Path, typer.Option("--prompts", exists=True)],
    in_dir: Annotated[Path, typer.Option("--in", exists=True, file_okay=False)],
    out_dir: Annotated[Path, typer.Option("--out")],
    checkpoint: Annotated[
        str, typer.Option("--checkpoint", help="facebook/sam3, a local HF dir, or a Meta *.pt")
    ] = "facebook/sam3",
    device: Annotated[str | None, typer.Option(help="cuda | cpu; default: auto")] = None,
) -> None:
    """Auto-label <in>/images.json with SAM 3 → <out>/annotations.json (provenance: auto).

    Progress is saved every 5 images to <out>/annotations.partial.json; rerun to resume.
    """
    from roofsight.labeling.pipeline import label_dataset, read_partial, write_partial
    from roofsight.labeling.prompts import PromptConfig
    from roofsight.labeling.sam3 import load_segmenter

    ds = read_coco(in_dir / "images.json")
    cfg = PromptConfig.load(prompts)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / "annotations.partial.json"
    done = read_partial(partial)
    if done:
        console.print(f"resuming: {len(done)} images already labeled")
    labeled = label_dataset(
        ds,
        in_dir / "images",
        cfg,
        load_segmenter(checkpoint, device),
        done=done,
        on_progress=lambda d: write_partial(d, partial),
    )
    labeled.write(out_dir / "annotations.json")
    partial.unlink(missing_ok=True)
    console.print(f"[green]labeled[/] {len(labeled.annotations)} instances → {out_dir}")


@app.command("review")
def review(
    dataset: Path,
    action: Annotated[str, typer.Argument(help="export | import | stats")],
    name: str = "roofsight-review",
) -> None:
    """FiftyOne round trip. export → open the app; import → merge edits back with provenance."""
    from roofsight.labeling.review import (
        export_to_fiftyone,
        import_from_fiftyone,
        merge_provenance,
        review_stats,
    )

    if action == "export":
        export_to_fiftyone(read_coco(dataset / "annotations.json"), dataset / "images", name)
        console.print(f"exported as FiftyOne dataset [bold]{name}[/]; run `fiftyone app launch`")
    elif action == "import":
        auto = read_coco(dataset / "annotations.json")
        edited = read_coco(import_from_fiftyone(name, dataset / "reviewed.raw.json"))
        merged = merge_provenance(auto, edited)
        merged.write(dataset / "annotations.json")
        console.print(review_stats(merged))
    elif action == "stats":
        console.print(review_stats(read_coco(dataset / "annotations.json")))
    else:
        raise typer.BadParameter("action must be export, import or stats")


# --------------------------------------------------------------------------------- train


@app.command("train")
def train_cmd(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    run_id: str | None = None,
) -> None:
    """Fine-tune RF-DETR-Seg. Writes runs/<id>/run.json and checkpoints."""
    from roofsight.train.config import TrainConfig
    from roofsight.train.trainer import train

    run_dir = train(TrainConfig.load(config), run_id)
    console.print(f"[green]trained[/] → {run_dir}")


@app.command("predict")
def predict_cmd(
    run: Annotated[
        Path | None, typer.Option("--run", help="Run dir; uses its best checkpoint")
    ] = None,
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint", exists=True)] = None,
    model: Annotated[
        str, typer.Option(help="rfdetr-seg-nano | rfdetr-seg-small")
    ] = "rfdetr-seg-small",
    split: Annotated[str | None, typer.Option("--split", help="Predict a whole split")] = None,
    dataset: Annotated[Path | None, typer.Option(help="Dataset dir; default from run.json")] = None,
    image: Annotated[list[Path] | None, typer.Option("--image", exists=True)] = None,
    out: Annotated[Path | None, typer.Option("--out")] = None,
    threshold: float = 0.3,
) -> None:
    """Run a checkpoint on images or on a split. Writes COCO results JSON (RLE masks)."""
    from roofsight.eval.predictions import write_predictions
    from roofsight.train.predict import load_model, predict_images, split_paths

    if run is not None:
        from roofsight.run import read_run

        rec = read_run(run)
        model = str(rec.config.get("model", model))
        checkpoint = checkpoint or run / "checkpoints" / "checkpoint_best_total.pth"
        dataset = (
            dataset or Path(str(rec.config.get("dataset_root", "datasets"))) / rec.dataset_version
        )
    if checkpoint is None:
        raise typer.BadParameter("--checkpoint or --run is required")
    net = load_model(checkpoint, model)

    if split is not None:
        s = _split(split)
        if dataset is None:
            raise typer.BadParameter("--dataset is required with --split unless --run is given")
        ds = read_coco(dataset / "annotations.json")
        preds = predict_images(net, split_paths(ds, dataset / "images", s), threshold)
        target = out or ((run or dataset) / f"predictions-{s}.json")
    elif image:
        preds = predict_images(net, list(enumerate(image, start=1)), threshold)
        target = out or Path("predictions.json")
    else:
        raise typer.BadParameter("give --split or at least one --image")
    write_predictions(preds, target)
    console.print(f"[green]{len(preds)} instances[/] → {target}")


# ---------------------------------------------------------------------------------- eval


@app.command("eval")
def eval_cmd(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    split: Annotated[str, typer.Option("--split")] = "test",
    dataset: Annotated[Path | None, typer.Option(help="Dataset dir; default from run.json")] = None,
    predictions: Annotated[Path | None, typer.Option(help="COCO results JSON")] = None,
    geometry_truth: Path | None = None,
    geometry_estimates: Path | None = None,
    runs_root: Path = Path("runs"),
) -> None:
    """Compute all metrics for a run on a split. Writes a new runs/<id>-eval with metrics.json."""
    from roofsight.eval.geometry import geometry_mae, read_estimates, read_ground_truth
    from roofsight.eval.metrics import evaluate
    from roofsight.eval.predictions import read_predictions
    from roofsight.run import read_run, start_run

    s = _split(split)
    src = read_run(run)
    ds_dir = dataset or Path(src.config.get("dataset_root", "datasets")) / src.dataset_version
    ds = read_coco(ds_dir / "annotations.json").subset(s)
    pred_path = predictions or run / f"predictions-{s}.json"
    preds = read_predictions(pred_path)
    report = evaluate(ds, preds)
    if geometry_truth and geometry_estimates:
        g = geometry_mae(read_ground_truth(geometry_truth), read_estimates(geometry_estimates))
        report.extra.update({k: float(v) for k, v in g.as_dict().items()})

    run_dir, _ = start_run(
        "eval",
        {
            "model": src.config.get("model", "unknown"),
            "params_m": src.config.get("params_m"),
            "source_run": src.run_id,
            "split": s,
            "predictions": str(pred_path),
        },
        src.dataset_version,
        src.seed,
        runs_root,
    )
    (run_dir / "metrics.json").write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    lat = run / "latency.json"
    if lat.exists():
        (run_dir / "latency.json").write_text(lat.read_text(encoding="utf-8"), encoding="utf-8")
    t = Table("metric", "value")
    for k, v in report.as_dict().items():
        if isinstance(v, float):
            t.add_row(k, f"{v:.4f}")
    console.print(t)
    console.print(f"[green]wrote[/] {run_dir / 'metrics.json'}")


@app.command("leaderboard")
def leaderboard_cmd(
    runs_root: Path = Path("runs"),
    md_out: Path = Path("docs/leaderboard.md"),
    tex_out: Path = Path("paper/results.tex"),
) -> None:
    """Regenerate docs/leaderboard.md and paper/results.tex from runs/."""
    from roofsight.eval.leaderboard import regenerate

    rows = regenerate(runs_root, md_out, tex_out)
    console.print(f"[green]{len(rows)} rows[/] → {md_out}, {tex_out}")


# -------------------------------------------------------------------------------- export


@export_app.command("coreml")
def export_coreml_cmd(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    out: Path | None = None,
) -> None:
    """PyTorch → ONNX (opset 17, 640×640) → Core ML ML Program (fp16). Then run `export verify`."""
    from roofsight.export.coreml_export import export_coreml
    from roofsight.export.onnx_export import export_onnx, onnx_sha256
    from roofsight.run import read_run
    from roofsight.train.config import TrainConfig
    from roofsight.train.trainer import build_model

    rec = read_run(run)
    cfg = TrainConfig.model_validate(rec.config)
    model = build_model(cfg)
    export_dir = out or run / "export"
    onnx_path = export_onnx(model, export_dir, cfg.resolution)
    mlpackage = export_coreml(onnx_path, export_dir / "RoofSight.mlpackage", cfg.resolution)
    (export_dir / "SHA256SUMS").write_text(
        f"{onnx_sha256(onnx_path)}  {onnx_path.name}\n", encoding="utf-8"
    )
    console.print(f"[green]exported[/] {onnx_path.name}, {mlpackage.name} → {export_dir}")


@export_app.command("verify")
def export_verify_cmd(
    reference: Annotated[Path, typer.Argument(help="Per-image instances JSON from PyTorch")],
    candidate: Annotated[Path, typer.Argument(help="Per-image instances JSON from Core ML/ONNX")],
    out: Path | None = None,
) -> None:
    """Compare two backends on the verify split. Passes at mask IoU ≥ 0.98 and box IoU ≥ 0.98."""
    from roofsight.export.io import read_instances
    from roofsight.export.verify import compare

    result = compare(read_instances(reference), read_instances(candidate))
    if out:
        out.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    console.print(
        f"mask IoU {result.mean_mask_iou:.4f}, box IoU {result.mean_box_iou:.4f}, "
        f"{result.n_matched}/{result.n_reference} matched on {result.n_images} images"
    )
    if not result.ok:
        err.print("[red]verification failed[/]")
        raise typer.Exit(1)
    console.print("[green]verified[/]")


if __name__ == "__main__":
    app()
