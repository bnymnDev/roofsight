"""``roofsight train``: fine-tune RF-DETR-Seg from the Roboflow checkpoint, write run.json."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from roofsight.coco import read_coco
from roofsight.run import start_run
from roofsight.train.config import PARAMS_M, TrainConfig, rfdetr_class_name
from roofsight.train.dataset import prepare_rfdetr_layout


def build_model(config: TrainConfig) -> Any:
    import rfdetr

    cls = getattr(rfdetr, rfdetr_class_name(config.model))
    return cls(resolution=config.resolution)


def train(config: TrainConfig, run_id: str | None = None) -> Path:
    run_dir, record = start_run(
        "train",
        {**config.model_dump(mode="json"), "params_m": PARAMS_M[config.model]},
        config.dataset_version,
        config.seed,
        config.runs_root,
        run_id,
    )
    ds = read_coco(config.dataset_dir / "annotations.json")
    layout = prepare_rfdetr_layout(ds, config.dataset_dir / "images", run_dir / "data")

    model = build_model(config)
    model.train(
        dataset_dir=str(layout),
        epochs=config.epochs,
        batch_size=config.batch_size,
        grad_accum_steps=config.grad_accum_steps,
        lr=config.lr,
        lr_encoder=config.lr_encoder,
        weight_decay=config.weight_decay,
        output_dir=str(run_dir / "checkpoints"),
        seed=config.seed,
        early_stopping=config.early_stopping,
        wandb=config.wandb,
        project=config.wandb_project,
        run=record.run_id,
    )
    return run_dir
