"""``configs/train/*.yaml`` as a pydantic model."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class Augmentations(BaseModel):
    """Roofs are seen from below: horizontal flip yes, vertical flip never."""

    color_jitter: float = 0.2
    random_crop_min_roof_fraction: float = 0.6
    horizontal_flip: bool = True
    vertical_flip: Literal[False] = False
    perspective: float = 0.05


class TrainConfig(BaseModel):
    model: Literal["rfdetr-seg-nano", "rfdetr-seg-small"]
    dataset_version: str
    dataset_root: Path = Path("datasets")
    resolution: int = 640
    epochs: int = 50
    batch_size: int = 8
    grad_accum_steps: int = 2
    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    weight_decay: float = 1e-4
    seed: int = 42
    early_stopping: bool = True
    augmentations: Augmentations = Field(default_factory=Augmentations)
    wandb: bool = False
    wandb_project: str = "roofsight"
    runs_root: Path = Path("runs")

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        with path.open(encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))

    @property
    def dataset_dir(self) -> Path:
        return self.dataset_root / self.dataset_version


def rfdetr_class_name(model: str) -> str:
    return {"rfdetr-seg-nano": "RFDETRSegNano", "rfdetr-seg-small": "RFDETRSegSmall"}[model]


PARAMS_M: dict[str, float] = {"rfdetr-seg-nano": 3.0, "rfdetr-seg-small": 14.0}
"""Approximate parameter counts in millions, for the leaderboard's params column."""
