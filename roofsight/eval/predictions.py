"""Prediction format: COCO results JSON, one entry per instance, RLE segmentation."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter


class Prediction(BaseModel):
    image_id: int
    category_id: int
    segmentation: dict[str, object]
    score: float = Field(ge=0, le=1)
    bbox: list[float] | None = None


_adapter = TypeAdapter(list[Prediction])


def read_predictions(path: Path) -> list[Prediction]:
    return _adapter.validate_json(path.read_text(encoding="utf-8"))


def write_predictions(preds: list[Prediction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([p.model_dump(exclude_none=True) for p in preds], ensure_ascii=False),
        encoding="utf-8",
    )
