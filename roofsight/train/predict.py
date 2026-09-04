"""``roofsight predict``: run a checkpoint on images or a split, write COCO results JSON.

The conversion from model output to :class:`Prediction` is pure so it is tested without torch.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from roofsight.categories import category_by_id
from roofsight.coco import CocoDataset, Split
from roofsight.eval.predictions import Prediction
from roofsight.masks import mask_bbox, mask_to_rle


class Detections(Protocol):
    """The subset of ``supervision.Detections`` we read."""

    mask: NDArray[np.bool_] | None
    class_id: NDArray[np.int_] | None
    confidence: NDArray[np.floating[Any]] | None


def detections_to_predictions(
    det: Detections, image_id: int, class_offset: int = 0
) -> list[Prediction]:
    """Convert one image's detections. ``class_offset`` maps model class index → category id
    (rfdetr uses the COCO category ids from the training file, so it is 0 by default)."""
    if det.mask is None or det.class_id is None or det.confidence is None:
        return []
    out: list[Prediction] = []
    for mask, cls, score in zip(det.mask, det.class_id, det.confidence, strict=True):
        cid = int(cls) + class_offset
        try:
            category_by_id(cid)
        except KeyError:
            continue
        m = np.asarray(mask, dtype=bool)
        if not m.any():
            continue
        out.append(
            Prediction(
                image_id=image_id,
                category_id=cid,
                segmentation=mask_to_rle(m),
                score=float(min(max(score, 0.0), 1.0)),
                bbox=mask_bbox(m),
            )
        )
    return out


def load_model(checkpoint: Path, model_name: str, resolution: int = 640) -> Any:
    import rfdetr

    from roofsight.train.config import rfdetr_class_name

    cls = getattr(rfdetr, rfdetr_class_name(model_name))
    return cls(pretrain_weights=str(checkpoint), resolution=resolution)


def predict_images(
    model: Any, paths: Iterable[tuple[int, Path]], threshold: float = 0.3
) -> list[Prediction]:
    from PIL import Image

    preds: list[Prediction] = []
    for image_id, path in paths:
        with Image.open(path) as im:
            det = model.predict(im.convert("RGB"), threshold=threshold)
        preds.extend(detections_to_predictions(det, image_id))
    return preds


def split_paths(ds: CocoDataset, images_root: Path, split: Split) -> list[tuple[int, Path]]:
    return [(im.id, images_root / im.file_name) for im in ds.subset(split).images]
