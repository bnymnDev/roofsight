from dataclasses import dataclass

import numpy as np

from roofsight.train.predict import detections_to_predictions, split_paths
from tests.conftest import make_dataset


@dataclass
class Det:
    mask: np.ndarray | None
    class_id: np.ndarray | None
    confidence: np.ndarray | None


def test_detections_to_predictions() -> None:
    masks = np.zeros((3, 64, 64), dtype=bool)
    masks[0, 10:20, 10:20] = True
    masks[1, 30:40, 30:40] = True  # unknown class → dropped
    det = Det(masks, np.array([2, 99, 4]), np.array([0.9, 0.5, 1.2]))
    preds = detections_to_predictions(det, image_id=7)
    assert [p.category_id for p in preds] == [2]  # third mask is empty
    assert preds[0].image_id == 7 and preds[0].bbox == [10.0, 10.0, 10.0, 10.0]
    assert preds[0].segmentation["size"] == [64, 64]
    assert detections_to_predictions(Det(None, None, None), 1) == []


def test_split_paths(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ds = make_dataset()
    paths = split_paths(ds, tmp_path, "test")
    assert [p.name for _, p in paths] == ["img_001.jpg", "img_002.jpg"]
    assert split_paths(ds, tmp_path, "train") == []
