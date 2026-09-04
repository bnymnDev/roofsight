"""Golden tests for every number ``roofsight eval`` reports."""

from __future__ import annotations

import numpy as np

from roofsight.coco import CocoDataset
from roofsight.eval.metrics import (
    boundary_f_score,
    coco_mask_ap,
    evaluate,
    small_obstacle_recall,
)
from roofsight.eval.predictions import Prediction
from roofsight.masks import mask_to_rle, segmentation_to_mask


def perfect(ds: CocoDataset, shift: int = 0, score: float = 0.9) -> list[Prediction]:
    preds = []
    for a in ds.annotations:
        m = segmentation_to_mask(a.segmentation, 64, 64)
        if shift:
            m = np.roll(m, shift, axis=1)
        preds.append(
            Prediction(
                image_id=a.image_id,
                category_id=a.category_id,
                segmentation=mask_to_rle(m),
                score=score,
            )
        )
    return preds


def test_perfect_predictions(dataset: CocoDataset) -> None:
    r = evaluate(dataset, perfect(dataset))
    assert r.mask_ap > 0.99 and r.mask_ap50 > 0.99
    assert r.small_obstacle_recall == 1.0
    assert r.n_small_obstacles == 2
    assert r.small_obstacle_recall_per_category == {"chimney": 1.0, "skylight": 1.0}
    assert r.boundary_f == 1.0
    assert set(r.per_category_ap) == {"roof_plane", "chimney", "dormer", "skylight", "roof_edge"}
    assert r.n_images == 2


def test_no_predictions(dataset: CocoDataset) -> None:
    r = evaluate(dataset, [])
    assert r.mask_ap == 0.0
    assert r.small_obstacle_recall == 0.0
    assert r.boundary_f == 0.0
    assert r.per_category_ap["chimney"] == 0.0


def test_shifted_predictions_lose_small_objects(dataset: CocoDataset) -> None:
    # 4 px shift: the 6 px wide chimney/skylight drop below IoU 0.5, the big planes do not
    preds = perfect(dataset, shift=4)
    recall, _, n = small_obstacle_recall(dataset, preds)
    assert n == 2 and recall == 0.0
    ap, _, per_ap, _ = coco_mask_ap(dataset, preds)
    assert 0.0 < ap < 0.99
    assert per_ap["roof_plane"] > per_ap["chimney"]
    f, p, rr = boundary_f_score(dataset, preds)
    assert 0.0 < f < 1.0 and abs(p - rr) < 0.2


def test_boundary_tolerance(dataset: CocoDataset) -> None:
    # 1 px shift stays inside the 1 px tolerance band → F = 1
    f, _, _ = boundary_f_score(dataset, perfect(dataset, shift=1))
    assert f == 1.0
    f2, _, _ = boundary_f_score(dataset, perfect(dataset, shift=3))
    assert f2 < 1.0


def test_small_recall_ignores_large_and_non_obstacles(dataset: CocoDataset) -> None:
    preds = [p for p in perfect(dataset) if p.category_id in (2, 4)]
    recall, per, n = small_obstacle_recall(dataset, preds)
    assert recall == 1.0 and n == 2
    preds = [p for p in perfect(dataset) if p.category_id == 2]
    recall, per, n = small_obstacle_recall(dataset, preds)
    assert recall == 0.5 and per == {"chimney": 1.0, "skylight": 0.0}
