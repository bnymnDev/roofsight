"""The numbers ``roofsight eval`` reports.

- Mask AP / AP50 per category and overall (pycocotools, segm)
- Small-obstacle recall @ IoU 0.5 for obstacles < 32² px: the number that matters for PV
- Roof-plane boundary F-score with 1 px tolerance at 640 px
- Geometry: pitch / azimuth MAE on the own-photo subset (see ``geometry.py``)
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from roofsight.categories import CATEGORIES, OBSTACLE_IDS, ROOF_PLANE_ID, category_by_id
from roofsight.coco import CocoDataset
from roofsight.eval.predictions import Prediction
from roofsight.masks import BoolMask, boundary, iou, segmentation_to_mask

SMALL_AREA_PX = 32 * 32
BOUNDARY_REF_SIZE = 640


@dataclass(slots=True)
class MetricReport:
    mask_ap: float
    mask_ap50: float
    per_category_ap: dict[str, float]
    per_category_ap50: dict[str, float]
    small_obstacle_recall: float
    small_obstacle_recall_per_category: dict[str, float]
    n_small_obstacles: int
    boundary_f: float
    boundary_precision: float
    boundary_recall: float
    n_images: int
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "mask_ap": self.mask_ap,
            "mask_ap50": self.mask_ap50,
            "per_category_ap": self.per_category_ap,
            "per_category_ap50": self.per_category_ap50,
            "small_obstacle_recall": self.small_obstacle_recall,
            "small_obstacle_recall_per_category": self.small_obstacle_recall_per_category,
            "n_small_obstacles": self.n_small_obstacles,
            "boundary_f": self.boundary_f,
            "boundary_precision": self.boundary_precision,
            "boundary_recall": self.boundary_recall,
            "n_images": self.n_images,
            **self.extra,
        }


def _coco_gt(ds: CocoDataset) -> COCO:
    gt = COCO()
    gt.dataset = ds.model_dump(exclude_none=True)
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def coco_mask_ap(
    ds: CocoDataset, preds: list[Prediction]
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    """Overall and per-category segm AP / AP50. Returns 0 for categories without ground truth."""
    if not ds.images:
        return 0.0, 0.0, {}, {}
    gt = _coco_gt(ds)
    empty = not preds
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes([p.model_dump(exclude_none=True) for p in preds]) if not empty else None
    per_ap: dict[str, float] = {}
    per_ap50: dict[str, float] = {}
    present = {a.category_id for a in ds.annotations}
    if dt is None:
        for cid in present:
            per_ap[category_by_id(cid).name] = 0.0
            per_ap50[category_by_id(cid).name] = 0.0
        return 0.0, 0.0, per_ap, per_ap50

    def run(cat_ids: list[int]) -> tuple[float, float]:
        ev = COCOeval(gt, dt, iouType="segm")
        ev.params.catIds = cat_ids
        ev.params.imgIds = [im.id for im in ds.images]
        with contextlib.redirect_stdout(io.StringIO()):
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        ap, ap50 = float(ev.stats[0]), float(ev.stats[1])
        return max(ap, 0.0), max(ap50, 0.0)

    for c in CATEGORIES:
        if c.id not in present:
            continue
        per_ap[c.name], per_ap50[c.name] = run([c.id])
    ap, ap50 = run(sorted(present))
    return ap, ap50, per_ap, per_ap50


def small_obstacle_recall(
    ds: CocoDataset, preds: list[Prediction], iou_threshold: float = 0.5
) -> tuple[float, dict[str, float], int]:
    """Recall of obstacle instances with area < 32² px at mask IoU ≥ 0.5, any score.

    Each ground-truth instance counts as found if any prediction of the same category on the
    same image overlaps it with IoU ≥ threshold. Predictions are not consumed, so two small
    GTs covered by one large prediction both count: the question is "would the planner have
    seen it", not AP.
    """
    by_image: dict[int, list[Prediction]] = {}
    for p in preds:
        by_image.setdefault(p.image_id, []).append(p)
    sizes = {im.id: (im.height, im.width) for im in ds.images}
    found: dict[int, int] = {}
    total: dict[int, int] = {}
    for a in ds.annotations:
        if a.category_id not in OBSTACLE_IDS or a.area >= SMALL_AREA_PX:
            continue
        h, w = sizes[a.image_id]
        gt_mask = segmentation_to_mask(a.segmentation, h, w)
        total[a.category_id] = total.get(a.category_id, 0) + 1
        hit = False
        for p in by_image.get(a.image_id, []):
            if p.category_id != a.category_id:
                continue
            if iou(gt_mask, segmentation_to_mask(p.segmentation, h, w)) >= iou_threshold:
                hit = True
                break
        if hit:
            found[a.category_id] = found.get(a.category_id, 0) + 1
    n = sum(total.values())
    per = {category_by_id(c).name: found.get(c, 0) / t for c, t in total.items()}
    overall = sum(found.values()) / n if n else 0.0
    return overall, per, n


def _scaled_tolerance(height: int, width: int, tolerance_px: int = 1) -> int:
    return max(1, round(tolerance_px * max(height, width) / BOUNDARY_REF_SIZE))


def boundary_f_score(
    ds: CocoDataset, preds: list[Prediction], tolerance_px: int = 1, score_threshold: float = 0.5
) -> tuple[float, float, float]:
    """Boundary precision / recall / F of the union of roof-plane masks per image.

    Tolerance is 1 px at 640 px and scales with image size. Images without roof planes in GT
    and prediction are skipped.
    """
    by_image: dict[int, list[Prediction]] = {}
    for p in preds:
        if p.category_id == ROOF_PLANE_ID and p.score >= score_threshold:
            by_image.setdefault(p.image_id, []).append(p)
    tp = fp = fn = 0
    for im in ds.images:
        h, w = im.height, im.width
        gt_union: BoolMask = np.zeros((h, w), dtype=bool)
        for a in ds.annotations:
            if a.image_id == im.id and a.category_id == ROOF_PLANE_ID:
                gt_union |= segmentation_to_mask(a.segmentation, h, w)
        pr_union: BoolMask = np.zeros((h, w), dtype=bool)
        for p in by_image.get(im.id, []):
            pr_union |= segmentation_to_mask(p.segmentation, h, w)
        if not gt_union.any() and not pr_union.any():
            continue
        tol = _scaled_tolerance(h, w, tolerance_px)
        gt_edge, pr_edge = boundary(gt_union, 0), boundary(pr_union, 0)
        gt_band, pr_band = boundary(gt_union, tol), boundary(pr_union, tol)
        tp += int((pr_edge & gt_band).sum())
        fp += int((pr_edge & ~gt_band).sum())
        fn += int((gt_edge & ~pr_band).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f, precision, recall


def evaluate(ds: CocoDataset, preds: Iterable[Prediction]) -> MetricReport:
    pl = list(preds)
    ap, ap50, per_ap, per_ap50 = coco_mask_ap(ds, pl)
    sor, sor_per, n_small = small_obstacle_recall(ds, pl)
    bf, bp, br = boundary_f_score(ds, pl)
    return MetricReport(
        mask_ap=ap,
        mask_ap50=ap50,
        per_category_ap=per_ap,
        per_category_ap50=per_ap50,
        small_obstacle_recall=sor,
        small_obstacle_recall_per_category=sor_per,
        n_small_obstacles=n_small,
        boundary_f=bf,
        boundary_precision=bp,
        boundary_recall=br,
        n_images=len(ds.images),
    )
