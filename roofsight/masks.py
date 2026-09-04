"""Mask utilities shared by labeling, eval and export. NumPy only, no torch."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pycocotools import mask as mask_utils

BoolMask = NDArray[np.bool_]


def polygon_to_mask(poly: Sequence[float], height: int, width: int) -> BoolMask:
    rle = mask_utils.frPyObjects([list(poly)], height, width)
    m = mask_utils.decode(mask_utils.merge(rle))
    return np.asarray(m, dtype=bool)


def segmentation_to_mask(
    segmentation: list[list[float]] | dict[str, object], height: int, width: int
) -> BoolMask:
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        return np.asarray(mask_utils.decode(rle), dtype=bool)
    if not segmentation:
        return np.zeros((height, width), dtype=bool)
    rles = mask_utils.frPyObjects([list(p) for p in segmentation], height, width)
    return np.asarray(mask_utils.decode(mask_utils.merge(rles)), dtype=bool)


def mask_to_rle(mask: BoolMask) -> dict[str, object]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(s) for s in rle["size"]], "counts": counts}


def mask_to_polygons(mask: BoolMask, min_points: int = 3) -> list[list[float]]:
    """Trace the outer contour of each connected component with a marching-squares walk.

    Good enough for COCO export of review-edited masks; not a replacement for OpenCV when
    sub-pixel accuracy matters.
    """
    from scipy import ndimage

    labeled, n = ndimage.label(mask)
    polys: list[list[float]] = []
    for k in range(1, n + 1):
        comp = labeled == k
        ys, xs = np.nonzero(comp)
        if len(xs) < min_points:
            continue
        # convex-ish outline via boundary pixels ordered by angle around the centroid;
        # exact for the tiny fixtures used in tests and a reasonable fallback otherwise.
        eroded = ndimage.binary_erosion(comp)
        by, bx = np.nonzero(comp & ~eroded)
        if len(bx) < min_points:
            by, bx = ys, xs
        cy, cx = by.mean(), bx.mean()
        order = np.argsort(np.arctan2(by - cy, bx - cx))
        poly: list[float] = []
        for i in order:
            poly.extend((float(bx[i]) + 0.5, float(by[i]) + 0.5))
        polys.append(poly)
    return polys


def iou(a: BoolMask, b: BoolMask) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 1.0


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two ``[x, y, w, h]`` boxes."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 1.0


def mask_bbox(mask: BoolMask) -> list[float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0 = float(xs.min()), float(ys.min())
    return [x0, y0, float(xs.max()) - x0 + 1.0, float(ys.max()) - y0 + 1.0]


def boundary(mask: BoolMask, tolerance_px: int = 1) -> BoolMask:
    """Boundary band of a mask: pixels within ``tolerance_px`` of the mask edge."""
    from scipy import ndimage

    edge: BoolMask = np.asarray(mask & ~ndimage.binary_erosion(mask), dtype=bool)
    if tolerance_px <= 0:
        return edge
    struct = ndimage.generate_binary_structure(2, 2)
    dilated: BoolMask = np.asarray(
        ndimage.binary_dilation(edge, struct, iterations=tolerance_px), dtype=bool
    )
    return dilated
