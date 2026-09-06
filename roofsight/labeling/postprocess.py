"""Post-processing of SAM 3 candidates.

1. NMS across synonyms of the same category (mask IoU).
2. Minimum area per category.
3. ``roof_plane`` split along ``roof_edge`` polylines where SAM 3 merged adjacent planes.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from roofsight.categories import OBSTACLE_IDS, ROOF_EDGE_ID, ROOF_PLANE_ID, category_by_name
from roofsight.labeling.candidates import Candidate
from roofsight.masks import BoolMask, iou


def nms(cands: list[Candidate], iou_threshold: float = 0.7) -> list[Candidate]:
    """Per category, keep the highest-scoring mask of each overlapping group."""
    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda c: c.score, reverse=True):
        if any(
            k.category_id == c.category_id and iou(k.mask, c.mask) >= iou_threshold for k in kept
        ):
            continue
        kept.append(c)
    return kept


def min_area(cands: list[Candidate], min_area_px: dict[int, int]) -> list[Candidate]:
    return [c for c in cands if c.area >= min_area_px.get(c.category_id, 0)]


def fit_line(mask: BoolMask) -> tuple[float, float, float, float] | None:
    """Least-squares line through a thin mask: ``(cx, cy, dx, dy)`` with a unit direction."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    cov = np.cov(np.stack([xs - cx, ys - cy]).astype(float))
    _, vecs = np.linalg.eigh(cov)
    dx, dy = (float(v) for v in vecs[:, -1])
    return cx, cy, dx, dy


def line_mask(
    line: tuple[float, float, float, float], shape: tuple[int, int], width_px: int = 3
) -> BoolMask:
    """Rasterize an infinite line as a band ``width_px`` wide across the whole image."""
    h, w = shape
    cx, cy, dx, dy = line
    ys, xs = np.mgrid[0:h, 0:w]
    # signed distance to the line: cross product with the unit direction
    dist = np.abs((xs + 0.5 - cx) * dy - (ys + 0.5 - cy) * dx)
    out: BoolMask = np.asarray(dist <= width_px / 2.0, dtype=bool)
    return out


def split_planes_by_edges(
    cands: list[Candidate],
    edge_width_px: int = 3,
    min_fragment_px: int = 64,
    touch_px: int = 4,
) -> list[Candidate]:
    """Cut each ``roof_plane`` along the ``roof_edge`` lines that touch it; keep the pieces.

    SAM 3 edge masks rarely reach the plane border, so an edge is extended to the straight line
    fitted through it before cutting. Only edges whose mask (dilated by ``touch_px``) overlaps
    the plane cut it; the extended line of a ridge on one house never slices the roof next
    door. A plane not crossed by any edge is returned unchanged. Fragments smaller than
    ``min_fragment_px`` are dropped (edge slivers, not planes).
    """
    edges = [c for c in cands if c.category_id == ROOF_EDGE_ID]
    if not edges:
        return cands
    shape = edges[0].mask.shape
    struct = ndimage.generate_binary_structure(2, 2)
    edge_lines: list[tuple[BoolMask, BoolMask]] = []  # (touch zone, extended line band)
    for e in edges:
        line = fit_line(e.mask)
        if line is None:
            continue
        touch = np.asarray(ndimage.binary_dilation(e.mask, struct, iterations=touch_px), dtype=bool)
        edge_lines.append((touch, line_mask(line, shape, edge_width_px)))

    out: list[Candidate] = []
    for c in cands:
        if c.category_id != ROOF_PLANE_ID:
            out.append(c)
            continue
        cutter: BoolMask = np.zeros(shape, dtype=bool)
        touched = False
        for touch, band in edge_lines:
            if np.any(touch & c.mask):
                cutter |= band
                touched = True
        if not touched:
            out.append(c)
            continue
        cut = c.mask & ~cutter
        labeled, n = ndimage.label(cut)
        pieces = [labeled == k for k in range(1, n + 1)]
        pieces = [p for p in pieces if p.sum() >= min_fragment_px]
        if len(pieces) <= 1:
            out.append(c)
            continue
        for p in pieces:
            # give the cut-away edge band back to the nearest piece so planes stay contiguous
            grown = np.asarray(
                ndimage.binary_dilation(p, iterations=edge_width_px) & c.mask, dtype=bool
            )
            out.append(Candidate(c.category_id, c.prompt, c.score, grown))
    return out


ON_ROOF_IDS: frozenset[int] = OBSTACLE_IDS | {category_by_name("tree_occlusion").id}
DEFAULT_ON_ROOF_OVERLAP: dict[int, float] = {
    **dict.fromkeys(OBSTACLE_IDS, 0.5),
    category_by_name("tree_occlusion").id: 0.2,
}


def keep_on_roof(
    cands: list[Candidate],
    dilate_fraction: float = 0.02,
    min_overlap: dict[int, float] | None = None,
) -> list[Candidate]:
    """Drop obstacles and occlusions that do not overlap the union of roof planes.

    The union is dilated by ``dilate_fraction`` of the longer image side so chimneys and
    antennas that stick out above the ridge still count. Without roof planes in the image
    nothing is on a roof and every such candidate is dropped.
    """
    thresholds = min_overlap or DEFAULT_ON_ROOF_OVERLAP
    planes = [c for c in cands if c.category_id == ROOF_PLANE_ID]
    others = [c for c in cands if c.category_id in ON_ROOF_IDS]
    if not others:
        return cands
    if not planes:
        return [c for c in cands if c.category_id not in ON_ROOF_IDS]
    shape = planes[0].mask.shape
    union: BoolMask = np.zeros(shape, dtype=bool)
    for p in planes:
        union |= p.mask
    radius = max(1, round(dilate_fraction * max(shape)))
    zone = np.asarray(ndimage.binary_dilation(union, iterations=radius), dtype=bool)
    out: list[Candidate] = []
    for c in cands:
        if c.category_id not in ON_ROOF_IDS:
            out.append(c)
            continue
        area = c.area
        if area == 0:
            continue
        overlap = float(np.logical_and(c.mask, zone).sum()) / area
        if overlap >= thresholds.get(c.category_id, 0.5):
            out.append(c)
    return out


def postprocess(
    cands: list[Candidate],
    nms_iou: float,
    min_area_px: dict[int, int],
) -> list[Candidate]:
    kept = min_area(nms(cands, nms_iou), min_area_px)
    kept = split_planes_by_edges(kept, min_fragment_px=max(64, min_area_px.get(ROOF_PLANE_ID, 64)))
    kept = min_area(kept, min_area_px)  # fragments must be planes, not slivers
    return keep_on_roof(kept)
