"""Post-processing of SAM 3 candidates.

1. NMS across synonyms of the same category (mask IoU).
2. Minimum area per category.
3. ``roof_plane`` split along ``roof_edge`` polylines where SAM 3 merged adjacent planes.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from roofsight.categories import ROOF_EDGE_ID, ROOF_PLANE_ID
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
    cands: list[Candidate], edge_width_px: int = 3, min_fragment_px: int = 64
) -> list[Candidate]:
    """Cut ``roof_plane`` masks along all ``roof_edge`` masks and keep the connected pieces.

    SAM 3 edge masks rarely reach the plane border, so each edge is extended to the straight
    line fitted through it before cutting. A plane that is not crossed by any edge is returned
    unchanged. Fragments smaller than ``min_fragment_px`` are dropped (edge slivers, not planes).
    """
    edges = [c for c in cands if c.category_id == ROOF_EDGE_ID]
    if not edges:
        return cands
    shape = edges[0].mask.shape
    cutter: BoolMask = np.zeros(shape, dtype=bool)
    for e in edges:
        line = fit_line(e.mask)
        if line is not None:
            cutter |= line_mask(line, shape, edge_width_px)

    out: list[Candidate] = []
    for c in cands:
        if c.category_id != ROOF_PLANE_ID:
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


def postprocess(
    cands: list[Candidate],
    nms_iou: float,
    min_area_px: dict[int, int],
) -> list[Candidate]:
    return split_planes_by_edges(min_area(nms(cands, nms_iou), min_area_px))
