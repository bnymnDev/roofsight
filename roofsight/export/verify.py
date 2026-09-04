"""Verification of an exported model against the PyTorch reference on the ``verify`` split.

The comparison itself is pure NumPy so it can be tested without torch or coremltools:
pairs of instance outputs from two backends, matched by best mask IoU per reference instance.
A Core ML model is "done" only when mean mask IoU ≥ 0.98 and mean box IoU ≥ 0.98.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from roofsight.masks import BoolMask, box_iou, iou, mask_bbox

MASK_IOU_MIN = 0.98
BOX_IOU_MIN = 0.98


@dataclass(frozen=True, slots=True)
class Instance:
    category_id: int
    score: float
    mask: BoolMask

    @property
    def bbox(self) -> list[float]:
        return mask_bbox(self.mask)


@dataclass(slots=True)
class VerifyResult:
    mean_mask_iou: float
    mean_box_iou: float
    n_reference: int
    n_matched: int
    n_images: int
    mask_iou_min: float = MASK_IOU_MIN
    box_iou_min: float = BOX_IOU_MIN

    @property
    def ok(self) -> bool:
        return (
            self.n_reference > 0
            and self.n_matched == self.n_reference
            and self.mean_mask_iou >= self.mask_iou_min
            and self.mean_box_iou >= self.box_iou_min
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "mean_mask_iou": self.mean_mask_iou,
            "mean_box_iou": self.mean_box_iou,
            "n_reference": self.n_reference,
            "n_matched": self.n_matched,
            "n_images": self.n_images,
            "thresholds": {"mask_iou": self.mask_iou_min, "box_iou": self.box_iou_min},
            "ok": self.ok,
        }


def compare(
    reference: Sequence[Sequence[Instance]],
    candidate: Sequence[Sequence[Instance]],
    score_threshold: float = 0.5,
) -> VerifyResult:
    """Compare per-image instance lists. Each reference instance above the threshold is matched
    to the candidate instance of the same category with the highest mask IoU."""
    if len(reference) != len(candidate):
        raise ValueError("reference and candidate must cover the same images")
    mask_ious: list[float] = []
    box_ious: list[float] = []
    n_ref = 0
    for ref_list, cand_list in zip(reference, candidate, strict=True):
        for r in ref_list:
            if r.score < score_threshold:
                continue
            n_ref += 1
            best: Instance | None = None
            best_iou = -1.0
            for c in cand_list:
                if c.category_id != r.category_id:
                    continue
                v = iou(r.mask, c.mask)
                if v > best_iou:
                    best, best_iou = c, v
            if best is None:
                continue
            mask_ious.append(best_iou)
            box_ious.append(box_iou(r.bbox, best.bbox))
    return VerifyResult(
        mean_mask_iou=float(np.mean(mask_ious)) if mask_ious else 0.0,
        mean_box_iou=float(np.mean(box_ious)) if box_ious else 0.0,
        n_reference=n_ref,
        n_matched=len(mask_ious),
        n_images=len(reference),
    )
