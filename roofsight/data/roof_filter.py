"""Roof presence filter: drop images without a visible roof before labeling.

Two backends. ``sam3`` (the SPEC default) prompts SAM 3 with "roof" and keeps images where the
mask covers ≥ 5 %; it needs a GPU and the checkpoint. ``file`` takes one score per image from a
JSON file, so a SAM 3 run elsewhere or reviewer decisions from FiftyOne plug in the same way.
Both produce one number per image; the threshold logic is shared.

Tried and rejected as CPU stand-ins (see docs/decisions.md): CLIP zero-shot, SegFormer-B0
ADE20K building fraction, Grounding DINO tiny. None separated roof from no-roof on a 16-image
check of real Mapillary frames.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path

from roofsight.coco import CocoDataset, ImageRecord
from roofsight.data.config import SplitConfig
from roofsight.data.split import assign_splits

log = logging.getLogger(__name__)

Scorer = Callable[[Path], float]
"""Returns a roof score in [0, 1] for an image path."""


def file_scorer(path: Path) -> Scorer:
    """Scores from a JSON file ``{"<file_name>": score}``: a SAM 3 run on another machine, or
    reviewer decisions (1 = roof, 0 = no roof) exported from FiftyOne."""
    import json

    table: dict[str, float] = json.loads(path.read_text(encoding="utf-8"))

    def score(image: Path) -> float:
        try:
            return float(table[image.name])
        except KeyError as e:
            raise KeyError(f"no score for {image.name} in {path}") from e

    return score


def score_all(records: Iterable[ImageRecord], images_root: Path, scorer: Scorer) -> None:
    """Fill ``roof_score`` on every record in place."""
    for i, r in enumerate(records, start=1):
        r.roof_score = round(scorer(images_root / r.file_name), 4)
        if i % 100 == 0:
            log.info("roof filter: scored %d images", i)


def apply_filter(
    ds: CocoDataset,
    images_root: Path,
    threshold: float,
    split: SplitConfig,
    frozen_test: Iterable[int] = (),
    delete_files: bool = True,
) -> tuple[CocoDataset, list[ImageRecord]]:
    """Drop records below ``threshold`` (scores must already be set), re-assign splits.

    Image ids are kept so an already frozen test set stays valid. Returns the filtered dataset
    and the dropped records."""
    kept: list[ImageRecord] = []
    dropped: list[ImageRecord] = []
    for r in ds.images:
        if r.roof_score is None:
            raise ValueError(f"image {r.id} has no roof_score; run the scorer first")
        (kept if r.roof_score >= threshold else dropped).append(r)
    if delete_files:
        for r in dropped:
            (images_root / r.file_name).unlink(missing_ok=True)
    ids = {r.id for r in kept}
    assignment = assign_splits(((r.id, r.region, 0) for r in kept), split, frozen_test)
    for r in kept:
        r.split = assignment[r.id]
    out = ds.model_copy(
        update={"images": kept, "annotations": [a for a in ds.annotations if a.image_id in ids]}
    )
    return out, dropped
