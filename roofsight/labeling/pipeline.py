"""``roofsight label``: images.json + prompts → auto-labeled COCO with provenance ``auto``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from roofsight.categories import ROOF_EDGE_ID, category_by_name
from roofsight.coco import Annotation, CocoDataset
from roofsight.labeling.candidates import Candidate
from roofsight.labeling.postprocess import postprocess
from roofsight.labeling.prompts import PromptConfig
from roofsight.labeling.sam3 import TextSegmenter, run_prompts
from roofsight.masks import mask_bbox, mask_to_rle

EdgeTyper = Callable[[Candidate], str]


def default_edge_typer(c: Candidate) -> str:
    """Prompt text decides the edge type: ``"ridge line"`` → ``ridge``. Falls back to ``eave``."""
    for t in ("ridge", "hip", "valley", "verge", "eave"):
        if t in c.prompt.lower():
            return t
    return "eave"


def load_image(path: Path) -> NDArray[np.uint8]:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def candidates_to_annotations(
    cands: list[Candidate], image_id: int, next_id: int, edge_typer: EdgeTyper
) -> list[Annotation]:
    anns: list[Annotation] = []
    for c in cands:
        area = float(c.area)
        if area <= 0:
            continue
        anns.append(
            Annotation(
                id=next_id,
                image_id=image_id,
                category_id=c.category_id,
                segmentation=mask_to_rle(c.mask),
                area=area,
                bbox=mask_bbox(c.mask),
                provenance="auto",
                score=round(c.score, 4),
                edge_type=edge_typer(c) if c.category_id == ROOF_EDGE_ID else None,  # type: ignore[arg-type]
            )
        )
        next_id += 1
    return anns


def label_dataset(
    ds: CocoDataset,
    images_root: Path,
    prompts: PromptConfig,
    segmenter: TextSegmenter,
    edge_typer: EdgeTyper = default_edge_typer,
) -> CocoDataset:
    thresholds = {
        category_by_name(n).id: cp.score_threshold for n, cp in prompts.categories.items()
    }
    min_area_px = {category_by_name(n).id: cp.min_area_px for n, cp in prompts.categories.items()}
    flat = prompts.flat()
    annotations: list[Annotation] = []
    next_id = 1
    for im in ds.images:
        image = load_image(images_root / im.file_name)
        cands = run_prompts(segmenter, image, flat, thresholds)
        cands = postprocess(cands, prompts.nms_iou, min_area_px)
        anns = candidates_to_annotations(cands, im.id, next_id, edge_typer)
        next_id += len(anns)
        annotations.extend(anns)
    return ds.model_copy(update={"annotations": annotations})
