"""``roofsight label``: images.json + prompts → auto-labeled COCO with provenance ``auto``."""

from __future__ import annotations

import json
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
    done: dict[int, list[Annotation]] | None = None,
    on_progress: Callable[[dict[int, list[Annotation]]], None] | None = None,
    every: int = 5,
) -> CocoDataset:
    """Auto-label every image not yet in ``done``.

    ``done`` maps image id → annotations from an earlier, interrupted run; ``on_progress`` is
    called with the growing mapping every ``every`` images so the caller can persist it. A
    SAM 3 pass over hundreds of images takes hours on CPU and must survive a restart.
    """
    thresholds = {
        category_by_name(n).id: cp.score_threshold for n, cp in prompts.categories.items()
    }
    min_area_px = {category_by_name(n).id: cp.min_area_px for n, cp in prompts.categories.items()}
    max_centroid_y = {
        category_by_name(n).id: cp.max_centroid_y
        for n, cp in prompts.categories.items()
        if cp.max_centroid_y is not None
    }
    flat = prompts.flat()
    result: dict[int, list[Annotation]] = dict(done or {})
    next_id = 1 + max((a.id for anns in result.values() for a in anns), default=0)
    todo = [im for im in ds.images if im.id not in result]
    for i, im in enumerate(todo, start=1):
        image = load_image(images_root / im.file_name)
        cands = run_prompts(segmenter, image, flat, thresholds)
        cands = postprocess(cands, prompts.nms_iou, min_area_px, max_centroid_y)
        anns = candidates_to_annotations(cands, im.id, next_id, edge_typer)
        next_id += len(anns)
        result[im.id] = anns
        if on_progress is not None and (i % every == 0 or i == len(todo)):
            on_progress(result)
    annotations = [a for im in ds.images for a in result.get(im.id, [])]
    return ds.model_copy(update={"annotations": annotations})


def read_partial(path: Path) -> dict[int, list[Annotation]]:
    """Read the ``on_progress`` dump written by :func:`write_partial`."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): [Annotation.model_validate(a) for a in v] for k, v in raw.items()}


def write_partial(done: dict[int, list[Annotation]], path: Path) -> None:
    payload = {str(k): [a.model_dump(exclude_none=True) for a in v] for k, v in done.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
