"""Adapter around the official ``sam3`` package. Weights are downloaded by the user.

Labeling only. Never shipped, never imported outside ``roofsight.labeling``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from roofsight.labeling.candidates import Candidate

Instances = list[tuple[float, NDArray[np.bool_]]]


class TextSegmenter(Protocol):
    """What the pipeline needs from a promptable segmenter. SAM 3 implements it; tests stub it."""

    def segment(self, image: NDArray[np.uint8], prompt: str) -> Instances:
        """Return ``(score, mask)`` per instance for one text prompt."""
        ...


class Sam3Segmenter:
    def __init__(self, checkpoint: Path, device: str = "cuda") -> None:
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as e:  # pragma: no cover - needs the real package
            raise RuntimeError(
                "sam3 is not installed. Install the official package and download the "
                "checkpoint; see docs/labeling.md"
            ) from e
        self._model: Any = build_sam3_image_model(checkpoint_path=str(checkpoint), device=device)
        self._processor: Any = Sam3Processor(self._model)

    def segment(self, image: NDArray[np.uint8], prompt: str) -> Instances:
        state = self._processor.set_image(Image.fromarray(image))
        out = self._processor.set_text_prompt(state=state, prompt=prompt)
        masks = np.asarray(out["masks"]).astype(bool)
        scores = np.asarray(out["scores"]).astype(float)
        return [(float(s), m) for s, m in zip(scores, masks, strict=True)]


def run_prompts(
    segmenter: TextSegmenter,
    image: NDArray[np.uint8],
    prompts: Iterable[tuple[int, str]],
    score_thresholds: dict[int, float],
) -> list[Candidate]:
    cands: list[Candidate] = []
    for category_id, prompt in prompts:
        for score, mask in segmenter.segment(image, prompt):
            if score < score_thresholds.get(category_id, 0.0):
                continue
            cands.append(Candidate(category_id, prompt, score, mask))
    return cands


def roof_fraction(
    segmenter: TextSegmenter, image: NDArray[np.uint8], prompt: str = "roof"
) -> float:
    """Roof presence filter: fraction of the image covered by the union of all roof masks."""
    union = np.zeros(image.shape[:2], dtype=bool)
    for _, mask in segmenter.segment(image, prompt):
        union |= mask
    return float(union.mean())
