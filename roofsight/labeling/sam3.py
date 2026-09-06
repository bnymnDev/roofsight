"""Adapters around SAM 3. Weights are downloaded by the user (gated on Hugging Face).

Two implementations of the same protocol:

- ``Sam3HfSegmenter``: the ``transformers`` port (``Sam3Model``). Pure PyTorch, runs on CPU,
  loads ``facebook/sam3`` by repo id or from a local directory. Slow without a GPU but the
  only option on a machine without CUDA.
- ``Sam3Segmenter``: Meta's ``sam3`` package. Needs CUDA (it imports ``triton``); faster.

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
"""``(score, mask)`` per instance."""


class TextSegmenter(Protocol):
    """What the pipeline needs from a promptable segmenter. SAM 3 implements it; tests stub it."""

    def segment(self, image: NDArray[np.uint8], prompt: str) -> Instances:
        """Instances for one text prompt."""
        ...

    def segment_many(self, image: NDArray[np.uint8], prompts: list[str]) -> list[Instances]:
        """Instances per prompt. Implementations reuse the image encoding across prompts."""
        ...


class Sam3HfSegmenter:
    """``transformers`` implementation. ``model`` is a Hub id (``facebook/sam3``) or a local dir."""

    def __init__(
        self,
        model: str = "facebook/sam3",
        device: str = "cpu",
        score_threshold: float = 0.3,
        mask_threshold: float = 0.5,
    ) -> None:
        import torch
        from transformers import Sam3Model, Sam3Processor

        self._torch = torch
        self._device = device
        self._processor: Any = Sam3Processor.from_pretrained(model)
        net: Any = Sam3Model.from_pretrained(model)
        self._model: Any = net.to(device).eval()
        self._score_threshold = score_threshold
        self._mask_threshold = mask_threshold

    def segment(self, image: NDArray[np.uint8], prompt: str) -> Instances:
        return self.segment_many(image, [prompt])[0]

    def segment_many(self, image: NDArray[np.uint8], prompts: list[str]) -> list[Instances]:
        pil = Image.fromarray(image)
        h, w = image.shape[:2]
        out: list[Instances] = []
        with self._torch.no_grad():
            pixels = self._processor(images=pil, return_tensors="pt").to(self._device)
            vision = self._model.get_vision_features(pixel_values=pixels["pixel_values"])
            for prompt in prompts:
                text = self._processor(text=prompt, return_tensors="pt").to(self._device)
                result = self._model(
                    vision_embeds=vision,
                    input_ids=text["input_ids"],
                    attention_mask=text.get("attention_mask"),
                )
                post = self._processor.post_process_instance_segmentation(
                    result,
                    threshold=self._score_threshold,
                    mask_threshold=self._mask_threshold,
                    target_sizes=[(h, w)],
                )[0]
                masks = post["masks"].cpu().numpy().astype(bool)
                scores = post["scores"].cpu().numpy().astype(float)
                out.append([(float(s), m) for s, m in zip(scores, masks, strict=True)])
        return out


class Sam3Segmenter:
    """Meta's ``sam3`` package; CUDA only."""

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

    def segment_many(self, image: NDArray[np.uint8], prompts: list[str]) -> list[Instances]:
        state = self._processor.set_image(Image.fromarray(image))
        out: list[Instances] = []
        for prompt in prompts:
            res = self._processor.set_text_prompt(state=state, prompt=prompt)
            masks = np.asarray(res["masks"]).astype(bool)
            scores = np.asarray(res["scores"]).astype(float)
            out.append([(float(s), m) for s, m in zip(scores, masks, strict=True)])
        return out


def load_segmenter(spec: str, device: str | None = None) -> TextSegmenter:
    """``*.pt`` → Meta's package on CUDA; anything else (Hub id or directory) → transformers."""
    if spec.endswith(".pt"):
        return Sam3Segmenter(Path(spec), device or "cuda")
    return Sam3HfSegmenter(spec, device or _default_device())


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


def run_prompts(
    segmenter: TextSegmenter,
    image: NDArray[np.uint8],
    prompts: Iterable[tuple[int, str]],
    score_thresholds: dict[int, float],
) -> list[Candidate]:
    pairs = list(prompts)
    per_prompt = segmenter.segment_many(image, [p for _, p in pairs])
    cands: list[Candidate] = []
    for (category_id, prompt), instances in zip(pairs, per_prompt, strict=True):
        for score, mask in instances:
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
