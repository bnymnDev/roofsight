from pathlib import Path

import numpy as np

from roofsight.coco import read_coco
from roofsight.labeling.pipeline import label_dataset
from roofsight.labeling.prompts import PromptConfig
from roofsight.labeling.sam3 import roof_fraction

PROMPTS = Path(__file__).resolve().parents[1] / "configs" / "labeling" / "prompts.yaml"


class StubSegmenter:
    """Returns one box per prompt whose position depends on the prompt hash."""

    def segment(self, image: np.ndarray, prompt: str) -> list[tuple[float, np.ndarray]]:
        h, w = image.shape[:2]
        m = np.zeros((h, w), dtype=bool)
        if prompt == "roof":
            m[10:60, 2:62] = True
            return [(0.9, m)]
        if "ridge" in prompt:
            m[34:36, 4:60] = True
            return [(0.8, m)]
        if prompt == "chimney":
            m[10:20, 10:18] = True
            return [(0.7, m)]
        return []

    def segment_many(
        self, image: np.ndarray, prompts: list[str]
    ) -> list[list[tuple[float, np.ndarray]]]:
        return [self.segment(image, p) for p in prompts]


def test_prompt_config_loads() -> None:
    cfg = PromptConfig.load(PROMPTS)
    assert len(cfg.categories) == 10
    assert ("chimney", 2) in [(c, i) for i, c in [(i, "chimney") for i, _ in cfg.flat()] if i == 2]


def test_label_dataset(dataset_dir: Path) -> None:
    ds = read_coco(dataset_dir / "annotations.json").model_copy(update={"annotations": []})
    cfg = PromptConfig.load(PROMPTS)
    cfg.categories["roof_plane"].min_area_px = 500  # 64×64 fixtures; real value is 2000
    out = label_dataset(ds, dataset_dir / "images", cfg, StubSegmenter())
    cats = sorted({a.category_id for a in out.annotations})
    assert cats == [1, 2, 10]
    assert all(a.provenance == "auto" for a in out.annotations)
    assert all(a.edge_type == "ridge" for a in out.annotations if a.category_id == 10)
    # the ridge split the roof into two planes on each of the two images
    assert sum(a.category_id == 1 for a in out.annotations) == 4
    assert len({a.id for a in out.annotations}) == len(out.annotations)
    out.write(dataset_dir / "auto.json")
    assert read_coco(dataset_dir / "auto.json") == out


def test_roof_fraction() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    assert abs(roof_fraction(StubSegmenter(), img) - (50 * 60) / (64 * 64)) < 1e-9
