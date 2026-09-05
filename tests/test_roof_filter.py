from pathlib import Path

import pytest

from roofsight.coco import read_coco
from roofsight.data.config import SplitConfig
from roofsight.data.roof_filter import apply_filter, score_all


def test_score_and_filter(dataset_dir: Path) -> None:
    ds = read_coco(dataset_dir / "annotations.json")
    score_all(ds.images, dataset_dir / "images", lambda p: 0.9 if p.name.endswith("1.jpg") else 0.1)
    assert [r.roof_score for r in ds.images] == [0.9, 0.1]
    out, dropped = apply_filter(ds, dataset_dir / "images", 0.5, SplitConfig(verify_count=0))
    assert [r.id for r in out.images] == [1]
    assert [r.id for r in dropped] == [2]
    assert not (dataset_dir / "images" / "img_002.jpg").exists()
    assert (dataset_dir / "images" / "img_001.jpg").exists()
    assert all(a.image_id == 1 for a in out.annotations)
    assert out.images[0].split is not None


def test_filter_requires_scores(dataset_dir: Path) -> None:
    ds = read_coco(dataset_dir / "annotations.json")
    with pytest.raises(ValueError, match="roof_score"):
        apply_filter(ds, dataset_dir / "images", 0.5, SplitConfig())
