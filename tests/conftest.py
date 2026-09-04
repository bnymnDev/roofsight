"""Tiny synthetic fixtures: 64×64 images with rectangular roofs and obstacles. CPU only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from roofsight.categories import coco_categories
from roofsight.coco import (
    Annotation,
    CocoDataset,
    CocoInfo,
    CocoLicense,
    ImageRecord,
    polygon_area,
    polygon_bbox,
)

H = W = 64


def rect(x0: float, y0: float, x1: float, y1: float) -> list[float]:
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def ann(aid: int, image_id: int, cat: int, poly: list[float], **kw: object) -> Annotation:
    return Annotation(
        id=aid,
        image_id=image_id,
        category_id=cat,
        segmentation=[poly],
        area=polygon_area(poly),
        bbox=polygon_bbox(poly),
        **kw,  # type: ignore[arg-type]
    )


def image(iid: int, **kw: object) -> ImageRecord:
    base: dict[str, object] = {
        "id": iid,
        "file_name": f"img_{iid:03d}.jpg",
        "width": W,
        "height": H,
        "source": "mapillary",
        "license": "CC-BY-SA-4.0",
        "attribution": "© tester, Mapillary, CC BY-SA 4.0",
        "split": "test",
        "region": "DE-NRW",
        "anonymized": True,
    }
    base.update(kw)
    return ImageRecord.model_validate(base)


def make_dataset() -> CocoDataset:
    """Two images. Image 1: two roof planes + a small chimney + a ridge. Image 2: one plane
    + a skylight (small) + a large dormer."""
    anns = [
        ann(1, 1, 1, rect(4, 20, 30, 50)),
        ann(2, 1, 1, rect(32, 20, 60, 50)),
        ann(3, 1, 2, rect(10, 12, 16, 20)),  # chimney, 48 px² → small
        ann(4, 1, 10, rect(4, 19, 60, 21), edge_type="ridge"),
        ann(5, 2, 1, rect(8, 24, 56, 56)),
        ann(6, 2, 4, rect(20, 30, 26, 36)),  # skylight, 36 px² → small
        ann(7, 2, 3, rect(24, 22, 60, 58)),  # dormer, 1296 px² → large
    ]
    return CocoDataset(
        info=CocoInfo(version="test", year=2026),
        licenses=[CocoLicense(id=1, name="CC-BY-SA-4.0")],
        images=[image(1), image(2)],
        annotations=anns,
        categories=coco_categories(),
    )


@pytest.fixture
def dataset() -> CocoDataset:
    return make_dataset()


@pytest.fixture
def dataset_dir(tmp_path: Path, dataset: CocoDataset) -> Path:
    d = tmp_path / "v-test"
    (d / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for im in dataset.images:
        arr = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
        Image.fromarray(arr).save(d / "images" / im.file_name)
    dataset.write(d / "annotations.json")
    return d
