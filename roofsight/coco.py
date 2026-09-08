"""COCO JSON as the canonical annotation format, with RoofSight's extra fields.

Every image carries ``source``, ``license`` and ``attribution``. Every annotation carries
``provenance``. ``roof_edge`` annotations carry ``edge_type``. Anything else is plain COCO, so
``pycocotools`` and every viewer that reads COCO work unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from roofsight.categories import EDGE_TYPES, ROOF_EDGE_ID, EdgeType, category_by_id

Provenance = Literal["auto", "auto_edited", "manual"]
Source = Literal["mapillary", "own", "commons"]
Split = Literal["train", "val", "test", "verify"]

#: Licenses a record may carry. The collection is CC-BY-SA 4.0; everything here can be
#: redistributed under it, and every record keeps its own license and attribution.
ALLOWED_LICENSES: frozenset[str] = frozenset(
    {
        "CC-BY-SA-4.0",
        "CC-BY-SA-3.0",
        "CC-BY-SA-2.5",
        "CC-BY-SA-2.0",
        "CC-BY-4.0",
        "CC-BY-3.0",
        "CC-BY-2.5",
        "CC-BY-2.0",
        "CC0-1.0",
        "public-domain",
    }
)
SPLITS: tuple[Split, ...] = ("train", "val", "test", "verify")


class CocoLicense(BaseModel):
    id: int
    name: str
    url: str = ""


class CocoInfo(BaseModel):
    description: str = "RoofSight: ground-view roof and rooftop-obstacle segmentation"
    version: str
    year: int
    contributor: str = "RoofSight contributors"
    url: str = "https://github.com/bnymnDev/roofsight"
    date_created: str = ""


class ImageRecord(BaseModel):
    """A COCO image plus the fields our license hygiene rule requires."""

    id: int
    file_name: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    source: Source
    license: str
    attribution: str
    split: Split | None = None
    region: str = ""
    source_id: str = ""
    anonymized: bool = False
    has_pose: bool = False
    phash: str = ""
    roof_score: float | None = None

    @field_validator("license")
    @classmethod
    def _license_allowed(cls, v: str) -> str:
        if v not in ALLOWED_LICENSES:
            raise ValueError(f"license {v!r} not allowed; allowed: {sorted(ALLOWED_LICENSES)}")
        return v

    @field_validator("attribution")
    @classmethod
    def _attribution_present(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("attribution must not be empty")
        return v


class Annotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    segmentation: list[list[float]] | dict[str, object]
    area: float = Field(ge=0)
    bbox: list[float] = Field(min_length=4, max_length=4)
    iscrowd: int = 0
    provenance: Provenance = "auto"
    score: float | None = None
    edge_type: EdgeType | None = None

    @field_validator("category_id")
    @classmethod
    def _known_category(cls, v: int) -> int:
        try:
            category_by_id(v)
        except KeyError as e:
            raise ValueError(str(e)) from e
        return v

    @model_validator(mode="after")
    def _edge_type_rule(self) -> Annotation:
        if self.category_id == ROOF_EDGE_ID and self.edge_type is None:
            raise ValueError("roof_edge annotations require edge_type")
        if self.category_id != ROOF_EDGE_ID and self.edge_type is not None:
            raise ValueError("edge_type is only valid on roof_edge annotations")
        if self.edge_type is not None and self.edge_type not in EDGE_TYPES:
            raise ValueError(f"unknown edge_type {self.edge_type!r}")
        return self


class CocoDataset(BaseModel):
    info: CocoInfo
    licenses: list[CocoLicense] = Field(default_factory=list)
    images: list[ImageRecord] = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)
    categories: list[dict[str, object]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _references(self) -> CocoDataset:
        image_ids = {im.id for im in self.images}
        if len(image_ids) != len(self.images):
            raise ValueError("duplicate image ids")
        ann_ids = {a.id for a in self.annotations}
        if len(ann_ids) != len(self.annotations):
            raise ValueError("duplicate annotation ids")
        for a in self.annotations:
            if a.image_id not in image_ids:
                raise ValueError(f"annotation {a.id} references unknown image {a.image_id}")
        return self

    def annotations_for(self, image_id: int) -> list[Annotation]:
        return [a for a in self.annotations if a.image_id == image_id]

    def subset(self, split: Split) -> CocoDataset:
        images = [im for im in self.images if im.split == split]
        ids = {im.id for im in images}
        return CocoDataset(
            info=self.info,
            licenses=self.licenses,
            images=images,
            annotations=[a for a in self.annotations if a.image_id in ids],
            categories=self.categories,
        )

    def to_json(self) -> str:
        return json.dumps(self.model_dump(exclude_none=True), indent=None, ensure_ascii=False)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")


def read_coco(path: Path) -> CocoDataset:
    return CocoDataset.model_validate_json(path.read_text(encoding="utf-8"))


def polygon_area(poly: Iterable[float]) -> float:
    """Shoelace area of a flat ``[x0, y0, x1, y1, ...]`` polygon."""
    pts = list(poly)
    n = len(pts) // 2
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x0, y0 = pts[2 * i], pts[2 * i + 1]
        x1, y1 = pts[2 * ((i + 1) % n)], pts[2 * ((i + 1) % n) + 1]
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0


def polygon_bbox(poly: Iterable[float]) -> list[float]:
    pts = list(poly)
    xs, ys = pts[0::2], pts[1::2]
    x0, y0 = min(xs), min(ys)
    return [x0, y0, max(xs) - x0, max(ys) - y0]
