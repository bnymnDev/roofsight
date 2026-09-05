"""``configs/data/*.yaml`` as a pydantic model."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Geographic bounding box, ``west,south,east,north`` in WGS84."""

    name: str
    region: str
    west: float
    south: float
    east: float
    north: float

    def as_query(self) -> str:
        return f"{self.west},{self.south},{self.east},{self.north}"


class MapillarySource(BaseModel):
    enabled: bool = True
    bboxes: list[BBox] = Field(default_factory=list)
    camera_type: str = "perspective"
    min_quality_score: float = 0.6
    per_bbox_limit: int = 200
    grid: int = 4
    per_cell_limit: int = 100
    image_size: str = "thumb_2048_url"


class OwnPhotoSource(BaseModel):
    enabled: bool = True
    root: Path = Path("datasets/own")
    attribution: str = "RoofSight contributors"
    region: str = "DE-NRW"


class DedupeConfig(BaseModel):
    hash_size: int = 16
    max_hamming: int = 6


class AnonymizeConfig(BaseModel):
    backend: str = "deface"  # deface | egoblur | none
    threshold: float = 0.2
    fail_if_missing: bool = True


class RoofFilterConfig(BaseModel):
    backend: str = "sam3"  # sam3 | file
    min_roof_fraction: float = 0.05  # fraction of the image covered by roof (or file score)
    prompt: str = "roof"


class SplitConfig(BaseModel):
    train: float = 0.7
    val: float = 0.1
    test: float = 0.15
    verify: float = 0.05
    verify_count: int = 50
    frozen_test: Path | None = None  # test image ids frozen at v0.1
    seed: int = 20260101


class DataConfig(BaseModel):
    version: str
    out: Path = Path("datasets")
    target_images: int = 500
    min_obstacle_fraction: float = 0.3
    mapillary: MapillarySource = Field(default_factory=MapillarySource)
    own: OwnPhotoSource = Field(default_factory=OwnPhotoSource)
    dedupe: DedupeConfig = Field(default_factory=DedupeConfig)
    anonymize: AnonymizeConfig = Field(default_factory=AnonymizeConfig)
    roof_filter: RoofFilterConfig = Field(default_factory=RoofFilterConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)

    @classmethod
    def load(cls, path: Path) -> DataConfig:
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)
