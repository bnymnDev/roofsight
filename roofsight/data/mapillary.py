"""Mapillary API v4 client: bbox queries over residential areas, perspective images only.

Images are CC-BY-SA 4.0. Every downloaded record keeps the Mapillary id and creator for the
``attribution`` field. Token comes from ``MAPILLARY_TOKEN``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

API = "https://graph.mapillary.com"
FIELDS = "id,thumb_2048_url,thumb_1024_url,camera_type,quality_score,creator,captured_at,geometry"
LICENSE = "CC-BY-SA-4.0"


@dataclass(frozen=True, slots=True)
class MapillaryImage:
    id: str
    url: str
    creator: str
    quality_score: float
    camera_type: str
    lon: float
    lat: float

    @property
    def attribution(self) -> str:
        return f"© {self.creator}, Mapillary, CC BY-SA 4.0, image {self.id}"


def parse_image(item: dict[str, Any], size_field: str = "thumb_2048_url") -> MapillaryImage | None:
    url = item.get(size_field) or item.get("thumb_1024_url")
    if not url:
        return None
    creator = item.get("creator") or {}
    coords = (item.get("geometry") or {}).get("coordinates") or [0.0, 0.0]
    return MapillaryImage(
        id=str(item["id"]),
        url=str(url),
        creator=str(creator.get("username", "unknown")),
        quality_score=float(item.get("quality_score") or 0.0),
        camera_type=str(item.get("camera_type") or ""),
        lon=float(coords[0]),
        lat=float(coords[1]),
    )


def keep(img: MapillaryImage, camera_type: str, min_quality: float) -> bool:
    """The filter from SPEC: perspective only, quality score, no 360°."""
    if img.camera_type != camera_type:
        return False
    return img.quality_score >= min_quality


class MapillaryClient:
    def __init__(self, token: str | None = None, client: httpx.Client | None = None) -> None:
        self.token = token or os.environ.get("MAPILLARY_TOKEN", "")
        if not self.token:
            raise RuntimeError("MAPILLARY_TOKEN is not set")
        self._client = client or httpx.Client(timeout=60)

    def search(
        self,
        bbox: str,
        limit: int = 200,
        camera_type: str = "perspective",
        min_quality: float = 0.6,
        size_field: str = "thumb_2048_url",
    ) -> Iterator[MapillaryImage]:
        params: dict[str, Any] = {
            "access_token": self.token,
            "fields": FIELDS,
            "bbox": bbox,
            "limit": min(limit, 2000),
        }
        url: str | None = f"{API}/images"
        yielded = 0
        while url and yielded < limit:
            r = self._client.get(url, params=params)
            r.raise_for_status()
            body = r.json()
            for item in body.get("data", []):
                img = parse_image(item, size_field)
                if img and keep(img, camera_type, min_quality):
                    yield img
                    yielded += 1
                    if yielded >= limit:
                        return
            url = (body.get("paging") or {}).get("next")
            params = {}

    def download(self, img: MapillaryImage, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            return dest
        with self._client.stream("GET", img.url) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return dest
