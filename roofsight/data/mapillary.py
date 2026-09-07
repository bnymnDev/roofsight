"""Mapillary API v4 client: bbox queries over residential areas, perspective images only.

Images are CC-BY-SA 4.0. Every downloaded record keeps the Mapillary id and creator for the
``attribution`` field. Token comes from ``MAPILLARY_TOKEN``.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

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


def grid_cells(bbox: str, n: int) -> list[str]:
    """Split ``west,south,east,north`` into an ``n × n`` grid of sub-boxes.

    The bbox endpoint does not paginate and returns 500 on boxes with too many images, so
    small cells with a modest ``limit`` each are the only way to get everything.
    """
    w, s, e, nn = (float(v) for v in bbox.split(","))
    dx, dy = (e - w) / n, (nn - s) / n
    cells = []
    for i in range(n):
        for j in range(n):
            cells.append(
                f"{w + i * dx:.6f},{s + j * dy:.6f},{w + (i + 1) * dx:.6f},{s + (j + 1) * dy:.6f}"
            )
    return cells


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

    def _get_cell(self, bbox: str, limit: int, retries: int = 3) -> list[dict[str, Any]]:
        """One bbox query. On 5xx: back off, halve the limit, retry; give up after ``retries``."""
        for attempt in range(retries + 1):
            params: dict[str, Any] = {
                "access_token": self.token,
                "fields": FIELDS,
                "bbox": bbox,
                "limit": limit,
            }
            r = self._client.get(f"{API}/images", params=params)
            if r.status_code < 500:
                r.raise_for_status()
                data: list[dict[str, Any]] = r.json().get("data", [])
                return data
            if attempt == retries:
                log.warning("mapillary: giving up on cell %s after %d retries", bbox, retries)
                return []
            time.sleep(1.5 * (attempt + 1))
            limit = max(10, limit // 2)
        return []

    def search(
        self,
        bbox: str,
        limit: int = 200,
        camera_type: str = "perspective",
        min_quality: float = 0.6,
        size_field: str = "thumb_2048_url",
        grid: int = 4,
        per_cell_limit: int = 100,
    ) -> Iterator[MapillaryImage]:
        """Up to ``limit`` images in ``bbox``: the box is queried as a ``grid × grid`` raster,
        cells are deduped by image id, and a cell that keeps failing is skipped, not fatal."""
        seen: set[str] = set()
        yielded = 0
        for cell in grid_cells(bbox, grid):
            for item in self._get_cell(cell, per_cell_limit):
                img = parse_image(item, size_field)
                if img is None or img.id in seen or not keep(img, camera_type, min_quality):
                    continue
                seen.add(img.id)
                yield img
                yielded += 1
                if yielded >= limit:
                    return

    def download(self, img: MapillaryImage, dest: Path, retries: int = 3) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            return dest
        tmp = dest.with_suffix(".part")
        for attempt in range(retries + 1):
            try:
                with self._client.stream("GET", img.url) as r:
                    r.raise_for_status()
                    with tmp.open("wb") as f:
                        for chunk in r.iter_bytes():
                            f.write(chunk)
                tmp.rename(dest)
                return dest
            except (httpx.HTTPError, OSError):
                if attempt == retries:
                    raise
                time.sleep(1.5 * (attempt + 1))
        return dest


def fetch_by_id(
    client: MapillaryClient, image_id: str, size_field: str = "thumb_2048_url"
) -> MapillaryImage | None:
    """Look up one image by id. The download URLs expire, so a manifest stores ids, not URLs."""
    r = client._client.get(
        f"{API}/{image_id}",
        params={"access_token": client.token, "fields": FIELDS},
    )
    # 404: deleted upstream; 400: Mapillary answers this for ids it no longer serves
    if r.status_code in (400, 404):
        return None
    r.raise_for_status()
    return parse_image(r.json(), size_field)
