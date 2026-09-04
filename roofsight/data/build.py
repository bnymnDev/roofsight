"""``roofsight data build``: download → dedupe → anonymize → roof filter → split → COCO.

Labeling is a separate step (``roofsight label``); this writes ``images.json`` with image
records only, which the labeling pipeline then fills with annotations.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from roofsight.categories import coco_categories
from roofsight.coco import CocoDataset, CocoInfo, CocoLicense, ImageRecord
from roofsight.data.anonymize import anonymize
from roofsight.data.config import DataConfig
from roofsight.data.dedupe import dedupe, phash
from roofsight.data.mapillary import LICENSE as MAPILLARY_LICENSE
from roofsight.data.mapillary import MapillaryClient
from roofsight.data.split import assign_splits

RoofFilter = Callable[[Path], float]
"""Returns the fraction of the image covered by roof. Implemented in labeling/ (SAM 3)."""

CC_BY_SA = CocoLicense(
    id=1, name="CC-BY-SA-4.0", url="https://creativecommons.org/licenses/by-sa/4.0/"
)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def collect_own_photos(config: DataConfig) -> list[ImageRecord]:
    root = config.own.root
    if not config.own.enabled or not root.exists():
        return []
    records: list[ImageRecord] = []
    for p in sorted(root.glob("*.jpg")):
        w, h = _image_size(p)
        sidecar = p.with_suffix(".arkit.json")
        records.append(
            ImageRecord(
                id=0,
                file_name=p.name,
                width=w,
                height=h,
                source="own",
                license="CC-BY-SA-4.0",
                attribution=config.own.attribution,
                region=config.own.region,
                source_id=p.stem,
                has_pose=sidecar.exists(),
            )
        )
    return records


def download_mapillary(config: DataConfig, raw_dir: Path) -> list[ImageRecord]:
    if not config.mapillary.enabled:
        return []
    client = MapillaryClient()
    records: list[ImageRecord] = []
    for bbox in config.mapillary.bboxes:
        for img in client.search(
            bbox.as_query(),
            limit=config.mapillary.per_bbox_limit,
            camera_type=config.mapillary.camera_type,
            min_quality=config.mapillary.min_quality_score,
            size_field=config.mapillary.image_size,
        ):
            dest = raw_dir / f"mapillary_{img.id}.jpg"
            client.download(img, dest)
            w, h = _image_size(dest)
            records.append(
                ImageRecord(
                    id=0,
                    file_name=dest.name,
                    width=w,
                    height=h,
                    source="mapillary",
                    license=MAPILLARY_LICENSE,
                    attribution=img.attribution,
                    region=bbox.region,
                    source_id=img.id,
                )
            )
    return records


def build(
    config: DataConfig,
    roof_filter: RoofFilter | None = None,
    records: list[ImageRecord] | None = None,
    raw_dir: Path | None = None,
) -> CocoDataset:
    """Run the pipeline. ``records``/``raw_dir`` allow tests and offline builds to skip download."""
    out = config.out / config.version
    raw = raw_dir or (config.out / "raw")
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if records is None:
        records = download_mapillary(config, raw) + collect_own_photos(config)
        own_root = config.own.root
    else:
        own_root = raw

    # dedupe
    hashed = []
    for r in records:
        src = (raw if r.source == "mapillary" else own_root) / r.file_name
        r.phash = phash(src, config.dedupe.hash_size)
        hashed.append((r.file_name, r.phash))
    keep = set(dedupe(hashed, config.dedupe.max_hamming))
    records = [r for r in records if r.file_name in keep]

    # anonymize + roof filter
    kept: list[ImageRecord] = []
    for r in records:
        src = (raw if r.source == "mapillary" else own_root) / r.file_name
        dst = images_dir / r.file_name
        anonymize(src, dst, config.anonymize.backend, config.anonymize.threshold)
        r.anonymized = True
        if roof_filter is not None and roof_filter(dst) < config.roof_filter.min_roof_fraction:
            dst.unlink(missing_ok=True)
            continue
        kept.append(r)

    for i, r in enumerate(kept, start=1):
        r.id = i

    frozen: list[int] = []
    if config.split.frozen_test and config.split.frozen_test.exists():
        frozen = json.loads(config.split.frozen_test.read_text())
    assignment = assign_splits(((r.id, r.region, 0) for r in kept), config.split, frozen)
    for r in kept:
        r.split = assignment[r.id]

    ds = CocoDataset(
        info=CocoInfo(
            version=config.version,
            year=datetime.now(UTC).year,
            date_created=datetime.now(UTC).date().isoformat(),
        ),
        licenses=[CC_BY_SA],
        images=kept,
        annotations=[],
        categories=coco_categories(),
    )
    ds.write(out / "images.json")
    return ds
