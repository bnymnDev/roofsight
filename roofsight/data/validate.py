"""``roofsight data validate``: the checks a dataset must pass before it is versioned."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from roofsight.categories import OBSTACLE_IDS, ROOF_EDGE_ID
from roofsight.coco import ALLOWED_LICENSES, CocoDataset


@dataclass(slots=True)
class ValidationReport:
    n_images: int = 0
    n_annotations: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(ds: CocoDataset, images_root: Path | None = None) -> ValidationReport:
    report = ValidationReport(n_images=len(ds.images), n_annotations=len(ds.annotations))
    anns_by_image: dict[int, int] = {}
    obstacle_images: set[int] = set()
    for a in ds.annotations:
        anns_by_image[a.image_id] = anns_by_image.get(a.image_id, 0) + 1
        if a.category_id in OBSTACLE_IDS:
            obstacle_images.add(a.image_id)
        if a.category_id == ROOF_EDGE_ID and a.edge_type is None:
            report.errors.append(f"annotation {a.id}: roof_edge without edge_type")
        if a.area <= 0:
            report.errors.append(f"annotation {a.id}: non-positive area")

    for im in ds.images:
        if im.license not in ALLOWED_LICENSES:
            report.errors.append(f"image {im.id}: license {im.license!r} not allowed")
        if not im.attribution.strip():
            report.errors.append(f"image {im.id}: missing attribution")
        if not im.anonymized:
            report.errors.append(f"image {im.id}: not anonymized")
        if im.split is None:
            report.errors.append(f"image {im.id}: no split")
        if im.id not in anns_by_image:
            report.warnings.append(f"image {im.id}: no annotations")
        if images_root is not None and not (images_root / im.file_name).exists():
            report.errors.append(f"image {im.id}: file {im.file_name} missing")

    if ds.images:
        frac = len(obstacle_images) / len(ds.images)
        if frac < 0.3:
            report.warnings.append(
                f"only {frac:.0%} of images have an obstacle instance (target ≥ 30 %)"
            )
    return report
