"""FiftyOne round trip for human review.

Export: COCO → a FiftyOne dataset, optionally narrowed to one split and to instances above a
score, so a reviewer sees the frozen test split instead of all 560 frames. What was exported
is recorded in a scope file; import replaces exactly that scope and leaves everything else
untouched, so reviewing a subset can never delete the rest of the dataset.

The merge itself is pure and tested without FiftyOne.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from roofsight.coco import Annotation, CocoDataset, Provenance, Split

SCOPE_SUFFIX = ".scope.json"


class ReviewScope(BaseModel):
    """What an export handed to the reviewer: these images, these annotation ids."""

    image_ids: list[int] = Field(default_factory=list)
    annotation_ids: list[int] = Field(default_factory=list)
    split: Split | None = None
    min_score: float | None = None

    @classmethod
    def of(cls, ds: CocoDataset, **kw: Any) -> ReviewScope:
        return cls(
            image_ids=[im.id for im in ds.images],
            annotation_ids=[a.id for a in ds.annotations],
            **kw,
        )

    def write(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> ReviewScope:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def select(
    ds: CocoDataset,
    images_root: Path | None = None,
    split: Split | None = None,
    min_score: float | None = None,
) -> tuple[CocoDataset, list[str]]:
    """Narrow a dataset for review: one split, instances above a score, files on disk.

    FiftyOne aborts on the first missing image file, so records without a file are dropped and
    reported.
    """
    images = ds.images if split is None else [im for im in ds.images if im.split == split]
    missing: list[str] = []
    if images_root is not None:
        on_disk = [im for im in images if (images_root / im.file_name).exists()]
        missing = [im.file_name for im in images if not (images_root / im.file_name).exists()]
        images = on_disk
    ids = {im.id for im in images}
    anns = [a for a in ds.annotations if a.image_id in ids]
    if min_score is not None:
        anns = [a for a in anns if a.score is None or a.score >= min_score]
    return ds.model_copy(update={"images": images, "annotations": anns}), missing


def merge_reviewed(
    auto: CocoDataset, reviewed: CocoDataset, scope: ReviewScope | None = None
) -> CocoDataset:
    """Fold a reviewed subset back into the full dataset.

    Inside the scope: an annotation the reviewer left untouched keeps its provenance, a changed
    one becomes ``auto_edited``, a deleted one is gone, and a new one is ``manual``. Everything
    outside the scope — other splits, instances filtered out of the export — is carried over
    unchanged. Without a scope the reviewed dataset replaces the annotations wholesale (the
    reviewer saw everything).
    """
    by_id = {a.id: a for a in auto.annotations}
    reviewed_anns: list[Annotation] = []
    for a in reviewed.annotations:
        orig = by_id.get(a.id)
        prov: Provenance
        if orig is None:
            prov = "manual"
        elif (
            orig.category_id == a.category_id
            and orig.segmentation == a.segmentation
            and orig.edge_type == a.edge_type
        ):
            prov = orig.provenance
        else:
            prov = "auto_edited"
        reviewed_anns.append(a.model_copy(update={"provenance": prov}))

    if scope is None:
        return reviewed.model_copy(update={"annotations": reviewed_anns})

    in_scope = set(scope.annotation_ids)
    kept = [a for a in auto.annotations if a.id not in in_scope]
    merged = kept + reviewed_anns
    merged.sort(key=lambda a: (a.image_id, a.id))
    return auto.model_copy(update={"annotations": merged})


# kept for callers that reviewed the whole dataset
def merge_provenance(auto: CocoDataset, reviewed: CocoDataset) -> CocoDataset:
    return merge_reviewed(auto, reviewed)


def review_stats(ds: CocoDataset) -> dict[str, int]:
    counts: dict[str, int] = {"auto": 0, "auto_edited": 0, "manual": 0}
    for a in ds.annotations:
        counts[a.provenance] += 1
    return counts


def export_to_fiftyone(
    ds: CocoDataset,
    images_root: Path,
    name: str,
    split: Split | None = None,
    min_score: float | None = None,
) -> tuple[Any, ReviewScope, list[str]]:  # pragma: no cover - needs FiftyOne
    import fiftyone as fo

    subset, missing = select(ds, images_root, split, min_score)
    scope = ReviewScope.of(subset, split=split, min_score=min_score)
    tmp = images_root.parent / f"{name}.fiftyone.json"
    subset.write(tmp)
    scope.write(images_root.parent / f"{name}{SCOPE_SUFFIX}")
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=str(images_root),
        labels_path=str(tmp),
        label_types=["segmentations"],
        extra_attrs=["provenance", "edge_type"],
        name=name,
        overwrite=True,
    )
    dataset.persistent = True
    return dataset, scope, missing


def import_from_fiftyone(name: str, out: Path) -> Path:  # pragma: no cover
    import fiftyone as fo

    dataset = fo.load_dataset(name)
    dataset.export(
        dataset_type=fo.types.COCODetectionDataset,
        labels_path=str(out),
        label_field="segmentations",
        export_media=False,
    )
    return out
