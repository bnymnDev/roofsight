"""FiftyOne round trip for human review.

Export: COCO → FiftyOne dataset with ``provenance`` as a label attribute. Import: the edited
FiftyOne dataset back to COCO, with provenance updated to ``auto_edited``/``manual`` by
comparing against the auto labels. The comparison itself is pure and tested without FiftyOne.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from roofsight.coco import Annotation, CocoDataset, Provenance


def merge_provenance(auto: CocoDataset, reviewed: CocoDataset) -> CocoDataset:
    """Set provenance on ``reviewed`` annotations by comparing with ``auto``.

    Same id, same mask and category → keep ``auto``. Same id, anything changed → ``auto_edited``.
    Id not in auto → ``manual``. Annotations deleted by the reviewer are simply absent.
    """
    by_id = {a.id: a for a in auto.annotations}
    out: list[Annotation] = []
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
        out.append(a.model_copy(update={"provenance": prov}))
    return reviewed.model_copy(update={"annotations": out})


def review_stats(ds: CocoDataset) -> dict[str, int]:
    counts: dict[str, int] = {"auto": 0, "auto_edited": 0, "manual": 0}
    for a in ds.annotations:
        counts[a.provenance] += 1
    return counts


def export_to_fiftyone(ds: CocoDataset, images_root: Path, name: str) -> Any:  # pragma: no cover
    import fiftyone as fo

    tmp = images_root.parent / f"{name}.fiftyone.json"
    ds.write(tmp)
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=str(images_root),
        labels_path=str(tmp),
        label_types=["segmentations"],
        extra_attrs=["provenance", "edge_type"],  # score is a native COCO field in FiftyOne
        name=name,
        overwrite=True,
    )
    dataset.persistent = True
    return dataset


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
