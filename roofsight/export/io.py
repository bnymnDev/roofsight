"""Per-image instance dumps used by ``roofsight export verify``.

Format: ``[[{"category_id", "score", "segmentation": RLE}, ...] per image]``. Written by the
PyTorch and Core ML runners (the latter from the ``RoofGeometryBench`` app).
"""

from __future__ import annotations

import json
from pathlib import Path

from roofsight.export.verify import Instance
from roofsight.masks import mask_to_rle, segmentation_to_mask


def read_instances(path: Path) -> list[list[Instance]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[list[Instance]] = []
    for image in raw:
        insts: list[Instance] = []
        for d in image:
            seg = d["segmentation"]
            h, w = seg["size"]
            insts.append(
                Instance(int(d["category_id"]), float(d["score"]), segmentation_to_mask(seg, h, w))
            )
        out.append(insts)
    return out


def write_instances(images: list[list[Instance]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        [
            {"category_id": i.category_id, "score": i.score, "segmentation": mask_to_rle(i.mask)}
            for i in insts
        ]
        for insts in images
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
