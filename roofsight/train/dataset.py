"""Lay out a COCO dataset the way ``rfdetr`` expects it: ``train/valid/test`` folders with
``_annotations.coco.json`` each. Symlinks, no copies."""

from __future__ import annotations

import os
from pathlib import Path

from roofsight.coco import CocoDataset, Split

RFDETR_SPLIT_DIRS: dict[Split, str] = {"train": "train", "val": "valid", "test": "test"}


def prepare_rfdetr_layout(ds: CocoDataset, images_root: Path, out: Path) -> Path:
    for split, dirname in RFDETR_SPLIT_DIRS.items():
        sub = ds.subset(split)
        d = out / dirname
        d.mkdir(parents=True, exist_ok=True)
        for im in sub.images:
            link = d / im.file_name
            if not link.exists():
                os.symlink((images_root / im.file_name).resolve(), link)
        # rfdetr reads plain COCO; our extra fields ride along untouched
        sub.write(d / "_annotations.coco.json")
    return out
