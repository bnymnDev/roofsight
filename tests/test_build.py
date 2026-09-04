from pathlib import Path

import numpy as np
from PIL import Image

from roofsight.coco import ImageRecord, read_coco
from roofsight.data.build import build
from roofsight.data.config import AnonymizeConfig, DataConfig, MapillarySource, SplitConfig


def test_offline_build(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(2)
    records = []
    for i in range(6):
        arr = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
        name = f"m_{i}.jpg"
        Image.fromarray(arr).save(raw / name)
        records.append(
            ImageRecord(
                id=0,
                file_name=name,
                width=48,
                height=48,
                source="mapillary",
                license="CC-BY-SA-4.0",
                attribution="© x, Mapillary, CC BY-SA 4.0",
                region="DE-NRW",
                source_id=str(i),
            )
        )
    # a duplicate of the first image, which must be dropped
    Image.open(raw / "m_0.jpg").save(raw / "dup.jpg")
    records.append(records[0].model_copy(update={"file_name": "dup.jpg", "source_id": "dup"}))

    cfg = DataConfig(
        version="v-test",
        out=tmp_path / "out",
        mapillary=MapillarySource(enabled=False),
        anonymize=AnonymizeConfig(backend="none"),
        split=SplitConfig(verify_count=1),
    )
    ds = build(
        cfg, roof_filter=lambda p: 0.0 if p.name == "m_5.jpg" else 0.5, records=records, raw_dir=raw
    )
    assert len(ds.images) == 5  # 7 - 1 duplicate - 1 filtered
    assert all(im.anonymized and im.split for im in ds.images)
    assert (tmp_path / "out" / "v-test" / "images.json").exists()
    assert read_coco(tmp_path / "out" / "v-test" / "images.json") == ds
    assert sorted(p.name for p in (tmp_path / "out/v-test/images").iterdir()) == [
        f"m_{i}.jpg" for i in range(5)
    ]
