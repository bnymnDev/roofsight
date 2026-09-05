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


def test_fetch_manifest(tmp_path: Path) -> None:
    import httpx

    from roofsight.coco import CocoDataset, CocoInfo
    from roofsight.data.build import fetch_manifest
    from roofsight.data.mapillary import MapillaryClient

    rng = np.random.default_rng(5)
    buf = __import__("io").BytesIO()
    Image.fromarray(rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)).save(buf, "JPEG")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith("https://graph.mapillary.com/gone"):
            return httpx.Response(404)
        if url.startswith("https://graph.mapillary.com/"):
            return httpx.Response(
                200,
                json={
                    "id": url.split("/")[3].split("?")[0],
                    "thumb_2048_url": "https://cdn/x.jpg",
                    "camera_type": "perspective",
                    "quality_score": 0.9,
                },
            )
        return httpx.Response(200, content=buf.getvalue())

    client = MapillaryClient(token="t", client=httpx.Client(transport=httpx.MockTransport(handler)))
    recs = [
        ImageRecord(
            id=1,
            file_name="mapillary_a.jpg",
            width=32,
            height=32,
            source="mapillary",
            license="CC-BY-SA-4.0",
            attribution="x",
            source_id="a",
            split="train",
        ),
        ImageRecord(
            id=2,
            file_name="mapillary_gone.jpg",
            width=32,
            height=32,
            source="mapillary",
            license="CC-BY-SA-4.0",
            attribution="x",
            source_id="gone",
            split="train",
        ),
    ]
    ds = CocoDataset(info=CocoInfo(version="v", year=2026), images=recs)
    cfg = DataConfig(version="v", out=tmp_path, anonymize=AnonymizeConfig(backend="none"))
    n, missing = fetch_manifest(ds, tmp_path / "images", cfg, client, raw_dir=tmp_path / "raw")
    assert n == 1 and [r.id for r in missing] == [2]
    assert (tmp_path / "images" / "mapillary_a.jpg").exists()
    # idempotent: nothing to fetch the second time
    assert fetch_manifest(ds, tmp_path / "images", cfg, client, raw_dir=tmp_path / "raw")[0] == 0
