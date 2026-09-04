import pytest
from pydantic import ValidationError

from roofsight.coco import Annotation, CocoDataset, ImageRecord, polygon_area, read_coco
from tests.conftest import ann, image, make_dataset, rect


def test_license_required() -> None:
    with pytest.raises(ValidationError, match="license"):
        image(1, license="CC0")
    with pytest.raises(ValidationError, match="attribution"):
        image(1, attribution="  ")


def test_edge_type_rules() -> None:
    with pytest.raises(ValidationError, match="edge_type"):
        ann(1, 1, 10, rect(0, 0, 4, 1))
    with pytest.raises(ValidationError, match="edge_type"):
        ann(1, 1, 1, rect(0, 0, 4, 4), edge_type="ridge")
    a = ann(1, 1, 10, rect(0, 0, 4, 1), edge_type="eave")
    assert a.edge_type == "eave"


def test_unknown_category() -> None:
    with pytest.raises(ValidationError):
        Annotation(id=1, image_id=1, category_id=99, segmentation=[], area=1, bbox=[0, 0, 1, 1])


def test_dangling_reference() -> None:
    ds = make_dataset()
    with pytest.raises(ValidationError, match="unknown image"):
        CocoDataset(info=ds.info, images=ds.images[:1], annotations=ds.annotations)


def test_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ds = make_dataset()
    ds.write(tmp_path / "a.json")
    back = read_coco(tmp_path / "a.json")
    assert back == ds
    assert back.subset("test").images == ds.images
    assert back.subset("train").images == []


def test_polygon_area() -> None:
    assert polygon_area(rect(0, 0, 10, 5)) == 50.0
    assert ImageRecord.model_fields["license"].annotation is str
