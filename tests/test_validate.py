from roofsight.data.validate import validate
from tests.conftest import image, make_dataset


def test_valid_dataset(dataset_dir) -> None:  # type: ignore[no-untyped-def]
    from roofsight.coco import read_coco

    r = validate(read_coco(dataset_dir / "annotations.json"), dataset_dir / "images")
    assert r.ok, r.errors


def test_missing_anonymization_and_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ds = make_dataset()
    ds.images[0] = image(1, anonymized=False)
    r = validate(ds, tmp_path)
    assert any("not anonymized" in e for e in r.errors)
    assert any("missing" in e for e in r.errors)
    assert not r.ok
