from roofsight.labeling.review import merge_provenance, review_stats
from tests.conftest import ann, make_dataset, rect


def test_merge_provenance() -> None:
    auto = make_dataset()
    reviewed = auto.model_copy(deep=True)
    # edit annotation 3, delete 6, add 99
    reviewed.annotations[2] = ann(3, 1, 2, rect(10, 12, 17, 20))
    reviewed.annotations = [a for a in reviewed.annotations if a.id != 6]
    reviewed.annotations.append(ann(99, 2, 6, rect(40, 40, 44, 44)))
    merged = merge_provenance(auto, reviewed)
    prov = {a.id: a.provenance for a in merged.annotations}
    assert prov[3] == "auto_edited"
    assert prov[99] == "manual"
    assert prov[1] == "auto"
    assert 6 not in prov
    assert review_stats(merged) == {"auto": 5, "auto_edited": 1, "manual": 1}


def test_select_drops_missing_files(dataset_dir) -> None:  # type: ignore[no-untyped-def]
    from roofsight.coco import read_coco
    from roofsight.labeling.review import select

    ds = read_coco(dataset_dir / "annotations.json")
    (dataset_dir / "images" / "img_002.jpg").unlink()
    kept, missing = select(ds, dataset_dir / "images")
    assert missing == ["img_002.jpg"]
    assert [im.id for im in kept.images] == [1]
    assert all(a.image_id == 1 for a in kept.annotations)


def test_select_by_split_and_score() -> None:
    from roofsight.labeling.review import select

    ds = make_dataset()
    ds.images[1].split = "train"
    ds.annotations[0].score = 0.9
    ds.annotations[1].score = 0.2
    kept, _ = select(ds, None, split="test")
    assert [im.id for im in kept.images] == [1]
    assert {a.image_id for a in kept.annotations} == {1}
    kept, _ = select(ds, None, split="test", min_score=0.5)
    ids = {a.id for a in kept.annotations}
    assert 1 in ids and 2 not in ids  # 2 scored 0.2
    assert 3 in ids  # no score at all is kept


def test_merge_reviewed_leaves_everything_outside_the_scope_alone() -> None:
    from roofsight.labeling.review import ReviewScope, merge_reviewed, select

    auto = make_dataset()
    auto.images[1].split = "train"
    subset, _ = select(auto, None, split="test")
    scope = ReviewScope.of(subset, split="test")
    assert set(scope.annotation_ids) == {1, 2, 3, 4}

    # the reviewer deletes annotation 2, edits 3 and adds one
    reviewed_anns = [a for a in subset.annotations if a.id != 2]
    reviewed_anns = [ann(3, 1, 2, rect(10, 12, 18, 20)) if a.id == 3 else a for a in reviewed_anns]
    reviewed_anns.append(ann(99, 1, 6, rect(40, 40, 44, 44)))
    reviewed = subset.model_copy(update={"annotations": reviewed_anns})

    merged = merge_reviewed(auto, reviewed, scope)
    prov = {a.id: a.provenance for a in merged.annotations}
    assert 2 not in prov  # deleted inside the scope
    assert prov[3] == "auto_edited"
    assert prov[99] == "manual"
    assert prov[1] == "auto"
    # the train image keeps every annotation it had, untouched
    assert {a.id for a in merged.annotations if a.image_id == 2} == {5, 6, 7}
