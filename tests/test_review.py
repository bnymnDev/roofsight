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
