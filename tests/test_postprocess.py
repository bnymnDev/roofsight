import numpy as np

from roofsight.labeling.candidates import Candidate
from roofsight.labeling.postprocess import min_area, nms, split_planes_by_edges


def box(x0: int, y0: int, x1: int, y1: int, h: int = 64, w: int = 64) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_nms_across_synonyms() -> None:
    a = Candidate(2, "chimney", 0.9, box(10, 10, 20, 20))
    b = Candidate(2, "brick chimney stack", 0.8, box(10, 10, 20, 21))
    c = Candidate(2, "chimney", 0.7, box(40, 40, 50, 50))
    d = Candidate(4, "skylight", 0.5, box(10, 10, 20, 20))  # other category, same box
    kept = nms([a, b, c, d], 0.7)
    assert [(k.category_id, k.prompt, k.score) for k in kept] == [
        (2, "chimney", 0.9),
        (2, "chimney", 0.7),
        (4, "skylight", 0.5),
    ]


def test_min_area() -> None:
    small = Candidate(6, "vent", 0.9, box(0, 0, 3, 3))
    big = Candidate(6, "vent", 0.9, box(0, 0, 10, 10))
    assert min_area([small, big], {6: 25}) == [big]


def test_split_plane_by_ridge() -> None:
    plane = Candidate(1, "roof", 0.9, box(4, 10, 60, 50))
    ridge = Candidate(10, "ridge line of roof", 0.8, box(4, 29, 60, 31))
    out = split_planes_by_edges([plane, ridge], edge_width_px=3, min_fragment_px=64)
    planes = [c for c in out if c.category_id == 1]
    assert len(planes) == 2
    assert sum(int(p.mask.sum()) for p in planes) >= 0.9 * int(plane.mask.sum())
    assert any(c.category_id == 10 for c in out)


def test_untouched_plane_survives() -> None:
    plane = Candidate(1, "roof", 0.9, box(4, 10, 60, 50))
    edge = Candidate(10, "eave line of roof", 0.8, box(4, 49, 60, 51))
    out = split_planes_by_edges([plane, edge])
    assert sum(c.category_id == 1 for c in out) == 1


def test_edge_cuts_only_the_plane_it_touches() -> None:
    from roofsight.labeling.postprocess import split_planes_by_edges

    left = Candidate(1, "roof", 0.9, box(2, 10, 28, 50))
    right = Candidate(1, "roof", 0.9, box(36, 10, 62, 50))
    ridge_left = Candidate(10, "ridge line of roof", 0.8, box(4, 29, 26, 31))
    out = split_planes_by_edges([left, right, ridge_left], edge_width_px=3, min_fragment_px=64)
    planes = [c for c in out if c.category_id == 1]
    assert len(planes) == 3  # left split in two, right untouched
    assert any(np.array_equal(p.mask, right.mask) for p in planes)


def test_keep_on_roof() -> None:
    from roofsight.labeling.postprocess import keep_on_roof, postprocess

    plane = Candidate(1, "roof", 0.9, box(10, 10, 50, 30))
    skylight_on = Candidate(4, "roof window", 0.8, box(20, 15, 26, 20))
    window_facade = Candidate(4, "roof window", 0.8, box(20, 45, 26, 52))
    chimney_above = Candidate(2, "chimney", 0.8, box(40, 8, 44, 14))  # sticks out over the ridge
    tree_street = Candidate(9, "tree", 0.7, box(0, 40, 12, 64))
    tree_over_roof = Candidate(9, "tree", 0.7, box(5, 5, 20, 25))
    kept = keep_on_roof(
        [plane, skylight_on, window_facade, chimney_above, tree_street, tree_over_roof]
    )
    assert [c.prompt for c in kept] == ["roof", "roof window", "chimney", "tree"]
    assert kept[3].mask[10, 10]  # the tree over the roof, not the street tree
    # no planes → nothing is on a roof
    assert keep_on_roof([window_facade, tree_street]) == []
    # the full chain still works
    assert len(postprocess([plane, window_facade], 0.7, {1: 100})) == 1
