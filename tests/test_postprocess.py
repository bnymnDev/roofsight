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
