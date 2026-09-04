from pathlib import Path

import numpy as np

from roofsight.export.io import read_instances, write_instances
from roofsight.export.verify import Instance, compare


def box(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros((64, 64), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_identical_passes(tmp_path: Path) -> None:
    ref = [[Instance(1, 0.9, box(4, 4, 40, 40)), Instance(2, 0.8, box(50, 50, 60, 60))], []]
    r = compare(ref, ref)
    assert r.ok and r.mean_mask_iou == 1.0 and r.n_matched == 2
    p = tmp_path / "i.json"
    write_instances(ref, p)
    r2 = compare(ref, read_instances(p))
    assert r2.ok


def test_drift_fails() -> None:
    ref = [[Instance(1, 0.9, box(4, 4, 40, 40))]]
    cand = [[Instance(1, 0.9, box(6, 4, 42, 40))]]
    r = compare(ref, cand)
    assert not r.ok and r.mean_mask_iou < 0.98


def test_missing_instance_fails() -> None:
    ref = [[Instance(1, 0.9, box(4, 4, 40, 40)), Instance(2, 0.8, box(50, 50, 60, 60))]]
    cand = [[Instance(1, 0.9, box(4, 4, 40, 40))]]
    r = compare(ref, cand)
    assert r.n_matched == 1 and not r.ok
    assert r.as_dict()["ok"] is False
