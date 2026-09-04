from pathlib import Path

import numpy as np
from PIL import Image

from roofsight.data.dedupe import dedupe, hamming, phash


def test_dedupe_keeps_first_of_group(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    b = a.copy()
    b[:2, :2] += 1  # near duplicate
    c = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    for name, arr in (("a", a), ("b", b), ("c", c)):
        Image.fromarray(arr).save(tmp_path / f"{name}.png")
    ha, hb, hc = (phash(tmp_path / f"{n}.png") for n in "abc")
    assert hamming(ha, hb) <= 6
    assert hamming(ha, hc) > 6
    assert dedupe([("a", ha), ("b", hb), ("c", hc)]) == ["a", "c"]
