from roofsight.data.config import SplitConfig
from roofsight.data.split import assign_splits, split_counts, stratum


def _items(n: int) -> list[tuple[int, str, int]]:
    return [(i, "DE-NRW" if i % 2 else "NL", i % 4) for i in range(1, n + 1)]


def test_frozen_test_stays_test() -> None:
    cfg = SplitConfig(verify_count=5)
    a = assign_splits(_items(200), cfg, frozen_test=[1, 2, 3])
    assert all(a[i] == "test" for i in (1, 2, 3))
    counts = split_counts(a)
    assert counts["verify"] == 5
    assert sum(counts.values()) == 200
    assert 0.6 < counts["train"] / 200 < 0.75


def test_deterministic_and_stable_under_growth() -> None:
    cfg = SplitConfig(verify_count=0)
    a = assign_splits(_items(100), cfg)
    b = assign_splits(_items(100), cfg)
    assert a == b
    frozen = [i for i, s in a.items() if s == "test"]
    grown = assign_splits(_items(150), cfg, frozen_test=frozen)
    assert all(grown[i] == "test" for i in frozen)


def test_stratum_buckets() -> None:
    assert stratum("X", 0) == "X|0"
    assert stratum("X", 2) == "X|1-2"
    assert stratum("X", 7) == "X|3+"
