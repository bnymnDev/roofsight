from roofsight.categories import CATEGORIES, OBSTACLE_IDS, category_by_id, category_by_name


def test_ids_are_stable_and_dense() -> None:
    assert [c.id for c in CATEGORIES] == list(range(1, 11))
    assert category_by_id(1).name == "roof_plane"
    assert category_by_id(10).name == "roof_edge"
    assert category_by_name("chimney").id == 2


def test_obstacles() -> None:
    assert set(OBSTACLE_IDS) == {2, 3, 4, 5, 6, 7, 8}
