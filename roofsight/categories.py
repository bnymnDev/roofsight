"""The fixed category list.

Category ids are stable and are never renumbered. A new category is appended with the next
free id and triggers a dataset version bump (see docs/dataset.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

EdgeType = Literal["eave", "ridge", "hip", "valley", "verge"]
EDGE_TYPES: Final[tuple[EdgeType, ...]] = ("eave", "ridge", "hip", "valley", "verge")

SUPERCATEGORY_ROOF: Final = "roof"
SUPERCATEGORY_OBSTACLE: Final = "obstacle"
SUPERCATEGORY_OCCLUSION: Final = "occlusion"
SUPERCATEGORY_EDGE: Final = "edge"


@dataclass(frozen=True, slots=True)
class Category:
    """One category of the RoofSight dataset, in COCO terms."""

    id: int
    name: str
    supercategory: str
    notes: str = ""

    def to_coco(self) -> dict[str, object]:
        return {"id": self.id, "name": self.name, "supercategory": self.supercategory}


CATEGORIES: Final[tuple[Category, ...]] = (
    Category(1, "roof_plane", SUPERCATEGORY_ROOF, "one instance per visible plane"),
    Category(2, "chimney", SUPERCATEGORY_OBSTACLE),
    Category(3, "dormer", SUPERCATEGORY_OBSTACLE, "whole dormer body incl. its roof"),
    Category(4, "skylight", SUPERCATEGORY_OBSTACLE, "roof windows, solar tubes"),
    Category(5, "antenna", SUPERCATEGORY_OBSTACLE, "TV/sat dishes, aerials"),
    Category(6, "vent", SUPERCATEGORY_OBSTACLE, "pipes, stacks, ventilation hoods"),
    Category(7, "snow_guard", SUPERCATEGORY_OBSTACLE, "snow rails/hooks"),
    Category(8, "existing_pv", SUPERCATEGORY_OBSTACLE, "installed modules / solar thermal"),
    Category(9, "tree_occlusion", SUPERCATEGORY_OCCLUSION, "vegetation covering roof area"),
    Category(10, "roof_edge", SUPERCATEGORY_EDGE, "polyline as thin mask; attribute edge_type"),
)

_BY_ID: Final[dict[int, Category]] = {c.id: c for c in CATEGORIES}
_BY_NAME: Final[dict[str, Category]] = {c.name: c for c in CATEGORIES}

OBSTACLE_IDS: Final[frozenset[int]] = frozenset(
    c.id for c in CATEGORIES if c.supercategory == SUPERCATEGORY_OBSTACLE
)
ROOF_PLANE_ID: Final = 1
ROOF_EDGE_ID: Final = 10


def category_by_id(category_id: int) -> Category:
    try:
        return _BY_ID[category_id]
    except KeyError as e:
        raise KeyError(f"unknown category id {category_id}") from e


def category_by_name(name: str) -> Category:
    try:
        return _BY_NAME[name]
    except KeyError as e:
        raise KeyError(f"unknown category name {name!r}") from e


def coco_categories() -> list[dict[str, object]]:
    return [c.to_coco() for c in CATEGORIES]
