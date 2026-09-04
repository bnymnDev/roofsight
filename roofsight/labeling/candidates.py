"""A raw detection before post-processing. Shared by the SAM 3 adapter and the tests."""

from __future__ import annotations

from dataclasses import dataclass

from roofsight.masks import BoolMask


@dataclass(slots=True)
class Candidate:
    category_id: int
    prompt: str
    score: float
    mask: BoolMask

    @property
    def area(self) -> int:
        return int(self.mask.sum())
