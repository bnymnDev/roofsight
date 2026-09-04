"""``configs/labeling/prompts.yaml``: text prompts per category, several synonyms each."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from roofsight.categories import CATEGORIES, category_by_name


class CategoryPrompts(BaseModel):
    prompts: list[str] = Field(min_length=1)
    min_area_px: int = 0
    score_threshold: float = 0.3


class PromptConfig(BaseModel):
    version: str
    nms_iou: float = 0.7
    categories: dict[str, CategoryPrompts]

    @model_validator(mode="after")
    def _known_categories(self) -> PromptConfig:
        for name in self.categories:
            category_by_name(name)  # raises on unknown
        missing = [c.name for c in CATEGORIES if c.name not in self.categories]
        if missing:
            raise ValueError(f"prompts missing for categories: {missing}")
        return self

    @classmethod
    def load(cls, path: Path) -> PromptConfig:
        with path.open(encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))

    def flat(self) -> list[tuple[int, str]]:
        """``(category_id, prompt)`` pairs in config order."""
        return [
            (category_by_name(name).id, p)
            for name, cp in self.categories.items()
            for p in cp.prompts
        ]
