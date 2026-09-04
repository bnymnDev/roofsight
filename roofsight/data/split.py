"""Stratified splits by region and obstacle count. ``test`` is frozen and only ever grows."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping

from roofsight.coco import SPLITS, Split
from roofsight.data.config import SplitConfig


def obstacle_bucket(n_obstacles: int) -> str:
    if n_obstacles == 0:
        return "0"
    if n_obstacles <= 2:
        return "1-2"
    return "3+"


def stratum(region: str, n_obstacles: int) -> str:
    return f"{region}|{obstacle_bucket(n_obstacles)}"


def assign_splits(
    items: Iterable[tuple[int, str, int]],
    config: SplitConfig,
    frozen_test: Iterable[int] = (),
) -> dict[int, Split]:
    """Assign a split per image id.

    ``items`` are ``(image_id, region, n_obstacles)``. Image ids in ``frozen_test`` always land in
    ``test``. Within each stratum the remaining images are shuffled with the config seed and cut
    by the configured fractions, so adding images never moves an existing image out of ``test``.
    """
    frozen = set(frozen_test)
    result: dict[int, Split] = {i: "test" for i in frozen}
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for image_id, region, n_obs in items:
        if image_id in frozen:
            continue
        by_stratum[stratum(region, n_obs)].append(image_id)

    fractions: Mapping[Split, float] = {
        "train": config.train,
        "val": config.val,
        "test": config.test,
        "verify": config.verify,
    }
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {total}")

    verify_budget = config.verify_count
    for name in sorted(by_stratum):
        ids = sorted(by_stratum[name])
        random.Random(f"{config.seed}:{name}").shuffle(ids)
        n = len(ids)
        cuts: list[tuple[Split, int]] = []
        acc = 0
        for split in SPLITS:
            count = round(fractions[split] * n) if split != "verify" else 0
            cuts.append((split, count))
            acc += count
        # whatever rounding left over goes to train
        cuts[0] = ("train", cuts[0][1] + (n - acc))
        pos = 0
        for split, count in cuts:
            for image_id in ids[pos : pos + count]:
                result[image_id] = split
            pos += count

    # verify: a fixed small set carved out of train, spread over strata
    train_ids = sorted(i for i, s in result.items() if s == "train")
    random.Random(f"{config.seed}:verify").shuffle(train_ids)
    for image_id in train_ids[:verify_budget]:
        result[image_id] = "verify"
    return result


def split_counts(assignment: Mapping[int, Split]) -> dict[Split, int]:
    counts: dict[Split, int] = dict.fromkeys(SPLITS, 0)
    for s in assignment.values():
        counts[s] += 1
    return counts
