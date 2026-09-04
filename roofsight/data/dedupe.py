"""Perceptual-hash dedupe. Near duplicates (consecutive Mapillary frames) collapse to one."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import imagehash
from PIL import Image


def phash(path: Path, hash_size: int = 16) -> str:
    with Image.open(path) as im:
        return str(imagehash.phash(im.convert("RGB"), hash_size=hash_size))


def hamming(a: str, b: str) -> int:
    return int(imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b))


def dedupe(hashes: Iterable[tuple[str, str]], max_hamming: int = 6) -> list[str]:
    """Keep the first of each near-duplicate group.

    ``hashes`` are ``(key, hex_hash)`` pairs in priority order. Returns the kept keys, in order.
    O(n²) is fine for the dataset sizes in SPEC (≤ 5 000).
    """
    kept: list[tuple[str, str]] = []
    for key, h in hashes:
        if all(hamming(h, kh) > max_hamming for _, kh in kept):
            kept.append((key, h))
    return [k for k, _ in kept]
