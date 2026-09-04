"""Latency measurement. iPhone numbers come from the ``RoofGeometryBench`` app as JSON."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel


class LatencyRecord(BaseModel):
    backend: str  # torch-cuda | onnxruntime-cpu | coreml-iphone
    device: str
    input_size: int = 640
    warmup: int
    iterations: int
    mean_ms: float
    median_ms: float
    p95_ms: float


def measure(
    fn: Callable[[], object], backend: str, device: str, warmup: int = 10, iterations: int = 50
) -> LatencyRecord:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return LatencyRecord(
        backend=backend,
        device=device,
        warmup=warmup,
        iterations=iterations,
        mean_ms=statistics.fmean(times),
        median_ms=statistics.median(times),
        p95_ms=times[min(len(times) - 1, int(0.95 * len(times)))],
    )


def read_latency(path: Path) -> list[LatencyRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [LatencyRecord.model_validate(d) for d in data]


def write_latency(records: list[LatencyRecord], path: Path) -> None:
    path.write_text(json.dumps([r.model_dump() for r in records], indent=2), encoding="utf-8")
