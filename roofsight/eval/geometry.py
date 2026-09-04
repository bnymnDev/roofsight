"""Geometry metrics on the own-photo subset: pitch and azimuth MAE against ARKit ground truth.

Estimates come from the Swift package (``RoofGeometryBench`` writes ``geometry.json``); ground
truth from the ``*.arkit.json`` sidecars. Azimuth error is circular.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, TypeAdapter


class PlaneEstimate(BaseModel):
    image_id: int
    plane_id: int
    pitch_deg: float
    azimuth_deg: float
    confidence: float = 1.0


_adapter = TypeAdapter(list[PlaneEstimate])


def read_estimates(path: Path) -> list[PlaneEstimate]:
    return _adapter.validate_json(path.read_text(encoding="utf-8"))


def read_ground_truth(path: Path) -> list[PlaneEstimate]:
    return _adapter.validate_json(path.read_text(encoding="utf-8"))


def azimuth_error(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


@dataclass(slots=True)
class GeometryReport:
    pitch_mae: float
    azimuth_mae: float
    n_planes: int
    n_missing: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "pitch_mae_deg": self.pitch_mae,
            "azimuth_mae_deg": self.azimuth_mae,
            "n_planes": self.n_planes,
            "n_planes_missing": self.n_missing,
        }


def geometry_mae(truth: list[PlaneEstimate], estimates: list[PlaneEstimate]) -> GeometryReport:
    est = {(e.image_id, e.plane_id): e for e in estimates}
    pitch_errs: list[float] = []
    az_errs: list[float] = []
    missing = 0
    for t in truth:
        e = est.get((t.image_id, t.plane_id))
        if e is None:
            missing += 1
            continue
        pitch_errs.append(abs(t.pitch_deg - e.pitch_deg))
        az_errs.append(azimuth_error(t.azimuth_deg, e.azimuth_deg))
    n = len(pitch_errs)
    return GeometryReport(
        pitch_mae=sum(pitch_errs) / n if n else float("nan"),
        azimuth_mae=sum(az_errs) / n if n else float("nan"),
        n_planes=n,
        n_missing=missing,
    )


def write_report(report: GeometryReport, path: Path) -> None:
    path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
