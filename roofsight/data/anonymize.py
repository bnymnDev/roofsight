"""Faces and license plates are blurred before an image enters the dataset.

Backends are external tools chosen in the data config. ``deface`` (pip) handles faces;
``egoblur`` (Meta, separate install) handles faces and plates. A backend that is not
installed is a hard error unless the config says otherwise, so an un-anonymized image can
never slip through by accident. ``roofsight data validate`` checks the ``anonymized`` flag.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

Backend = Callable[[Path, Path, float], None]


def _deface(src: Path, dst: Path, threshold: float) -> None:
    exe = shutil.which("deface")
    if exe is None:
        raise RuntimeError("deface is not installed: uv pip install deface")
    subprocess.run(
        [exe, str(src), "--output", str(dst), "--thresh", str(threshold), "--replacewith", "blur"],
        check=True,
        capture_output=True,
    )


def _egoblur(src: Path, dst: Path, threshold: float) -> None:
    exe = shutil.which("egoblur")
    if exe is None:
        raise RuntimeError("egoblur is not installed; see docs/dataset.md")
    subprocess.run(
        [
            exe,
            "--input",
            str(src),
            "--output",
            str(dst),
            "--face-threshold",
            str(threshold),
            "--lp-threshold",
            str(threshold),
        ],
        check=True,
        capture_output=True,
    )


def _none(src: Path, dst: Path, threshold: float) -> None:
    shutil.copyfile(src, dst)


BACKENDS: dict[str, Backend] = {"deface": _deface, "egoblur": _egoblur, "none": _none}


def anonymize(src: Path, dst: Path, backend: str = "deface", threshold: float = 0.2) -> Path:
    try:
        fn = BACKENDS[backend]
    except KeyError as e:
        raise ValueError(f"unknown anonymizer backend {backend!r}") from e
    dst.parent.mkdir(parents=True, exist_ok=True)
    fn(src, dst, threshold)
    return dst
