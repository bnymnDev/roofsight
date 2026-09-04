"""``run.json``: the reproducibility record every train/eval/export run writes.

Results without a ``run.json`` are not accepted into the leaderboard.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

RUN_FILE = "run.json"


class RunRecord(BaseModel):
    run_id: str
    kind: str  # train | eval | export | label | data
    git_sha: str
    git_dirty: bool
    config_hash: str
    config: dict[str, Any] = Field(default_factory=dict)
    dataset_version: str
    seed: int
    hardware: dict[str, str] = Field(default_factory=dict)
    created_at: str
    roofsight_version: str

    def write(self, run_dir: Path) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / RUN_FILE
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path


def read_run(run_dir: Path) -> RunRecord:
    path = run_dir / RUN_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: results without run.json are not accepted")
    return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))


def config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def git_state(cwd: Path | None = None) -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout
        return sha, bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True


def hardware_info() -> dict[str, str]:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import torch

        info["torch"] = str(torch.__version__)
        info["cuda"] = torch.version.cuda or "none"
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = "not installed"
    return info


def new_run_id(kind: str, now: datetime | None = None) -> str:
    ts = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{kind}"


def start_run(
    kind: str,
    config: dict[str, Any],
    dataset_version: str,
    seed: int,
    runs_root: Path = Path("runs"),
    run_id: str | None = None,
) -> tuple[Path, RunRecord]:
    from roofsight import __version__

    sha, dirty = git_state()
    rid = run_id or new_run_id(kind)
    record = RunRecord(
        run_id=rid,
        kind=kind,
        git_sha=sha,
        git_dirty=dirty,
        config_hash=config_hash(config),
        config=config,
        dataset_version=dataset_version,
        seed=seed,
        hardware=hardware_info(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        roofsight_version=__version__,
    )
    run_dir = runs_root / rid
    record.write(run_dir)
    return run_dir, record
