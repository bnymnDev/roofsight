from pathlib import Path

import pytest

from roofsight.run import config_hash, read_run, start_run


def test_run_record(tmp_path: Path) -> None:
    run_dir, rec = start_run("train", {"lr": 1e-4}, "v0.1", 42, tmp_path)
    back = read_run(run_dir)
    assert back == rec
    assert back.config_hash == config_hash({"lr": 1e-4})
    assert back.run_id.endswith("-train") and back.seed == 42
    assert "python" in back.hardware


def test_missing_run_json(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not accepted"):
        read_run(tmp_path)
