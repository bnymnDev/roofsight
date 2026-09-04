import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from roofsight import __version__
from roofsight.cli import app
from roofsight.eval.predictions import write_predictions
from roofsight.export.io import write_instances
from roofsight.export.verify import Instance
from roofsight.run import start_run
from tests.test_metrics import perfect

runner = CliRunner()


def test_version() -> None:
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0 and __version__ in r.output


def test_data_validate_and_split(dataset_dir: Path) -> None:
    r = runner.invoke(app, ["data", "validate", str(dataset_dir)])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["data", "split", str(dataset_dir)])
    assert r.exit_code == 0 and "test" in r.output


def test_eval_and_leaderboard(dataset_dir: Path, dataset, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    runs = tmp_path / "runs"
    run_dir, _ = start_run(
        "train", {"model": "rfdetr-seg-nano", "params_m": 3.0}, "v-test", 1, runs, run_id="t1"
    )
    write_predictions(perfect(dataset), run_dir / "predictions-test.json")
    r = runner.invoke(
        app,
        [
            "eval",
            "--run",
            str(run_dir),
            "--split",
            "test",
            "--dataset",
            str(dataset_dir),
            "--runs-root",
            str(runs),
        ],
    )
    assert r.exit_code == 0, r.output
    evals = [d for d in runs.iterdir() if d.name.endswith("-eval")]
    assert len(evals) == 1
    m = json.loads((evals[0] / "metrics.json").read_text())
    assert m["mask_ap"] > 0.99
    md, tex = tmp_path / "lb.md", tmp_path / "r.tex"
    r = runner.invoke(
        app, ["leaderboard", "--runs-root", str(runs), "--md-out", str(md), "--tex-out", str(tex)]
    )
    assert r.exit_code == 0 and "rfdetr-seg-nano" in md.read_text()


def test_export_verify(tmp_path: Path) -> None:
    m = np.zeros((64, 64), dtype=bool)
    m[4:40, 4:40] = True
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write_instances([[Instance(1, 0.9, m)]], a)
    write_instances([[Instance(1, 0.9, np.roll(m, 3, axis=1))]], b)
    assert runner.invoke(app, ["export", "verify", str(a), str(a)]).exit_code == 0
    r = runner.invoke(app, ["export", "verify", str(a), str(b), "--out", str(tmp_path / "v.json")])
    assert r.exit_code == 1
    assert json.loads((tmp_path / "v.json").read_text())["ok"] is False
