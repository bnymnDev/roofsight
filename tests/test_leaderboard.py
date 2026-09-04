import json
from pathlib import Path

from roofsight.eval.leaderboard import collect, regenerate
from roofsight.run import start_run


def _eval_run(root: Path, model: str, ap: float, with_latency: bool) -> None:
    run_dir, _ = start_run(
        "eval", {"model": model, "params_m": 3.0}, "v0.1", 1, root, run_id=f"r-{model}"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"mask_ap": ap, "small_obstacle_recall": 0.5, "boundary_f": 0.7})
    )
    if with_latency:
        (run_dir / "latency.json").write_text(
            json.dumps([{"backend": "onnxruntime-cpu", "median_ms": 88.0}])
        )


def test_leaderboard(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _eval_run(runs, "a", 0.4, True)
    _eval_run(runs, "b", 0.6, False)
    start_run("train", {"model": "c"}, "v0.1", 1, runs, run_id="r-train")  # no metrics → ignored
    (runs / "junk").mkdir()
    rows = collect(runs)
    assert [r.model for r in rows] == ["b", "a"]
    assert rows[1].cpu_ms == 88.0 and rows[0].cpu_ms is None
    md, tex = tmp_path / "lb.md", tmp_path / "res.tex"
    regenerate(runs, md, tex)
    text = md.read_text()
    assert "do not edit by hand" in text and "| b |" in text and "60.0" in text
    assert r"\begin{tabular}" in tex.read_text()


def test_empty(tmp_path: Path) -> None:
    md, tex = tmp_path / "lb.md", tmp_path / "res.tex"
    assert regenerate(tmp_path / "none", md, tex) == []
    assert "no runs yet" in md.read_text()
