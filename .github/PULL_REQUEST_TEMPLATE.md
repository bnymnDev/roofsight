## What

<!-- One or two sentences: what changes and why. Link the issue if there is one. -->

## Checklist

- [ ] Branched from and targeting `develop` (`main` only moves by merging `develop`)
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy` pass
- [ ] `uv run pytest` passes on CPU with the tiny fixtures
- [ ] `swift test` passes in `swift/RoofGeometry` if the Swift package changed
- [ ] Metric changes come with a golden test in `tests/test_metrics.py`
- [ ] Category or prompt changes bump the dataset version and add a changelog entry in `docs/dataset.md`
- [ ] `docs/leaderboard.md` and `paper/results.tex` were regenerated with `uv run roofsight leaderboard`, not edited by hand
- [ ] No datasets, weights or images committed; every new image record has `source`, `license` and `attribution`
- [ ] `CHANGELOG.md` updated if the change is user-visible
