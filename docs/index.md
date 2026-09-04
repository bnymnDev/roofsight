# roofsight

Ground-view roof and rooftop-obstacle segmentation for on-site PV planning. One smartphone
photo from the street, and you get roof planes, chimneys, skylights, vents and the rest as
instance masks, plus pitch and azimuth per plane when the phone's pose is known.

- [Dataset](dataset.md): sources, categories, versions, licensing, contributing images
- [Labeling](labeling.md): SAM 3 prompts, post-processing, review with FiftyOne, provenance
- [Benchmark](benchmark.md): metrics, splits, how to submit a run
- [Leaderboard](leaderboard.md): generated from `runs/`
- [Geometry](geometry.md): pitch and azimuth from masks and device pose (the Swift package)
- [Model card](model_card.md): the shipped Core ML models
- [Decisions](decisions.md): why things are the way they are
