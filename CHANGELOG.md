# Changelog

## Unreleased

- Repository scaffold: `roofsight` package with `data`, `labeling`, `train`, `eval`, `export`
  and the `roofsight` CLI
- Fixed category list (ids 1–10) and COCO models with license, attribution, provenance and
  `edge_type` validation
- Dataset build: Mapillary v4 client, perceptual-hash dedupe, anonymizer backends, roof
  presence filter hook, stratified splits with a frozen test set, `data validate`
- SAM 3 labeling pipeline: prompt config, synonym NMS, minimum area, plane splitting along
  fitted edge lines, FiftyOne round trip with provenance
- Metrics: mask AP / AP50, small-obstacle recall, boundary F, pitch/azimuth MAE, latency
  records; leaderboard generator for `docs/leaderboard.md` and `paper/results.tex`
- Export: ONNX → Core ML path, backend verification at mask/box IoU ≥ 0.98, model card
- `RoofGeometry` Swift package: pitch and azimuth from eave/verge/hip lines with a gravity
  constraint, LiDAR plane fit, ARKit boundary, synthetic-roof tests
- Docs: dataset, labeling, benchmark, geometry, model card, decisions
