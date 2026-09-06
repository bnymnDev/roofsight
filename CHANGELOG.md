# Changelog

## Unreleased

- Dataset v0.1 build 1: 1 418 anonymized Mapillary frames, 560 after the SAM 3 roof filter (1.5 %) from twelve NRW suburbs, manifest
  `datasets/v0.1/images.json` in git, `roofsight data fetch` restores the images by id
- Mapillary client queries every bbox as a grid with retries; roof filter is a separate step
  (`data filter`, backends `sam3` and `file`)
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
