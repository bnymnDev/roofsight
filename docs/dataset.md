# The RoofSight dataset

Ground-level photos of residential houses with instance masks for roof planes, rooftop
obstacles, vegetation occlusion and roof edges. COCO format. CC-BY-SA 4.0.

## Versions

| Version | Images | Regions | Status | Notes |
|---|---|---|---|---|
| v0.1 | 500 (target) | DE-NRW | in progress | test split frozen at release; own-photo subset with ARKit pose |
| v0.2 | 2 000 (target) | DE, NL, AT | planned | more roof styles; satellite recall study subset |
| v1.0 | 5 000+ (target) | DE, NL, AT | planned | 100 % of train reviewed |

`test` is frozen at v0.1 and only ever grows by appending. `verify` is 50 images used only
for export verification (`roofsight export verify`) and never for training or reporting.

## Categories

Ids are stable and never renumbered. A new category gets the next free id and bumps the
dataset version.

| id | name | supercategory | notes |
|----|------|---------------|-------|
| 1 | `roof_plane` | roof | one instance per visible plane; gable, hip, shed, flat |
| 2 | `chimney` | obstacle | |
| 3 | `dormer` | obstacle | whole dormer body including its roof |
| 4 | `skylight` | obstacle | roof windows (Velux-type), solar tubes |
| 5 | `antenna` | obstacle | TV/sat dishes, aerials |
| 6 | `vent` | obstacle | pipes, stacks, ventilation hoods |
| 7 | `snow_guard` | obstacle | snow rails/hooks |
| 8 | `existing_pv` | obstacle | installed modules, solar thermal |
| 9 | `tree_occlusion` | occlusion | vegetation covering roof area in the image |
| 10 | `roof_edge` | edge | polyline as thin mask; attribute `edge_type` ∈ {eave, ridge, hip, valley, verge} |

Every annotation carries `provenance` ∈ {`auto`, `auto_edited`, `manual`}.

## Sources and licensing

| Source | License | Attribution | Geometry ground truth |
|---|---|---|---|
| Mapillary street-level images (API v4, bbox queries over residential areas; `camera_type=perspective`, quality score ≥ 0.6, no 360°) | CC-BY-SA 4.0 | `© <creator>, Mapillary, CC BY-SA 4.0, image <id>` | no |
| Own photos with an ARKit sidecar (`<name>.arkit.json`: intrinsics, pose, gravity, heading, depth map when LiDAR is present) | CC-BY-SA 4.0 | `RoofSight contributors` | yes |

Every image record carries `source`, `license` and `attribution`. `roofsight data validate`
fails on a record without them, on an image that was not anonymized, and on a `roof_edge`
without `edge_type`.

## Pipeline

```
download ──▶ dedupe ──▶ anonymize ──▶ roof filter ──▶ auto-label ──▶ review ──▶ split
Mapillary    phash      deface /      SAM 3 "roof"    SAM 3         FiftyOne   stratified by
API v4       ≤ 6 bits   egoblur       ≥ 5 % of image  prompts                  region × obstacles
```

```sh
export MAPILLARY_TOKEN=MLY|...
uv run roofsight data build --config configs/data/v0.1.yaml --sam3-checkpoint weights/sam3.pt
uv run roofsight label --prompts configs/labeling/prompts.yaml --in datasets/v0.1 --out datasets/v0.1 --checkpoint weights/sam3.pt
uv run roofsight review datasets/v0.1 export      # then: fiftyone app launch
uv run roofsight review datasets/v0.1 import
uv run roofsight data validate datasets/v0.1
```

Datasets and weights are never committed. `datasets/` holds DVC pointers only; the remote is
S3/R2. `dvc pull` fetches a version.

## Contributing images

Own photos are the only images with geometry ground truth, so they are the most valuable
contribution. What we need:

1. A phone photo of a house from the street, roof clearly visible, taken with the capture
   view of the `RoofGeometryBench` app (it writes the ARKit sidecar).
2. Nothing identifying: the anonymizer blurs faces and plates, but do not photograph people
   on purpose, and skip house numbers if you can.
3. Consent that the image is published under CC-BY-SA 4.0.

Drop the `.jpg` + `.arkit.json` pairs into `datasets/own/` and open a pull request against the
DVC pointer, or send a link in an issue.

## Changelog

- **v0.1** (in progress): initial category list (ids 1–10), NRW residential streets, own-photo
  subset, frozen test split.
