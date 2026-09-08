# The RoofSight dataset

Ground-level photos of residential houses with instance masks for roof planes, rooftop
obstacles, vegetation occlusion and roof edges. COCO format. CC-BY-SA 4.0.

## Versions

| Version | Images | Regions | Status | Notes |
|---|---|---|---|---|
| v0.1 | 560 after roof filter (1 418 downloaded), auto-labeled, 500 reviewed (target) | DE-NRW, 12 suburbs | auto-labeled, review pending | manifest in git, test ids frozen (213); own-photo subset with ARKit pose to come |
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
| Wikimedia Commons (CirrusSearch, `deepcat:` category trees, 1280 px thumbnails) | CC-BY-SA 4.0/3.0/2.x, CC-BY, CC0, public domain — recorded per image | `<author>, Wikimedia Commons, <license>, <file page url>` | no |
| Own photos with an ARKit sidecar (`<name>.arkit.json`: intrinsics, pose, gravity, heading, depth map when LiDAR is present) | CC-BY-SA 4.0 | `RoofSight contributors` | yes |

Every image record carries `source`, `license` and `attribution`. `roofsight data validate`
fails on a record without them, on an image that was not anonymized, and on a `roof_edge`
without `edge_type`.

### What Commons is and is not good for

Commons has the framing a PV planner sees: the whole building, shot from the pavement, roof
large in the frame, at 3000 px. Two limits decide how it is used.

- **The building stock skews to the notable.** People upload listed buildings, Gründerzeit
  villas and half-timbered houses, not the 1975 house next door. Targeted searches for
  `Einfamilienhaus`, `Satteldach` or `Neubaugebiet` return villas, historic town centres and
  construction sites. Commons therefore brings roof and obstacle variety, not typical PV
  candidates; those come from the own-photo subset.
- **Deep categories wander.** `deepcat:"Dormer windows"` returns Oxford colleges and São Paulo.
  Only categories with a region in their name are used, and the SAM 3 roof filter is what
  removes what still slips through.

Sizes: 21 920 files under "Houses in North Rhine-Westphalia", 17 018 Lower Saxony, 56 236
Netherlands, 35 543 Austria. Commons has no snow-guard category, so `snow_guard` stays
unrepresented until own photos cover it.

Two things about talking to Wikimedia. They reject some HTTP clients by TLS fingerprint with a
403 and their robot-policy notice; `httpx` is among them, so the Commons client uses the
standard library's `urllib`. And their media hosts throttle by client address: a data-centre
address gets 429 with a ten-minute `Retry-After` even at one request every three seconds,
while an ordinary connection pulls thousands of files without one. **Run the Commons
collection from the machine you work on, not from a server.** The client caps the wait it
honours and raises `CommonsThrottledError` after four throttled requests in a row rather than
crawling for days.

## Pipeline

```
download ──▶ dedupe ──▶ anonymize ──▶ roof filter ──▶ auto-label ──▶ review ──▶ split
Mapillary    phash      deface /      SAM 3 "roof"    SAM 3         FiftyOne   stratified by
API v4       ≤ 6 bits   egoblur       ≥ 5 % of image  prompts                  region × obstacles
```

```sh
export MAPILLARY_TOKEN=MLY|...
uv run roofsight data build --config configs/data/v0.1.yaml --no-roof-filter     # download, dedupe, anonymize, split
uv run roofsight data filter datasets/v0.1 --config configs/data/v0.1.yaml \
    --backend sam3 --sam3-checkpoint weights/sam3.pt                                # or --backend file --scores review.json
uv run roofsight label --prompts configs/labeling/prompts.yaml --in datasets/v0.1 --out datasets/v0.1 --checkpoint weights/sam3.pt
uv run roofsight review datasets/v0.1 export      # then: fiftyone app launch
uv run roofsight review datasets/v0.1 import
uv run roofsight data validate datasets/v0.1
```

The roof filter is a separate step so a build without a GPU still runs; see
[decisions](decisions.md#no-cpu-stand-in-for-the-sam-3-roof-filter) for why nothing lighter
than SAM 3 is used. The bbox endpoint of Mapillary does not paginate and fails on large boxes,
so every box is queried as a 4 × 4 raster with retries; a cell that keeps failing is skipped.

Images and weights are never committed. What is in git is the **manifest**
`datasets/<version>/images.json`: every image record with its Mapillary id, license,
attribution, perceptual hash and split. That makes a build reproducible without hosting
the pixels:

```sh
uv run roofsight data fetch datasets/v0.1 --config configs/data/v0.1.yaml   # re-download by id + anonymize
```

The token is checked before the first download; a bad token (a trailing `.` from a paste is
the usual culprit) aborts with exit code 2 and never looks like missing images. Images that
Mapillary has since removed are reported and kept in the manifest; `--prune-missing` drops
them.
Annotations and weights go through DVC (S3/R2 remote, to be set up); `dvc pull` fetches them.

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

- **v0.1 build 1** (2026-09-05): 1 440 frames from 12 NRW suburbs (Köln, Bonn, Düsseldorf,
  Münster, Aachen, Essen, Dortmund, Paderborn; 120 per box), 22 near-duplicates removed,
  1 418 anonymized with `deface`; splits 1013 / 142 / 213 / 50 (train / val / test / verify);
  213 test ids frozen in `configs/data/frozen_test_v0.1.json`. No roof filter yet (no GPU in the
  build environment); roughly half the frames show no usable roof and will be removed in review.
- **v0.1 build 1, roof filter** (2026-09-06): SAM 3 (`transformers` port, CPU, 22 s per frame)
  scored every frame with the prompt "roof"; threshold 1.5 % of the image (see
  [decisions](decisions.md#roof-filter-threshold-15-for-mapillary-frames-not-the-5-of-the-spec)),
  560 frames kept, 858 dropped; splits re-assigned with the frozen test ids preserved.
- **v0.1 build 1, auto-labels** (2026-09-06): SAM 3 with the 35 prompts of `prompts.yaml` v0.1.1
  over all 560 frames on CPU (79 s per frame, 12.3 h); 21 870 instances, 39 per frame, 91 % of
  frames with at least one obstacle, 20 frames without any instance. All provenance `auto`.
  Committed gzipped until the DVC remote exists (see decisions). Known gaps for review:
  `snow_guard` never fires, `roof_edge` is 88 % `ridge` (the eave/verge prompts rarely win).
- **v0.1** (planned): initial category list (ids 1–10), reviewed roof presence, SAM 3 auto-labels,
  own-photo subset.
