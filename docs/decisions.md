# Decisions

Why things are the way they are. Newest at the bottom.

## RF-DETR-Seg as the primary model

Apache-2.0, NMS-free (one less thing to port to Core ML), Roboflow ships pretrained
segmentation checkpoints at Nano/Small sizes that fit a phone. YOLO26-seg is faster to try but
AGPL, so it is reported as a baseline and never shipped.

## COCO JSON with extra fields, not a new format

Every viewer, `pycocotools` and `rfdetr` read it unchanged. Our fields (`source`, `license`,
`attribution`, `provenance`, `edge_type`, `split`) ride along and are validated by pydantic
models on our side.

## `roof_edge` as a thin mask (open)

The alternative is a separate line-detection head or post-hoc edge extraction from
`roof_plane` masks. Thin masks let the same model produce everything the geometry needs; the
question is whether RF-DETR trains well on them. Decide after the M2 numbers. If not, edges
come from the plane-mask contours split at corners.

## Small-obstacle recall does not consume predictions

AP-style matching punishes a single prediction covering two adjacent skylights. For a PV
planner that prediction is fine: the area is blocked either way. The metric asks whether every
small obstacle was noticed, not whether instances were separated.

## Plane splitting extends edges to lines

SAM 3 edge masks stop short of the plane border, so cutting along the raw mask leaves the two
planes connected through a few pixels. The line fitted through the edge mask, rasterized
across the whole image, cuts cleanly. Curved edges do not exist on the roofs we target.

## Gravity, not a full pose, inside the geometry package

The estimator needs two directions in the camera frame: down and north. That keeps the Swift
package free of ARKit and testable with a synthetic camera, and it means any pose source works
(ARKit, ARCore, EXIF + compass) as long as it can give those two vectors.

## Hip plan angle is 45°

True for regular hip roofs, wrong for irregular ones. A verge is preferred whenever visible
(its 90° is exact); hips are the fallback with the same confidence penalty as a short edge.
Multi-view would remove the assumption; that is v0.2.

## Mapillary coverage risk (open)

Residential streets in NRW may be thin on Mapillary. If the first build yields < 300 usable
images, the own-photo campaign (a weekend walk, 200 houses) moves from M3 to M1.

## Google Solar API for the recall study (open)

Cost and terms of service need checking before M4. Fallback: an off-the-shelf aerial model on
NRW's open orthophotos (opengeodata.nrw.de).

## No CPU stand-in for the SAM 3 roof filter

The first v0.1 build ran on a machine without a GPU, so three cheap replacements for the SAM 3
roof-presence filter were tried on 16 real Mapillary frames from the NRW boxes, hand-labelled
roof / no roof:

| Candidate | Result |
|---|---|
| CLIP ViT-B/32 zero-shot, full image or upper crops, several prompt sets | near random: clear roofs scored < 0.06, a hedge scored 0.98 |
| SegFormer-B0 ADE20K, building + house pixel fraction | noise barriers along a motorway 0.34, a house filling a third of the frame 0.003 |
| Grounding DINO tiny, prompt "a roof of a house" | 23 s per image on CPU and boxes covering the whole frame on a motorway |

Street-level frames are dominated by road, sky and trees; the roof is small, high and often
partly hidden. Nothing lighter than a promptable segmenter separated the cases. Rather than
ship a filter that deletes usable images, v0.1 keeps every downloaded frame and the roof
decision is made in the FiftyOne review: reviewers tag frames without a usable roof, the tags
become a scores file, and `roofsight data filter --backend file` drops them. When a GPU is
available, `--backend sam3` does the same automatically.

## Roof filter threshold 1.5 % for Mapillary frames, not the 5 % of the SPEC

The SPEC's "keep if the roof mask covers ≥ 5 % of the image" was written with phone photos in
mind: an installer stands in front of the house and the roof fills a good part of the frame.
Mapillary frames are dashcam shots at 2048 px from the middle of the road; the same roof is
far away and small. SAM 3 over all 1 418 frames of build 1 gave this distribution of the
roof-mask fraction (union of all roof instances):

| threshold | 0.5 % | 1 % | 1.5 % | 2 % | 3 % | 5 % |
|---|---|---|---|---|---|---|
| frames kept | 934 | 724 | 560 | 440 | 293 | 93 |

267 frames had no roof at all. At 5 % the build would miss the 500-image target by a factor of
five; at 1.5 % it keeps 560 with a margin for the review to drop more. Frames at 1.5–2 % show
whole houses with clearly delineated roofs, big enough for `roof_plane` and the larger
obstacles (chimney, dormer, existing PV), too small for vents and snow guards. Those categories
will come mainly from the own-photo subset, where the 5 % rule applies unchanged. Per-image
scores stay in `datasets/v0.1/roof_stats.json` so the threshold can be revisited without
re-running SAM 3.

## Auto-labels committed gzipped until the DVC remote exists

The rule is that datasets go through DVC. The first labeling pass ran on an ephemeral machine
without an S3/R2 remote, and 21 870 SAM 3 annotations over 560 images (13.5 MB JSON, 2–3 MB
gzipped) are too much work to lose. So `datasets/v0.1/annotations.json.gz` is in git next to
the manifest, as a stopgap with three properties: it is small, it is the *auto* layer only
(provenance `auto`, unreviewed), and it moves to DVC the moment a remote is configured.
Reviewed annotations will not be committed to git.
