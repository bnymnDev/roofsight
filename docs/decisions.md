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
