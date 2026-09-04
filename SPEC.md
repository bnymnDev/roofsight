# SPEC — roofsight v0.1

## Goal
From a single ground-level smartphone photo of a house, segment roof planes and rooftop obstacles, and (with ARKit pose) estimate pitch and azimuth of each roof plane — on device, in real time, accurate enough for a first PV layout on site without drone or satellite data.

Public deliverables:
1. **RoofSight dataset** — ground-view roof/obstacle segmentation, COCO format, CC-BY-SA.
2. **Benchmark** — fixed splits, `roofsight eval`, public leaderboard.
3. **Models** — fine-tuned RF-DETR-Seg (Nano/Small), PyTorch + ONNX + Core ML.
4. **RoofGeometry** Swift package — pitch/azimuth/edge geometry from mask + device pose.
5. **Preprint** — dataset paper with baselines and ground-view vs. satellite obstacle recall.

## Non-goals (v0.1)
- Satellite/aerial imagery (only used as reference for the recall study, never as input)
- Yield calculation, module layout, shading simulation — Voltaris does that
- Multi-view / photogrammetry reconstruction (v0.2 candidate)
- Training SAM 3 or any foundation model

## Categories (stable ids)
| id | name | notes |
|----|------|-------|
| 1 | roof_plane | one instance per visible plane; gable, hip, shed, flat |
| 2 | chimney | |
| 3 | dormer | whole dormer body incl. its roof |
| 4 | skylight | roof windows (Velux-type), solar tubes |
| 5 | antenna | TV/sat dishes, aerials |
| 6 | vent | pipes, stacks, ventilation hoods |
| 7 | snow_guard | snow rails/hooks |
| 8 | existing_pv | already installed modules / solar thermal |
| 9 | tree_occlusion | vegetation covering roof area in the image |
| 10 | roof_edge | polyline as thin mask: eaves, ridge, hip, valley, verge; attribute `edge_type` |

Instance masks for all. `roof_plane` and `roof_edge` additionally carry attributes used by geometry: `edge_type ∈ {eave, ridge, hip, valley, verge}`.

## Data sources
- **Mapillary** street-level images, CC-BY-SA 4.0, via API v4 with bbox queries over residential areas (start: NRW, then DE/NL/AT for roof style diversity). Filter: `camera_type=perspective`, quality score, exclude 360°.
- **Own photos**: phone captures with ARKit metadata sidecar (`*.arkit.json`: intrinsics, pose, gravity, heading, depth map if LiDAR). These are the only images with ground-truth geometry.
- Target v0.1: 500 images; v0.2: 2 000; v1.0: 5 000+. Aim for ≥ 30 % images with ≥ 1 obstacle instance.

Pipeline: download → dedupe (perceptual hash) → anonymize (faces/plates, `egoblur` or `deface`) → roof presence filter (SAM 3 prompt "roof", keep if mask ≥ 5 % of image) → auto-label → human review → split.

## Labeling (SAM 3)
- Text prompts per category in `configs/labeling/prompts.yaml`; several synonyms per category, e.g. `chimney: ["chimney", "brick chimney stack"]`.
- Post-processing: NMS across synonyms, min area per category, `roof_plane` split by `roof_edge` polylines where SAM 3 merges adjacent planes.
- Export to FiftyOne for review; reviewer accepts/edits/deletes; every annotation records `provenance ∈ {auto, auto_edited, manual}`.
- Review target: every image in test split fully reviewed; train split ≥ 50 % reviewed in v0.1, 100 % in v1.0.
- SAM 3 stays in `labeling/`. Weights are downloaded by the user, license noted in docs.

## Splits
Stratified by region and by obstacle count. `test` is frozen at v0.1 and only ever grows by appending. `verify` = 50 images used only for export verification.

## Models
Primary: **RF-DETR-Seg Nano and Small**, fine-tuned from Roboflow pretrained checkpoints, 640 px. Reason: Apache-2.0, NMS-free, Core ML friendly.
Baselines for the paper:
- SAM 3 zero-shot with the labeling prompts (upper bound / what you get for free)
- YOLO26-seg n/s (reported, not shipped — AGPL)
- Mask2Former-Swin-T (accuracy reference)

Training: `configs/train/*.yaml` (epochs, lr, aug). Augmentations: color jitter, random crop keeping ≥ 60 % of roof, horizontal flip, slight perspective (roofs are seen from below; no vertical flip). Log to a local `runs/` directory + optional W&B.

## Metrics (`roofsight eval`)
- Mask AP and AP50 per category and overall (COCO)
- Obstacle recall @ IoU 0.5 for small objects (< 32² px) — the number that matters for PV
- Roof-plane boundary F-score (1 px tolerance scaled to 640)
- Latency: PyTorch (CUDA), ONNX Runtime (CPU), Core ML on iPhone (reported from `RoofGeometryBench` app target, JSON copied into `runs/`)
- Geometry (own-photo subset only): pitch MAE (°), azimuth MAE (°) vs. ARKit-derived ground truth

Leaderboard columns: model, params, mask AP, small-obstacle recall, boundary F, CPU ms, iPhone ms, dataset version, run id.

## Export
`roofsight export coreml`: PyTorch → ONNX (opset 17, static 640×640) → Core ML `mlpackage` (ML Program, fp16, Neural Engine). Verify: run both on `verify` split, require mean mask IoU ≥ 0.98 and box IoU ≥ 0.98. Publish as GitHub release asset with SHA256 and `model_card.md`.

## RoofGeometry (Swift package)
Input: mask/polygon per roof plane and edges (from the Core ML output), camera intrinsics, camera pose (ARKit `simd_float4x4`), gravity vector, true-north heading, optional depth map.

Output per roof plane:
```swift
struct RoofPlaneEstimate {
  let id: Int
  let pitchDegrees: Double
  let azimuthDegrees: Double        // 0 = N, 90 = E
  let confidence: Double
  let eaveLine: Line3D?
  let ridgeLine: Line3D?
}
```
Method v0.1:
1. Eave and ridge lines from `roof_edge` masks → 2D line fits.
2. Back-project with intrinsics; eave is assumed horizontal in the world (gravity constraint) → recovers its 3D direction.
3. Plane normal from eave direction and the vanishing direction of the plane's slope edges (verge/hip) with gravity; pitch = angle to horizontal, azimuth from heading.
4. If LiDAR depth is available: fit plane directly to depth samples inside the mask, use step 3 only as fallback/consistency check.
Pure Swift, no ARKit types below the boundary layer. Unit tests with synthetic roofs (known pitch/azimuth rendered to 2D).

## Ground-view vs. satellite recall study (paper)
For the own-photo subset (addresses known), pull satellite roof segmentation via Google Solar API or an off-the-shelf aerial model; count obstacles found from ground vs. from above per category. Hypothesis: ground view finds more skylights, vents, antennas on street-facing planes; satellite finds more on rear planes. Report per-category recall and the union.

## Docs
- README: what/why, one image with predicted masks over a photo, quick start (download dataset + model, run on an image), leaderboard snippet, Core ML integration snippet, citation.
- `docs/dataset.md` (versions, categories, licensing, how to contribute images), `docs/benchmark.md`, `docs/geometry.md`, `docs/labeling.md`, `docs/model_card.md`.

## Milestones
- **M0 (week 1–2):** repo scaffold, Mapillary downloader, anonymizer, SAM 3 labeling pipeline with prompts, FiftyOne review, COCO writer. 100 images end to end.
- **M1 (week 3):** dataset v0.1 = 500 images, test split fully reviewed. `roofsight eval` with golden tests. SAM 3 zero-shot baseline numbers.
- **M2 (week 4–5):** RF-DETR-Seg N/S fine-tune, first leaderboard, ONNX + Core ML export with verification, model release v0.1.
- **M3 (week 6–7):** RoofGeometry Swift package with synthetic tests, `RoofGeometryBench` iOS target for latency + demo, Voltaris integration (private).
- **M4 (week 8–10):** dataset v0.2 (2 000 images), satellite recall study, paper draft, arXiv preprint, launch (HN, r/MachineLearning, r/solar, LinkedIn, PV-Fachforen).

## Open questions (record in docs/decisions.md)
- Does `roof_edge` as thin mask train well with RF-DETR, or use a separate line-detection head / post-hoc edge extraction from `roof_plane` masks? Decide after M2 numbers.
- Mapillary coverage of residential streets in NRW may be thin; if < 300 usable images, add own-photo campaign early (weekend walk, 200 houses).
- Google Solar API cost and ToS for the recall study — check before M4; fallback is an aerial model on open orthophotos (NRW opengeodata).
