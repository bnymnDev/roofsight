# Model card

Filled in by `roofsight export coreml` for each release; the numbers below are placeholders
until the first release run exists. No hand-typed metrics.

## RoofSight RF-DETR-Seg Nano / Small

| | Nano | Small |
|---|---|---|
| Architecture | RF-DETR-Seg (Roboflow), DINOv2 backbone, NMS-free | same |
| Input | RGB 640×640, scale 1/255 | same |
| Output | per-instance class logits, boxes, masks | same |
| Precision | fp16 ML Program, Neural Engine | same |
| Training data | RoofSight v0.1 train split | same |
| Mask AP / small-obstacle recall / boundary F | see [leaderboard](leaderboard.md) | see leaderboard |
| Release | `RoofSight-nano-v0.1.mlpackage.zip` + SHA256 | `RoofSight-small-v0.1.mlpackage.zip` + SHA256 |

## Intended use

First on-site PV layout from a single ground-level photo. Not a substitute for a structural
survey, and not a yield estimate. The consuming app decides what to do with a low-confidence
plane; the model does not.

## Limitations

- Trained on residential roofs in DE/NL/AT; expect degraded results on flat commercial roofs
  and non-European roof styles.
- Obstacles smaller than roughly 6 px at 640 px are below what the annotation resolution
  supports.
- Rear planes are invisible from the street. Ground view finds street-facing obstacles that
  satellites miss, and misses the rest; the paper quantifies both.

## Licenses

Weights: Apache-2.0. Training data: CC-BY-SA 4.0. The SAM 3 model used for auto-labeling is
not part of the shipped model.
