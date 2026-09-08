# Labeling with SAM 3

Auto-labels come from SAM 3 with text prompts. Humans review them in FiftyOne. Nothing from
SAM 3 is shipped; the `sam3` package is imported only inside `roofsight/labeling/`.

## Weights and implementations

SAM 3 weights are gated: request access at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
(Meta approves by hand), then log in with `hf auth login` or set `HF_TOKEN`. The license is not
Apache-2.0; nothing from SAM 3 is shipped. Weights are never committed.

Two implementations behind one interface, chosen by `--checkpoint`:

| `--checkpoint` | Implementation | Runs on | Notes |
|---|---|---|---|
| `facebook/sam3` (default) or a local HF directory | `transformers` (`Sam3Model`) | CPU or CUDA | 3.4 GB safetensors, pure PyTorch; slow on CPU but works without `triton` |
| `path/to/sam3.pt` | Meta's `sam3` package | CUDA only | imports `triton`; faster |

The image is encoded once and all prompts of a category set reuse the encoding.

## Prompts

`configs/labeling/prompts.yaml` has several synonyms per category, a score threshold and a
minimum area in pixels:

```yaml
chimney:
  prompts: ["chimney", "brick chimney stack", "chimney pipe on roof"]
  min_area_px: 60
  score_threshold: 0.3
```

A prompt change bumps the prompt config version and the dataset version.

## Post-processing

1. **NMS across synonyms.** The same chimney found by three prompts is one chimney: per
   category, masks with IoU ≥ `nms_iou` collapse to the highest-scoring one.
2. **Minimum area.** Per category, in pixels.
3. **Plane splitting.** SAM 3 likes to merge adjacent planes into one "roof". Every
   `roof_edge` mask is extended to the straight line fitted through it and each plane is cut
   along the lines of the edges that touch it; pieces above the plane's minimum area become
   separate planes. Edge masks rarely reach the plane border, which is why the line is
   extended, and only touching edges cut so the ridge of one house never slices the roof next
   door.

4. **Position.** A category may forbid mask centroids below a fraction of the image height
   (`max_centroid_y`). "roof" also finds car roofs and bonnets at the bottom of a street photo;
   `roof_plane` uses 0.75.
5. **On the roof.** Obstacles must have ≥ 50 % of their mask inside the union of roof planes
   dilated by 2 % of the image side (chimneys stick out above the ridge); `tree_occlusion`
   ≥ 20 %. "roof window" also finds façade windows, "tree" finds every tree on the street.
   Without a roof plane in the image nothing is on a roof.

`roof_edge` gets its `edge_type` from the prompt that found it ("ridge line of roof" →
`ridge`).

The first pass over real frames, before rules 4 and 5 and with the plane cut applied to every
plane an extended line crossed, produced 126 instances per image; with them it is about 60,
and what remains sits on roofs. Labeling runs at ~95 s per image on 4 CPU cores with all 35
prompts (one image encoding, 35 cheap decodes), so `roofsight label` saves progress every
5 images and resumes when rerun.

## Review

```sh
# the frozen test split only, and only what SAM 3 was reasonably sure about
uv run roofsight review datasets/v0.1 export --split test --min-score 0.5
uv run fiftyone app launch roofsight-review   # http://localhost:5151; Windows: set PYTHONUTF8=1
# accept / edit / delete in the app, then:
uv run roofsight review datasets/v0.1 import
uv run roofsight review datasets/v0.1 stats
```

Pass the dataset name to `fiftyone app launch`; without it the app opens empty and its start-up
query fails with a 500.

An export narrowed by `--split` or `--min-score` writes a scope file next to the dataset, and
`import` replaces exactly that scope: other splits and instances below the score are carried
over unchanged. Reviewing a subset can therefore never delete the rest of the dataset.

### How much is there to review

The v0.1 auto-labels average 42 instances per test frame, most of them low-scoring. Raising the
export threshold cuts the pile before a human sees it:

| `--min-score` | instances in the test split | per frame |
|---|---|---|
| none (0.3) | 6461 | 42 |
| 0.4 | 4458 | 29 |
| 0.5 | 3235 | 21 |
| 0.6 | 2432 | 16 |

0.5 is the recommended first pass: it halves the work and the frames still keep their planes.
`roof_edge` is the exception, its median score is 0.36, so review edges in a second pass with
`--min-score 0.3` once the obstacles are clean.

Import compares every annotation against the auto labels and sets `provenance`:
unchanged → `auto`, changed mask, category or edge type → `auto_edited`, new → `manual`.
Deleted annotations are simply gone.

Review targets: every image in `test` is fully reviewed; `train` ≥ 50 % in v0.1, 100 % in v1.0.
