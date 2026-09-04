# Labeling with SAM 3

Auto-labels come from SAM 3 with text prompts. Humans review them in FiftyOne. Nothing from
SAM 3 is shipped; the `sam3` package is imported only inside `roofsight/labeling/`.

## Weights

Download the SAM 3 checkpoint from Meta's release page and note its license (it is not
Apache-2.0). Pass the path with `--checkpoint`. Weights are never committed.

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
   `roof_edge` mask is extended to the straight line fitted through it and the plane is cut
   along all of them; connected pieces above 64 px become separate planes. Edge masks rarely
   reach the plane border, which is why the line is extended.

`roof_edge` gets its `edge_type` from the prompt that found it ("ridge line of roof" →
`ridge`).

## Review

```sh
uv run roofsight review datasets/v0.1 export
fiftyone app launch
# accept / edit / delete, then:
uv run roofsight review datasets/v0.1 import
uv run roofsight review datasets/v0.1 stats
```

Import compares every annotation against the auto labels and sets `provenance`:
unchanged → `auto`, changed mask, category or edge type → `auto_edited`, new → `manual`.
Deleted annotations are simply gone.

Review targets: every image in `test` is fully reviewed; `train` ≥ 50 % in v0.1, 100 % in v1.0.
