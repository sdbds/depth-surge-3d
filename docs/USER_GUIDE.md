# Stereo Quality User Guide

## Choose A Mode

Use `quality` for final offline renders where thin silhouettes and disocclusion
edges matter. Use `fast` for previews or when a prior Fast render must remain
byte-compatible. Quality is selected automatically only for a new omitted-mode
job whose effective renderer device is CUDA; an omitted CPU job uses Fast.

```bash
python depth_surge_3d.py input.mp4 \
  --stereo-render-mode quality \
  --occlusion-fill background \
  --occlusion-fill-max-px 8
```

The Web UI exposes the same two modes. Its Quality fill limit is an integer from
1 through 32, expressed at 1080p and scaled to output height. Start at 8. A
larger value permits wider local disocclusions to use the bounded repair planner;
it is not a global background-smear distance.

Quality on CPU is valid but slow. The CLI and Web UI warn about that selection.
No runtime error or memory-pressure path silently changes Quality to Fast.

## Inspect Diagnostics

Quality jobs write committed diagnostics below `04_stereo_diagnostics/`:

```text
04_stereo_diagnostics/
  metadata.json
  <frame>/
    left_coverage.png
    right_coverage.png
    left_repair.png
    right_repair.png
    stats.json
    manifest.json
```

The exact filenames are recorded by each manifest. Coverage values count valid
subpixel lanes. Repair masks distinguish local, exemplar, and global fallback
repairs. Frame statistics include hole ratios, repair counts, and maximum
connected widths. A frame is reusable only after its manifest and root summary
are durable and validate against the current inputs and algorithm identity.

## Resume Rules

A schema-5 job retains its saved `stereo_render_mode` and
`occlusion_fill_max_px`. An explicit CLI or Web resume override wins and causes
incompatible downstream stereo artifacts to be rebuilt. Historical schema 1 to
4 jobs migrate to Fast with an 8 px limit.

With `keep_intermediates=true`, diagnostics remain available after final video
publication. With pruning enabled, the authenticated metadata transitions to
`payload_pruned`; deleted payloads are never presented as reusable diagnostics.

## Edge Review

Compare synchronized Fast and Quality crops at curls, feathers, narrow barrels,
ribbons, and other foreground/background depth transitions. Reject a Quality
render if it introduces foreground erosion, temporal flicker, black unresolved
holes, or donors crossing a foreground depth boundary. Use the repair masks to
separate a depth-edge problem from a later encoding artifact.
