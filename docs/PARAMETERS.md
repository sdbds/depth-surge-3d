# Parameters Reference

Run `python depth_surge_3d.py --help` for the complete CLI generated from the
current parser. The options below are the controls that materially affect depth
and stereo output.

## Input And Output

- `input_video`: source video path.
- `-o, --output-dir`: output root. Default: `./output`.
- `--resume DIRECTORY`: resume a processing directory.
- `-s, --start` and `-e, --end`: optional time range.
- `-f, --format`: `side_by_side` or `over_under`.
- `--vr-resolution`: named per-eye resolution, `auto`, or `custom`.
- `--target-fps`: output frame rate. Omit it to preserve the source rate.
- `--no-audio`: omit source audio.
- `--no-intermediates`: remove restartable working files after validated output.

## Stereo Rendering

- `--stereo-strength`: full near-to-far horizontal disparity as a percentage of
  render width. Default: `2.0`; valid range: `0.0` to `5.0`.
- `--convergence`: canonical relative disparity placed at zero parallax.
  Default: `0.5`; valid range: `0.0` to `1.0`.
- `--occlusion-fill`: `background` fills horizontal disocclusion gaps from
  farther visible pixels; `none` leaves unresolved subpixel lanes black.
- `--stereo-io-workers`: bounded decode/encode worker count. Default is derived
  from the host CPU count; valid range: `1` to `8`.

The renderer first resizes canonical disparity to the target frame size, then
computes pixel disparity using the target width:

```text
d = (relative_disparity - convergence) * target_width * stereo_strength / 100
left_target_x  = source_x + d / 2
right_target_x = source_x - d / 2
```

Near surfaces therefore move right in the left eye and left in the right eye.
Both eyes share the same depth key; only the target-coordinate sign changes.
Stereo strength scales this geometry only. Antialiasing quality is fixed at 16
horizontal samples per output pixel and is not reduced or increased by the
strength value.

With `--occlusion-fill none`, a partially covered silhouette pixel averages its
valid lanes with black unresolved lanes because the renderer exposes RGB rather
than alpha. A dark contour at a disocclusion is therefore intentional in this
mode. Use `background` when the bounded farther-surface extension is preferred.

## Scene Scaling

- `--scene-detection` / `--no-scene-detection`: enable deterministic scene
  segmentation. Default: enabled.
- `--scene-cut-threshold`: luma-histogram cut threshold. Default: `0.55`.
- `--min-scene-frames`: minimum candidate segment length. Default: `8`.

Raw model output is converted by representation, then each final scene uses
fixed 2nd/98th percentile bounds. Canonical values are reproducible across
chunks and resume boundaries. Empty or flat scenes use `0.5`.

## Raw Depth Storage

- `--raw-storage-dtype`: `auto`, `float16`, or `float32`. The first inference
  chunk fixes the directory dtype and records it in metadata.
- `--migrate-legacy`: `archive` (default) or `delete` for generated stages from
  an older on-disk schema. Deletion is always explicit.

The pipeline stores native-resolution model output in `02_depth_raw`; it does
not persist an upsampled copy. Canonical `uint16` maps live in
`03_disparity_maps` and carry fingerprints for the model, raw stage, scene
manifest, and bounds.

## Model And Projection

- `--depth-model-version`: `v2`, `v3`, or `see_through`.
- `--model`: checkpoint path, model name, or repository as appropriate.
- `--metric`: select metric output where the adapter supports it.
- `--device`: `auto`, `cuda`, or `cpu`.
- `--fisheye-projection`: `stereographic`, `equidistant`, `equisolid`, or
  `orthogonal`.
- `--fisheye-fov`: projection field of view in degrees.
- `--no-distortion`: keep rectilinear stereo images.
- `--crop-factor` and `--fisheye-crop-factor`: post-render crop controls.

## Temporal Post-Processing

- `--temporal-postprocessor`: `off` (default for new jobs) or `vdpp`.

VDPP is an experimental depth-only pass after canonicalization. It works with
all current depth backends, but V2 usually benefits less because V2 already has
model-native shot-aware temporal inference. Generating missing VDPP artifacts
requires CUDA, downloads the pinned 116,485,370-byte v1.0 checkpoint on first
use, adds one uint16 PNG per selected frame, and adds processing time. A fully
validated stabilized cache can be rendered without CUDA.

The released checkpoint fixes window 32, overlap 4, stride 28, downsize mode,
and FP32 precision. These are not user controls. On resume, omitting the option
keeps the saved mode; explicitly passing `off` or `vdpp` changes the requested
artifact and invalidates only affected downstream stages.

## Tuning

Start with the defaults. Reduce `--stereo-strength` when depth edges feel
uncomfortable or expose too much hidden background. Move `--convergence`
toward the canonical value of the subject that should sit on the display plane.
Changing scene detection affects scaling boundaries, not stereo geometry.
