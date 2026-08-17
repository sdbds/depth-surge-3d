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

- `--depth-model-version`: `v2`, `v3`, `see_through`, or `moge2`. Select
  MoGe-2 with `--depth-model-version moge2`.
- `--model-size {vits,vitb,vitl}`: MoGe-2 Small, Base, or Large. MoGe-2
  defaults to `vitb`.
- `--stereo-geometry-mode {relative,metric_camera}`: geometry interpretation.
  Default: `relative`. `metric_camera` is Experimental.
- `--virtual-baseline-mm`: finite `0` to `100`. Default: `63.0`. This is the
  separation between virtual capture cameras, not viewer IPD.
- `--metric-convergence-distance {auto|metres}`: `auto` or a finite distance
  from `0.1` to `1000` metres. Default: `auto`.
- `--max-disparity-percent`: finite `0` to `5`. Default: `2.0`. This caps total
  left-to-right disparity in retained final-output coordinates.
- `--model`: checkpoint path, model name, or repository as appropriate.
- `--metric`: select metric output where the adapter supports it.
- `--device`: `auto`, `cuda`, or `cpu`.
- `--fisheye-projection`: `stereographic`, `equidistant`, `equisolid`, or
  `orthogonal`.
- `--fisheye-fov`: projection field of view in degrees.
- `--no-distortion`: keep rectilinear stereo images.
- `--crop-factor` and `--fisheye-crop-factor`: post-render crop controls.

Relative geometry remains the default, and flat rectilinear SBS is the
main playback path. Experimental `metric_camera` requires MoGe-2 pinhole focal
data, `--format side_by_side`, `--no-distortion`, and canonical square source
SAR `1:1`. Missing or `N/A` SAR is normalized to `1:1`; malformed, explicit
non-square, or otherwise invalid SAR fails before model loading. Letterbox bars
inside a square-SAR image may still bias automatic convergence.

The automatic convergence distance is resolved once from clip-global valid
metric samples and persisted before metric stereo rendering or preview. The
crop-aware projection converts pre-crop source disparity to retained-output
coordinates, clamps total disparity there, then uses the same center crop and
axis-aligned resize rule as the rest of the pipeline. Changing final output
width does not change the projection fraction.

MoGe-2 performs per-frame depth and focal estimation. Temporal stability on video is not guaranteed; depth or focal drift may be visible across frames.

The mode does not establish calibrated physical scale, physically correct
reconstruction, improved stereo quality, temporal stability, viewing comfort
or safety, or superiority over relative mode. The adapter level is fixed at
`9`, is report-only, and is not exposed as a public setting or CLI flag.

### Pinned MoGe-2 Variants

| UI/setting | Parameters | Repository | Revision |
| --- | ---: | --- | --- |
| Small / `vits` | 35M | `Ruicheng/moge-2-vits-normal` | `679230677b4d282c6f304189a93e98e14f085902` |
| Base / `vitb` default | 104M | `Ruicheng/moge-2-vitb-normal` | `54ad3a693e61907ea4633d13dec6ee682fa09419` |
| Large / `vitl` | 326M | `Ruicheng/moge-2-vitl` | `39c4d5e957afe587e04eec59dc2bcc3be5ecd968` |

## Temporal Post-Processing

- `--temporal-postprocessor`: `off` (default for new jobs) or `vdpp`.

VDPP is an experimental depth-only pass after relative-disparity
canonicalization. It works with all current depth backends in `relative`
geometry mode, but V2 usually benefits less because V2 already has model-native
shot-aware temporal inference. It is not applied to `metric_camera` geometry.
Generating missing VDPP artifacts requires CUDA, downloads the pinned
116,485,370-byte v1.0 checkpoint on first use, adds one uint16 PNG per selected
frame, and adds processing time. A fully validated stabilized cache can be
rendered without CUDA.

The released checkpoint fixes window 32, overlap 4, stride 28, downsize mode,
and FP32 precision. These are not user controls. On resume, omitting the option
keeps the saved mode; explicitly passing `off` or `vdpp` changes the requested
artifact and invalidates only affected downstream stages.

## Tuning

Start with the defaults. Reduce `--stereo-strength` when depth edges feel
uncomfortable or expose too much hidden background. Move `--convergence`
toward the canonical value of the subject that should sit on the display plane.
Changing scene detection affects scaling boundaries, not stereo geometry.
