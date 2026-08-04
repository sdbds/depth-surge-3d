# Depth Contract and Torch DIBR Design

## Status

Approved in conversation on 2026-08-04. This is an intentionally breaking
change. Existing stereo settings, cached depth maps, and resumable jobs are not
compatible with the new renderer.

## Problem

The current pipeline destroys depth semantics by normalizing every frame to
`0..1`, then treats those values as physical distance by multiplying them by a
fixed ten metres. It converts that fabricated distance to disparity with camera
baseline and focal-length parameters that were never calibrated from the input.

Stereo images are then generated with inverse `cv2.remap`, clipped coordinates,
and reflected borders. That operation has no visibility ordering and produces
no real disocclusion mask. The later hole filler infers holes from black pixels,
which can modify legitimate image content while missing actual occlusions.

## Goals

- Preserve the meaning and scale class of every estimator output.
- Produce one canonical rendering representation: relative disparity where
  `0.0` is far and `1.0` is near.
- Keep disparity scaling stable across frames and outer processing chunks.
- Reset temporal scaling at scene boundaries.
- Generate both eyes with depth-aware forward splatting and an explicit
  visibility mask.
- Run the same Torch renderer on CUDA and CPU, with CUDA as the primary path.
- Give preview, batch processing, resume, and file-backed processing one render
  implementation and one setting schema.
- Bound peak GPU memory for 4K frames.

## Non-goals

- Backward compatibility with `baseline`, `focal_length`,
  `hole_fill_quality`, old depth PNG metadata, or old resumable jobs.
- Learned view synthesis or diffusion inpainting.
- Physical camera reconstruction from unknown monocular input.
- A second NumPy rendering implementation.
- Replacing the existing frame-on-disk pipeline in this change.

## Chosen Architecture

The implementation has three explicit boundaries:

1. Estimators return raw `DepthBatch` objects instead of normalized arrays.
2. `DepthProcessor` converts those batches to temporally stable canonical
   relative disparity before cache or intermediate storage.
3. A shared Torch stereo renderer consumes only canonical relative disparity.

This keeps model-specific semantics out of rendering and keeps rendering details
out of the estimators.

## Depth Output Contract

Create `inference/depth/types.py` with:

- `DepthRepresentation`: `RELATIVE_DEPTH`, `METRIC_DEPTH`, or
  `INVERSE_DEPTH`.
- `DepthBatch.values`: finite or non-finite `float32` array shaped `[N,H,W]`.
- `DepthBatch.representation`: required enum value.
- `DepthBatch.confidence`: optional `float32` array with the same shape.

Estimator adapters declare representation explicitly:

- Video Depth Anything relative models: `INVERSE_DEPTH`.
- Video Depth Anything metric models: `METRIC_DEPTH`.
- Depth Anything 3 outputs: `METRIC_DEPTH` for metric models and
  `RELATIVE_DEPTH` otherwise.
- See-Through Marigold: `RELATIVE_DEPTH`.

Estimators may resize output but must not apply per-frame min/max
normalization. Metric values remain in metres until canonicalization.

## Canonical Disparity

Create a stateful `TemporalDisparityScaler` in
`processing/frames/depth_normalizer.py`.

For each raw batch it will:

1. Mark NaN, infinity, and non-positive physical depth invalid.
2. Convert depth representations to a disparity score. Depth values use a safe
   reciprocal; inverse-depth values pass through unchanged.
3. Detect scene boundaries from normalized 32-bin luma histograms of the source
   frames. A Bhattacharyya distance greater than `0.55`, subject to a minimum
   scene length of eight frames, starts a scene.
4. Compute robust low/high disparity bounds from the valid values in each
   scene segment using the 2nd and 98th percentiles.
5. Move the active bounds toward the segment target with an EMA alpha of `0.1`
   for each frame. The state survives outer chunks and resets before the first
   frame of a new scene.
6. Map disparity to `[0,1]`, clip outliers, set invalid pixels to zero, and
   return `float32` canonical values with `0=far` and `1=near`.

A flat or invalid frame produces zero disparity and never divides by zero. The
normalizer is owned by one `DepthProcessor` run, not by individual estimator
calls or multiprocessing workers.

Canonical intermediate PNG files use 16-bit unsigned encoding. Their metadata
must include:

- `schema_version`
- `representation: relative_disparity`
- `near_value: 1.0`
- `far_value: 0.0`
- percentile and temporal-scaling algorithm versions

## User-facing Stereo Controls

Remove `baseline`, `focal_length`, and `hole_fill_quality` from constants, CLI,
Web payloads, templates, projector APIs, persisted settings, and validation.

Add:

- `stereo_strength`: near-to-far binocular disparity span as a percentage of
  frame width. Default `2.0`; valid range `0.0..5.0`.
- `convergence`: canonical disparity value placed on the zero-parallax plane.
  Default `0.5`; valid range `0.0..1.0`.
- `occlusion_fill`: `none` or `inpaint`. Default `inpaint`.

For canonical disparity `r`, frame width `w`, strength `s`, and convergence
`c`, total binocular disparity is:

`d = (r - c) * w * (s / 100)`

The left target coordinate uses `x - d/2`; the right target uses `x + d/2`.

## Torch Forward Splat

Create `rendering/forward_splat.py` and `rendering/stereo_renderer.py`.

The low-level splat accepts batched Torch tensors and performs horizontal
bilinear forward projection:

1. Project each source pixel to the two neighboring integer target columns.
2. Discard out-of-frame contributions instead of clipping or reflecting them.
3. Build a target z-buffer with `scatter_reduce(amax)` over canonical disparity,
   where larger values are nearer.
4. Keep only contributions within `1e-4` canonical disparity of the nearest
   target depth.
5. Accumulate visible colour and bilinear weights with `scatter_add`.
6. Divide colour by accumulated weight and mark zero-weight pixels invalid.

Coordinate, depth, colour accumulation, and z-buffer tensors remain float32,
including under mixed-precision model inference. The final image is clamped and
converted back to its input dtype.

Rendering is split into complete row bands. Horizontal projection never crosses
rows, so row tiling bounds temporary memory without changing geometry. The
renderer chooses the largest band whose estimated temporary storage is no more
than the smaller of 256 MiB and 25 percent of currently free CUDA memory. It
falls back to 64 rows when memory information is unavailable.

`StereoRenderResult` contains left image, right image, left valid mask, right
valid mask, and corresponding occlusion masks. Masks are derived from splat
weights, never image colour.

`occlusion_fill=inpaint` transfers rendered frames and masks to host memory and
uses a new `inpaint_occlusions(image, mask)` helper with those explicit masks.
Black source pixels that have valid splat weights remain untouched. `none`
leaves holes black for debugging and export. Delete the old
`depth_to_disparity`, `create_shifted_image`, and colour-derived automatic hole
mask helpers after all callers migrate.

## Execution Model

CUDA rendering runs in the main processing process. The existing four-process
stereo pool must not be used with CUDA because every worker would initialize a
separate CUDA context. File-backed processing loads one frame and disparity map,
renders them, writes both eyes, and releases temporary tensors before advancing.

CPU mode invokes the same Torch operations on `torch.device("cpu")`. It is the
reference path for exact unit tests; CUDA parity tests use numerical tolerances.

`StereoPairGenerator` receives its device from `StereoProjector`. Preview and
full-video paths both call `StereoRenderer.render_pair`.

## Cache and Resume

Increment the depth cache schema and include the canonicalization algorithm in
the cache key. Cache loading rejects missing or older schema metadata with a
normal cache miss. Resume validation rejects jobs containing removed settings or
non-canonical depth metadata and explains that the job must restart.

No heuristic migration is permitted because old `0..1` files do not record
whether their model emitted depth or inverse depth.

## Error Handling

- Invalid estimator shapes raise a descriptive `ValueError` at the estimator
  boundary.
- Shape mismatches between image and disparity are resized once by the shared
  renderer using linear interpolation, then revalidated.
- CUDA out-of-memory retries the same frame once with half the row-band size.
  A second failure reports the attempted frame size and band size; it does not
  silently switch devices.
- Non-finite render tensors raise before image encoding.
- Unsupported or removed setting names fail validation instead of being ignored.

## Verification

Tests must be written before implementation and cover:

- representation-specific conversion direction and preservation of metric raw
  values at estimator boundaries;
- robust normalization under outliers, stable bounds across chunks, scene reset,
  flat maps, and non-finite inputs;
- constant-disparity translation and left/right symmetry;
- bilinear weights and out-of-frame holes;
- near-surface z-buffer victory when foreground and background collide;
- explicit masks that preserve valid black source pixels;
- CPU/CUDA parity when CUDA is available;
- bounded row tiling equivalence to an untiled small render;
- projector, file-backed processing, preview, CLI, Web, cache, and resume use of
  the new setting and metadata schema;
- absence of removed settings from code, templates, examples, and documentation.

The complete existing test suite must remain green. Add a non-gating benchmark
that reports 1080p and 4K render time plus peak CUDA memory. Performance numbers
are recorded as observations, not portable pass/fail thresholds.

## Acceptance Criteria

- No estimator performs per-frame min/max normalization.
- No production stereo path calls the old depth-to-disparity or inverse-remap
  functions.
- A synthetic foreground collision always renders the nearer colour.
- A valid black source region is never selected for filling solely by colour.
- The same canonical sequence has no scale discontinuity at an outer chunk
  boundary beyond the configured EMA evolution.
- Old cache and resume state are rejected clearly.
- All tests pass on CPU; CUDA-specific tests pass on the available NVIDIA GPU.
