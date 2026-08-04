# Depth Contract and Torch DIBR Design

## Status

Revision 3, updated after external review on 2026-08-04. The user approved a
Torch/CUDA-first renderer and explicitly rejected backward compatibility in the
final state. Implementation is split into three independently verifiable
commits so contract, geometry, and persistence failures can be isolated.

## Problem

The current pipeline normalizes every depth frame independently to `0..1`, then
treats those values as physical distance by multiplying them by a fixed ten
metres. It applies camera baseline and focal-length parameters that were never
calibrated from the monocular input.

Stereo images are generated with inverse `cv2.remap`, clipped coordinates, and
reflected borders. That operation has no visibility ordering and produces no
real disocclusion mask. The hole filler then infers holes from black pixels,
which can modify legitimate source content while missing actual occlusions.

## Goals

- Preserve the representation class of every estimator output.
- Produce canonical relative disparity where `0.0` is far and `1.0` is near.
- Use one immutable set of robust bounds for every frame in a scene.
- Make clean runs, chunked runs, and resumed runs byte-for-byte reproducible on
  the same backend.
- Detect scenes before depth normalization and persist frame-to-scene mapping.
- Generate both eyes with depth-aware Torch forward splatting and explicit
  visibility masks.
- Keep the standard rectified-stereo sign: a near object has
  `u_left - u_right > 0`.
- Fill horizontal DIBR disocclusions on the GPU without a CPU inpainting
  bottleneck.
- Give preview and full processing one renderer and one setting schema.
- Bound temporary GPU memory for 4K rendering and bound the legacy in-memory
  depth API on host memory.

## Non-goals

- Final compatibility with `baseline`, `focal_length`, `hole_fill_quality`, old
  depth metadata, or old depth/downstream resume stages.
- Learned view synthesis or diffusion inpainting.
- Physical camera reconstruction from unknown monocular input.
- A second NumPy rendering implementation.
- Replacing the frame-on-disk video pipeline.

## Final Data Flow

```text
00_original_frames
  -> scene pre-pass
01_scene_data/scene_manifest.json (status=candidate)
  -> raw model inference with explicit representation
02_depth_raw/*.npz + metadata.json
  -> GLOBAL BARRIER: every raw frame is complete
01_scene_data/depth_samples.npz
  -> deterministic fixed-point scene merge and final bounds
01_scene_data/depth_bounds.json
01_scene_data/scene_manifest.json (status=final)
  -> pure canonicalization
03_disparity_maps/*.png + metadata.json
  -> Torch forward splat + GPU background fill
04_left_frames/*.png + 04_right_frames/*.png
```

The first three stages are restartable independently, but canonicalization may
not overlap raw inference. Every raw frame and every candidate scene sample set
must exist before deterministic scene merging and final bounds begin.
Canonicalization is then a pure function of one raw map, its representation,
the final scene manifest, and persisted final bounds.

## Depth Output Contract

Create `inference/depth/types.py` with:

- `DepthRepresentation`: `RELATIVE_DEPTH`, `METRIC_DEPTH`, or
  `INVERSE_DEPTH`.
- `DepthBatch.values`: a `float32` array shaped `[N,H,W]`. Non-finite values are
  allowed at this boundary and downstream code must mark them invalid.
- `DepthBatch.representation`: a required enum value.

There is no confidence field until a concrete consumer exists.

Estimator adapters declare representation explicitly:

- Video Depth Anything relative models: `INVERSE_DEPTH`.
- Video Depth Anything metric models: `METRIC_DEPTH`.
- Depth Anything 3 metric models: `METRIC_DEPTH`.
- Depth Anything 3 non-metric models: `RELATIVE_DEPTH`.
- See-Through Marigold: `RELATIVE_DEPTH`.

Estimators must return their native model-output resolution and must not resize
to source-frame dimensions or apply per-frame min/max normalization. Metric
values remain in metres until raw persistence. Native dimensions are recorded
in metadata; canonical disparity is resized exactly once at the render boundary.

Each adapter has a contract test that checks both its declared enum and the full
near/far direction through canonical conversion. These tests do not load large
models: they feed a two-region synthetic estimator result through the adapter
boundary and assert `canonical_near > canonical_far`.

## Scene Pre-pass

Create `processing/frames/scene_analyzer.py`. It runs once after frame extraction
and before depth inference, reads the persisted source frames, and writes an
atomic `01_scene_data/scene_manifest.json` containing:

- schema and algorithm versions;
- `status: candidate` or `status: final`;
- ordered frame names and scene IDs;
- candidate cut frame indexes;
- `scene_detection`, `scene_cut_threshold`, and `min_scene_frames` settings.

The detector uses normalized 32-bin luma histograms and Bhattacharyya distance.
Defaults are `scene_detection=true`, `scene_cut_threshold=0.55`, and
`min_scene_frames=8`. They are typed processing settings and persisted rather
than hidden constants. Disabling detection assigns every frame to scene zero.

Candidate cuts are provisional and the first manifest is always written with
`status: candidate`. Canonicalization accepts only `status: final`.

After all raw-depth samples exist, adjacent candidate scenes are merged when the
maximum difference between their low/high bounds is no more than 10 percent of
their combined disparity span. Merging is deterministic:

1. Preserve candidate scenes and their sample arrays in source-frame order.
2. Scan left to right. When two neighbors merge, concatenate their original
   sample arrays in that order, recompute 2nd/98th percentile bounds, replace
   the pair at the lower index, and compare that merged scene with its new right
   neighbor. When a pair does not merge, advance the cursor by one.
3. Finish the pass, then repeat complete left-to-right passes until a pass makes
   no merge. Every successful merge reduces scene count, so termination occurs
   in at most `candidate_scene_count - 1` passes.
4. Never resample frames after a merge.

Final bounds are calculated from each fixed-point sample union, not combined
from child percentiles. Write `depth_bounds.json` first. Then atomically replace
the manifest with final scene IDs, `status: final`, and the bounds-file
fingerprint. A crash before the final manifest replacement leaves a candidate
manifest, so resume reruns finalization rather than canonicalizing provisional
IDs. A final manifest with missing or mismatched bounds is also rejected and
finalized again.

The normalizer never reads RGB frames and never owns scene-detection state. It
consumes raw depth plus persisted scene IDs and bounds only.

## Raw Depth and Scene Bounds

`DepthProcessor` converts model output to float16 and writes each native-resolution
map atomically as a zlib-compressed `.npz` file in `02_depth_raw`. Float16 is a
storage encoding only: canonical conversion reads it into float32 before safe
reciprocal, sampling, percentile, or arithmetic. The directory's
`metadata.json` records representation, model fingerprint, frame count, native
dimensions, storage dtype, compression, and schema version.

Finite model values outside the float16 representable range fail raw persistence
with the offending model and value range; they are never silently clipped.
Non-finite model values remain permitted and canonicalize as invalid.

Bounds are deterministic and bounded-memory:

1. Convert selected raw values to an unscaled disparity score. Positive depth
   uses a safe reciprocal; inverse depth passes through.
2. For each candidate scene, select at most 32 frame indexes evenly across its
   duration.
3. From each selected frame, take a uniform grid of at most 64 by 64 valid
   pixels.
4. Persist every candidate scene's pooled float32 samples, selected frame
   indexes, and source-frame order in `01_scene_data/depth_samples.npz`. The
   maximum payload per candidate scene is 131,072 floats, or 512 KiB before
   compression.
5. Calculate provisional bounds for the merge guard, run the fixed-point merge,
   and calculate final 2nd/98th percentile bounds from final sample unions.
6. Persist only final bounds in `01_scene_data/depth_bounds.json`, including the
   sample-file fingerprint and algorithm version.

Empty or flat scenes receive equal bounds and canonicalize to zero disparity.
There is no EMA or other mutable scaling state.

## Pure Canonicalization

Create `processing/frames/depth_normalizer.py` with a stateless function:

```python
canonicalize_depth(
    values: np.ndarray,
    representation: DepthRepresentation,
    bounds: SceneDepthBounds,
) -> np.ndarray
```

It marks NaN, infinity, and non-positive physical depth invalid, converts to a
disparity score, maps the persisted scene bounds to `[0,1]`, clips outliers, and
sets invalid pixels to zero. Output is float32 with `0=far` and `1=near`.

Canonical maps are encoded as uint16 PNG in `03_disparity_maps`.
`DepthProcessor`, not the global cache, owns and atomically writes
`03_disparity_maps/metadata.json`. It contains:

- schema and canonicalization algorithm versions;
- `representation: relative_disparity`;
- `near_value: 1.0` and `far_value: 0.0`;
- source raw-depth metadata fingerprint;
- scene-manifest and bounds fingerprints;
- encoding scale `65535.0`.

Global cache save and restore copy or recreate the same required metadata. The
stereo stage reads only the local `03_disparity_maps/metadata.json`.

## Raw-depth Disk Budget and Retention

Before raw inference, estimate native output dimensions from the selected depth
resolution while preserving source aspect ratio. Refuse to start unless current
free space covers the expected peak:

- raw allowance: `frames * native_width * native_height * 2 * 1.25` bytes;
- canonical allowance: `frames * native_width * native_height * 2 * 1.10`
  bytes;
- with `keep_intermediates=true`, peak is the sum;
- otherwise peak is the larger allowance plus two frame payloads for atomic
  write overlap.

The estimate intentionally assumes no compression benefit. After the first raw
map reveals actual native dimensions, repeat the check with those dimensions
before processing the remaining frames. Failure reports required, available,
and output-path bytes before more model work is performed.

When `keep_intermediates=false`, each raw `.npz` is deleted only after its
canonical PNG has been atomically written, read back, and validated against
final metadata. If a crash leaves both files, resume validates the canonical
file and then removes its redundant raw file. `keep_intermediates=true` retains
raw maps for replay and inspection.

The global raw-depth barrier remains: retention lowers post-canonicalization and
final disk usage but cannot lower the peak required before final scene bounds
exist.

## User-facing Controls

The final commit removes `baseline`, `focal_length`, and `hole_fill_quality`
from constants, CLI, Web payloads, templates, projector APIs, examples,
persisted settings, and validation.

Add:

- `stereo_strength`: near-to-far binocular disparity span as a percentage of
  frame width. Default `2.0`; valid range `0.0..5.0`.
- `convergence`: canonical disparity placed on the zero-parallax plane. Default
  `0.5`; valid range `0.0..1.0`.
- `occlusion_fill`: `none` or `background`. Default `background`.
- Scene-analysis settings described above in the advanced settings section.

For canonical disparity `r`, frame width `w`, strength `s`, and convergence
`c`, total binocular disparity is:

`d = (r - c) * w * (s / 100)`

The corrected rectified-stereo projection is:

- `x_left = x + d/2`
- `x_right = x - d/2`

Therefore a near point (`r > c`) appears farther right in the left image than
in the right image, so `u_left - u_right > 0`.

## Torch Forward Splat

Create `rendering/forward_splat.py` and `rendering/stereo_renderer.py`.

The low-level splat accepts batched Torch tensors and performs horizontal
bilinear forward projection:

1. Project each source pixel to the two neighboring integer target columns.
2. Discard out-of-frame contributions instead of clipping or reflecting them.
3. Build a target z-buffer with `scatter_reduce(amax)` over signed pixel
   disparity, where a larger value represents a nearer surface.
4. Keep contributions whose signed pixel disparity is within `0.25` pixel of
   the nearest target contribution.
5. Accumulate visible colour and bilinear weights with `scatter_add`.
6. Divide colour by accumulated weight and mark zero-weight pixels invalid.

The visibility tolerance is defined in projected pixel units, not canonical
depth units, so its meaning is stable across scenes and uint16 round trips.
When `stereo_strength=0`, all samples have zero disparity and no depth-based
occlusion is applied.

Coordinates, pixel disparity, colour accumulation, and z-buffer remain float32,
including under mixed-precision model inference. The final image is clamped and
converted back to its input dtype.

Rendering is split into complete row bands. A fixed 256 MiB temporary-memory
budget and a conservative documented estimate of 160 bytes per source pixel
determine band height from frame width. The result does not depend on runtime
free-memory queries. CUDA out-of-memory retries the frame once with half the
calculated band height.

`StereoRenderResult` contains left image, right image, left/right valid masks,
left/right hole masks, and internal projected-disparity buffers. Masks derive
from accumulated splat weight, never image colour.

## GPU Background Fill

`occlusion_fill=background` fills horizontal DIBR gaps on the same Torch device:

1. For every row, propagate the nearest valid index from the left and right.
2. Gather both candidate colours and their projected disparities.
3. Select the farther candidate, identified by smaller disparity; use pixel
   distance only as a tie-breaker.
4. Leave a row black only when the complete row is invalid.

This fills disocclusions from background surfaces instead of treating them as
texture-removal regions. `none` leaves explicit holes black for debugging.
Delete the old `depth_to_disparity`, `create_shifted_image`, automatic black-hole
mask, and OpenCV stereo inpainting helpers after all callers migrate.

## Execution and Memory Model

CUDA rendering stays in the main process. No worker process imports or touches
CUDA. A bounded two-slot pipeline overlaps CPU frame decoding, main-process GPU
rendering, and CPU PNG writing; worker threads handle only OpenCV I/O and numpy
arrays. GPU background fill removes the former serial CPU inpainting stage.

CPU mode invokes the same Torch renderer on `torch.device("cpu")` and is the
reference path for exact tests. CUDA parity tests use numerical tolerances.

The production video pipeline always uses file-backed raw and canonical depth.
The legacy in-memory depth API calculates `N*native_H*native_W*4` before
inference and rejects requests above 512 MiB with guidance to use file-backed
processing. This limit does not claim to bound source-frame memory owned by
callers.

## Cache and Resume

Increment stage and depth-cache schemas. Resume validation operates per stage:

- `00_original_frames` remains valid when frame count, dimensions, and source
  video fingerprint match. It is never discarded solely because depth or stereo
  schema changed.
- Old or missing scene metadata invalidates `01_scene_data` and every later
  stage. A schema-valid `status: candidate` manifest resumes scene finalization
  but is never accepted by canonicalization.
- Old raw-depth metadata invalidates `02_depth_raw` and every later stage.
- Old or missing canonical metadata invalidates `03_disparity_maps` and every
  later stage.
- Stereo setting changes invalidate only stereo and later stages.

The user-facing resume message lists preserved and invalidated stages. No
heuristic migration of old depth PNGs is permitted because they do not record
whether the source model emitted depth or inverse depth.

Partial raw inference resumes by frame filename. Scene samples and bounds are
finalized only after every raw frame exists, which is the global barrier. A
crash between writing final bounds and replacing the candidate manifest reruns
the deterministic finalization step. Partial canonicalization resumes safely
because it is a pure function of persisted raw maps, a final manifest, and
fingerprinted final bounds.

## Error Handling

- Invalid estimator rank, dtype, or frame count raises at the estimator
  boundary.
- Shape mismatches between image and canonical disparity are resized once by
  the shared renderer and then revalidated.
- A second CUDA out-of-memory failure reports frame size and attempted band
  heights; it never silently switches devices.
- Non-finite render tensors raise before image encoding.
- Unsupported or removed setting names fail validation instead of being ignored.
- Metadata writes use a temporary file followed by atomic replacement.

## Three Verifiable Implementation Slices

### Slice 1: Depth Contract and Canonical Disparity

Add `DepthBatch`, adapter declarations/tests, scene pre-pass, raw-depth storage,
persisted candidate samples, deterministic fixed-point scene merging and bounds,
pure canonicalization, disk preflight/retention, and local metadata. Keep the
current inverse-remap renderer temporarily through a narrow adapter that
bypasses the old physical formula and supplies a positive, near-first
pixel-disparity map directly. Run unit tests and one real short clip; inspect
near/far direction and clean/chunk/resume identity before continuing.

### Slice 2: Torch DIBR

Add corrected-sign forward splat, pixel-unit visibility, explicit masks, fixed
memory tiling, GPU background fill, and the bounded I/O pipeline. Replace every
preview and production remap caller. Run synthetic geometry tests, CPU/CUDA
parity, 1080p/4K benchmarks, and the same real clip before continuing.

### Slice 3: Settings, Cache, and Resume Break

Replace old settings end to end, update stage directories and schemas, preserve
valid original frames on resume, reject old depth/downstream stages clearly,
remove temporary adapters and dead image-processing functions, and update docs.
Run the complete test suite and repeat the real-clip comparison.

Each slice is committed separately. A failure in a real render can therefore be
bisected to representation/scaling, projection/fill, or persistence/settings.

## Verification

Tests are written before their production changes and cover:

- every estimator's declared representation and near/far direction through the
  canonical adapter;
- no estimator performs per-frame min/max normalization;
- deterministic scene IDs, false-cut merge guard, robust bounds, outliers, flat
  maps, and non-finite values;
- left-to-right fixed-point merge order, pooled-sample union, and final bounds
  recomputation after every merge;
- candidate manifests cannot canonicalize, including a simulated crash after
  final bounds are written but before the final manifest replacement;
- clean, chunked, and resumed canonical output identity;
- native-resolution float16 compressed raw round trips through float32
  canonicalization without source-frame upsampling;
- disk-budget refusal before inference and raw retention/cleanup after validated
  canonical writes;
- local metadata creation and global-cache metadata round trip;
- constant-disparity translation and left/right symmetry;
- a non-symmetric sign test asserting a near foreground centroid has
  `x_left > x_right`;
- bilinear weights, out-of-frame holes, and a near-surface z-buffer victory;
- pixel-unit visibility tolerance across different scene bounds and uint16
  round trips;
- valid black source pixels are never selected as holes;
- GPU background fill chooses the farther horizontal candidate;
- CPU/CUDA parity when CUDA is available;
- row-band equivalence, deterministic band calculation, and OOM retry;
- 512 MiB in-memory depth rejection;
- resume preserves `00_original_frames` while invalidating incompatible depth
  and downstream stages;
- projector, file-backed processing, preview, CLI, Web, cache, and resume all
  use the final setting schema;
- removed settings and old render helpers are absent from production code,
  templates, examples, and current documentation.

The complete existing suite must remain green after every slice. Add a
non-gating benchmark reporting 1080p and 4K render time plus peak CUDA memory.
Performance numbers are observations, not portable test thresholds.

## Acceptance Criteria

- For a synthetic near foreground, `u_left - u_right > 0`.
- All five estimator mode declarations produce `canonical_near > canonical_far`.
- No estimator performs per-frame min/max normalization.
- Canonical output is identical across a resume boundary and a clean run.
- Canonicalization cannot start until all raw frames exist and the scene
  manifest has `status: final` with matching final-bounds fingerprint.
- No production stereo path calls inverse remap or colour-derived hole masking.
- A foreground/background collision always renders the nearer colour.
- Valid black source content is unchanged unless its explicit splat mask is
  invalid.
- `03_disparity_maps/metadata.json` is written locally and required by stereo.
- Raw depth is stored compressed at native model resolution, and insufficient
  disk space fails before long-running inference begins.
- Resume with the previous schema preserves valid original frames and rebuilds
  only scene/depth/downstream stages.
- CPU tests pass and CUDA-specific tests pass on the available NVIDIA GPU.
