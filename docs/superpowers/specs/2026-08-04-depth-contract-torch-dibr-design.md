# Depth Contract and Torch DIBR Design

## Status

Revision 8, updated after implementation review on 2026-08-05. The user approved a
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
- Bound temporary GPU memory for 4K rendering and host memory for both the
  file-backed stereo pipeline and the legacy in-memory depth API.

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

Representation conversion is deliberately separate for all three enum values:

- `METRIC_DEPTH`: finite values greater than zero are valid and use
  `score = 1 / value`; zero and negative values are invalid.
- `INVERSE_DEPTH`: every finite value is valid and uses `score = value`.
- `RELATIVE_DEPTH`: every finite value, including zero, is valid and uses the
  affine-safe monotonic reversal `score = -value`.

Relative depth never uses a reciprocal because its arbitrary affine offset has
no physical zero. Adapter tests include a near region whose raw relative-depth
value is exactly zero.

Each adapter has a contract test that checks both its declared enum and the full
near/far direction through canonical conversion. These tests do not load large
models: they feed a two-region synthetic estimator result through the adapter
boundary and assert `canonical_near > canonical_far`.

Those synthetic tests verify adapter wiring, not the upstream model's real
polarity. Release verification therefore runs one annotated real clip through
each non-physical output family independently: Video Depth Anything relative,
Depth Anything 3 non-metric, and See-Through Marigold. For each model, fixed
near and far ROIs are selected before inference and the native raw medians must
match the declared representation before canonical output is inspected. A pass
from one model cannot stand in for either of the other two.

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
Frame decode and histogram calculation run through an ordered bounded thread
pool; only the adjacent-histogram comparison and cut decisions remain serial.
The ordered result makes worker scheduling irrelevant to the manifest bytes.
Defaults are `scene_detection=true`, `scene_cut_threshold=0.55`, and
`min_scene_frames=8`. They are typed processing settings and persisted rather
than hidden constants. Disabling detection assigns every frame to scene zero.

Candidate cuts are provisional and the first manifest is always written with
`status: candidate`. Canonicalization accepts only `status: final`.

After all raw-depth samples exist, adjacent candidate scenes are merged when the
maximum difference between their low/high bounds is no more than 10 percent of
their combined disparity span. Merging is deterministic:

When combined span is zero, equal low/high bounds are mergeable; unequal bounds
are not. No division by zero is performed.

1. Preserve candidate scenes and their sample arrays in source-frame order.
   Concatenate those arrays once into one immutable pooled buffer. Each mutable
   block stores a contiguous `[start, stop)` span and its cached bounds.
2. Scan left to right. Comparisons use cached block bounds. When two neighbors
   merge, extend the lower block's span, recompute 2nd/98th percentile bounds
   once from that pooled-buffer view, replace the pair at the lower index, and
   compare that merged scene with its new right neighbor. When a pair does not
   merge, advance the cursor by one.
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

`raw_storage_dtype` is a typed setting with `auto`, `float16`, and `float32`;
default is `auto`. `DepthProcessor` keeps the first inferred chunk in memory,
checks its finite range, selects float16 when representable or float32 otherwise,
and records the chosen dtype before writing any raw file. An explicit float16
request fails at this first-chunk gate when its values are not representable.

Each native-resolution map is then written atomically as a zlib-compressed
`.npz` file in `02_depth_raw`. Float16 is a storage encoding only: canonical
conversion reads it into float32 before reciprocal, negation, sampling,
percentile, or arithmetic. The directory's `metadata.json` records
representation, model fingerprint, frame count, native dimensions, requested
and selected storage dtype, storage provenance, compression, schema version,
`promoted_frame_count`, and `storage_status: ready | promoting`. Raw-depth
schema version 2 owns the `promoted_frame_count` fingerprint field; version 1
directories fail explicitly as a schema mismatch.

Each chunk atomically commits only its new frame payloads and increments an
in-memory completed counter. The complete metadata document is flushed once at
the raw-stage boundary, not rewritten for every chunk. After a crash, resume
reconstructs the counter once from expected frame filenames. Resume validates
NPZ membership plus the embedded NPY shape and dtype from array headers without
inflating payload data; full data decoding and the same shape/dtype checks occur
when a raw map is actually consumed. Membership compares basenames inside the
single raw directory and never resolves every expected and actual path.

A later finite value outside float16 range is never clipped and does not force
completed frames through model inference again. The only permitted in-place
fingerprint transition is float16-to-float32 storage promotion while every
semantic model, source, preprocessing, representation, and shape field remains
identical:

1. Stop before committing the offending frame and preflight the final float32
   directory size plus one atomic-rewrite payload.
2. Atomically set `storage_status: promoting`, preserve the prior fingerprint,
   and set the target dtype and provenance to
   `promoted_float16_to_float32`. Persist `promoted_frame_count` as the number
   of already completed maps that were originally quantized to float16.
3. In frame-name order, load each completed float16 file and atomically rewrite
   it as float32. This conversion is exact with respect to the already stored
   float16 value and invokes no estimator.
4. Validate that every completed file is float32, atomically publish the new
   ready fingerprint, then write the held offending chunk as float32 and
   continue inference.

A crash during promotion leaves a resumable `promoting` directory; resume scans
file dtypes and rewrites only remaining float16 files before inference can
continue. Mixed dtypes are valid only inside this transaction and are never
accepted by sampling or canonicalization. `auto` promotes automatically. An
explicit float16 request stops with instructions to resume using float32, which
enters the same promotion path rather than deleting completed raw files.
Non-finite model values remain permitted and canonicalize as invalid.

Bounds are deterministic and bounded-memory:

1. Convert selected raw values to an unscaled disparity score using the same
   representation contract as canonicalization: metric depth uses a safe
   reciprocal, inverse depth passes through, and relative depth is negated.
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

Empty or flat scenes receive equal bounds and canonicalize every pixel to the
neutral canonical midpoint `0.5`, not canonical zero. With the default
`convergence=0.5` this produces zero binocular disparity. With a non-default
convergence it produces one constant, scene-independent offset. There is no EMA
or other mutable scaling state.

## Pure Canonicalization

Create `processing/frames/depth_normalizer.py` with a stateless function:

```python
canonicalize_depth(
    values: np.ndarray,
    representation: DepthRepresentation,
    bounds: SceneDepthBounds,
) -> np.ndarray
```

It marks non-finite values invalid for every representation and additionally
marks non-positive values invalid only for `METRIC_DEPTH`. It applies the three
representation-specific score rules, maps persisted scene bounds to `[0,1]`,
clips outliers, and sets invalid pixels to the neutral midpoint `0.5`. Output is
float32 with `0=far`, `0.5=neutral`, and `1=near`.

Canonical maps are encoded deterministically as
`np.rint(np.clip(r, 0, 1) * 65535).astype(np.uint16)` in
`03_disparity_maps`; decoding converts to float32 and divides by `65535.0`.
Thus canonical float32 midpoint is exactly `0.5`, while its encoded round trip
is `32768 / 65535` and is compared with one-quantum tolerance.
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

Before raw inference, ask the estimator for its native output shape when it
declares one; otherwise estimate dimensions from the selected depth resolution
while preserving source aspect ratio. See-Through explicitly declares its
square `processing_resolution x processing_resolution` output, matching its
square input preprocessing rather than the generic aspect-ratio estimate.
Refuse to start unless current free space covers the expected peak:

- raw allowance:
  `frames * native_width * native_height * storage_bytes * 1.25`;
- canonical allowance: `frames * native_width * native_height * 2 * 1.10`
  bytes;
- with `keep_intermediates=true`, peak is the sum;
- otherwise peak is the larger allowance plus two frame payloads for atomic
  write overlap.

`storage_bytes` is 2 or 4 for an explicit dtype; the initial `auto` check assumes
4 so a later promotion is budgeted conservatively. The estimate intentionally
assumes no compression benefit. After the first chunk reveals actual native
dimensions and selected dtype, repeat the check before writing that chunk or
processing remaining frames. Promotion performs its own four-byte preflight
before mutating metadata. Failure reports required, available, and output-path
bytes before more model work is performed.

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
- `raw_storage_dtype`: `auto`, `float16`, or `float32`. Default `auto`.
- `stereo_io_workers`: integer in `1..16`. Default is
  `min(4, max(1, cpu_count - 2))`.
- `migrate_legacy`: `archive` or `delete`. Default `archive`; deletion must be
  supplied explicitly in the current CLI, Web, or configuration request.

For canonical disparity `r`, frame width `w`, strength `s`, and convergence
`c`, total binocular disparity is:

`d = (r - c) * w * (s / 100)`

The corrected rectified-stereo projection is:

- `x_left = x + d/2`
- `x_right = x - d/2`

Therefore a near point (`r > c`) appears farther right in the left image than
in the right image, so `u_left - u_right > 0`.

The renderer first resizes canonical `r` to the exact render target height and
width with bilinear interpolation. It then calculates `d` from that resized
field. In the formula, `w` is always render target width, never source-video
width or native model-output width. Pixel disparity is never resized after it is
calculated. For See-Through, this single anisotropic resize reverses the model's
intentional square preprocessing transform and restores alignment with the
rectangular source frame; the square raw and canonical maps remain recorded as
their true native shape.

## Torch Forward Splat

Create `rendering/forward_splat.py` and `rendering/stereo_renderer.py`.

The low-level splat accepts batched Torch tensors and performs horizontal
bilinear forward projection:

1. Project each source pixel to the two neighboring integer target columns.
2. Discard out-of-frame contributions instead of clipping or reflecting them.
3. Let only bilinear contributions with weight greater than or equal to `0.5`
   cast the primary target-depth vote. This is the nearest target column for
   each source pixel and prevents a tiny foreground tail from becoming a
   full-coverage z-buffer owner.
4. For each target pixel, define `z_win` as the voters' `scatter_reduce(amax)`
   over total signed pixel disparity `d`, where larger `d` is nearer. If that
   target has contributions but no voter, fall back to the `amax` over all its
   contributions. A target with no contribution remains invalid.
5. Apply one-sided visibility to every contribution, including low-weight
   antialiasing tails: keep it when `d >= z_win - 0.25`. Nearer fractional
   coverage may blend over a farther voter; farther surfaces cannot bleed
   through a nearer winner.
6. Accumulate visible colour and bilinear weights with `scatter_add`.
7. Divide colour by accumulated weight and mark zero-weight pixels invalid.

Both eyes use the same z-key `d`. The eye sign affects only target coordinate:
left uses `+d/2` and right uses `-d/2`. The right eye must never use `-d` as its
z-key.

The one-sided visibility tolerance is defined in projected pixel units, not
canonical depth units, so its meaning is stable across scenes and uint16 round
trips. The no-voter fallback keeps stretched sloped surfaces inside projected
support valid instead of misclassifying their antialiasing tails as holes. The
two-tap bilinear kernel cannot cover every target column when local per-eye
stretch reaches `2x` or more. For a full-range canonical ramp spanning `N`
source pixels, this begins at `N <= render_width * stereo_strength / 200`.
Such columns have no positive-weight contribution, remain invalid after splat,
and are intentionally handled as isolated gaps by bounded background fill. A
zero-weight neighbor is not a contribution: changing `weights > 0` to
`weights >= 0` would create false coverage rather than extend kernel support.
When `stereo_strength=0`, all samples have zero disparity and no depth-based
occlusion is applied.

Coordinates, pixel disparity, colour accumulation, and z-buffer remain float32,
including under mixed-precision model inference. The final image is clamped and
converted back to its input dtype.

Rendering is split into complete row bands. A fixed 256 MiB temporary-memory
budget determines band height from frame width. Eyes render sequentially. The
one-eye logical live-set estimate is phase-specific because projection buffers
are released before deterministic sorted reduction:

| Phase | Dominant live allocations | Approx. bytes/source pixel |
| --- | --- | ---: |
| Projection and z voting | Source RGB/disparity, projected coordinates, two `int64` target indexes, bilinear weights/z values, vote/visibility masks, target z buffers | 132 |
| Stable sorted reduction | `int64` safe indexes, `order`, sorted indexes, unique indexes and counts; sorted weights/values; target colour, weight, and disparity accumulators | 112 |

The implementation constant is `SPLAT_BYTES_PER_PIXEL=192`, reserving at least
60 bytes per pixel above the larger explicit live set for `argsort` workspace,
Torch reduction temporaries, horizontal-fill propagation/gather buffers,
allocator alignment, and bookkeeping. Splat contribution buffers are
released or reused before fill, so these phases contribute their maximum rather
than their sum.
Band rows are
`max(1, floor(256 MiB / (render_width * SPLAT_BYTES_PER_PIXEL)))`, capped at
render height. The result does not depend on runtime free-memory queries. CUDA
out-of-memory retries the frame once with half the calculated band height.

Background fill runs before each completed row band leaves the device. Each
band's colour and masks are copied into preallocated host arrays, then its
z-buffer, projected disparity, propagation indexes, and fill temporaries are
released or reused for the next band. There are no full-frame GPU-resident
buffers. `StereoRenderResult` contains only host-resident left/right images and
left/right valid and hole masks; projected disparity is an internal band-scoped
buffer. Masks derive from accumulated visible splat weight, never image colour.

## GPU Background Fill

`occlusion_fill=background` fills horizontal DIBR gaps on the same Torch device:

1. For every row, propagate the nearest valid index from the left and right.
2. Gather both candidate colours and their projected disparities.
3. Select the farther candidate, identified by smaller disparity; use pixel
   distance only as a tie-breaker.
4. Fill only horizontal invalid runs no wider than
   `ceil(render_width * stereo_strength / 200) + 2` pixels, the maximum
   expected per-eye near-to-far displacement plus bilinear coverage.
5. Leave wider runs and completely invalid rows black with their hole masks set.

This fills disocclusions from background surfaces instead of treating them as
texture-removal regions. `none` leaves explicit holes black for debugging.
The method does not synthesize unseen texture. A run whose valid candidates
resolve to one background boundary can become a constant-colour horizontal
strip; this is an accepted limitation of the deterministic baseline and is
reported in renderer metadata. Learned filling remains a separate future
backend, not a hidden fallback.
Delete the old `depth_to_disparity`, `create_shifted_image`, automatic black-hole
mask, and OpenCV stereo inpainting helpers after all callers migrate.

## Execution and Memory Model

CUDA rendering stays in the main process. No worker process imports or touches
CUDA. A bounded producer/consumer pipeline overlaps CPU frame decoding,
main-process GPU rendering, and CPU PNG writing. Worker threads handle only
OpenCV I/O and numpy arrays. GPU background fill removes the former serial CPU
inpainting stage.

The production pipeline has a fixed `STEREO_HOST_BUDGET=512 MiB` and a
conservative `HOST_STEREO_BYTES_PER_PIXEL=24`. One permit covers a source RGB
frame, native and resized canonical float32 fields, two uint8 eye outputs, four
boolean masks, PNG encoding overlap, and array/task overhead for the frame's
complete lifetime. The 24-byte estimate is `3 source + 8 canonical + 6 eye
output + 4 mask + 3 reserve` bytes per render-target pixel. The eight canonical
bytes cover the worst shape-mismatch point where both the caller-owned native
field and the renderer's resized field are live. Source, canonical, and mask
payloads are released before writer encoding, leaving sufficient space for
encoded payloads; each slot also reserves 1 MiB of fixed task/codec overhead.
Define:

`slot_bytes = render_width * render_height * HOST_STEREO_BYTES_PER_PIXEL + 1 MiB`

Permit count is:

`min(2 * stereo_io_workers, floor(STEREO_HOST_BUDGET /
      slot_bytes))`

It must be at least one or the frame is rejected with required bytes. Queue
capacity equals this count, and all decode and write queues additionally share
the same lifecycle semaphore. One permit is acquired before decode and released
only after both eye writes complete, so queued and actively decoded, rendered,
or encoded frames share the same bound. More workers cannot increase resident
frame payload beyond this permit count. Queue wait and permit wait are measured
so I/O and memory backpressure are visible.

CPU mode invokes the same Torch renderer on `torch.device("cpu")` and is the
reference path for exact tests. CUDA parity tests use numerical tolerances.

The production video pipeline always uses file-backed raw and canonical depth.
The legacy in-memory depth API calculates `N*native_H*native_W*4` before
inference and rejects requests above 512 MiB with guidance to use file-backed
processing. This limit does not claim to bound source-frame memory owned by
callers.

## Model Fingerprint

Raw metadata contains and resume validates one canonical model fingerprint over:

- estimator backend and model family;
- exact repository/model revision, never a floating `main` label;
- weight-file SHA-256 or immutable hub artifact digest;
- model size, metric flag, and declared `DepthRepresentation`;
- native depth resolution and preprocessing algorithm version;
- inference precision settings, requested and selected `raw_storage_dtype`, and
  storage provenance (`native_float16`, `native_float32`, or
  `promoted_float16_to_float32`).

`get_model_info()` is a UI/reporting surface and is never hashed wholesale.
Fingerprint construction allowlists immutable identity and output-affecting
fields such as model name/version, revision or artifact digest, architecture,
metric mode, device/precision, and native processing settings. Mutable runtime
state such as `loaded`, plus presentation-only capability flags, is excluded.
Device remains part of the identity because changing numerical backends can
change output bytes. Every reported key must be classified in either the
identity allowlist or an explicit presentation-only list; an unknown key is a
hard error so a newly added output-affecting field cannot be silently omitted.

Partial raw files are reusable by frame name only when the complete fingerprint,
source-frame fingerprint, schema, and expected native shape match. Any mismatch
invalidates all of `02_depth_raw` except the explicitly transactional,
storage-only float16-to-float32 promotion defined above. A promoted fingerprint
is deliberately distinct from native float32 because widening does not recover
precision already quantized by float16. A ready directory never mixes raw maps
from different fingerprints. The same final fingerprint participates in global
depth-cache keys and local canonical-metadata fingerprints.

## Cache and Resume

Increment stage and depth-cache schemas. Resume validation operates per stage:

- `00_original_frames` remains valid when frame count, dimensions, and source
  video fingerprint match. It is never discarded solely because depth or stereo
  schema changed.
- Old or missing scene metadata invalidates `01_scene_data` and every later
  stage. A schema-valid `status: candidate` manifest resumes scene finalization
  but is never accepted by canonicalization.
- Missing or old raw-depth metadata, or any model/source/shape fingerprint
  mismatch, invalidates all of `02_depth_raw` and every later stage. A storage
  mismatch is reusable only when it qualifies for the explicit
  float16-to-float32 promotion; every other storage mismatch invalidates it. A
  partial directory is resumed only after this complete validation.
- Old or missing canonical metadata invalidates `03_disparity_maps` and every
  later stage.
- Stereo setting changes invalidate only stereo and later stages.

Removed settings have two distinct validation paths:

- A removed name supplied explicitly through CLI, Web, or a new configuration
  file is an error.
- A removed name found while reading an on-disk legacy `settings.json` is
  stripped in migration mode, listed in the resume report, and replaced by the
  new default or explicit new setting. The original file is retained as
  `settings.legacy.json`; the migrated schema is written separately.

Legacy `02_depth_maps` is an explicitly known invalid stage, as are its old
stereo and downstream directories. The resume report lists each directory and
size before mutation. Legacy migration never prompts or waits for acceptance.
The default `migrate_legacy=archive`, including every non-interactive run, moves
invalid generated directories under `legacy_v1/<original-directory-name>` and
excludes them from stage discovery. Deletion occurs only when
`migrate_legacy=delete` is supplied explicitly in the current invocation. A
value inherited from legacy disk settings cannot authorize deletion. A
pre-existing archive destination is an error rather than an overwrite.
`keep_intermediates` controls current-schema raw retention only and does not
authorize legacy deletion. `00_original_frames` is never archived or deleted by
this migration, including when its manifest is invalid. A missing or mismatched
source-video fingerprint aborts resume before migration instead of authorizing
replacement or deletion of extracted frames.

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
- Unsupported or removed setting names supplied explicitly by the user fail
  validation instead of being ignored; legacy on-disk settings use the
  migration path defined above.
- Metadata writes use a temporary file followed by atomic replacement.

## Three Verifiable Implementation Slices

New settings land with the slice that owns their behavior. Slice 1 implements
scene controls and `raw_storage_dtype`; Slice 2 implements stereo controls and
`stereo_io_workers`; Slice 3 implements `migrate_legacy` and removes obsolete
names from every entry point. Slice 3 does not defer settings required to verify
Slices 1 and 2.

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

### Real-model polarity record

On 2026-08-05, all three non-physical model families were run independently on
four consecutive frames beginning at 2.0 seconds in the official Video Depth
Anything `Tokyo-Walk_rgb.mp4` example. The fixed normalized near ROI covered the
foreground pedestrian (`x=.405..475`, `y=.30..68`); the far ROI covered the
distant street (`x=.66..76`, `y=.40..56`). Processing resolution was 384.

| Model artifact | Declared representation | Near ROI medians | Far ROI medians | Result |
| --- | --- | ---: | ---: | --- |
| VDA Large, weight SHA-256 `43df27c6...d0e6fa`, repository `4f5ae231` | `INVERSE_DEPTH` | `906.34..907.99` | `86.84..88.47` | near is larger |
| DA3-SMALL non-metric, snapshot `e08cab65` | `RELATIVE_DEPTH` | `0.6183..0.6197` | `1.6520..1.7039` | near is smaller |
| See-Through Marigold, snapshot `aa7a892f` | `RELATIVE_DEPTH` | `0.7109..0.8125` | `0.9922` | near is smaller |

Thus the three adapter declarations match real upstream output polarity. This
record does not replace the synthetic contract tests; it verifies the fact that
those tests deliberately cannot establish.

Tests are written before their production changes and cover:

- every estimator's declared representation and near/far direction through the
  canonical adapter, including a relative-depth near value of exactly zero;
- annotated real-clip polarity checks for Video Depth Anything relative, Depth
  Anything 3 non-metric, and See-Through Marigold independently;
- no estimator performs per-frame min/max normalization;
- deterministic scene IDs, false-cut merge guard, robust bounds, outliers, flat
  maps, and non-finite values;
- ordered parallel histogram extraction plus cached scene-block bounds and one
  pooled sample allocation during fixed-point merging;
- left-to-right fixed-point merge order, pooled-sample union, and final bounds
  recomputation after every merge;
- candidate manifests cannot canonicalize, including a simulated crash after
  final bounds are written but before the final manifest replacement;
- clean, chunked, and resumed canonical output identity;
- empty and flat scenes produce exact float32 canonical `0.5`; uint16 PNG
  round-trip tests expect `32768 / 65535` within one encoding quantum;
- native-resolution float16 compressed raw round trips through float32
  canonicalization without source-frame upsampling;
- disk-budget refusal before inference and raw retention/cleanup after validated
  canonical writes;
- local metadata creation and global-cache metadata round trip;
- constant-disparity translation and left/right symmetry;
- a non-symmetric sign test asserting a near foreground centroid has
  `x_left > x_right`;
- bilinear weights, out-of-frame holes, and a near-surface z-buffer victory;
- a vertical near/far step where a low-weight foreground tail produces the
  weighted result `(w_bg*C_bg + w_fg*C_fg) / (w_bg + w_fg)` over the background
  voter rather than replacing the column with pure foreground;
- a full-range disparity ramp spanning 20 source pixels whose interior projected
  support contains no hole, including target columns with no primary voter;
- a local `2x` stretch whose alternating zero-support columns remain explicitly
  invalid after splat and are closed by one-pixel bounded background fill;
- left and right eyes independently choose the near surface with shared `d` as
  z-key;
- pixel-unit visibility tolerance across different scene bounds and uint16
  round trips;
- native canonical disparity is bilinearly resized before target-width pixel
  disparity is calculated;
- valid black source pixels are never selected as holes;
- GPU background fill chooses the farther horizontal candidate;
- GPU background fill respects its derived run-width limit and documents the
  constant-strip limitation;
- CPU/CUDA parity when CUDA is available;
- row-band equivalence, in-band background fill, absence of full-frame GPU
  intermediates, deterministic band calculation, and OOM retry;
- the documented int64 scatter-index allocation is included in the 192-byte
  row-band budget;
- 512 MiB stereo-pipeline permit accounting under default and maximum worker
  counts, plus 512 MiB legacy in-memory depth rejection;
- first-chunk raw dtype selection, explicit float16 rejection, atomic
  float16-to-float32 promotion without reinference, and crash-resume during
  promotion;
- NPZ header-only resume validation, stage-boundary metadata flush, and an exact
  count of frames whose values were quantized before float32 promotion;
- uninterrupted and crash-resumed promotion produce byte-identical canonical
  output, and completed frame names never re-enter the estimator;
- raw resume rejection on model, weight, preprocessing, representation, shape,
  or any non-promotable storage fingerprint mismatch;
- resume preserves `00_original_frames` while invalidating incompatible depth
  and downstream stages;
- legacy disk settings are migrated while explicitly supplied removed settings
  fail; unattended legacy migration archives by default, and deletion requires
  an explicit current-invocation setting;
- projector, file-backed processing, preview, CLI, Web, cache, and resume all
  use the final setting schema;
- removed settings and old render helpers are absent from production code,
  templates, examples, and current documentation.

The complete existing suite must remain green after every slice. Add a
non-gating benchmark reporting 1080p and 4K GPU render time, peak CUDA memory,
end-to-end wall-clock time per frame, effective FPS, PNG writer utilization, and
queue stall time. Performance numbers are observations, not portable test
thresholds.

## Acceptance Criteria

- For a synthetic near foreground, `u_left - u_right > 0`.
- All five estimator mode declarations produce `canonical_near > canonical_far`;
  a relative-depth near value of exactly zero remains valid.
- Real-model ROI checks independently confirm the declared raw polarity of Video
  Depth Anything relative, Depth Anything 3 non-metric, and See-Through
  Marigold; this release gate cannot be satisfied by synthetic adapter stubs.
- No estimator performs per-frame min/max normalization.
- Canonical output is identical across a resume boundary and a clean run.
- Canonicalization cannot start until all raw frames exist, raw metadata has
  `storage_status: ready`, and the scene manifest has `status: final` with a
  matching final-bounds fingerprint.
- Scene merging is deterministic at zero and nonzero span, and every final
  bound is recomputed from the fixed-point union of persisted samples using
  cached block bounds and one source-ordered pooled sample allocation.
- Empty and flat scenes produce exact float32 canonical `0.5`; encoded maps use
  the documented uint16 midpoint and quantization tolerance.
- No production stereo path calls inverse remap or colour-derived hole masking.
- Canonical disparity is bilinearly resized to render dimensions before `d` is
  calculated with render target width.
- A foreground/background collision renders the nearer colour independently in
  both eyes using common z-key `d`.
- A low-weight foreground tail cannot overwrite the background-side pixel of a
  depth edge.
- A stretched 20-pixel full-range disparity ramp has no holes inside its
  projected support, including columns without a primary depth voter.
- A local stretch of at least `2x` leaves only genuine zero-support columns
  invalid after splat, and bounded background fill closes the isolated gaps.
- Valid black source content is unchanged unless its explicit splat mask is
  invalid.
- Background fill never crosses its derived maximum gap width; wider gaps stay
  explicitly invalid, and the constant-colour-strip limitation is documented.
- `03_disparity_maps/metadata.json` is written locally and required by stereo.
- Raw depth is stored compressed at native model resolution, and insufficient
  disk space fails before long-running inference begins.
- Partial raw resume requires an exact model/source/shape/storage fingerprint;
  the sole storage-only exception atomically promotes completed float16 files
  to a distinct float32 provenance without rerunning their estimator frames.
- Resume with the previous schema preserves valid original frames and rebuilds
  only scene/depth/downstream stages; removed legacy settings and
  `02_depth_maps` follow the reported migration policy.
- Unattended legacy migration archives by default and never deletes generated
  data without an explicit current-invocation `migrate_legacy=delete`.
- The row-band budget includes int64 scatter indexes and in-band fill, with no
  full-frame GPU intermediate. Total queued and active stereo frame payload is
  bounded by the 512 MiB host permit budget regardless of worker count.
- The 1080p/4K benchmark reports both GPU render measurements and end-to-end
  wall-clock throughput, including writer utilization and queue stalls.
- CPU tests pass and CUDA-specific tests pass on the available NVIDIA GPU.
