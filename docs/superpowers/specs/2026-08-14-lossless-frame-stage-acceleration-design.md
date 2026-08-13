# Lossless Frame-Stage Acceleration Design

## Status

The user selected the conservative optimization approach on 2026-08-14:
accelerate only work whose decoded output can remain pixel-identical to the
current implementation. This written specification is awaiting final user
review before an implementation plan is produced.

## Problem

The pipeline persists every major result as a numbered frame stage so that a
long video can resume after interruption. That recovery model is valuable, but
several CPU frame stages currently process one frame pair at a time or repeat
work whose inputs do not change.

A measured 2,014-frame, 1920x1080-per-eye job exposed the following costs:

| Stage | Approximate time | Observed throughput | Main avoidable work |
| --- | ---: | ---: | --- |
| Canonical disparity | 97 s | 20.8 fps | Independent maps handled serially |
| Stereo generation | 660 s | 3.05 fps | Separate renderer concern, excluded here |
| Crop, factor 1.0 | 941 s | 2.14 fps | Decode and re-encode pixels that do not change |
| VR assembly | About 1,000 s | About 2 fps | Independent frame pairs handled serially |

The same job used no fisheye distortion, no upscaler, side-by-side output, and
NVENC encoding. Stereo I/O was explicitly configured to one worker. The depth
inference stage has already been optimized separately and is not part of this
design.

Targeted local benchmarks on real frames established that the proposed paths
do not require an image-quality trade-off:

- Parallel VR assembly scaled from 3.74 fps with one worker to 13.96 fps with
  four workers and 24.19 fps with eight workers. Decoded pixel hashes matched.
- Parallel canonicalization scaled from 11.15 fps with one worker to 54.71 fps
  with four workers and 71.68 fps with eight workers. Encoded canonical pixels
  matched.
- Creating hard links for 32 no-op crop outputs completed at about 2,600 files
  per second on the same volume and preserved the exact source bytes.
- A 1920x1080 fisheye coordinate map took about 0.21 seconds to build while the
  remap itself took about 0.017 seconds. Reusing one map therefore removes the
  dominant repeated computation without changing the remap.

## Goals

- Preserve the current frame-on-disk, resumable stage architecture.
- Preserve decoded pixels exactly for every optimized path on the same OpenCV
  and NumPy backend.
- Eliminate decode and PNG re-encode work when non-fisheye cropping is a true
  no-op.
- Compute fisheye coordinate maps once per successful uniform-resolution stage,
  not once per eye per frame.
- Process independent canonical, transformed stereo-pair, and VR assembly work
  concurrently with bounded CPU and memory use.
- Keep progress callbacks, previews, manifests, and final validation on the
  caller thread and in source-frame order.
- Add no user-facing tuning setting. Worker selection must be automatic and
  conservative.
- Leave failed stages non-reusable so the next run clears partial output and
  starts that stage cleanly.

## Non-goals

- Replacing numbered frame stages with a streaming FFmpeg pipeline.
- Feeding stereo images directly into the encoder and removing VR PNG frames.
- Changing NVENC quality, preset, rate-control, pixel-format, or codec options.
- Combining the two stereo-eye renders or changing CUDA transfer behavior.
- Changing `stereo_io_workers`; the existing setting remains authoritative.
- Enabling half precision or batching in Real-ESRGAN.
- Changing PNG compression levels. Lower compression is lossless in pixels but
  substantially increases intermediate storage and downstream I/O.
- Optimizing source extraction, which is already near source playback rate and
  produces frames required by depth estimation and resume.
- Changing scene analysis, which is already parallel and not a material part of
  the measured runtime.
- Changing any depth, stereo geometry, crop geometry, resize interpolation, or
  fisheye projection formula.

## Quality And Recovery Invariants

The following invariants are hard requirements rather than best-effort goals:

1. A transformed output decoded with `cv2.IMREAD_UNCHANGED` must be exactly
   equal, including dtype and shape, to the current serial implementation.
2. A no-op crop output must contain the exact source PNG bytes. This is at least
   as strong as the current behavior, whose decoded pixels equal the source.
3. Frame names, ordering, stage identity inputs, metadata schema, bit depth, and
   output shape must not change.
4. A stage may call `complete_stage` only after every worker has succeeded.
5. A worker exception, unreadable input, failed write, or failed link/copy must
   make the stage return failure and must not write completion metadata.
6. Concurrent workers may only write distinct output paths. Shared arrays are
   read-only after construction.
7. The main thread alone calls progress and preview APIs. This avoids relying on
   thread safety in UI or websocket code.
8. Existing completed stages remain reusable. No algorithm-version bump is
   needed because output semantics and manifest inputs do not change.

Pixel identity, rather than coincidental PNG byte identity after re-encoding,
is the compatibility contract for transformation paths. OpenCV may encode the
same pixels differently across library versions, which is already outside the
same-backend reproducibility boundary.

## Architecture

### Bounded worker selection

Add one small frame-stage parallelism helper rather than duplicating worker
math across processors. It accepts a frame count and a conservative estimate of
peak bytes held by one active item.

The worker count is:

```text
cpu_limit    = max(1, logical_cpu_count - 2)
memory_limit = max(1, 1 GiB // estimated_bytes_per_item)
workers      = min(frame_count, 8, cpu_limit, memory_limit)
```

Missing CPU-count information is treated as one logical CPU. Empty work is
rejected by the caller before worker calculation. The one-GiB value is a
concurrent-item budget, not a claim about total process memory. One item is
still allowed when its conservative estimate alone exceeds the budget because
the stage must be able to make progress.

Each processor supplies a stage-specific estimate:

- Canonicalization: native map pixels multiplied by 32 bytes.
- Distortion or transformed crop: per-eye source pixels multiplied by 48 bytes
  for a pair.
- VR assembly: target per-eye pixels multiplied by 48 bytes for both decoded
  inputs, optional resized arrays, the combined image, and encoder buffers.

The eight-worker ceiling follows measured scaling and prevents OpenCV calls
from consuming every logical CPU. At 1920x1080, VR assembly receives eight
workers on a sufficiently large CPU; at higher resolutions, the memory limit
reduces concurrency automatically.

Use `ThreadPoolExecutor`, not process workers. OpenCV and NumPy release the GIL
for the expensive operations, threads avoid serializing large arrays, and each
task already has independent file paths. Futures are consumed in source order
for deterministic callbacks. On failure, pending futures are cancelled;
already-running tasks may finish their distinct writes, but no completion
manifest is emitted.

This helper chooses worker counts only. Stage-specific workers remain local to
their processors so file handling, result types, and failure messages stay
clear.

### No-op crop materialization

`DistortionProcessor.crop_frames` determines the clamped crop factor before
decoding any image. When distortion is disabled and that factor is exactly
`1.0`, the desired cropped image is the unmodified stereo source.

For this case, materialize each destination as follows:

1. Clear stale crop-stage outputs through the existing stage-manifest API.
2. Attempt `os.link(source, destination)` for each left and right PNG.
3. If a hard link is unsupported, crosses a volume boundary, or is denied,
   fall back to `shutil.copy2(source, destination)` for that file.
4. Report progress in frame-pair order.
5. Run the existing PNG header and `complete_stage` validation.

Hard links are safe in this pipeline because a finalized source frame is
immutable, downstream stages only read it, and cleanup unlinks paths rather
than modifying file contents in place. Removing either path does not remove the
other directory entry while it still exists. The copy fallback preserves the
same bytes and supports filesystems without hard-link capability.

When distortion is enabled or the clamped factor is below `1.0`, keep the
current crop functions and interpolation unchanged, but process independent
frame pairs with the bounded thread pool.

### Fisheye map reuse and distortion parallelism

Split fisheye work into two pure operations:

- Coordinate-map construction using the existing projection formulas.
- `cv2.remap` using supplied `float32` x/y maps, `INTER_LINEAR`, and
  `BORDER_REFLECT_101`.

The public `apply_fisheye_distortion` behavior remains unchanged by delegating
to those operations. `DistortionProcessor` performs a header preflight, builds
one immutable map pair for the stage's `(width, height, fov, projection)` key,
and shares those maps among workers. It therefore builds one map pair instead
of two map pairs per frame.

The preflight rejects unreadable headers, left/right shape mismatches, and
nonuniform frame shapes. This makes the existing successful-stage contract
explicit: `complete_stage` records one output shape and already rejects a stage
whose PNG dimensions vary.

Each worker decodes one left/right pair, selects the prebuilt map by shape,
applies the existing remap to both eyes, and writes the two existing output
paths. Progress remains sampled at the current frequency and is emitted by the
main thread.

### Parallel canonical disparity writes

`DepthMapProcessor._write_canonical_stage` keeps its current cache validation,
metadata, scene bounds, representation conversion, preview sampling, and
atomic-write rules.

One worker receives a raw-depth path, output path, and scene id, then:

1. Loads the raw map from `RawDepthStore`.
2. Calls the existing `canonicalize_depth` with the same representation and
   persisted scene bounds.
3. Calls the existing `encode_canonical_png`.
4. Writes through `_atomic_write_png` to its unique path.
5. Returns the output path to the caller.

The main thread consumes results in source order and sends the same sampled
previews as today. It writes `metadata.json` only after all workers finish and
then calls the existing full-stage validator. Shared store metadata, scene
bounds, and representation values are read-only during this phase.

### Parallel VR assembly

`VRFrameAssembler` validates and fingerprints the source manifest exactly as it
does now. A bounded worker then handles one pair:

1. Decode the two source PNGs.
2. Compare each decoded shape with `per_eye_width` and `per_eye_height`.
3. If a shape already matches, use that array directly. Otherwise call the
   existing `resize_image`, preserving `INTER_CUBIC`.
4. Call the existing `create_vr_frame` for side-by-side or over-under layout.
5. Write the existing destination PNG and return its path.

Skipping `cv2.resize` for an already matching shape is semantically exact and
was verified to produce identical pixels. It also avoids obscuring the more
important worker scaling with a call that has no useful effect.

The main thread sends previews and progress updates in frame order. Final
`complete_stage` shape validation remains unchanged and executes only after all
outputs succeed.

## Data Flow

The persistent data flow does not change:

```text
02_depth_raw/*.npz
  -> bounded canonical workers
03_disparity_maps/*.png
  -> existing stereo renderer
04_left_frames/*.png + 04_right_frames/*.png
  -> optional bounded distortion workers with shared maps
05_left_distorted/*.png + 05_right_distorted/*.png
  -> no-op link/copy OR bounded crop workers
06_left_cropped/*.png + 06_right_cropped/*.png
  -> existing optional upscaler
07_left_upscaled/*.png + 07_right_upscaled/*.png
  -> bounded VR assembly workers
99_vr_frames/*.png
  -> existing encoder
```

If distortion or upscaling is disabled, the existing source-directory
selection continues to bypass that stage in the same way it does today.

## Error Handling

- Validate left/right counts, names, configured output directories, and stage
  identity before scheduling work.
- A worker reports success only after every output for its item has been
  written: two images for distortion/crop, or one image for canonical/VR work.
- Convert worker exceptions into the processor's current `False` return or the
  canonical stage's current exception contract. Include the failing frame path
  in the diagnostic.
- Cancel futures that have not started after the first failure.
- Do not delete partial outputs in the failure handler. The existing
  non-reusable-stage path clears them at the start of the next attempt, avoiding
  races with workers that were already finishing.
- Do not write metadata early or mark a stage complete from a worker.
- Preserve atomic temporary-file replacement for canonical PNGs. The other
  stages retain their current distinct final-path writes.

## Tests

### Worker selection

- Verify CPU, eight-worker, frame-count, and memory caps independently.
- Verify one worker on a one-CPU or unknown-CPU host.
- Verify representative 1080p, 4K, and 8K estimates reduce concurrency as
  specified.

### Crop

- Verify a non-distorted factor of `1.0` does not call image decode or encode.
- On hard-link-capable storage, verify source and destination identify the same
  file and contain identical bytes.
- Force `os.link` to raise `OSError`; verify `copy2` fallback and byte identity.
- Verify deleting one hard-link path leaves the other readable.
- Compare factors below `1.0` against the serial reference pixel for pixel.
- Verify failure leaves no completion manifest and a retry clears partial work.

### Distortion

- Compare all supported projection types against the current serial helper
  pixel for pixel.
- Process multiple same-shape pairs and verify map construction occurs once for
  the stage rather than once per eye per frame.
- Verify mixed shapes, left/right shape mismatch, and unreadable input fail
  before completion.

### Canonicalization

- Compare serial and threaded canonical PNGs using dtype, shape, and exact array
  equality across all depth representations and more than one scene.
- Verify output names and preview frame numbers remain in source order even when
  workers finish out of order.
- Inject a worker failure and verify metadata is not written.
- Verify a completed existing canonical stage is still reused without work.

### VR assembly

- Compare threaded side-by-side and over-under outputs against serial reference
  arrays for matching and mismatching input sizes.
- Verify the same-size path does not call resize.
- Verify progress and preview ordering under deliberately out-of-order worker
  completion.
- Verify one failed decode/write prevents `complete_stage`.
- Verify existing completed VR stages remain reusable.

### Regression and benchmark verification

- Run the complete unit-test suite after targeted tests pass.
- Re-run the representative real-frame benchmarks outside timing-sensitive unit
  tests. Performance thresholds are evidence for release review, not CI
  assertions.
- Re-run a short end-to-end video with keep-intermediates enabled, compare every
  optimized stage to the serial reference by decoded pixel hash, then compare
  final frame count, FPS, dimensions, audio presence, and successful decode.

## Acceptance Criteria

The implementation is accepted when all of the following hold:

- Every optimized transformed frame is pixel-identical to the serial reference.
- No-op crop output is byte-identical to its source and works through both hard
  link and forced-copy paths.
- Resume metadata and output naming remain compatible with existing batches.
- Worker count is bounded by frame count, CPU reserve, the eight-worker ceiling,
  and the memory-derived cap, with an unavoidable floor of one active item.
- Progress and preview callbacks occur on the caller thread in source order.
- A worker failure cannot produce a reusable stage.
- The full test suite passes.
- On the measured 1080p workload class, no-op crop completes in seconds rather
  than minutes, canonicalization is materially faster than 20.8 fps, and VR
  assembly is materially faster than 2 fps. Exact wall times are not contractual
  because source disks and CPU topology vary.

## Expected Impact

For the measured no-distortion, no-upscale job, no-op crop removes roughly 15
minutes of redundant decode/encode work. Eight-worker VR assembly demonstrated
more than a sixfold microbenchmark improvement, although HDD contention and
preview overhead will reduce end-to-end scaling. Canonicalization demonstrated
more than a threefold improvement at eight workers.

The fisheye path was not active in that job, but map reuse removes about 0.42
seconds of repeated coordinate construction per stereo pair at 1080p before
parallel remap and PNG I/O are considered.

These gains are additive to the already completed depth-inference optimization.
They retain frame-stage recovery and do not spend quality, change geometry, or
increase intermediate file sizes.

## Rejected Alternatives

### Direct FFmpeg layout and encode

Piping left/right frames through FFmpeg `hstack` or `vstack` would remove the VR
PNG write/read cycle and may be faster still. It also couples assembly to final
encoding, weakens reuse of `99_vr_frames`, and changes failure recovery. That is
an architectural optimization, not the selected conservative one.

### Faster encoder presets

Changing NVENC from the current quality-oriented preset can reduce encode time,
but preset and rate-control changes alter compression behavior and can reduce
quality. Encoding is therefore intentionally untouched.

### Real-ESRGAN batching or FP16

Batching both eyes may be useful when upscaling is enabled. FP16 cannot satisfy
the exact-output requirement, and the measured job had upscaling disabled.
Upscaler work should be measured and designed separately against perceptual or
numeric quality criteria.

### Stereo renderer fusion

The current renderer may be able to reuse source/depth transfers while creating
both eyes. That touches CUDA scheduling, memory lifetime, occlusion handling,
and a high-value image-quality boundary. It is excluded until it has its own
pixel-equivalence benchmark and failure analysis.

### Unlimited or user-configured worker counts

More threads can oversubscribe OpenCV, exhaust memory at high resolutions, and
reduce HDD throughput. A new setting also moves tuning burden to users. The
bounded automatic policy gives the measured gain without expanding the product
surface.

## Implementation Boundary

The later implementation plan may modify only the focused frame-stage modules,
the fisheye image helper/export, a small parallelism helper, and their tests:

- `processing/frames/depth_processor.py`
- `processing/frames/distortion_processor.py`
- `processing/frames/vr_assembler.py`
- `processing/frames/frame_stage_parallelism.py` as a new focused helper
- `utils/imaging/image_processing.py` and its existing export modules
- Focused unit tests for those modules and the helper

No UI, setting schema, encoder, estimator, renderer, or stage-manifest schema
change belongs in this implementation.
