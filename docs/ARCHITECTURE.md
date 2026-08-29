# Technical Architecture

## Pipeline

Depth Surge 3D converts monocular video through restartable, fingerprinted
stages:

```text
source video
  -> 00_original_frames
  -> 01_scene_data (candidate cuts, final segments, samples, bounds)
  -> 02_depth_raw (native model output and representation metadata)
  -> global barrier: all raw inference complete
  -> selected Stage 3:
       relative: 03_disparity_maps (canonical relative disparity)
         -> optional 03_disparity_stabilized (VDPP, one transaction per final shot), or
       03_metric_geometry (metric inverse depth, validity, focal data)
  -> 04_left_frames + 04_right_frames (forward-splat DIBR)
  -> optional projection, crop, and upscale stages
  -> 99_vr_frames
  -> encoded video with optional source audio
```

The frame, scene, raw-depth, canonical, and stereo stages have separate
metadata. `00_original_frames/metadata.json` anchors the exact encoded PNG
bytes to the source-video SHA-256 after extraction completes. A stage is reused
only when its schema, frame manifest, settings, model identity, and upstream
fingerprints match.

VDPP is a derived artifact stage, not an estimator backend. It consumes the
same canonical uint16 relative disparity produced by V2, V3, See-Through, and
relative MoGe-2.
The fixed adapter uses 32-frame windows, four retained observations, stride 28,
FP32, and the upstream affine continuation rule. A window never crosses a final
scene boundary. The depth-model owner is released before VDPP construction,
and VDPP is released before stereo rendering, so both neural models are never
resident together.

## Depth Contract

Every estimator returns a `DepthBatch` containing float32 values and one
explicit `DepthRepresentation`:

- `METRIC_DEPTH`: positive physical distance; converted with a safe reciprocal.
- `INVERSE_DEPTH`: near-is-large score; passed through.
- `RELATIVE_DEPTH`: affine-relative distance; converted with linear negation.

Non-finite values are invalid. Non-positive values are invalid only for metric
physical distance. Estimators never perform per-frame min/max normalization.

The backend registry is the single owner of backend variants, capabilities,
factories, availability, and report identity. A `DepthBatch` may also carry a
typed `PinholeCameraBatch`; MoGe-2 supplies one normalized horizontal focal
value per frame. Existing raw schema v2 depth-only data remains readable. New
raw schema v3 writes exact depth payload membership plus camera-model metadata;
`pinhole_fx` frames atomically store depth and a scalar float32 focal value.

Raw output is stored at model-native resolution. The first inference chunk
selects `float16` or `float32` storage for the whole raw directory and records
the choice. A preflight estimate checks disk space before long inference.
Remote DA3 and See-Through repositories are resolved to one immutable Hub
snapshot before loading; that resolved artifact identity is part of the raw
fingerprint.

## Relative And Metric Geometry

Relative mode remains the default and derives `03_disparity_maps`. Experimental
metric-camera mode derives `03_metric_geometry`. These are independent Stage-3
directories: only the selected missing stage is generated, and a valid inactive
stage is preserved. The metric store holds native-resolution inverse depth,
depth validity, and normalized horizontal focal data. When intermediates are
retained, it also holds one viewable uint8 depth-preview PNG per frame; these
PNGs are diagnostics and are not projection inputs.

Both paths produce a common `StereoGeometryFrame` containing near-score,
total-disparity fraction, and source-validity arrays. The common stereo
renderer therefore does not branch on depth-model identity. Metric projection
uses the virtual capture-camera baseline, focal estimate, depth, and one
convergence distance. `virtual_baseline_mm` is not viewer IPD.

The metric formula is evaluated in pre-crop source coordinates, transformed by
the exact retained center-crop width, and clamped as total left-to-right
disparity in retained final-output coordinates. It is then converted back to
renderer coordinates. The common center crop and axis-aligned resize run later,
so changing final output width does not alter the capped projection fraction.
Unclamped inverse depth remains the z-order key.

Every metric geometry build resolves and persists one deterministic
clip-global automatic convergence value from valid positive metric samples,
even when the active render requests an explicit distance. This finalized
metadata is a barrier: metric stereo and live preview do not start before it
exists. Baseline, explicit-versus-auto convergence, disparity cap, and render
width are stereo settings rather than metric-stage identity.

MoGe's predicted mask is depth confidence, not source-image alpha. Metric
projection excludes masked pixels from convergence and clamp statistics, gives
them the lowest visibility rank, and projects them on the infinite-background
plane. Their source colors therefore remain available to fill the background
instead of being replaced with black.

## Scene Canonicalization

Scene cuts are analyzed from source-frame luma histograms before depth
canonicalization. Candidate manifests carry `status: candidate`; only a
`status: final` manifest may drive canonical output.

After all raw frames exist, deterministic depth samples are pooled per candidate
segment. Adjacent segments are merged left-to-right, repeatedly until stable.
Merged samples are the union of their child sample sets. Final 2nd/98th
percentiles are then computed once for each final segment.

Canonicalization is a pure function of raw depth, representation, final scene
ID, and final bounds. Output is relative disparity in `[0, 1]`, where `1` is
near. Empty and flat scenes produce `0.5`. Canonical maps use `uint16` PNG and
store exact upstream fingerprints in `metadata.json`.

## Stereo Renderer

Canonical relative disparity is resized to the render target with bilinear
interpolation. The host then computes both full-frame eye offset maps once in
float64 and narrows them to signed int32 fine-lane offsets:

```text
q = (r - convergence) * target_width * stereo_strength / 200
left_offset  = ceil(16 * q - 0.5)
right_offset = ceil(-16 * q - 0.5)
```

Each source pixel represents one projected horizontal unit footprint. The
renderer preserves its occupancy with 16 fixed samples: source column `x`
writes lanes `16*x + offset + {0,...,15}`. Every fine lane has one strict
z-buffer winner. The ordering key packs nonnegative float32 disparity bits and
the inverted full-frame source index into one int64 value:

```text
key = (float32_bits(r) << 32) | (0xffffffff - source_index)
```

One integer `amax` therefore selects the largest disparity and, for exact depth
ties, the lowest source index. There is no epsilon visibility window,
floating-point atomic sum, or fixed layer count. CPU and CUDA reduction order
cannot change the winner.

Invalid splat pixels are represented by explicit masks, never inferred from
color. `background` occlusion fill runs on the fine grid and extends the farther
of the two discrete boundary winners across bounded horizontal gaps. Equal
depth uses distance and then the left boundary as tie-breaks. `none` leaves
unresolved lanes black. Sixteen lanes are combined with a fixed balanced
addition tree, multiplied by `0.0625`, and converted with ties-to-even rounding.
The public valid mask means at least one lane was covered before fill; the
public hole mask means all 16 lanes remain unresolved after fill.

## Memory And I/O

CUDA Fast scatter first attempts two bounded Triton kernels. The first computes
the 16 packed winner candidates per source pixel directly into the winner
buffer with integer atomic max; the second decodes and gathers the winners
without expanded target, candidate, or decoded-index tensors. A Triton import
or compilation failure disables this path for that CUDA device and preserves
the byte-identical PyTorch scatter fallback.

CUDA Quality visibility reuses the same packed winner reduction, but its
default compact path does not gather the general fine-grid disparity, validity,
or winner-index planes. A Quality-specific Triton kernel reads the 16 winners
for each output pixel, accumulates the exact uint8 BGR sum, multiplies by the
binary-exact `1/16`, and emits only the downsampled float32 BGR value plus one
packed missing-lane mask. The existing native histogram and background repair
consume that mask unchanged. Triton unavailability or compilation failure
falls back to the general byte-identical Quality band path.

Both paths use a deterministic byte budget. On an RTX 4090, the Triton live set
measured `855.819` temporary bytes per band source pixel during a complete 4K
render. Its configured `1088 B/source-pixel` retains 27.6 percent measured
headroom and permits 64 complete rows, or 34 bands per eye, under the 256 MiB
temporary budget. The PyTorch fallback retains the conservative
`1280 B/source-pixel` bound, 54 rows, and 40 bands per eye because it still
materializes expanded indexes, masks, and packed candidates.

The two full-frame int32 eye maps are built from host float64 geometry; all
float64 working tensors are released when map construction returns. Fast CUDA
rendering then attempts one bounded preload of the source image, near score,
source-valid mask, and both int32 eye maps so both eyes can reuse them without
per-band host transfers. For a 4K uint8 relative frame this static set is about
126.6 MiB and remains separate from the 256 MiB scatter-temporary budget. If
the preload cannot be allocated, every partial allocation is released and the
renderer falls back to slicing the host arrays for each band. Both paths are
byte-identical. CPU rendering always uses the band-sliced path.

Background fill and downsampling still complete before the next band, final
eye outputs remain host arrays, and no full-frame expanded index, winner, or
16-lane buffer is retained on the device.

Stereo decode and output work is bounded by both `stereo_io_workers` and a host
byte budget. End-to-end throughput measurements include decode, render, image
encoding, and writes; renderer median, p95, mean, host geometry, and offset-transfer
costs are reported explicitly.

Relative Quality jobs use a transient final-output path only when direct VR
encoding is enabled, intermediates are not retained, upscaling is disabled,
schema-5 publication is active without a recovery reservation, and distortion
plus crop can be reproduced exactly in memory. Rendered pairs are released in
source order through the existing bounded pipeline, transformed and assembled
with the same pixel operations as the file-backed stages, encoded to lossless
PNG in memory, and written to FFmpeg through `image2pipe`. The exact piped PNG
bytes are authenticated by the encoding-input manifest. This path never writes
or rereads `04_left_frames`, `04_right_frames`, or cropped RGB PNGs. Unsupported
settings, retained intermediates, and metric geometry continue through the
file-backed pipeline. Recovery first completes an evidence-bound streamed-v3
sibling when one is fully authenticated; all other recoverable states use the
file-backed pipeline.

Transient Quality diagnostics use one ordered JSONL extent and publish the same
aggregate summary and completion metadata without per-frame diagnostics files.
FFmpeg stdin, progress, and diagnostics are drained with fixed-size buffers;
pipe backpressure propagates to the stereo lifecycle permits, so memory remains
bounded by configured pipeline capacity rather than frame count.

The legacy in-memory depth API rejects projected outputs over 512 MiB. The main
video pipeline remains file-backed.

Metric-camera disk preflight covers both missing raw payloads and the complete
metric store. The native output estimate is checked before inference and the
preflight is repeated after the first stored batch if its measured shape or
selected dtype differs. The metric-stage bound does not assume compression:

```text
A              = max(4096, filesystem allocation unit)
payload_bound  = sum(5 * height * width)
metadata_bound = max(16 MiB, A * frame_count)
atomic_overlap = ceil(1.25 * largest_frame_payload) + A
required_bytes = ceil(1.25 * payload_bound) + metadata_bound + atomic_overlap
```

The five bytes per pixel are float32 inverse depth plus Boolean validity. A
64-KiB allocation unit is used when the filesystem value cannot be queried.
Preflight cannot reserve capacity against other writers. Post-preflight
`ENOSPC` removes only the current temporary write, keeps prior committed frames
and writing metadata resumable, withholds completion metadata, and reports the
preflight bound, current free bytes, and failing path.

## Resume And Migration

Resume decisions are deterministic and reported before mutation. The selected
settings file is fixed for the whole operation. Web resume resolves only the
source path and SHA-256 recorded by that file. Source frames require the saved
source-video SHA-256, their extraction-time content fingerprint, a contiguous
frame manifest, matching dimensions, and readable payloads. Raw reuse
additionally requires the exact fingerprint of the loaded estimator and
validates every persisted NPZ. Canonical and stereo PNGs are decoded and checked
for their declared dtype and shape before reuse.

Settings schema v4 combines the MoGe geometry settings with
`temporal_postprocessor`. V1-v3 jobs migrate upward with a missing temporal mode
set to `off`; v4 is parsed strictly, and future schemas fail before any job
mutation. Resume distinguishes an omitted override from an explicit `off`.

Every writer takes a fail-fast OS lock before the authoritative audit and holds
it through migration, model execution, downstream writes, and cleanup. A
complete stabilized artifact is validated from stored identities, shot
manifests, and per-file SHA-256 values before any CUDA or model probe. It can
therefore render on a machine without CUDA, the VDPP checkpoint, or the
historical base estimator. Incomplete VDPP work resumes at shot granularity;
an interrupted long shot is recomputed from its beginning.

A depth or stereo schema change does not by itself invalidate valid
`00_original_frames`. Raw model changes invalidate raw depth and every
downstream stage. Canonical changes invalidate canonical disparity and
downstream stages. Render-setting changes invalidate stereo output and
downstream stages.

Relative and metric Stage-3 fingerprints are independent. A mode switch
selects the matching stage, preserves a compatible inactive stage, and
invalidates stereo and downstream output without interpreting one geometry
format as the other. Metric stereo fingerprints also include crop width, SAR,
baseline, requested/effective convergence, and disparity cap.

For `keep_intermediates=false`, raw frame NPZ files are removed only at the
validated selected-Stage-3 barrier. A later render or encode failure keeps that
completed Stage 3 for resume. Full successful-finalization cleanup removes all
intermediate payloads, including payloads from both Stage-3 directories;
cleanup does not run after a failed job.

The 16-sample renderer changes the stereo algorithm identity from v1 to v3.
Resuming v1 metadata preserves source, scene, raw-depth, and canonical stages,
then regenerates stereo and every tracked downstream frame stage. Encoded video
files are outside that frame-stage invalidation and are not implicitly deleted.

Old generated directories are archived under `legacy_v1/` by default.
Non-interactive runs never delete them implicitly; deletion requires
`--migrate-legacy delete`. Removed keys found in an on-disk settings file are
listed and stripped during migration, while the same keys supplied explicitly
are validation errors. Web migration is deferred to the processing thread until
the model fingerprint is available. Failed archive/settings transactions move
stages back to their original paths. Explicit deletion commits after stage
movement and settings migration; staging cleanup is post-commit garbage
collection, so an interrupted cleanup is reported without an impossible partial
rollback.

## Main Components

- `depth_surge_3d.py`: CLI, validation, and resume entry point.
- `app.py`: Flask and Socket.IO Web UI.
- `processing/frames/depth_processor.py`: scene, raw-depth, canonical, and cache
  orchestration.
- `inference/depth/backend_registry.py`: backend variants, capabilities,
  construction, availability, and report identity.
- `processing/frames/metric_geometry.py`: restartable metric Stage 3, disk
  bound, and clip-global convergence.
- `processing/frames/depth_normalizer.py`: representation conversion and pure
  scene canonicalization.
- `processing/frames/stereo_generator.py`: bounded stereo I/O pipeline.
- `processing/video/video_encoder.py`: file-backed and bounded `image2pipe`
  encoding, schema-5 reservation, validation, and final publication.
- `processing/video/streamed_vr_manifest.py`: exact streamed PNG input evidence.
- `processing/frames/temporal_stabilizer.py`: VDPP artifact audit, shot resume,
  disk/device preflight, and file-backed generation.
- `inference/depth/vdpp_temporal_postprocessor.py`: exact bounded 32/4 neural
  recurrence and explicit model lifecycle.
- `core/render_disparity.py`: strict base/stabilized producer validation.
- `rendering/forward_splat.py`: packed 16-lane z-buffer for one row band.
- `rendering/stereo_renderer.py`: host geometry, fine-grid fill, downsampling,
  and bounded eye rendering.
- `rendering/stereo_geometry.py`: common relative and metric renderer input.
- `io/resume.py`: stage validation, reports, and legacy migration.
