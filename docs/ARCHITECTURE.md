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
  -> 03_disparity_maps (canonical relative disparity)
  -> optional 03_disparity_stabilized (VDPP, one transaction per final shot)
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
same canonical uint16 relative disparity produced by V2, V3, and See-Through.
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

Raw output is stored at model-native resolution. The first inference chunk
selects `float16` or `float32` storage for the whole raw directory and records
the choice. A preflight estimate checks disk space before long inference.
Remote DA3 and See-Through repositories are resolved to one immutable Hub
snapshot before loading; that resolved artifact identity is part of the raw
fingerprint.

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

CUDA scatter bands use a deterministic byte budget. The measured live set
includes source color and canonical depth, transferred int32 offsets, expanded
target/source indexes and in-bounds masks, packed candidates and winners,
gathered fine-grid color/depth/validity, fill indexes and selected color, and
scatter workspace. On an RTX 4090, the 16-sample renderer measured at most
`854.931` allocated bytes per source pixel during a complete 4K render. The
configured `1280 B/source-pixel` exceeds that measurement plus 25 percent
headroom. At 4K this permits 54 complete rows, or 40 bands per eye, under the
256 MiB temporary budget.

The two full-frame int32 eye maps remain host-resident and are sliced for each
band. They are built from host float64 geometry; all float64 temporaries are
released when map construction returns, before row-band device rendering
begins. Background fill and downsampling complete before the next band, and
final uint8 eye outputs are host arrays. No full-frame image, index, or 16-lane
buffer is retained on the device.

Stereo decode and output work is bounded by both `stereo_io_workers` and a host
byte budget. End-to-end throughput measurements include decode, render, image
encoding, and writes; renderer median, p95, mean, host geometry, and offset-transfer
costs are reported explicitly.

The legacy in-memory depth API rejects projected outputs over 512 MiB. The main
video pipeline remains file-backed.

## Resume And Migration

Resume decisions are deterministic and reported before mutation. The selected
settings file is fixed for the whole operation. Web resume resolves only the
source path and SHA-256 recorded by that file. Source frames require the saved
source-video SHA-256, their extraction-time content fingerprint, a contiguous
frame manifest, matching dimensions, and readable payloads. Raw reuse
additionally requires the exact fingerprint of the loaded estimator and
validates every persisted NPZ. Canonical and stereo PNGs are decoded and checked
for their declared dtype and shape before reuse.

Settings schema v3 adds `temporal_postprocessor`. V1/v2 jobs migrate upward
with the missing value set to `off`; v3 is parsed strictly, and future schemas
fail before any job mutation. Resume distinguishes an omitted override from an
explicit `off`.

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
- `processing/frames/depth_normalizer.py`: representation conversion and pure
  scene canonicalization.
- `processing/frames/stereo_generator.py`: bounded stereo I/O pipeline.
- `processing/frames/temporal_stabilizer.py`: VDPP artifact audit, shot resume,
  disk/device preflight, and file-backed generation.
- `inference/depth/vdpp_temporal_postprocessor.py`: exact bounded 32/4 neural
  recurrence and explicit model lifecycle.
- `core/render_disparity.py`: strict base/stabilized producer validation.
- `rendering/forward_splat.py`: packed 16-lane z-buffer for one row band.
- `rendering/stereo_renderer.py`: host geometry, fine-grid fill, downsampling,
  and bounded eye rendering.
- `io/resume.py`: stage validation, reports, and legacy migration.
