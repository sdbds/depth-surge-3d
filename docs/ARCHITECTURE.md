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
interpolation before pixel disparity is computed:

```text
d = (r - convergence) * target_width * stereo_strength / 100
left_target_x  = source_x + d / 2
right_target_x = source_x - d / 2
```

Both eyes use the same signed `d` as the depth key. A banded CUDA forward splat
performs bilinear horizontal scattering. Each target column first elects a
near-surface winner from contributions with weight at least `0.5`; when no such
vote exists, all contributions participate. Visibility is the one-sided test
`d >= winner - 0.25 px`, preserving low-weight antialiasing tails without
letting them control the depth vote.

Invalid splat pixels are represented by explicit masks, never inferred from
color. `background` occlusion fill propagates farther visible pixels
horizontally within each row and band. It cannot reconstruct unseen texture;
wide gaps can become a constant-color run from the selected background edge.

## Memory And I/O

CUDA scatter bands use a deterministic byte budget. The per-source-pixel budget
includes color/value tensors, weights, projected disparity, and int64 scatter
indices. Background fill executes inside each full-row band, so no full-frame
GPU buffer remains except the final uint8 eye outputs.

Stereo decode and output work is bounded by both `stereo_io_workers` and a host
byte budget. End-to-end throughput measurements include decode, render, image
encoding, and writes; GPU kernel time is reported separately.

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

A depth or stereo schema change does not by itself invalidate valid
`00_original_frames`. Raw model changes invalidate raw depth and every
downstream stage. Canonical changes invalidate canonical disparity and
downstream stages. Render-setting changes invalidate stereo output and
downstream stages.

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
- `rendering/stereo_projector.py`: banded CUDA forward splat and occlusion fill.
- `io/resume.py`: stage validation, reports, and legacy migration.
