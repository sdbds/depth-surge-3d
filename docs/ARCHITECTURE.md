# Technical Architecture

## Pipeline

Depth Surge 3D converts monocular video through restartable, fingerprinted stages:

```text
source video
  -> 00_original_frames (extracted PNGs + metadata.json)
  -> 01_scene_data (candidate cuts, final segments, samples, bounds)
  -> 02_depth_raw (native model output and representation metadata)
  -> global barrier: all raw inference complete
  -> selected Stage 3:
       relative: 03_disparity_maps (canonical uint16 relative disparity)
         -> optional 03_disparity_stabilized (VDPP, one transaction per final shot), or
       metric_camera: 03_metric_geometry (metric inverse depth, validity, focal NPZ)
  -> 04_left_frames + 04_right_frames (forward-splat DIBR)
     + 04_stereo_diagnostics/summary.json (Quality/Fast aggregate diagnostics)
  -> optional projection, crop, and Real-ESRGAN upscale stages
  -> 99_vr_frames (when keep_intermediates=true) OR direct streamed bgr24 FFmpeg pipe
  -> encoded MP4 (hevc_nvenc / libx265 / libx264) with optional source audio
```

Each stage records its own metadata (`metadata.json`). A stage is reused on resume when its schema, frame manifest, settings, model identity, and upstream fingerprints match.

## Depth Contract & Backends

Every depth estimator returns a `DepthBatch` containing float32 values and one explicit `DepthRepresentation`:

- `METRIC_DEPTH`: positive physical distance; converted with a safe reciprocal.
- `INVERSE_DEPTH`: near-is-large score; passed through.
- `RELATIVE_DEPTH`: affine-relative distance; converted with linear negation.

Non-finite values are rejected; non-positive values are rejected for metric depth. Estimators never apply per-frame min/max normalization.

`inference/depth/backend_registry.py` is the single registry of backend variants, capabilities, factories, availability, and report identity:

- **Depth Anything V3 (`v3`)**: frame-by-frame relative/inverse depth with optional FlashAttention 2 (`attention_backend.py`) and adaptive VRAM chunking.
- **Video-Depth-Anything V2 (`v2`)**: shot-aware 32-frame sliding windows (10-frame overlap) with batched host-to-device input assembly (`_assemble_vda_input`) and automatic uniform-resolution fallback on CUDA OOM.
- **See-Through Marigold (`see_through`)**: opaque diffusion depth with CPU scheduler tables, uint8 GPU upload normalization, and batched VAE latent decoding.
- **MoGe-2 (`moge2`)**: pinned variants producing metric depth, confidence mask, and normalized horizontal focal length (`PinholeCameraBatch`); preprocesses via uint8 `cv2.INTER_AREA` downscaling before float32 conversion, and writes `03_metric_geometry` NPZ frames in parallel.

Raw depth is stored at model-native resolution in `02_depth_raw/` (raw schema v3 writes exact depth payload membership plus optional `pinhole_fx` focal scalars; schema v2 remains readable). Source frame decoding for the next chunk overlaps with GPU inference using a RAM-scaled prefetch budget (`RAW_DEPTH_PREFETCH_BUDGET_BYTES`, 512 MiB–2 GiB).

## Scene Canonicalization & Metric Geometry

Relative (`03_disparity_maps`) and experimental metric-camera (`03_metric_geometry`) are independent Stage-3 directories: only the selected missing stage is generated, and a valid inactive
stage is preserved. When `keep_intermediates=false`, raw frame NPZ files are removed only at the validated selected-Stage-3 barrier, keeping the completed Stage 3 for resume if stereo rendering or encoding fails later; full cleanup does not run after a failed job. Post-preflight
`ENOSPC` during metric-geometry writes preserves resumable metadata and reports the preflight bound, free bytes, and failing path.

### Relative Disparity (`03_disparity_maps`)
Scene cuts are detected from source-frame luma histograms (`01_scene_data/scene_manifest.json`). After all raw depth frames complete, deterministic depth samples are pooled per candidate segment, adjacent similar segments are merged left-to-right until stable, and final 2nd/98th percentiles are computed once per final shot. Canonical maps are encoded as `uint16` PNGs in `[0, 1]` (`1` = near, flat/empty = `0.5`) using parallel worker threads.

### Metric Camera Geometry (`03_metric_geometry`)
Experimental `metric_camera` mode persists native-resolution `inverse_depth`, `valid`, and scalar `focal_x_normalized` per frame in `zipfile(compresslevel=1)` NPZ files (`io/metric_npz.py`), plus one deterministic clip-global automatic convergence distance in `metadata.json`. Metric projection evaluates disparity in pre-crop coordinates, scales by the retained center-crop width, clamps total left-to-right disparity in final output coordinates, and keeps unclamped inverse depth as the z-order key.

### Optional VDPP Temporal Post-Processing (`03_disparity_stabilized`)
VDPP is a derived Stage-3 stabilizer that consumes canonical uint16 relative disparity from any backend. It operates strictly within final shot boundaries using 32-frame windows, 4-frame overlap, and stride 28 (`inference/depth/vdpp_temporal_postprocessor.py`):

- Runs with CUDA `bfloat16`/`float16` `autocast` when available (outputs normalized back to `float32`).
- Resizes and transfers each window to host memory in one batched `F.interpolate` + D2H copy.
- Validates CUDA working memory analytically in `preflight()` without running dummy neural passes.
- Caches decoded `uint16` source frames per shot (up to 512 MiB) in `TemporalDepthStabilizer` to avoid redundant PNG decodes across scan, inference, calibration, and commit passes.
- Depth estimators are unloaded before VDPP loads, and VDPP is released before stereo rendering.

## Stereo Renderers (`fast` and `quality`)

Both modes share a common `StereoGeometryFrame` and a 16-lane horizontal subpixel forward-splat z-buffer (`rendering/forward_splat.py`). Every fine lane resolves one strict winner using a packed `int64` key:

```text
key = (float32_bits(near_score) << 32) | (0xffffffff - source_index)
```

One integer `amax` selects the nearest surface and breaks exact ties deterministically by lowest source index.

### Fast Mode (`rendering/stereo_renderer.py`)
- Computes relative eye offsets on CUDA via a native FP32 Triton kernel (`rendering/fast_offsets_triton.py`, ~0.17 ms/frame at 4K) with CPU fallback.
- Scales default CUDA band memory dynamically up to 1.5 GiB (`free_vram // 4`, targeting ≤4 bands/eye) so 4K and 1080p frames minimize band-split kernel overhead.
- Performs bounded horizontal background-extension fill on the fine grid and balanced 16-lane downsampling.

### Quality Mode (`rendering/quality/` + `native/quality_kd_native.cpp`)
- Compiled C++17 pybind11 extension (`_quality_kd_native`, built with `/O2 /fp:precise` on Windows and `-O3` on Linux/macOS):
  - Reuses a persistent 7-worker thread pool (`PersistentParallelPool`, cleanly shut down via Python `atexit`) across all parallel native kernels, eliminating per-call OS thread creation overhead.
  - Builds the 4-connected edge band via a parallel separable O(H·W) L1 distance transform (exact for all radii; ~17 ms/frame at 4K).
  - Performs fused stable-surface region extraction, geodesic edge-locked region refinement, hybrid local Chebyshev-ring + implicit k-d tree geometry resampling, foreground matte expansion, and background-proxy disocclusion repair.
- Scales the Quality host memory preflight dynamically to physical RAM (`2–4 GiB` via `core/host_memory.py`), enabling 4K Quality renders without arbitrary 512 MiB host rejection while still guarding against runaway allocations.
- Scales default CUDA splat band budgets up to 1.5 GiB (`free_vram // 4`) and reuses a persistent single-thread left-eye repair executor.
- Vectorizes metric offset slabs and metric edge-band updates in NumPy (`rendering/quality/geometry.py`).
- Aggregates per-frame coverage and repair statistics in O(1) host memory (`StreamingDiagnosticsAccumulator`) and writes one `04_stereo_diagnostics/summary.json` at stage completion (`rendering/quality/diagnostic_transaction.py`).

## Memory, I/O, Encoding & Resume

- **Host memory budgeting (`core/host_memory.py`, `processing/frames/stereo_control.py`)**: Detects available physical RAM across Windows (`GlobalMemoryStatusEx`), Linux (`/proc/meminfo`), and POSIX (`os.sysconf`) to size the 3-stage bounded streaming stereo pipeline (decode → render → encode/write).
- **Portable atomic I/O (`io/durable_publish.py`, `io/nofollow_io.py`, `io/file_identity.py`)**: Uses standard `os.lstat`, `os.scandir(follow_symlinks=False)`, and buffered sibling-temp + `os.replace` writes (`sync=False` for per-frame intermediates, `sync=True` for settings/control/final outputs), working reliably across local NTFS/ext4/APFS as well as OneDrive, exFAT, SMB/NAS, Docker overlayfs, and WSL2.
- **Process locking (`io/job_lock.py`)**: Acquires a fail-fast OS advisory lock (`msvcrt.locking` / `fcntl.flock`) on the job lock file before writing owner metadata.
- **Zero-copy streamed encoding (`processing/video/video_encoder.py`)**: When intermediates are not kept, streams contiguous `uint8` BGR frames directly to FFmpeg via `-f rawvideo -pixel_format bgr24` (`stdin.write(contiguous.data)`) with zero intermediate PNG writes and zero BGR→RGB copies. Supports `hevc_nvenc`, `libx265`, and `libx264`.
- **Fast media verification (`processing/video/final_container_validation.py`, `final_audio.py`)**: Verifies encoded MP4 streams and source audio with a single metadata-only `ffprobe` call (no full-video software decode or multi-GB hashing) while preserving exact sample-accurate `atrim=end_sample=...` audio sync.
- **O(N) stage cleanup (`rendering/quality/cleanup_transaction.py`)**: Containment-checked `os.scandir(follow_symlinks=False)` cleanup after successful final video publication when `keep_intermediates=false`.
- **Schema-5 settings & resume (`io/settings_artifact.py`, `io/operations.py`, `io/resume.py`)**: Persists `processing-settings-v5.json` atomically. Validates upstream stage manifests and payloads on resume, migrates v1–v4 jobs forward, and archives invalidated legacy stages under `legacy_v1/` unless `--migrate-legacy delete` is requested.

## Main Components

- `depth_surge_3d.py` / `src/depth_surge_3d/cli.py`: CLI entry point, validation, and job lifecycle.
- `app.py` / `templates/index.html`: Flask + Socket.IO Web UI with live JPEG preview and progress ETA.
- `core/host_memory.py`: Cross-platform RAM detection and dynamic budget helpers.
- `core/settings.py` & `io/settings_artifact.py`: Settings validation and schema-5 serialization.
- `inference/depth/`: Backend registry, DA3, VDA V2, See-Through, MoGe-2, FlashAttention adapter, and VDPP post-processor.
- `processing/frames/depth_processor.py`: Scene analysis, raw depth inference, canonical disparity, and metric geometry stages.
- `processing/frames/temporal_stabilizer.py`: VDPP shot orchestration, bounded source caching, and calibration.
- `processing/frames/stereo_generator.py` & `stereo_control.py`: Unified Fast/Quality × Relative/Metric streaming stereo pipeline.
- `processing/video/`: `video_encoder.py`, `final_container_validation.py`, `final_audio.py`.
- `processing/orchestration/`: `pipeline_orchestrator.py` and `video_processor.py`.
- `rendering/`: `forward_splat.py`, `stereo_renderer.py`, `stereo_geometry.py`, `fast_offsets_triton.py`, `fast_splat_triton.py`, `fast_output_triton.py`, `stereo_projector.py`, and `rendering/quality/`.
- `native/quality_kd_native.cpp`: Compiled C++17 Quality stereo acceleration extension.
- `io/`: `operations.py`, `resume.py`, `job_lock.py`, `durable_publish.py`, `nofollow_io.py`, `file_identity.py`, `metric_npz.py`.
