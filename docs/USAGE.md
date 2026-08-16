# Usage Guide

## Web UI

On Linux or macOS:

```bash
./run_ui.sh
```

On Windows:

```powershell
.\run_ui.ps1
```

Open `http://localhost:5000`. The Web UI exposes the same depth, scene, stereo,
projection, and retention settings as the CLI.

## Command Line

```bash
# Basic conversion
python depth_surge_3d.py input_video.mp4

# A short test clip with stronger depth
python depth_surge_3d.py input_video.mp4 --start 00:30 --end 00:45 \
  --stereo-strength 2.5

# Put the canonical value 0.6 on the display plane
python depth_surge_3d.py input_video.mp4 --convergence 0.6

# High-resolution over-under output
python depth_surge_3d.py input_video.mp4 --format over_under \
  --vr-resolution 16x9-4k

# Keep rectilinear stereo images
python depth_surge_3d.py input_video.mp4 --no-distortion

# Use float32 raw storage when a model exceeds float16 range
python depth_surge_3d.py input_video.mp4 --raw-storage-dtype float32

# Remove restartable working files only after validated output
python depth_surge_3d.py input_video.mp4 --no-intermediates
```

Run `python depth_surge_3d.py --help` for every option and its current default.

### MoGe-2 Examples

Install the optional extra first with `uv sync --extra moge2`. Relative
geometry is the default:

```bash
uv run depth-surge-3d clip.mp4 --depth-model-version moge2 --model-size vitb
```

The Experimental metric-camera path uses flat rectilinear side-by-side output:

```bash
uv run depth-surge-3d clip.mp4 --depth-model-version moge2 --model-size vitb \
  --stereo-geometry-mode metric_camera --format side_by_side --no-distortion \
  --virtual-baseline-mm 63 --metric-convergence-distance auto \
  --max-disparity-percent 2
```

The six public selection and projection flags are `--depth-model-version
moge2`, `--model-size`, `--stereo-geometry-mode`, `--virtual-baseline-mm`,
`--metric-convergence-distance`, and `--max-disparity-percent`. Metric mode
requires square source SAR `1:1`; missing or `N/A` SAR is normalized to `1:1`,
while malformed or explicit non-square SAR is rejected. Letterbox bars remain
part of the picture and can bias automatic convergence even when SAR is square.

Automatic convergence is one persisted clip-global value resolved before
metric stereo or preview. The disparity cap is applied as total left-to-right
disparity in retained final-output coordinates before the shared center crop
and resize. Final output width does not alter that projection fraction.

## Output Structure

Each conversion uses a self-contained output directory:

```text
output/video_timestamp/
|-- 00_original_frames/
|   `-- metadata.json
|-- 01_scene_data/
|   |-- scene_manifest.json
|   |-- depth_samples.npz
|   `-- depth_bounds.json
|-- 02_depth_raw/
|-- 03_disparity_maps/
|-- 03_metric_geometry/
|-- 04_left_frames/
|-- 04_right_frames/
|-- 05_left_distorted/
|-- 05_right_distorted/
|-- 06_left_cropped/
|-- 06_right_cropped/
|-- 07_left_upscaled/
|-- 07_right_upscaled/
|-- 99_vr_frames/
|-- <batch>-settings.json
`-- output.mp4
```

Only the selected Stage 3 is derived. A compatible inactive
`03_disparity_maps` or `03_metric_geometry` stage is preserved during mode
switches. With `--no-intermediates`, raw frame NPZ files are deleted only after
the selected Stage 3 payloads and metadata validate. If later rendering fails,
that completed Stage 3 remains available for resume. Full intermediate cleanup,
including payloads from both Stage-3 directories, runs only after final output
is successfully finalized.

## Resume

```bash
python depth_surge_3d.py --resume ./output/video_timestamp/
```

Resume validates the selected settings file, its recorded source-video path and
SHA-256, the extraction-time frame-content fingerprint, contiguous frame names,
loaded estimator artifact identity, metadata, and decoded payload shape and
dtype. Extracted source frames remain reusable across downstream schema changes
when those checks pass. A candidate scene manifest cannot be used for
canonicalization; final scene bounds are required.

If an explicit float16 run encounters an out-of-range value, widen completed
raw files in place and continue without re-inferring them. Canonical disparity
and stereo output are then rebuilt against the widened raw fingerprint:

```bash
python depth_surge_3d.py --resume ./output/video_timestamp/ \
  --raw-storage-dtype float32
```

Generated directories from the old schema are archived by default:

```bash
python depth_surge_3d.py --resume ./output/old_job/ --migrate-legacy archive
```

Delete them only with an explicit request:

```bash
python depth_surge_3d.py --resume ./output/old_job/ --migrate-legacy delete
```

The resume report lists every preserved, resumed, invalidated, archived, or
deleted stage. Destructive migration starts only after the configured estimator
loads and its exact fingerprint has been checked.

Relative and metric geometry use independent fingerprints. Switching modes
selects the matching Stage 3 without interpreting relative disparity as metric
geometry or deleting a still-valid inactive stage. Metric stereo begins only
after its clip-global convergence metadata is finalized.

Stereo renderer v3 invalidates v1 left/right metadata and all tracked
downstream frame stages while preserving valid source, scene, raw-depth, and
canonical data. An existing encoded video is not a tracked frame stage and is
not implicitly deleted; the resumed run encodes its new result through the
normal output path.

## Viewing

The final video can be opened in a player or headset that supports side-by-side
or over-under stereo video. Match the player layout to `--format`.
