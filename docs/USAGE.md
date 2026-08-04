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

Optional stages may be empty. With `--no-intermediates`, restartable working
payloads are removed only after their replacement output has been written and
validated.

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

## Viewing

The final video can be opened in a player or headset that supports side-by-side
or over-under stereo video. Match the player layout to `--format`.
