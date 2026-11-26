# Web UI Guide

## Local Access

Open `http://127.0.0.1:5000` on the computer running the UI. Use `--port`
to choose another port. The server binds to loopback only; HTTP and live
progress connections accept the local UI's own origin, including its port.
There is no authenticated LAN or multi-user mode. Do not publish the UI via
a reverse proxy or port forwarding.

Uploads create separate jobs under `output/`, even for repeated filenames.
For Open, Analyze, Stitch, Process, or Resume, select an individual job folder
directly inside `output/`, not the collection, a stage subfolder, or another
location. Jobs that read or write artifacts must contain ordinary files and
directories, not symbolic links or Windows junctions. Keep this folder under
the desktop user's control. The Web model selector uses registered presets;
use the CLI for custom checkpoints.

Stitch saves in the selected job. Give it a video filename (for example,
`second edit.mp4`), not a destination path. An existing name, including the
source movie, is never replaced. Wait for any current writer to finish first.

Polling progress or refreshing the page does not shut down media processes.
Stop applies to the current conversion; closing the server stops its own
work without terminating other desktop applications' encoders.

## Stereo Renderer

The Stereo Pair Generation step exposes `Fast` and `Quality`. New jobs default
to Fast on every execution device. Quality remains an explicit offline choice,
defaults to an 8 px local fill limit at 1080p, and accepts `1..32`. Selecting
Quality with an effective CPU device displays a warning rather than silently
downgrading the job.

The resume panel defaults to the saved stereo mode. Choosing Fast or Quality is
an explicit override and invalidates incompatible downstream stereo artifacts.
Leaving the Quality fill-limit field empty preserves its saved value.

Committed Quality diagnostics are stored under
`04_stereo_diagnostics/`. Frame manifests authenticate coverage and repair PNGs
plus scalar statistics. Incomplete generations are not reused.

## Temporal Post-Processing

The Depth Analysis step exposes one segmented control:

```text
Off | VDPP (Experimental)
```

`Off` preserves the historical pipeline. `VDPP` adds a depth-only temporal
stage after canonical disparity and before stereo. It has no strength, window,
overlap, precision, or resolution controls because those values are fixed by
the released checkpoint.

Starting a new VDPP job requires an effective CUDA device. The first generation
downloads roughly 111 MiB of weights and adds processing time and one uint16 PNG
per source frame. V2 already has shot-aware model-native temporal inference, so
VDPP is mainly intended for evaluating framewise V3, See-Through, and relative
MoGe-2 output. It is not available with `metric_camera` geometry.

## Resume

Completed jobs are read-only for Resume. To create a different version, start
a new job; resuming will not migrate an already published one.

The resume selector defaults to `Use saved setting`, not `Off`. This prevents a
visual default from silently changing a persisted VDPP job. Jobs discovered by
the Web UI hydrate the selector with their validated saved value; selecting
`Off` or `VDPP` explicitly sends an override.

A complete stabilized artifact is content-verified before device checks. Such a
job can resume on CPU without loading CUDA, VDPP, its checkpoint, or the base
depth estimator. Missing or partial VDPP output still requires CUDA. Only one
Web or CLI writer may own an output directory; another writer fails immediately
and reports the recorded process owner.

With VDPP enabled, progress has a separate `Temporal Depth Stabilization` step.
Checkpoint progress reports bytes, and shot generation reports finalized frames
out of total frames. An interrupted shot is recomputed from its beginning while
earlier committed shots remain reusable.
