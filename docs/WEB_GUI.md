# Web UI Guide

## Stereo Renderer

The Stereo Pair Generation step exposes `Quality` and `Fast`. Quality is the
offline highest-quality path and defaults to an 8 px local fill limit at 1080p;
the input accepts `1..32`. Fast retains the historical renderer. Selecting
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
