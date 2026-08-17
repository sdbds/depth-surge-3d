# VDPP Canonical Calibration Fix

## Status

Approved direction, revised after PRO review. This document narrows and
supersedes the v1 VDPP input and output range policy in
`2026-08-16-vdpp-temporal-postprocessor-design.md`. It does not change depth
estimation, base canonical disparity, or stereo rendering contracts.

## Problem

The released VDPP model predicts a non-negative residual and returns
`scaled_input + residual`. The current integration feeds scene-normalized
canonical disparity directly to the model and clips its uncalibrated output to
`[0, 1]`. Real output from the reported MoGe-2 sky shot reached `3.198`; an
existing stabilized artifact had mean `0.998`, so clipping destroyed depth
contrast and moved midpoint-coded sky pixels from `32768` to `65535`.

The upstream demo min/max-normalizes each input frame and then min/max-normalizes
the whole output video. The input normalization is part of the checkpoint's
actual operating domain. The final visualization normalization is not a valid
render contract here because it would force every shot to span `[0, 1]` and
silently redefine `stereo_strength`.

## Decisions And Invariants

1. Base `02_depth_raw` and `03_disparity_maps` files and fingerprints do not
   change.
2. VDPP keeps `0=far`, `1=near`; an accepted fit must have positive polarity.
3. Uint16 code `32768` is preserved by a **midpoint-code preservation
   heuristic**. It is not a validity mask and it is not a universal neutral or
   zero-parallax value; zero parallax depends on the configurable convergence.
4. The heuristic intentionally cannot distinguish MoGe invalid sky from a
   legitimate surface quantized to `32768`. Both are preserved bit-for-bit and
   excluded from normalization and fitting. This collision is the accepted
   compatibility cost of leaving the base canonical schema unchanged.
5. A true validity sidecar derived from raw depth would require a separate base
   schema, fingerprint, cache, and lifecycle design. Raw payloads are not a
   reliable VDPP input because `keep_intermediates=false` removes them before
   temporal stabilization and global canonical-cache restores may contain no raw
   payloads.
6. Calibration is one affine transform per finalized shot, never per frame or
   per inference window.
7. OLS preserves the fit-pair mean before clipping and minimizes squared error to
   base disparity. It does **not** guarantee exact source variance or dynamic
   range; fixed quality gates reject weak fits.
8. Resident working memory remains bounded independently of shot length. The
   private float32 memmap and final PNG payloads remain disk-backed and scale
   with shot length.
9. Old saturated stabilized artifacts are incompatible and must be regenerated;
   valid base disparity remains reusable.

## Ownership

- `VDPPTemporalPostprocessor` continues to own only the pinned 32-frame,
  four-overlap model recurrence. Its output remains native-resolution float32
  with no canonical clipping.
- `TemporalDepthStabilizer` owns base decoding, per-frame input normalization,
  pre-scan, stable statistics, shot fitting, quality gates, memmap lifecycle,
  encoded fallback, and progress.
- `StabilizedDepthStore.commit_shot()` accepts ordered native-shape `uint16`
  frames that are already final. The store hashes and atomically writes those
  exact values; it does not decode and re-encode them.

A semantic base fallback still commits a complete v2 stabilized artifact with
shot diagnostics. It never returns the base-path list or bypasses the requested
stabilized stage.

## Fixed Constants

The v2 producer uses these exact values:

```text
MIDPOINT_CODE = 32768
MODEL_MIDPOINT_VALUE = 0.5
STATS_TILE_PIXELS = 262144
MIN_PAIR_COUNT = 2
VARIANCE_EPSILON = 1e-12
MIN_POSITIVE_SCALE = 1e-8
MIN_CORRELATION = 0.50
MIN_POSTCLIP_CONTRAST_RATIO = 0.50
MAX_POSTCLIP_MEAN_DRIFT = 0.01
MAX_PRECLIP_OUT_OF_RANGE_FRACTION = 0.01
```

All variance values are population variances in decoded float64 canonical
units. `preclip_out_of_range_fraction` is the sum of the fractions below zero
and above one over fit-eligible pairs, before clipping.

The fixed gates are deliberately conservative. Eight read-only eight-frame
samples across the reported episode produced correlation and post-clip contrast
ratios of `0.5051..0.9734`, and pre-clip out-of-range fractions of
`0..0.009409`. A complete shot outside these bounds falls back to base rather
than publishing an uncertain stabilization.

## Bounded Pre-Scan

Before checkpoint resolution or model inference, scan one source PNG at a time
and retain only scalar counts:

- validate uint16 grayscale type, native shape, and frame order;
- count total pixels, exact midpoint-code pixels, and frames;
- count frames with at least two distinct non-midpoint uint16 codes;
- count individually flat frames: no non-midpoint pixels, or one unique
  non-midpoint code.

If every pixel is midpoint-coded, commit a bit-exact copy with mode
`all_midpoint`. If no frame has non-midpoint range, commit a bit-exact copy with
mode `base_fallback` and reason `source_no_range`. Both decisions happen before
CUDA preflight or model execution.

An individually flat frame inside an otherwise non-degenerate shot is fed to
VDPP as all `0.5` so recurrence remains ordered, but it is excluded from the fit
and copied bit-for-bit during final commit. VDPP is not allowed to invent
structure for that frame.

## Inference Pass

For each pending non-degenerate shot:

1. Create one private float32 memmap shaped `[shot_length, H, W]`.
2. The loader allocates exactly one float32 host window shaped
   `[requested_length, H, W]` and fills it in place. For each source frame:
   - compare the original uint16 values to `MIDPOINT_CODE`;
   - for a ranged frame, min/max-normalize only non-midpoint pixels to `[0, 1]`;
   - write `MODEL_MIDPOINT_VALUE` at midpoint-code pixels;
   - for a flat frame, fill the entire model input with
     `MODEL_MIDPOINT_VALUE`.
3. Run the existing pinned recurrence unchanged.
4. For each final `(shot_local_index, raw_output)` emitted by `process_shot()`:
   - require finite values and exact native shape;
   - write the frame once at that index in the raw memmap;
   - reread the corresponding base uint16 PNG;
   - if the base frame has range, merge non-midpoint `(raw, base / 65535)` pairs
     into the shot statistics in fixed row-major tiles.

Only final emitted frames enter statistics. Four-frame recurrence overlaps are
not counted twice.

## Stable Statistics And Fit

Statistics use a Chan merge in float64, not raw sums followed by large-number
subtraction. The state is:

```text
count, mean_x, mean_y, M2_x, M2_y, C_xy
```

For state `A` and the next fixed row-major tile `B`, with `n = n_a + n_b`:

```text
delta_x = mean_x_b - mean_x_a
delta_y = mean_y_b - mean_y_a
mean_x = mean_x_a + delta_x * n_b / n
mean_y = mean_y_a + delta_y * n_b / n
M2_x = M2_x_a + M2_x_b + delta_x^2 * n_a * n_b / n
M2_y = M2_y_a + M2_y_b + delta_y^2 * n_a * n_b / n
C_xy = C_xy_a + C_xy_b + delta_x * delta_y * n_a * n_b / n
```

Derived values are:

```text
variance_x = M2_x / count
variance_y = M2_y / count
covariance = C_xy / count
correlation = C_xy / sqrt(M2_x * M2_y)
scale = covariance / variance_x
shift = mean_y - scale * mean_x
```

Fallback is selected before fitting when `count < MIN_PAIR_COUNT`, or either
variance is at most `VARIANCE_EPSILON`. After fitting, non-finite parameters,
`scale < MIN_POSITIVE_SCALE`, or negative polarity select base fallback.

A non-finite model output is a hard model failure. Finite model output followed
by non-finite aggregate statistics or fitted parameters is a semantic
`base_fallback`, because the source frames remain valid but the calibration is
not identifiable.

## Quality Pass And Commit Pass

The affine cannot be accepted from OLS alone. Before writing final PNGs, scan
the raw memmap and source uint16 frames a second time in fixed tiles. Over the
same fit-eligible pairs, compute:

```text
preclip = raw * scale + shift
candidate = clip(preclip, 0, 1)
preclip_low_fraction
preclip_high_fraction
candidate_mean
candidate_std
postclip_contrast_ratio = candidate_std / source_std
postclip_mean_drift = abs(candidate_mean - source_mean)
```

Use OLS only when every condition holds:

```text
correlation >= 0.50
postclip_contrast_ratio >= 0.50
postclip_mean_drift <= 0.01
preclip_low_fraction + preclip_high_fraction <= 0.01
```

Otherwise the entire shot uses `base_fallback`. This quality scan is separate
from the commit scan so no candidate PNG is written before the fallback decision
is final.

The final commit pass yields already encoded uint16 frames:

- accepted ranged frames: apply the shot affine, clip to `[0, 1]`, encode, then
  restore every source `MIDPOINT_CODE` pixel to exact code `32768`;
- individually flat frames: copy the original uint16 frame;
- any shot fallback: copy every original uint16 frame.

Fallback is therefore pixel-level bit-exact, not decode/encode equivalent. The
shot manifest is published only after every encoded frame is atomically written
and hashed.

## Calibration Diagnostics

Shot manifest schema becomes version 2 and includes this self-hashed payload:

```json
{
  "calibration": {
    "mode": "ols | base_fallback | all_midpoint",
    "pair_count": 0,
    "midpoint_count": 0,
    "midpoint_fraction": 0.0,
    "flat_frame_count": 0,
    "source_mean": null,
    "source_std": null,
    "raw_mean": null,
    "raw_std": null,
    "covariance": null,
    "correlation": null,
    "scale": null,
    "shift": null,
    "candidate_mean": null,
    "candidate_std": null,
    "postclip_contrast_ratio": null,
    "postclip_mean_drift": null,
    "preclip_low_fraction": null,
    "preclip_high_fraction": null,
    "fallback_reason": null
  }
}
```

Unavailable numeric values are JSON `null`, never NaN or infinity. Fallback
reasons are stable enum strings: `source_no_range`, `too_few_pairs`,
`source_variance`, `raw_variance`, `nonfinite_statistics`, `nonfinite_fit`,
`nonpositive_scale`, `correlation`, `contrast`, `mean_drift`, or
`preclip_out_of_range`. Diagnostics participate in the shot manifest fingerprint
but not the stage semantic fingerprint.

## Work Files, Locking, And Cleanup

Private files live only under
`03_disparity_stabilized/.vdpp-work/shot_<id>.raw.f32.mmap`; cleanup never scans
or deletes outside that named directory. The production pipeline already owns a
job-level `JobWriterLock`; there is no independent stage writer lock. The
orchestrator passes the acquired job lock to the stabilizer so private-directory
mutation can be asserted.

Ordering is:

1. Perform read-only base, scene, and lightweight stabilized audits.
2. Under the already acquired job writer lock, remove known private work files,
   including on a complete-cache hit.
3. If the lightweight artifact is complete, return without CUDA checks.
4. For incomplete work, collect runtime identity and perform the runtime-aware
   audit to identify pending or invalid shots.
5. Run the bounded pre-scan over those shots. This fixes degenerate decisions
   before any model allocation and identifies which pending shots need a raw
   memmap.
6. Preflight disk for pending final PNGs and the largest pending non-degenerate
   shot.
7. Resolve the checkpoint only when at least one pending shot is non-degenerate,
   then load/preflight the model, prepare the store, and mutate pending shot
   state. If all pending shots are semantic copies, prepare and commit them
   without constructing VDPP.

Every memmap owner uses `try/finally`. Before unlink on Windows it flushes the
mapping, releases all slices/views, explicitly closes the underlying mapping,
and then unlinks. A cleanup failure is reported but cannot replace an already
active model, cancellation, decode, or disk exception. With no active exception,
cleanup failure is surfaced.

## Disk And Memory Bounds

For pending work, define:

```text
frame_u16_bytes = H * W * 2
conservative_png_bound = ceil(frame_u16_bytes * 1.10)
pending_frames = frames in pending/invalid shots after runtime-aware audit
largest_pending_shot = max pending non-degenerate shot length, or 0

required_bytes = ceil(1.10 * (
    pending_frames * conservative_png_bound
    + largest_pending_shot * H * W * 4
    + conservative_png_bound
    + 1 MiB metadata/manifest allowance
))
```

The second term is the largest raw float32 memmap and the third is one atomic PNG
temporary. Reusable shot payloads are not charged again.

The conservative host-resident array bound is:

```text
normalized input window:          32 * H * W * 4
decoded source uint16:             1 * H * W * 2
midpoint boolean mask:             1 * H * W * 1
native raw output:                 1 * H * W * 4
calibrated float32 + uint16:       1 * H * W * (4 + 2)
Chan tile scratch:                 at most 8 * STATS_TILE_PIXELS * 8 bytes
```

The loader fills the normalized host window in place; an unnormalized float32
window must not coexist with it. GPU memory retains the existing pinned VDPP
recurrence bound. Tests instrument owned arrays and prove that a longer shot
changes disk size, not peak resident array shape.

## Cache Identity

The stabilized producer algorithm becomes `vdpp-canonical-shot-v2`. Its
execution plan records all behavior-changing strings and constants:

```text
input_normalization = "per-frame-minmax-excluding-midpoint-code-v2"
midpoint_code_policy = "preserve-u16-32768-heuristic-v2"
calibration_fit = "positive-ols-chan-v1"
degenerate_policy = "prescan-flat-frame-copy-u16-v1"
fit_quality_policy = "corr-contrast-mean-drift-v1"
clip_policy = "preclip-fraction-gate-then-clip-v1"
base_fallback_policy = "copy-source-u16-v1"
calibration_precision = "float64-chan-float32-output-v1"
midpoint_code = 32768
model_midpoint_value = 0.5
stats_tile_pixels = 262144
min_pair_count = 2
variance_epsilon = 1e-12
min_positive_scale = 1e-8
min_correlation = 0.50
min_postclip_contrast_ratio = 0.50
max_postclip_mean_drift = 0.01
max_preclip_out_of_range_fraction = 0.01
```

Runtime identity additionally records `numpy_version` and `opencv_version`.
Partial shots generated by a different NumPy/OpenCV runtime are not mixed. The
algorithm, execution plan, and runtime changes invalidate v1 stabilized
artifacts without changing any depth-model setting or base canonical cache key.

## Verification

Automated tests must first fail against v1 behavior and then prove:

1. Per-frame non-midpoint min/max normalization matches the pinned upstream
   input policy while midpoint-code pixels receive model value `0.5`.
2. A legitimate source pixel quantized to `32768` demonstrates the documented
   heuristic collision: it is counted and preserved, never claimed as validity.
3. Chan tile statistics match a direct float64 reference on low-variance and
   long synthetic data without cancellation-induced gate changes.
4. A deterministic fake VDPP with a large positive residual produces an
   accepted, non-saturated affine result when quality gates pass.
5. `count < 2`, source/raw zero variance, too-small or negative scale,
   non-finite fit, and every quality-gate failure select a bit-exact whole-shot
   base fallback with the expected stable reason.
6. Non-finite model output, decode, shape, ordering, cancellation, and disk
   faults remain hard failures; cleanup never hides the primary exception.
7. Individually flat frames in a normal shot are model-fed as `0.5`, excluded
   from statistics, and copied bit-for-bit in an otherwise accepted shot.
8. A 61-frame synthetic shot proves overlap frames enter statistics once,
   midpoint preservation crosses the 32-frame boundary, and one affine applies
   without a boundary jump. Existing recurrence-equivalence tests remain
   unchanged.
9. Instrumentation proves the host bound above, memmaps close before Windows
   unlink, stale work is removed on cache hits, and disk preflight uses pending
   shots plus the largest pending memmap.
10. Shot diagnostics are self-hashed; threshold, policy, NumPy, and OpenCV
    identity changes invalidate or prevent mixed partial resume as specified.

The implementation must commit `scripts/verify_vdpp_calibration.py`; it drives
the reported real artifact as a local, non-CI quality gate:

```text
job basename:
  1786983915_BanG Dream Its MyGO S01E03-[1080p][BDRIP][AV1.OPUS]_20260818_002515
base canonical fingerprint:
  8292b1291fe6c552fc3843deadbbf5657efe5a1d734f48f1ef3389dc4253cf75
source raw fingerprint:
  b7529c0a7e348ab599ef1446015d210022603e89d3ff960ae419f7aa50f8cc65
scene manifest fingerprint:
  08defe1eee63a921191df2db93e1c71637548233df30e051c194e5e911f3967a
frame_000500.png SHA-256:
  cc19d33a8dbceee4056ac0e141980228e335e6d3fb6c930a42c7f185791cecb8
frame 500 shot range, zero-based:
  [207, 562)
frame 500 non-midpoint source mean/std:
  0.05490560770408996 / 0.027351981535192124
frame 500 midpoint count/fraction at 608x1080:
  522561 / 0.7958104897660818
```

The verifier rejects mismatched input hashes, reads the shot diagnostics, and
requires: shot mean drift at most `0.01`, correlation at least `0.50`, post-clip
standard-deviation ratio in `[0.50, 1.05]`, pre-clip out-of-range fraction at
most `0.01`, every frame-500 source midpoint code preserved as `32768`, and a
non-degenerate accepted result that is not pixel-identical to base. Base and raw
fingerprints must remain unchanged. The copyrighted episode is not added to the
repository; synthetic CI fixtures cover the same logic.
