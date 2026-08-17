# VDPP Canonical Calibration Fix

## Status

Approved design. This document narrows and supersedes the v1 VDPP input and
output range policy in
`2026-08-16-vdpp-temporal-postprocessor-design.md`. It does not change depth
estimation, base canonical disparity, or stereo rendering contracts.

## Problem

The released VDPP model predicts a non-negative residual and returns
`scaled_input + residual`. The current integration feeds scene-normalized
canonical disparity directly to the model and clips its uncalibrated output to
`[0, 1]`. Real output from the reported MoGe-2 sky shot reached `3.198`; an
existing stabilized artifact had mean `0.998`, so clipping destroyed depth
contrast and moved neutral sky pixels from `0.5` to `1.0`.

The upstream demo avoids saturation by min/max-normalizing each input frame and
then min/max-normalizing the whole output video. The input normalization is part
of the checkpoint's actual operating domain. The final visualization
normalization is not a valid render contract here because it would force every
shot to span `[0, 1]` and silently redefine `stereo_strength`.

## Invariants

1. Base `02_depth_raw` and `03_disparity_maps` files and fingerprints do not
   change.
2. VDPP keeps `0=far`, `1=near`, and a positive shot-wide calibration scale.
3. An input pixel encoded as exact canonical midpoint `32768 / 65535` is a
   neutral anchor and remains that exact value after stabilization.
4. Calibration is one affine transform per finalized shot, never per frame or
   per inference window.
5. Working memory remains bounded independently of shot length.
6. Old saturated stabilized artifacts are incompatible and must be regenerated;
   valid base disparity remains reusable.

## Data Flow

For each finalized shot:

1. Decode each requested base canonical frame as float32.
2. Mark exact uint16 value `32768` as the neutral-anchor mask.
3. Compute min and max over non-anchor pixels in that frame. Map those pixels to
   `[0, 1]` before VDPP. Keep anchor pixels at `0.5`. A frame with no non-anchor
   range is filled with `0.5`.
4. Run the existing pinned 32-frame, four-overlap VDPP recurrence unchanged.
5. Stream each native-resolution raw VDPP output into a shot-local float32
   memmap. While streaming, pair non-anchor raw outputs `x` with their original
   base canonical values `y` and accumulate float64 sufficient statistics.
6. Solve one least-squares affine transform for the shot:

   ```text
   scale = covariance(x, y) / variance(x)
   shift = mean(y) - scale * mean(x)
   calibrated = raw * scale + shift
   ```

7. Require finite `scale` and `shift` and `scale > 0`. Restore neutral anchors to
   exact `0.5`, clip only final numerical outliers to `[0, 1]`, encode uint16,
   and commit through the existing shot-atomic store.
8. Delete the memmap in `finally` on success, failure, or cancellation.

The calibration deliberately matches the shot's source disparity location and
scale while allowing VDPP to change spatial structure and temporal behavior.

## Degenerate Shots

Calibration must not invent depth or invert polarity.

- If every pixel is a neutral anchor, emit exact `0.5` for the shot without
  using fitted parameters.
- If the original non-anchor disparity has no range, emit the original base
  canonical frames.
- If the VDPP output variance is numerically zero, or the fitted parameters are
  non-finite or non-positive, emit the original base canonical frames.
- Model, decode, shape, ordering, or disk failures remain hard failures. They do
  not silently fall back to base disparity.

The base fallback is a semantic fallback for a mathematically unidentifiable
calibration, not an error-recovery path.

## Storage And Memory

The transient memmap is stored below the stabilized stage in a private work
directory and is never included in metadata or artifact hashes. It contains at
most one shot of float32 native-resolution VDPP output. RAM retains only the
existing VDPP window/overlap state plus one native output, one base frame, one
mask, and float64 scalar statistics.

Disk preflight includes:

```text
all final uint16 stabilized frames
+ the largest pending shot as float32
+ atomic-write and 10% filesystem allowance
```

Stale private work files are removed before generation and ignored by resume
validation.

## Cache Identity

The stabilized producer algorithm becomes `vdpp-canonical-shot-v2`. Its
execution plan records:

```text
input_normalization = "per-frame-minmax-excluding-neutral-u16-midpoint-v2"
output_range_policy = "shot-global-affine-to-base-then-clip-v2"
neutral_anchor_policy = "preserve-u16-midpoint-v1"
calibration_precision = "float64-statistics-float32-output"
```

Changing the stabilized algorithm and execution plan invalidates v1 VDPP
artifacts. No depth-model setting or base canonical cache key changes.

## Verification

Automated tests must first fail against v1 behavior and then prove:

1. A fake VDPP forward that adds a large positive residual does not produce a
   saturated artifact after shot calibration.
2. Output ordering and near/far polarity remain positive.
3. Exact midpoint anchors survive bit-for-bit.
4. Calibration uses one shot-wide fit rather than per-frame fits.
5. Constant, all-anchor, zero-variance, non-finite, and negative-scale cases use
   the specified fallback or hard failure.
6. Peak retained arrays are independent of shot length, transient work files are
   removed on every exit path, and disk preflight includes the memmap bound.
7. `vdpp-canonical-shot-v1` metadata is rejected while base canonical metadata
   remains valid.
8. Existing upstream recurrence-equivalence tests continue to pass because the
   model recurrence itself is unchanged.

The real reported frame-500 quality gate uses the existing MoGe-2 base artifact:

- sky anchor remains `32768`;
- stabilized mean is below `0.95`;
- fewer than 1% of non-anchor pixels clip to either endpoint;
- non-anchor correlation with base disparity is positive;
- base and raw artifact hashes remain unchanged.

