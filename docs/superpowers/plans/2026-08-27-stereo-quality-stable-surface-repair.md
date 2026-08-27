# Stereo Quality Stable-Surface Repair Implementation Plan

**Goal:** Fix the independent NVENC and DIBR artifact sources while reducing
the Quality renderer's cost.

**Spec:** `docs/superpowers/specs/2026-08-27-stereo-quality-stable-surface-repair-design.md`

## Constraints

- Keep exactly the public `fast` and `quality` modes.
- Preserve Fast output and algorithm identity.
- Preserve the 16-lane visibility evidence and valid foreground lanes.
- Keep host allocations below the existing 512 MiB preflight.
- Treat encoded-video QA and pre-encode PNG QA as separate gates.

## Completed Work

- [x] Add an exact `hevc_nvenc` CQ18 quality contract and unit test.
- [x] Prove the low-bitrate encoder contribution by comparing decoded default,
  CQ18, and source PNG pixels.
- [x] Reject transition-depth endpoints, split disconnected stable components,
  and snap the full RGB edge band to stable native geometry.
- [x] Add a one-pixel, displacement-gated foreground matte expansion for outer
  antialiasing pixels.
- [x] Replace the v9 second splat and global donor planner with a one-splat,
  missing-lane-only background proxy repair.
- [x] Move row-bounded lane-hole histograms into the existing native extension.
- [x] Update the Quality identity to
  `torch-horizontal-16x-rgb-geodesic-repair-v10` and update memory preflight.
- [x] Add synthetic regression tests for geometry, matte, proxy composition,
  donor exclusion, native histograms, one-splat routing, identity, and CQ18.
- [x] Render and inspect frames 89, 111, 171, 176, 231, 301, and 401 before
  final encoding.
- [x] Record cold selected-frame core timings of 2.56 through 4.58 seconds,
  below the existing 5.0-second fixture gate and substantially below v9.
- [x] Run the focused Quality, diagnostics, resume, CLI, Web, and encoder suite:
  527 tests passed.

## Remaining Integration

- [x] Run the complete repository test suite: 2956 passed and 17 skipped; the
  sole failure is the unchanged environment-only packaged CLI lookup, which
  also fails on `main` because `depth-surge-3d` is not installed on `PATH`.
- [x] Run compile, focused lint, and diff checks.
- [ ] Commit the isolated branch.
- [ ] Integrate into dirty `main` without staging or overwriting unrelated user
  changes.
