# VDPP Canonical Calibration Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace VDPP's saturating direct clip with deterministic shot-global calibration that preserves midpoint-coded pixels, publishes strictly validated v2 diagnostics, and keeps inference memory bounded independently of shot length.

**Architecture:** A new Torch-free `core.vdpp_calibration` module owns every numerical constant, reducer, Chan merge, canonical derived diagnostic, fallback decision, and strict reader/writer validator. `TemporalDepthStabilizer` pre-scans base PNGs, normalizes one model window at a time, stores raw VDPP output in a private memmap, runs separate fit/quality/commit passes, and gives final uint16 frames plus canonical diagnostics to `StabilizedDepthStore`. One shared shot-record auditor is used by partial storage audit, resume preflight, and complete render validation. A lightweight output-root helper removes stale private memmaps immediately after the writer lock is acquired or accepted.

**Tech Stack:** Python 3.10+, NumPy 1.x/2.x, OpenCV, PyTorch/VDPP at the inference boundary only, pytest, uint16 PNG, float32 `numpy.memmap`.

## Global Constraints

- Base raw depth and canonical disparity payloads and identities remain byte-compatible.
- Producer becomes `vdpp-canonical-shot-v2`; v1 stabilized artifacts are invalidated without changing base caches.
- Exact arithmetic order, positive-zero canonicalization, planned-tile ULP policy, and diagnostics derivation follow the approved specification literally.
- A complete stabilized cache is reusable without CUDA. Incomplete work remains CUDA-only, but copy-only pending shots do not load the checkpoint or model.
- No production behavior is added until its focused test fails for the expected reason.
- The external copyrighted fixture remains a non-CI sign-off blocker until `PENDING_RECAPTURE` is replaced honestly.

---

### Task 1: Deterministic Calibration Core

**Files:**
- Create: `src/depth_surge_3d/core/vdpp_calibration.py`
- Create: `tests/unit/test_vdpp_calibration.py`

- [x] Add RED tests for exact constants, frame normalization, candidate arithmetic, empty/one-value reducers, the 17-value two-pass fixture, sequential Chan merges, positive zero, planned-tile ULP boundaries, the 4,096-tile variance fixture, and the canonical correlation one-ULP fixture.
- [x] Implement immutable scalar/pair states, exact tile reducers, merge functions, bounded normalization, and canonical moment finalization.
- [x] Add RED tests for every fallback boundary/reason and strict exact-key/nullability/derived-formula validation, including self-hashed tampering inputs and `-0.0`.
- [x] Implement canonical diagnostics construction and strict validation using persisted variances and `float.hex()` equality.
- [x] Run `tests/unit/test_vdpp_calibration.py` and keep the module free of Torch imports.

### Task 2: V2 Execution And Runtime Identity

**Files:**
- Modify: `src/depth_surge_3d/inference/depth/vdpp_contract.py`
- Modify: `src/depth_surge_3d/core/render_disparity.py`
- Modify: `tests/unit/test_render_disparity.py`
- Modify: `tests/unit/test_temporal_stabilizer.py`

- [x] Add RED tests for the exact v2 execution-plan policy fields/constants and NumPy/OpenCV runtime fields.
- [x] Import calibration constants into plan construction rather than duplicating literals; bump the producer algorithm to v2.
- [x] Prove v1 complete and partial artifacts are incompatible while base canonical artifacts remain accepted.

### Task 3: Final-U16 Store And Shared Shot Audit

**Files:**
- Modify: `src/depth_surge_3d/core/render_disparity.py`
- Modify: `src/depth_surge_3d/processing/frames/temporal_storage.py`
- Modify: `tests/unit/test_temporal_storage.py`
- Modify: `tests/unit/test_render_disparity.py`

- [x] Add RED tests showing `commit_shot()` accepts only ordered native-shape uint16 frames and validates canonical diagnostics before any payload write.
- [x] Add manifest schema v2 diagnostics/fingerprint and JSON `allow_nan=False`; write exact uint16 payloads without decode/re-encode.
- [x] Extract a tri-state shared shot-record audit and use it for partial store audit and complete render validation.
- [x] Add tamper tests that recompute all surrounding hashes yet are rejected for derived fields, missing/unknown keys, non-canonical bounds, and negative zero.

### Task 4: Bounded Coordinator Calibration Pipeline

**Files:**
- Modify: `src/depth_surge_3d/processing/frames/temporal_stabilizer.py`
- Modify: `tests/unit/test_temporal_stabilizer.py`

- [x] Add RED tests for bounded pre-scan, all-midpoint/source-no-range early copy, flat-frame model input/copy, exact float32 per-frame min/max, and deferred checkpoint/model construction.
- [x] Implement pre-scan records, window normalization, raw memmap lifecycle, exact fit accumulation, fallback ordering, separate quality scan, and final accepted/fallback uint16 iterators.
- [x] Add RED end-to-end fake-VDPP tests for a large residual, a 61-frame overlap shot, each fallback reason, hard model/decode/order failures, and midpoint preservation.
- [x] Instrument array/memmap bounds and update disk preflight to include the largest pending non-degenerate memmap and atomic PNG allowance.

### Task 5: Stale Private-Work Cleanup At Every Entrypoint

**Files:**
- Create: `src/depth_surge_3d/vdpp_work.py`
- Modify: `src/depth_surge_3d/cli.py`
- Modify: `app.py`
- Modify: `src/depth_surge_3d/rendering/stereo_projector.py`
- Modify: `src/depth_surge_3d/processing/orchestration/pipeline_orchestrator.py`
- Create: `tests/unit/test_vdpp_work.py`
- Modify: relevant entrypoint tests

- [x] Add RED tests for lock ownership, idempotence, known-file cleanup, unknown/symlink/junction refusal, and a fresh-interpreter no-Torch import.
- [x] Implement root-derived, lock-verified cleanup of only `.vdpp-work/shot_<id>.raw.f32.mmap` regular files.
- [x] Call it as the first job-private mutation after lock acquisition/acceptance in CLI, Web, projector, and direct orchestrator paths, including cache hits and VDPP-off runs.

### Task 6: Resume Uses The Shared Audit

**Files:**
- Modify: `src/depth_surge_3d/io/resume.py`
- Modify: `tests/unit/test_resume.py`

- [x] Add RED tests for valid, invalid-diagnostics, missing-record, and structural-corruption building artifacts.
- [x] Invoke the shared read-only shot audit after stage identity/state checks; report valid records as provisional pending runtime comparison and invalid records as regeneration work.
- [x] Prove complete validation, partial store audit, and resume preflight classify the same records consistently.

### Task 7: Independent Final-PNG Quality Verifier

**Files:**
- Create: `scripts/verify_vdpp_calibration.py`
- Modify: `tests/unit/test_vdpp_quality_harness.py`

- [x] Add RED synthetic tests for final-PNG source/output moments, pair/midpoint/flat counts, all midpoint locations, endpoint counts, non-identity, and quantization tolerances.
- [x] Implement an independent reducer/merge path that does not import producer accumulators.
- [x] Encode the reported 355-frame fixture and fail before CUDA/model loading while its ordered payload fingerprint is `PENDING_RECAPTURE`; provide no bypass.

### Task 8: Full Verification And Branch Finish

- [x] Run focused calibration, storage, stabilizer, render, resume, cleanup, and entrypoint tests with the repository venv on `PATH`.
- [x] Run the complete unit suite and relevant static/format checks.
- [x] Inspect `git diff --check`, status, and scoped diff; verify no base-depth/cache identities changed.
- [x] Record the external fixture as the only remaining sign-off blocker unless it has been recaptured during implementation.

### Task 9: Review Follow-Up For Runtime-Bound Partial Resume

- [x] Add a deterministic numeric behavior probe covering the pinned two-pass reducers and sequential Chan merge, then persist it with interpreter/platform and NumPy/OpenCV identity.
- [x] Prove a building artifact resets for the same NumPy version with a different probe and for a different NumPy version with the same probe.
- [x] Document the conservative OpenCV version policy: decoded uint16 pixels are semantic, while PNG byte hashes provide integrity rather than cross-runtime encoder identity.
- [x] Reuse one full-frame normalization mask and instrument its in-place reversal.
- [x] Rerun focused and complete verification, then publish the review fix on the existing branch.
