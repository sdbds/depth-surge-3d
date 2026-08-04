# Torch DIBR Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace inverse remap and colour-derived hole filling with one depth-aware Torch forward-splat renderer used by preview and production.

**Architecture:** Canonical disparity is resized before target-width pixel disparity is computed. Both eyes render sequentially in deterministic complete-row bands with primary depth voting, no-voter fallback, one-sided visibility, and in-band horizontal background filling. Production frame I/O uses a host-memory semaphore shared across queued and active work.

**Tech Stack:** PyTorch 2.x, NumPy, OpenCV, pytest, CUDA when available.

## Global Constraints

- `x_left=x+d/2`, `x_right=x-d/2`, and both eyes use total signed `d` as z-key.
- Primary z votes require bilinear weight `>=0.5`; no-voter targets fall back to all-contribution `amax`.
- Visibility is `d >= z_win - 0.25` in pixel-disparity units.
- GPU temporary budget is 256 MiB with `SPLAT_BYTES_PER_PIXEL=192`; no full-frame GPU intermediate exists.
- Host stereo payload budget is 512 MiB with 16 bytes per pixel plus 1 MiB per slot.

---

### Task 1: Low-level Forward Splat

**Files:**
- Create: `src/depth_surge_3d/rendering/forward_splat.py`
- Test: `tests/unit/test_forward_splat.py`

**Interfaces:**
- Produces: `forward_splat_band(image, disparity, eye_sign) -> SplatBandResult`.

- [ ] **Step 1: Write failing CPU tests for translation, sign, bilinear weights, out-of-frame holes, near collision, fractional foreground edge blend, no-voter fallback, 20-pixel ramp coverage, and shared z-key in each eye**
- [ ] **Step 2: Run the test module and confirm import failure**
- [ ] **Step 3: Implement scatter indexes, primary/all z reductions, one-sided visibility, colour accumulation, and explicit masks**
- [ ] **Step 4: Run CPU tests green and CUDA parity tests when CUDA is available**
- [ ] **Step 5: Commit with `feat: add depth-aware Torch forward splat`**

### Task 2: In-band Background Fill and Renderer

**Files:**
- Create: `src/depth_surge_3d/rendering/stereo_renderer.py`
- Modify: `src/depth_surge_3d/rendering/__init__.py`
- Test: `tests/unit/test_stereo_renderer.py`

**Interfaces:**
- Produces: `StereoRenderer.render(frame, canonical, settings) -> StereoRenderResult` with host images and masks.

- [ ] **Step 1: Write failing tests for resize-before-disparity, target-width scaling, farther-candidate fill, run-width limit, valid black pixels, deterministic bands, and no full-frame GPU buffers**
- [ ] **Step 2: Run the test module and confirm expected failure**
- [ ] **Step 3: Implement bilinear canonical resize, row-band sizing, sequential eyes, and band-scoped horizontal fill**
- [ ] **Step 4: Implement one half-band OOM retry and stable second-failure diagnostics**
- [ ] **Step 5: Run renderer tests green and commit with `feat: add bounded Torch stereo renderer`**

### Task 3: Bounded Production I/O Pipeline

**Files:**
- Modify: `src/depth_surge_3d/processing/frames/stereo_generator.py`
- Test: `tests/unit/test_stereo_generator.py`

**Interfaces:**
- Consumes: `StereoRenderer`, canonical files/metadata, `stereo_io_workers`.
- Produces: a shared lifecycle permit count calculated from the 512 MiB host budget.

- [ ] **Step 1: Write failing tests for worker validation, 4K slot calculation, maximum-worker memory bound, resume skips, atomic writes, and queue/permit release on errors**
- [ ] **Step 2: Run focused tests and verify failures against the multiprocessing renderer**
- [ ] **Step 3: Replace the process pool with decoder/writer threads and a shared bounded semaphore; keep CUDA in the main process**
- [ ] **Step 4: Run stereo generator tests green and commit with `feat: add bounded stereo I/O pipeline`**

### Task 4: Preview and Single-image Migration

**Files:**
- Modify: `src/depth_surge_3d/rendering/stereo_projector.py`
- Modify: `app.py`
- Test: `tests/unit/test_stereo_projector.py`
- Test: `tests/unit/test_see_through_entrypoints.py`

**Interfaces:**
- Consumes: estimator `DepthBatch`, pure single-scene canonicalization, `StereoRenderer`.

- [ ] **Step 1: Write failing tests proving image and Web preview use the same renderer and corrected sign**
- [ ] **Step 2: Run focused tests and verify the old remap calls fail expectations**
- [ ] **Step 3: Route single-image and preview paths through canonicalization and `StereoRenderer`**
- [ ] **Step 4: Run focused tests green and commit with `feat: unify preview and production stereo rendering`**

### Task 5: Remove Old Renderer and Benchmark

**Files:**
- Modify: `src/depth_surge_3d/utils/imaging/image_processing.py`
- Create: `scripts/benchmark_stereo_renderer.py`
- Test: `tests/unit/test_image_processing.py`
- Test: complete suite

**Interfaces:**
- Removes: `depth_to_disparity`, `create_shifted_image`, black-pixel hole masking, OpenCV stereo inpainting.

- [ ] **Step 1: Add failing absence tests for production imports and legacy helper names**
- [ ] **Step 2: Remove migrated helpers and callers**
- [ ] **Step 3: Add a non-gating 1080p/4K benchmark reporting GPU render time, peak CUDA memory, wall time, FPS, writer utilization, and queue stalls**
- [ ] **Step 4: Run the complete suite and benchmark smoke mode**
- [ ] **Step 5: Commit with `refactor: remove inverse-remap stereo path`**

