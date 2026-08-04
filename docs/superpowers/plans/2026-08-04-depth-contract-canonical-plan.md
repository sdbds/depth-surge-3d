# Depth Contract and Canonical Disparity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace frame-normalized estimator arrays with an explicit depth representation contract and deterministic scene-level canonical disparity files.

**Architecture:** Estimators return native-resolution `DepthBatch` values. A scene pre-pass and file-backed raw stage establish immutable scene bounds, then a pure canonicalizer writes schema-validated uint16 disparity PNGs. The existing remap renderer temporarily consumes signed pixel disparity through a narrow adapter so this slice remains independently runnable.

**Tech Stack:** Python 3.9+, NumPy, OpenCV, pytest, compressed NPZ, JSON metadata.

## Global Constraints

- No estimator performs per-frame min/max normalization or source-frame upsampling.
- `METRIC_DEPTH` uses reciprocal, `INVERSE_DEPTH` passes through, and `RELATIVE_DEPTH` negates finite values including zero.
- Raw inference completes globally before scene finalization and canonicalization.
- Canonical float32 is `0=far`, `0.5=neutral`, `1=near`; uint16 encoding uses `np.rint(value * 65535)`.
- All metadata and frame payload writes use temporary files followed by atomic replacement.
- Production processing is file-backed; the legacy array API rejects depth payloads above 512 MiB.

---

### Task 1: Estimator Output Contract

**Files:**
- Create: `src/depth_surge_3d/inference/depth/types.py`
- Modify: `src/depth_surge_3d/inference/depth/__init__.py`
- Modify: `src/depth_surge_3d/inference/depth/video_depth_estimator.py`
- Modify: `src/depth_surge_3d/inference/depth/video_depth_estimator_da3.py`
- Modify: `src/depth_surge_3d/inference/depth/video_depth_estimator_see_through.py`
- Test: `tests/unit/test_depth_contract.py`
- Test: existing estimator unit tests

**Interfaces:**
- Produces: `DepthRepresentation(Enum)` and `DepthBatch(values: np.ndarray, representation: DepthRepresentation)`.
- Produces: every `estimate_depth_batch(...) -> DepthBatch` at native model-output resolution.

- [ ] **Step 1: Write failing contract and adapter-direction tests**

```python
def test_relative_zero_is_valid_and_nearer_after_canonicalization():
    batch = DepthBatch(np.array([[[0.0, 0.8]]], np.float32), DepthRepresentation.RELATIVE_DEPTH)
    assert batch.values.dtype == np.float32
    assert batch.representation is DepthRepresentation.RELATIVE_DEPTH
```

- [ ] **Step 2: Run `pytest tests/unit/test_depth_contract.py -q` and confirm failure because the contract module is absent**
- [ ] **Step 3: Implement the enum/dataclass and change each adapter to return unnormalized native float32 values with its declared representation**
- [ ] **Step 4: Run the contract and three estimator test modules until green**
- [ ] **Step 5: Commit with `feat: add explicit depth output contract`**

### Task 2: Pure Canonicalization and Encoding

**Files:**
- Create: `src/depth_surge_3d/processing/frames/depth_normalizer.py`
- Test: `tests/unit/test_depth_normalizer.py`

**Interfaces:**
- Consumes: `DepthRepresentation`.
- Produces: `SceneDepthBounds(low: float, high: float)`.
- Produces: `depth_to_score(values, representation) -> tuple[np.ndarray, np.ndarray]`.
- Produces: `canonicalize_depth(values, representation, bounds) -> np.ndarray`.
- Produces: `encode_canonical_png(values) -> np.ndarray` and `decode_canonical_png(values) -> np.ndarray`.

- [ ] **Step 1: Write failing tests for all three representations, non-finite values, metric zero, relative zero, flat bounds, clipping, and midpoint encoding**
- [ ] **Step 2: Run `pytest tests/unit/test_depth_normalizer.py -q` and confirm expected import failure**
- [ ] **Step 3: Implement score conversion and stateless canonicalization; invalid and flat values return float32 `0.5`**
- [ ] **Step 4: Implement deterministic uint16 encode/decode and run the test module green**
- [ ] **Step 5: Commit with `feat: add pure scene-bound depth canonicalization`**

### Task 3: Deterministic Scene Analysis and Finalization

**Files:**
- Create: `src/depth_surge_3d/processing/frames/scene_analyzer.py`
- Test: `tests/unit/test_scene_analyzer.py`

**Interfaces:**
- Produces: `analyze_scenes(frame_files, output_dir, *, enabled, threshold, min_frames) -> dict` with `status="candidate"`.
- Produces: `sample_scene_depths(raw_files, manifest, representation) -> dict[int, np.ndarray]`.
- Produces: `finalize_scenes(candidate_manifest, samples) -> tuple[dict, dict[int, SceneDepthBounds]]`.

- [ ] **Step 1: Write failing tests for deterministic cuts, disabled detection, minimum scene length, zero-span merge, left-to-right fixed-point merge, pooled sample order, and candidate rejection**
- [ ] **Step 2: Run `pytest tests/unit/test_scene_analyzer.py -q` and verify the module is missing**
- [ ] **Step 3: Implement 32-bin luma histogram analysis and atomic candidate manifest writing**
- [ ] **Step 4: Implement at-most-32-frame and 64x64-grid sampling plus deterministic fixed-point merging and final bound recomputation**
- [ ] **Step 5: Run scene tests green and commit with `feat: add deterministic scene depth bounds`**

### Task 4: Raw Storage, Fingerprints, Promotion, and Disk Budget

**Files:**
- Create: `src/depth_surge_3d/processing/frames/depth_storage.py`
- Test: `tests/unit/test_depth_storage.py`

**Interfaces:**
- Produces: `build_model_fingerprint(estimator, settings, native_shape, dtype, provenance) -> dict`.
- Produces: `RawDepthStore` with `write_batch`, `validate_resume`, `promote_to_float32`, `load`, and `complete_files`.
- Produces: `estimate_depth_disk_bytes(...)` and `require_disk_space(...)`.

- [ ] **Step 1: Write failing tests for auto dtype choice, explicit float16 rejection, compressed atomic files, metadata validation, semantic fingerprint rejection, resumable promotion, and no-reinference promotion**
- [ ] **Step 2: Run `pytest tests/unit/test_depth_storage.py -q` and verify failure**
- [ ] **Step 3: Implement canonical JSON hashing, atomic JSON/NPZ writes, ready/promoting metadata, and strict resume validation**
- [ ] **Step 4: Implement ordered float16-to-float32 promotion with crash recovery and distinct provenance**
- [ ] **Step 5: Implement uncompressed worst-case disk preflight and run storage tests green**
- [ ] **Step 6: Commit with `feat: add resumable native depth storage`**

### Task 5: File-backed Depth Processor

**Files:**
- Modify: `src/depth_surge_3d/processing/frames/depth_processor.py`
- Modify: `src/depth_surge_3d/core/constants.py`
- Modify: `src/depth_surge_3d/io/operations.py`
- Modify: `src/depth_surge_3d/utils/domain/depth_cache.py`
- Test: `tests/unit/test_depth_processor.py`
- Test: `tests/unit/test_depth_cache.py`
- Test: `tests/unit/test_io_operations.py`

**Interfaces:**
- Consumes: estimator `DepthBatch`, `RawDepthStore`, scene manifest/final bounds, canonicalizer.
- Produces: `generate_depth_map_files(...) -> list[Path]` pointing to `03_disparity_maps/*.png` with required local metadata.

- [ ] **Step 1: Add failing processor tests for native raw files, global barrier, candidate/final crash resume, metadata ownership, clean/resume identity, retention, and 512 MiB array rejection**
- [ ] **Step 2: Run the focused processor/cache/I/O tests and confirm the new expectations fail**
- [ ] **Step 3: Add `01_scene_data`, `02_depth_raw`, and `03_disparity_maps` directories and rewrite the file-backed processor around the new stages**
- [ ] **Step 4: Update cache save/restore to copy canonical metadata and reject schema/fingerprint mismatches**
- [ ] **Step 5: Run focused tests green and commit with `feat: build file-backed canonical disparity pipeline`**

### Task 6: Temporary Legacy Renderer Adapter and Pipeline Wiring

**Files:**
- Modify: `src/depth_surge_3d/processing/frames/stereo_generator.py`
- Modify: `src/depth_surge_3d/processing/orchestration/pipeline_orchestrator.py`
- Test: `tests/unit/test_stereo_generator.py`
- Test: `tests/unit/test_pipeline_orchestrator.py`

**Interfaces:**
- Consumes: canonical uint16 files and metadata.
- Produces: signed pixel disparity `d=(r-0.5)*render_width*0.02` for the temporary inverse-remap implementation.

- [ ] **Step 1: Write failing tests that require canonical metadata, use signed target-width disparity, and assert near foreground has `x_left > x_right`**
- [ ] **Step 2: Run the focused tests and verify failures in the old physical-depth path**
- [ ] **Step 3: Replace depth-scale decoding and `depth_to_disparity` calls with the narrow canonical-to-signed-pixel adapter**
- [ ] **Step 4: Wire the orchestrator to the canonical directory and run focused tests green**
- [ ] **Step 5: Run the complete test suite and commit with `feat: route pipeline through canonical disparity`**

