# Lossless Frame-Stage Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove redundant frame transformations and parallelize independent CPU frame work while preserving decoded output pixels, resume metadata, and failure behavior.

**Architecture:** A focused helper selects a bounded worker count from frame count, CPU reserve, and an estimated one-GiB concurrency budget, then provides bounded in-flight scheduling with prompt failure detection and ordered caller-thread callbacks. Canonicalization, transformed crop, distortion, and VR assembly keep their existing stage ownership and pixel operations. Non-distorted factor-1.0 crop uses hard links with byte-exact copy fallback, and fisheye maps are built once per uniform stage.

**Tech Stack:** Python 3, `concurrent.futures`, `pathlib`, NumPy, OpenCV, pytest.

## Global Constraints

- Preserve decoded dtype, shape, and every pixel on the same NumPy/OpenCV backend.
- Preserve frame names, stage identity inputs, metadata schema, algorithm versions, bit depth, and final output shape.
- Keep progress and preview calls on the caller thread and consume results in source order.
- Use at most eight workers, reserve two logical CPUs, and cap concurrent estimated item memory at one GiB with a minimum of one worker.
- Base pair-stage estimates on the larger of source IHDR dimensions and target dimensions; target-only estimates are forbidden.
- On failure, cancel pending futures and complete `shutdown(wait=True, cancel_futures=True)` before returning or raising.
- Observe `FIRST_COMPLETED` independently of callback order, stop submission at
  the first failure, and buffer successful results by source index.
- Do not add a setting, dependency, encoder change, PNG-compression change, renderer change, or stage-manifest schema change.
- Do not write stage completion metadata until every scheduled item succeeds.
- Preserve unrelated pre-existing worktree changes. Commit commands below are checkpoints for a clean isolated worktree; do not stage a shared dirty file in the current workspace.

---

### Task 1: Bounded Frame-Stage Worker Policy

**Files:**
- Create: `src/depth_surge_3d/processing/frames/frame_stage_parallelism.py`
- Create: `tests/unit/test_frame_stage_parallelism.py`

**Interfaces:**
- Produces: `calculate_frame_stage_workers(frame_count: int, estimated_bytes_per_item: int, *, max_workers: int = 8, memory_budget_bytes: int = 1024**3) -> int`.
- Produces: `run_ordered_frame_tasks(items, worker, *, worker_count, on_ordered_result=None) -> None` with at most `worker_count` futures in flight.
- Consumed by: depth, distortion/crop, and VR processors in later tasks.

- [ ] **Step 1: Write the failing policy tests**

```python
import pytest

from src.depth_surge_3d.processing.frames.frame_stage_parallelism import (
    calculate_frame_stage_workers,
)


@pytest.mark.parametrize(
    ("cpu_count", "frame_count", "item_bytes", "expected"),
    [
        (16, 100, 1, 8),       # explicit ceiling
        (6, 100, 1, 4),        # reserve two CPUs
        (16, 3, 1, 3),         # frame-count ceiling
        (1, 100, 1, 1),        # minimum progress
        (None, 100, 1, 1),     # unknown CPU count
        (16, 100, 600_000_000, 1),
        (16, 100, 2_000_000_000, 1),
    ],
)
def test_calculate_frame_stage_workers_obeys_all_caps(
    monkeypatch, cpu_count, frame_count, item_bytes, expected
):
    monkeypatch.setattr("os.cpu_count", lambda: cpu_count)
    assert calculate_frame_stage_workers(frame_count, item_bytes) == expected


@pytest.mark.parametrize(("frame_count", "item_bytes"), [(0, 1), (1, 0), (-1, 1)])
def test_calculate_frame_stage_workers_rejects_invalid_work(frame_count, item_bytes):
    with pytest.raises(ValueError):
        calculate_frame_stage_workers(frame_count, item_bytes)
```

- [ ] **Step 2: Run the focused test and verify the import fails**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_frame_stage_parallelism.py -q`

Expected: collection fails because `frame_stage_parallelism` does not exist.

- [ ] **Step 3: Implement the pure worker calculation**

```python
from __future__ import annotations

import os


MAX_FRAME_STAGE_WORKERS = 8
FRAME_STAGE_MEMORY_BUDGET_BYTES = 1024**3
FRAME_STAGE_CPU_RESERVE = 2


def calculate_frame_stage_workers(
    frame_count: int,
    estimated_bytes_per_item: int,
    *,
    max_workers: int = MAX_FRAME_STAGE_WORKERS,
    memory_budget_bytes: int = FRAME_STAGE_MEMORY_BUDGET_BYTES,
) -> int:
    if frame_count <= 0 or estimated_bytes_per_item <= 0:
        raise ValueError("Frame work and item memory must be positive")
    logical_cpus = os.cpu_count() or 1
    cpu_limit = max(1, logical_cpus - FRAME_STAGE_CPU_RESERVE)
    memory_limit = max(1, memory_budget_bytes // estimated_bytes_per_item)
    return max(1, min(frame_count, max_workers, cpu_limit, memory_limit))
```

- [ ] **Step 4: Run the helper tests**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_frame_stage_parallelism.py -q`

Expected: all tests pass.

- [ ] **Step 5: Checkpoint the focused diff**

Run: `git diff --check -- src/depth_surge_3d/processing/frames/frame_stage_parallelism.py tests/unit/test_frame_stage_parallelism.py`

Clean-worktree commit: `git commit -m "perf: bound frame-stage parallelism"`

### Task 2: Lossless Crop Fast Path And Shared Fisheye Maps

**Files:**
- Modify: `src/depth_surge_3d/utils/imaging/image_processing.py`
- Modify: `src/depth_surge_3d/utils/imaging/__init__.py`
- Modify: `src/depth_surge_3d/utils/__init__.py`
- Modify: `src/depth_surge_3d/processing/frames/distortion_processor.py`
- Modify: `tests/unit/test_distortion_processor.py`
- Test: `tests/unit/test_file_operations.py`

**Interfaces:**
- Consumes: `calculate_frame_stage_workers` from Task 1.
- Produces: `remap_fisheye(image: np.ndarray, x_map: np.ndarray, y_map: np.ndarray) -> np.ndarray`.
- Preserves: `apply_fisheye_distortion(image, fov_degrees, projection_type)` public behavior.
- Produces private crop materialization that attempts `os.link` and falls back to `shutil.copy2`.

- [ ] **Step 1: Add failing fisheye-equivalence and map-reuse tests**

```python
def test_remap_fisheye_matches_existing_helper_exactly():
    image = np.random.default_rng(7).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    x_map, y_map = calculate_fisheye_coordinates(64, 48, 100.0, "equisolid")
    expected = apply_fisheye_distortion(image, 100.0, "equisolid")
    actual = remap_fisheye(image, x_map, y_map)
    np.testing.assert_array_equal(actual, expected)


def test_distortion_builds_one_map_pair_for_the_stage(
    temp_frames, mock_progress_tracker, tmp_path
):
    left_output = tmp_path / "left_distorted"
    right_output = tmp_path / "right_distorted"
    left_output.mkdir()
    right_output.mkdir()
    processor = DistortionProcessor()
    settings = {"fisheye_fov": 100.0, "fisheye_projection": "equisolid"}
    with patch(
        "src.depth_surge_3d.processing.frames.distortion_processor.calculate_fisheye_coordinates",
        wraps=calculate_fisheye_coordinates,
    ) as calculate:
        assert processor.apply_distortion(
            temp_frames["left_files"],
            temp_frames["right_files"],
            {"left_distorted": left_output, "right_distorted": right_output},
            settings,
            mock_progress_tracker,
        ) is True
    assert calculate.call_count == 1
```

Also assert that all projection modes remain pixel-identical, and that mixed,
left/right-mismatched, or unreadable PNG headers fail without `metadata.json`.

- [ ] **Step 2: Add failing no-op crop link/copy tests**

```python
def test_factor_one_crop_uses_hard_links_without_decode(temp_frames, monkeypatch):
    monkeypatch.setattr(cv2, "imread", Mock(side_effect=AssertionError("must not decode")))
    monkeypatch.setattr(cv2, "imwrite", Mock(side_effect=AssertionError("must not encode")))
    assert processor.crop_frames(temp_frames, settings, total_frames=3) is True
    for source, output in zip(source_files, output_files):
        assert source.read_bytes() == output.read_bytes()
        assert os.path.samefile(source, output)


def test_factor_one_crop_falls_back_to_copy2(temp_frames, monkeypatch):
    monkeypatch.setattr(os, "link", Mock(side_effect=OSError("unsupported")))
    assert processor.crop_frames(temp_frames, settings, total_frames=3) is True
    assert source.read_bytes() == output.read_bytes()
    assert not os.path.samefile(source, output)
```

Delete one hard-link path and assert the other path still decodes. Change the
existing crop exception test to `crop_factor=0.8` so it exercises the transformed
path rather than the new no-op path.

- [ ] **Step 3: Run the distortion tests and verify the new assertions fail**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_distortion_processor.py -q`

Expected: failures for missing remap API, repeated map construction, and decode/re-encode on factor 1.0.

- [ ] **Step 4: Separate coordinate construction from remapping**

```python
def remap_fisheye(image, x_map, y_map):
    height, width = image.shape[:2]
    if x_map.shape != (height, width) or y_map.shape != (height, width):
        raise ValueError("Fisheye maps must match image dimensions")
    return cv2.remap(
        image,
        x_map,
        y_map,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def apply_fisheye_distortion(image, fov_degrees, projection_type="stereographic"):
    height, width = image.shape[:2]
    maps = calculate_fisheye_coordinates(width, height, fov_degrees, projection_type)
    return remap_fisheye(image, *maps)
```

Export `remap_fisheye` through both existing imaging utility export modules.

- [ ] **Step 5: Preflight one uniform shape and parallelize distortion pairs**

Read all source IHDR values before work. Require identical width, height,
channels, and bit depth across both eyes and all frames. Construct the maps
once, mark them read-only, and submit one pair per future. Use
`estimated_bytes_per_item = width * height * 48`. Each worker decodes, calls
`remap_fisheye` twice, and writes distinct outputs. Consume futures in source
order and update progress on the caller thread. On failure, cancel pending work
and call `shutdown(wait=True, cancel_futures=True)` before returning. Call
`complete_stage` only after success.

- [ ] **Step 6: Implement factor-1.0 materialization and parallel transformed crop**

```python
@staticmethod
def _materialize_unchanged_frame(source: Path, destination: Path) -> bool:
    try:
        os.link(source, destination)
    except OSError:
        try:
            shutil.copy2(source, destination)
        except OSError:
            return False
    return True
```

Determine the clamped factor once. For non-distorted `1.0`, materialize pairs
sequentially and report ordered progress. Otherwise keep `_crop_single_frame_pair`
pixel logic unchanged. Read every source IHDR and calculate
`max(max_source_eye_pixels, per_eye_width * per_eye_height) * 48`, then submit
pairs to a bounded pool and consume results in order. Apply the same cancel,
wait-for-running-workers, and no-completion-on-failure rule as distortion.

- [ ] **Step 7: Run focused tests and exact-pixel comparisons**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_distortion_processor.py tests/unit/test_file_operations.py -q`

Expected: all tests pass, including existing stage reuse and crop-factor clamping tests.

- [ ] **Step 8: Checkpoint the focused diff**

Run: `git diff --check -- src/depth_surge_3d/utils/imaging src/depth_surge_3d/utils/__init__.py src/depth_surge_3d/processing/frames/distortion_processor.py tests/unit/test_distortion_processor.py`

Clean-worktree commit: `git commit -m "perf: accelerate lossless distortion and crop stages"`

### Task 3: Parallel Canonical Disparity Persistence

**Files:**
- Modify: `src/depth_surge_3d/processing/frames/depth_processor.py`
- Modify: `tests/unit/test_canonical_depth_pipeline.py`

**Interfaces:**
- Consumes: `calculate_frame_stage_workers` from Task 1.
- Preserves: `_write_canonical_stage` return type `list[Path]`, canonical metadata, atomic PNG writes, cache reuse, and preview sampling.

- [ ] **Step 1: Add a failing serial-versus-threaded exactness test**

Run the small file-backed fixture once with worker selection patched to one and
store every decoded `uint16` canonical array. Remove only local canonical PNGs
and `metadata.json`, rerun with four workers while reusing raw depth, and assert:

```python
assert serial_names == parallel_names
for serial, parallel in zip(serial_arrays, parallel_arrays):
    assert serial.dtype == parallel.dtype == np.uint16
    np.testing.assert_array_equal(parallel, serial)
```

- [ ] **Step 2: Add failing ordering and failure tests**

Use per-frame delays to make workers complete in reverse order. Capture callback
frame numbers and `threading.get_ident()` values, then assert source order and
the test caller's thread id. Make the first canonical worker raise while a
second worker blocks on a `threading.Event`; assert the stage call does not
return until the event releases the running worker, and assert
`03_disparity_maps/metadata.json` is absent.

- [ ] **Step 3: Run the canonical tests and verify they fail under serial code**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_canonical_depth_pipeline.py -q`

Expected: the worker-count patch is unused and concurrency/ordering assertions fail.

- [ ] **Step 4: Implement ordered bounded futures in `_write_canonical_stage`**

Take `native_shape` from `raw_store.metadata`, calculate item memory as
`height * width * 32`, and submit one worker per zipped raw/output/scene item.
The worker must execute the existing load, `canonicalize_depth`,
`encode_canonical_png`, and `_atomic_write_png` calls without changing their
arguments. Consume futures in source order and emit the existing sampled
previews from the caller thread. Cancel pending futures on failure. Write JSON
metadata and validate only after all futures succeed. The exception path must
call `shutdown(wait=True, cancel_futures=True)` before re-raising; the success
path must also shut down with `wait=True` before metadata is written.

- [ ] **Step 5: Run canonical and storage regression tests**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_canonical_depth_pipeline.py tests/unit/test_depth_processor.py tests/unit/test_depth_storage.py tests/unit/test_canonical_metadata_contract.py -q`

Expected: all tests pass.

- [ ] **Step 6: Checkpoint the focused diff**

Run: `git diff --check -- src/depth_surge_3d/processing/frames/depth_processor.py tests/unit/test_canonical_depth_pipeline.py`

Clean-worktree commit: `git commit -m "perf: parallelize canonical disparity writes"`

### Task 4: Parallel Pixel-Exact VR Assembly

**Files:**
- Modify: `src/depth_surge_3d/processing/frames/vr_assembler.py`
- Modify: `tests/unit/test_vr_assembler.py`

**Interfaces:**
- Consumes: `calculate_frame_stage_workers` from Task 1.
- Preserves: `assemble_vr_frames` return type `bool`, source selection, layout, naming, stage identity, preview API, and final shape validation.

- [ ] **Step 1: Add failing exactness and same-size bypass tests**

Create deterministic left/right images, build expected side-by-side and
over-under arrays with the current serial operations, run assembly, and compare
decoded outputs exactly. Patch `resize_image` to raise for matching dimensions
and verify assembly still succeeds. Include a mismatching-size case and assert
the current `INTER_CUBIC` result exactly.

- [ ] **Step 2: Add failing ordered-callback and failure-manifest tests**

Delay workers so completion order differs from source order. Assert progress and
preview frame numbers are ordered and all callback thread ids match the caller.
Make one worker fail while another running worker blocks; assert assembly does
not return until the running worker exits and VR metadata remains absent even if
another output was already written. Add a source frame larger than the target,
patch worker calculation, and assert its item-byte argument is based on the
larger source IHDR pixel count.

- [ ] **Step 3: Run VR tests and verify the new tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_vr_assembler.py -q`

Expected: same-size bypass and worker-concurrency assertions fail.

- [ ] **Step 4: Implement the same-size bypass and ordered bounded futures**

In `_assemble_single_vr_frame`, use the decoded array directly only when its
`(width, height)` already equals the target; otherwise call `resize_image`
unchanged. Return the written `Path` rather than sending a preview in the worker.
In `assemble_vr_frames`, preflight every left/right source IHDR, reject
unreadable headers, and find `max_source_eye_pixels`. Calculate workers with
`max(max_source_eye_pixels, per_eye_width * per_eye_height) * 48`, submit
distinct pairs, consume in source order, and issue preview/progress callbacks on
the caller thread. On failure, cancel pending futures and finish
`shutdown(wait=True, cancel_futures=True)` before returning. Preserve the final
`complete_stage` call after successful shutdown.

- [ ] **Step 5: Run VR and orchestration regression tests**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_vr_assembler.py tests/unit/test_pipeline_orchestrator.py tests/unit/test_video_processor_new.py -q`

Expected: all tests pass.

- [ ] **Step 6: Checkpoint the focused diff**

Run: `git diff --check -- src/depth_surge_3d/processing/frames/vr_assembler.py tests/unit/test_vr_assembler.py`

Clean-worktree commit: `git commit -m "perf: parallelize VR frame assembly"`

### Task 5: Full Regression And Performance Verification

**Files:**
- Verify only; no production file should change in this task.

**Interfaces:**
- Consumes all prior tasks.
- Produces measured evidence and a final changed-file report.

- [ ] **Step 1: Run formatting and whitespace validation**

Run: `git diff --check`

Expected: no whitespace errors in changed files.

- [ ] **Step 2: Run the complete unit suite**

Run: `.venv\Scripts\python.exe -m pytest tests/unit -q`

Expected: all tests pass with no new warnings attributable to this change.

- [ ] **Step 3: Run real-frame stage benchmarks**

Use the previously measured 1080p frame sample. Record one-worker and automatic
worker throughput for canonicalization and VR assembly. For factor-1.0 crop,
record elapsed materialization plus `complete_stage` validation. Do not change
PNG compression or input data between runs.

Expected: exact decoded hashes match; no-op crop completes in seconds rather
than minutes; canonical and VR throughput materially exceed their measured
20.8 fps and 2 fps pipeline baselines.

- [ ] **Step 4: Run a short end-to-end resumability smoke test**

Run a short source with intermediates retained, rerun it to exercise stage
reuse, and confirm final frame count, dimensions, FPS, decode, and audio-presence
metadata. Interrupt or inject one frame-stage failure, rerun, and confirm the
incomplete stage clears partial output before completing.

- [ ] **Step 5: Review scope and report**

Run: `git status --short` and `git diff --stat`

Confirm that no setting, encoder, renderer, estimator, stage-manifest schema, or
unrelated user file was changed. In the shared dirty workspace, leave changes
unstaged and report targeted tests, full-suite results, benchmark numbers, and
any residual disk-dependent risk.
