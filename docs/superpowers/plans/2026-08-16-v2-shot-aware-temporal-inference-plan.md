# V2 Shot-Aware Temporal Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the file-backed V2 pipeline reproduce Video-Depth-Anything's fixed 32-frame offline temporal algorithm within each candidate RGB shot, with bounded memory, cache-correct resume, and uniform resolution fallback.

**Architecture:** `VideoDepthEstimator.iter_sequence_depth` returns a stateful bounded iterator whose failed window can be retried without advancing temporal state. `DepthMapProcessor` detects that method on the estimator type, partitions work by candidate cuts, owns disk loading/writes and shot-atomic resume, and wraps V2 execution in a requested/effective resolution-plan loop. Backend-specific fingerprint selection keeps DA3 and See-Through raw identities unchanged.

**Tech Stack:** Python 3.9+, NumPy, PyTorch, OpenCV, vendored Video-Depth-Anything transforms/utilities, pytest.

## Global Constraints

- V2 constants are fixed at window 32, overlap 10, keyframes `[0, 12, 24, 25, 26, 27, 28, 29, 30, 31]`, interpolation length 8, and step 22.
- Every source frame is emitted exactly once in source order; temporal state resets at every candidate cut.
- `DepthBatch` and raw `.npz` payload formats do not change.
- A V2 raw stage uses one effective input size and precision; a plan change invalidates all V2 raw payloads before new writes.
- DA3 and See-Through estimator calls and raw fingerprints remain unchanged.
- Legacy `temporal_window_size` and `temporal_window_overlap` settings remain validated and fingerprinted for at least this release, but fixed V2 inference ignores them.
- No production behavior is added without first observing its focused test fail for the expected reason.

---

### Task 1: Bounded V2 Offline Sequence Iterator

**Files:**
- Modify: `src/depth_surge_3d/inference/depth/video_depth_estimator.py`
- Modify: `tests/unit/test_video_depth_estimator.py`

**Interfaces:**
- Produces: `VideoDepthEstimator.iter_sequence_depth(frame_count, load_frames, *, target_fps, input_size, fp32) -> Iterator[tuple[int, DepthBatch]]`
- Produces: a private retryable `_VDASequenceDepthIterator`; an exception from one fixed-window forward leaves its window position and retained state unchanged.
- Preserves: `estimate_depth_batch` calls upstream `infer_video_depth` once for the complete in-memory array supplied by an external caller.

- [ ] **Step 1: Replace obsolete chunk-routing tests with failing sequence-contract tests**

Add parameterized tests that drive a fake fixed-window forward at lengths `1, 22, 23, 24, 25, 31, 32, 33, 53, 54, 55, 100` and assert:

```python
yielded = list(estimator.iter_sequence_depth(
    frame_count,
    load_frames,
    target_fps=30,
    input_size=518,
    fp32=False,
))
starts = [start for start, _batch in yielded]
values = np.concatenate([batch.values for _start, batch in yielded], axis=0)
assert starts[0] == 0
assert sum(len(batch.values) for _start, batch in yielded) == frame_count
assert values[:, 0, 0].tolist() == list(range(frame_count))
assert max(map(len, loader_requests)) <= 32
assert all(request == sorted(set(request)) for request in loader_requests)
```

Also assert the later requests for a 55-frame shot are `[32, ..., 53]` and `[54]`, not global or carried indexes.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```powershell
& 'E:\Code\depth-surge-3d\.venv\Scripts\python.exe' -m pytest tests/unit/test_video_depth_estimator.py -q
```

Expected: failures because `iter_sequence_depth` and the fixed-window state machine do not exist and obsolete chunk helpers are still selected.

- [ ] **Step 3: Implement the retryable fixed-window iterator and exact upstream finalization**

Add fixed constants and implement these responsibilities:

```python
VDA_INFER_LEN = 32
VDA_OVERLAP = 10
VDA_KEYFRAMES = (0, 12, 24, 25, 26, 27, 28, 29, 30, 31)
VDA_INTERP_LEN = 8
VDA_FRAME_STEP = 22
VDA_INFERENCE_ALGORITHM = "vda-offline-shot-v1"
```

The iterator must load `0:min(32, L)` first, then only `s+10:min(s+32, L)` for later `s = 22, 44, ...`; pad with the retained final transformed input; retain ten selected input tensors, two alignment maps, and eight pending output maps; and advance state only after a successful `_infer_fixed_window` call.

For relative output, `_finalize_window` computes scale/shift from current positions 0/1 against retained references, applies and clamps positions 2-31, interpolates prior pending maps against positions 2-9, and carries positions 24-31. Metric output uses scale 1 and shift 0 but follows the same interpolation path.

- [ ] **Step 4: Add callback validation, retry-state, and in-memory compatibility tests**

Cover short count, non-`uint8`, wrong rank/channels/geometry, callback exception propagation, model-not-loaded, a fixed-window OOM followed by a successful second `next(iterator)`, and `estimate_depth_batch` calling `infer_video_depth` exactly once for 70 frames.

- [ ] **Step 5: Run focused tests, format, and commit**

Run:

```powershell
& 'E:\Code\depth-surge-3d\.venv\Scripts\python.exe' -m pytest tests/unit/test_video_depth_estimator.py tests/unit/test_depth_contract.py -q
& 'E:\Code\depth-surge-3d\.venv\Scripts\python.exe' -m black src/depth_surge_3d/inference/depth/video_depth_estimator.py tests/unit/test_video_depth_estimator.py
```

Commit: `feat: add bounded V2 sequence inference`

---

### Task 2: Backend-Specific Identity And Shot Payload Transactions

**Files:**
- Modify: `src/depth_surge_3d/processing/frames/depth_storage.py`
- Modify: `tests/unit/test_depth_storage.py`
- Modify: `tests/unit/test_canonical_depth_pipeline.py`

**Interfaces:**
- Produces: V2-only raw settings selection for scene parameters.
- Produces: classified `model_info.inference_algorithm` identity.
- Produces: `RawDepthStore.discard_frames(frame_names: list[str]) -> None` for shot-atomic restart.

- [ ] **Step 1: Write failing fingerprint-matrix and discard tests**

Use fake estimators with and without an `iter_sequence_depth` method. Assert changing `scene_cut_threshold` changes only the V2 raw model fingerprint, while changing compatibility temporal settings still changes all three backend fingerprints. Assert an unclassified algorithm field is rejected before adding it to `MODEL_IDENTITY_INFO_KEYS`.

For storage, create three payloads, discard two named frames, and assert their files are removed, the unrelated file remains, `completed_count == 1`, and metadata is flushed.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```powershell
& 'E:\Code\depth-surge-3d\.venv\Scripts\python.exe' -m pytest tests/unit/test_depth_storage.py tests/unit/test_canonical_depth_pipeline.py -q
```

- [ ] **Step 3: Implement backend-dependent identity without widening common key tuples**

Add `inference_algorithm` to classified model identity. Keep `DEPTH_MODEL_SETTING_KEYS` unchanged, then have `build_current_model_fingerprint` add `scene_detection`, `scene_cut_threshold`, and `min_scene_frames` only when `iter_sequence_depth` is defined on the estimator type. Add `scene_algorithm_version` and the runtime execution plan later at the raw semantic-fingerprint layer, not to DA3/See-Through model settings.

- [ ] **Step 4: Implement shot payload discard and verify GREEN**

`discard_frames` validates names against the manifest, deletes only their `.npz` files, recomputes `completed_count` through payload validation, and atomically flushes metadata.

- [ ] **Step 5: Run focused tests and commit**

Commit: `feat: make V2 raw identity shot aware`

---

### Task 3: Route File-Backed V2 Inference By Candidate Shot

**Files:**
- Modify: `src/depth_surge_3d/processing/frames/depth_processor.py`
- Create: `tests/unit/test_v2_temporal_pipeline.py`

**Interfaces:**
- Consumes: `iter_sequence_depth` from Task 1 and `discard_frames` from Task 2.
- Produces: `_candidate_shot_ranges(manifest, frame_count) -> list[tuple[int, int]]`.
- Produces: V2 sequence writes using shot-local loader indexes; framewise estimators keep `_infer_raw_chunk` unchanged.

- [ ] **Step 1: Write failing shot partition and callback mapping tests**

Build a fake sequence estimator that records callback indexes and yields deterministic `DepthBatch` values. For candidate cuts `[3, 7]`, assert calls are made for shot lengths 3, 4, and the tail; no request crosses a cut; and a second shot beginning at global frame 3 receives local indexes starting at zero while the loaded pixels come from global frame 3.

- [ ] **Step 2: Write failing shot-atomic resume tests**

Create raw payload state where shot 1 is complete and shot 2 is partial. Assert shot 1 is not inferred, every shot-2 payload is discarded before recomputation, and an exception during shot 2 prevents canonical generation while leaving a detectably partial shot for the next resume.

- [ ] **Step 3: Run the new tests and confirm RED**

Run:

```powershell
& 'E:\Code\depth-surge-3d\.venv\Scripts\python.exe' -m pytest tests/unit/test_v2_temporal_pipeline.py -q
```

- [ ] **Step 4: Implement the optional sequence path with strict loader ownership**

Detect support using `getattr(type(estimator), "iter_sequence_depth", None)`, never a `MagicMock` attribute or backend name. Derive ranges from persisted `candidate_cuts`. The loader maps local indexes to `frame_files[shot_start + index]`, raises `OSError` naming an unreadable path, and returns a stacked BGR `uint8` array in requested order.

Validate yielded starts as ordered, contiguous, non-overlapping, and exactly the shot length before treating the shot as complete. Create the raw store from the first yield and write each finalized batch immediately.

- [ ] **Step 5: Run sequence and unchanged-framewise regression tests and commit**

Run:

```powershell
& 'E:\Code\depth-surge-3d\.venv\Scripts\python.exe' -m pytest tests/unit/test_v2_temporal_pipeline.py tests/unit/test_depth_processor.py tests/unit/test_video_depth_estimator_da3.py tests/unit/test_video_depth_estimator_see_through.py -q
```

Commit: `feat: process V2 depth by candidate shot`

---

### Task 4: Uniform V2 Resolution Plan And OOM Restart

**Files:**
- Modify: `src/depth_surge_3d/processing/frames/depth_processor.py`
- Modify: `tests/unit/test_v2_temporal_pipeline.py`
- Modify: `tests/unit/test_depth_cache.py`
- Modify: `tests/unit/test_resume.py`

**Interfaces:**
- Produces: semantic fingerprint field `execution_plan` with requested/effective size, precision, and `v2-uniform-halving-v1` policy.
- Produces: initial persisted-plan adoption and candidate sequence `requested -> max(384, current // 2) -> ... -> 384`.
- Produces: one same-size retry for a failed late window, followed by whole-stage restart at the next candidate.

- [ ] **Step 1: Write failing plan identity and persisted-plan tests**

Assert V2 metadata stores:

```python
{
    "requested_input_size": 518,
    "effective_input_size": 384,
    "precision": "fp16",
    "fallback_policy": "v2-uniform-halving-v1",
}
```

Assert changing effective size changes the raw semantic hash and global cache key. Assert DA3 metadata has no execution plan. Assert malformed, source-incompatible, or policy-incompatible persisted plans are rejected, while a compatible partial 384 plan resumes at 384 without probing 518.

- [ ] **Step 2: Write failing first-window and late-window OOM tests**

Test requested-size success, first-window OOM selecting 384 before any payload write, late-window OOM retrying the same iterator position once, repeated late OOM deleting all current-plan raw payloads and restarting from frame zero at 384, and final-candidate OOM raising an error containing both requested and effective sizes.

- [ ] **Step 3: Run focused tests and confirm RED**

Run:

```powershell
& 'E:\Code\depth-surge-3d\.venv\Scripts\python.exe' -m pytest tests/unit/test_v2_temporal_pipeline.py tests/unit/test_depth_cache.py tests/unit/test_resume.py -q
```

- [ ] **Step 4: Implement the outer execution-plan loop**

Resolve requested input size before fingerprint/cache lookup. Adopt only a structurally valid local plan whose semantic fingerprint matches after removing `execution_plan`. On fallback, clear CUDA state, reset raw and canonical directories, rebuild the semantic hash/cache settings, perform one cache lookup for the new plan, and restart inference. Never change size inside an active raw directory.

- [ ] **Step 5: Verify plan behavior and commit**

Commit: `feat: add uniform V2 resolution fallback`

---

### Task 5: Compatibility Warning, Web Cleanup, Documentation, And Final Verification

**Files:**
- Modify: `src/depth_surge_3d/processing/frames/depth_processor.py`
- Modify: `templates/index.html`
- Modify: `README.md`
- Modify: `docs/PERFORMANCE.md`
- Modify: `tests/unit/test_v2_temporal_pipeline.py`
- Modify: `tests/unit/test_settings.py`
- Modify: `tests/unit/test_direct_vr_encode_template.py`

**Interfaces:**
- Preserves: validation, saved settings, constructor parameters, and fingerprint participation for `temporal_window_size`/`temporal_window_overlap`.
- Removes: Web controls, request fields, value listeners, and claims that arbitrary V2 window sizes tune quality or VRAM.

- [ ] **Step 1: Write failing compatibility and template tests**

Assert default 32/10 emits no warning. Assert 64/10 emits one non-blocking warning at V2 raw-stage entry, before cache lookup or estimator work, through the progress channel when present and stderr otherwise. Assert DA3 emits no compatibility warning. Assert the rendered template contains no temporal control IDs or submitted temporal fields while settings validation still accepts and preserves both keys.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```powershell
& 'E:\Code\depth-surge-3d\.venv\Scripts\python.exe' -m pytest tests/unit/test_v2_temporal_pipeline.py tests/unit/test_settings.py tests/unit/test_direct_vr_encode_template.py -q
```

- [ ] **Step 3: Implement warning and remove misleading Web controls**

Emit exactly one message containing the supplied values and `VDA uses fixed window 32 and overlap 10`. Remove the hidden HTML block, payload properties, model-toggle reference, and slider listeners without changing unrelated model controls.

- [ ] **Step 4: Update active documentation**

Describe V2 as fixed 32/10 offline inference that carries temporal state across the whole detected shot and resets at candidate cuts. Replace variable V2 chunk-size guidance with uniform effective-resolution fallback guidance. Do not rewrite archived documents.

- [ ] **Step 5: Run formatting, static checks, focused tests, and the complete suite**

Run:

```powershell
& 'E:\Code\depth-surge-3d\.venv\Scripts\python.exe' -m black --check src tests
& 'E:\Code\depth-surge-3d\.venv\Scripts\python.exe' -m flake8 src tests
& 'E:\Code\depth-surge-3d\.venv\Scripts\python.exe' -m mypy src
& 'E:\Code\depth-surge-3d\.venv\Scripts\python.exe' -m pytest tests/unit -q
```

- [ ] **Step 6: Inspect persisted metadata and commit**

Run a CPU fake-estimator pipeline fixture and inspect `02_depth_raw/metadata.json` to prove `model_info.inference_algorithm`, `execution_plan`, and V2 scene identity are inside `semantic_fingerprint`. Confirm `git diff --check` and commit: `feat: complete V2 shot temporal pipeline`.
