# VDPP Temporal Post-Processor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional, CUDA-generated VDPP stage that stabilizes canonical relative-disparity PNGs by finalized shot, remains bounded in memory, resumes only integrity-checked shot artifacts, and leaves the default-off pipeline byte- and cache-compatible.

**Architecture:** A schema-first settings resolver and an OS-backed job lock feed an artifact-first execution planner. `TemporalDepthStabilizer` owns files, scene ranges, metadata, resume, checkpoint resolution, and progress. `VDPPTemporalPostprocessor` owns the pinned neural model and the exact 32/4 padded-space recurrence. A shared render-disparity validator lets stereo consume either the historical base canonical producer or the new content-addressed stabilized producer without weakening either schema.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, OpenCV, pytest, vendored VDPP v1.0 source (`73cc2b4dc6b3b5cfb2e37f51e452461e03fe26f5`), standard-library OS file locking.

## Global Constraints

- `temporal_postprocessor=off` is the default. It must preserve the existing eight progress steps, generated payloads, raw/base-canonical identities, and cache keys.
- VDPP constants are fixed: batch 1, window 32, overlap 4, stride 28, `downsize=True`, FP32, no tail padding, no overlap interpolation, and one affine scale/shift per later window.
- Windows never cross finalized scene cuts. Every selected source frame is emitted once, in order, and recurrence state resets at every shot.
- The base depth owner is released before VDPP construction. VDPP is released before stereo. Cleanup cannot depend on Python garbage collection.
- Generation is CUDA-only, but a complete validated stabilized artifact is renderable without CUDA, the checkpoint, the vendored import, or a base estimator.
- One fail-fast OS-backed writer lock covers the authoritative audit through the last job mutation. A future settings schema fails before lock-file creation or any mutation.
- Stabilized outputs use shot-atomic commits, per-file SHA-256, immutable shot manifests, and separate semantic/runtime/state/payload/artifact/metadata fingerprints.
- No production behavior is added before a focused test fails for the expected reason.

---

### Task 1: Settings Schema V3 And Presence-Aware Overrides

**Files:**
- Modify: `src/depth_surge_3d/core/constants.py`
- Modify: `src/depth_surge_3d/core/settings.py`
- Modify: `src/depth_surge_3d/io/operations.py`
- Modify: `src/depth_surge_3d/io/resume.py`
- Modify: `depth_surge_3d.py`
- Modify: `app.py`
- Modify: `tests/unit/test_settings.py`
- Modify: `tests/unit/test_resume.py`
- Modify: `tests/unit/test_see_through_entrypoints.py`

**Interfaces:**
- Produces: `PROCESSING_SETTINGS_SCHEMA_VERSION = 3`.
- Produces: `load_processing_settings(raw_settings, saved_version) -> SettingsLoadResult` with schema-first, upward-only migration.
- Produces: `resolve_temporal_postprocessor(*, persisted, override, is_resume) -> Literal["off", "vdpp"]`.
- Preserves: `temporal_postprocessor` is excluded from depth-model and canonical-cache identity key sets.

- [ ] **Step 1: Add failing migration and override matrix tests**

Cover absent version/v1/v2 migration, prerelease v2 values, strict v3 required field and unknown-key rejection, future-version no-mutation failure, new-job omitted -> `off`, resume omitted -> persisted, and explicit resume `off`/`vdpp` overrides.

- [ ] **Step 2: Run focused tests and confirm RED**

```powershell
& 'E:\Code\depth-surge-3d\.venv\Scripts\python.exe' -m pytest tests/unit/test_settings.py tests/unit/test_resume.py -q
```

- [ ] **Step 3: Implement schema-first parsing and one-way migration**

Read `metadata.settings_schema_version` before legacy filtering. Reject versions above 3 without backup writes or cleanup. Migrate only known lower versions, insert `temporal_postprocessor="off"` when absent, validate a present prerelease value, and parse v3 strictly.

- [ ] **Step 4: Implement presence-aware CLI/Web resolution**

Add the CLI choice with `default=None`. Preserve property presence in Web JSON rather than replacing absence with `off`. Keep persisted settings on omitted resume and treat explicit `off` as an invalidating user change.

- [ ] **Step 5: Verify default-off identities and commit**

Assert toggling the setting does not alter raw/canonical identities. Commit: `feat: add schema v3 temporal postprocessor setting`.

---

### Task 2: OS-Backed Writer Lock And Artifact-First Plan Types

**Files:**
- Create: `src/depth_surge_3d/io/job_lock.py`
- Create: `src/depth_surge_3d/processing/orchestration/execution_plan.py`
- Modify: `src/depth_surge_3d/io/__init__.py`
- Create: `tests/unit/test_job_lock.py`
- Create: `tests/unit/test_execution_plan.py`

**Interfaces:**
- Produces: `JobWriterLock(output_dir).acquire(blocking=False)` context manager using `msvcrt.locking` on Windows and `fcntl.flock` on POSIX.
- Produces: `PipelineExecutionPlan(can_run_cache_only, needs_base_depth_model, needs_vdpp_model, selected_render_source)`.
- Produces: pure, read-only `probe_saved_settings_schema(output_dir)` that creates no path.

- [ ] **Step 1: Write failing lock contention, crash-stale-record, and diagnostics tests**

Use a child process to prove a second writer fails immediately while the first holds the OS lock. Assert diagnostic JSON includes PID, hostname, process identity, and acquisition time; stale text alone does not block reacquisition.

- [ ] **Step 2: Write failing future-schema and cache-only plan tests**

Assert future schemas fail before `.depth-surge.lock` exists. Assert a fully validated stabilized artifact selects cache-only without invoking supplied CUDA/model/checkpoint probes.

- [ ] **Step 3: Implement the lock and immutable plan data types**

The lock owns an open handle, locks one initialized byte, writes diagnostics only after acquisition, and unlocks/closes in `finally`. Planning callbacks remain lazy so a cache-only decision cannot accidentally import or construct a model.

- [ ] **Step 4: Run process-level tests and commit**

Commit: `feat: add authoritative job writer lock`.

---

### Task 3: Render-Disparity Contract And Intermediate Registry

**Files:**
- Modify: `src/depth_surge_3d/core/constants.py`
- Create: `src/depth_surge_3d/core/render_disparity.py`
- Modify: `src/depth_surge_3d/processing/frames/stereo_generator.py`
- Modify: `tests/unit/test_constants.py`
- Create: `tests/unit/test_render_disparity.py`
- Modify: `tests/unit/test_stereo_generator.py`

**Interfaces:**
- Produces: `INTERMEDIATE_DIRS["disparity_stabilized"] = "03_disparity_stabilized"`.
- Produces: `validate_render_disparity_input(files, directory) -> RenderDisparityArtifact`.
- Accepts: exact base `scene-percentile-v1` or exact stabilized `vdpp-canonical-shot-v1`; no generic producer fallback.

- [ ] **Step 1: Add failing producer matrix tests**

Lock common representation/polarity/range/shape/name/uint16-header checks and producer-specific self-hash rules. Reject unknown producer names, a base metadata file relabeled as stabilized, and stabilized metadata missing its complete artifact fingerprint.

- [ ] **Step 2: Refactor stereo to consume the validated artifact**

Retain the historical `source_canonical_fingerprint` metadata key. Fill it with the selected artifact fingerprint, which is the base canonical fingerprint for historical input and stabilized `artifact_fingerprint` for VDPP input.

- [ ] **Step 3: Verify historical base fixtures unchanged and commit**

Commit: `feat: validate derived render disparity sources`.

---

### Task 4: Stabilized Metadata And Shot-Atomic Store

**Files:**
- Create: `src/depth_surge_3d/processing/frames/temporal_storage.py`
- Create: `tests/unit/test_temporal_storage.py`
- Modify: `tests/unit/test_resume.py`

**Interfaces:**
- Produces: `StabilizedDepthStore` with atomic PNG writes, immutable shot manifests, audit/repair, and final completion barrier.
- Produces: separate `semantic_fingerprint`, `partial_resume_runtime_fingerprint`, `state_fingerprint`, `payload_fingerprint`, `artifact_fingerprint`, and `metadata_fingerprint`.
- Produces: strict final-scene `shot_plan` from half-open `final_cuts`.

- [ ] **Step 1: Write failing fingerprint-boundary and metadata self-hash tests**

Assert semantic identity excludes mutable state/runtime, partial runtime excludes preflight peak counters, state changes after a committed shot, and artifact identity is exactly the canonical hash of semantic plus payload fingerprints.

- [ ] **Step 2: Write failing shot transaction and corruption tests**

Cover interruption before manifest, interruption before top-level completion append, one modified PNG byte, malformed/missing manifest, orphan payload cleanup, selective repair of one completed shot, and preservation of later independent shots.

- [ ] **Step 3: Implement preallocated input loading and deterministic uint16 encoding**

Decode directly into a caller-supplied `[S,H,W]` float32 buffer one file at a time. Validate path, order, dtype, rank, shape, and `[0,1]`. Encode with the existing canonical rounding policy to a same-directory temporary, hash final bytes, then atomically replace.

- [ ] **Step 4: Implement resume decisions and complete-stage validation**

Semantic mismatch resets the stage. Partial runtime mismatch resets an unfinished stage. A complete content-addressed stage ignores runtime drift but still validates every manifest and byte digest.

- [ ] **Step 5: Run storage/resume tests and commit**

Commit: `feat: add shot-atomic stabilized depth store`.

---

### Task 5: Pin And Vendor VDPP Source Plus Secure Checkpoint Resolution

**Files:**
- Create: `src/depth_surge_3d/_vendor/vdpp/**`
- Create: `src/depth_surge_3d/_vendor/vdpp_utils/normal_utils.py`
- Create: `src/depth_surge_3d/_vendor/vdpp/UPSTREAM.json`
- Create: `src/depth_surge_3d/_vendor/vdpp/LICENSE`
- Create: `src/depth_surge_3d/_vendor/vdpp/NOTICE.md`
- Create: `src/depth_surge_3d/inference/depth/vdpp_artifact.py`
- Modify: `pyproject.toml`
- Create: `tests/unit/test_vdpp_vendor.py`
- Create: `tests/unit/test_vdpp_artifact.py`

**Interfaces:**
- Pins: release `v1.0`, revision `73cc2b4dc6b3b5cfb2e37f51e452461e03fe26f5`.
- Pins: checkpoint size `116485370`, SHA-256 `7368315b126093f0335147f42a1920f255d529613bfffc5c6cf4ef832deb73a7`.
- Produces: `ensure_vdpp_checkpoint(models_dir, progress_callback) -> Path` with streaming `.part`, bounded timeout, artifact lock, exact verification, and atomic replace.

- [ ] **Step 1: Add failing vendor-manifest and package-data tests**

Assert every vendored inference file matches its declared original-path SHA-256, imports stay package-relative, demo/assets/DAv2 code are absent, Apache-2.0 ships, and built wheel contents include the subset.

- [ ] **Step 2: Port the minimal pinned source mechanically**

Copy only `vdpp/**` and `utils/normal_utils.py`; add package markers and change imports only as required for package-relative resolution. Record original and vendored hashes plus the mechanical changes in `UPSTREAM.json`.

- [ ] **Step 3: Add failing checkpoint success/corruption/concurrency tests**

Mock streaming HTTP. Cover correct reuse, wrong existing digest, wrong content length, timeout, interrupted `.part`, concurrent resolution, byte-progress reporting, and no network call before an explicit generation request.

- [ ] **Step 4: Implement secure load boundary and commit**

Require `torch.load(..., map_location="cpu", weights_only=True)` and strict state-dict loading; never retry unsafe pickle. Commit: `feat: vendor pinned VDPP inference source`.

---

### Task 6: Bounded 32/4 VDPP Adapter

**Files:**
- Create: `src/depth_surge_3d/inference/depth/vdpp_temporal_postprocessor.py`
- Create: `tests/unit/test_vdpp_temporal_postprocessor.py`
- Create: `tests/integration/test_vdpp_cuda.py`

**Interfaces:**
- Produces: `VDPPTemporalPostprocessor.process_shot(frame_count, load_window) -> Iterator[tuple[int, np.ndarray]]`.
- Produces: `model_identity()`, `execution_plan(native_shape)`, `preflight(pending_length, native_shape)`, and idempotent `release()`.
- Preserves: exact upstream padded-space forward/alignment/resize semantics, including zero determinant -> scale 0, shift 0.

- [ ] **Step 1: Write failing window/tail/order/state-release tests**

Parameterize lengths `1,25,31,32,33,60,61,100`. Assert calls match the spec table, output indexes are contiguous, later windows discard only their first four observations, no tail padding occurs, and a loader/model/cancellation error clears compact retained state.

- [ ] **Step 2: Write failing deterministic upstream-equivalence tests**

Use a fake forward and compare bounded output numerically with the pinned upstream `infer_video_depth` continuation calculation for non-multiple-of-14 inputs, normal and degenerate affine systems, clipping branch separation, and final native resize.

- [ ] **Step 3: Implement exact FP32 bounded recurrence**

Preallocate host input, retain only `current_padded[-4:].detach().clone()`, validate all scalar/tensor finiteness, apply the upstream out-of-place affine expression, and yield one native float32 frame at a time before storage clipping.

- [ ] **Step 4: Implement continuation-path CUDA preflight**

For pending lengths above 32, exercise retained overlap plus the real next-window size, affine multiply/add, and one native transfer. Record allocated and reserved peaks separately and synchronously release all trial tensors.

- [ ] **Step 5: Add real-checkpoint CUDA smoke test gate and commit**

Skip when CUDA/checkpoint are absent, but fail release verification when the platform claims VDPP support and the gated smoke test is requested. Commit: `feat: add bounded VDPP temporal adapter`.

---

### Task 7: File-Backed Temporal Coordinator And Disk Preflight

**Files:**
- Create: `src/depth_surge_3d/processing/frames/temporal_stabilizer.py`
- Modify: `src/depth_surge_3d/processing/frames/__init__.py`
- Create: `tests/unit/test_temporal_stabilizer.py`

**Interfaces:**
- Produces: `TemporalDepthStabilizer.generate_files(base_files, settings, directories, progress_tracker) -> list[str]`.
- Consumes: validated base canonical metadata and final scene ranges only.
- Owns: disk preflight, CUDA/checkpoint/model laziness, shot resume, deterministic encoding, progress, and final global validation.

- [ ] **Step 1: Write failing off/cache-only/device/disk tests**

Assert off never constructs the stage; a complete stabilized hit invokes neither CUDA nor checkpoint/model factories; generation rejects CPU/MPS before download/mutation; insufficient disk fails before checkpoint resolution; and no requested VDPP run falls back to base files.

- [ ] **Step 2: Write failing shot-range/resume/progress tests**

Assert no load crosses a final cut, all frames are produced once, a partial shot is fully recomputed, completed shots resume, runtime drift resets only an unfinished stage as specified, and progress reports finalized frames plus checkpoint bytes in `Temporal Depth Stabilization`.

- [ ] **Step 3: Implement authoritative audit and generation loop**

Build semantic identity and execution plan before construction. Audit existing metadata, resolve pending shots, perform disk/device checks, lazily resolve the checkpoint and adapter, run preflight before repair/new commits, then process one shot at a time. Always close the shot iterator and release VDPP in nested `finally` blocks.

- [ ] **Step 4: Verify failure preservation and commit**

Assert OOM/model/download/cancellation preserve base canonical and earlier valid shots, mark the current shot incomplete, and never enter stereo. Commit: `feat: add file-backed VDPP stabilization stage`.

---

### Task 8: Orchestration, Lazy Base Ownership, And Cache-Only Execution

**Files:**
- Modify: `src/depth_surge_3d/processing/orchestration/pipeline_orchestrator.py`
- Modify: `src/depth_surge_3d/processing/orchestration/video_processor.py`
- Modify: `src/depth_surge_3d/rendering/stereo_projector.py`
- Modify: `src/depth_surge_3d/io/resume.py`
- Modify: `depth_surge_3d.py`
- Modify: `app.py`
- Modify: `tests/unit/test_pipeline_orchestrator.py`
- Modify: `tests/unit/test_video_processor_new.py`
- Modify: `tests/unit/test_stereo_projector.py`
- Modify: `tests/unit/test_resume.py`

**Interfaces:**
- Inserts: base canonical -> owner release -> optional VDPP -> stereo.
- Strengthens: `StereoProjector.unload_model()` is idempotent, synchronizes the effective CUDA device, runs estimator/allocator cleanup despite failures, and clears `_model_loaded` in `finally`.
- Produces: locked, authoritative artifact-first plan before settings migration, cleanup, or any model load.

- [ ] **Step 1: Add failing model-order and all-path release tests**

Cover normal generation, base cache hit, base failure, cancellation, VDPP construction/checkpoint/preflight/generation failures, VDPP cache hit, and top-level cleanup. Assert the owner flag is false before VDPP construction and both models are never resident together.

- [ ] **Step 2: Add failing cache-only entry-point tests**

Patch CUDA checks, estimator imports/construction, checkpoint resolution, and VDPP import to raise. A complete valid stabilized artifact must still reach stereo. A missing base stage must lazily load only the selected estimator.

- [ ] **Step 3: Refactor projector and entry points to lazy loading**

Remove eager video-path `load_model()` calls before resume planning. Keep image behavior unchanged. Enrich the plan with live base identity only when base generation is required.

- [ ] **Step 4: Hold the writer lock across all authoritative mutations**

Probe schema first, acquire lock, repeat the full audit, then allow migrations, resets, writes, encoding, cleanup, and final status. Release only in the outermost `finally`.

- [ ] **Step 5: Run orchestration/resume regressions and commit**

Commit: `feat: orchestrate artifact-first VDPP processing`.

---

### Task 9: Web, CLI, Dynamic Progress, And User-Facing Errors

**Files:**
- Modify: `templates/index.html`
- Modify: `app.py`
- Modify: `depth_surge_3d.py`
- Modify: `src/depth_surge_3d/core/constants.py`
- Modify: `tests/unit/test_direct_vr_progress.py`
- Modify: `tests/unit/test_direct_vr_encode_template.py`
- Modify: `tests/unit/test_resume_template.py`
- Modify: `tests/unit/test_see_through_entrypoints.py`

**Interfaces:**
- Adds: segmented `Off | VDPP (Experimental)` control and explanatory tooltip.
- Adds: CLI `--temporal-postprocessor {off,vdpp}` with omission preserved.
- Adds: dynamic nine-step weights when enabled: depth `0.28`, temporal `0.07`, all other weights unchanged.

- [ ] **Step 1: Write failing template/payload/hydration tests**

Assert no tuning controls exist, a new request sends the selected explicit value, a resume form hydrates the persisted value, omission is distinguishable from explicit off, and cached VDPP may resume without CUDA while generation may not.

- [ ] **Step 2: Write failing progress and durable-error tests**

Lock the eight-step off plan exactly. Assert the VDPP plan sums to 1.0 and never moves backward. Important warnings/OOM must always hit the process log as well as the throttled tracker.

- [ ] **Step 3: Implement UI, request parsing, CLI, and dynamic progress**

Use a Bootstrap segmented radio group, not free-form inputs. Do not expose window, overlap, strength, precision, or resolution. Hydrate resume state from validated settings before submitting overrides.

- [ ] **Step 4: Run template/entry-point tests and commit**

Commit: `feat: expose optional VDPP stabilization`.

---

### Task 10: Documentation, Quality Harness, Packaging, And Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/PARAMETERS.md`
- Modify: `docs/USAGE.md`
- Modify: `docs/INSTALLATION.md`
- Modify: `docs/PERFORMANCE.md`
- Modify: `docs/DEVELOPMENT.md`
- Modify: `docs/WEB_GUI.md`
- Create: `scripts/evaluate_vdpp_quality.py`
- Create: `tests/unit/test_vdpp_quality_harness.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add a deterministic, versioned quality-evaluation harness**

The harness records source/checkpoint/runtime identities and computes the pinned temporal warping-error and spatial-degradation metrics for the agreed clip manifest. Keep thresholds outside production behavior until the quality gate is run on representative DA3, See-Through, and V2 clips.

- [ ] **Step 2: Document the experimental boundary and operational contract**

Explain CUDA generation versus CPU cache reuse, added disk/time, fixed 32/4 semantics, shot resets, checkpoint location/license/hash, schema-v3 backward/forward boundary, interruption cost, error recovery, and why V2 usually needs VDPP less than framewise models.

- [ ] **Step 3: Build wheel and inspect vendored/package artifacts**

```powershell
& 'E:\Code\depth-surge-3d\.venv\Scripts\python.exe' -m build
```

Confirm source, license, notice, and `UPSTREAM.json` are installed while demo/assets are absent.

- [ ] **Step 4: Run formatting, static checks, focused integration tests, and complete suite**

```powershell
& 'E:\Code\depth-surge-3d\.venv\Scripts\python.exe' -m black --check src tests app.py depth_surge_3d.py scripts
& 'E:\Code\depth-surge-3d\.venv\Scripts\python.exe' -m flake8 src tests app.py depth_surge_3d.py scripts
& 'E:\Code\depth-surge-3d\.venv\Scripts\python.exe' -m mypy src
& 'E:\Code\depth-surge-3d\.venv\Scripts\python.exe' -m pytest -q
git diff --check
```

- [ ] **Step 5: Inspect representative metadata and finish the branch**

Run a fake-adapter end-to-end job, inspect stabilized top-level and shot manifests, prove default-off historical output remains unchanged, and verify a complete artifact renders with model/CUDA probes disabled. Commit: `feat: complete VDPP temporal post-processing`.
