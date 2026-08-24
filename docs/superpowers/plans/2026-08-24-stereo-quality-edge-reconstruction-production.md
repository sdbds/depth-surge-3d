# Stereo Quality Edge Reconstruction Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an offline highest-quality stereo renderer that removes fine-edge DIBR bleeding while retaining the current Fast renderer byte-for-byte, persisting deterministic diagnostics, and recovering every settings/final-publication crash boundary.

**Architecture:** Promote the gate-proven Task 0 algorithms into a focused `depth_surge_3d.rendering.quality` package and keep `StereoRenderer` as the typed Fast/Quality dispatch boundary. The file pipeline owns native primitive inputs, one-frame Quality capacity, compact coverage/repair planes, and staged diagnostics; final encoding consumes a replayable sequence through the authenticated reservation protocol. Existing Fast functions remain in place and are called by the same code path as before.

**Tech Stack:** Python 3.10+, NumPy 1.26, OpenCV, PyTorch, pybind11/C++17 build-time native extension, pytest, Hatchling, FFmpeg/FFprobe.

**Spec:** `docs/superpowers/specs/2026-08-19-stereo-quality-edge-reconstruction-canonical.md`

## Global Constraints

- Fast RGB and all four legacy masks must remain byte-identical for the frozen regression corpus and every existing Fast test.
- `StereoRenderSettings` retains zero through three positional arguments; both new fields are keyword-only and default to Fast/8.
- Quality never silently falls back to Fast, never changes output under memory pressure, and fails the complete frame on a semantic/resource-cap violation.
- Quality uses the build-time precompiled `_quality_kd_native` extension; runtime JIT and a new runtime dependency are forbidden.
- The host live-set budget is 512 MiB, the device temporary budget is 256 MiB, Quality processes left before right, and the file pipeline has lifecycle capacity one.
- Canonical JSON, checked counters, exact tie orders, persisted paths, crash recovery, and index-last retirement are semantic contracts.
- Every production behavior begins with a focused failing test and every task ends with a reviewable commit.

---

### Task 1: Freeze Public Settings, Resolution, and Typed Dispatch

**Files:**
- Modify: `src/depth_surge_3d/core/constants.py`
- Modify: `src/depth_surge_3d/core/settings.py`
- Modify: `src/depth_surge_3d/rendering/stereo_renderer.py`
- Modify: `src/depth_surge_3d/cli.py`
- Test: `tests/unit/test_settings.py`
- Test: `tests/unit/test_stereo_renderer.py`
- Test: `tests/unit/test_cli.py`

**Interfaces:**
- Produces: `StereoRenderMode = Literal["fast", "quality"]`.
- Produces: keyword-only `StereoRenderSettings.stereo_render_mode` and `StereoRenderSettings.occlusion_fill_max_px`.
- Produces: `resolve_stereo_render_mode(persisted, override, is_resume, renderer_device_type) -> StereoRenderMode`.
- Preserves: `StereoSplatSettings(max_eye_shift_fraction, occlusion_fill)` with no new field.

- [x] **Step 1: Add RED settings and constructor tests**

```python
def test_quality_fields_are_keyword_only_and_numpy_integer_normalizes():
    with pytest.raises(TypeError):
        StereoRenderSettings(2.0, 0.5, "background", "quality", 8)
    settings = StereoRenderSettings(
        stereo_render_mode="quality", occlusion_fill_max_px=np.int64(9)
    )
    assert settings.stereo_render_mode == "quality"
    assert settings.occlusion_fill_max_px == 9

def test_schema_four_migrates_to_fast_and_limit_eight():
    parsed = parse_saved_processing_settings(
        schema_four_values, saved_version=4
    )
    assert parsed.settings["stereo_render_mode"] == "fast"
    assert parsed.settings["occlusion_fill_max_px"] == 8
```

- [x] **Step 2: Run the focused tests and confirm missing fields/schema-5 behavior fail**

Run: `python -m pytest tests/unit/test_settings.py tests/unit/test_stereo_renderer.py tests/unit/test_cli.py -q`

- [x] **Step 3: Implement schema 5 validation and device-aware omission resolution**

```python
stereo_render_mode: Literal["fast", "quality"] = field(
    default="fast", kw_only=True
)
occlusion_fill_max_px: int = field(default=8, kw_only=True)
```

Validate `vr_resolution` against exact presets or `custom:<width>x<height>` and add CLI choices `--stereo-render-mode` and `--occlusion-fill-max-px`. Resolve omitted new CUDA jobs to Quality only at the application boundary; library defaults stay Fast.

- [x] **Step 4: Dispatch before legacy geometry construction**

Add `_render_relative_fast(...)` containing the current body. `render()` branches on `settings.stereo_render_mode` before any `build_relative_geometry` call; initially the Quality branch imports and calls the Task 4 entry point.

- [x] **Step 5: Run focused tests and frozen Fast hashes**

Run: `python -m pytest tests/unit/test_settings.py tests/unit/test_stereo_renderer.py tests/unit/test_cli.py -q`

- [x] **Step 6: Commit**

```text
feat: add typed stereo quality settings
```

### Task 2: Promote Exact Native Geometry and Region Core

**Files:**
- Create: `src/depth_surge_3d/rendering/quality/__init__.py`
- Create: `src/depth_surge_3d/rendering/quality/implicit_kd.py`
- Create: `src/depth_surge_3d/rendering/quality/native_kd.py`
- Create: `src/depth_surge_3d/rendering/quality/indexed_heap.py`
- Create: `src/depth_surge_3d/rendering/quality/region_solver.py`
- Create: `src/depth_surge_3d/rendering/quality/native_region_solver.py`
- Create: `src/depth_surge_3d/rendering/quality/geometry.py`
- Modify: `hatch_build.py`
- Modify: `scripts/build_quality_task0_native.py`
- Test: `tests/unit/test_quality_production_core.py`
- Test: `tests/unit/test_quality_native_wheel.py`

**Interfaces:**
- Produces: `build_relative_quality_geometry(...) -> QualityGeometryResult`.
- Produces: `build_metric_quality_geometry(...) -> QualityGeometryResult`.
- Produces: single-eye `build_relative_eye_offsets_into(...)` and `build_geometry_eye_offsets_into(...)`.
- Consumes: wheel-root `_quality_kd_native` with frozen build info `quality-implicit-region-kd-median-of-medians-v1`.

- [x] **Step 1: Add RED import/parity tests against the Task 0 scalar fixtures**

```python
def test_production_geometry_matches_gate_prototype(relative_fixture):
    actual = production.build_relative_quality_geometry(**relative_fixture)
    expected = prototype.build_relative_quality_geometry(**relative_fixture)
    assert array_hashes(actual) == array_hashes(expected)
```

Include half-lane ULPs, `G/Q` 0.5/1/2/4, sparse-region rejection, geodesic ties, CPU/CUDA-context equality, and native build metadata.

- [x] **Step 2: Run tests and confirm production modules are absent**

Run: `python -m pytest tests/unit/test_quality_production_core.py tests/unit/test_quality_native_wheel.py -q`

- [x] **Step 3: Mechanically promote gate-proven modules and rewrite only package imports**

Promote exact implementations from `scripts/quality_task0`; production code must not import `scripts`. Keep oracle-only helpers private and retain all checked arena/visit caps.

- [x] **Step 4: Build the extension and run parity tests**

Run: `python scripts/build_quality_task0_native.py`

Run: `python -m pytest tests/unit/test_quality_production_core.py tests/unit/test_quality_native_wheel.py -q`

- [x] **Step 5: Build and inspect a wheel**

Run: `python -m build --wheel`

Assert a clean subprocess imports both `depth_surge_3d.rendering.quality` and `_quality_kd_native` from the installed wheel.

- [x] **Step 6: Commit**

```text
feat: promote quality geometry native core
```

### Task 3: Promote Bounded Repair Planning

**Files:**
- Create: `src/depth_surge_3d/rendering/quality/repair_records.py`
- Create: `src/depth_surge_3d/rendering/quality/native_repair.py`
- Create: `src/depth_surge_3d/rendering/quality/local_strip.py`
- Create: `src/depth_surge_3d/rendering/quality/fallback.py`
- Create: `src/depth_surge_3d/rendering/quality/exemplar.py`
- Create: `src/depth_surge_3d/rendering/quality/planner.py`
- Test: `tests/unit/test_quality_production_repair.py`

**Interfaces:**
- Consumes: `QualityVisibilityAnalysis`, `QualityRepairBudgets`, and native implicit k-d index.
- Produces: `plan_quality_repairs(analysis, *, render_shape, local_limit_px, budgets) -> QualityRepairPlan`.
- Produces: a final sorted 16-byte record table with backend values 1/2/3 and immutable scalar diagnostics.

- [x] **Step 1: Add RED production parity and global-component tests**

```python
def test_repair_plan_covers_every_invalid_lane_once(pass_a_fixture):
    plan = plan_quality_repairs(**pass_a_fixture)
    assert validate_record_coverage(plan.records, pass_a_fixture["invalid_mask"]) == 0
    assert set(plan.records["backend"]) <= {1, 2, 3}
```

Cover equal-depth splits, cross-band components, local direction, safe donors, shared left/right budgets, exemplar exhaustion, no-donor failure, and fallback visit overflow.

- [x] **Step 2: Run tests and confirm planner import fails**

Run: `python -m pytest tests/unit/test_quality_production_repair.py -q`

- [x] **Step 3: Promote the gate-proven repair modules**

Keep record dtype offsets `0,4,6,10,13,14,15`, record/graph/repair arena caps at 64 MiB each, and the exact local/exemplar/fallback budget order.

- [x] **Step 4: Implement the global sparse component coordinator**

`planner.py` sorts records, performs exact horizontal/vertical/in-pixel unions, streams canonical component ranges, calls local then fallback-precompute then exemplar, finalizes fallback, and returns aggregate counters without a dense repair map.

- [x] **Step 5: Run repair tests and Task 0 oracle tests together**

Run: `python -m pytest tests/unit/test_quality_production_repair.py tests/unit/test_quality_task0_local_strip.py tests/unit/test_quality_task0_fallback.py tests/unit/test_quality_task0_exemplar.py -q`

- [x] **Step 6: Commit**

```text
feat: add bounded quality repair planner
```

### Task 4: Implement Quality Visibility Replay and Compact Renderer

**Files:**
- Create: `src/depth_surge_3d/rendering/quality/types.py`
- Create: `src/depth_surge_3d/rendering/quality/visibility.py`
- Create: `src/depth_surge_3d/rendering/quality/renderer.py`
- Modify: `src/depth_surge_3d/rendering/stereo_renderer.py`
- Modify: `src/depth_surge_3d/rendering/forward_splat.py`
- Test: `tests/unit/test_quality_renderer.py`
- Test: `tests/unit/test_stereo_renderer.py`

**Interfaces:**
- Produces: `QualityStereoControls`, `RelativeStereoPrimitiveInput`, `MetricStereoPrimitiveInput`, and `CompactStereoRenderResult`.
- Produces: `render_relative_primitives_compact(...)` and `render_metric_primitives_compact(...)`.
- Produces: public `StereoRenderer.render(... quality ...) -> StereoRenderResult` by materializing four independent boolean masks.

- [x] **Step 1: Add RED Pass A/Pass B tests with independent fine-grid oracle**

```python
def test_quality_replay_never_changes_valid_winner_lanes(edge_fixture):
    compact = render_relative_primitives_compact(**edge_fixture)
    oracle = scalar_quality_render(**edge_fixture)
    assert np.array_equal(compact.left_image, oracle.left_image)
    assert np.array_equal(compact.left_coverage_count, oracle.left_coverage_count)
```

Cover pure/mixed proxies, 16-lane ties-to-even, per-run region ownership, partial/full holes, repair-bit values, none mode, background mode, one-row/full-band equality, and both OOM retry boundaries.

- [x] **Step 2: Run focused tests and observe missing renderer failure**

Run: `python -m pytest tests/unit/test_quality_renderer.py -q`

- [x] **Step 3: Add winner-index visibility extraction without changing Fast**

Expose an internal winner-key result used only by `quality.visibility`. Pass A reduces each active band into planar coverage/pure proxy arrays plus sparse records, then releases all fine tensors.

- [x] **Step 4: Add Pass B replay and compact result**

Recompute visibility with the immutable single-eye offset map, verify mask OR/popcount equality, scatter only final record colours into invalid lanes, use the fixed reduction tree, and return RGB plus `uint8` coverage/repair planes.

- [x] **Step 5: Add exact OOM rollback and sequential-eye lifetimes**

An OOM at `h0` resets the current eye and retries from row zero at `h1`; an OOM at `h1` raises with both heights. Preserve shared budget checkpoints and completed earlier-eye compact output only.

- [x] **Step 6: Wire the public wrapper and rerun frozen Fast tests**

Run: `python -m pytest tests/unit/test_quality_renderer.py tests/unit/test_stereo_renderer.py tests/unit/test_stereo_edge_coverage.py -q`

- [x] **Step 7: Commit**

```text
feat: add compact two-pass quality renderer
```

### Task 5: Integrate Native Primitive File Pipeline and Diagnostics

**Files:**
- Create: `src/depth_surge_3d/rendering/quality/diagnostics.py`
- Create: `src/depth_surge_3d/rendering/quality/diagnostic_transaction.py`
- Modify: `src/depth_surge_3d/processing/frames/metric_geometry.py`
- Modify: `src/depth_surge_3d/processing/frames/stereo_generator.py`
- Modify: `src/depth_surge_3d/io/resume.py`
- Test: `tests/unit/test_quality_diagnostics.py`
- Test: `tests/unit/test_stereo_generator.py`
- Test: `tests/unit/test_resume.py`

**Interfaces:**
- Produces: four discriminated plan variants `fast_relative`, `quality_relative`, `fast_metric`, `quality_metric`.
- Produces: replayable lazy work provider, capacity-one Quality lifecycle, per-frame `diagnostics-v3/<frame>/manifest.json`, and root summary.
- Consumes: compact RGB/coverage/repair without public boolean-mask allocation.

- [ ] **Step 1: Add RED file-pipeline tests**

Assert decoder workers retain uint16 relative payloads/native metric objects, Quality capacity is exactly one, work enumeration is replayable and lazy, and impossible plan/input combinations fail before rendering.

- [ ] **Step 2: Add RED diagnostics transaction and resume tests**

Inject failures after every payload write, payload sync, frame-manifest replace, directory sync, root-summary replace, cleanup transition, and `keep_intermediates=false` pruning boundary. Assert partial generations are never reused as committed frames.

- [ ] **Step 3: Run focused tests and confirm expected failures**

Run: `python -m pytest tests/unit/test_quality_diagnostics.py tests/unit/test_stereo_generator.py tests/unit/test_resume.py -q`

- [ ] **Step 4: Implement the discriminated pipeline and compact writer**

Persist RGB atomically, write coverage/repair PNGs and canonical statistics only for Quality, publish frame manifest last, then update the root summary. Fast writes the same RGB bytes and legacy stage metadata while collecting only its specified bounded counters.

- [ ] **Step 5: Implement strict resume/audit disposition**

Validate hashes, schemas, frame order, algorithm identities, source content manifest, and compact pixel counts before reuse; distinguish absent, incomplete, pruned, and validation-failed states exactly.

- [ ] **Step 6: Run file-pipeline and resume suites**

Run: `python -m pytest tests/unit/test_quality_diagnostics.py tests/unit/test_stereo_generator.py tests/unit/test_resume.py -q`

- [ ] **Step 7: Commit**

```text
feat: persist quality diagnostics transaction
```

### Task 6: Promote Bounded Metric NPZ Content I/O

**Files:**
- Create: `src/depth_surge_3d/io/metric_npz.py`
- Create: `src/depth_surge_3d/io/content_manifest.py`
- Modify: `src/depth_surge_3d/processing/frames/metric_geometry.py`
- Test: `tests/unit/test_metric_npz.py`
- Test: `tests/unit/test_quality_content_manifest.py`

**Interfaces:**
- Produces: `write_metric_npz(...)`, `audit_metric_npz(...)`, and `audit_metric_sequence(...)` with one-frame ownership.
- Preserves: supported historical NumPy NPZ layouts through the frozen allowlist.

- [ ] **Step 1: Add RED production golden/strict-parser tests**

Reuse all owned/current-Numpy/legacy goldens; reject malformed flags, descriptors, duplicate members, CRCs, dtype/shape mismatch, trailing data, symlinks, and allocation failures.

- [ ] **Step 2: Run tests and confirm production APIs are missing**

Run: `python -m pytest tests/unit/test_metric_npz.py tests/unit/test_quality_content_manifest.py -q`

- [ ] **Step 3: Promote the gate-proven reader/writer and bind the metric store**

Ensure no complete second member is resident, validate native arrays in place, freeze them read-only, and release the owned frame before advancing.

- [ ] **Step 4: Run goldens, RSS checks, and metric pipeline tests**

Run: `python -m pytest tests/unit/test_metric_npz.py tests/unit/test_quality_content_manifest.py tests/unit/test_metric_geometry.py -q`

- [ ] **Step 5: Commit**

```text
feat: add bounded metric content io
```

### Task 7: Implement Canonical Settings Artifact V5 and Reservation Recovery

**Files:**
- Create: `src/depth_surge_3d/io/canonical_json.py`
- Create: `src/depth_surge_3d/io/settings_artifact.py`
- Create: `src/depth_surge_3d/io/reservation_protocol.py`
- Modify: `src/depth_surge_3d/io/operations.py`
- Modify: `src/depth_surge_3d/io/resume.py`
- Test: `tests/unit/test_settings_artifact_v5.py`
- Test: `tests/unit/test_reservation_protocol.py`

**Interfaces:**
- Produces: strict `ProcessingSettingsArtifactV5` parse/encode with duplicate-key rejection and content fingerprint.
- Produces: `SettingsArtifactTransactionV1`, `ReservedArtifactTransaction`, `JobControlV1`, and recovery entry order.
- Produces: `job_output_name_v1`, `PortableFinalComponentV1`, and legacy contained-component audit.

- [ ] **Step 1: Add RED schema and byte-contract tests**

Test the exact six top-level keys, all owned closed subobjects, scalar strings, u64/binary64 distinctions, UTC microseconds, revision equations, canonical bytes, and source schema provenance.

- [ ] **Step 2: Add RED filename and crash-window tests**

Freeze every canonical device-name/superscript golden, 239/240/241 byte and UTF-16 boundaries, write-ahead semantic payload recovery, complete-full/complete-short/renamed phases, allocation evidence at readiness, and truncate-before-sync recovery.

- [ ] **Step 3: Run tests and observe missing transaction APIs**

Run: `python -m pytest tests/unit/test_settings_artifact_v5.py tests/unit/test_reservation_protocol.py -q`

- [ ] **Step 4: Promote reservation core and implement strict settings codec**

Use descriptor phase to decide whether readiness allocation evidence is rechecked. Never re-extend an exact complete-short payload; retry sync/truncate release proof and retain it on `ReservationIncompleteError`.

- [ ] **Step 5: Replace ad hoc settings writes with the transaction**

Create/migrate schema 5 before frame mutation, persist immutable expected target and producer marker, and reconcile JobControl-owned terminal extents before ordinary stage reuse.

- [ ] **Step 6: Run all settings/resume/reservation tests**

Run: `python -m pytest tests/unit/test_settings.py tests/unit/test_settings_artifact_v5.py tests/unit/test_reservation_protocol.py tests/unit/test_resume.py -q`

- [ ] **Step 7: Commit**

```text
feat: add recoverable settings artifact v5
```

### Task 8: Stream Final Encoding and Irreversible Publication V4

**Files:**
- Create: `src/depth_surge_3d/processing/video/encoding_provider.py`
- Create: `src/depth_surge_3d/processing/video/final_publication.py`
- Modify: `src/depth_surge_3d/processing/video/video_encoder.py`
- Modify: `src/depth_surge_3d/processing/orchestration/pipeline_orchestrator.py`
- Test: `tests/unit/test_encoding_provider.py`
- Test: `tests/unit/test_final_publication.py`
- Test: `tests/unit/test_video_encoder.py`

**Interfaces:**
- Produces: `EncodingSequenceProvider`, canonical encoding-input manifest, strict FFprobe parsers, and scalar image2/concat argv construction.
- Produces: sibling-video validation and publication order `video -> input manifest -> final manifest` with actual V/I/P audit classification.

- [x] **Step 1: Add RED provider/FFprobe/progress tests**

Cover N=1 and `4_194_304`, signed image2 endpoints, generation replay, provider mutation, FFmpeg 5/6/7 JSON wrappers, duplicate authoritative keys, bounded output, and progress monotonicity.

- [x] **Step 2: Add RED publication/reencode crash tests**

Inject at final replacement, input-manifest replacement, final-manifest replacement, post-boundary reencode, and descriptor retirement. Assert the approved irreversible boundary keeps actual evidence and never promises restoration of the old triplet.

- [x] **Step 3: Run focused tests and confirm old list-based encoder fails**

Run: `python -m pytest tests/unit/test_encoding_provider.py tests/unit/test_final_publication.py tests/unit/test_video_encoder.py -q`

- [x] **Step 4: Promote provider and integrate reservations**

Audit once, launch only after producer-marker/settings commit, revalidate the same generation after FFmpeg, validate sibling media, then publish each artifact using its reserved extent and index-last retirement.

- [x] **Step 5: Run encoding, reservation, and direct-VR integration tests**

Run: `python -m pytest tests/unit/test_encoding_provider.py tests/unit/test_final_publication.py tests/unit/test_video_encoder.py tests/integration/test_direct_vr_ffmpeg_encoding.py -q`

- [x] **Step 6: Commit**

```text
feat: stream and recover final publication v4
```

### Task 9: Complete CLI/Web/Resume Compatibility and Documentation

**Files:**
- Modify: `app.py`
- Modify: `src/depth_surge_3d/cli.py`
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `docs/USER_GUIDE.md`
- Test: `tests/unit/test_app.py`
- Test: `tests/unit/test_cli.py`
- Test: `tests/unit/test_resume.py`

**Interfaces:**
- Consumes: the single settings validator and resolver from Task 1.
- Produces: two visible modes (`fast`, `quality`), limit input 1..32, CPU warning, persisted schema-5 overrides, and mode-specific resume identity.

- [ ] **Step 1: Add RED CLI/Web/resume tests**

Assert new CUDA omission resolves Quality, CPU omission resolves Fast, explicit CPU Quality warns, schema-5 resume retains persisted values, explicit override wins, and schema 1..4 migration stays Fast/8.

- [ ] **Step 2: Run focused tests and confirm UI/settings mismatch**

Run: `python -m pytest tests/unit/test_app.py tests/unit/test_cli.py tests/unit/test_resume.py -q`

- [ ] **Step 3: Implement controls and update operational documentation**

Use a mode selector and bounded numeric input; document that Quality is offline/highest-quality, Fast is legacy-compatible, diagnostics live under the job root, and failures never silently downgrade.

- [ ] **Step 4: Run CLI/Web/resume tests**

Run: `python -m pytest tests/unit/test_app.py tests/unit/test_cli.py tests/unit/test_resume.py -q`

- [ ] **Step 5: Commit**

```text
feat: expose offline stereo quality mode
```

### Task 10: Repository, Package, Fault, and Real-fixture Release Gate

**Files:**
- Create: `scripts/verify_stereo_quality_fixture.py`
- Create: `docs/superpowers/reports/2026-08-24-stereo-quality-production.json`
- Modify: this plan's checkboxes only after each command succeeds.

**Interfaces:**
- Produces: one canonical production gate report tied to the tested commit/diff and user fixture settings.

- [ ] **Step 1: Run focused Quality and Fast compatibility tests**

Run: `python -m pytest tests/unit/test_quality_*.py tests/unit/test_stereo_renderer.py tests/unit/test_stereo_generator.py tests/unit/test_resume.py tests/unit/test_video_encoder.py -q`

- [ ] **Step 2: Run the complete repository suite in the supported environment**

Run: `$env:PATH='E:\Code\depth-surge-3d\.venv\Scripts;' + $env:PATH; python -m pytest -q`

- [ ] **Step 3: Run formatting, lint, type, and diff checks**

Run scoped Black/flake8/mypy for every changed production/test file, record pre-existing unrelated baseline debt, then run `git diff --check` and inspect `git status --short`.

- [ ] **Step 4: Build and smoke-test the wheel**

Install the wheel into a clean temporary virtual environment, import the native extension, run CLI help, and render the deterministic Quality edge fixture.

- [ ] **Step 5: Render the user fixture in both modes**

Use `H:\3dtest\1787051840_f908a5f038277cf447b8a6a9b5072311_20260818_191720\f908a5f038277cf447b8a6a9b5072311_1787051942-settings.json`. Freeze source hashes, render the seven-frame review set, export matching Fast/Quality crops around curls, feather, gun barrels, and ribbon, and record coverage/repair statistics.

- [ ] **Step 6: Inspect the actual crops and reject regressions**

Verify Quality removes boundary stretching/halos without foreground erosion, temporal flicker, black holes, or cross-depth donor leakage; Fast outputs must match the pre-change hashes.

- [ ] **Step 7: Publish the canonical production report and commit**

```text
feat: complete stereo quality production gate
```
