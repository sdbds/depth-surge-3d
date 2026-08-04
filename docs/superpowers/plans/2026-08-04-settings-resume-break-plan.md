# Settings, Cache, and Resume Break Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the intentional schema break by exposing only the new controls, validating stage fingerprints, preserving valid original frames, and handling legacy generated directories without silent deletion.

**Architecture:** One typed settings validator is shared by CLI, Web, projector, persisted settings, and resume. Resume validates each numbered stage independently and performs a one-way legacy migration: archive by default, delete only with an explicit current invocation. Old depth semantics are never inferred or reused.

**Tech Stack:** Python argparse, Flask payloads, JSON, pathlib, pytest.

## Global Constraints

- Removed user-supplied setting names fail validation; legacy on-disk names are reported and stripped during migration.
- Unattended migration defaults to `archive`; only explicit `migrate_legacy=delete` authorizes deletion.
- Valid `00_original_frames` survive every depth/stereo schema invalidation.
- No old depth PNG is heuristically migrated.

---

### Task 1: Shared Final Settings Schema

**Files:**
- Create: `src/depth_surge_3d/core/settings.py`
- Modify: `src/depth_surge_3d/core/constants.py`
- Test: `tests/unit/test_settings.py`

**Interfaces:**
- Produces: `validate_settings(values, *, source) -> dict` where source is `explicit` or `legacy_disk`.
- Produces defaults for stereo, scene, storage, I/O, and migration controls.

- [ ] **Step 1: Write failing tests for all defaults/ranges, unknown names, removed explicit names, and stripped legacy names**
- [ ] **Step 2: Run tests and verify import failure**
- [ ] **Step 3: Implement typed validation and replace old constants/ranges**
- [ ] **Step 4: Run settings tests green and commit with `feat: add final processing settings schema`**

### Task 2: CLI and Projector Controls

**Files:**
- Modify: `depth_surge_3d.py`
- Modify: `src/depth_surge_3d/rendering/stereo_projector.py`
- Test: `tests/unit/test_stereo_projector.py`
- Test: CLI parser tests in `tests/unit/test_imports.py`

**Interfaces:**
- Exposes: `stereo_strength`, `convergence`, `occlusion_fill`, scene controls, `raw_storage_dtype`, `stereo_io_workers`, and `migrate_legacy`.

- [ ] **Step 1: Write failing parser/signature tests asserting old names are absent and new names validate**
- [ ] **Step 2: Remove old CLI/projector parameters and add new controls**
- [ ] **Step 3: Run focused tests green and commit with `feat: expose final CLI stereo controls`**

### Task 3: Web Settings and Template

**Files:**
- Modify: `app.py`
- Modify: `templates/index.html`
- Test: `tests/unit/test_resume_template.py`
- Test: `tests/unit/test_headset_preset_template.py`
- Test: `tests/unit/test_see_through_entrypoints.py`

**Interfaces:**
- Consumes: shared settings validator.

- [ ] **Step 1: Write failing template/payload tests for new controls and absence of baseline, focal length, and hole-fill quality**
- [ ] **Step 2: Update visible controls, JavaScript payloads, defaults, and server validation**
- [ ] **Step 3: Run Web-focused tests green and commit with `feat: migrate Web settings to DIBR controls`**

### Task 4: Stage Resume Validation and Legacy Migration

**Files:**
- Create: `src/depth_surge_3d/io/resume.py`
- Modify: `src/depth_surge_3d/io/operations.py`
- Modify: `app.py`
- Modify: `depth_surge_3d.py`
- Test: `tests/unit/test_resume.py`
- Test: `tests/unit/test_io_operations.py`

**Interfaces:**
- Produces: `build_resume_report(output_dir, current_settings) -> ResumeReport`.
- Produces: `apply_legacy_migration(report, mode) -> None`.

- [ ] **Step 1: Write failing tests for original-frame preservation, candidate manifest handling, fingerprint invalidation, default archive, explicit delete, archive collision, and legacy settings backup**
- [ ] **Step 2: Run tests and verify failure**
- [ ] **Step 3: Implement per-stage validation and deterministic report generation without mutation**
- [ ] **Step 4: Implement atomic legacy settings backup plus archive/delete execution; never move valid original frames**
- [ ] **Step 5: Wire CLI/Web resume and run focused tests green**
- [ ] **Step 6: Commit with `feat: add stage-aware resume migration`**

### Task 5: Cache Schema and Dead Compatibility Removal

**Files:**
- Modify: `src/depth_surge_3d/utils/domain/depth_cache.py`
- Modify: `src/depth_surge_3d/utils/batch_analysis.py`
- Modify: `src/depth_surge_3d/utils/imaging/video_processing.py`
- Modify: current documentation and examples containing removed settings
- Test: `tests/unit/test_depth_cache.py`
- Test: `tests/unit/test_batch_analysis.py`
- Test: complete suite

**Interfaces:**
- Requires: exact final canonical metadata/model fingerprint for cache hits.
- Removes: legacy depth-directory discovery and compatibility defaults.

- [ ] **Step 1: Write failing cache-schema and repository absence tests**
- [ ] **Step 2: Update cache keys/copy rules and batch analysis for final directories**
- [ ] **Step 3: Remove dead compatibility code and obsolete docs/examples**
- [ ] **Step 4: Run `rg -n "baseline|focal_length|hole_fill_quality|02_depth_maps"` and classify only archived design/history references as allowed**
- [ ] **Step 5: Run the full test suite and commit with `refactor: remove legacy depth schema`**

