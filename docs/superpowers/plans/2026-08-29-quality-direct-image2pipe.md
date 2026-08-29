# Quality Direct Image2Pipe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream eligible relative Quality frames directly into FFmpeg without materializing stereo or cropped PNG stages.

**Architecture:** Add source-ordered delivery to the existing bounded stereo pipeline, a one-sync transient diagnostics spool, and an image2pipe encoding session whose manifest authenticates the exact PNG payloads sent to FFmpeg. Select it only for the approved no-retention configuration and retain the current file-backed pipeline as the recovery and unsupported-settings fallback.

**Tech Stack:** Python 3.11+, NumPy, OpenCV, FFmpeg image2pipe, pytest, existing schema-5 reservation/publication APIs.

**Spec:** `docs/superpowers/specs/2026-08-29-quality-direct-image2pipe-design.md`

## Global Constraints

- Rendered and transformed pixels must remain byte-exact before lossy video encoding.
- Pipeline residency must remain capacity-bounded; no frame-count-sized image list is allowed.
- Retained intermediates and existing recovery behavior remain file-backed.
- An active final reservation disables the transient path.
- Do not commit or push during this execution unless the user explicitly requests it.

---

### Task 1: Source-Ordered Bounded Delivery

**Files:**
- Modify: `src/depth_surge_3d/processing/frames/stereo_control.py`
- Test: `tests/unit/test_stereo_control.py`

**Interfaces:**
- Consumes: existing `StereoProviderItem.source_index` and lifecycle permits.
- Produces: `run_streaming_stereo_pipeline(..., ordered_render: bool = False)`.

- [ ] **Step 1: Write a failing test** whose decoders finish out of order and asserts that `render` and the single writer observe source order when `ordered_render=True`.
- [ ] **Step 2: Run the focused test** with `pytest tests/unit/test_stereo_control.py -k ordered_render -vv` and confirm the missing keyword or wrong ordering causes the failure.
- [ ] **Step 3: Implement bounded coordinator reordering** using at most the existing capacity and release all owned items on cancellation.
- [ ] **Step 4: Run the focused and complete stereo-control tests** and confirm both ordered and default completion-order behavior pass.

### Task 2: Transient Quality Diagnostics

**Files:**
- Modify: `src/depth_surge_3d/rendering/quality/diagnostic_transaction.py`
- Modify: `src/depth_surge_3d/rendering/quality/__init__.py`
- Test: `tests/unit/test_quality_diagnostics.py`

**Interfaces:**
- Produces: `TransientDiagnosticsSpool(context)`, `append(frame_name, stats_raw)`, `complete(...)`, and `abort()`.
- Guarantees: one ordered JSONL temporary, one sync/replace at completion, no per-frame files, bounded line parsing, and current aggregate metadata schema.

- [ ] **Step 1: Write failing tests** for complete aggregate publication without a `frames/` tree, source-order rejection, malformed statistics rejection, and temporary cleanup on abort.
- [ ] **Step 2: Run the focused tests** and verify failure is caused by the absent spool API.
- [ ] **Step 3: Implement the spool** by reusing the existing statistics parser, summary builder, streamed metadata builder, and durable temporary primitive.
- [ ] **Step 4: Run all Quality diagnostics tests** and verify retained per-frame transactions are unchanged.

### Task 3: Bounded FFmpeg Input Pipe

**Files:**
- Modify: `src/depth_surge_3d/processing/video/bounded_process.py`
- Test: `tests/unit/test_bounded_process.py`

**Interfaces:**
- Produces: `FfmpegInputPipeProcessV1.start(arguments)`, `write(payload)`, `finish()`, and `abort()`.
- Guarantees: chunked stdin writes, concurrent bounded progress/diagnostic drains, cancellation, stall watchdog, and unconditional child reaping.

- [ ] **Step 1: Write failing real-process tests** using the Python executable as a finite stdin consumer for successful transfer, early child failure, abort, and diagnostic overflow.
- [ ] **Step 2: Run the focused tests** and confirm the new process adapter is absent.
- [ ] **Step 3: Implement the adapter** with fixed buffers and a watchdog that terminates the child when progress stalls or cancellation is requested.
- [ ] **Step 4: Run bounded-process and video-encoder process tests** and confirm no process or thread remains live.

### Task 4: Streamed PNG Manifest

**Files:**
- Create: `src/depth_surge_3d/processing/video/streamed_vr_manifest.py`
- Test: `tests/unit/test_streamed_vr_manifest.py`

**Interfaces:**
- Produces: `StreamedVrManifestWriter(path, frame_count, first_index, width, height)`, `append(frame_name, png_payload)`, `complete() -> StreamedVrManifestEvidence`, and `abort()`.
- Evidence fields: `path`, `fingerprint`, `raw_sha256`, `raw_byte_count`, `frame_count`, `first_index`.

- [ ] **Step 1: Write failing tests** for a hand-derived two-frame canonical manifest, wrong order, wrong PNG shape, short/extra sequence, and bounded cleanup.
- [ ] **Step 2: Run the focused tests** and confirm import failure.
- [ ] **Step 3: Implement streamed canonical output** with a patched fingerprint placeholder and a final bounded SHA-256 pass.
- [ ] **Step 4: Parse the result through the production final-publication parser** and run the complete manifest test file.

### Task 5: Image2Pipe Encoding Session

**Files:**
- Modify: `src/depth_surge_3d/processing/video/video_encoder.py`
- Modify: `src/depth_surge_3d/processing/video/final_publication.py`
- Test: `tests/unit/test_video_encoder.py`
- Test: `tests/unit/test_final_publication.py`
- Test: `tests/integration/test_direct_vr_ffmpeg_encoding.py`

**Interfaces:**
- Produces: `VideoEncoder.open_streamed_vr_session(...) -> StreamedVrEncodingSession`.
- Session methods: `write(frame_name, vr_frame)`, `finish_and_publish() -> bool`, and `abort()`.

- [ ] **Step 1: Write failing command and publication tests** that require one `image2pipe` PNG input, no image start number, the existing codec/audio arguments, and an assembled-VR normalized input token.
- [ ] **Step 2: Write a failing FFmpeg integration test** that streams three in-memory frames and validates layout, rate, frame count, audio behavior, and final authentication.
- [ ] **Step 3: Run both focused test groups** and confirm failures identify the missing session and image2pipe normalized-argument rule.
- [ ] **Step 4: Implement the session** using the bounded process and streamed manifest, then stage the completed manifest into a newly prepared reservation, move the encoded temporary to its indexed sibling, validate the container, and invoke `FinalPublicationTransactionV4`.
- [ ] **Step 5: Extend normalized-argument validation** so assembled `image2pipe` input requires one framerate, zero start-number options, PNG codec declaration, and the existing frame bound.
- [ ] **Step 6: Run all video encoder, final publication, and direct FFmpeg tests**.

### Task 6: In-Memory Transform And Quality Generator Path

**Files:**
- Modify: `src/depth_surge_3d/processing/frames/stereo_generator.py`
- Modify: `src/depth_surge_3d/processing/frames/distortion_processor.py`
- Test: `tests/unit/test_stereo_generator.py`
- Test: `tests/unit/test_distortion_processor.py`

**Interfaces:**
- Produces: `DistortionProcessor.prepare_stream_transform(...)` and a callable that maps one stereo pair to target-sized eyes.
- Extends: `StereoPairGenerator.create_stereo_pairs_from_manifest(..., frame_consumer=None)`.

- [ ] **Step 1: Write failing transform equivalence tests** comparing in-memory results to the existing fused file path for distortion and the approved no-op rectilinear case.
- [ ] **Step 2: Write a failing generator test** asserting ordered consumer delivery, no left/right/cropped PNG files, complete transient diagnostics, and unchanged pixel arrays.
- [ ] **Step 3: Run focused tests** and confirm the new interfaces are absent.
- [ ] **Step 4: Extract the pure transform** from existing operations without changing file-backed behavior.
- [ ] **Step 5: Add the transient generator branch** that always renders from the durable upstream manifest, uses one ordered writer, appends statistics to the transient spool, and never calls `publish_quality_frame`.
- [ ] **Step 6: Run stereo generator, distortion, diagnostics, and stereo-control tests**.

### Task 7: Orchestrator Selection And End-To-End Verification

**Files:**
- Modify: `src/depth_surge_3d/processing/orchestration/pipeline_orchestrator.py`
- Modify: `docs/ARCHITECTURE.md`
- Test: `tests/unit/test_pipeline_orchestrator.py`
- Test: `tests/integration/test_moge2_pipeline.py`

**Interfaces:**
- Produces: a pure eligibility decision and a transient Step 3-to-Step 8 orchestration branch.
- Falls back to: the existing `_execute_remaining_steps` and `_execute_post_stereo_steps` paths.

- [ ] **Step 1: Write failing eligibility table tests** for every required gate, including active-reservation recovery and unsupported resize/upscale cases.
- [ ] **Step 2: Write a failing orchestration test** proving the stream session starts before stereo rendering, is aborted on failure, and publishes only after successful generation.
- [ ] **Step 3: Implement the branch** and preserve current progress/finalization semantics.
- [ ] **Step 4: Run the focused unit and integration tests**.
- [ ] **Step 5: Run exact pixel A/B verification** on a retained Quality fixture and compare every pre-encode VR PNG hash.
- [ ] **Step 6: Run a full-video A/B benchmark** and record stage wall time, throughput, and decoded-frame equivalence.
- [ ] **Step 7: Run formatting, static checks, `git diff --check`, and the complete test suite** before reporting completion.
