# Raw RGB24 VR Transport Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution
> with test and review checkpoints. Keep prior worktree changes intact.

**Goal:** Remove PNG compression/decompression from transient video generation
without sacrificing quality, memory bounds, or recovery guarantees.

**Architecture:** Add a versioned raw-frame identity alongside existing PNG
identities, retain the bounded process and publication state machine, then select
RGB24 for new sessions. Legacy PNG readers and file-backed paths remain intact.

**Tech Stack:** Python, NumPy, OpenCV, FFmpeg, pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-31-raw-vr-transport.md`

## Global Constraints

- No changes to presets, CRF/CQ, frame rate, chroma sampling, or audio selection.
- No unbounded frame retention; shared host admission stays 512 MiB.
- Preserve legacy PNG v2/v3 recovery and immutable authenticated pipe payloads.
- No commit/push or cleanup of unrelated files.

## Task 1: Raw Stream Evidence

Files: `streamed_vr_evidence.py`, `streamed_vr_manifest.py`,
`final_publication.py`, and new `tests/unit/test_streamed_vr_raw.py`.

Interfaces: `StreamedVrInputManifestV1(..., input_format="rgb24")` accepts exact
immutable RGB24 payloads; `validate_raw_rgb24_input_v1(arguments)` returns the
declared `(width, height)` or raises `ValueError`; parsed projections retain
`raw_pixel_format="rgb24"` for independent transport checks.

- [x] Add failing real-spool v4 round-trip and malformed-identity tests.
- [x] Run `.venv/Scripts/python.exe -m pytest tests/unit/test_streamed_vr_raw.py -q`.
- [x] Implement raw command validation, versioned identities, and strict bounded
      parsing. Keep PNG identity and canonical v2/v3 bytes unchanged.
- [x] Run raw tests plus existing streamed-manifest and final-publication suites.

## Task 2: New Sessions And Memory

Files: `video_encoder.py`, `pipeline_orchestrator.py`,
`tests/unit/test_video_encoder.py`, `tests/unit/test_pipeline_orchestrator.py`.

Interfaces: `_build_streamed_vr_command(..., input_format="rgb24")` emits one
explicit raw input; new sessions choose that mode. `write` converts BGR8 to
immutable RGB24 bytes and appends exactly those bytes before bounded pipe writes.

- [x] Add failing command, byte/pixel identity, immutable-source, and memory tests.
- [x] Implement explicit raw selection and `9 * final_pixels` pipe peak.
- [x] Update existing new-session assertions to v4/raw; retain explicit PNG
      compatibility fixtures instead of replacing their coverage.
- [x] Run encoder, orchestration, bounded-process, and recovery tests.

## Task 3: Integrated Verification

Files: `docs/ARCHITECTURE.md`; reproducible probe evidence under
`build/raw-vr-transport-20260831/` (not shipped as a production API).

- [x] Compare real final videos for supported codecs, audio, and layouts.
- [x] Measure a warmed full transient-stage PNG/raw/raw/PNG comparison using real
      frames, real schema-5 publication, and unchanged rendering settings.
- [x] Compare decoded/video/diagnostic identities and inspect bounded capacities.
- [x] Obtain independent code review, resolve findings, run affected suites,
      formatting and `git diff --check`, then report verified scope and limits.

## Results And Limits

- Final affected-suite run: 471 passed, 1 explicitly deselected, 295.35 s.
  The excluded test is the 90,000-frame diagnostic-metadata recovery fixture;
  it was not rerun successfully in this round. The raw manifest has a separate
  8,192-frame bounded-reader regression.
- NVENC/x264, both layouts, with/without audio: eight real publication comparisons
  preserved complete MP4 bytes, decoded frames/audio, and stream metadata.
- Forced lower-level x265 is not selectable in schema-5 settings and hits the
  same existing informational-stderr rejection in PNG and raw tests. This is
  negative parity coverage, not a successful x265 publication claim.
- Full 48-frame generation through cleanup and terminal audit: Quality
  58.58 s -> 55.69 s (4.9 percent); Fast 43.86 s -> 43.75 s (within noise).
  Both modes preserve assembled pixels, MP4 bytes, and diagnostic summaries.
- Independent review found no remaining actionable issue. A suspected fourth
  frame allocation was withdrawn after lifetime analysis and allocation probes.
- Black checks and `git diff --check` pass. No commit, push, native rebuild, or
  unrelated cleanup. Details: `build/raw-vr-transport-20260831/verification.json`.
