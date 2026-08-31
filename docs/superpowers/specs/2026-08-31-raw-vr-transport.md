# Raw RGB24 VR Transport

## Goal

Reduce total conversion time without changing image quality, frame order,
encoding settings, audio, bounded ownership, or restart semantics. Remove the
lossless PNG encode/decode round trip from transient assembled-VR output.

## Evidence And Decision

The previous session encoded each assembled BGR8 frame to PNG, copied the encoded
bytes, hashed them, and piped them to FFmpeg's PNG decoder. The existing default
PNG policy already beat the tested compression alternatives. A real 24-frame
3840x1080 transport/encoding ABBA probe measured NVENC 4.44 s to 1.06 s and
x264 5.93 s to 1.92 s with raw RGB24. Within each encoder all four MP4 SHA-256
identities, decoded frames, and stream metadata matched. This excludes stereo
rendering and publication. The later 48-frame full-stage ABBA result was
58.58 s to 55.69 s for Quality (4.9 percent), and effectively unchanged for Fast
(43.86 s to 43.75 s). That scope includes rendering through cleanup and terminal
audit, not upstream extraction or depth inference. The fully overlapped Fast
PNG writer was not the stage's limiting operation.

Choose RGB24 rather than BGR24 to preserve the PNG decoder's input pixel format
to FFmpeg's existing color conversion. Do not change presets, CRF/CQ, frame rate,
chroma sampling, audio selection, or validation passes. No new dependency.

## Transport

New `open_streamed_vr_session` calls select `rawvideo`, `rgb24`, and an explicit
`video_size` before the image input. Input arrays retain the existing BGR8
contract. Convert to RGB24 and freeze immutable bytes before hashing and writing.
Keep source arrays unchanged, including read-only and strided views. Use the
existing bounded pipe, backpressure, cancellation, drain, and failure cleanup.

## Evidence And Compatibility

Retain reading/writing legacy PNG manifest algorithms v2/v3. Add input-manifest
algorithm `encoding-input-sequence-v4`, schema 3, only for fully evidence-bound
raw RGB24 streaming. Every frame identity has exactly `byte_count`, `raw_frame`,
`relative_path`, and `sha256`. `raw_frame` has exactly `height`, `width`, and
`pixel_format` (`rgb24`). Payload length is exactly `3 * width * height` and
fits uint64. Dimensions are positive uint32 values. Names end in `.rgb` and
follow the same ordered frame stems and logical parent as the existing provider;
these are stream identities, not claims that raw frame files exist on disk.

The provider still supplies the existing frame order and generation audit. Only
the raw stream's evidence maps its nominal path to `.rgb`; file-backed provider
and PNG directory contracts are unchanged. Never stage or validate raw payloads
as PNG files.

V4 requires invocation and completed-output evidence. The invocation's raw input
options must bind the first input, match its dimensions, and be unambiguous.
Reject wrong formats, duplicate/missing options, option aliases/overrides, mixed
PNG/raw identities, wrong byte counts, wrong sequence names, and version/transport
mismatches. The final container contract must require raw-manifest evidence for
a raw transport and must not accept raw evidence for image2/image2pipe.

Completed legacy PNG siblings remain recoverable without re-encoding. Completed
raw siblings use the same authenticated handoff and publication path. Incomplete
or unsupported jobs retain the existing file-backed fallback; no weakened audit
or unbounded frame retention is allowed.

## Memory

Charge all simultaneously live parent-process buffers: assembled BGR8, converted
RGB24, and immutable RGB24 bytes, totaling `9 * final_pixels`, plus existing
held transform outputs and distortion maps. Preserve the larger transform-phase
charge when it dominates. The shared host admission budget stays 512 MiB.

## Acceptance

- Real encoded output and per-frame pixels agree with PNG for NVENC and x264;
  exercise the supported x265 path and audio separately.
- Raw identities and normalized arguments are validated independently before
  publication/recovery; old v2/v3 test fixtures remain valid.
- Tests cover order, lengths, dtype/shape/strides, immutability, malformed
  declarations, truncation, transport confusion, cancellation, failure and resume.
- Measure actual transient stereo plus encode/publish work on real frames in
  alternating order. Report scope, output hashes, and capacity/memory separately.
- Do not commit, push, remove unrelated files, or change the native renderer.

## Verification Scope

The user-selectable NVENC and x264 paths are tested with side-by-side and
over-under layouts, with and without audio. The existing lower-level x265 path
cannot be selected by schema-5 settings; forcing that resolved codec exposed
the same legacy PNG failure from x265 informational stderr under the existing
fail-closed process policy. Do not weaken that policy or claim successful x265
publication as part of this transport change.

Evidence is under `build/raw-vr-transport-20260831/`: `probe-results.json`,
`allocation-check/allocations.json`, `full-stage-abba48/full-stage.json`, and the
supported-codec/audio/layout results. Large frame-allocation checks cover
contiguous, transposed and negative-stride views; observed new allocations stay
at two frame buffers, plus small object overhead, after counting the existing
source backing buffer separately. This is not a total-process RSS measurement.
