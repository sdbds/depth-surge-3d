# Quality Direct Image2Pipe Design

## Goal

Reduce Quality-mode wall time without changing rendered pixels or encoder quality by
removing the durable `04_left_frames` / `04_right_frames` PNG round trip when the
result is going directly to the final VR video.

## Eligibility

The transient path is selected only when all of these conditions hold:

- `stereo_render_mode == "quality"`
- `stereo_geometry_mode == "relative"`
- `direct_vr_encode == true`
- `keep_intermediates == false`
- `upscale_model == "none"`
- no active final-encoding reservation is being recovered
- either fused fisheye distortion is enabled, or distortion is disabled with a
  no-op crop and the rendered eye size already equals the requested per-eye size

Every other configuration continues through the current file-backed stages.

## Data Flow

The durable guide and relative-disparity manifest remains the upstream authority.
Quality decode and render retain the current bounded worker pipeline, but the
coordinator releases rendered frames to one writer in source order. The writer:

1. applies the same in-memory distortion and crop operations used by the file path;
2. assembles the two eyes into the configured VR layout;
3. encodes that assembled uint8 frame to a lossless PNG in memory;
4. writes the PNG bytes to FFmpeg through `image2pipe`;
5. appends the exact PNG identity to a streamed encoding-input manifest; and
6. appends the already prepared Quality statistics to one transient JSONL extent.

The existing lifecycle permits remain the memory bound. FFmpeg pipe backpressure
propagates through the single writer and stops additional rendering when the fixed
pipeline capacity is full.

## Diagnostics And Publication

Transient Quality jobs do not publish per-frame RGB, statistics, or frame-manifest
files. They publish the Quality input-content manifest, one source-ordered statistics
JSONL file, its aggregate summary, and complete diagnostics metadata. The ordered
diagnostic fingerprint is derived from each frame name and its exact canonical
statistics bytes. This aggregate is sufficient for final cleanup preflight but is
deliberately not reusable as a retained stereo frame stage.

The streamed encoding-input manifest records the exact assembled PNG bytes actually
sent to FFmpeg. It uses the existing assembled-VR manifest shape with logical frame
paths; the paths name the stream records rather than persisted files. After FFmpeg
closes successfully, the manifest is copied into the reserved input extent, the
encoded temporary is atomically moved to the reservation sibling, and the existing
container validation and final-publication transaction publish the result.

## Failure And Recovery

Before the final reservation exists, interruption terminates and reaps FFmpeg,
removes the private video and manifest temporaries, and leaves the upstream guide and
depth artifacts intact. A retry reruns the stereo stage from those durable inputs.

If a final reservation already exists, a new transient path is not entered. A
complete streamed-v3 sibling is recovered directly only after its manifest,
invocation, and output evidence authenticate the same reservation; other states
continue through file-backed recovery. `keep_intermediates=true` also always
retains the existing durable per-frame transaction behavior.

## Quality Contract

- Distortion, crop, and VR layout use the same functions and parameters as the file
  path.
- Every image2pipe payload is a lossless PNG of the final pre-encode VR frame.
- Tests compare exact pre-encode frame bytes and decoded final frames against the
  file-backed path.
- Encoder selection, codec quality arguments, frame rate, audio selection, final
  container validation, and atomic publication remain unchanged.

## Non-Goals

- No storage-device recommendation or scratch-drive selection.
- No weakening of per-frame durability for retained intermediates.
- No transient upscaling in this iteration.
- No rawvideo transport until profiling shows in-memory PNG encoding is the new
  limiting stage.
