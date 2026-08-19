# Direct VR FFmpeg Encoding Design

## Status

The user approved the recommended orchestration-boundary design on 2026-08-14.
The feature is an opt-in aggressive optimization. The existing assembled-PNG
path remains the default and keeps its current output, preview, and recovery
behavior.

This document is partially superseded by
`2026-08-19-stereo-quality-edge-reconstruction-canonical.md`. Its
"Relationship to Final Encoding Specifications" authority map is controlling:
this document retains the decisions about opt-in behavior, source-selection
policy, layout, encoder quality, and basic UX, but not the eager source-resolver
return interface; the canonical specification replaces input validation,
audio and `-shortest` policy, command normalization, publication/manifests,
container validation, resume, and cleanup. The sentence above about unchanged
recovery behavior does not override that later transaction contract.

The accepted quality contract is visual and encoding quality parity, not
bit-identical pixels. When resizing is required, FFmpeg bicubic scaling may
differ numerically from OpenCV `INTER_CUBIC` without being treated as a
regression.

## Problem

The current final pipeline has two serial materialization steps:

1. `VRFrameAssembler` reads every final left/right eye pair, optionally resizes
   it, combines it into side-by-side or over-under layout, and writes a PNG to
   `99_vr_frames`.
2. `VideoEncoder` asks FFmpeg to read that assembled PNG sequence and encode the
   final video.

For the measured long-video workload, `99_vr_frames` accounts for about 8.7 GB
of avoidable writes. FFmpeg can read the two existing eye sequences, perform
the same layout operation in its filter graph, and feed the result directly to
the selected encoder. The intermediate stage is still useful for inspection
and restart granularity, so this behavior must be optional rather than a new
unconditional default.

## Goals

- Add an independent, user-visible direct-encoding option that defaults off.
- When enabled for a new job, do not create or write `99_vr_frames`.
- Let FFmpeg read the final left and right PNG sequences directly.
- Preserve side-by-side and over-under layout semantics.
- Preserve output dimensions, exact frame rate selection, encoder choice,
  encoder quality arguments, pixel format, audio trim, and output naming.
- Avoid scaling when the eye sequences already have the requested per-eye
  dimensions.
- Use quality-oriented bicubic scaling when dimensions differ.
- Validate sequence correspondence and continuity before launching FFmpeg.
- Report final encoding progress without generating assembled preview PNGs.
- Write the final video atomically so a failed encode cannot replace a previous
  valid output with a truncated file.
- Preserve all upstream resume artifacts after an encoding failure.
- Keep the existing assembled-PNG path behaviorally unchanged when the option
  is disabled.

## Non-goals

- Removing `99_vr_frames` or the `VRFrameAssembler` from the default pipeline.
- Making direct encoding the default.
- Producing bit-identical resized pixels to OpenCV `INTER_CUBIC`.
- Changing the NVENC preset, tune, codec, software CRF, software preset, or
  `yuv420p` output contract.
- Removing any upstream eye-image stage.
- Streaming depth estimation, stereo rendering, distortion, cropping, or AI
  upscaling directly into FFmpeg.
- Adding an assembled VR-frame preview during direct encoding.
- Adding direct encoding to the separate batch-stitching workflow, whose input
  contract is already assembled VR frames.
- Adding a resume-time UI override. Resume uses the strategy saved by the
  interrupted job.
- Automatically deleting a pre-existing `99_vr_frames` directory merely
  because direct encoding is selected.

## User-Facing Contract

Add a boolean processing setting named `direct_vr_encode`.

- Default: `false`.
- Validation: a strict boolean; truthy strings and integers are rejected.
- Placement: near the existing video encoder and intermediate-retention
  controls.
- Label: `Direct FFmpeg VR Encoding`.
- Help: a compact tooltip may identify it as an advanced option that skips the
  assembled VR-frame stage. No modal or extra confirmation is required.
- Persistence: include it in the current browser settings save/load/reset flow
  and in the `/process` request payload.

`direct_vr_encode` is independent from `keep_intermediates`:

- Direct encoding always omits the new job's `99_vr_frames` stage.
- `keep_intermediates=true` retains upstream left/right stages after success.
- `keep_intermediates=false` cleans upstream stages only after final encoding
  succeeds, using the existing cleanup policy.
- An encoding failure retains upstream stages regardless of the retention
  setting so that resume can retry the final encode.

Older saved settings do not contain the new field. Legacy-disk validation fills
in the default `false`, so the settings schema version does not change. The new
setting is an execution strategy and must not be added to any frame-stage
artifact identity or `_VR_SETTING_KEYS`: toggling it does not change the
content contract of an already assembled VR frame.

## Chosen Architecture

The pipeline orchestrator owns the strategy decision. Existing processing
components keep narrow responsibilities:

- `ProcessingOrchestrator` chooses the default or direct branch after optional
  upscaling.
- `VRFrameAssembler` chooses and validates the final eye source files. The
  source-selection logic remains the single authority for cropped versus
  upscaled inputs.
- `VideoEncoder` constructs and executes FFmpeg commands for already validated
  inputs. It does not inspect the whole processing directory to infer pipeline
  state.
- `create_output_directories` can omit named intermediate directories for a
  new job. It does not delete an omitted directory that already exists.

This keeps recovery policy out of the encoder and avoids a generalized input
type hierarchy for only two concrete encoding paths.

### Directory setup

Extend output-directory creation with an optional set of intermediate keys to
omit. Existing callers get the current behavior by default. Direct mode passes
`{"vr_frames"}`.

For an output root that does not already contain `99_vr_frames`, no such
directory is created and no `vr_frames` entry is added to the returned
directory map. For an output root that already contains the directory, the
helper leaves it untouched but still omits it from the active map. This meets
both requirements: new direct jobs avoid the stage, while selecting an
optimization never becomes an implicit destructive operation.

The existing successful cleanup scans `INTERMEDIATE_DIRS` independently of the
active map. Therefore `keep_intermediates=false` may remove a stale pre-existing
`99_vr_frames` after the new final video is safely complete, exactly as the
current cleanup contract permits.

### Source resolution

Expose the existing VR source-file resolution as the shared entry point for
both branches. It returns sorted left and right file lists or failure.

- If `upscale_model != "none"`, only `07_left_upscaled` and
  `07_right_upscaled` are valid.
- Otherwise, use `06_left_cropped` and `06_right_cropped`.
- Both lists must be non-empty, have equal length, match `total_frames` when it
  is positive, and have identical stems in the same order.

The direct encoder adds image2-specific validation without tightening the
legacy assembler's accepted filename behavior:

- Every name equals `frame_<index padded to at least six digits>.png`, as
  produced by `frame_%06d.png`; short unpadded forms such as `frame_1.png` are
  rejected.
- Both eyes have exactly the same names.
- Numeric indices are strictly consecutive.
- The first numeric index becomes FFmpeg's explicit `-start_number`.
- The source directories are stable and all files share the same image2
  pattern `frame_%06d.png`; indices longer than six digits remain valid because
  `%06d` is a minimum width.
- The first PNG header in each eye is readable.

Upstream completion manifests remain responsible for proving uniform shape and
complete payloads across the stage. Direct encoding reads only the first header
from each eye to choose the filter graph; it does not add another full IHDR
scan.

## Pipeline Data Flow

### Default branch

The disabled branch remains:

1. Resolve final eye sources.
2. Assemble and persist `99_vr_frames` with the current OpenCV implementation.
3. Reuse or complete the VR stage manifest as today.
4. Encode `99_vr_frames/frame_%06d.png` with `VideoEncoder.create_video`.

No new condition is inserted inside the frame workers, and no encoder command
or preview behavior changes on this branch.

### Direct branch

The enabled branch is:

1. Complete distortion, crop, and optional upscaling normally.
2. Ask `VRFrameAssembler` to resolve the final left/right source files.
3. Validate the canonical image2 sequence contract and read the first eye
   headers.
4. Mark Step 7 as intentionally skipped/deferred to FFmpeg in console output;
   do not call `assemble_vr_frames` or `complete_stage` for VR assembly.
5. Call a dedicated `VideoEncoder` stereo-sequence method with both file lists,
   the expected frame count, output directory, source video, settings, and the
   optional progress tracker.
6. Build one FFmpeg filter graph that resizes only when needed, combines the
   eyes, and feeds the existing selected encoder.
7. Atomically publish the completed final video.
8. Run the existing status update and successful-cleanup flow.

There is no direct-mode `metadata.json` for Step 7. The restart boundary moves
from assembled VR PNGs back to the final left/right stage.

## FFmpeg Command Contract

### Inputs

Resolve the output frame rate through the existing `_resolve_output_fps`
method. Pass the same rate and explicit start number to both image2 inputs:

```text
-framerate <fps> -start_number <n> -i <left>/frame_%06d.png
-framerate <fps> -start_number <n> -i <right>/frame_%06d.png
```

If audio preservation is enabled, append the current trimmed audio input after
both image inputs. Its index is therefore 2. Continue preferring
`original_audio.flac`, with the original video as the existing fallback.

### Filter graph

Let `W` and `H` be `per_eye_width` and `per_eye_height`.

If both first-eye headers already report `W x H`, omit scale filters:

```text
[0:v][1:v]hstack=inputs=2:shortest=1[vr]
```

or:

```text
[0:v][1:v]vstack=inputs=2:shortest=1[vr]
```

If either eye differs, normalize both inputs before stacking:

```text
[0:v]scale=W:H:flags=bicubic+accurate_rnd[left];
[1:v]scale=W:H:flags=bicubic+accurate_rnd[right];
[left][right]hstack=inputs=2:shortest=1[vr]
```

Scaling both eyes when only one header differs is intentional. It keeps both
filter inputs on the same normalization path instead of mixing one transformed
eye with one passthrough eye.

Use `vstack` for over-under. Exact target sizing intentionally matches the
current OpenCV behavior; aspect ratio is not preserved independently because
the current assembler also resizes to the requested width and height.

`shortest=1` prevents framesync from repeating the last frame if an input ends
unexpectedly. Add `-frames:v <validated count>` as an upper bound.

### Stream mapping and encoding

Always map the filtered video explicitly:

```text
-map [vr]
```

When preserving audio, map the third input's optional first audio stream:

```text
-map 2:a:0? -c:a aac -shortest
```

The optional map preserves current behavior for source videos without audio.
Do not rely on automatic stream selection now that the command has two video
inputs.

Call the existing `_build_encoder_cmd` with the direct temporary output path.
The current helper owns and preserves these exact arguments:

| Encoder path | Arguments preserved from `_build_encoder_cmd` |
| --- | --- |
| NVENC | `-c:v hevc_nvenc -pix_fmt yuv420p -preset p7 -tune hq` |
| Software | `-c:v libx264` or `-c:v libx265`, followed by `-pix_fmt yuv420p -crf 18 -preset medium` |

The helper also remains the sole authority for automatic NVENC detection,
explicit-NVENC fallback, and unknown-encoder fallback. The direct path does not
introduce bitrate, rate-control, GPU-selection, or profile flags that the
current helper does not produce.

The direct path replaces only these command regions:

- all image input arguments, because there are two image2 inputs with explicit
  start numbers;
- the filter graph;
- explicit video and optional audio stream mapping;
- progress and logging arguments;
- the validated video-frame limit; and
- the final argument passed to `_build_encoder_cmd`, which is the sibling
  temporary output path rather than the published output path.

### Progress

Run direct encoding with FFmpeg's machine-readable progress protocol:

```text
-progress pipe:1 -nostats -loglevel error
```

The process and reader contract is explicit:

```python
process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
    bufsize=1,
)
```

- The calling processing thread reads the one merged stream; no second pipe or
  additional reader thread is created.
- Read the stream line by line. Parse stripped lines beginning with `frame=` as
  progress records.
- Keep the last 50 non-progress lines in `collections.deque(maxlen=50)` for a
  bounded failure diagnostic.
- After the merged stream reaches EOF, call `process.wait()` and use that return
  code. EOF is never treated as evidence of process success. Because the only
  pipe has already been continuously drained, `wait()` cannot deadlock on a
  full child-output pipe.
- If stream reading raises, terminate the child, use `communicate()` to drain
  remaining merged output, and wait for process cleanup before returning
  failure. Use a bounded wait and kill fallback so the error path cannot leave
  FFmpeg running.

`poll()` alone is not a waiting primitive, and `communicate()` cannot provide
incremental progress when used as the sole reader. The line-reader plus final
`wait()` strategy supplies both progress and deterministic process completion.

For each parsed frame, call the existing tracker with phase `video_encoding`,
step name `Direct VR Encoding`, and the validated total. The existing
application-level progress throttle remains the final emission limit. A
tracker callback failure is logged and ignored because UI transport must not
invalidate a successfully encoded video.

No assembled VR preview is emitted. The UI keeps the most recent upstream
preview while the progress values continue to advance.

## Output Commit And Failure Semantics

The direct method encodes to a sibling temporary path that retains the `.mp4`
suffix, for example `.name.direct.tmp.mp4`. Retaining the suffix lets FFmpeg
select the correct muxer.

- Remove only that exact stale temporary path before starting.
- Never remove or truncate an existing final output before FFmpeg succeeds.
- On return code zero, require the temporary file to exist and be non-empty,
  then use `Path.replace` to publish it atomically.
- On nonzero return, launch exception, progress-reader failure, or missing/empty
  output, remove the temporary path and return failure.
- Include the bounded FFmpeg diagnostic tail in console output on failure.
- Leave all upstream frame stages and manifests untouched.

The orchestrator marks the job failed through its existing finalization path.
A resumed job revalidates and reuses the upstream stages, then starts final
encoding from frame one. Direct encoding itself has no partially reusable
artifact.

## Resume And Mode Transitions

| Situation | Required behavior |
| --- | --- |
| Resume a failed direct job | Load saved `direct_vr_encode=true`, reuse valid eye stages, restart final encoding |
| Resume an older job with no setting | Default to `false` and retain the legacy assembled-frame contract |
| Run direct mode where no VR directory exists | Do not create `99_vr_frames` |
| Run direct mode where a VR directory already exists | Ignore it and do not delete or modify it |
| Use normal mode after a direct run | Create and populate `99_vr_frames` normally |
| Use direct mode after a normal run | Ignore the existing VR stage; successful cleanup may remove its contents only when retention is disabled |
| Direct FFmpeg failure | Keep upstream stages and any previous valid final video |

The `/resume` endpoint continues to use settings saved by the interrupted job;
it does not gain a direct-mode override. Programmatic callers may choose a
different strategy for a fresh processing run, but stage reuse remains governed
by existing artifact identities rather than this execution-only flag.

## Error Handling

Fail before launching FFmpeg when:

- FFmpeg is unavailable.
- The selected source directories are missing or empty.
- Left and right counts differ.
- `total_frames` is positive and does not match the source count.
- Eye filenames differ or are not canonical numbered PNG names.
- Numeric indices contain a gap or have different starts.
- Either first PNG header is unreadable.
- Per-eye target dimensions are absent or non-positive.
- The VR format is neither `side_by_side` nor `over_under` after validated
  settings reach the encoder.

Fail after launch when FFmpeg exits nonzero, the progress reader fails, the
temporary output is missing or empty, or atomic publication fails. Every such
failure returns `false` to the orchestrator and leaves the job resumable from
its eye sequences.

## Testing Strategy

### Settings and web contract

- Assert the default is `false`.
- Assert explicit validation accepts only booleans.
- Assert legacy settings without the field normalize to `false`.
- Assert the process payload, browser persistence, reset defaults, and loaded
  settings include the checkbox.

### Directory behavior

- Assert default directory setup still creates every intermediate directory.
- Assert direct setup omits a new `99_vr_frames` and the `vr_frames` map entry.
- Assert omission does not delete a pre-existing `99_vr_frames`.
- Assert existing successful cleanup still handles all configured intermediate
  directories.

### Source resolution and sequence validation

- Assert cropped sources are selected without upscaling.
- Assert upscaled sources are mandatory when upscaling is enabled.
- Assert matching zero-based and one-based sequences produce the correct start
  number and count.
- Reject empty, count-mismatched, stem-mismatched, noncanonical, gapped, and
  unreadable sequences.
- Assert only the first header from each eye is needed for filter selection.

### Command construction

- Assert side-by-side uses `hstack` and over-under uses `vstack`.
- Assert exact-size sources omit scale filters.
- Assert mismatched dimensions scale both eyes with
  `bicubic+accurate_rnd` to the per-eye target.
- Assert both inputs use the same exact fractional frame rate and start number.
- Assert the validated count reaches `-frames:v`.
- Assert video maps from `[vr]` and audio maps from optional input 2.
- Assert no audio input or audio codec arguments appear when preservation is
  disabled.
- Assert current NVENC and software encoder argument lists are reused without
  quality changes.

### Orchestration and failures

- Assert the default branch still calls `assemble_vr_frames` followed by the
  current `create_video`.
- Assert direct mode does not call frame assembly and calls the dedicated stereo
  encoder with resolved files.
- Assert source-validation failure prevents FFmpeg launch.
- Assert progress records produce ordered tracker updates.
- Assert `Popen` merges standard error into the sole piped stream, waits for the
  child return code after draining it, and retains at most 50 diagnostic lines.
- Assert a progress-reader exception terminates and reaps the child process.
- Assert failed encoding removes only the direct temporary file, preserves an
  old final video, and does not trigger successful cleanup.
- Assert successful encoding atomically replaces the final path and retains the
  existing finalization behavior.

### Real FFmpeg integration

When FFmpeg is available, build tiny deterministic left/right PNG sequences and
exercise both layouts with the software encoder. Use ffprobe and decoded sample
frames to verify:

- output dimensions;
- frame rate and frame count;
- left/right or top/bottom placement;
- optional audio mapping;
- absence of all VR-stage PNGs and metadata.

Also run a deterministic, structured resize fixture through the legacy and
direct resize paths before video compression. Report PSNR and SSIM against the
OpenCV `INTER_CUBIC` result. Emit a non-blocking quality-review warning below
either of these calibrated floors:

- PSNR: 30 dB
- SSIM: 0.95

A warning requires explicit review in the implementation completion report but
does not fail CI by itself. The fixture must contain gradients, edges, texture,
and text-like detail; pure random noise is not representative visual content.

These floors are deliberately below locally measured accepted-path values.
With FFmpeg n8.0.1, structured slight downscaling measured 41.12 dB and 0.9978,
while a real UI-image half downscale measured 32.45 dB and 0.9766. A universal
40 dB floor would therefore flag accepted half-downscale behavior, and a random
noise sample measured only 22.91 dB and 0.9461 despite using the approved
filter. The enforceable cross-platform contract remains the scaler flags,
exact dimensions, and unchanged encoder arguments; the calibrated warnings add
an automated regression signal without pretending libswscale builds are
bit-stable.

### Repository verification

- Run the focused unit and integration tests first.
- Run the complete pytest suite.
- Run Black check, Flake8, and Mypy with the repository's existing commands.
- Inspect the settings control at desktop and mobile widths and verify that its
  label, tooltip, and neighboring controls do not overlap.

## Acceptance Criteria

1. A default-settings run follows the legacy path with no command, artifact,
   progress, or preview regression.
2. A new direct-mode run creates no `99_vr_frames` directory, PNG, or metadata.
3. FFmpeg reads validated final left/right image2 sequences and produces the
   requested SBS or OU dimensions.
4. Exact-size sources are not rescaled; mismatched sources use
   `bicubic+accurate_rnd`.
5. Encoder and audio quality settings match the legacy path.
6. Output frame rate and selected clip audio behavior match the current path.
7. The UI reports direct encoding progress without materializing preview
   frames.
8. A failed encode preserves upstream recovery artifacts, any previous valid
   final video, and no stale direct temporary output.
9. Resume honors the saved direct-mode value and restarts final encoding from
   the validated eye sequences.
10. All focused and repository-wide checks pass, and local FFmpeg smoke tests
    confirm both layouts.

## Alternatives Rejected

### Generalized encoder input specification

A common input hierarchy could model assembled and stereo sequences. It adds
types and dispatch for only two paths while obscuring the important pipeline
branch. A dedicated stereo method is smaller and keeps the legacy method stable.

### Encoder infers sources from processing directories

Passing the whole directory map to `VideoEncoder` would reduce orchestrator
code, but it would couple encoding to upscaler selection, stage recovery, and
directory conventions. Those decisions already belong to the assembler and
orchestrator.

### Replace the legacy path unconditionally

Always encoding from eye sequences would maximize disk savings but remove the
existing final-frame inspection and restart checkpoint for every user. The
request explicitly accepts that trade-off only as an optional aggressive mode.

### Preserve OpenCV bit identity through a Python pipe

Python could resize and stack every pair and pipe raw frames to FFmpeg. That
would avoid PNG writes but would not let FFmpeg read the eye sequences directly,
would retain the Python per-frame bottleneck, and would add pipe lifecycle and
backpressure complexity. The accepted contract does not require bit identity.
