# V2 Shot-Aware Temporal Inference Design

## Status

The user approved the design direction on 2026-08-15 after reviewing the
difference between memory batching, model-native temporal context, and generic
temporal post-processing. This document is the implementation gate. No code
change should start until the user reviews this written specification.

## Linus Gate

### Is this a real problem?

Yes. The Web UI submits `temporal_window_size` and
`temporal_window_overlap`, but the main file-backed pipeline never applies
those values to the inference that produces raw depth. `DepthMapProcessor`
first partitions the video into batches of at most 32 frames. The local V2
adapter only enters its own chunking branch for inputs over 60 frames, so that
branch is unreachable from the main pipeline.

This is not only a missing parameter. Upstream Video Depth Anything already
implements a fixed long-sequence algorithm inside `infer_video_depth`: 32-frame
windows, ten carried frames, affine alignment for relative depth, and overlap
interpolation. Calling that method independently for each outer batch resets
its state at every outer boundary. The current local fallback merely discards
duplicate overlap predictions and is not the upstream algorithm.

### Is there a simpler solution?

Yes. Fix V2 as V2. Do not build a universal temporal framework before a second
backend needs the same contract. Add one bounded V2 sequence path that matches
the upstream offline algorithm, and leave DA3 and See-Through on their current
batch path.

### What can this break?

V2 raw depth will intentionally change, so old V2 raw, canonical, stereo, and
downstream output must not be reused. Source frames and scene analysis remain
valid. DA3 and See-Through output, call patterns, and cache identities must not
change. Existing external callers of the V2 constructor must continue to load,
even though custom temporal window values are no longer presented as supported
tuning controls.

## Upstream Contract

The accepted behavior follows the official offline implementation rather than
inventing a project-specific approximation:

- `INFER_LEN = 32`
- `OVERLAP = 10`
- `KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]`
- `INTERP_LEN = 8`
- frame step `32 - 10 = 22`

The upstream source labels these inference settings as fixed. It replaces the
first ten inputs of every later window with two key frames and eight overlap
frames from the preceding input window. For relative depth it aligns the next
window to retained reference depths with one scale and shift, then interpolates
the eight duplicate overlap predictions. Metric depth skips scale and shift but
retains overlap interpolation.

References:

- [Official VDA offline implementation](https://github.com/DepthAnything/Video-Depth-Anything/blob/main/video_depth_anything/video_depth.py)
- [Video Depth Anything paper, long-sequence inference](https://arxiv.org/abs/2501.12375)

## Accepted Scope

This change repairs V2 native temporal inference in the restartable,
file-backed pipeline.

It does not add temporal behavior to DA3, DA3MONO, DA3METRIC, or See-Through.
An overlap loop around a deterministic frame-independent model only repeats the
same work. DA3's multi-view and streaming implementations have different state
and reference-view semantics and require a separate design.

The product contract becomes **temporal consistency within each detected
shot**, not one temporal state across unrelated cuts.

## Invariants

1. Every source frame produces exactly one raw depth map, in source order.
2. A V2 temporal window never crosses a candidate RGB scene cut.
3. V2 uses the fixed upstream 32/10/key-frame algorithm; window dimensions are
   not user tuning knobs.
4. Host and device working memory are bounded independently of video length.
5. `DepthBatch` representation and native raw storage contracts do not change.
6. A resumed V2 shot is either reused completely or recomputed completely.
7. DA3 and See-Through retain their existing memory-batch behavior.
8. One shot uses one input resolution and precision from start to finish.
9. OOM handling never silently changes resolution for only one window.

## Ownership

`DepthMapProcessor` continues to own:

- source frame files and candidate scene boundaries,
- disk reads,
- raw-store creation and writes,
- resume decisions,
- progress reporting,
- the global raw-depth completion barrier.

`VideoDepthEstimator` owns:

- VDA input transformation,
- temporal window composition,
- the fixed-window model forward call,
- key-frame state,
- relative scale/shift alignment,
- overlap interpolation,
- release of finalized ordered depth batches.

The processor must not reconstruct VDA key-frame rules. The estimator must not
write project stage files. This leaves one owner for each kind of state.

## Narrow Sequence Interface

Only V2 gains a sequence iterator. Its logical contract is:

```python
def iter_sequence_depth(
    self,
    frame_count: int,
    load_frames: Callable[[Sequence[int]], np.ndarray],
    *,
    target_fps: int,
    input_size: int,
    fp32: bool,
) -> Iterator[tuple[int, DepthBatch]]:
    ...
```

Indexes passed to `load_frames` are local to the current shot. Each yielded
integer is the local start index of a contiguous finalized `DepthBatch`.
Yields must be ordered, non-overlapping, gap-free, and total exactly
`frame_count` values.

`DepthMapProcessor` uses this optional method when present and retains the
existing `estimate_depth_batch` loop otherwise. Do not add a strategy class,
backend inheritance hierarchy, capability Boolean, or generic temporal config
object in this change. The method itself is the single capability signal.

The existing in-memory `estimate_depth_batch` API remains supported. It calls
upstream `infer_video_depth` once for the complete array supplied by its caller.
The local `num_frames > 60` branch, discard-only overlap helper, and nested
chunking path are deleted rather than retained as a second temporal algorithm.

## Shot Partitioning

The RGB scene pass already runs before raw inference. V2 sequence calls are
partitioned by `candidate_cuts`, not `final_cuts` and not the current contents
of `scene_ids`.

This distinction matters during resume: a persisted manifest may already have
`status: final`, whose `scene_ids` can merge candidate segments after inspecting
depth distributions. `candidate_cuts` remains the stable pre-inference temporal
reset contract. Final canonical merging does not retroactively join V2 temporal
state across a cut.

When scene detection is disabled, `candidate_cuts` is empty and the source is
one temporal shot. A hard cut resets key frames, overlap buffers, affine
references, and pending output.

## Bounded V2 Data Flow

For each shot:

1. The first window loads source indexes 0 through 31, repeating the final
   source frame when the shot is shorter than 32 frames.
2. The estimator transforms those frames and performs one direct 32-frame VDA
   forward call.
3. Every later upstream window begins at offsets 22, 44, 66, and so on while
   the offset is less than the shot length.
4. Positions 0 and 1 use prior input positions 0 and 12. Positions 2 through 9
   use prior input positions 24 through 31. Positions 10 through 31 load the
   next 22 source frames, padding with the final source frame at the tail.
5. For relative V2, upstream `compute_scale_and_shift` aligns the two retained
   reference predictions. The same affine transform is applied to the new
   overlap and non-overlap predictions, and negative aligned values are clamped
   to zero as upstream does.
6. Upstream `get_interpolate_frames` combines the previous and current eight
   overlap predictions.
7. Frames that no future window can modify are yielded immediately. Only the
   ten selected input tensors, two alignment references, eight pending depth
   maps, and the current window remain live.
8. Tail padding predictions are discarded so the yielded count equals the
   original shot length.

The implementation must factor one `_infer_fixed_window` path around the VDA
model's direct `forward` operation. It must not call `infer_video_depth` once per
32-frame outer window, because that method performs its own segmentation and
would create nested temporal windows again.

The existing outer V2 batch size is not a valid GPU-memory control. Upstream
`infer_video_depth` pads short calls to a 32-frame model input, so sending four
source frames does not produce a four-frame VDA forward pass. The repaired path
makes that real behavior explicit.

## Resume Semantics

Candidate shots are the atomic V2 resume unit.

- If every raw payload for a shot exists and validates, reuse the complete shot.
- If any payload in a shot is missing or invalid, discard all raw payloads for
  that shot and recompute it from its first frame.
- Completed earlier shots remain reusable because temporal state resets at each
  candidate cut.
- A failure never permits canonical generation; the existing global raw-depth
  barrier remains mandatory.

Persisting GPU attention tensors or temporal checkpoints is deliberately out of
scope. Recomputing one incomplete shot is simpler and cannot restore subtly
incompatible state.

## Cache And Fingerprint Contract

V2 reports this new identity field:

```text
inference_algorithm = vda-offline-shot-v1
```

`inference_algorithm` becomes a classified model-identity field. Its presence
invalidates old V2 raw depth and every downstream product while preserving
source frames and valid scene analysis.

Because candidate cuts now affect V2 raw predictions, the V2 raw fingerprint
also includes:

- `scene_detection`,
- `scene_cut_threshold`,
- `min_scene_frames`,
- the RGB scene-analysis algorithm version.

These values are added only to the V2 raw contract. They must not invalidate
DA3 or See-Through raw caches because those backends do not use candidate cuts
during inference.

## Settings And UI Compatibility

The Web UI removes the experimental window-size and overlap controls and stops
claiming that larger arbitrary windows improve consistency. Active README and
performance documentation describe fixed VDA windows and shot-aware resets.
Archived historical documents are not rewritten.

For one compatibility release:

- keep `temporal_window_size` and `temporal_window_overlap` in validated saved
  settings with their existing defaults,
- keep their current fingerprint participation so unaffected backend caches do
  not churn,
- keep the existing `temporal_window_overlap` arguments on V2 and
  `StereoProjector` constructors callable,
- ignore non-default compatibility values for the fixed upstream path and emit
  one clear warning per processing invocation when either value differs from
  32/10.

The compatibility keys are not shown in the Web UI and are not described as
effective. Removing them from the settings schema is a later migration, not
part of this change.

## Error Handling

- Missing or unreadable source frames fail the current shot with the source
  path in the error.
- A model result with the wrong frame count, shape, dtype, or representation
  fails before it is committed as a complete shot.
- CUDA OOM aborts with a message recommending a smaller explicit depth
  resolution or smaller V2 checkpoint.
- The current per-window reduced-resolution retry is removed from the sequence
  path. Mixing resolutions inside a shot is not a valid temporal result and is
  not represented in the cache identity.
- Partial shot payloads remain detectably incomplete and are discarded on the
  next resume.

## Rejected Alternatives

### Pass the two Web settings into the existing estimator

Rejected. It would make the controls appear live while preserving the outer
state reset and the incorrect discard-only overlap behavior. It also conflicts
with upstream's fixed inference constants.

### Wrap every estimator in the V2 overlap loop

Rejected. Memory batching is generic; temporal context is not. See-Through and
monocular DA3 checkpoints do not gain temporal information by seeing the same
frame twice. DA3 main-series streaming has its own state contract.

### Load an entire shot and call upstream `infer_video_depth` once

Rejected. It is algorithmically simple but violates the file-backed bounded
memory contract for long takes. A multi-minute 1080p shot can consume tens of
gigabytes of host memory before inference begins.

### Use experimental VDA streaming as an automatic fallback

Rejected for this change. Upstream documents a quality drop between streaming
and offline inference. Choosing it silently would change the output contract.
It may be offered later as a separate explicit backend mode.

## Verification

- Add window-index tests at lengths 1, 22, 23, 24, 25, 31, 32, 33, 53, 54,
  55, and a multi-window long sequence.
- Compare the bounded sequence result with upstream `infer_video_depth` on the
  same synthetic shot using a deterministic fake forward pass. Frame order,
  scale/shift alignment, overlap interpolation, padding, and crop length must
  match.
- Test relative and metric branches separately.
- Prove that a loader call requests at most 32 source frames for the first
  window and at most 22 new source frames for later windows.
- Test that candidate cuts reset all temporal state and no loader request spans
  two shots.
- Test complete-shot reuse and partial-shot discard/recompute.
- Test that changing scene settings invalidates V2 raw depth but does not alter
  DA3 or See-Through raw fingerprints.
- Test that the new V2 algorithm identity preserves source and scene stages but
  invalidates raw depth and all downstream stages.
- Retain the existing framewise processor test and assert unchanged DA3 and
  See-Through call behavior.
- Test that Web requests no longer submit temporal tuning fields and that active
  user-facing text says consistency is within detected shots.
- Run the focused estimator, depth processor, storage, settings, cache, resume,
  and Web tests, followed by the complete unit-test suite.

## Acceptance Criteria

The change is accepted only when all of the following are true:

1. A V2 shot longer than 32 frames is numerically equivalent to the official
   offline sequence algorithm for the same frames, settings, and precision.
2. No V2 temporal window crosses a candidate cut.
3. Peak sequence working memory does not grow with shot length.
4. Every frame is stored once and the global raw-depth barrier remains intact.
5. Interrupted work never reuses a partial V2 shot.
6. Old V2 raw output cannot survive the algorithm identity change.
7. DA3 and See-Through output and cache behavior remain unchanged.
8. The UI exposes no temporal setting that the estimator ignores.

## Non-Goals

- DA3-Streaming integration.
- A generic optical-flow or confidence-based temporal stabilizer.
- Temporal consistency for frame-independent estimators.
- Cross-cut affine scale continuity.
- User-configurable VDA window, overlap, key-frame, or interpolation sizes.
- Persisted temporal-state checkpoints inside a shot.
- A new estimator class hierarchy or generalized strategy framework.
- Changing scene detection, scene merging, canonical disparity, stereo
  rendering, or video encoding algorithms.
