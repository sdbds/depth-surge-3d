# V2 Shot-Aware Temporal Inference Design

## Status

The user approved the design direction on 2026-08-15 after reviewing the
difference between memory batching, model-native temporal context, and generic
temporal post-processing. This document is the implementation gate. No code
change should start until the user reviews this written specification.

The specification was revised after external review to make the current
settings flow, cache identities, runtime resolution plan, memory bound, loader
contract, and the separate post-inference stabilization problem explicit.

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

## Current Data Flow And Cache Reality

The two temporal settings are not lost at the HTTP boundary. Their current path
is:

```text
Web controls
  -> request JSON
  -> validated processing settings
  -> saved processing settings
  -> DepthMapProcessor.generate_depth_map_files(settings)
```

The path stops there for inference. Web and CLI construction call
`create_stereo_projector` with model path, device, metric mode, and backend, but
not with either temporal setting. `StereoProjector` therefore constructs V2
with its default overlap of 10, and V2 has no constructor parameter for window
size. `DepthMapProcessor._infer_raw_chunk` passes frames, FPS, input size, and
precision to `estimate_depth_batch`; it does not pass temporal settings.

Today, setting `temporal_window_size=64` therefore has this exact effect:

1. The value is validated, saved, and delivered to `DepthMapProcessor`.
2. It changes local raw-depth and global canonical-cache identities.
3. It does not change outer batching, V2 construction, or model inference.
4. The user pays a cache miss for output that is semantically unchanged.

Both values currently participate in two identity layers for every backend:

| Identity layer | Current location | Current participation |
| --- | --- | --- |
| Local raw depth | `DEPTH_MODEL_SETTING_KEYS` in `depth_storage.py` | Stored under `02_depth_raw/metadata.json -> semantic_fingerprint.depth_settings` |
| Global canonical cache | `DEPTH_CACHE_SETTING_KEYS` in `depth_cache.py` | Hashed directly and also indirectly through `model_fingerprint` |
| V2 inference | `VideoDepthEstimator` | Window size unused; overlap remains constructor default 10 in the main path |
| DA3 / See-Through inference | Their adapters | Both values unused |

This revision deliberately preserves the two existing cache fields for one
compatibility release. It does not claim they affect output. Removing them now
would change every backend's cache key, including unaffected DA3 and
See-Through entries. Their eventual removal requires an explicit settings and
cache migration, outside this change.

The existing `_retry_chunk_with_reduced_resolution` is not graceful fallback in
the main file-backed path. It is reachable only from the adapter's internal
chunk branch for inputs over 60 frames, while the processor supplies at most 32.
The uniform execution plan below replaces a dead path with an actual bounded
fallback contract.

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
An inference-time overlap loop around a deterministic frame-independent model
only repeats the same work. DA3's multi-view and streaming implementations have
different state and reference-view semantics and require a separate design.

This statement applies only to model-native inference context. Generic
post-inference stabilization is a separate and real product requirement. It
would consume already predicted depth or canonical disparity, motion evidence,
and occlusion confidence to reduce frame-to-frame jitter for any backend. It is
not implemented or rejected by this V2 repair; it is separated because it has
different data, cache, quality, and failure contracts.

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
8. One V2 raw stage uses one effective input resolution and precision from
   start to finish.
9. An effective size is fixed before any payload under that execution plan is
   committed. A later plan change invalidates prior V2 raw payloads before new
   writes, is visible to the user, and changes the raw-depth identity.

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

The callback contract is strict:

- requested indexes are valid, unique, ascending local shot indexes;
- the estimator requests real source indexes only and performs tail repetition
  itself;
- the callback returns `uint8` BGR in shape `[len(indexes), H, W, 3]` and in the
  exact requested order;
- every returned frame has the same height and width;
- the processor raises `OSError` naming the unreadable path when a decode
  fails;
- the estimator validates returned count, dtype, rank, channel count, and
  common geometry before transformation, then propagates callback exceptions.

Pixel content cannot prove semantic ordering. The processor guarantees order by
constructing the callback directly from the requested index list; tests verify
that mapping. The estimator does not reread files or duplicate processor I/O.

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
5. One `_finalize_window` routine handles both representations. For relative
   V2 it uses current positions 0 and 1 against the two retained references to
   compute scale and shift, applies that transform to positions 2 through 31,
   and clamps negative aligned values to zero. Metric V2 uses identity scale
   and shift.
6. The same routine calls upstream `get_interpolate_frames` once for the
   previous pending tail and current positions 2 through 9, then appends current
   positions 10 through 31. It also updates the retained alignment reference.
   Current positions 0 and 1 are key frames, so a helper that interpolates
   `current[:8]` would be incorrect.
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

## Tail Examples

All examples use zero-based shot-local indexes and reproduce upstream's loop
condition `window_start < shot_length`.

### 25-frame shot

- Window starts are 0 and 22.
- The first input is frames 0 through 24 followed by seven copies of frame 24.
- The second input carries `[0, 12, 24, 24, 24, 24, 24, 24, 24, 24]` in its
  first ten positions and uses frame 24 for all padded new positions.
- Only source frame 24 lies in the visible interpolated tail. Predictions are
  cropped to exactly frames 0 through 24.

### 31-frame shot

- Window starts are 0 and 22.
- The first input is frames 0 through 30 followed by one copy of frame 30.
- The second input carries `[0, 12, 24, 25, 26, 27, 28, 29, 30, 30]`; its new
  positions are padded with frame 30.
- Visible overlap frames 24 through 30 are interpolated, and padded frame 31 is
  discarded.

### 33-frame shot

- Window starts are 0 and 22.
- The first input is frames 0 through 31.
- The second input carries `[0, 12, 24, 25, 26, 27, 28, 29, 30, 31]`, places
  source frame 32 at position 10, and repeats frame 32 for the remaining tail.
- Frames 24 through 31 are interpolated, frame 32 comes from the second window,
  and every later padded prediction is discarded.

## Explicit Working-Memory Bound

Let source geometry be `Hs x Ws`, transformed model geometry be `Hi x Wi`, and
returned depth geometry be `Hd x Wd`. Excluding model weights, allocator
bookkeeping, and the framework workspace for one model forward, the sequence
orchestrator must not retain more than:

```text
host decoded RGB       <= 32 * Hs * Ws * 3 bytes
host transformed input <= 32 * 3 * Hi * Wi * 4 bytes
host depth state       <= 50 * Hd * Wd * 4 bytes
device input/state     <= 42 * 3 * Hi * Wi * 4 bytes
device depth output    <= 32 * Hd * Wd * 4 bytes
```

The conservative 50-map host depth term covers the current 32 predictions,
eight previous pending maps, two alignment references, and eight newly blended
maps at the interpolation peak. The 42-frame device input term covers one
current window plus ten retained key-frame tensors before old state is released.
Later source loads request at most 22 new decoded frames, but the first-window
32-frame bound is used for the invariant.

Model activations and CUDA workspace are additionally bounded by exactly one
32-frame `_infer_fixed_window` call at the chosen effective resolution. Tests
must measure the live orchestration objects, and an optional CUDA integration
test records allocator peak; neither bound may scale with shot or video length.

## V2 Resolution Execution Plan

The V2 raw stage distinguishes requested resolution from effective resolution:

```text
requested_input_size: value derived from depth_resolution
effective_input_size: one value used by every V2 window in this raw stage
precision: fp16 or fp32 inference mode
fallback_policy: v2-uniform-halving-v1
```

This execution plan is resolved before the first raw payload is written and is
included in the raw semantic fingerprint. The same effective size applies to
all shots so one raw directory never mixes inference resolutions.

Resolution selection proceeds as follows:

1. A global cache or complete raw stage whose full identity matches the
   requested-size plan may be reused without a capacity probe.
2. Complete or partial local raw metadata may provide a previously negotiated
   plan. It is adopted only when its base source, model, requested settings, and
   fallback-policy identity match; complete payloads are then reusable and an
   incomplete stage resumes at that exact effective size.
3. Otherwise, the first real fixed V2 window is run at the requested input size
   as a capacity probe. A successful result and its temporal state are reused;
   the probe is not repeated work.
4. On CUDA OOM, all tensors from the failed probe are released and CUDA cache is
   cleared. When `current_size > 384`, the next candidate is
   `max(384, current_size // 2)`; at 384 or below there is no fallback candidate.
5. Candidates are tried until one succeeds or the final candidate fails. The
   successful value becomes the effective input size for the entire raw stage.
6. When fallback changes the effective size, the semantic fingerprint and
   global-cache key are rebuilt before any raw payload is written. The fallback
   identity gets one cache lookup before inference continues.

If a later window raises CUDA OOM, the processor clears CUDA state and retries
that window once at the same effective size. A second failure invalidates all
V2 raw payloads for the current execution plan, selects the next smaller
candidate, rebuilds the fingerprint, and restarts the raw stage from frame zero.
It never continues with mixed resolutions. If no smaller candidate exists, the
stage fails with the requested and effective sizes in the error.

Any fallback is non-blocking but visible. CLI writes one warning to stderr; Web
emits one warning through the existing progress-message channel. Both state the
requested size, selected effective size, and that the selection applies to the
whole V2 raw stage.

## Resume Semantics

Candidate shots are the atomic V2 resume unit.

- If every raw payload for a shot exists and validates, reuse the complete shot.
- If any payload in a shot is missing or invalid, discard all raw payloads for
  that shot and recompute it from its first frame.
- Completed earlier shots remain reusable because temporal state resets at each
  candidate cut, unless a changed effective resolution creates a new execution
  plan for the entire raw stage.
- A failure never permits canonical generation; the existing global raw-depth
  barrier remains mandatory.

For `vda-offline-shot-v1`, resume validation requires a structurally valid
`execution_plan`. It compares requested size, precision, and fallback-policy
version with the current request; the persisted effective size is accepted as
the already negotiated runtime value. A missing, malformed, or incompatible
plan invalidates V2 raw depth. DA3 and See-Through metadata require no plan.

Persisting GPU attention tensors or temporal checkpoints is deliberately out of
scope. Recomputing one incomplete shot is simpler and cannot restore subtly
incompatible state.

## Cache And Fingerprint Contract

V2 reports this new identity field:

```text
inference_algorithm = vda-offline-shot-v1
```

`inference_algorithm` becomes a classified model-identity field. It is stored
at this exact local metadata path:

```text
02_depth_raw/metadata.json
  -> semantic_fingerprint
  -> model_info
  -> inference_algorithm
```

`RawDepthStore` includes the complete `semantic_fingerprint` in its own storage
fingerprint. Canonical metadata stores both `source_raw_fingerprint` and the
hash of the raw semantic fingerprint as `source_model_fingerprint`. The global
canonical cache requires that same hash. The algorithm field therefore
invalidates old V2 raw depth and every downstream product without changing
`DepthBatch` or individual `.npz` payload formats. Source frames and valid scene
analysis remain reusable.

The selected V2 execution plan is stored beside the model identity at:

```text
02_depth_raw/metadata.json
  -> semantic_fingerprint
  -> execution_plan
```

Its effective input size and fallback-policy version are consequently covered
by local raw validation, canonical metadata, resume validation, and the global
cache's `model_fingerprint`.

Because candidate cuts now affect V2 raw predictions, the V2 raw fingerprint
also includes:

- `scene_detection`,
- `scene_cut_threshold`,
- `min_scene_frames`,
- the RGB scene-analysis algorithm version.

These values are added only to the V2 raw contract. Raw identity selection is
therefore explicitly backend-dependent:

| Identity field group | V2 | DA3 | See-Through |
| --- | --- | --- | --- |
| Existing model, resolution, dtype, and preprocessing identity | Yes | Yes | Yes |
| Compatibility `temporal_window_*` fields | Yes, unchanged for one release | Yes, unchanged for one release | Yes, unchanged for one release |
| Candidate-scene settings and scene algorithm version | Yes | No | No |
| `vda-offline-shot-v1` algorithm identity | Yes | No | No |
| V2 requested/effective resolution execution plan | Yes | No | No |

The backend-dependent selector is tested directly. Adding a field to V2 must
not be implemented by extending the common key tuple, because that would
invalidate DA3 and See-Through caches for settings that do not affect their raw
output.

This matrix governs raw model identity only. The global canonical cache keeps
scene settings for every backend because scene bounds change canonical
disparity even when raw estimator output is frame-independent.

## Settings And UI Compatibility

The Web UI removes the experimental window-size and overlap controls and stops
claiming that larger arbitrary windows improve consistency. Active README and
performance documentation describe fixed VDA windows and shot-aware resets.
Archived historical documents are not rewritten.

For one compatibility release:

- keep `temporal_window_size` and `temporal_window_overlap` in validated saved
  settings with their existing defaults,
- keep them in both `DEPTH_MODEL_SETTING_KEYS` and
  `DEPTH_CACHE_SETTING_KEYS`, so local and global cache keys do not churn,
- keep the existing `temporal_window_overlap` arguments on V2 and
  `StereoProjector` constructors callable,
- ignore non-default compatibility values for the fixed upstream path and emit
  one clear warning per processing invocation when either value differs from
  32/10.

The compatibility keys are not shown in the Web UI and are not described as
effective. Removing them from the settings schema is a later migration, not
part of this change.

The compatibility warning does not block processing. CLI writes it once to
stderr. Web sends it once through the existing progress-message channel so it
is visible in the active job. The message includes both supplied values and
states that VDA uses fixed window 32 and overlap 10. Default 32/10 values emit
no warning.

## Error Handling

- Missing or unreadable source frames fail the current shot with the source
  path in the error.
- A loader count, dtype, rank, channel, or geometry violation fails before the
  fixed-window forward call.
- A model result with the wrong frame count, shape, dtype, or representation
  fails before it is committed as a complete shot.
- CUDA OOM follows the uniform execution-plan negotiation and restart rules;
  no completed raw directory can contain mixed effective resolutions.
- Partial shot payloads remain detectably incomplete and are discarded on the
  next resume.

## Separate Future Work: Output Stabilization

Frame-to-frame depth jitter from DA3MONO, DA3METRIC, and See-Through remains a
real product problem after this V2 repair. The correct generic boundary is
after every backend has been converted to canonical relative disparity and
before stereo rendering:

```text
raw model depth
  -> canonical relative disparity
  -> optional shot-aware temporal stabilizer
  -> stereo rendering
```

A separate specification must define motion estimation, occlusion masks,
confidence fallback when a backend provides no confidence, edge preservation,
cut resets, cache identity, resume behavior, and measurable temporal-quality
acceptance criteria. That stage may also be offered for V2, but it must be
independently switchable so native V2 consistency is not confused with generic
filtering.

This future stage is not an overlap rerun and does not call the depth estimator
twice. Keeping it out of the V2 inference repair avoids changing every model's
canonical or stereo output while fixing one confirmed V2 data-flow defect.

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
- Test one shared `_finalize_window` path: metric uses identity alignment,
  relative aligns positions 2 through 31, and both interpolate positions 2
  through 9 rather than `current[:8]`.
- Prove that a loader call requests at most 32 source frames for the first
  window and at most 22 new source frames for later windows.
- Test that the processor callback maps indexes in requested order. Separately
  test estimator rejection of short count, wrong dtype, rank, channel count,
  geometry, decode failure, and callback exception propagation.
- Assert the 25-, 31-, and 33-frame worked examples exactly.
- Instrument retained arrays and tensors to enforce the documented 32/50/42
  orchestration bounds independently of sequence length.
- Test that candidate cuts reset all temporal state and no loader request spans
  two shots.
- Test complete-shot reuse and partial-shot discard/recompute.
- Test requested-resolution success, first-window uniform fallback, persisted
  plan resume, late-OOM same-size retry, whole-stage lower-resolution restart,
  final-candidate failure, and visible CLI/Web warnings.
- Test that changing scene settings invalidates V2 raw depth but does not alter
  DA3 or See-Through raw fingerprints.
- Test that the new V2 algorithm identity preserves source and scene stages but
  invalidates raw depth and all downstream stages.
- Inspect `02_depth_raw/metadata.json` to prove algorithm and execution-plan
  fields are inside the persisted semantic fingerprint and covered by its hash.
- Test the backend identity matrix so adding V2-only fields cannot invalidate
  DA3 or See-Through raw caches.
- Test current and compatibility behavior for
  `temporal_window_size=64`: it remains in both cache identities, does not alter
  V2 window composition, and produces one non-blocking warning.
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
4. The requested and effective input sizes are persisted, and no raw stage
   mixes effective resolutions.
5. Every frame is stored once and the global raw-depth barrier remains intact.
6. Interrupted work never reuses a partial V2 shot.
7. Old V2 raw output cannot survive the algorithm identity change.
8. DA3 and See-Through output and cache behavior remain unchanged.
9. The UI exposes no temporal setting that the estimator ignores.

## Non-Goals

- DA3-Streaming integration.
- Implementing the separately specified generic post-inference stabilizer in
  this V2 change.
- Adding model-native temporal inference to frame-independent estimators.
- Cross-cut affine scale continuity.
- User-configurable VDA window, overlap, key-frame, or interpolation sizes.
- Persisted temporal-state checkpoints inside a shot.
- A new estimator class hierarchy or generalized strategy framework.
- Changing scene detection, scene merging, canonical disparity, stereo
  rendering, or video encoding algorithms.
