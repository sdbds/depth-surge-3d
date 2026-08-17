# Optional VDPP Temporal Post-Processor Design

## Status

The user selected VDPP on 2026-08-16 as the first generic temporal depth
post-processor and explicitly rejected an optical-flow/RAFT path. This document
is the implementation gate. It specifies one optional, experimental VDPP stage;
it does not authorize a generic plug-in framework or a second stabilizer.

The integration is deliberately narrower than the upstream demo. It preserves
this project's canonical disparity contract, shot boundaries, bounded-memory
pipeline, cache identities, and resume behavior instead of loading and
normalizing a complete video in memory.

## Linus Gate

### Is this a real problem?

Yes. DA3MONO, DA3METRIC, and See-Through estimate frames independently. Their
spatial depth can be useful while their frame-to-frame scale and local geometry
jitter in video. The V2 repair gives V2 model-native temporal context, but it
does not solve output jitter for frame-independent estimators.

VDPP is relevant because it consumes depth only. It can refine the common
canonical disparity representation after any current backend without rerunning
the depth estimator or coupling the stage to RGB motion estimation.

### Is there a simpler solution?

Yes. Add exactly one optional stage at the existing canonical-to-render boundary:

```text
03_disparity_maps
  -> temporal_postprocessor=off  -> stereo rendering
  -> temporal_postprocessor=vdpp -> 03_disparity_stabilized -> stereo rendering
```

Do not put VDPP inside every estimator. Do not add RAFT, optical flow, RGB
warping, a stabilizer registry, strength sliders, or configurable temporal
windows. The first release has one setting with two values: `off` and `vdpp`.

### What can this break?

The dangerous boundaries are not the model call itself. They are:

- accidentally changing raw-depth or global canonical cache identities;
- silently applying per-frame min/max normalization and destroying scene scale;
- crossing a scene cut with temporal attention;
- accepting a derived disparity directory as canonical without validating its
  producer and source lineage;
- mixing old and new post-processor output during resume;
- keeping the depth estimator and VDPP resident on the GPU simultaneously;
- copying the upstream demo's whole-video tensor/list and making memory grow
  with video length;
- claiming general model or anime-domain quality before measuring it.

This specification makes each of those cases explicit.

## Decision Summary

The user-facing setting is:

```text
temporal_postprocessor = "off" | "vdpp"
```

Its default is `off`. Old jobs whose settings file predates the field are read
as `off`. Enabling it is an explicit request and never happens automatically.

VDPP is exposed after every current depth backend because all of them converge
to the same canonical relative-disparity contract. This includes DA3MONO,
DA3METRIC, See-Through, and V2. The intended first beneficiaries are the
frame-independent backends. V2 plus VDPP remains allowed for A/B testing, but it
is not recommended or selected automatically because V2 already has native
temporal inference.

The first release is marked **Experimental**. VDPP generation is CUDA-only,
uses the released VDPP v1.0 checkpoint, runs in FP32 with the upstream
half-resolution refinement path, and has no user-tunable inference parameters.

## Upstream Contract And Immutable Identity

The implementation is pinned to the source and weight release that belong
together:

| Field | Required value |
| --- | --- |
| Repository | `https://github.com/injun-baek/VDPP` |
| Release tag | `v1.0` |
| Source revision | `73cc2b4dc6b3b5cfb2e37f51e452461e03fe26f5` |
| Checkpoint URL | `https://github.com/injun-baek/VDPP/releases/download/v1.0/vdpp.pth` |
| Checkpoint size | `116485370` bytes |
| Checkpoint SHA-256 | `7368315b126093f0335147f42a1920f255d529613bfffc5c6cf4ef832deb73a7` |
| License | Apache-2.0 |
| Architecture | VDPP, DINOv2-small encoder, temporal DPT/VDA-small head |
| Released model size | 111M parameters, as reported by the upstream README |

The constructor configuration is also fixed: `encoder="vits"`, `features=64`,
`out_channels=[48, 96, 192, 384]`, `use_bn=false`,
`use_clstoken=false`, `num_frames=32`, `max_depth=1.0`, and `pe="ape"`.
Changing any value requires a new algorithm/model identity even if a checkpoint
happens to load.

Release verification found one upstream artifact discrepancy: the v1.0
checkpoint contains zero-valued `shift_head.0.weight` and
`shift_head.0.bias` tensors, while the pinned public model class does not
declare or call `shift_head`. The integration registers an inert 1x1
single-channel compatibility layer before `load_state_dict(strict=True)` so
the released checkpoint is still loaded exactly. The layer is never used by
forward inference. This policy is persisted as
`checkpoint_compatibility="released-zero-shift-head-v1"`; changing or removing
it invalidates stabilized caches.

Primary references:

- [VDPP v1.0 source](https://github.com/injun-baek/VDPP/tree/73cc2b4dc6b3b5cfb2e37f51e452461e03fe26f5)
- [VDPP v1.0 release](https://github.com/injun-baek/VDPP/releases/tag/v1.0)
- [CVPR 2026 Workshop paper](https://openaccess.thecvf.com/content/CVPR2026W/ECV/papers/Yoon_VDPP_Video_Depth_Post-Processing_for_Speed_and_Scalability_CVPRW_2026_paper.pdf)
- [CVPR 2026 supplementary material](https://openaccess.thecvf.com/content/CVPR2026W/ECV/supplemental/Yoon_VDPP_Video_Depth_CVPRW_2026_supplemental.pdf)

The paper and released code have one important difference that must not be
blurred. The supplementary material says training used non-overlapping
16-frame sequences and 224x224 random crops. The released model class and
`infer_video_depth` default to 32 inference frames with four overlap frames.
This integration follows the released v1.0 inference code, not an inferred
16-frame product variant:

```text
WINDOW = 32
OVERLAP = 4
STRIDE = 28
BATCH = 1
DOWNSIZE = true
PRECISION = fp32
```

`DOWNSIZE=true` is intentional. The paper trains and evaluates the efficient
path at a 0.5 spatial ratio and identifies downsampling as necessary for its
memory result. The upstream CLI happens to default its `--downsize` flag to
false, while the model method defaults to true. The product pins true and puts
that choice in the stage fingerprint. No upstream FPS or 5.5 GB memory number
is treated as a guarantee for this project, its resolutions, or its PyTorch
build.

## Current Pipeline Boundary

The current file-backed depth path is:

```text
00_original_frames
  -> 01_scene_data
  -> 02_depth_raw
  -> global raw completion barrier
  -> 03_disparity_maps
  -> 04_left_frames / 04_right_frames
```

`03_disparity_maps` is the correct input boundary because it removes backend
differences before post-processing:

- every payload is a uint16 PNG;
- decoded values are float32 in `[0, 1]`;
- `0.0` means far and `1.0` means near;
- each finalized scene uses immutable 2nd/98th-percentile bounds;
- metric depth, inverse depth, and relative depth have already been converted
  to one near-increasing representation;
- every frame has the same native shape and ordered name manifest.

VDPP therefore does not need an estimator backend name, a metric/relative
branch, RGB frames, or raw model confidence. Its identity is derived from the
selected canonical stage, not from string heuristics such as `"v2"` or
`"see_through"`.

The new physical directory is:

```text
03_disparity_stabilized
```

Its directory key is `disparity_stabilized`. Existing directories are not
renumbered. When the setting is off, stereo receives `03_disparity_maps`
exactly as it does today. When the setting is `vdpp`, stereo receives the
validated files from `03_disparity_stabilized`.

The directory is added to the central intermediate-directory registry so
cleanup, disk reporting, resume migration, and `keep_intermediates=false`
handle it deliberately. It is never discovered by a loose `03_*` glob.
`create_output_directories` adds `disparity_stabilized` to its omitted set when
the mode is off, so a new default-off job does not gain an otherwise unused
empty directory. Omission never deletes a dormant directory from an older
VDPP run.

## Invariants

1. The default-off generated media/intermediate payloads and cache identities
   are behaviorally and byte-for-byte unchanged. The saved settings manifest
   may add the explicit default field.
2. Every canonical input frame produces exactly one stabilized output frame in
   the same order, name, and native shape.
3. A VDPP window never crosses a finalized scene boundary.
4. VDPP receives canonical relative disparity directly. There is no per-frame
   min/max normalization, polarity inversion, RGB input, or optical flow.
5. Output is finite float32, clipped once to `[0, 1]`, and encoded with the
   existing deterministic uint16 canonical encoder.
6. Temporal working memory is bounded independently of video and shot length.
7. The depth estimator is released before VDPP is constructed; VDPP is released
   before stereo rendering begins.
8. Raw depth and base canonical cache identities do not include the new
   setting. VDPP has a separate source- and checkpoint-bound identity.
9. Resume reuses only complete shots produced by the exact same VDPP identity.
10. Missing weights, download errors, model errors, and OOM never delete or
    invalidate valid base canonical disparity.
11. A requested VDPP run never silently falls back to unstabilized output.
12. The fixed window, overlap, downsize, precision, and range policy are
    algorithm constants, not user settings.
13. An authoritative job run has one writer. It acquires the output-directory
    lock before audit-driven mutation and holds it through finalization.
14. Cache-only execution is planned from persisted artifacts before either the
    base depth model or VDPP is imported, constructed, or loaded.

## Input And Output Semantics

### Input loader contract

The stage coordinator loads one ordered window from canonical PNG files. The
loader must return exactly the requested indexes and must validate:

- every path exists and decodes;
- storage dtype is uint16;
- rank is two;
- shape equals `metadata.native_shape`;
- decoded values are finite float32 in `[0, 1]`;
- returned count and order exactly match the request.

A short read, wrong order, wrong dtype, wrong geometry, decode failure, or
callback exception fails the current shot before its completion record is
committed. The exception identifies the source path or requested range.

### Deliberate difference from the upstream demo

The upstream demo normalizes each image-depth result with that frame's own min
and max before VDPP and holds every result in a Python list before
`torch.stack`. Those are demo composition choices, not acceptable project
contracts.

Per-frame min/max would erase the cross-frame scale that the canonical scene
barrier deliberately preserves. It also divides by zero on a constant frame
unless special-cased. This integration instead feeds the already bounded,
scene-normalized canonical values directly to VDPP. The checkpoint's learned
median-based differentiable scaler remains active.

The demo performs another whole-video min/max operation for visualization.
This integration does not. VDPP's residual output is clipped to `[0, 1]` once
and encoded. There is no per-shot or whole-video affine remapping in v1. This
keeps the source canonical scale meaningful and avoids a second full-shot
statistics barrier. The integration-domain difference is why the feature stays
experimental until the quality gate is measured.

### Derived render-disparity contract

The stabilized output remains a renderable relative disparity:

```text
representation = "relative_disparity"
near_value = 1.0
far_value = 0.0
encoding = "uint16_png"
encoding_scale = 65535.0
```

It is not allowed to masquerade as the base `scene-percentile-v1` producer.
The code adds a producer-specific stabilized metadata schema and a shared
`validate_render_disparity_input` helper. That helper accepts exactly two
known producers:

1. the existing base canonical schema and `scene-percentile-v1`; or
2. the new stabilized schema and `vdpp-canonical-shot-v1`.

Both paths must pass the common representation, frame manifest, shape, PNG
header, and self-hash checks. Producer-specific validation remains strict.
`StereoGenerator` uses the shared helper rather than weakening its current
validation to accept arbitrary metadata.

The existing stereo metadata field `source_canonical_fingerprint` is retained
for cache compatibility. For a VDPP run it contains the selected stabilized
render-disparity `artifact_fingerprint`. Its historical name does not permit an
unvalidated producer.

## Bounded Shot Algorithm

Finalized scene ranges from `01_scene_data` are half-open global frame ranges
`[shot_start, shot_end)`. If scene detection is disabled, the complete selected
video is one shot. Temporal state resets before every shot.

For one shot of length `N`:

```python
window = 32
overlap = 4
stride = 28

start = 0
end = min(start + window, N)
current_padded = forward_padded(load(start, end), downsize=True)
emit(resize_to_native(current_padded))
retain_device_float32(current_padded[-4:])

while end < N:
    start += stride
    end = min(start + window, N)
    current_padded = forward_padded(load(start, end), downsize=True)

    scale, shift = upstream_scale_shift(
        current_padded[:4],
        retained_previous_output,
    )
    current_padded = current_padded * scale + shift

    emit(resize_to_native(current_padded[4:]))
    retain_device_float32(current_padded[-4:])
```

The real implementation validates counts and shapes and does not retain state
after the last window. `upstream_scale_shift` is the released v1.0
`compute_scale_and_shift` calculation over all four overlap maps and all their
pixels with an all-valid mask. One scale and one shift are applied to the
entire later window.

There is no overlap interpolation. The first four predictions of each later
window are alignment observations and are discarded after alignment. Retained
reference maps are the unquantized, already aligned float32 output of the
previous window. Reading them back from uint16 PNG would introduce a
window-boundary quantization error and is forbidden. Alignment state is also
retained before the storage-range clip. Clipping is an encoding branch for
finalized output, not part of the upstream window-to-window recurrence. The
four-map retained tensor must be a compact detached clone. A slice view such as
`current_padded[-4:]` can keep the complete 32-map storage alive and violates
the stated memory bound.

The degenerate affine case follows the pinned upstream FP32 implementation
exactly. `scale` and `shift` are initialized to zero. When the determinant is
exactly zero they remain zero, so the aligned later window becomes zero. For a
nonzero determinant the upstream numerators and its `determinant + 1e-6`
denominator are used without an identity fallback or clamp. A non-finite
determinant, scale, shift, or aligned output fails the current shot and records
the scalar diagnostics. Changing the zero-determinant behavior, adding a clamp,
or substituting identity alignment requires a new algorithm version and cache
invalidation; it cannot still claim upstream v1.0 equivalence.

The upstream `infer_video_depth` first resizes an input whose height or width
is not a multiple of 14 upward with bilinear interpolation and
`align_corners=True`. Model forward, four-frame affine alignment, and retained
overlap state all operate at that padded shape. Only finalized output frames
are resized back to the original canonical shape, one at a time, using the
same bilinear/`align_corners=True` operation. Aligning after the resize would
change the global scale/shift sums and is not equivalent. The bounded adapter
preserves this order for every window. With `DOWNSIZE=true`, the inner
refinement shape for each axis is:

```text
padded = ceil_to_multiple(original, 14)
working = max(ceil_to_multiple(padded / 2, 14), 224)
```

### Tail examples

There is no tail padding:

| Shot length | Forward calls | Emitted result |
| --- | --- | --- |
| 1 | `[0:1]` | frame 0 |
| 25 | `[0:25]` | frames 0-24 |
| 31 | `[0:31]` | frames 0-30 |
| 32 | `[0:32]` | frames 0-31 |
| 33 | `[0:32]`, `[28:33]` | first 32, then only frame 32 |
| 60 | `[0:32]`, `[28:60]` | first 32, then frames 32-59 |
| 61 | `[0:32]`, `[28:60]`, `[56:61]` | first 32, then 28, then frame 60 |

Short first and final windows are forwarded at their actual length, matching
the released source. A later window always has at least five frames because a
new window is created only when at least one unprocessed frame remains after
the four-frame overlap.

## Artifact-First Planning And Ownership

### Artifact-first execution planning

Web and CLI must not load a depth model merely to decide whether persisted
artifacts already satisfy the requested job. Before any base-model or VDPP
import, construction, checkpoint resolution, or CUDA check, they build a
file-backed execution plan with these explicit decisions:

```text
can_run_cache_only
needs_base_depth_model
needs_vdpp_model
selected_render_source = base | stabilized
```

Planning has two phases. A provisional read-only inspection may drive UI
preview, but it is not authoritative. Job execution acquires the exclusive
output-directory writer lock, repeats the complete audit under that lock, and
only then mutates settings, caches, or output. The audit reads the saved schema
version before settings normalization and validates the artifact chain from the
latest requested downstream stage toward its inputs.

A complete stabilized stage is a historical, content-addressed artifact. Its
saved semantic lineage and stabilized payload digests are sufficient to select
it for rendering; current device selection and the ability to load the old base
estimator are irrelevant. The saved source lineage must still match the current
job source fingerprint and requested settings; artifact-first does not mean
reusing output for a changed input video. If that stage and the required
downstream inputs are valid, the plan sets `can_run_cache_only=true`, loads
neither neural model, and does not require the VDPP checkpoint or importable
vendored VDPP code.

If stabilized generation is needed but valid base canonical disparity already
exists, the plan loads only VDPP. If base canonical generation is needed, the
base estimator is loaded lazily, its current inference identity is resolved,
and the upstream portion of the plan is enriched before mutation. Existing raw
cache rules, including their current device participation, still apply when
base inference is actually required; they are not used to reject an already
complete downstream artifact.

Any identity needed for the model-free audit must come from saved metadata,
validated settings, and pinned lightweight manifests. If an existing identity
helper constructs or loads a neural model as a side effect, implementation must
split pure identity description from model loading. Artifact validation must
not hide an eager load behind a nominally read-only fingerprint call.

This is an intentional entry-point refactor. The current Web and CLI resume
paths load `StereoProjector` before `build_resume_report`; implementation must
move that load behind this artifact audit.

### Model ownership and release protocol

`StereoProjector` is the sole production owner of the base estimator's loaded
state. `ProcessingOrchestrator` owns stage order and invokes an injected owner
release callback. Production Web and CLI construction must inject
`StereoProjector.unload_model`; the `VideoProcessor` fallback that calls the raw
estimator directly is not an allowed VDPP production path because it cannot
maintain `StereoProjector._model_loaded`.

The existing owner method may be strengthened rather than adding a second
release API, but its contract is normative:

- it is idempotent whether the model was loaded, released after a cache hit, or
  partially initialized;
- after the final base-model CUDA operation it synchronizes the relevant CUDA
  device before model references and allocator cache are released;
- estimator unload and allocator cleanup run even if synchronization or another
  release step reports an error;
- `StereoProjector._model_loaded` becomes false in a `finally` path, so owner
  state cannot disagree with estimator residency;
- the existing top-level `finally: unload_model()` remains safe after the
  earlier canonical-stage release;
- normal completion, base-canonical cache hits, depth failures, cancellation,
  and VDPP construction/checkpoint failures all leave the owner released.

After `DepthMapProcessor.generate_depth_map_files` returns, the existing
`ProcessingOrchestrator._execute_pipeline` `finally` invokes that owner callback.
The outer `ProcessingOrchestrator.process` `finally` is the idempotent safety
net. Only after owner release completes may VDPP be constructed. A successful
owner callback return is the boundary assertion: the callback must verify
internally that its loaded flag is false before returning, and lifecycle tests
inspect owner state before CUDA is handed to VDPP. This avoids adding a second
public state probe solely for orchestration.

The orchestrator then selects the effective render-disparity input:

```python
base_files = depth_processor.generate_depth_map_files(...)
release_depth_model_owner()

effective_files = base_files
if settings["temporal_postprocessor"] == "vdpp":
    effective_files = temporal_stabilizer.generate_files(
        base_files,
        settings,
        directories,
        progress_tracker,
    )

return execute_remaining_steps(depth_files=effective_files, ...)
```

On an artifact-first cache hit, the base generation and release calls are
no-ops and `effective_files` comes directly from the validated stabilized
artifact. On a generation path, VDPP is released in a `finally` block inside
the temporal stage and downstream work begins only after a complete, validated
stabilized stage exists. At no point may base-model and VDPP parameter tensors
be resident together.

The new `TemporalDepthStabilizer` stage coordinator owns:

- canonical metadata and PNG validation;
- finalized shot ranges;
- file loading and deterministic encoding;
- stabilized metadata, partial-shot cleanup, and resume;
- progress and preview events;
- checkpoint resolution and lazy post-processor construction;
- the global stabilized completion barrier.

The new `VDPPTemporalPostprocessor` owns:

- the pinned model architecture and strict checkpoint load;
- upstream spatial transforms;
- one-window forward calls;
- upstream scale/shift alignment;
- unquantized four-frame retained state;
- device tensor release.

The post-processor never reads project settings, discovers scene cuts, writes
stage files, or knows estimator backend names. The coordinator never
reimplements neural network layers or temporal alignment math.

The narrow logical model interface is:

```python
class VDPPTemporalPostprocessor:
    def process_shot(
        self,
        frame_count: int,
        load_window: Callable[[int, int], np.ndarray],
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ordered native-shape aligned float32 frames before storage clip."""

    def model_identity(self) -> dict[str, object]: ...
    def execution_plan(self, native_shape: tuple[int, int]) -> dict[str, object]: ...
    def release(self) -> None: ...
```

`load_window(start, end)` uses shot-local half-open indexes and returns exactly
`end - start` maps shaped `[S, H, W]`. The coordinator alone maps shot-local
indexes to global canonical paths. The adapter validates finite model output;
the adapter performs padded-space recurrence and the final upstream resize; the
coordinator alone applies the `[0, 1]` storage clip and canonical PNG encoder.

The shot iterator owns its retained device tensor in a `try/finally` and clears
it on normal exhaustion, loader/model failure, cancellation, or explicit
iterator close. The coordinator always closes the iterator in its own
`finally`. `release()` is idempotent and also clears any abandoned shot state;
correctness must not depend on Python garbage collection releasing CUDA memory.

## Settings And Cache Identity

`temporal_postprocessor` is added to `DEFAULT_SETTINGS` and typed choice
validation. It is serialized in job settings and accepted by Web and CLI.
`PROCESSING_SETTINGS_SCHEMA_VERSION` is bumped from 2 to 3. Forward-reading old
jobs and preventing an old interpretation of new jobs are different problems;
default insertion alone solves only the first.

The raw settings metadata version is inspected before
`validate_settings(..., source="legacy_disk")` can filter unknown fields. The
reader follows one-way rules:

```python
if saved_version > PROCESSING_SETTINGS_SCHEMA_VERSION:
    fail_without_backup_write_or_job_mutation()
elif saved_version < PROCESSING_SETTINGS_SCHEMA_VERSION:
    migrate_known_versions_upward_only()
else:
    parse_v3_strictly()
```

Absent legacy version markers and known v1/v2 files may migrate upward. The v2
to v3 migration inserts `temporal_postprocessor="off"` only when the field is
absent; a present prerelease value is validated and preserved rather than
overwritten. A v3 file must contain a valid `off | vdpp` value and rejects
unknown settings fields. Future versions fail closed without making a backup,
rewriting settings, changing status, deleting caches, or acquiring a plan that
can execute. Downward migration is forbidden.

No new implementation can retroactively repair a previously released v2
binary whose legacy reader discards unknown fields. Opening a v3 job with a
pre-v3 client is explicitly unsupported; the compatibility promise is that v3
and later readers preserve old jobs and never silently reinterpret a newer job,
not that old executables can safely edit v3 state. User documentation must state
that boundary.

Its complete data flow is:

```text
Web segmented control / CLI flag
  -> request or argparse settings
  -> schema-first migration / strict validation
  -> saved processing settings
  -> ProcessingOrchestrator artifact-first source selection
  -> TemporalDepthStabilizer only when value is vdpp
```

It may be present in the full settings dictionary already passed through the
pipeline, but `DepthMapProcessor`, `DepthBatch`, and depth estimators do not
consume it or include it in their identities.

The existing `temporal_window_size` and `temporal_window_overlap` compatibility
fields do not configure VDPP. VDPP's 32/4 values come only from its pinned
algorithm contract. The separate V2 specification continues to own the legacy
fields and their current cache participation.

The field participates as follows:

| Identity | Participation | Reason |
| --- | --- | --- |
| Saved job settings | Direct | Reproduce the requested product path |
| Local `02_depth_raw` semantic fingerprint | No | Post-processing cannot change inference |
| Global canonical cache key | No | Base `03_disparity_maps` is unchanged |
| Base canonical metadata | No | It remains a reusable upstream stage |
| `03_disparity_stabilized` semantic identity | Direct | Selects the VDPP stage |
| Stereo and later stages | Indirect | Their selected source disparity fingerprint changes |

The implementation must not add the field to `DEPTH_MODEL_SETTING_KEYS` or
`DEPTH_CACHE_SETTING_KEYS`. That would invalidate expensive depth results for a
downstream choice and would repeat the original temporal-settings cache error.

## Stabilized Metadata

`03_disparity_stabilized/metadata.json` is written atomically after each
completed shot and once more when the stage becomes complete. Its logical
shape is:

```json
{
  "schema_version": 1,
  "algorithm_version": "vdpp-canonical-shot-v1",
  "status": "building",
  "representation": "relative_disparity",
  "near_value": 1.0,
  "far_value": 0.0,
  "encoding": "uint16_png",
  "encoding_scale": 65535.0,
  "num_frames": 1234,
  "semantic_identity": {
    "frame_names": ["frame_000001.png"],
    "native_shape": [518, 910],
    "source_canonical_fingerprint": "...",
    "scene_manifest_fingerprint": "...",
    "postprocessor_settings": {
      "temporal_postprocessor": "vdpp"
    },
    "model_identity": {
      "name": "vdpp",
      "upstream_release": "v1.0",
      "upstream_revision": "73cc2b4dc6b3b5cfb2e37f51e452461e03fe26f5",
      "checkpoint_sha256": "7368315b126093f0335147f42a1920f255d529613bfffc5c6cf4ef832deb73a7",
      "checkpoint_size": 116485370,
      "architecture": "vits-temporal-dpt",
      "model_config": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "use_bn": false,
        "use_clstoken": false,
        "num_frames": 32,
        "max_depth": 1.0,
        "pe": "ape"
      },
      "vendor_port_version": 1,
      "checkpoint_compatibility": "released-zero-shift-head-v1"
    },
    "execution_plan": {
      "batch_size": 1,
      "window_size": 32,
      "overlap": 4,
      "stride": 28,
      "tail_padding": false,
      "downsize": true,
      "precision": "fp32",
      "input_normalization": "canonical-scene-percentile-v1-no-frame-renorm",
      "output_range_policy": "native-resize-then-clip-0-1-v1",
      "alignment": "vdpp-global-affine-v1",
      "spatial_transform": "ceil14-half-ceil14-min224-bilinear-align-corners-v1",
      "padded_input_shape": [518, 910],
      "working_shape": [266, 462]
    },
    "shot_plan": [
      {"shot_id": 0, "start": 0, "end": 120}
    ]
  },
  "semantic_fingerprint": "...",
  "execution_provenance": {
    "torch_version": "...",
    "xformers_version": "...",
    "attention_backend": "xformers",
    "attention_operator": "...",
    "cuda_runtime": "...",
    "cuda_driver": "...",
    "cudnn_version": "...",
    "device_name": "...",
    "compute_capability": [8, 9],
    "float32_matmul_precision": "highest",
    "allow_tf32": false,
    "deterministic_algorithms": true,
    "cudnn_deterministic": true,
    "cudnn_benchmark": false,
    "preflight_max_memory_allocated": 0,
    "preflight_max_memory_reserved": 0
  },
  "partial_resume_runtime_fingerprint": "...",
  "completed_shots": [
    {
      "shot_id": 0,
      "manifest": "shot_manifests/shot_000000.json",
      "manifest_sha256": "...",
      "shot_payload_sha256": "..."
    }
  ],
  "state_fingerprint": "...",
  "payload_fingerprint": null,
  "artifact_fingerprint": null,
  "metadata_fingerprint": "..."
}
```

The example lists only one frame name for readability; the real manifest lists
all selected source frame names. The execution plan is built by the same code
that supplies forward-call arguments; there are no duplicated precision or
downsize literals.

The fingerprints have separate jobs and are never substituted for one another:

- `semantic_fingerprint` hashes immutable source lineage, final scene manifest,
  ordered frame/shot plan, pinned source and checkpoint identity, model config,
  spatial transforms, 32/4 recurrence, precision, normalization, and range
  policy. It excludes status, completed shots, device, and runtime versions.
- `partial_resume_runtime_fingerprint` hashes the actual PyTorch and xFormers
  versions/operator, CUDA runtime and driver, compute capability, effective
  attention backend, cuDNN version, GPU model, TF32/matmul mode, and
  deterministic/cuDNN flags. It is a compatibility gate for adding neural
  output to a partial stage, not part of a completed artifact's semantic
  identity.
- `state_fingerprint` hashes `status` and the ordered `completed_shots` records.
  It changes during progress and is never consumed as a downstream source ID.
- `payload_fingerprint` hashes the ordered shot payload digests and is populated
  only after every shot validates.
- `artifact_fingerprint` is the canonical hash of
  `{semantic_fingerprint, payload_fingerprint}`. Stereo and every later stage
  use this immutable value as the selected render-disparity source identity.
- `metadata_fingerprint` is a self-hash over every other metadata field. It
  detects metadata damage but is not an artifact identity.

Each completed shot has an immutable
`03_disparity_stabilized/shot_manifests/shot_NNNNNN.json` containing its exact
half-open range and an ordered list of `{name, sha256}` entries:

```json
{
  "schema_version": 1,
  "shot_id": 0,
  "start": 0,
  "end": 120,
  "files": [
    {"name": "frame_000001.png", "sha256": "..."}
  ],
  "shot_payload_sha256": "...",
  "manifest_fingerprint": "..."
}
```

`shot_payload_sha256` is the canonical JSON hash of the ordered name/digest
pairs; concatenated strings with ambiguous boundaries are forbidden.
`manifest_fingerprint` excludes only that self-hash. PNG hashes are calculated
from final encoded bytes before the shot is committed. The top-level
`manifest_sha256` hashes the final manifest file bytes, including its self-hash,
and binds that value together with the shot payload SHA-256.

`status` becomes `complete` only when every expected shot manifest and payload
digest validates. A complete stage is reusable on a different runtime because
no new neural output is being mixed into it. A partial stage may continue only
when both semantic and partial-runtime fingerprints match. A runtime mismatch
resets the entire unfinished stabilized stage, including its previously
completed shots, before any new payload is written.

The new byte-integrity guarantee applies to stabilized outputs. The existing
base canonical producer continues to provide its current metadata, decode, name,
shape, bit-depth, and range validation; this specification does not silently
upgrade it to detect arbitrary replacement by another structurally valid PNG.
Accordingly, references to corrupt canonical input below mean missing,
undecodable, truncated, or structurally invalid input, while stabilized
corruption includes any byte change.

## Resume And Invalidation

One finalized shot is the smallest reusable temporal unit. Windows inside a
shot share four unquantized output maps, so arbitrary frame-level resume is not
valid. Shots share no temporal state and can be validated independently.

Every payload is written to a same-directory temporary file and atomically
replaced. Its immutable shot manifest is then written atomically, and only then
is the shot appended to top-level `completed_shots`. Completion requires name,
shape, bit-depth, count, per-file SHA-256, manifest SHA-256, and shot payload
SHA-256 validation. If a process stops mid-shot, its orphan payloads or manifest
are not trusted. The next run deletes the files in that incomplete shot range
and recomputes the complete shot.

If a completed shot has a missing, malformed, or digest-mismatched PNG or
manifest, only that shot is removed from completion state and recomputed when
the semantic and partial-runtime fingerprints still match. Because later shots
reset temporal state, they remain internally reusable. Stereo and downstream
output are nevertheless invalidated when the selected stabilized source fails
validation; rendering does not begin from a partially repaired source.

The resume matrix is:

| Change | Raw | Base canonical | Stabilized | Stereo/downstream |
| --- | --- | --- | --- | --- |
| Old job, field absent, current `off` | Preserve | Preserve | Ignore/dormant | Preserve |
| `off -> vdpp` | Preserve | Preserve | Generate or reuse exact match | Invalidate |
| `vdpp -> off` | Preserve | Preserve | Keep dormant | Invalidate |
| VDPP checkpoint/source/algorithm changes | Preserve | Preserve | Invalidate all | Invalidate |
| VDPP interrupted with same identity | Preserve | Preserve | Resume complete shots | Not reusable yet |
| One completed VDPP shot is corrupt | Preserve | Preserve | Recompute that shot | Invalidate |
| Partial VDPP runtime fingerprint changes | Preserve | Preserve | Reset unfinished stage | Invalidate |
| Complete VDPP runtime fingerprint changes | Preserve | Preserve | Reuse hashed artifact | Follow artifact identity |
| Canonical fingerprint changes | Follow current rules | Regenerate/reuse | Invalidate all | Invalidate |
| Final scene manifest changes | Follow current rules | Follow current rules | Invalidate all | Invalidate |

When mode is off, resume does not delete a valid stabilized directory. It is a
dormant derived cache that can be reused if the user re-enables VDPP with the
same identity. The resume report omits the optional stage while off rather than
calling it current render input. Normal successful cleanup may still remove it
when `keep_intermediates=false`; dormancy does not override the existing
retention policy.

There is no global cross-job VDPP cache in v1. The existing global cache stops
at base canonical disparity. Local job resume is sufficient for the first
release and avoids doubling global cache storage before value is measured.

### Single-writer protocol

One process may mutate a job output directory at a time. The raw schema-version
probe happens before any lock-file creation so a future schema can fail without
mutation. For a supported schema, authoritative new-job or resume execution:

1. acquires a fail-fast, OS-backed exclusive lock in the output directory;
2. repeats schema parsing and the entire artifact audit after acquisition;
3. performs settings migration, cleanup, stabilization, stereo, encoding, and
   final status writes while holding that lock; and
4. releases the lock in the top-level `finally` after the last mutation.

A second writer fails clearly; it does not wait and then execute a stale plan.
The lock file records PID, hostname, process start identity, and acquisition
time for diagnostics, but the operating-system lock is authoritative. Process
death releases the OS lock; a stale diagnostic record alone never blocks or
authorizes a run and may be replaced only after lock acquisition. A read-only
UI preview may inspect without the lock but cannot migrate, delete, repair, or
promise that its provisional plan remains current. The checkpoint download
artifact lock is separate and does not replace the job writer lock.

### Shot-atomic recovery cost

Shot atomicity is a deliberate v1 product limitation. If scene detection is
disabled, produces no cuts, or fails open to one range, an interruption near the
end of a long video recomputes that entire shot. Memory remains bounded, but
resume time does not. Window-level checkpoints are a non-goal for v1. A future
version would need to persist the next source index plus the compact four-map
padded, aligned, unclipped float32 overlap state, and bind that state to the
exact semantic and partial-runtime fingerprints. Persisting only a frame index
is not sufficient.

## Memory Bound And Model Lifetime

Let:

```text
H, W   = canonical native shape
Hp, Wp = each axis rounded upward to a multiple of 14
P      = 4 * H * W bytes for one host float32 map
Pp     = 4 * Hp * Wp bytes for one padded device float32 map
S      = current window length, 1 <= S <= 32
I(S)   = padded device input allocation, bounded by S * Pp
O(S)   = current model output, S * Pp
T(S)   = out-of-place multiply temporary, S * Pp
F(S)   = aligned output allocation, S * Pp
R      = compact retained overlap, 4 * Pp on a continuation window
E      = one finalized native resize result, at most Pp
A(S)   = model-internal forward activations/workspace, excluding I/O and params
Q(S)   = alignment mask/reduction/scalar scratch
M      = model parameters and persistent buffers
```

The coordinator decodes at most 32 input maps into one preallocated buffer and
receives one finalized native-shape output map at a time. The four prior
float32 reference maps remain adapter-owned at padded shape on the device. The
algorithm-owned host tensor bound is therefore at most approximately:

```text
33 * P + bounded PNG codec buffers
```

The pinned upstream affine expression is out of place:

```python
affined_depths = depths * scale + shift
```

The live-allocation bound must therefore include both the multiplication
temporary and aligned result; it must not assume an unproved in-place rewrite.
Conservatively retaining the input reference through alignment gives:

```text
device_live_peak <= M + max(
    R + I(S) + O(S) + A(S),
    R + I(S) + O(S) + T(S) + F(S) + Q(S),
    R + F(S) + E
)
```

For a full continuation window, the second term contains at most 132 padded-map
equivalents plus explicit `Q(32)` and `M`; this is intentionally more
conservative than depending on Python reference destruction to reduce it to 100.
`A(S)` and allocator-reserved memory depend on the installed attention kernel
and CUDA allocator, but are bounded because sequence length is at most 32 and
working shape is fixed by the canonical shape and pinned transform. The formula
describes live allocations, not a promise that CUDA reserved bytes equal live
bytes. Neither host nor device expression contains video or shot length. Tests
instrument the structural lifetimes and CUDA peaks rather than merely inspect
the loop.

The host bound requires one preallocated `[S, H, W]` float32 input buffer. PNGs
are decoded into their assigned slices one at a time; the implementation must
not keep 32 decoded arrays and then allocate a second stacked copy. Finalized
device outputs likewise resize to native shape, transfer, and encode one frame
at a time. Padded recurrence state is not copied to host between windows.

No code path may build a list of all shot/video tensors, call `torch.stack` on
the entire sequence, or call upstream `infer_video_depth` with the complete
shot in production.

The lifetime order is artifact-first and strict:

```text
schema probe -> acquire job lock -> authoritative artifact audit
  -> complete stabilized hit: load no depth model and begin stereo
  -> otherwise, only if base canonical is missing:
       load selected depth estimator
       -> write/validate base canonical
       -> owner releases depth estimator
  -> only if stabilized output is missing:
       assert base owner is released
       -> load VDPP
       -> write/validate stabilized stage
       -> release VDPP in finally
  -> begin stereo rendering
```

## Disk Bound

VDPP adds one uint16 PNG per selected frame. Before model construction, disk
preflight adds the conservative uncompressed allowance:

```text
num_frames * H * W * 2 * 1.10 + one atomic PNG temporary
```

This allowance is required even when `keep_intermediates=false`, because the
file-backed stereo barrier needs the complete stabilized stage before later
cleanup. No float32 shot or video spool is written. A disk-space failure occurs
before checkpoint download and leaves base canonical output reusable.

## CUDA Preflight And OOM Policy

The first release supports VDPP computation on CUDA only. A complete,
validated stabilized cache can still be rendered on CPU because it requires no
VDPP import, checkpoint, or neural execution. If generation or partial resume
is required, an effective CPU or MPS device fails before checkpoint download or
stabilized-stage mutation with a clear message. This avoids pretending that an
unbenchmarked 111M temporal model is a practical CPU feature. CPU/MPS
generation can be added later with their own real checkpoint tests and
performance contract.

If every stabilized shot is already complete and valid, VDPP is not loaded and
no preflight runs. Otherwise, after strict checkpoint load and before any new
payload commit or destructive stabilized-stage repair, the adapter finds the
longest shot that still requires generation after semantic/runtime resume rules
have been applied. It resets CUDA peak statistics and runs an inference-mode,
shape-faithful trial on constant 0.5 input using the exact native shape, padded
and downsize transforms, FP32 mode, and effective attention operator.

For a pending shot of at most 32 frames, preflight forwards its actual length
and performs one native resize/host emission. For a pending shot longer than 32,
preflight exercises the continuation path, not only the first window:

1. forward the first 32 maps;
2. retain a compact detached clone of its last four padded outputs and release
   the rest of the first-window state;
3. forward `min(32, pending_shot_length - 28)` maps for the next window;
4. run the exact all-valid `compute_scale_and_shift` path;
5. evaluate the exact out-of-place `depths * scale + shift` expression; and
6. resize and transfer at least one finalized non-overlap result.

Thus a pending shot of at least 60 frames preflights a full 32-frame
continuation with 4 retained maps. A 33-frame shot preflights its real five-map
continuation rather than inventing a shape it will not execute. The coordinator
records and reports `torch.cuda.max_memory_allocated()` and
`torch.cuda.max_memory_reserved()` separately in execution provenance, together
with native/working shapes and the sequence lengths exercised. All trial
tensors and recurrence state are synchronously released before production.

Preflight verifies that the job can execute the structural peak path under the
then-current allocator state. It is not a promise against later fragmentation,
another process consuming VRAM, driver failure, or data-independent allocator
changes. Preflight failure leaves base canonical output and previously committed
valid stabilized shots untouched and reports the plan and CUDA OOM cause.

The stage does not reduce temporal window size, switch precision, change
working resolution, or turn itself off after OOM. Those would create a
different, unverified model contract. A rare later OOM marks the current shot
incomplete, atomically preserves earlier complete shots, releases VDPP, and
aborts before stereo. The user can resume after freeing VRAM or switch the
setting to off while reusing base canonical disparity.

## Source And Checkpoint Distribution

The application must not clone a Git repository or mutate `sys.path` during a
job. The required upstream inference subset is vendored inside the installed
Python package at the pinned revision. Only `vdpp/**` and
`utils/normal_utils.py` are needed; the upstream demo, assets, DAv2 submodule,
and visualization utilities are excluded.

Vendored files retain upstream copyright/license headers. Mechanical package
import changes are documented in a small `UPSTREAM.json` manifest containing
the repository, revision, original paths, file hashes, and
`vendor_port_version`. The complete Apache-2.0 license and a third-party notice
ship with the package.

The minimal inference subset uses `torch`, `einops`, `easydict`, and optionally
`xformers`, which are already project dependencies. The implementation does not
copy the upstream `requirements.txt` pins or add its demo-only visualization,
DAv2, pandas, wandb, or audio dependencies.

The checkpoint resolves to:

```text
models/VDPP/vdpp.pth
```

It is downloaded only after VDPP is requested, CUDA validation succeeds, and a
valid local stabilized cache has not already satisfied the job. Download uses
a bounded timeout, streams to `vdpp.pth.part`, reports byte progress, verifies
the exact size and SHA-256, and atomically replaces the final path. A final file
with the wrong hash is never loaded. Concurrent jobs use an artifact lock and
re-check the final hash after acquiring it.

Checkpoint loading uses `torch.load(..., map_location="cpu", weights_only=True)`
and `load_state_dict(..., strict=True)` before moving the model to CUDA. If the
installed PyTorch cannot read the pinned state dictionary safely, the feature
fails rather than retrying unsafe pickle loading.

The loaded module is always `.eval()` and every preflight/production forward
runs under `torch.inference_mode()`. Autocast, gradient checkpointing, compile,
and quantization are not silently enabled because they would create a different
execution and numerical contract.

Upstream reports testing Python 3.10 and CUDA 12.6. This project uses a
different supported runtime matrix. Real-checkpoint smoke tests on every
claimed release platform are required; successful upstream testing is not
substituted for local compatibility evidence.

## Web, CLI, And Progress

Web exposes one two-mode control in depth settings:

```text
Temporal post-processing: Off | VDPP (Experimental)
```

It is a segmented mode control with a tooltip, not a group of tuning inputs.
The tooltip states that VDPP is a separate depth-only post-process, adds a model
download and processing time, requires CUDA to generate, and is usually
unnecessary for V2. The control does not claim guaranteed improvement for every
model or clip.

CLI adds:

```text
--temporal-postprocessor {off,vdpp}
```

The argument parser default is `None` (or the attribute is suppressed), not the
string `off`, because omission and an explicit override have different resume
semantics:

```text
new job + omitted       -> resolve to off before v3 settings are saved
resume + omitted        -> retain the persisted v3 value
resume + explicit value -> apply that value and invalidate only derived/later stages
```

Web requests preserve the same presence distinction. A resume form hydrates the
control from persisted settings; its visual default must not overwrite `vdpp`.
An explicit `off` request is a real user change and selects base canonical
disparity. Web and CLI feed the same typed resolver after schema migration.
There are no `vdpp_window`, `vdpp_overlap`, `vdpp_strength`,
`vdpp_resolution`, or `vdpp_precision` controls.

The Web device status already reports CUDA availability. For a new job, the
start action rejects the incompatible combination of VDPP and an effective
non-CUDA device before job creation. For resume, artifact-first planning first
checks whether a complete hashed stabilized stage can satisfy the request. It
permits that cache-only path without CUDA and rejects a non-CUDA device only if
the authoritative locked plan requires VDPP generation. CLI follows the same
new-job/resume rule. The UI describes CUDA as required to **generate** VDPP
output, not to read cached PNGs.

Progress uses a distinct visible step name, `Temporal Depth Stabilization`, and
reports finalized frames out of total frames. Checkpoint download reports bytes
inside the same step. It does not reuse `Depth Map Generation` timing.

Because the step is optional, the progress plan is dynamic:

- with VDPP off, the existing eight steps and `[0.02, 0.35, 0.20, 0.08, 0.02,
  0.18, 0.08, 0.07]` weights remain unchanged;
- with VDPP on, the existing depth weight is split into `0.28` for Depth Map
  Generation and `0.07` for Temporal Depth Stabilization; all later weights
  remain unchanged and the total remains 1.0.

This prevents overall progress from moving backward when stabilization begins
and gives the optional stage its own ETA clock. Cache hits complete the step
without constructing the model. Important failures and OOM are always written
to the normal process log as well as the Web progress channel; a throttled
progress event is never the only record.

## Error Handling

- A future settings schema fails before lock-file creation, backup, migration,
  cache audit with side effects, model construction, or status mutation.
- Output-directory writer-lock contention fails immediately and identifies the
  recorded owner for diagnosis; it never steals a live OS lock.
- Missing or invalid canonical metadata fails before model construction.
- A missing, undecodable, structurally invalid, short, or reordered canonical
  payload fails the owning shot with a path/range-specific error. Arbitrary
  valid-PNG content replacement is outside the existing base contract.
- Any stabilized file or shot-manifest digest mismatch invalidates that shot
  even when the replacement remains a structurally valid uint16 PNG.
- A malformed final scene manifest fails before a temporal window is planned.
- A checkpoint size/hash mismatch fails before `torch.load`.
- A state-dict mismatch fails before CUDA transfer.
- A non-finite, wrong-rank, wrong-count, or wrong-shape model result fails before
  the shot is marked complete.
- CUDA preflight/OOM follows the explicit policy above and never deletes base
  canonical disparity.
- User cancellation stops at the next loader/window/output boundary, leaves the
  current shot incomplete, releases VDPP and the base-model owner, and preserves
  prior complete shots.
- A requested VDPP failure returns a failed job. It does not silently render
  from base canonical disparity.
- When the setting is off, or when a complete stabilized cache satisfies a VDPP
  request, missing VDPP source/checkpoint is irrelevant: there is no import,
  network request, warning, or VDPP device allocation.

## Verification

### Upstream equivalence

- Use a deterministic fake `forward` and compare the bounded shot iterator with
  pinned upstream `infer_video_depth` on the same synthetic input.
- Cover lengths 1, 2, 3, 4, 5, 15, 16, 25, 31, 32, 33, 59, 60, 61, and a long
  multi-window sequence.
- Cover native shapes whose axes are and are not multiples of 14.
- Match frame order, resize/crop behavior, scale/shift calculation, overlap
  discard, short-tail behavior, and final count within a documented FP32
  tolerance.
- Prove scale/shift and retained overlap use the padded multiple-of-14 output,
  and that only finalized frames are resized back to native shape. Include a
  case that would produce a different affine fit if alignment happened after
  resizing.
- Prove that the retained reference is post-alignment float32 output rather
  than encoded/redecoded PNG, and that it is retained before output clipping.
- Test the actual upstream `compute_scale_and_shift` path, including degenerate
  determinant behavior, rather than monkeypatching alignment away. Assert exact
  zero determinant produces zero scale, zero shift, and zero aligned output;
  assert any NaN/Inf fails the shot rather than invoking identity fallback.
- Compare with upstream using `downsize=True`; do not compare against its CLI
  default-false path and call that equivalent.

### Data and memory contracts

- Assert no loader call requests more than 32 maps and every later full request
  advances by 28 source indexes.
- Instrument peak decoded input count at 32, retained reference count at 4, and
  one-at-a-time finalized host output transfer independently of shot length.
- Instrument a full continuation so retained input, current model output,
  multiply temporary, aligned output, alignment scratch, and resize output are
  accounted for separately. The test must fail if it silently assumes an
  in-place affine operation.
- Assert retained overlap does not alias the 32-map current tensor and its
  underlying device storage contains exactly four padded maps.
- Instrument the adapter so production never accumulates a whole-shot list or
  calls full-shot `infer_video_depth`.
- Test candidate videos of equal shape at 64 frames and 10,000 frames and prove
  the retained tensor counts are identical.
- Test that no planned request crosses a finalized scene cut and that all state
  resets at each cut.
- Test scene detection disabled as one bounded long shot.
- On CUDA, preflight both a first-window-only shot and a >=60-frame pending shot;
  assert the latter executes 4 retained + 32 current continuation alignment and
  records nonzero maximum allocated and maximum reserved bytes independently.

### Canonical semantics

- Decode canonical uint16 with the existing helper and encode stabilized output
  with the existing deterministic helper.
- Use frames with identical structure but different canonical ranges to prove
  the coordinator performs no per-frame min/max normalization.
- Test constant zero, constant one, clipped-boundary, and high-gradient maps.
- Reject NaN/Inf, wrong dtype, wrong shape, wrong name order, and short loader
  results before completion metadata changes.
- Prove output names, count, native shape, polarity, and `[0, 1]` range.

### Cache and resume

- Assert `temporal_postprocessor` is absent from local raw and global canonical
  identity builders.
- Assert toggling it preserves `00_original_frames`, `01_scene_data`,
  `02_depth_raw`, and `03_disparity_maps`.
- Assert toggling it changes the selected render-disparity fingerprint and
  invalidates stereo/downstream output.
- Test the settings matrix: v2/missing field migrates to `off`; v2/present valid
  field is preserved; v3/`vdpp` is preserved; v3/missing, invalid, or unknown
  fields fail strictly; and a future schema fails with the settings file and
  job tree byte-for-byte unchanged.
- Test CLI new-job omission, resume omission, and explicit resume `off` as three
  distinct cases. Test Web resume hydration from persisted `vdpp` and request
  omission versus explicit override.
- Test full stabilized reuse without importing VDPP, resolving its checkpoint,
  loading the base estimator, or requiring its historical device.
- Test interruption in a shot, deletion/recompute of only that incomplete shot,
  and reuse of earlier complete shots.
- Test interruption in a single long scene restarts the whole shot and does not
  claim window-level resume.
- Replace one completed-shot PNG with a different valid uint16 PNG of identical
  dimensions; its file/shot/artifact digest must fail and downstream must
  invalidate. Test missing and modified shot manifests separately.
- Test source canonical, scene manifest, algorithm, source revision, checkpoint,
  downsize, precision, and range-policy semantic identity changes. Change each
  runtime-fingerprint field independently: reuse a complete hashed stage, but
  reset the entire partial stage before producing more shots.
- Assert mutable status/completed-shot changes affect only state and metadata
  fingerprints, while payload bytes affect payload/artifact fingerprints and
  downstream source selection.
- Test that off mode leaves a valid dormant stabilized directory untouched and
  never selects it for stereo.
- Test that the render-disparity validator accepts only the existing canonical
  producer and exact VDPP producer, rejecting arbitrary lookalike metadata.

### Artifact, device, and lifecycle

- Test atomic checkpoint download, timeout, interrupted `.part`, exact size,
  exact SHA-256, concurrent lock, and corrupt-final-file behavior without
  downloading in ordinary CI.
- Run two writers against one job directory. The second fails fast, the first
  retains the lock through downstream mutation, and the post-lock audit catches
  a TOCTOU change made after a provisional read. Simulate process death to prove
  OS lock release, not stale metadata deletion, controls recovery.
- Test `weights_only=True` and strict state-dict loading.
- Test early CPU/MPS rejection before download and stabilized mutation when VDPP
  generation is required. Separately, render a complete exact-match cache with
  CUDA unavailable, VDPP unimportable, checkpoint absent, and the base estimator
  deliberately unloadable.
- Test shape-faithful CUDA preflight before the first new payload commit,
  including actual continuation affine and emission peaks.
- Test that production injects the `StereoProjector` owner callback, owner
  `_model_loaded` is false before VDPP construction, and VDPP is released before
  stereo. Cover normal generation, canonical cache hit, depth exception,
  cancellation, synchronization/unload failure, and VDPP initialization failure;
  repeated top-level unload must remain safe.
- Test loader failure, model failure, cancellation, and explicit generator
  close after the first yielded frame; all must release compact overlap state
  without waiting for garbage collection.
- Test preflight OOM and late OOM: canonical remains valid, current VDPP shot is
  incomplete, and no new stereo payload is written.
- Run a real pinned-checkpoint CUDA smoke test at release time on the supported
  Windows and Linux runtime matrix. It is an explicit release test, not normal
  CI.
- Build the wheel/sdist and assert the vendored source, `UPSTREAM.json`,
  Apache-2.0 license, and third-party notice are present and their manifest
  hashes match.

### Product quality gate

Correct plumbing is not evidence that canonical inputs fall inside the
checkpoint's best domain. Before removing the Experimental label, the
implementation checks in `benchmarks/vdpp/evaluation_manifest.json`. It pins:

- dataset release URLs, licenses, archive checksums, exact clip IDs and half-open
  frame ranges, and source-frame digests;
- canonical preprocessing/algorithm versions, complete base and candidate
  settings, model source revisions, and checkpoint hashes;
- the metric implementation revision and command, runtime provenance, hardware,
  and fixed seeds `[0, 1, 2]`;
- DA3MONO and DA3METRIC configurations plus a V2 control where the hardware
  matrix can run it.

For each backend, metric, and sequence, run baseline and VDPP three times. Take
the median across the three repeats for that sequence, then the unweighted
median across the fixed sequences. Preserve unrounded per-repeat/per-sequence
JSON and the resolved manifest with every result. A backend/domain combination
may be called recommended only when all three aggregate comparisons pass:

```text
TGSE_vdpp  <= 0.99 * TGSE_base
AbsRel_vdpp <= 1.02 * AbsRel_base
delta1_vdpp >= delta1_base - 0.02
```

The TGSE requirement is at least 1 percent relative improvement. AbsRel permits
at most 2 percent relative regression. Delta-1 permits at most 0.02 absolute,
that is two percentage points, regression. No rounded display value is used for
the decision.

Temporal slit-scans and final stereo output are also inspected for edge
breathing, foreground/background swaps, clipping, lag, and over-smoothing. A
separate checked-in anime/See-Through clip manifest uses a blinded base versus
VDPP rubric and records reviewer sign-off because Sintel ground truth does not
establish anime quality. A failed gate leaves that combination Experimental or
documented as not recommended; it does not trigger hidden backend exceptions.

This gate uses the paper's ground-truth TGSE where it is meaningful. It does
not add RAFT merely to manufacture an optical-flow-based production metric.

## Acceptance Criteria

The implementation is accepted only when all of the following are true:

1. `off` is the default and preserves previous generated payloads, directory
   tree, pipeline behavior, and cache identities; only the saved settings
   manifest may record the new explicit default.
2. The bounded adapter is numerically equivalent to pinned upstream v1.0 for
   the chosen 32/4/downsize/FP32 plan under deterministic fake forward tests.
3. No window crosses a finalized scene cut and every input frame is emitted
   exactly once.
4. Peak retained sequence state is independent of video and shot length.
   The symbolic bound includes retained overlap, model output, both out-of-place
   affine buffers, alignment scratch, activations, and parameters, and the CUDA
   preflight exercises a real continuation peak when one is pending.
5. VDPP consumes and emits the explicit canonical polarity, range, order, and
   shape without per-frame min/max normalization.
6. Raw and base canonical identities do not change when VDPP is toggled.
7. Stabilized metadata separates semantic, partial-runtime, mutable state,
   payload, artifact, and metadata fingerprints and binds source canonical,
   scene manifest, source revision, checkpoint hash, and the complete effective
   execution plan.
8. Per-file and per-shot digests prevent changed stabilized bytes from being
   mistaken for complete output, and valid complete shots can resume only under
   the defined semantic/runtime rules.
9. Stereo accepts only a fully validated base or stabilized render-disparity
   producer and fingerprints the selected source.
10. The depth estimator and VDPP are never resident together by orchestration
     design. `StereoProjector` remains the synchronized owner on normal, cache,
     cancellation, initialization-failure, and repeated-finally paths.
11. CUDA incompatibility, artifact failure, and OOM preserve reusable base
    canonical output and never silently fall back.
12. Web and CLI expose only `off | vdpp`, identify VDPP as Experimental, and
    show a non-regressing optional progress step.
13. The pinned source license, source manifest, checkpoint size, and SHA-256 are
     shipped and verified as specified.
14. Settings schema v3 migrates known older jobs upward, distinguishes omitted
    resume values from explicit `off`, parses v3 strictly, and rejects future
    versions without modifying the job.
15. A complete hashed stabilized artifact can render without CUDA, VDPP import
    or checkpoint, or a loadable base estimator because planning precedes model
    loading.
16. One OS-backed writer lock covers authoritative audit through the final job
    mutation, and a concurrent writer fails rather than racing metadata.

## Implementation Slices

1. Add settings schema v3, one-way migration, omitted/explicit override
   semantics, CLI/Web control, and the dynamic default-off progress plan without
   constructing VDPP.
2. Add the output-directory writer lock and artifact-first planner, then move
   current Web/CLI model loading behind the authoritative locked audit.
3. Add the shared render-disparity validator, split-fingerprint stabilized
   metadata, immutable shot manifests, and payload hashes while keeping current
   canonical/stereo tests green.
4. Vendor the minimal pinned Apache-2.0 source subset and implement secure,
   lazy checkpoint resolution.
5. Implement the bounded 32/4 shot iterator and continuation-path CUDA preflight
   test-first against the pinned upstream reference.
6. Add the file-backed `TemporalDepthStabilizer`, shot-atomic resume, and cache
   invalidation/runtime compatibility matrix.
7. Insert the stage through the `StereoProjector` owner release callback after
   canonical preparation and before stereo, then verify lifetime, cancellation,
   cache-only, preflight, and OOM paths.
8. Update architecture, parameters, usage, installation, performance,
   troubleshooting, and third-party notices; run the real-checkpoint and
   product-quality release gates.

Each slice keeps `temporal_postprocessor=off` green. The model artifact is not
downloaded by unit tests.

## Rejected Alternatives

### RAFT or another optical-flow dependency

Rejected. VDPP's value is depth-only temporal refinement. Adding an older flow
model would add RGB decoding, another large checkpoint, occlusion/warping
contracts, and more failure modes without being required by the chosen model.

### Put VDPP inside each depth estimator

Rejected. It would duplicate integration and cache logic, keep two neural
models resident together, and make a generic post-process depend on backend
names. Canonical disparity is the existing shared boundary.

### Run VDPP on raw model output

Rejected. Raw outputs have backend-specific metric, inverse, and relative
semantics, ranges, invalid pixels, and native identities. The canonical stage
already solves those differences once.

### Copy the upstream demo literally

Rejected. Whole-video lists/tensors violate bounded memory. Per-frame min/max
destroys scene scale. The demo also has no scene-cut, cache, resume, artifact
verification, or project metadata contract.

### Load a complete shot and call upstream `infer_video_depth`

Rejected. It is convenient but memory grows linearly with shot length. The
bounded wrapper must instead prove equivalence to the same 32/4 algorithm.

### Expose window, overlap, strength, precision, or resolution controls

Rejected. The released checkpoint supplies one tested architecture and code
path. Tuning structural constants without training evidence creates cache and
quality combinations the project cannot validate.

### Auto-enable VDPP for DA3 or See-Through

Rejected. The upstream claim is broad, but this integration changes input
normalization and includes an untested anime domain. Default-on requires local
quality evidence, not a paper headline.

### Silently fall back to base canonical after VDPP failure

Rejected. It would render output different from the user's saved setting and
could make a failed experimental feature look successful. The valid canonical
stage remains available for an explicit off-mode resume.

### Add a global stabilized cache immediately

Rejected for v1. Local shot-aware resume solves interrupted work. A second
global PNG cache doubles storage and maintenance before reuse value is known.

## Documentation Changes Required With Implementation

- `docs/ARCHITECTURE.md`: optional derived disparity stage, source selection,
  artifact-first planning, writer lock, and model ownership/lifetime order.
- `docs/PARAMETERS.md`: the one setting, schema v3 boundary, default, CUDA
  requirement, and no tunable temporal parameters.
- `docs/USAGE.md`: Web/CLI omission/override examples, old-client limitation,
  shot-atomic resume cost, and explicit off-mode recovery after OOM.
- `docs/INSTALLATION.md`: lazy checkpoint location, size, source license, and
  hash verification.
- `docs/PERFORMANCE.md`: symbolic memory bound, fixed downsize plan, and no
  promise copied from upstream hardware.
- `docs/TROUBLESHOOTING.md`: CUDA preflight, corrupt checkpoint, cache/resume,
  and how to disable VDPP without recomputing depth.
- third-party notice and vendored source manifest: Apache-2.0 attribution,
  revision, included paths, and mechanical import changes.

## Non-Goals

- Optical flow, RAFT, RGB warping, or occlusion-mask generation.
- A generic temporal post-processor registry or plug-in API.
- Model-native temporal inference for DA3 or See-Through.
- Replacing or modifying V2's native 32/10 shot-aware inference.
- Training, fine-tuning, quantization, distillation, or checkpoint conversion.
- CPU or MPS VDPP generation in the first release; cached PNG rendering remains
  device-independent.
- User-configurable VDPP temporal or spatial execution parameters.
- Cross-shot temporal continuity.
- Window-level resume inside a shot in v1.
- A global stabilized-output cache in v1.
- Adding byte digests to the existing base canonical producer in this change.
- Changing `DepthBatch`, raw model storage, base canonical normalization, DIBR,
  or video encoding algorithms.
