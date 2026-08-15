# MoGe-2 Flat SBS Design

## Status

Approved on 2026-08-15. This design adds MoGe-2 as an optional depth backend
and adds an experimental metric-camera projection for flat, rectified
side-by-side video. Spherical, VR180, and point-map rendering are deliberately
separate future work.

## Linus Three-Question Check

### Is this a real problem?

Yes, with a narrow definition. MoGe-2 can provide sharper single-image metric
depth and an estimated pinhole focal length. Those outputs can improve
cross-shot scale consistency and make flat-SBS disparity follow a camera model
instead of an arbitrary per-scene percentage.

This does not prove that MoGe-2 will be temporally stable on video. The metric
path remains experimental until real clips show useful results.

### Is there a simpler solution?

Yes. A full XYZ point-map renderer is unnecessary for parallel, rectified SBS.
For a pure horizontal virtual-camera translation, metric depth `Z` and
horizontal focal length `fx` are sufficient. The implementation must reuse the
existing deterministic forward splat and change only how source validity,
z-order, and eye offsets are supplied.

### What can this break?

The largest risks are:

- changing the existing relative renderer while generalizing its inputs;
- invalidating valid version-2 raw-depth caches;
- silently falling back to another backend when MoGe is unavailable;
- presenting per-frame monocular estimates as physically calibrated video;
- allowing metric disparity to exceed a comfortable configured limit;
- coupling a render-setting change to unnecessary model reinference.

The design addresses these risks with byte-identical relative-render tests,
backward raw-schema reading, hard errors instead of fallbacks, an explicit
`Experimental` label, a disparity cap, and stage-specific fingerprints.

## Decision

Add one backend ID, `moge2`, with three model sizes. Keep the existing
`depth_model_version` setting name for compatibility. Add two explicit flat-SBS
geometry modes:

- `relative`: the existing scene-canonicalized projection and the global
  default;
- `metric_camera`: a new MoGe-2-capable projection based on metric depth,
  normalized focal length, virtual baseline, and convergence distance.

Do not make MoGe-2 or `metric_camera` the default. Do not persist or render
MoGe point maps or normal maps.

## Goals

- Expose MoGe-2 Small, Base, and Large through CLI, Web, settings, resume, and
  model reporting.
- Preserve exact existing behavior for V2, V3, See-Through, and relative SBS.
- Preserve MoGe-2 metric depth and normalized horizontal focal length through
  a typed inference and persistence contract.
- Generate physically motivated, bounded flat-SBS disparity from metric depth.
- Keep z-buffer ordering independent from disparity safety clamping.
- Reuse the existing deterministic, bounded-memory forward splat and
  background fill.
- Give preview and production the same geometry builder when convergence has
  been resolved.
- Pin upstream source and default weights to immutable revisions.
- Keep model, raw-depth, derived-geometry, and stereo invalidation boundaries
  explicit.

## Non-goals

- VR180, dome, equirectangular, or other spherical output.
- Six-degree-of-freedom novel views.
- Rendering or persisting XYZ point maps or normal maps.
- Synthesizing source content outside the original camera view.
- Claiming headset-IPD calibration or universally comfortable disparity.
- Temporal stabilization, optical-flow alignment, or learned video depth.
- Making metric-camera mode available to a backend that does not emit a
  validated pinhole focal length.
- Applying the metric pinhole claim to fisheye-distorted source or output.
- Shipping metric-camera mode in over-under or another stereo packing before a
  separate compatibility check.
- Optimizing MoGe batch size before real VRAM measurements exist.
- Replacing the existing scene-relative path.

## Upstream Contract and Pins

The integration uses `moge.model.v2.MoGeModel` from the official repository:

- repository: <https://github.com/microsoft/MoGe>
- pinned source commit:
  `925b8ed835a7a9cdb7578ba15c658a0afc969030`
- model API:
  <https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/model/v2.py>

The backend registry maps the existing size vocabulary to these immutable
Hugging Face artifacts:

| Size | Setting | Repository | Revision | Parameters |
| --- | --- | --- | --- | ---: |
| Small | `vits` | `Ruicheng/moge-2-vits-normal` | `679230677b4d282c6f304189a93e98e14f085902` | 35M |
| Base | `vitb` | `Ruicheng/moge-2-vitb-normal` | `54ad3a693e61907ea4633d13dec6ee682fa09419` | 104M |
| Large | `vitl` | `Ruicheng/moge-2-vitl` | `39c4d5e957afe587e04eec59dc2bcc3be5ecd968` | 326M |

`vitb` is the MoGe-2 default. The official Small and Base checkpoints contain
a normal head, but Depth Surge 3D discards normal output. Large uses the
depth-only checkpoint so it does not request an unused normal capability.

Default artifacts always resolve with the pinned revision. A custom remote
artifact must use an explicit `repo_id@revision`; a floating repository name is
an error. A local custom artifact must resolve to a `model.pt` file and uses its
SHA-256 as artifact identity.

## Dependency Policy

MoGe-2 is an optional project extra because the upstream package declares CLI
and demo dependencies that the other backends do not need. The project adds a
PEP 508 dependency pinned to the source commit under the `moge2` extra. The
supported install command is:

```text
uv sync --extra moge2
```

Pip-oriented documentation and a pinned `requirements-moge2.txt` provide the
same commit URL. The normal install does not pull MoGe's Gradio, trimesh, and
other optional-backend dependencies.

The Web option remains visible when the extra is absent, but selection fails
before model download or frame mutation and reports the exact installation
command. The application never substitutes DA3, V2, or See-Through.

MoGe code is MIT licensed. Its bundled DINOv2 code is Apache 2.0 licensed. The
installation and attribution documentation must retain both notices. No
upstream source is copied into this repository.

## Backend Registry

Create one Python registry for backend identity and construction instead of
adding a fourth copy of backend-specific `if/elif/else` dispatch. Each entry
declares:

- stable backend ID and display name;
- estimator factory;
- model-size aliases and default size;
- whether output is metric-capable;
- whether output includes a pinhole horizontal focal length;
- supported flat-SBS geometry modes.

The setting key remains `depth_model_version`; accepted values become `v2`,
`v3`, `see_through`, and `moge2`. Unknown values are validation errors. The
current behavior in which an unknown backend falls into the V3 branch must be
removed.

The registry is the Python source of truth for validation, model resolution,
factory selection, capability checks, and model reporting. HTML owns only its
presentation labels. Server-side validation remains authoritative.

## Inference Data Contract

Extend the existing `DepthBatch` with one typed, optional camera capability:

```python
@dataclass(frozen=True)
class PinholeCameraBatch:
    focal_x_normalized: np.ndarray  # float32 [N]


@dataclass(frozen=True)
class DepthBatch:
    values: np.ndarray
    representation: DepthRepresentation
    camera: PinholeCameraBatch | None = None
```

The existing contract remains:

- `values` is float32 `[N,H,W]`;
- `representation` is explicit;
- non-finite metric values are permitted at this boundary and become invalid
  downstream;
- existing estimators construct `DepthBatch` without a camera value.

`PinholeCameraBatch` validation requires:

- float32 shape `[N]` matching the depth batch;
- every focal value finite and greater than zero.

Do not add an opaque metadata dictionary. Do not add point maps, full
intrinsics, normal maps, or confidence fields.

MoGe-2 always returns `DepthRepresentation.METRIC_DEPTH`. The adapter invokes
the official mask application, so rejected pixels become non-finite metric
depth. A valid metric source pixel is therefore:

```text
isfinite(Z) and Z > 0 and isfinite(float32(1 / Z))
```

There is no second persisted copy of the model mask. Metric geometry derives
an explicit Boolean source-valid mask from this rule. The reciprocal is
evaluated only at initially valid pixels; reciprocal overflow invalidates the
pixel rather than storing infinity as a z-order value.

## MoGe-2 Adapter

Add a dedicated MoGe-2 estimator beside the existing depth adapters. It must:

1. Resolve the model-size alias to the pinned repository and revision, unless
   an explicit immutable custom artifact was supplied.
2. Resolve the Hugging Face snapshot through an extended
   `resolve_hf_snapshot(repo_id, revision=...)` API.
3. Pass the snapshot's concrete `model.pt` path to
   `MoGeModel.from_pretrained`; passing the snapshot directory is invalid for
   the upstream loader.
4. Move the model to the selected device and use evaluation/inference mode.
5. Convert input BGR uint8 frames to RGB float32 in `[0,1]`.
6. Preserve aspect ratio and downsample the longest edge to at most the
   selected `depth_resolution`. Never upscale a source frame for inference.
7. Use OpenCV area resampling for this output-affecting model preprocessing and
   fingerprint it as `moge2-rgb-area-max-edge-v1`.
8. Invoke `infer(force_projection=False, apply_mask=True,
   resolution_level=moge_resolution_level, use_fp16=...)`.
9. Extract metric depth and normalized `intrinsics[...,0,0]`; discard points,
   normal, and the remaining intrinsics immediately.
10. Validate keys, ranks, batch counts, spatial shape, depth dtype, and focal
    values before returning `DepthBatch`.

`force_projection=False` avoids recomputing a point map that this project does
not consume. It does not change the upstream depth or recovered intrinsics.

CUDA uses FP16 autocast unless the caller explicitly requests FP32. CPU uses
FP32. The adapter never silently changes device or precision. Device,
precision, source commit, weight revision, model size, depth resolution,
preprocessing identity, and resolution level all enter the model fingerprint.

The official API can accept a batch, but the first integration declares
`max_batch_size=1`. Existing DA3/V2 VRAM heuristics must not be applied to
MoGe-2. Later batching requires measured Small, Base, and Large memory data and
a separate change.

## User Settings

Add these typed settings:

| Setting | Default | Validation | Stage ownership |
| --- | --- | --- | --- |
| `stereo_geometry_mode` | `relative` | `relative`, `metric_camera` | stage 3 selection |
| `virtual_baseline_mm` | `63.0` | finite `0..100` | stereo |
| `metric_convergence_distance` | `auto` | `auto` or finite `0.1..1000` metres | stereo |
| `max_disparity_percent` | `2.0` | finite `0..5` | stereo |
| `moge_resolution_level` | `9` | integer `0..9` | raw model output |

`stereo_strength` and normalized `convergence` remain the relative-mode
controls. They are inactive in `metric_camera`. The metric settings remain
inactive in `relative`. Web controls hide inactive groups, and CLI startup
prints the effective geometry mode and its active parameters. No setting is
silently reinterpreted under another name.

`virtual_baseline_mm` deliberately does not use the name `ipd`: it defines a
virtual capture-camera baseline, not the viewer's physical pupil distance.

`metric_camera` requires `vr_format=side_by_side` and
`apply_distortion=false`. A post-stereo fisheye remap no longer represents the
pinhole camera described by MoGe's focal length. Web selects SBS, turns the
distortion control off when the user selects metric mode, and keeps both states
visible; server-side validation rejects forged or resumed metric requests that
use another packing or still enable distortion. Identical center crop and
uniform resize remain permitted because they transform both eye images
consistently.

The CLI adds:

- `--depth-model-version moge2`;
- `--model-size {vits,vitb,vitl}`;
- `--stereo-geometry-mode`;
- `--virtual-baseline-mm`;
- `--metric-convergence-distance`;
- `--max-disparity-percent`;
- `--moge-resolution-level`.

An explicitly supplied `--model-size` and `--model` are mutually exclusive at
the CLI boundary. Normalized effective settings persist the resolved
`model_size` plus repository/path and revision. A custom model records
`model_size: custom`.

## Flat-SBS Metric Projection

For one valid pixel, define:

```text
Z       = metric depth in metres
q       = 1 / Z
fx_n    = horizontal focal length normalized by input image width
b       = virtual_baseline_mm / 1000
Z0      = resolved metric convergence distance in metres
W       = per-eye render width in pixels
m       = max_disparity_percent / 100

p_raw   = fx_n * b * (q - 1 / Z0)
p       = clamp(p_raw, -m, +m)
D_px    = p * W

left_shift_px  = +D_px / 2
right_shift_px = -D_px / 2
```

`p_raw` and `p` are total binocular disparity as a fraction of one per-eye
frame width. For `Z < Z0`, `p` is positive and `u_left - u_right > 0`, so the
point appears in front of the zero-parallax plane. For `Z > Z0`, the sign is
negative and the point appears behind it.

The z-buffer near score is the unclamped `q`, never `p`. Clamping changes only
eye displacement. This prevents multiple close surfaces that hit the safety
limit from becoming false depth ties.

`max_disparity_percent=2.0` is a conservative project default derived from the
existing default total relative-disparity span. It is not described as a
universal comfort standard. Stereo metadata records the valid-pixel fraction
whose raw disparity was clamped, both per frame and for the complete stage.

## Automatic Convergence

An explicit `metric_convergence_distance` is used verbatim. `auto` resolves one
distance for the whole source-video identity; it never resolves independently
per scene or frame.

The metric-geometry stage reuses the deterministic depth-sampling policy:

- select at most 32 frame indexes per candidate scene, in source order;
- sample a uniform grid of at most 64 by 64 valid positive metric pixels from
  each selected frame;
- concatenate samples in source-frame order;
- select the float32 median metric depth as `Z0`.

The stage persists the selected frame indexes, sampling algorithm identity, and
resolved clip-global median. No valid sample is a hard error before stereo
rendering. A single clip-global value avoids scene-cut convergence pumping. It
does not claim to handle optical zoom or per-frame focal drift.

A pre-processing preview cannot know clip-global `auto` convergence. The
preview endpoint returns an explicit `convergence_unresolved` state until
metric geometry exists. The user may supply an explicit metric convergence for
an immediate preview. Once resolved metadata exists, preview and production
invoke the same geometry builder with the same `Z0`; there is no provisional
frame-median preview presented as final output.

## Common Renderer Geometry

Separate projection policy from the low-level splat with one immutable frame
input:

```python
@dataclass(frozen=True)
class StereoGeometryFrame:
    near_score: np.ndarray                 # float32 [H,W]
    total_disparity_fraction: np.ndarray   # float64 [H,W]
    source_valid: np.ndarray               # bool [H,W]
```

Two small builders produce it:

- relative builder: `near_score` is the existing canonical disparity,
  `total_disparity_fraction` is the existing relative formula, and every
  canonical source pixel is valid;
- metric builder: `near_score` is inverse metric depth,
  `total_disparity_fraction` is the clamped metric formula, and validity is
  explicit.

The renderer receives no backend ID and contains no MoGe branch. It uses
`near_score` for packed z-order keys, the signed disparity fraction for the two
eye offsets, and `source_valid` to exclude invalid source pixels before winner
selection. Offset construction remains host float64 and narrows to the existing
int32 fine-lane representation.

The generalized forward splat accepts any finite, nonnegative float32
`near_score`. Relative scores remain in `[0,1]`; inverse metric depth may exceed
one. Positive float32 bit ordering remains monotonic and can use the existing
packed int64 winner key. Invalid metric pixels store near score zero but are
excluded by `source_valid`, so they cannot win or fill a target.

The relative builder must reproduce the current offsets, z-keys, masks, and
eye pixels byte for byte on CPU. Generalization is not permission to change
relative interpolation, rounding, tie ordering, background fill, or banding.

## Metric Geometry Resizing

Metric geometry is persisted at MoGe's native output size and resized once at
the render boundary. Resize inverse depth without bleeding invalid zero values:

1. Bilinearly resize `inverse_depth * valid`.
2. Bilinearly resize `valid.astype(float32)` with the same geometry.
3. Divide the first result by the second where the resized weight is nonzero.
4. Mark a target source pixel valid when resized valid weight is at least
   `0.5`; otherwise set its near score and disparity to zero and exclude it.

Use `align_corners=False`, matching the existing canonical resize geometry.
Normalized `fx` does not change under aspect-preserving inference resize or
render resize. Any later fisheye transform, crop, or SBS assembly remains
downstream and unchanged for relative mode. Metric mode forbids the fisheye
transform; identical center crop, uniform resize, and SBS assembly remain
downstream and unchanged.

## Raw-Depth Schema Compatibility

Introduce raw-depth schema version 3 without rejecting valid version-2 data.
The reader supports versions 2 and 3; new writes use version 3.

Version 3 metadata adds `camera_model: none | pinhole_fx`. Per-frame NPZ
membership is exact:

- `camera_model=none`: `values.npy` only;
- `camera_model=pinhole_fx`: `values.npy` plus a zero-dimensional float32
  `focal_x_normalized.npy`.

MoGe-2 requires `pinhole_fx`. Version-2 raw data may be reused only for a
depth-only backend and is validated with the version-2 fingerprint and payload
rules. It is not rewritten merely because the application learned version 3.

Each version-3 payload is written through one temporary NPZ and one atomic
rename, so depth and focal length cannot be committed independently. Resume
validates archive membership, depth header, focal scalar dtype, finite positive
focal value, frame manifest, native shape, representation, model fingerprint,
and camera model before reuse.

The raw semantic fingerprint includes source commit, weight artifact, model
size, precision, device, resolution level, depth resolution, preprocessing
algorithm, metric representation, and camera model. A missing or corrupt focal
field is a hard raw-stage validation failure, not an invitation to infer a
default focal length.

## Stage 3 Metric Geometry

`metric_camera` derives a new restartable `03_metric_geometry` stage after the
raw-depth barrier. Each frame NPZ stores:

- float32 `inverse_depth`, with invalid locations set to zero;
- Boolean `valid` at native model resolution;
- zero-dimensional float32 `focal_x_normalized`.

Its atomic metadata records:

- schema and algorithm versions;
- ordered frame manifest and native shape;
- representation `metric_inverse_depth` and `near_value: larger`;
- source raw-depth fingerprint;
- clip-global deterministic sample manifest;
- resolved automatic convergence depth;
- storage dtype and compression.

The metric stage does not contain baseline, explicit convergence, disparity
cap, or render width. Those are stereo settings, so changing them regenerates
only stereo and downstream stages.

`relative` continues to derive and consume `03_disparity_maps`. The two stage-3
directories may coexist. Only the selected geometry mode's stage is required;
the pipeline does not eagerly generate both. An existing inactive stage is
preserved when its fingerprint remains valid.

The conservative disk preflight budgets metric geometry without assuming
compression: four bytes per inverse-depth pixel plus one byte per valid-mask
pixel, a 25 percent payload allowance, metadata, and one atomic frame overlap.
With `keep_intermediates=true`, raw and selected derived-stage allowances are
summed. With it false, raw payloads are removed only after the selected stage-3
payload and metadata validate.

## Cache, Resume, and Invalidation

Resume remains stage-specific:

- a valid source frame stage is never invalidated solely by adding MoGe-2;
- backend, size, repository, revision, precision, device, resolution level,
  depth resolution, preprocessing, or camera-model changes invalidate raw
  depth and every downstream stage;
- changing `stereo_geometry_mode` selects the required stage 3 and invalidates
  stereo and downstream output, but preserves any compatible raw and inactive
  stage-3 data;
- baseline, metric convergence, disparity cap, relative strength, relative
  convergence, or occlusion-fill changes invalidate stereo and downstream
  output only as appropriate to the selected mode;
- the metric geometry fingerprint must match raw depth exactly;
- clean, chunked, and resumed derivation of a given stage must produce the same
  payload values and metadata.

If `keep_intermediates=false`, the requested stage 3 is absent, and compatible
raw depth was already removed, the resume report explicitly states that MoGe
inference is required. It preserves source frames and all independently valid
stages. It never treats relative disparity as metric geometry or invents a
focal length.

Switching from one geometry mode to the other does not delete the previous
stage 3. Once both have been generated, future switches can reuse them while
their upstream fingerprints remain valid.

## Web and CLI Behavior

The Web depth-backend selector adds `MoGe-2`. When selected:

- the model-size control shows Small, Base, and Large with the exact upstream
  checkpoint family;
- the model-type control is hidden and effective depth type is metric;
- V2 temporal controls remain hidden;
- a numeric `moge_resolution_level` control appears;
- the flat-SBS geometry selector offers `relative` and experimental
  `metric_camera`;
- relative mode shows existing strength and normalized convergence controls;
- metric mode shows virtual baseline, metric convergence, and maximum total
  disparity controls, selects SBS, and disables fisheye distortion explicitly.

Web resume restores the persisted backend, size, geometry mode, and active
settings. Server validation rejects a forged `metric_camera` request for a
backend without `pinhole_fx`, regardless of client state.

CLI and Web report the effective repository, immutable revision, model size,
device, precision, depth resolution, resolution level, camera capability,
geometry mode, and active projection settings before expensive processing.

## Failure Behavior

Fail explicitly in these cases:

- the `moge2` extra is not installed;
- pinned source or weights cannot resolve online or from local cache;
- `model.pt` is absent or incompatible with `moge.model.v2`;
- model output omits depth, mask, or intrinsics;
- output rank, frame count, spatial shape, or dtype is invalid;
- normalized `fx` is non-finite or non-positive;
- no valid positive metric depth exists for automatic convergence;
- `metric_camera` is requested from a backend without pinhole focal output;
- `metric_camera` is requested with a packing other than `side_by_side`;
- `metric_camera` is requested while fisheye distortion is enabled;
- raw version-3 payload membership or focal data is invalid;
- metric geometry does not match its raw fingerprint;
- disk preflight fails;
- CUDA inference or rendering remains out of memory after existing renderer
  band retry behavior.

MoGe inference OOM reports model size, input dimensions, resolution level,
precision, and device. It does not reduce resolution, change model, switch to
CPU, or fall back to relative geometry automatically.

## Verification

### Unit tests

Add tests for:

- all four backend IDs, explicit unknown-ID rejection, and capability lookup;
- Small/Base/Large repository and immutable revision mapping;
- CLI model-size/model-override mutual exclusion and normalized effective
  settings;
- BGR-to-RGB conversion, no-upscale behavior, aspect-preserving area resize,
  and resolution-level forwarding;
- CUDA FP16 versus CPU FP32 selection without silent fallback;
- required MoGe output keys, shapes, metric representation, mask-to-invalid
  conversion, and normalized focal extraction;
- `DepthBatch` camera validation and existing no-camera construction;
- raw version-2 reuse, version-3 exact membership, atomic camera payload,
  corrupt focal rejection, and clean/chunk/resume identity;
- metric-geometry conversion, invalid mask, automatic convergence sampling,
  metadata, disk preflight, and retention;
- metric-mode rejection of fisheye output while preserving relative-mode
  packing and distortion behavior;
- zero disparity at `Z0`;
- positive near and negative far disparity with `u_left-u_right > 0` for a
  foreground point;
- proportional scaling with focal length and virtual baseline;
- total-disparity clamping at both signs;
- unclamped inverse-depth z-order when two surfaces share a clamped offset;
- invalid source exclusion from z-buffer, fill, and output coverage;
- mask-aware metric resize without invalid-zero bleeding;
- relative projection offsets, packed keys, masks, fill, and final CPU pixels
  byte-identical to the pre-change fixture;
- mode-specific settings and invalidation boundaries;
- preview unresolved-auto behavior and parity after convergence resolves;
- Web model controls, payload validation, resume restoration, and missing-extra
  error behavior.

### Integration tests

Mocked-model integration covers CLI and Web processing through raw depth,
selected stage 3, stereo, resume, and output assembly. Test both geometry modes
with the same synthetic MoGe result. Verify that changing only metric stereo
settings does not invoke the estimator or rebuild metric geometry.

Exercise `keep_intermediates=false`: after validated metric geometry, raw files
may be removed; a later missing-mode switch must report required reinference
while preserving source frames.

### Real-model release checks

These checks are explicit release verification, not ordinary CI downloads:

1. Load each pinned Small, Base, and Large model.
2. Infer one fixed image at a documented depth resolution and resolution level.
3. Verify a positive finite focal length and a nonempty set of finite positive
   metric-depth pixels.
4. Run fixed indoor-near, outdoor-far, and scene-cut clips through both
   `relative` and `metric_camera`.
5. Record model load time, inference time per frame, peak VRAM, native output
   shape, per-frame focal variation, static-ROI metric-depth variation,
   static-ROI output-disparity variation, hole fraction, and clamped-pixel
   fraction.
6. Inspect the A/B outputs for edge tearing, foreground sign, scale pumping,
   focal breathing, convergence placement, and viewing discomfort.

Performance results are observations, not portable thresholds. The project
must not claim that metric mode is better, physically calibrated, or temporally
stable merely because the formula and smoke tests pass.

## Acceptance Criteria

- `moge2` is selectable through CLI and Web with Small, Base, and Large.
- Default source code and weight identities are immutable and fingerprinted.
- Missing dependencies or artifacts fail before frame-stage mutation and never
  select another backend.
- MoGe returns typed metric depth plus validated normalized horizontal focal
  length.
- Existing raw version-2 data remains reusable for existing backends.
- Version-3 raw payload commits depth and focal length in one atomic file.
- `relative` output remains byte-identical on the fixed CPU regression corpus.
- `metric_camera` uses the documented depth/focal/baseline/convergence formula
  and standard foreground sign.
- `metric_camera` accepts only side-by-side packing on the rectilinear output
  path and rejects `apply_distortion=true`; relative packing and distortion
  behavior is unchanged.
- z-buffer ordering uses unclamped inverse depth; safety clamping changes only
  eye displacement.
- Invalid metric pixels never contribute to splat winners, fill sources, or
  valid output coverage.
- Automatic convergence is one deterministic clip-global value and is never
  silently approximated by a pre-processing preview.
- Configured total disparity is never exceeded, and clamped-pixel fractions
  are persisted.
- Metric stereo-setting changes reuse valid metric geometry and do not invoke
  MoGe.
- Resume never interprets relative canonical maps as metric geometry.
- Clean, chunked, and resumed processing produce identical selected stage-3
  values and stereo output on the same backend.
- Real release checks run all three pinned variants and produce an A/B report
  before `metric_camera` can lose its `Experimental` label.
- The complete existing and new CPU test suite passes; CUDA-specific tests pass
  on the available NVIDIA system.
- README, installation, parameters, architecture, example settings, changelog,
  and license attribution are updated with no claim of universal physical
  correctness.

## Implementation Slices

Implementation planning should preserve three independently verifiable slices:

### Slice 1: Backend and Raw Contract

Add the optional pinned dependency, backend registry, three model variants,
MoGe adapter, camera-extended `DepthBatch`, raw version-3 payloads, version-2
compatibility, fingerprints, CLI/Web model selection, and adapter/persistence
tests. Do not add metric rendering yet.

### Slice 2: Metric Geometry and Common Renderer Input

Add metric settings, `03_metric_geometry`, deterministic auto convergence,
`StereoGeometryFrame`, explicit source validity, metric projection, mode-aware
resume, and renderer regression tests. Prove relative output is byte-identical
before continuing.

### Slice 3: Product Surface and Release Evidence

Complete preview states, Web controls, CLI reporting, documentation,
attribution, end-to-end tests, three-variant smoke checks, A/B sample renders,
and performance/stability reporting. Keep `metric_camera` experimental.

Each slice receives its own test-first implementation plan tasks and commit so
model-loading, geometry, and product-surface regressions can be bisected.
