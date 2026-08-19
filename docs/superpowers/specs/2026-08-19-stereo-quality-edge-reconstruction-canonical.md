# Stereo Quality Edge Reconstruction Canonical Specification

## Status

This is the only implementation baseline for the stereo edge-reconstruction
change. It consolidates the chat-approved design and all completed PRO review
rounds. It is pending user review before implementation planning. No production
implementation is authorized by this document alone.

The earlier draft paths (the second now represented only in Git history)
`2026-08-19-stereo-quality-edge-reconstruction-design.md` and
`2026-08-19-stereo-quality-edge-reconstruction-revision-2.md` are superseded in
full. They are historical review records and must not be used to implement or
verify behavior. Existing Revision 5 behavior remains authoritative only where
this document explicitly retains it.

## Relationship to the Existing Renderer Contract

This specification extends and, where explicitly stated, supersedes:

- `docs/superpowers/specs/2026-08-14-stereo-edge-coverage-design.md`;
- `docs/superpowers/specs/2026-08-14-stereo-edge-coverage-revision-5.md`.

Revision 5 remains authoritative for the fixed 16-sample horizontal footprint,
packed strict z-buffer, deterministic source-index tie-break, balanced 16-lane
downsampling tree, complete-row banding, and the Fast pixel result.

The earlier prohibition on RGB-guided geometry reconstruction and a public
quality mode is superseded. It addressed an earlier serration defect. The
reported defect survives the 16-sample renderer and was isolated to cross-edge
low-resolution geometry interpolation plus unsafe disocclusion reconstruction.

## Reported Fixture and Root-cause Evidence

The local fixture root is:

```text
H:\3dtest\1787051840_f908a5f038277cf447b8a6a9b5072311_20260818_191720
```

The supplied SBS screenshot matches `frame_000089.png` with 95 SIFT/RANSAC
inliers. Its mapped source region is approximately `x=1291..3594,
y=51..563` in the SBS output. The screenshot is symptom evidence, not a
committed test asset.

The immutable diagnosis artifacts are:

| Artifact | SHA-256 |
|---|---|
| settings JSON | `13b66e657877c8227b100f10265114f1a88162faff4f6a950d8b2d7234d41fbe` |
| source frame 89 | `cca5b9dd367ab23d4931c73ec98b1d091c431c69385e5d1077608f4ec3fd060b` |
| raw depth frame 89 | `5096c6dd730eb05e6bd5f9777c4541e8a0fbf7c816844296e0434499e26be387` |
| canonical disparity frame 89 | `66d524dd394d82ad9b96c1fda89b725b1b9ff954a25b26fcd0ea0eed8138f77c` |
| Fast left eye frame 89 | `9f2b60cbc1aee589ed00afa6d854eaaa009ae708acc62534a47dddcb68473bfd` |
| Fast right eye frame 89 | `be77f8fc1f8552c4d8240350d1e1a5c6f1b7d89efca059b260eed5490426534a` |
| supplied screenshot | `364e7590cc3ce8c5b724ddf5b319fd1ac7caaee8b74bc17a2b1eef2c9100122` |

The fixture uses 1920x1080 source and per-eye output, 1080x608 relative stereo
geometry in width-height notation, `stereo_strength=1.5`, `convergence=0.5`,
and `occlusion_fill=background`.

Diagnosis established:

- current rerendering reproduces both saved stage-04 eyes byte-exactly;
- stage 99 is an exact concatenation of stage-04 eyes, so later transforms are
  not causal;
- bilinear geometry without fill has 12,171 left and 11,902 right full-hole
  pixels, with a seven-pixel maximum horizontal run;
- current fill changes 21,641 left and 22,214 right pixels, then reports no
  full holes;
- nearest geometry removes much of the soft fringe but creates steps and raises
  the true maximum hole width to 11 pixels;
- 8- and 10-pixel caps change no tested output under bilinear geometry, while a
  6-pixel cap exposes black cracks;
- the current fill repeats one horizontal boundary colour without a background
  barrier, and current frame writing discards coverage diagnostics.

The primary cause is therefore cross-edge bilinear geometry. Boundary-copy
fill is a secondary amplifier. Source antialiasing, motion blur, and raw model
misalignment remain irreducible inputs and must not be mislabeled as renderer
defects.

## Goals

- Preserve the current renderer as selectable, byte-compatible Fast mode.
- Add an offline-first Quality mode while retaining only two public modes.
- Prevent interpolation from constructing geometry between visibly separated
  foreground and background surfaces.
- Use RGB only to position a geometry-supported boundary; do not infer a new
  surface from line art or texture alone.
- Preserve smooth interpolation within one surface.
- Repair only actual uncovered fine lanes and never overwrite valid winners.
- Prevent every repair backend from copying across a source-depth region edge.
- Avoid repeated contaminated boundary colours through safe donors, local strip
  continuation, bounded exemplar synthesis, and explicit fallback accounting.
- Preserve coverage and repair diagnostics through generation, transactions,
  resume, and debugging.
- Reuse upstream source, depth, canonical disparity, and metric geometry when
  only stereo mode or Quality fill limit changes.
- Produce deterministic CPU/CUDA output independent of band height and I/O
  worker order.

## Non-goals

- Recovering a thin structure with no distinct evidence in input geometry.
- Treating arbitrary anime outlines as foreground geometry.
- Semantic segmentation, matting, learned view synthesis, neural inpainting,
  or a new third-party dependency.
- Optical-flow or sequential temporal stabilization in Quality v1.
- Changing model resolution, calibration, strength, convergence, camera
  equations, crop, distortion, upscaling, VR assembly, or encoding.
- Hiding artifacts by blurring colour or lowering stereo strength.
- Making the fixed 16-lane sample count user-configurable.

## Review Disposition

| Finding | Disposition |
|---|---|
| Full-frame repair conflicts with banded fine-grid rendering | Replace the one-pass Quality path with banded analysis, global compact planning, and banded final rendering. |
| Metric derived fields could lose their algebraic relationship | Snap primitive fields only, copy the exact Fast baseline outside edge bands, then rerun existing derivation and clamp formulas. |
| Exemplar behavior and complexity were open-ended | Define one pure planner, exact pyramid/update/score rules, fixed memory and evaluation budgets, and deterministic fallback. |
| Dense watershed markers prevent useful motion | Remove watershed. Use eroded or skeleton seeds plus an exact integer geodesic label solver. |
| Fine-lane components and pixel proxies were ambiguous | Bind every unresolved lane segment to a source-region ID and admit only pure single-region proxy pixels as donors. |
| Diagnostics had no transaction or resume contract | Add a separately versioned diagnostics stage, per-frame commit manifests, ordered consolidation, and explicit legacy state. |
| Quality memory, disk, and I/O were unbudgeted | Add mode-specific GPU/host constants, sparse-plan limits, disk preflight, and renderer plus full-pipeline gates. |
| Older settings and resume overrides were ambiguous | Migrate every saved schema v1-v4 to Fast and resolve omitted resume overrides separately from new-job defaults. |
| Six reviewed frames were not hash-bound | Add one complete seven-frame manifest and canonical manifest hash. |
| Statistics were not reproducible | Fix bit assignments, denominators, run definitions, percentile method, count units, and canonical JSON rules. |
| An incorrect RGB edge inside the band was untested | Add a multi-edge fixture with an exact selected-boundary expectation. |
| Segment records lost far-side intent and could overlap | Encode `far_side`, define canonical run reconstruction, and assert mask disjointness, OR equality, and popcount equality. |
| Exemplar pyramid did not define unique state transitions | Define all five per-level arrays plus exact reduction, initialization, eligibility, termination, and target-only upsampling. |
| A one-lane pure proxy could contaminate repairs | Separate `pure_proxy` context from fully covered, barrier-cleared `safe_donor` copy sources. |
| Pixel-level union lost fine-lane connectivity | Make segment records component nodes and union only exact horizontal or vertical lane adjacency. |
| Run width and interpolation arithmetic were ambiguous | Define fine-coordinate spans, canonical component keys, a scalar float64 interpolation oracle, and exact nearest-marker arithmetic. |
| Fallback range and exemplar ROI conflicted | Execute fallback in the full-frame planner with explicit per-core, component, and eye budgets. |
| Compact diagnostics conflicted with the public mask API | Keep a compact pipeline result and materialize the four legacy boolean arrays only in the public wrapper, with memory accounted. |
| Diagnostics regeneration could leave stale downstream output | Every diagnostics-triggered stereo rerender invalidates tracked downstream stages before writing. |
| Persisted summaries cannot prove lane-level provenance | Treat lane statistics as producer-attested and independently cross-checked during generation; resume verifies committed hashes only. |
| A global Quality default was unsafe on CPU | Make the resolved new-job default device-aware: gated CUDA defaults to Quality, CPU defaults to Fast; explicit CPU Quality remains supported and warned. |
| Multiple overlapping design files were unsafe | Consolidate every active rule into this canonical specification and mark all earlier drafts superseded. |
| Local strip continuation and equal-depth behavior were non-unique | Split equal-depth pre-fill runs at the exact fine-lane midpoint and define a complete scalar strip oracle with bounded enumeration. |
| `MemoryError` could change RGB under one identity | Preallocate fixed scratch before plan mutation and make every allocation failure fatal; only semantic budgets may select fallback. |
| Metric Quality received geometry after bilinear damage | Carry native `MetricGeometryFrame` primitives into a Quality-only renderer entry point and compute projection statistics after one-sided resampling. |
| Stage and per-frame fill limits were conflated | Fingerprint configured/scaled limits plus policy versions only for background Quality; store its geometry-dependent limits only in frame diagnostics. |
| Fast component counts had no graph | Mark Fast component counts explicitly unavailable and validate Fast lane/pixel counters without a Quality plan. |
| Diagnostics lacked a strict state machine and schemas | Define exact metadata, frame manifest, stats, JSONL, summary, hashing, building/complete, and aggregate-rebuild contracts. |
| Metric clamp sidecars were outside the transaction | Put new-render clamp statistics in committed frame stats and derive the legacy-compatible clamp summary from diagnostics. |
| Disk reserve omitted RGB transaction overlap | Replace the mask-only estimate with allocation-rounded final payload, aggregate, and one-frame atomic-overlap bounds. |
| Exemplar core ownership was ambiguous | Partition the target bbox into nonoverlapping half-open 384-pixel interiors; halos never own targets. |
| Horizontal union could wrap between rows | Require equal rows and consecutive columns, with a dedicated row-boundary fixture. |
| Fallback lookup had no capacity or complexity contract | Use a capped deterministic per-region implicit k-d index and add a high-fallback 4K stress gate. |
| Tiled geodesic verification had no equivalent algorithm | Require one global host indexed heap and remove the unsupported tiled-equivalence claim. |
| Cleanup could leave retained manifests pointing at deleted stereo RGB | Add a terminal `payload_pruned` diagnostics state which preserves authenticated summaries and final-video identity without claiming reusable frame payloads. |
| Legacy partial repair and mask-policy changes had no state matrix | Make any damaged legacy stage and `false -> true` mask transition redraw all frames; make `true -> false` a manifest-only migration. |
| Quality capacity did not separate active scratch from queued slots | Force Quality v1 to one lifecycle slot and define phase-specific active-render byte formulas under the 512 MiB cap. |
| The global Dijkstra queue was unbounded | Use an indexed binary heap with one uint32 entry per band pixel, dense int32 positions, no stale entries, and typed overflow failure. |
| CUDA OOM retry did not define host rollback | Reset all partial Pass A state or all partial Pass B output at row zero, while preserving only the explicitly immutable state. |
| Fixed 64 KiB JSON estimates could undercount variable arrays | Derive deterministic schema bounds from `H`, `W`, `N`, known strings, integer/float widths, and maximum histogram cardinality before mutation. |
| Strict JSON keys did not fix numeric representation | Assign every numeric field an integer or binary64 JSON type and require source-order scalar aggregation with exact empty-value rules. |
| Quality `occlusion_fill=none` claimed a nonexistent repair graph | Add explicit no-repair availability and nullable budget counters without constructing sparse repair state. |
| Local strip mapping mirrored directional texture | Change to a direction-preserving translated strip and add oriented-gradient plus asymmetric-glyph fixtures. |
| Device type caused semantically identical Quality cache misses | Exclude device type from Quality RGB identity and keep hardware only in non-semantic execution provenance. |
| A final video could be published before its resolved encoder identity became recoverable | Persist and fsync a strict `final_video_manifest.json` from the actual executed command before any prune transition; never infer missing evidence after restart. |
| Reclaim estimates could double-count hard links or logical file sizes | Count unique physical allocations per volume only when every hard link is authorized for deletion; otherwise count zero and revalidate before mutation. |
| Renderer settings and direct-call defaults did not expose a unique mode dispatch | Append keyword-only mode/limit fields to `StereoRenderSettings`, keep `StereoSplatSettings` Fast-only, and make public `settings=None` permanently Fast. |
| Dense `repair_bits` ownership and offset temporaries contradicted the host formula | Keep dense repair bits outside the immutable plan, reuse one planar analysis allocation after planning, and build one eye offset into a preallocated int32 map with row scratch. |
| A legal 7x7 donor could read unsafe pixels through Sobel | Require an original same-region safe 9x9 support and score only its central 7x7 gradients without patch-edge reflection. |
| Quality none lacked its own OOM rollback | Reset its partial RGB, compact diagnostics, histograms, counters, and device state from row zero while preserving immutable geometry, offset, and the earlier eye. |
| Exact heap/k-d loops may miss the Python performance gates | Require a Task 0 prototype and authorize a project-owned prebuilt C++/Torch extension, but no runtime JIT, dependency, semantic relaxation, or silent Fast fallback. |
| Fast's historical 24-byte and interim 28-byte slots omitted native geometry and decoder temporaries | Replace the scalar coefficient with relative/metric lifecycle bounds over `Q`, per-frame `G_i`, compressed input bytes, JSON, and every overlapping slot. |
| Damaged retained aggregates in a pruned job had no audit disposition | Report historical diagnostics as irrecoverably damaged while preserving a separately valid final video; never demote the pruned state to `building`. |
| Final-video input identity was coupled to diagnostics manifests and the wrong stage | Add a retained content-only encoding-input sequence manifest over the exact resolved 06/07 eye files or 99 VR files; mask/stats identity never participates. |
| Metric Quality had no unique fill-control type or stage-plan variant | Add `QualityStereoControls` and a four-way discriminated stage-plan union; never borrow relative or Fast-only settings. |
| Strict uint64 JSON fields could overflow only during consolidation | Derive and checked-multiply every frame/root counter bound before mutation, then use checked addition while aggregating. |
| Prune authorization persisted mutable stage keys rather than paths | Commit strict output-root-relative `PruneEntry` objects and resume deletion only from those versioned paths. |
| Final encoding referred to nonexistent output reserve and unspecified validation | Explicitly decline a compressed-video size guarantee, always encode to a sibling temporary, reserve manifests, and require one exact ffprobe/full-decode contract. |
| Target Sobel could consume barrier or provisional values outside its 7x7 patch | Score a target gradient only when its reflected 3x3 support is entirely processed non-barrier context. |
| Quality-none identity included an unused fill limit | Make all repair-limit identity and diagnostics fields null for Quality none, so a limit-only change preserves RGB and downstream stages. |
| Public integer validation could accept booleans | Reject Python/NumPy booleans before the `Integral` check and normalize accepted values to Python `int`. |

## Public Settings Contract

The processing settings schema advances from 4 to 5. An explicit Quality
payload contains:

```json
{
  "stereo_render_mode": "quality",
  "occlusion_fill_max_px": 8
}
```

`stereo_render_mode` accepts exactly `fast` and `quality`.
`occlusion_fill_max_px` accepts non-boolean `numbers.Integral` values 1 through
32, normalizes them to Python `int`, and controls only local Quality
continuation. Python and NumPy booleans are rejected before the `Integral`
test. It is a 1080p-equivalent distance. For render height `H`:

```text
safe_limit_px = max(1, floor(occlusion_fill_max_px * H / 1080 + 0.5))
```

The existing `occlusion_fill` setting remains exactly `none` or `background`:

- `none` skips reconstruction and retains Revision 5 black uncovered lanes;
- `background + fast` uses the current bounded boundary-copy implementation;
- `background + quality` uses the reconstruction contract below.

The removed `processing_mode` name remains rejected and is not an alias.

Saved schema-5 settings always contain both new fields. Omission is resolved
before persistence and is device-aware:

- a new job on a CUDA renderer resolves to `quality` only after all Quality
  release gates pass;
- a new job on CPU resolves to `fast`;
- an explicit `quality` choice on CPU is supported, emits a visible performance
  warning before rendering, and must pass the CPU correctness and resource
  gates below;
- every saved schema 1 through 4 migrates to `fast` and limit 8;
- a schema-5 resume with no override retains its persisted values.

There is no persisted `auto` mode. The resolver receives the selected renderer
device, writes the resolved two-mode value, and therefore never reinterprets a
saved job because hardware changed. The Web UI exposes a two-value Fast/Quality
segmented control and an advanced fill-limit control. CLI options are
`--stereo-render-mode {fast,quality}` and
`--occlusion-fill-max-px 1..32`, both with parser default `None` so omission is
distinguishable from override.

### Typed Renderer API

Append the two new fields to the existing public relative-render settings as
keyword-only fields, preserving its current positional constructor exactly:

```python
@dataclass(frozen=True)
class StereoRenderSettings:
    stereo_strength: float = 2.0
    convergence: float = 0.5
    occlusion_fill: Literal["none", "background"] = "background"
    stereo_render_mode: Literal["fast", "quality"] = field(
        default="fast", kw_only=True
    )
    occlusion_fill_max_px: int = field(default=8, kw_only=True)
```

Existing zero- through three-positional-argument construction is unchanged;
attempting to pass either new field positionally is a `TypeError`. Construction
validates the mode and applies this exact limit check even when Fast will ignore
the value:

```python
if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
    raise ValueError("occlusion_fill_max_px must be a non-boolean integer")
if not 1 <= int(value) <= 32:
    raise ValueError("occlusion_fill_max_px must be within 1..32")
```

`StereoSplatSettings` retains its existing two-field positional shape and is
explicitly Fast-only; it gains neither field.

The file pipeline uses one separate immutable control object:

```python
@dataclass(frozen=True)
class QualityStereoControls:
    occlusion_fill: Literal["none", "background"]
    occlusion_fill_max_px: int
```

It validates `occlusion_fill` against the two exact literals and applies the same
non-boolean integer normalization. Relative Quality derives it directly from
`StereoRenderSettings`; Metric Quality receives one explicitly from its stage
plan. It contains no relative strength/convergence or Fast shift field.

`StereoRenderer.render(frame, canonical, settings=None)` always constructs the
default settings above and therefore remains Fast on CPU and CUDA. Device-aware
default selection belongs only to CLI/Web job resolution, which passes an
explicit resolved settings object. `render()` dispatches on
`stereo_render_mode` before calling `build_relative_geometry`: Fast enters the
unchanged path, while Quality enters the new relative primitive/region compact
path and only the public wrapper materializes the four legacy masks.

`render_geometry(frame, StereoGeometryFrame, StereoSplatSettings)` remains a
public Fast-only API. It rejects any other settings type and never inspects a
mode field. Metric Quality is intentionally file-pipeline-internal through
`MetricStereoPrimitiveInput` and `render_metric_primitives_compact`; this version
adds no public metric-primitive renderer.

## Persistence and Identity Contract

The stereo RGB and diagnostics algorithm identities are:

```text
Fast RGB schema/algorithm: 1 / torch-horizontal-16x-zbuffer-v3
Quality RGB schema/algorithm: 2 / torch-horizontal-16x-rgb-geodesic-repair-v5
Diagnostics schema/algorithm: 1 / stereo-coverage-sidecar-v1
```

Fast output colour and legacy masks remain governed by Revision 5. Its
fingerprint retains the old render-setting shape and ignores the Quality-only
fill limit. Migrated legacy jobs may reuse an existing valid Fast v3 stage.
Resume replaces blanket legacy-schema invalidation with a semantic check for
that exact case.

Quality stage identity is constructed before rendering. It retains every
output-affecting current stage field, including geometry mode, ordered frame
names, render shape, `occlusion_fill`, encoding, upstream geometry identity,
relative or metric projection settings, Quality RGB algorithm identity, and
source RGB guide fingerprint. Its repair-policy fields are conditional but keep
one strict key set:

```text
Quality background:
    configured_limit_1080p = resolved integer
    scaled_safe_limit_px = resolved integer
    predicted_gap_policy = "max-four-neighbour-eye-shift-v1"
    local_limit_formula = "min-scaled-safe-predicted-plus2-v1"

Quality none:
    configured_limit_1080p = null
    scaled_safe_limit_px = null
    predicted_gap_policy = null
    local_limit_formula = null
```

Quality none never computes a local limit. The saved user setting remains
available as non-semantic job provenance but does not enter RGB or diagnostics
identity.

`renderer_device_type` is not a Quality semantic identity field: CPU and CUDA
must produce the same bytes and may reuse one another's valid Quality stage.
Fast retains its existing device-bearing metadata solely for v3 cache
compatibility. Hardware, driver, and library versions belong in benchmark or
runtime execution provenance outside the RGB and diagnostics fingerprints.

For background Quality, `scaled_safe_limit_px` is stage-constant because render
shape is stage-constant. Frame-dependent `predicted_gap_px` and `local_limit_px`
are not stage fields; both eyes record them in frame stats and the frame manifest
repeats those values as a transaction-level assertion. They are already
determined by the upstream geometry fingerprint, guide, settings, and policy
versions. Quality none neither computes nor records any of these limits.

Changing mode, background-Quality limit, or Quality identity invalidates stage
04 and tracked downstream stages, but not source, depth, canonical disparity,
stabilization, or metric geometry. A limit-only change invalidates neither Fast
nor Quality none and does not rewrite their diagnostics or downstream stages.

Legacy reused Fast output has no reconstructable lane diagnostics and reports
`legacy_fast_unavailable`; masks or counts must not be fabricated from final
RGB. Every newly rendered Fast or Quality frame carries diagnostics.

## End-to-end Architecture

Quality geometry is constructed once on the host. Each eye is then processed
independently and sequentially through three phases. Left-eye compact planning
state is released before right-eye analysis begins; only the completed left
output and diagnostics remain resident.

```text
Quality primitive geometry + source-region labels
    -> Pass A: banded visibility analysis
    -> global compact repair planning on host
    -> Pass B: banded visibility replay and final 16-lane reduction
    -> eye RGB + compact diagnostics
```

`occlusion_fill=none` needs no repair plan and therefore uses one banded
visibility pass. It still emits compact coverage diagnostics. Fast retains the
Revision 5 single-pass renderer exactly.

### Quality Eye-offset Builder

Quality never calls either current helper which returns both full-frame eye maps
and materializes full-frame float64 expressions. It allocates one contiguous
`int32[H,W]` output and one `float64[W]` row scratch, then invokes:

```python
def build_quality_eye_offsets_into(
    total_disparity_fraction: np.ndarray,
    *,
    eye: Literal["left", "right"],
    output_int32: np.ndarray,
    row_float64_scratch: np.ndarray,
) -> None:
    ...
```

Rows and columns are visited in ascending order. For each source binary64 value
`f`, the scalar oracle performs these separately rounded binary64 operations:

```text
shift = float64(f * float64(W))
shift = float64(shift * float64(0.5))
shift = float64(shift * float64(16.0))
left  = ceil(float64( shift - float64(0.5)))
right = ceil(float64(-shift - float64(0.5)))
```

Select only the requested eye, range-check before narrowing, and store int32.
A row-vectorized implementation is allowed only when it matches that scalar
oracle exactly. The map is immutable through the current eye's analysis/repair
passes and every CUDA OOM retry; it is never rebuilt after an OOM. Release it
only after that eye commits in memory, then reuse the same allocation for the
other eye. The row scratch is included in fixed runtime overhead. Host allocation
or construction failure is fatal rather than a band-height retry.

### Native File-pipeline Input Boundary

The current file pipeline expands metric primitives inside decoder workers and
copies completed arrays again in `StereoGeometryFrame.__post_init__`. Fast and
Quality must both enter before that unbudgeted construction. Decoder workers
load only source RGB and one owned native primitive object:

```python
@dataclass(frozen=True)
class RelativeStereoPrimitiveInput:
    encoded_canonical: np.ndarray  # owned contiguous uint16 [Gh,Gw]
    encoding_scale: np.float32

@dataclass(frozen=True)
class MetricStereoPrimitiveInput:
    metric: MetricGeometryFrame
    virtual_baseline_mm: np.float64
    convergence_distance_m: np.float64
    max_disparity_percent: np.float64
    retained_crop_width: int
```

`metric` supplies native float32 `inverse_depth`, bool `valid`, and float32
`focal_x_normalized`. The store loader allocates those two arrays exactly once,
validates them in place, marks them read-only, and uses an internal owned factory
which performs no `MetricGeometryFrame` copy. The public constructor retains its
existing defensive-copy behavior. Relative decode likewise returns the uint16
PNG allocation directly rather than casting it to float32 in a worker. The
render shape comes from the uint8 source frame.

The internal decode/render union becomes:

```text
RelativeStereoPrimitiveInput | MetricStereoPrimitiveInput
```

- relative Fast and Quality receive `RelativeStereoPrimitiveInput` and convert
  it only on the serial render thread;
- metric Fast constructs the exact current bilinear geometry on that serial
  thread through a no-copy owned builder, then calls the public-compatible Fast
  `render_geometry` core;
- metric Quality sends `MetricStereoPrimitiveInput` with no precomputed stats to
  `render_metric_primitives_compact(frame, primitives, controls)`, which performs
  region solving, one-sided primitive resampling, projection, clamping, and
  final stats in that order;
- `render_geometry(frame, StereoGeometryFrame, StereoSplatSettings)` is
  structurally Fast-only and rejects a `StereoRenderSettings` object;
- metric Quality never silently accepts a render-size `StereoGeometryFrame`.

The metric Quality entry point is exactly:

```python
def render_metric_primitives_compact(
    frame: np.ndarray,
    primitives: MetricStereoPrimitiveInput,
    controls: QualityStereoControls,
) -> CompactStereoRenderResult:
    ...
```

The file planner is a discriminated union, not a `geometry_mode` plus runtime
settings-type cross product:

```text
FastRelativePlan    {kind="fast_relative",    primitives=relative,
                     settings=StereoRenderSettings(stereo_render_mode="fast")}
QualityRelativePlan {kind="quality_relative", primitives=relative,
                     settings=StereoRenderSettings(stereo_render_mode="quality")}
FastMetricPlan      {kind="fast_metric",      primitives=metric,
                     settings=StereoSplatSettings}
QualityMetricPlan   {kind="quality_metric",   primitives=metric, controls}

StereoStagePlan = FastRelativePlan | QualityRelativePlan |
                  FastMetricPlan | QualityMetricPlan
```

Each variant owns one matching decode and render function; an impossible
variant/input/settings combination is rejected when the plan is built, not
inferred in the frame loop. Relative Quality derives its controls exactly once
from its validated settings at dispatch; no second stored controls object can
disagree. Metric Quality stores the only explicit `QualityStereoControls`.

The compact Quality result carries its final `MetricProjectionStats` to the
writer. `_DecodeFrame`, `_DecodedMessage`, `_WriteItem`, tests, and mocks update
their unions accordingly. This keeps the old public Fast entry point compatible
while making it impossible to implement only relative Quality correctly.

### Source-region Identity and Fine-lane Components

Source geometry regions replace a retained global graph over fine-grid winners,
which would conflict with banded rendering. Fine-grid connectivity is retained
only for sparse unresolved segment records.

Source regions are constructed globally before either eye renders. Every valid
fine-lane winner can recover its full-frame source index from the existing
packed visibility key, then gather its uint32 source-region ID. A repair donor
must carry the same region ID as the far boundary selected for that unresolved
lane run. This prevents crossing a source depth edge without retaining a full
eye fine grid or merging an unbounded winner graph across render bands.

Target repair components are global and use one segment record as one graph
node. Two nodes may union only when they have the same source-region ID and are
fine-grid adjacent:

- horizontal adjacency requires equal rows, `left_column + 1 == right_column`,
  bit 15 in the left record, and bit 0 in the right record; a row's last pixel
  never connects to the next row's first pixel;
- vertical adjacency between rows `y` and `y+1` requires a nonzero bitwise AND
  of their lane masks;
- records inside one output pixel union only when their masks contain adjacent
  lane bits, exactly
  `(((mask_a << 1) & mask_b) | ((mask_b << 1) & mask_a)) & 0xffff != 0`;
  separated masks do not union merely because region IDs match.

Connectivity is evaluated after all Pass A records are sorted, so components
cross band boundaries and may span the full image height without retaining a
dense fine grid.

## Primitive and Derived Geometry Contract

### Exact Fast Baseline First

Quality first invokes the existing Torch float32 bilinear helpers with
`align_corners=False` to build a complete Fast baseline. Every output outside a
classified edge band is copied directly from this baseline. Quality must not
reimplement ordinary bilinear coordinates in NumPy and then claim bit identity.

Only primitive resampling inside an edge band changes. Derived fields are
recomputed from the selected primitives in their existing evaluation order.

### Relative Dependency Order

The relative primitive is canonical near score `r`. The derived total disparity
fraction is recomputed after one-sided resampling:

```text
total_disparity_fraction = (float64(r) - float64(convergence))
                           * float64(stereo_strength) / 100
```

Outside edge bands, both `near_score` and `total_disparity_fraction` are copied
from the exact Fast baseline rather than recomputed.

### Metric Dependency Order

Metric Quality preserves this unique order:

```text
primitive valid_weight          = float32(valid)
primitive weighted_inverse      = float32(inverse_depth * valid_weight)
resample both primitives
resized_valid                   = resized_weight >= float32(0.5)
resized_inverse                 = resized_weighted_inverse / resized_weight
                                  where resized_weight > 0
resized_inverse[~resized_valid] = float32(0)
raw_output_fraction             = existing float64 pinhole formula
clamped_output_fraction         = existing max-disparity clamp
near_score                      = resized_inverse
total_disparity_fraction        = existing retained-crop conversion
source_valid                    = all true, preserving the existing
                                  infinite-background policy
clamp statistics                = recomputed from final resized_valid and
                                  raw/clamped fractions
```

Within an edge band, both primitive fields use the same selected-region mask and
the same bilinear spatial weights. Weights are renormalized only across corners
whose region ID matches the selected output region. Derived values are never
interpolated independently. A low-resolution validity change always blocks a
region-graph edge, even when both samples happen to clamp to the same visible
shift.

The metric tests compare Quality output outside edge bands to the exact Fast
baseline, verify the inverse-depth/displacement equation inside bands, and
independently recount clamp statistics.

### Exact One-sided Interpolation Oracle

Inside an edge band, production must match this scalar oracle. Let source shape
be `(SH, SW)`, render shape `(DH, DW)`, and output coordinate `(y, x)`. Every
operation below is IEEE-754 binary64, evaluated in the written order without
contraction or reassociation, until the explicit final cast:

```text
sx = (((float64(x) + 0.5) * float64(SW)) / float64(DW)) - 0.5
sy = (((float64(y) + 0.5) * float64(SH)) / float64(DH)) - 0.5
cx = min(max(sx, 0.0), float64(SW - 1))
cy = min(max(sy, 0.0), float64(SH - 1))
x0 = floor(cx); x1 = min(x0 + 1, SW - 1); wx = cx - float64(x0)
y0 = floor(cy); y1 = min(y0 + 1, SH - 1); wy = cy - float64(y0)
ox = 1.0 - wx; oy = 1.0 - wy
w00 = oy * ox; w01 = oy * wx; w10 = wy * ox; w11 = wy * wx
```

For selected region `r`, let `mij` be binary64 1.0 only when that corner's
canonical region ID is `r`, otherwise 0.0. Compute:

```text
rw00 = w00 * m00; rw01 = w01 * m01
rw10 = w10 * m10; rw11 = w11 * m11
retained = ((rw00 + rw01) + rw10) + rw11
numerator = (((float64(v00) * rw00) + (float64(v01) * rw01))
             + (float64(v10) * rw10)) + (float64(v11) * rw11)
result = float32(numerator / retained)
```

Relative geometry uses this once for its primitive. Metric geometry computes
the weights once, then applies the same already-quantized `rw00..rw11` values
and the same ordered expression separately to `valid_weight` and
`weighted_inverse`. It does not call a reduction helper, `einsum`, or a fused
kernel whose association differs from the oracle. Synthetic tests use an
independent scalar implementation and include values where one-ULP changes
alter the projected lane.

If `retained == 0.0`, compare matching-region low-resolution sample `(iy, ix)`
against the unclipped `(sy, sx)` using binary64 in this order:

```text
dx = sx - float64(ix)
dy = sy - float64(iy)
distance2 = (dx * dx) + (dy * dy)
```

Choose the lowest `distance2`, then lowest `iy`, then lowest `ix`. Production
must disable fused multiply-add for this comparison or use the scalar oracle
path. This fallback is rare and deliberately favors a unique numeric contract
over vectorized throughput.

## Deterministic RGB-guided Region Solver

Quality does not use marker-controlled watershed. It uses an integer geodesic
solver with sparse stable seeds.

### Low-resolution Region IDs

For every primitive sample, derive its final one-eye displacement in output
pixels through the selected relative or metric formula. Two four-neighbour
samples are disconnected when either condition holds:

- their metric validity differs;
- their absolute one-eye displacement difference is at least `1.0` output
  pixel.

Union every other neighbour pair. Visit low-resolution samples in row-major
order and examine neighbours in up then left order. The canonical component key
is the minimum row-major linear index among all members, independent of the
union-find root. Assign positive uint32 region IDs by ascending canonical key.
Zero is reserved for unknown and `0xffffffff` is reserved for no region.

### Mapping and Edge Band

Map low-resolution sample centre `(iy, ix)` to render coordinates with the
same half-pixel convention as `align_corners=False`. The formula uses binary64
and the written operation order:

```text
fy = (iy + 0.5) * render_height / geometry_height - 0.5
fx = (ix + 0.5) * render_width  / geometry_width  - 0.5
py = clamp(floor(fy + 0.5), 0, render_height - 1)
px = clamp(floor(fx + 0.5), 0, render_width  - 1)
```

Nearest-label upsampling supplies the initial full-resolution region map. A
render pixel `(y,x)` chooses the low-resolution label at
`clamp(floor(sy+0.5), 0, SH-1), clamp(floor(sx+0.5), 0, SW-1)`, where `(sy,sx)`
is the exact unclipped output-to-source coordinate from the interpolation
oracle. Thus no library-specific `nearest` versus `nearest-exact` behavior is
implicit. A mapped boundary is any four-neighbour pair with different
non-sentinel IDs. Its
four-connected dilation radius is:

```text
radius = ceil(max(render_width / geometry_width,
                  render_height / geometry_height)) + 1
```

Pixels outside the band are immutable seeds. Inside each connected band
component, erode each participating region by `radius` iterations with the
four-connected cross kernel and constant-zero border. If erosion removes a
connected region fragment completely, use the exact morphological skeleton of
that fragment as its seed. The skeleton is built by repeated cross-kernel
opening and erosion, accumulating `current AND NOT opened` until `current` is
empty. If numerical shape degeneracy still leaves no seed, retain the fragment's
lowest row-major pixel. Original low-resolution sample centres are not all
retained as markers.

### Integer Geodesic Assignment

Assign unknown band pixels by multi-source four-neighbour Dijkstra. For adjacent
uint8 BGR pixels `p` and `q`, traversal cost is the exact uint64 integer:

```text
256 + 8 * max(abs(Bp-Bq), abs(Gp-Gq), abs(Rp-Rq))
```

Production owns the complete host region, uint64 distance, owner/tie state, and
one global indexed binary heap for the render frame. It is not an
independent-tile algorithm and has no fixed-halo approximation. Implementations
may chunk array storage, but every relaxation still enters the same global heap
and follows the same priority order; no tile may finalize a pixel independently.

The queue contract is exact:

```text
heap_pixel:    uint32 [edge_band_pixel_count]
heap_position: int32  [H,W]
```

`heap_position=-1` means unseen/not queued and `-2` means settled; every other
value is that pixel's current heap slot. At most one live heap entry exists for
an unsettled band pixel, so maximum heap length is
`edge_band_pixel_count <= H*W`. Heap comparison reads distance, owner region
rank/ID, row, and column from the dense canonical arrays using the priority key
below. A lower-cost or equal-cost better-owner relaxation updates those arrays
and performs indexed insert or in-place decrease-key; stale duplicate pushes are
forbidden. Allocate the heap once after the band mask is known. Capacity
overflow or an inconsistent position is `QualityRegionQueueBudgetError`, which
fails the frame without changing mode or output policy. The phase-specific host
formula below includes the worst-case `H*W` heap plus its dense position array.

Neighbour visitation order is up, left, right, down. Priority order is total
cost, then descending region rank, then ascending region ID, row, and column.
Region rank is the maximum float32 `near_score` bit pattern in that source
region after positive-zero normalization. A lower cost always wins. This makes
a flat-colour band reduce to a deterministic geometry-distance partition while
a real RGB discontinuity raises the cost of crossing that discontinuity. No
Sobel confidence threshold or floating Lab conversion participates in label
selection.

For Dijkstra boundary ties, choose the greater region rank and then lower region
ID. Zero-weight interpolation follows the exact scalar nearest-marker oracle
above.

Quality requires a three-channel uint8 BGR guide. Other guide dtypes or shapes
raise an explicit error; Fast dtype behavior is unchanged.

### Worked Movement Contract

For the reported 1080-to-1920 horizontal scale, the scale factor is
`1.777...` and the band radius is 3. In a one-dimensional two-region fixture
with mapped geometry boundary `b`, fixed left and right markers begin outside
the inclusive corridor `[b-3, b+3]`. A clean full-channel BGR step placed at
`b+0`, `b+1`, `b+2`, or `b+3` must produce its region boundary at that exact
step. This proves that marker density does not collapse the permitted motion.

A second fixture places three parallel edges inside one band: a 32-level
foreground highlight at `b+1`, the 192-level foreground/background transition
at `b+2`, and a 16-level chromatic fringe at `b+3`. The selected region boundary
must be `b+2`. A scalar integer-geodesic oracle, independent of production queue
and union helpers, defines both fixtures.

## Two-pass Visibility Contract

### Temporary Pass A Winner Data

`forward_splat_band` keeps its packed strict z-buffer. Pass A additionally
decodes each valid packed key to a full-frame uint32 winner source index before
the packed buffer is released. Invalid lanes use `0xffffffff`. The band gathers
source-region IDs from the full-resolution Quality region map. Winner index and
region arrays exist only for the active GPU band.

Pass A reduces each band to output-resolution or sparse host records, copies
those records to their final host positions, and releases every fine-grid
tensor before advancing to the next band.

### Compact Analysis Representation

For one eye, Pass A retains exactly these dense arrays:

```text
coverage_count: uint8  [H, W]
pure_region_id: uint32 [H, W], 0xffffffff when absent or mixed
pure_bgr:       three independently owned uint8 [H,W] planes in B,G,R order,
                zero when pure_region_id is absent
```

`coverage_count` is the number of valid pre-fill lanes. A pixel is a pure proxy
for region `r` only when it has at least one valid lane and every valid lane's
winner belongs to `r`. Invalid lanes are ignored. Sum the valid lane colours in
ascending lane order using int32, divide each channel by `coverage_count`, and
round to nearest with ties to even. The planar layout is semantic storage, not a
temporary interleaved copy. A pixel containing winners from two regions is never
a pure proxy, even if one region owns 15 of 16 lanes.

`pure_proxy` is context only. A pixel is a `safe_donor` for region `r` only
when it is a pure proxy, `coverage_count == 16`, and every in-frame pixel in its
clipped Chebyshev neighbourhood of radius
`max(1, floor(H / 1080 + 0.5))` also has coverage 16 and pure region `r`. Thus a
one-lane or boundary-adjacent proxy can influence context scoring but can never
be copied by local, exemplar, or fallback repair. The fallback-index phase
evaluates this predicate while streaming source pixels and does not retain a
full mask. After that index is discarded, the exemplar phase may materialize a
temporary boolean safe-donor mask inside its reused 64 MiB repair arena. The
mask is never live with the fallback index and is not a fourth retained analysis
array.

Each maximal contiguous invalid sequence is a **pre-fill run** with inclusive
fine coordinates `[s,e]`. Inspect valid anchors `L=s-1` and `R=e+1` when they
exist. Unequal near scores assign the complete run to the farther anchor using
Revision 5 depth ordering. Equal near scores preserve the current per-lane
distance/tie rule by splitting into at most two **repair runs**:

```text
left repair run  = every j in [s,e] where (j - L) <= (R - j)
right repair run = every remaining j
```

Thus the left half has `ceil((e-s+1)/2)` lanes and the right half has the
remainder; an exact midpoint lane belongs left. Each nonempty repair run stores
the corresponding anchor's source-region ID and `far_side`. This is the only
equal-depth policy. If only `R` exists, the pre-fill run must touch the left
frame edge and becomes one right repair run; the symmetric rule applies to only
`L`. A row-wide run with no valid anchor is a typed no-donor error in background
mode.

Each repair run is split at output-pixel boundaries into records with this fixed
little-endian, unaligned 16-byte dtype:

```text
pixel_index: uint32
lane_mask:   uint16
region_id:   uint32
fill_bgr:    uint8[3]
backend:     uint8
far_side:    uint8
reserved:    uint8
```

Field byte offsets are exactly 0, 4, 6, 10, 13, 14, and 15 in the order shown;
the dtype asserts `itemsize == 16` at construction.

`lane_mask` bit `i` corresponds to fine lane `i`. Backend values are 0
unplanned, 1 local strip, 2 exemplar, and 3 safe fallback. `far_side` is 0 for
the left boundary and 1 for the right boundary. The reserved byte is zero and
participates in plan hashing. Records are ordered by pixel index, least set
lane, region ID, then far side. Multiple records may exist for one pixel when
its unresolved runs bind to different source regions.
An unplanned record initially has zero `fill_bgr` and backend 0. After the
deterministic fallback-precompute phase, a still-unplanned backend-0 record may
hold its provisional fallback colour in `fill_bgr`; Pass B never scatters a
backend-0 value. Exemplar success overwrites it and sets backend 2, while final
fallback retains it and sets backend 3.

Every record mask is nonzero and contiguous within its output pixel. Masks in
one pixel are pairwise disjoint. Convert a record to its inclusive full-row
fine-coordinate interval using output column and the least/greatest set bits.
Two sorted records reconstruct one repair run only when their intervals are
fine-grid consecutive in the same row and both `region_id` and `far_side`
match. Because construction splits only at output-pixel boundaries, two
consecutive pieces of one repair run occupy adjacent output pixels. The selected
boundary anchor is then uniquely derived as `run_start - 1` for left and
`run_end + 1` for right; it must be a valid lane. A one-sided candidate is legal
only when the opposite run end touches the frame edge.

Hole-run statistics continue to describe the original maximal pre-fill run,
before an equal-depth split. Repair planning, backend statistics, records, and
components operate on repair runs.

For each eye and pixel, the bitwise OR of all record masks must equal the Pass A
invalid mask, and the sum of record-mask popcounts must equal the invalid-lane
count. Pass B repeats both checks against replayed invalid masks. These two
checks, plus pairwise disjointness, prevent scatter order from deciding colour.

The complete Pass A record table is limited to 64 MiB per eye. Backend choice
does not change record count, so exceeding this cap raises
`QualityRepairBudgetError` immediately; fallback cannot pretend to make the
table smaller. Within a valid table, components are processed whole in
canonical component order. Exemplar-budget exhaustion routes remaining records
to deterministic backend 3 without changing their table representation. The
renderer never allocates an unbounded table or silently uses Fast.

The plan uses two independently releasable fixed arenas: 64 MiB for 16-byte
records and 64 MiB for graph work. With at most 4,194,304 records, the graph
arena is partitioned into at most 16 MiB uint32 union parents, 4 MiB uint8 union
ranks, 16 MiB uint32 member order, 16 MiB uint32 component keys/roots, and
12 MiB nonrecursive in-place sort/scan workspace. Records are heapsorted in
place before union, so there is no second record-order array. After path
compression, member order is heapsorted by canonical component key then record
order; component ranges are streamed rather than retained as a descriptor list.
Partitions may be released and reused only in that sequence. If any fixed
partition is insufficient, raise `QualityRepairBudgetError`; no heap allocation
or alternate graph is permitted. The host preflight below also guarantees
`16*H*W <= UINT32_MAX` before any uint32 fine index is constructed.

### Global Host Planning

Sort Pass A records and use each record as a sparse node. Union only the exact
fine-lane adjacencies defined above, including across render-band boundaries.
For a node, its canonical fine index is
`row * W * 16 + column * 16 + least_set_lane`. A component key is the minimum
canonical fine index among its member nodes, and component IDs are assigned by
ascending key, independent of union-find roots. This planning result is
independent of Pass A band height.

Components describe pre-fill connectivity and are not renumbered after local
repair. Local planning visits reconstructed runs within each component; exemplar
and fallback see only that component's still-unplanned records. A successful
local record may therefore bridge two residual subsets without changing their
component ID, bounding box, statistics identity, or budget order.

The planner has this pure interface:

```python
def plan_quality_repairs(
    analysis: QualityVisibilityAnalysis,
    *,
    render_shape: tuple[int, int],
    local_limit_px: int,
    budgets: QualityRepairBudgets,
) -> QualityRepairPlan:
    ...
```

It mutates no renderer tensor and performs no file I/O. The returned plan owns
the filled 16-byte segment table, deterministic scalar/histogram statistics, and
measured budget counters. It owns no dense `repair_bits` or output RGB array.

### Compact-map Materialization

Background planning creates no dense repair map. After every record has its
final backend and colour, no planner may read `pure_region_id` or `pure_bgr`
again. Release the region plane and the G/R colour planes, zero the independently
owned B plane, and reuse that exact contiguous `Q`-byte allocation as
`repair_bits`. Scan `coverage_count` and final records in canonical order to set
the fixed bits below; no additional dense allocation is permitted. The
`coverage_count` and derived `repair_bits` then become immutable compact
diagnostics. The B-plane ownership transfer, rather than allocator-dependent
free/reallocate behavior, is part of the host formula.

For Quality none there is no plan or `pure_bgr`: its one visibility pass writes
its own `coverage_count` and `repair_bits` output planes directly. A completed
eye therefore owns exactly RGB `3*Q` plus the two compact planes `2*Q` in either
fill mode.

### Pass B Replay

Pass B reruns the same banded packed visibility using the same immutable host
int32 eye-offset map. For each band it selects plan records by output-pixel
range. The union of their lane masks must equal Pass B invalid lanes exactly,
and every masked lane must still be invalid. Any mismatch is an internal
deterministic error.

Scatter each record's `fill_bgr` only into its masked invalid lanes. Keep every
valid winner colour unchanged. Execute the existing fixed balanced 16-lane
addition tree and ties-to-even output conversion in the original location, then
release the band. Pass B never uploads or materializes a full-frame fine grid.

This replay makes output independent of band height while retaining exact lane
coverage. One output pixel may use different background colours for different
unresolved lane runs without mixing their donor components.

### CUDA OOM Rollback

For a frame, let `h0` be the deterministic initial Quality band height and
`h1=max(1,floor(h0/2))`. The frame begins at `h0`; after the first CUDA OOM in
any eye/pass, every later CUDA pass for that frame uses `h1`. These are the only
attempted heights.

- Pass A OOM discards the current eye's complete partial dense analysis, every
  appended sparse record, Pass-A/run counters, and temporary device state. It
  preserves immutable geometry/source-region data, the current eye-offset map,
  and any already completed earlier eye. It restarts the current eye at row 0
  with `h1`.
- Pass B OOM preserves the immutable record plan, current eye-offset map,
  `coverage_count`, and derived `repair_bits`, but discards the current eye's
  complete partial RGB, Pass-B validation/scatter counters, and temporary device
  state. It restarts Pass B at row 0 with `h1`.
- Quality-none visibility OOM discards the current eye's complete partial RGB,
  `coverage_count`, `repair_bits`, hole-run histograms/counters, and temporary
  device state. It preserves immutable Quality geometry, source-region map,
  current eye-offset map, and any completed earlier eye, then restarts that eye
  at row 0 with `h1`. It is not interpreted as Pass A rollback and creates no
  records.
- An OOM while already using `h1` is fatal and reports `h0` and `h1`. It never
  resumes after the failed band, rebuilds a different repair plan, lowers an
  evaluation budget, changes fill quality, or switches to Fast.

Before any restart, synchronize and run the existing CUDA release routine.
Rollback tests inject OOM after every band position in Pass A, Pass B, and the
none visibility pass; duplicate records, retained counter increments, partially
retained RGB, or retained partial none diagnostics are errors.

## Bounded Repair Planner

### Local Limit and Strip Fill

The safe local limit retains the original 1080p-equivalent setting formula. The
predicted gap is based on the largest four-neighbour full-resolution one-eye
shift jump plus the existing two-pixel footprint guard:

```text
safe_limit_px      = max(1, floor(setting * H / 1080 + 0.5))
predicted_gap_px   = ceil(max_neighbour_abs_q_jump_px) + 2
local_limit_px     = min(safe_limit_px, predicted_gap_px)
```

Each repair run tries local fill exactly when
`run_lane_count <= 16 * local_limit_px`. For a run beginning at full-row fine
column `start_fine` with positive length `L`, statistics derive:

```text
first_pixel = floor(start_fine / 16)
last_pixel  = floor((start_fine + L - 1) / 16)
touched_pixel_span = last_pixel - first_pixel + 1
physical_width_px  = float64(L) / 16.0
```

The cap uses repair-run lane count, not touched-pixel span. The public pre-fill
run statistics use the same formulas on the unsplit pre-fill run.

#### Scalar Direction-preserving Local-strip Oracle

For repair run `[s,e]` in row `y`, let `dir=-1` and `anchor_fine=s-1` for a left
far side, or `dir=+1` and `anchor_fine=e+1` for right. Let
`boundary_col=floor(anchor_fine/16)`. Enumerate unique touched target columns
from the boundary into the hole:

```text
left far side:  first_pixel, first_pixel+1, ..., last_pixel
right far side: last_pixel, last_pixel-1, ..., first_pixel
```

Call their count `P`. This order is the direction from known background into the
hole. Target index `j` always receives donor index `j`; no reversal is permitted
later.

The actual boundary context is ordered spatially toward the hole:

```text
context_col[i] = boundary_col + (2-i)*dir, i=0,1,2
```

For a left far side this is `boundary-2, boundary-1, boundary`; for a right far
side it is the symmetric decreasing sequence. All three coordinates must be in
frame and be pure proxies of the repair run's region. Do not skip an intervening
non-proxy and do not shorten context. Otherwise local repair is ineligible.

The bounded horizontal search is:

```text
search_limit_px = max(1, floor(64 * H / 1080 + 0.5))
row_deltas      = [0, -1, +1, -2, +2]
```

Enumerate candidate slots by `offset=1..search_limit_px` outermost and the five
`row_deltas` innermost. Slot `(offset,k)` has row `cy=y+row_deltas[k]`, first
donor column `cx=boundary_col+dir*offset`, and ordinal `(offset-1)*5+k`. Its
candidate context and direction-preserving donor sequence are:

```text
candidate_context_col[i] = cx + (3-i)*dir, i=0,1,2
donor_col[j]              = cx - j*dir,     j=0..P-1
```

Thus donor coordinates advance in the same spatial direction as target
coordinates. Every slot consumes one of exactly
`5*search_limit_px` per-run evaluation slots, including an immediately rejected
out-of-frame slot; there is no pressure-dependent early termination.

A slot is eligible only when `cy`, all three candidate-context coordinates, and
all `P` donor coordinates are in frame; both contexts must be same-region pure
proxies and every donor must be a same-region `safe_donor`. Every donor must
also lie strictly on the far side:
`(donor_col[j] - boundary_col) * dir > 0`. This forbids crossing or repeating
the actual boundary. Define `C[i,c]` from
`pure_bgr[c][y,context_col[i]]` and `D[i,c]` from
`pure_bgr[c][cy,candidate_context_col[i]]`, channels in B,G,R order. Compute in
this exact integer order:

```text
bgr_l1 = int64(0)
for i = 0..2:
    for c = B,G,R:
        bgr_l1 += int64(abs(int32(C[i,c]) - int32(D[i,c])))

YC[i] = (29*int32(C[i,B]) + 150*int32(C[i,G])
         + 77*int32(C[i,R]) + 128) >> 8
YD[i] = (29*int32(D[i,B]) + 150*int32(D[i,G])
         + 77*int32(D[i,R]) + 128) >> 8

luma_first_difference_l1 = int64(0)
for i = 0..1:
    dc = int32(YC[i+1] - YC[i])
    dd = int32(YD[i+1] - YD[i])
    luma_first_difference_l1 += int64(abs(int32(dc - dd)))

score = int64(2) * bgr_l1 + luma_first_difference_l1
```

Choose the smallest `(score, ordinal)`. Fill target column `j` from the three
`pure_bgr[c][cy,donor_col[j]]` values; only that target column's repair-run lane
mask receives the colour. This is a translated, direction-preserving strip
rather than reflection, supplies distinct donor pixels, and never repeats the
boundary colour. If no slot is eligible, retain backend 0 and enter exemplar
repair. A standalone scalar implementation is the local-strip test oracle.

### Exemplar Input and ROI

The component repair function consumes a clipped pure proxy, safe-donor mask,
`pure_region_id`, target records, exact region ID, provisional target colours,
and `QualityRepairBudgets`. The full-frame planner computes the one provisional
fallback lookup before exemplar and later applies the stored result; the
component function never interprets absence from its
clipped ROI as absence from the frame. It returns a full-level colour only for
targets actually completed by exemplar iteration. Other targets remain backend
0 for the outer planner.

Let the component target bbox be half-open `[y0,y1) x [x0,x1)`. Its read/search
domain is that bbox expanded by 128 actual output pixels and clipped to frame.
Partition the unexpanded bbox into nonoverlapping half-open interiors with
origins `(y0+384*i, x0+384*j)` in row-major `(i,j)` order:

```text
interior_y = [origin_y, min(origin_y + 384, y1))
interior_x = [origin_x, min(origin_x + 384, x1))
```

The last row or column is simply partial. A core working ROI expands its unique
interior by 64 pixels on every side and clips to the component read/search
domain and frame, so it is at most 512x512. Target owner is the unique core
whose **interior** contains that target; halo overlap never affects ownership.

A nonowned target in a halo is read-only. If an earlier row-major owner has
completed it, it is processed context; otherwise it is a barrier for the
current core. The current core never copies to it or marks it processed. A
synthesized target, including earlier-core context, never becomes a donor.

### Exact Pyramid State

Use full, half, and quarter levels, omitting a coarser level when its shorter
working-ROI side would be below 32 pixels. Each level owns exactly:

```text
working_bgr:    uint8 [h,w,3]
donor_mask:     bool  [h,w]
target_mask:    bool  [h,w]
processed_mask: bool  [h,w]
barrier_mask:   bool  [h,w]
```

At full resolution:

- `working_bgr` is `pure_bgr` for same-region pure proxies, an already completed
  earlier-core colour for such component context, the planner-provided
  full-frame provisional safe-donor colour for current targets, and zero for
  barriers;
- `donor_mask` is true only for same-region `safe_donor` pixels;
- `target_mask` is true only for targets owned by the current core;
- `processed_mask` is true for all same-region pure proxies and completed
  earlier-core context, and false for current targets and barriers;
- `barrier_mask` is false only for same-region pure proxies, targets owned by
  the current core, and completed earlier-core context; later/noncompleted halo
  targets are barriers.

These invariants always hold: donor and target are disjoint, target and barrier
are disjoint, donor implies processed, and barrier implies not processed.

Build coarser levels bottom-up from each clipped set of existing 2x2 children.
For parent `p`:

```text
p.barrier = any(child.barrier)
p.target = (not p.barrier) and any(child.target)
p.donor = (not p.barrier) and all(child.donor)
p.processed = (not p.target) and (not p.barrier)
              and all(child.processed)
```

If `p.barrier`, set `p.working_bgr` to zero. Otherwise set it to the per-channel
integer mean of every existing child's `working_bgr`, using int32 accumulation
in child order top-left, top-right, bottom-left, bottom-right and round-to-
nearest ties-to-even. A cell containing both a target child and a barrier child
is therefore a barrier, not a coarse target; that target is repaired only at a
finer level. No synthetic padding is added.

Process levels coarsest to full. Before a finer level begins, overwrite only
its target working values whose parent is a non-barrier coarse target:

```text
fine.working_bgr[y,x] = coarse.working_bgr[floor(y/2), floor(x/2)]
```

Do not modify donor, proxy, completed-context, or barrier values, and do not
mark a finer target processed. If a coarse level ended with an unprocessed
target, its already-defined provisional working value is still replicated; the
finer target remains unprocessed and gets a fresh opportunity. No fallback is
committed merely because a coarse target was unfinished.

### Exact Patch Iteration

At each level, `processed_mask` is the known-context mask. Recompute the
four-neighbour frontier among `target_mask AND NOT processed_mask` after every
patch copy. A frontier target has at least one non-barrier, processed
four-neighbour. Its complete 7x7 target patch must lie inside both the frame and
working ROI and contain no barrier; patches are never clipped. Select the
eligible frontier centre with greatest processed sample count in that 7x7
patch; ties use row then column. Require at least eight processed samples.

Candidate donor centres are enumerated in row-major order. Their central 7x7
copy patch and its one-pixel Sobel border, a complete 9x9 support, must lie inside
the frame and working ROI. Every pixel in that 9x9 support must be an original
same-region `safe_donor`; provisional, completed target, proxy-only, or barrier
pixels make the candidate ineligible. For the target patch's known offsets,
calculate:

```text
Y = (29*B + 150*G + 77*R + 128) >> 8
Gx = [-1 0 1; -2 0 2; -1 0 1] applied in int32
Gy = transpose(Gx) applied in int32
score = 2 * sum(BGR_L1)
        + sum(abs(Gx_target-Gx_donor) + abs(Gy_target-Gy_donor))
```

Compute donor Sobel values from the 9x9 support and retain only gradients aligned
with its central 7x7 copy patch; donor patch boundaries never use reflection.
For each known central target offset, its BGR term always contributes. Its
gradient term contributes only when all coordinates in its target 3x3 Sobel
support are non-barrier and `processed_mask=true`. At a true working-ROI/frame
boundary, apply reflect-101 first and test the mapped coordinates; there is no
synthetic processed padding. Thus neither a barrier nor an unfinished target's
provisional colour outside the central 7x7 can affect the score. Use the aligned
donor gradient only for the same target-gradient-eligible offsets. Accumulate
the int64 score in row-major sample/channel order, BGR terms first and eligible
gradient terms second. Both contributing offset sets are target-defined and
therefore identical for every candidate in an iteration, so no division occurs.
Lowest score wins; a tie uses donor row then column.

Copy every currently unprocessed target pixel in the selected 7x7 patch from
the aligned donor patch and mark it processed. The update is immediately usable
as context but never changes `donor_mask`. A level ends when every target is
processed, no eligible frontier or candidate remains, 8,192 patch iterations
have run, or any applicable donor-evaluation budget is exhausted.

Budgets count every scored donor patch across all pyramid levels:

```text
per_core_evaluations      = 2,000,000
per_component_evaluations = 8,000,000
per_eye_evaluations       = 32,000,000
```

Components run by canonical component key, cores by row-major origin, levels
coarsest to full, and iterations in the order above. Before an iteration, `M`
is the minimum remaining core, component, and eye budget. If `N` row-major
candidates exceed `M`, score candidates at unique indexes `floor(k*N/M)` for
`k=0..M-1`. When `M == 0`, terminate before candidate selection. If a core
budget reaches zero, skip exemplar work for the rest of that core. If a
component budget reaches zero, skip its remaining cores. If the eye budget
reaches zero, skip all remaining components. Only a target processed at full
resolution receives backend 2; every skipped or unfinished full-level target
proceeds to outer fallback in canonical component/record order.

### Fallback and Memory Failure

After local planning and before exemplar work, the outer full-frame planner
indexes original safe donors by source-region ID for every still-backend-0
record. It computes and stores one provisional colour per such record. After
exemplar, every record still at backend 0 adopts that already stored colour as
backend 3. The nearest same-region safe donor must be within this
1080p-equivalent distance:

```text
fallback_limit_px = max(1, floor(256 * H / 1080 + 0.5))
```

Distance between output-pixel coordinates is exact integer squared Euclidean;
ties use donor row then column. The search sees the full analysis frame, not the
128-pixel exemplar expansion. Set backend 3. Pure-but-partial proxies and
synthesized pixels are ineligible. If no safe donor exists within the limit,
fail the frame explicitly rather than pulling foreground colour or leaving a
black hole.

The scalar fallback oracle enumerates every same-region safe donor inside the
clipped square search window in row-major order, rejects `distance2` above the
squared limit, and applies the distance/row/column key.

Production uses a deterministic per-region implicit two-dimensional k-d index:

- collect row-major uint32 pixel indexes only for safe donors whose region is
  referenced by at least one backend-0 record;
- partition each region's contiguous range in place, starting with row axis and
  alternating row/column by depth;
- choose the lower median by the total key `(axis coordinate, other coordinate,
  pixel_index)` using deterministic median-of-medians partitioning;
- child ranges are implicit and require no pointer per donor;
- the scalar query starts with no donor and `radius2=fallback_limit_px**2`. At
  each nonempty range it visits and counts the median node first, accepts it only
  when `distance2 <= radius2`, and keeps the smallest exact
  `(distance2, row, column)` key;
- let signed-int64 `delta = target_axis - node_axis`. The lower child is nearer
  when `delta <= 0`, otherwise the upper child is nearer. Query the nearer child,
  then query the farther child exactly when `delta*delta <= bound`, where `bound`
  is the smaller of `radius2` and the selected donor's distance, or `radius2`
  when none has been selected. Axis selection alternates exactly as in build.

The index must return the scalar-oracle donor exactly. It may occupy at most 48
MiB of the per-eye 64 MiB repair scratch arena, covering the uint32 donor array,
region descriptors, construction stack, and query stack. The remaining 16 MiB
stores one uint32 visited-node count for each queried backend-0 record; the
64 MiB record cap proves the query count cannot exceed 4,194,304. Compute total,
maximum, and scalar p95 in place before resetting this arena. Build complexity is
`O(N log N)` over indexed donors,
working memory is `O(N)`, expected query work is `O(log N)`, and exact worst case
remains `O(N)`; queries are never truncated because that would change output.
Diagnostics record indexed donor count, query count, total visited nodes,
maximum, and p95 visited nodes. Query p95 applies the exact scalar linear rule
below to the per-query visit counts and is `0.0` when there are no queries.
Wall-clock build time is benchmark telemetry, never hashed frame diagnostics.
The 4K fallback stress gate below constrains the practical case.

Before background-mode Pass A appends its first record, allocate the complete
64 MiB record, 64 MiB graph, and 64 MiB repair arenas. An OOM retry resets but
retains those arenas. Local fill uses no dynamic allocation. Fallback index
construction, its visited-count array, and all provisional queries use the first
phase of repair scratch. After every provisional colour and fallback diagnostic
scalar is stored, reset that same arena and use its complete 64 MiB for
safe-donor materialization and exemplar; the index is no longer queried. Index
and exemplar state are therefore never simultaneous allocations.

If either preallocation, index construction, or any later allocation raises
`MemoryError`, fail the frame and commit no RGB or diagnostics. Earlier planned
values are discarded. Runtime memory pressure never selects fallback and never
changes output under one RGB identity. Only the fixed core/component/eye
evaluation budgets may deterministically route records to backend 3.

If any residual record has no safe donor within the limit or required index
state exceeds its 48 MiB partition, propagate an actionable render error. There
is no unconstrained
OpenCV inpaint path and no silent Fast downgrade.

## Compact Coverage and Statistics

The file-production path uses an internal compact result and retains only:

```text
coverage_count: uint8 [H,W]
repair_bits:    uint8 [H,W]
```

The existing public `StereoRenderResult` API remains structurally unchanged: it
still owns concrete `left_valid_mask`, `right_valid_mask`, `left_hole_mask`, and
`right_hole_mask` NumPy boolean arrays. `StereoRenderer.render()` is a public
wrapper which materializes those four arrays. They are allocations, not
zero-copy views, and their four bytes per output pixel are included in the
public API memory gate. The frame generator calls an internal compact render
entry point and does not materialize them.

For Quality, public valid masks are `coverage_count > 0`. Under `none`, public
hole masks are `coverage_count == 0`; under successful `background`, they are
all false because any final unresolved lane is a render error. Fast retains its
existing exact mask computation. The fixed repair-bit mapping is:

| Bit | Hex | Meaning |
|---:|---:|---|
| 0 | `0x01` | pre-fill partial hole, coverage 1..15 |
| 1 | `0x02` | pre-fill full hole, coverage 0 |
| 2 | `0x04` | at least one lane locally filled |
| 3 | `0x08` | at least one lane remained after local fill |
| 4 | `0x10` | at least one lane exemplar filled |
| 5 | `0x20` | at least one lane fallback filled |
| 6 | `0x40` | at least one lane finally unresolved |
| 7 | `0x80` | reserved and always zero |

Bits 0 and 1 always describe pre-fill coverage. With `occlusion_fill=none`,
bits 2 through 5 remain zero and bit 6 is set exactly when coverage is below 16.
With background fill, bit 3 is set whenever a lane remains after the local
phase, including a run which bypassed local because it exceeded the cap. Newly
rendered Fast treats its existing boundary-copy fill as backend 1; bits 4 and 5
remain zero there.

All pixel counts and ratios are per eye and frame. A pixel ratio denominator is
exactly `H*W`. A horizontal pre-fill hole run is a maximal contiguous invalid
fine-lane sequence in one row. Report lane count, touched-pixel span from the
exact formula above, and physical width `lane_count / 16.0`. Maximum and p95 for
each measure operate over all nonempty runs, not one maximum per row. P95 uses
the exact scalar linear rule below; an empty run set reports integer maxima 0
and binary64 p95 values `0.0`.

Backend lane count is the number of lane bits assigned to that backend. Backend
pixel count is the number of pixels with at least one such bit. In Quality,
backend component count is the number of canonical sparse segment components
which used that backend at least once. Fast does not construct source regions,
segment records, or this graph, so all Fast backend component values are JSON
`null` with `availability="unavailable_fast_no_segment_graph"`; they are not
invented from coverage or final RGB.

`final_unresolved_lane_count` is the sum of unresolved lane bits after final
rendering. Repair bit 6 is set for a pixel exactly when its unresolved lane
count is nonzero. The public legacy `hole_mask` remains true only when all 16
lanes are finally unresolved.

Statistics contain only integers, finite nonnegative floats, strings, booleans,
lists, dictionaries, and JSON `null` at fields explicitly declared nullable by
the strict schema below. Normalize negative floating zero to positive zero.
Reject NaN and infinity. Canonical bytes are ASCII JSON with sorted keys,
`separators=(",", ":")`, `ensure_ascii=True`, and `allow_nan=False`; SHA-256 of
those bytes is the statistics fingerprint.

Lane-level counts are producer-attested, not independently reconstructable from
persisted PNG sidecars. For Quality background, one counter is accumulated from
final segment records and a separate Pass B counter from actual scatter masks;
lane totals, per-backend totals, mask OR, popcount, and final unresolved counts
must agree before commit. Quality none derives coverage/unresolved totals from
its one visibility pass and has no plan counter. Fast instead counts writes
inside the existing fill helper
and independently derives pre-fill, filled, and final lane totals from the
active band's pre/post validity arrays before releasing them. It never runs the
Quality planner for diagnostics. Resume verifies committed hashes and
PNG-derived pixel counts when masks exist, but does not claim to rederive lane
or component provenance. Documentation uses "committed diagnostics" rather than
"independently auditable lane plan."

## Diagnostics Stage and Transaction

Stereo RGB and diagnostics use the separate identities fixed in the Persistence
and Identity Contract. Fast RGB metadata and fingerprint remain byte-compatible
with existing v3. Every path below is relative to the job output root:

```text
04_stereo_diagnostics/
    metadata.json
    stereo_coverage_frames.jsonl
    stereo_coverage_summary.json
    frames/frame_000089/
        left_coverage.png
        left_repair.png
        right_coverage.png
        right_repair.png
        stats.json
        manifest.json
```

Coverage and repair PNGs are single-channel uint8 at render shape and exist only
when `keep_intermediates=true`. Per-frame `stats.json` and `manifest.json` exist
for every new Fast or Quality render through `building` and `complete`,
regardless of mask retention. A later successful `payload_pruned` transition is
the sole normal path allowed to remove them.

Diagnostics normalize the existing persisted `stereo_geometry_mode` value
`metric_camera` to the JSON token `metric`; `relative` remains `relative`. This
normalization is diagnostics-only and does not rename the existing RGB-stage
field or public setting.

### Canonical JSON and Hashing

All JSON objects reject missing or extra keys. Canonical JSON is the ASCII byte
encoding already defined for statistics. A 64-character lowercase hexadecimal
SHA-256 hashes the exact stated bytes. Every self-fingerprinted object computes
`fingerprint = SHA256(canonical_json(object without fingerprint))`.

For source-ordered frame names, define:

```text
ordered_frame_manifest_fingerprint = SHA256(canonical_json([
  {"frame_name": name, "manifest_sha256": SHA256(raw_manifest_bytes)},
  ...
]))
```

The JSONL bytes are each validated `stats.json` object re-encoded canonically,
followed by ASCII LF, in exact source order. Nonempty JSONL has a final LF; an
empty sequence is zero bytes. `frames_jsonl_sha256` hashes those exact bytes.

### Encoding Input Sequence Manifest

Final encoding never uses a diagnostics manifest or generic frame-stage
metadata as image-content identity. Before FFmpeg starts, resolve its actual
input files and stream the canonical representation for the future job-root
`encoding_input_manifest.json` with exactly:

```text
schema_version:    1
algorithm_version: "encoding-input-sequence-v1"
mode:              "direct_stereo" | "assembled_vr"
frame_names:       list[string]
left:              list[EncodingImageIdentity] | null
right:             list[EncodingImageIdentity] | null
frames:            list[EncodingImageIdentity] | null
fingerprint:       string
```

`EncodingImageIdentity` has exactly:

```text
relative_path: string
sha256:        string
byte_count:    integer
png_header:    {width: integer, height: integer, bit_depth: integer,
                color_type: integer}
```

Direct mode resolves the exact 06 cropped or 07 upscaled left/right sequences
selected by `VRFrameAssembler.resolve_vr_source_files()`. `left` and `right` are
non-null, have the same positive length and source-ordered frame names, and
`frames` is null. Assembled mode resolves the exact source-ordered
`99_vr_frames` sequence; `frames` is non-null and `left`/`right` are null. Every
path is output-root-relative POSIX form, resolves beneath the acquired job root,
and is opened as a regular file without following a symlink, junction, mount
point, or other reparse point.
The manifest contains only files FFmpeg reads as image inputs. Diagnostics masks,
stats, their manifests/fingerprints, mask-retention policy, mtimes, and semantic
stage fingerprints are forbidden.

`frame_names` are unique canonical stems `frame_<six-or-more-digits>` in
gap-free numeric order; each identity path's filename is exactly its stem plus
`.png`. For every selected parent, the complete `frame_*.png` directory listing
must equal the manifest list, so an unmanifested trailing image cannot be read by
image2. Both commands pass the manifest's first index as `-start_number` and its
length as `-frames:v`; assembled encoding gains these arguments rather than
depending on directory exhaustion.

Hash every complete raw PNG, not decoded pixels or a sampled/stat identity. Any
image byte, byte count, header, order, or relative-path change changes the
self-fingerprint. A diagnostics-only or mask true-to-false migration leaves it
unchanged. Semantic stage identity may be recorded only in separate execution
provenance which is not an input to this object or final-video identity.

Audio has no entry in this image manifest. The normalized FFmpeg audio path token
contains the complete raw audio/source-file SHA-256 and byte count actually used;
sampled or stat-only fingerprints are forbidden. Clip selection and audio options
remain authenticated by normalized executed arguments.

The encoder streams one image identity at a time, so it never materializes an
`O(N)` Python object tree. The pre-FFmpeg pass writes canonical bytes at the
beginning of the reserved temporary, records their length and raw SHA-256 plus
the self-fingerprint, then physically fills the rest of the schema bound. After
successful encoding it reopens and rehashes the same contained image paths,
streams the resulting canonical bytes over that same extent, and requires the
length, raw hash, and self-fingerprint to match the pre-FFmpeg values. It also
rehashes the audio source against its captured full identity. A changed file,
header, order, or identity aborts publication; a match truncates the temporary
to the canonical length and fsyncs it before publication.
The job writer lock protects project mutations throughout. The committed manifest
is retained as a final artifact, never intermediate cleanup input.

### Final Video Publication Manifest

Every successful final encode, regardless of `keep_intermediates`, retains
`final_video_manifest.json` at the job root. It has exactly:

```text
schema_version:                1
algorithm_version:             "final-video-publication-v2"
relative_path:                 string
sha256:                        string
byte_count:                    integer
input_stage_fingerprint:       string
resolved_encoding_arguments:   list[string]
encoding_settings_fingerprint: string
fingerprint:                   string
```

`resolved_encoding_arguments` is the normalized copy of the exact argument
vector passed to the successful FFmpeg process, including the actually selected
encoder after `auto` resolution, frame rate, clip bounds, audio policy/options,
filters, pixel format, VR format/resolution, and output options. Normalization
replaces every input/output path token with a deterministic token containing its
authenticated source, audio, encoding-input-manifest, or final relative-path
identity.
It does not include an absolute path, process ID, temporary UUID, wall time,
hardware probe output, or a newly re-resolved setting. The fingerprint is:

```text
encoding_settings_fingerprint = SHA256(canonical_json({
    "input_stage_fingerprint": input_stage_fingerprint,
    "resolved_encoding_arguments": resolved_encoding_arguments,
}))
```

`relative_path` follows output-root containment rules and rejects absolute or
`..` components. `input_stage_fingerprint` is exactly the validated
`EncodingInputSequenceManifest.fingerprint`; no diagnostics/RGB-stage manifest,
generic mtime metadata, or independently selected hash may substitute. An audio
path token likewise contains the complete raw authenticated audio/source identity
actually used. A settings fingerprint alone is never an input identity.

After resolving `auto`, the encoder constructs the runtime argument vector and
its normalized copy together before execution, verifies their positional
one-to-one mapping, and retains both unchanged with the running process. On
FFmpeg success and process exit, it revalidates every encoding input, then opens
the sibling temporary output without following a link. It applies the exact
container validator below, hashes and measures the file, fsyncs that open handle,
and closes it. Only then does it atomically replace the final path and durably
sync the final-output directory where supported. It reopens the final path
without following a link and verifies the same byte count and SHA-256. It next
atomically commits the revalidated canonical encoding-input manifest and then
the canonical self-fingerprinted final-video manifest from separately fsynced
job-root temporaries, syncing the job-root directory after each replacement.
Windows uses `FlushFileBuffers`; unsupported directory-sync operations are
recorded but do not permit skipping file flushes. The raw final-video manifest
SHA-256 plus its bound input-manifest fingerprint are the only evidence accepted
by a later prune.

Container validation is `final-video-container-validation-v1` and has one
implementation. Run `ffprobe -v error -count_frames -show_streams -show_format
-of json <sibling-temporary>` and reject malformed JSON or any probe/decode
error. Require exactly one video stream, no data/subtitle/attachment streams, the
resolved output width and height, and `pix_fmt=yuv420p`. The exact codec-name map
is `libx264 -> h264` and `libx265|hevc_nvenc -> hevc`; no other resolved encoder
token validates. Require reduced `avg_frame_rate` equal to the resolved output
rational and `nb_read_frames ==
EncodingInputSequenceManifest.frame_names.length`. If
`nb_read_frames` is unavailable or noninteger, validation fails rather than
substituting duration. When resolved audio policy is absent, require zero audio
streams. When it is present, pre-probe the authenticated audio source: require
exactly one AAC output audio stream iff that source exposes a selected usable
audio stream, otherwise zero. Finally run `ffmpeg -v error -xerror -i
<sibling-temporary> -map 0:v:0 -map 0:a? -f null -`; because all other stream
types were rejected, this completely decodes every accepted stream. Require exit
status zero with no error diagnostic. File existence or nonzero length alone is
never validation.

A stale, missing, malformed, or mismatching encoding-input or final-video
manifest never authenticates an existing video. In particular, a crash after
video publication but before final-video-manifest commit requires a new encode to
a temporary and atomic replacement; resume must not inspect the container or
re-resolve `auto` to guess provenance.
Both manifests are retained final artifacts and are never listed for
intermediate cleanup. Final-encoding disk preflight physically reserves their
schema-derived temporary allocations before starting FFmpeg.

### Metadata State Machine

`04_stereo_diagnostics/metadata.json` has exactly these keys:

```text
schema_version:                     1
algorithm_version:                  "stereo-coverage-sidecar-v1"
status:                             "building" | "complete" |
                                    "legacy_fast_unavailable" |
                                    "payload_pruned"
rgb_stage_fingerprint:              string
source_guide_fingerprint:           string
frame_names:                        list[string]  # unique source-order stems
render_shape:                       [H, W]
render_mode:                        "fast" | "quality"
geometry_mode:                      "relative" | "metric"
occlusion_fill:                     "none" | "background"
mask_payloads_enabled:              bool
stats_schema_version:               1
frame_manifest_schema_version:      1
ordered_frame_manifest_fingerprint: string | null
frames_jsonl_sha256:                string | null
summary_sha256:                     string | null
metric_clamp_summary_sha256:        string | null
pruned_from:                        null | PrunedFrom
final_video_manifest_sha256:        string | null
prune_entries:                      null | list[PruneEntry]
fingerprint:                        string
```

`PrunedFrom` has exactly `previous_status`,
`previous_diagnostics_fingerprint`,
`ordered_frame_manifest_fingerprint`, `frames_jsonl_sha256`, `summary_sha256`,
and nullable `metric_clamp_summary_sha256`. `previous_status` is `complete` or
`legacy_fast_unavailable`. `PruneEntry` has exactly:

```text
stage_key:                 string
relative_path:             string
operation:                 "delete_tree"
cleanup_contract_version: 1
```

`prune_entries` records the concrete paths approved by the read-only audit, in
pipeline order with no duplicates. Each path is canonical output-root-relative
POSIX form, contains no absolute prefix or `..`, and names a directory which was
inside the acquired job root at audit and revalidation. The diagnostics-frame
entry uses `stage_key="stereo_diagnostics_frames"` and
`relative_path="04_stereo_diagnostics/frames"`. Stage keys are display/audit
labels only; cleanup never resolves them through the current
`INTERMEDIATE_DIRS` mapping.

Resume supports the committed `cleanup_contract_version` or performs no deletion
and reports an actionable version error. It validates containment and a
non-following directory identity for every entry, then deletes only the persisted
relative paths. Mapping changes in a future release cannot redirect or omit a
historical authorization. Final video, both final manifests, retained JSONL/root
summary, and the diagnostics metadata root are forbidden prune targets.
An already-absent persisted path is an idempotent successful no-op. A symlink,
junction, mount point, or other reparse-point replacement at that path is never
followed or recursively deleted; it fails cleanup with an actionable identity
error and leaves every other unprocessed entry untouched. Recursive traversal
also never follows a descendant link or reparse point; it may unlink that entry
itself but cannot visit or delete its target.

State-dependent nullability is exact:

| Status | Ordered manifest | JSONL/summary | Current clamp hash | Prune fields |
|---|---|---|---|---|
| `building` | null | both null | null | all null |
| `complete` | non-null | both non-null | metric non-null, relative null | all null |
| `legacy_fast_unavailable` | hash of `[]` | both non-null | legacy metric non-null, relative null | all null |
| `payload_pruned` | null | both non-null | null | all non-null |

`payload_pruned` always has `mask_payloads_enabled=false`; the previous policy
is authenticated indirectly by `pruned_from.previous_diagnostics_fingerprint`.

Before any frame transaction, atomically write `building` metadata with the four
aggregate hashes null and all three prune fields null. After every frame
manifest validates, write JSONL, summary, and metric compatibility summary to
same-directory temporaries, replace them, then atomically write `complete`
metadata last with their hashes and prune fields null. Relative mode requires
`metric_clamp_summary_sha256=null`; metric mode requires the hash of
`04_left_frames/clamp_summary.json`.
The diagnostics stage fingerprint is the metadata `fingerprint`; because it is
computed without itself but over all aggregate hashes, JSONL, summary, ordered
manifests, and metric compatibility output participate without a recursive hash.

A valid `building` resume validates and reuses committed frame manifests,
rerenders only missing/invalid frames, and rebuilds aggregates. A valid
`complete` stage with missing or corrupt JSONL, root summary, or metric clamp
summary is atomically demoted to `building` and rebuilds those derived files
without rerendering RGB when every frame transaction remains valid. A bad frame
manifest or its payload requires that frame's stereo rerender.

An absent diagnostics root may become `legacy_fast_unavailable` only for a saved
schema 1 through 4 job whose complete Fast v3 RGB stage and, for metric, every
old per-frame clamp sidecar plus summary all validate. It writes no frame
directories, an empty JSONL, and a strict root summary with
`availability="legacy_fast_unavailable"`; ordered manifest hash is the hash of
canonical `[]`. Its complete-like metadata hashes those two aggregate files,
has `mask_payloads_enabled=false`, and has all prune fields null. A legacy metric
stage additionally hashes its validated old clamp summary; legacy relative uses
null. It never fabricates counts.

Any missing or corrupt RGB, metadata, old clamp sidecar, or clamp summary in a
legacy candidate forbids partial repair: set `P=N`, redraw the complete stereo
stage, and finish as ordinary `available` diagnostics. A legacy stage also sets
`P=N` when masks are requested because they cannot be reconstructed from RGB.
The current per-frame legacy repair behavior is not used after schema 5.

After final video encoding and both durable final manifests validate,
`keep_intermediates=false` first atomically transitions `complete` or
`legacy_fast_unavailable` metadata to `payload_pruned`. The new object copies
the prior hashes into `pruned_from`, retains the current JSONL and root summary
hashes, sets the current ordered-manifest and metric-compatibility hashes null,
records the raw `final_video_manifest.json` SHA-256 and ordered `prune_entries`,
and fingerprints that state. Only after this commit may cleanup delete every
listed intermediate stage, per-frame diagnostics directories/manifests, and the
metric compatibility summary. Extra old payloads inside a listed delete-tree
root may disappear when that exact entry resumes; an unlisted path is never
deleted. Their presence never makes them reusable.

Finalization order is exact: durably publish and validate the final video,
durably commit and validate `encoding_input_manifest.json`, durably commit and
validate `final_video_manifest.json`, commit `payload_pruned`, run authorized
cleanup from its persisted `prune_entries`, then mark the settings/runtime record
completed. A crash after the prune commit resumes cleanup and final status only;
it never reopens a render transaction or consults current stage mappings. A crash
after both manifests commit but before prune validates their canonical bytes,
fingerprints, normalized executed arguments, and video payload, then retries only
the prune transition. A crash before final-video-manifest commit must reencode
even if a complete MP4 or encoding-input manifest is present.
With `keep_intermediates=true`, finalization skips only prune and cleanup: the
encoding-input and publication manifests still commit before the
settings/runtime completion mark.

A valid `payload_pruned` state authenticates only metadata, retained JSONL/root
summary, the raw encoding-input/publication manifests, and the final video.
Historical statistics remain readable but no stereo or upstream intermediate
payload is reusable. Merely opening or auditing a completed job performs no
render. Any requested processing resume, mask generation, or invalid final-video
recovery sets `P=N` and starts a fresh `building` stage after preflight.

Audit never demotes `payload_pruned` to `building`. It reports the three facts
`final_video_valid`, `historical_diagnostics_valid`, and the constant
`stereo_payload_reusable=false` independently. `final_video_valid` requires the
video, encoding-input manifest, and publication manifest to validate as one
bound identity. When those remain valid but JSONL or root summary is
missing/corrupt, ordinary inspection performs no writes or render, preserves
final-video success, and
reports historical diagnostics as damaged and irrecoverable. A later explicit
request to process again, regenerate masks, or rebuild diagnostics follows the
matrix with `P=N`; deleted frame payload is never assumed available. Invalid
publication evidence similarly cannot be repaired from retained aggregates.

The mask-policy transition matrix is mandatory:

| Existing state | Requested policy | Action | `P` | Manifest-only `R` |
|---|---|---|---:|---:|
| no reusable current or legacy stage | either | create a new available diagnostics stage | `N` | 0 |
| `building`/`complete`, same policy | same | reuse valid transactions, redraw invalid | invalid frame count | 0 |
| `building`/`complete`, false | true | masks are unrecoverable; redraw every frame | `N` | 0 |
| `building`/`complete`, true | false | validate RGB/stats, rewrite all manifests without masks, rebuild aggregates, then delete masks | invalid frame count | valid frame count |
| intact legacy Fast | false | write `legacy_fast_unavailable` | 0 | 0 |
| legacy Fast | true, or any legacy damage | full upgrade to available diagnostics | `N` | 0 |
| `payload_pruned` | either | no payload reuse; full rebuild when processing is requested | `N` | 0 |

`P` always counts full frame transactions which encode RGB and stats; `R`
counts manifest-only mask-removal migrations and never includes a frame already
in `P`. A true-to-false migration does not invalidate downstream stages because
RGB and stats hashes are unchanged. Any `P>0` follows normal downstream
invalidation. These values feed progress, disk preflight, and execution before
any destructive mutation.

The stereo executor creates exactly `P+R` lifecycle work items: the source-order
`P` render items first and source-order `R` manifest-migration items second.
Completion advances once per committed item; aggregate consolidation is the
existing separate final phase and does not invent frame work. Only `P` items
enter decode/render/write capacity or set `repaired_outputs`; `R` items use the
transaction writer without decoding or invoking a renderer. This rule fixes the
progress denominator, Quality's one-slot permit use, and downstream invalidation
from the same audited matrix.

For each `R` frame, validate existing RGB/stats/masks, atomically replace its
manifest with null mask payloads, and only then treat the old masks as
unreferenced. Delete unreferenced masks after all `R` manifests commit and before
aggregate consolidation. A crash may leave extra masks, but never a manifest
which requires a deleted mask.

### Frame Statistics Schema

Each `stats.json` has exactly:

```text
schema_version:     1
frame_name:         string
render_mode:        "fast" | "quality"
geometry_mode:      "relative" | "metric"
occlusion_fill:     "none" | "background"
render_shape:       [H, W]
eyes:               {"left": EyeStats, "right": EyeStats}
metric_projection:  null | MetricProjectionStats
```

`MetricProjectionStats` has exactly `valid_pixel_count`,
`clamped_pixel_count`, and `clamped_fraction`. It is null in relative mode and
required in metric mode. `EyeStats` has exactly:

```text
pixel_count
state_pixel_counts
state_pixel_ratios
hole_runs
backend_lane_counts
backend_pixel_counts
backend_component_counts
final_unresolved_lane_count
quality_limits
quality_budgets
```

`state_pixel_counts` and `state_pixel_ratios` each have exactly
`prefill_partial`, `prefill_full`, `local_filled`, `post_local_residual`,
`exemplar_filled`, `fallback_filled`, and `final_unresolved`. Ratios use
`pixel_count`.

`hole_runs` has exactly `count`, `lane_count_histogram`,
`touched_pixel_span_histogram`, `lane_count_max`, `lane_count_p95`,
`touched_pixel_span_max`, `touched_pixel_span_p95`,
`physical_width_px_max`, and `physical_width_px_p95`. A histogram is an
ascending list of unique `[positive_value, positive_count]` pairs. Maxima and
p95 are independently derived from histograms; physical width derives from the
lane histogram divided by 16. Empty histograms use the zero rules above.

`backend_lane_counts` and `backend_pixel_counts` each have exactly `local`,
`exemplar`, and `fallback`. `backend_component_counts` has exactly
`availability`, `local`, `exemplar`, and `fallback`. Quality background
availability is `quality_segment_graph` and all three counts are integers. Fast
availability is `unavailable_fast_no_segment_graph`; Quality with
`occlusion_fill=none` uses
`unavailable_quality_none_no_repair_graph`. Both unavailable cases require all
three values to be null. Quality none never constructs segment records or a
component graph merely for diagnostics.

`quality_limits` is null for Fast and Quality none. For Quality background it
has exactly `max_neighbour_abs_q_jump_px`, `predicted_gap_px`, and
`local_limit_px`.
`quality_budgets` is null for Fast. For Quality it has exactly:

```text
availability
segment_record_count
segment_table_bytes
exemplar_evaluations
fallback_indexed_donor_count
fallback_query_count
fallback_visited_nodes_total
fallback_visited_nodes_max
fallback_visited_nodes_p95
```

Quality background uses `availability="quality_repair_plan"` and requires every
counter to have its numeric type below. Quality none uses
`availability="unavailable_quality_none_no_repair_plan"` and requires all eight
counters to be null, not fabricated zeros. Its backend lane and pixel counts are
ordinary integer zeros because no repair write occurred; only graph-derived
component and budget values are unavailable.

### Frame Manifest and Transaction

Each `manifest.json` has exactly:

```text
schema_version:                1
algorithm_version:             "stereo-coverage-sidecar-v1"
frame_name:                    string
render_mode:                   "fast" | "quality"
geometry_mode:                 "relative" | "metric"
occlusion_fill:                "none" | "background"
render_shape:                  [H, W]
rgb_stage_fingerprint:         string
mask_payloads_enabled:         bool
quality_limits:                null | {"left": QualityLimits,
                                       "right": QualityLimits}
payloads:                      PayloadMap
fingerprint:                   string
```

`QualityLimits` has exactly `max_neighbour_abs_q_jump_px`, `predicted_gap_px`,
and `local_limit_px`. `quality_limits` is null for Fast and Quality none. For
Quality background, its two objects must equal the corresponding `stats.json`
eye values exactly; this repetition lets the manifest assert the
geometry-dependent limits without placing a frame list in stage identity.

`PayloadMap` has exactly `left_rgb`, `right_rgb`, `left_coverage`,
`left_repair`, `right_coverage`, `right_repair`, and `stats`. RGB and stats
entries are required objects; mask entries are objects exactly when enabled and
otherwise null. Each object has exactly `relative_path`, `sha256`, and
`byte_count`; paths are output-root-relative POSIX paths and cannot contain
`..` or an absolute prefix.

For one frame, encode every payload to a temporary in its destination
directory. Replace left RGB, right RGB, enabled masks, and stats in that order,
then write the canonical manifest last. Metric projection statistics are inside
stats, so there is no new-render per-frame `04_left_frames/clamp_stats` file.
Any failure removes the manifest and every final payload in that transaction,
including both RGB images. Resume verifies manifest self-fingerprint, every raw
payload hash/byte count, exact stats schema, path containment, and PNG header.

Any frame rerender triggered by diagnostics sets the existing
`repaired_outputs` condition and invalidates every tracked downstream frame
stage before replacement begins, even if regenerated RGB later hashes equally.

### Root Summary Schema

`04_stereo_diagnostics/stereo_coverage_summary.json` has exactly:

```text
schema_version:                       1
algorithm_version:                    "stereo-coverage-sidecar-v1"
availability:                         "available" |
                                      "legacy_fast_unavailable"
frame_names:                          list[string]
frame_count:                          int
render_shape:                         [H, W]
render_mode:                          "fast" | "quality"
geometry_mode:                        "relative" | "metric"
occlusion_fill:                       "none" | "background"
ordered_frame_manifest_fingerprint:   string
frames_jsonl_sha256:                  string
eyes:                                 {"left": EyeAggregate,
                                       "right": EyeAggregate} | null
metric_projection:                    MetricProjectionAggregate | null
```

For available diagnostics, `EyeAggregate` has exactly `pixel_count`, summed
`state_pixel_counts`, recomputed `state_pixel_ratios`, merged `hole_runs`, summed
`backend_lane_counts`, summed `backend_pixel_counts`, summed-or-null
`backend_component_counts`, summed `final_unresolved_lane_count`,
`quality_limit_ranges`, and `quality_budget_totals`. Merged hole histograms are
sorted and all max/p95 fields are recomputed from them. Quality limit ranges
contain min/max for each of the three per-frame limit fields. Both Quality fields
are null for Fast. `quality_limit_ranges` is also null for Quality none; only
Quality background supplies its min/max object. Quality none uses the explicitly
unavailable budget object defined below. Component availability and nullability
must be the same in every frame.

The nested `state_pixel_counts`, `state_pixel_ratios`, `hole_runs`,
`backend_lane_counts`, `backend_pixel_counts`, and `backend_component_counts`
use exactly the same key sets as `EyeStats`. `quality_limit_ranges` is either
null or has exactly:

```text
max_neighbour_abs_q_jump_px_min
max_neighbour_abs_q_jump_px_max
predicted_gap_px_min
predicted_gap_px_max
local_limit_px_min
local_limit_px_max
```

`quality_budget_totals` is either null or has exactly:

```text
availability
segment_record_count                 # sum
segment_table_bytes_max              # maximum eye/frame value
exemplar_evaluations                 # sum
fallback_indexed_donor_count         # sum
fallback_query_count                 # sum
fallback_visited_nodes_total         # sum
fallback_visited_nodes_max           # global maximum
fallback_visited_nodes_frame_p95_max # max of per-frame p95 values
```

It is null for Fast. Quality background uses
`availability="quality_repair_plan"` and numeric totals. Quality none uses
`availability="unavailable_quality_none_no_repair_plan"` with every total null.

`MetricProjectionAggregate` is required only for metric mode and has exactly
`valid_pixel_count`, `clamped_pixel_count`, `clamped_fraction`, ordered
`clamped_fractions`, `affected_frame_count`, `mean_clamped_fraction`, and
`max_clamped_fraction`. Global `clamped_fraction` divides summed clamped by
summed valid; mean is the arithmetic mean of ordered per-frame fractions.

### Numeric JSON Types and Aggregation

JSON integer means a non-boolean integer in `0..2**64-1`. JSON binary64 means a
finite Python/IEEE-754 binary64 value serialized as a floating token, so zero is
`0.0`, never integer `0`; negative zero is normalized first. Strict parsing
rejects an integer token in a binary64 field and vice versa.

The exact field types are:

- JSON integers: dimensions, frame/pixel/run/count fields, every histogram
  value/count pair, `lane_count_max`, `touched_pixel_span_max`, non-null backend
  counts, `final_unresolved_lane_count`, `predicted_gap_px`, `local_limit_px`,
  all non-null budget counters except visited-node p95, `byte_count`,
  `valid_pixel_count`, `clamped_pixel_count`, and `affected_frame_count`;
- JSON binary64: all state ratios, `lane_count_p95`,
  `touched_pixel_span_p95`, both physical-width fields,
  `max_neighbour_abs_q_jump_px`, `fallback_visited_nodes_p95`, every clamp
  fraction, mean, maximum, and every float min/max range;
- nullable integers: unavailable backend component counts and unavailable
  Quality budget counters whose corresponding ordinary type is integer;
- nullable binary64: unavailable `fallback_visited_nodes_p95` and any explicitly
  nullable binary64 aggregate; no other numeric null is legal.

The same rules apply to aggregate fields: q-jump range endpoints and
`fallback_visited_nodes_frame_p95_max` are binary64; predicted/local range
endpoints and all summed/max counters are integers. Schema versions are positive
JSON integers. Strings, booleans, arrays, and objects accept only their declared
types.

Before any mutation, diagnostics cardinality preflight proves that every possible
integer is representable. Define:

```text
U64_MAX = 2**64 - 1
R_cap   = min(16*Q, QUALITY_RECORD_ARENA_BYTES // 16)
E_cap   = 32_000_000
D_cap   = Q
```

`R_cap` is the maximum per-eye record/query count, `E_cap` is the fixed per-eye
exemplar evaluation cap, and `D_cap` is the conservative maximum indexed donor
and visited-node count for one query. Per-eye/per-frame upper bounds are:

```text
pixel and backend-pixel counts             <= Q
lane, unresolved, run, histogram counts    <= 16*Q
component and segment-record counts        <= R_cap
segment-table bytes                         <= 16*R_cap
exemplar evaluations                        <= E_cap
fallback indexed donors                     <= D_cap
fallback queries                            <= R_cap
fallback visited-node total                 <= R_cap*D_cap
fallback visited-node maximum               <= D_cap
metric valid/clamped pixels                 <= Q
```

For every summed field in one root eye aggregate, use `N` times its applicable
per-frame bound. Metric root counts use `N*Q`; frame and affected-frame counts use
`N`. Histogram values use `16*W` for lane length and `W` for touched span, while
their merged counts use `N*16*Q`. File/video/manifest `byte_count` values are
checked directly from the nonnegative platform size and must also fit uint64.

The preflight first validates that `N`, `H`, `W`, `Q=H*W`, every `G_i`, every
`F_i`, and every `J_frame_i` are nonnegative uint64 values, deriving each product
with checked arithmetic rather than an unchecked language multiplication. Any
future cross-eye aggregate must use its mathematical factor, including checked
`2*N` when a field actually combines both eyes; the current root schema keeps
the two eye aggregates separate and therefore applies `N` to each.

All products and sums use `checked_u64_mul` and `checked_u64_add`; evaluation is
ordered left to right and an operand or result outside `0..U64_MAX` raises
`DiagnosticsCardinalityError(field, operands)` during read-only preflight. JSON
size calculation runs only after these proofs, so its 20-byte integer token is a
derived fact. Frame generation and source-order consolidation repeat checked
addition defensively. Hitting exactly `U64_MAX` is legal; overflow never reaches
Python-bigint serialization or a partially replaced aggregate.

All derived binary64 values use source-order scalar operations. Define
`add64(a,b)=float64(float64(a)+float64(b))`; a scalar sum starts at positive
`float64(0.0)` and applies `add64` once per source-ordered value. Ratios are
`float64(float64(integer_numerator)/float64(integer_denominator))`. Global clamp
fraction first sums integer counts and then divides. Arithmetic mean uses the
source-order scalar binary64 sum divided by binary64 count; an empty allowed
mean is `0.0`. Min/max compare normalized binary64 values in source order.
For the only schema ratios whose integer denominator may be zero, metric clamp
fractions with no valid pixels, the result is exactly `0.0` without performing a
division. State-pixel denominators are `H*W` and must be positive.

Linear p95 is defined without a library reduction. For total sample count `n>0`:

```text
rank   = float64(float64(0.95) * float64(n - 1))
lo     = floor(rank)
hi     = ceil(rank)
alpha  = float64(rank - float64(lo))
result = float64(float64(value_at_rank(lo))
                 + float64(alpha * float64(value_at_rank(hi)
                                             - value_at_rank(lo))))
```

`value_at_rank` walks the ascending histogram counts without expanding them.
For an empty histogram, integer maxima/counts are `0` and binary64 p95/physical
values are `0.0`. Physical-width maximum and p95 are the corresponding
binary64 lane measures divided by `float64(16.0)`. Vectorized, tree-reordered,
parallel, extended-precision, or fused reductions are forbidden for persisted
statistics.

Legacy-unavailable summary uses the real frame names/count/shape/modes and
hashes, with `eyes=null` and `metric_projection=null`. Duplicate, missing, or
out-of-order names fail consolidation.

For new metric renders, atomically derive the existing-compatible
`04_left_frames/clamp_summary.json` from `MetricProjectionAggregate`; the
orchestrator/runtime summary reads the same aggregate. Missing or corrupt
derived clamp summary is aggregate damage and is rebuildable without RGB. A
legacy reused Fast metric stage continues validating and trusting its existing
per-frame clamp sidecars and summary. Starting any schema-5 metric rerender
deletes its old `clamp_stats` directory and derived summary before writing, so
old and new completion rules never coexist.

The derived compatibility file retains exactly the current six keys:
`schema_version`, `frame_names`, `clamped_fractions`, `affected_frame_count`,
`mean_clamped_fraction`, and `max_clamped_fraction`. Their values are copied or
derived from the corresponding aggregate fields; the diagnostics-only summed
valid/clamped counts are not added to this legacy shape.

Writer threads never append JSONL or update root aggregates. Stereo invalidation
deletes `04_stereo_diagnostics`, both diagnostic aggregate files, and new-render
derived clamp summary only after preflight authorizes replacement. During
`building` or `complete`, intermediates-disabled runs create no masks but retain
stats/manifests and aggregates; intermediates-enabled runs retain every manifest
payload. No normal cleanup removes a currently manifest-recorded file. The only
exception is the committed `payload_pruned` transition, after which the old
manifests are no longer part of current state and only the paths in committed
`prune_entries` may be deleted.

## Settings Migration and Override Resolution

Every saved processing schema from 1 through 4 migrates to:

```json
{
  "stereo_render_mode": "fast",
  "occlusion_fill_max_px": 8
}
```

This direct rule matches the current non-incremental migration implementation
and prevents any old job from being reinterpreted through the new Quality
default. Schema 5 requires both fields.

CLI and Web resume must distinguish omission from an explicit override. The
resolver contract is:

```python
def resolve_stereo_render_mode(
    *,
    persisted: object,
    override: object,
    is_resume: bool,
    renderer_device: Literal["cpu", "cuda"],
    quality_default_gates_passed: bool,
) -> Literal["fast", "quality"]:
    ...
```

- new CUDA job, omitted override: `quality` only after the release gates pass,
  otherwise `fast`;
- new CPU job, omitted override: `fast`;
- schema-v1-v4 resume, omitted override: migrated `fast`;
- schema-v5 resume, omitted override: persisted value;
- any resume with explicit override: validated override.

`--stereo-render-mode` and `--occlusion-fill-max-px` therefore use parser
default `None`, like the temporal postprocessor override. The fill-limit
resolver follows the same omission rules with new-job/migration default 8.
Web resume loads the persisted values before rendering controls and submits an
override only when the user changes them.

The implementation keeps every omitted new job on Fast until all Quality
performance, memory, disk, and visual gates pass on the clean CUDA candidate.
The final gate commit changes only the CUDA omitted-job resolver to Quality.
CPU omission stays Fast. Explicit Quality remains available on either device
after correctness gates pass, with the CPU warning required above.

## Resource and Capacity Contract

### GPU

Fast retains:

```text
GPU_TEMP_BUDGET = 256 MiB
FAST_SPLAT_BYTES_PER_PIXEL = 1280
```

Quality uses the same temporary budget independently in Pass A and Pass B:

```text
QUALITY_SPLAT_BYTES_PER_PIXEL = 1536
```

The clean CUDA gate must show peak live allocation plus 25 percent headroom no
greater than 1,536 bytes per source pixel for forced 1080p and 4K bands. No
full-frame device winner, region, repair, or fine-grid buffer is allowed.

### Host Lifecycle

Let `Q=H*W`, let `G_i` be frame `i`'s native primitive pixel count and
`G=max_i(G_i)`, let
`D_hist=272*(W+1)` be two eyes' dense uint64 lane/span histogram accumulators,
and let `J_frame` and `J_root` be the maximum raw frame-transaction and root-file
JSON bounds derived by disk preflight.

Fast keeps its RGB bytes but replaces every scalar bytes-per-output-pixel slot
estimate. Before mutation, read and validate every source PNG and native
canonical/metric header. For frame `i`, let `G_i` be its native primitive pixel
count, `F_i` the sum of the exact compressed source-image and native-geometry
payload byte counts which a bounded decoder may retain, and `J_frame_i` its raw
transaction JSON bound. `G_i > Q` is legal for every backend and is charged by
the formulas; no benchmark-only `G<=Q` assumption exists.

Decoder workers hold a lifecycle permit, load the source plus native primitive
exactly once, and never construct full-resolution geometry. The serial render
thread uses explicit owned outputs and fixed reuse:

- relative decode owns `3*Q + 2*G_i`; construction may own source 3Q, encoded
  uint16 `2*G_i`, float32 native canonical `4*G_i`, and resized near-score 4Q together,
  giving `7*Q + 6*G_i`. It then releases native storage and constructs final
  13Q geometry without `StereoGeometryFrame.__post_init__` copies;
- metric decode owns `3*Q + 5*G_i`. Because invalid native inverse is canonical
  positive zero, the builder resizes it directly as weighted inverse, uses one
  `4*G_i` validity-weight plane, and retains at most two 4Q resize outputs. Its peak is
  `11*Q + 9*G_i`. It derives resized validity/inverse in place, releases native
  state, computes projection statistics before in-place clamp, and uses one 8Q
  float64 fraction output; final geometry is again 13Q with no constructor copy;
- first-eye render/second-eye offset construction remains at most 25Q and second
  render at most 22Q under the one-eye-map schedule;
- writer ownership is at most `23*Q + J_frame_i`: two RGB/compact eyes plus
  bounded encoded RGB/mask payloads and canonical transaction bytes.

Both builders must match the current Fast scalar/bilinear oracles byte for byte.
They construct internal owned `StereoGeometryFrame` values through a validating
no-copy factory; the public constructor keeps its copy contract. A decoder uses
a bounded streaming reader which retains no more than `F_i` compressed bytes;
any backend with additional full-array temporaries violates this contract.

Define exact per-frame lifecycle bounds:

```text
STEREO_HOST_BUDGET        = 512 MiB
FAST_SLOT_OVERHEAD        = 1 MiB
FAST_JSON_STREAM_BYTES    = 1 MiB

fast_relative_item[i] = max(
    3*Q + 2*G_i + F_i,       # decoder
    7*Q + 6*G_i,             # serial geometry construction
    25*Q,                    # render
    23*Q + J_frame_i,        # writer
)
fast_metric_item[i] = max(
    3*Q + 5*G_i + F_i,
    11*Q + 9*G_i,
    25*Q,
    23*Q + J_frame_i,
)
fast_slot_bytes = max(item[i] for the selected geometry mode)
                + FAST_SLOT_OVERHEAD
fast_active_scratch = D_hist + FAST_JSON_STREAM_BYTES
fast_capacity = min(2*workers,
                    floor((512 MiB - fast_active_scratch) / fast_slot_bytes))
require fast_active_scratch + fast_capacity*fast_slot_bytes <= 512 MiB
fast_consolidation_peak = D_hist + J_frame + J_root + 1 MiB + 16 MiB
require fast_consolidation_peak <= 512 MiB
```

The capacity charge deliberately repeats the single active construction peak in
every slot rather than relying on a smaller empirical overlap estimate. Before
enqueue, the main thread serializes dense histogram results into that slot's
`J_frame_i` reservation and releases the dense arrays; writer items never retain
both forms. Capacity below one raises the typed host-budget error before any
stage mutation.

Fast builds only one eye's int32 offset at a time with one float64 row scratch.
Relative Fast follows its existing near-score/settings operation order; metric
Fast follows its existing total-fraction order. Retain the fraction plane until
the second eye map is complete, release it before second-eye rendering, and never
overlap two eye maps.

Quality v1 deliberately forces `quality_capacity=1`. Its lifecycle permit is
held from decode through transaction write, so no second decoded/writing frame
overlaps the active renderer. This is an offline-quality choice, not an estimate
derived from the Fast floor-division formula. Parallel Quality frame lifecycle
slots require a later measured design revision.

Quality fixed allocations are:

```text
QUALITY_RECORD_ARENA_BYTES   = 64 MiB
QUALITY_GRAPH_ARENA_BYTES    = 64 MiB
QUALITY_REPAIR_ARENA_BYTES   = 64 MiB  # fallback phase, then exemplar phase
QUALITY_JSON_STREAM_BYTES    = 1 MiB
QUALITY_RUNTIME_OVERHEAD     = 16 MiB
```

The persistent render coefficient is exactly 20 bytes per output pixel: source
BGR 3, `near_score` 4, `total_disparity_fraction` 8, `source_valid` 1, and final
source-region ID 4. The indexed region solver adds uint64 distance 8, int32 heap
position 4, worst-case uint32 heap pixel 4, and four one-byte band/morphology
work masks per output pixel, plus one uint32 rank per possible native region.
Solver scratch is released before Pass A.

The 13 geometry bytes hold the exact Fast baseline while the region solver runs
and the final one-sided geometry afterward; both versions are never live
simultaneously.

Every Quality-owned host phase must satisfy its corresponding byte bound:

```text
quality_relative_decode_peak = max_i(3*Q + 2*G_i + F_i) + 16 MiB
quality_metric_decode_peak   = max_i(3*Q + 5*G_i + F_i) + 16 MiB
quality_lowres_region_peak  = 3*Q + 22*G + 16 MiB
quality_geometry_build_peak = 32*Q + 17*G + 16 MiB
quality_region_peak         = 40*Q + 13*G + 16 MiB
quality_planning_peak       = 37*Q + D_hist + 64 MiB + 64 MiB + 64 MiB + 16 MiB
quality_pass_b_peak         = 42*Q + D_hist + 64 MiB + 16 MiB
quality_none_visibility_peak = 34*Q + D_hist + 16 MiB
quality_writer_peak         = 23*Q + D_hist + J_frame + 16 MiB
quality_consolidation_peak  = D_hist + J_frame + J_root + 1 MiB + 16 MiB
quality_public_peak         = quality_pass_b_peak + 4*Q
quality_none_public_peak    = quality_none_visibility_peak + 4*Q

quality_active_peak = max(all applicable peaks)
require quality_active_peak <= 512 MiB
```

Only the applicable relative or metric decode row participates. The native
primitive load follows the same one-allocation/`F_i` decoder contract as Fast;
Quality capacity one prevents overlap with another lifecycle item.

The low-resolution region phase's `22*G` covers owned primitives, float64
displacement, uint32 union parent/canonical-key storage, uint8 rank, and final
region IDs with the documented in-place reuse. Union work is released before
the full-resolution solver.

The metric geometry-build bound's `17*G` includes owned native inverse/valid,
float32 valid-weight and weighted-inverse, plus low-resolution region IDs. Its
`32*Q` output
slots are exactly source BGR 3, region map 4, three float32 primitive/resample
planes 12, one float64 fraction plane 8, final float32 near-score 4, and final
bool validity 1. Buffers are reused in that sequence; a fourth primitive plane
or second fraction plane is forbidden. Relative geometry is bounded by the same
formula. Planning's
`37*Q` is persistent 20, one int32 eye-offset map 4, current dense analysis 8,
and the already completed eye RGB/compact diagnostics 5. Pass B adds the current
eye RGB/diagnostics 5, releases the 64 MiB graph and 64 MiB repair arenas, and
retains only the 64 MiB record arena. Writer's `23*Q` covers both
eye RGB/compact diagnostics plus the worst 1.25-times encoded image payloads;
`J_frame` covers variable JSON buffers. Quality none's `34*Q` is persistent 20,
one eye-offset map 4, and both completed eye RGB/compact results 10; it has no
dense repair analysis or record arena.

The planning coefficient does not omit a dense repair map: none exists while
donor planning reads the 8-byte analysis set. After planning, the B plane's
existing `Q`-byte allocation becomes `repair_bits` while the G/R and region
allocations are released, exactly as specified above. The one-row float64 offset
scratch is within `QUALITY_RUNTIME_OVERHEAD`; no second eye map or full-frame
float64 offset temporary is legal.

`D_hist` is exact: per eye, uint64 lane bins need `16*(W+1)*8` bytes and span
bins need `(W+1)*8` bytes. Serialization streams nonzero bins in ascending order
instead of constructing unbounded Python dictionaries.

Quality none allocates none of the three fixed repair-plan arenas, builds no
plan/analysis table, and uses only its one-pass visibility/public formulas.
Background Quality uses planning/Pass-B/public formulas as applicable.

Metric native inverse/valid plus low-resolution region IDs (`9*G`) remain live
through region solving and are released immediately after one-sided geometry
construction; relative input is smaller and uses the same conservative region
formula.

Canonical JSONL and aggregate generation reads one strict frame transaction at
a time and streams JSONL directly to its temporary and SHA-256. Histogram pairs
are emitted from the dense arrays without constructing Python pair dictionaries;
frame-name and metric-fraction lists are streamed in source order. No full JSONL
or variable-cardinality Python object tree may be materialized. At most one
`J_frame` input, one `J_root` canonical output, and the fixed 1 MiB serializer
buffer coexist during consolidation; that phase is included above.

At 4K with `G<=Q`, the fixed render/planning phases peak at about 501.7 MiB in
planning. The public wrapper's applicable Pass-B phase, including its four
concrete boolean masks, is about 444.9 MiB. Variable `J_frame`/`J_root` phases
are evaluated separately and can reject a pathologically large job rather than
invalidate this statement. The renderer constructs only one eye-offset map,
reuses all three fixed arenas between eyes, and releases left-eye planning state
before right-eye Pass A while retaining the completed left result.

Before allocating or mutating stage state, calculate every applicable phase and
raise `QualityHostBudgetError(phase, required_bytes, 512 MiB)` if any bound is
too large. Reject `16*Q > UINT32_MAX` through the same preflight before
evaluating record or heap indexes. The clean memory gate measures all
stereo-owned NumPy, Python, OpenCV, and allocator-resident memory and must remain
within both the formula
and 512 MiB. Arena or later allocation failure is fatal, never an alternate
output path.

### Disk and I/O

Preflight estimates the complete pending stage-04 transaction, not diagnostics
alone. Reuse the metric-stage allocation-unit query with its 64 KiB filesystem
unit fallback; 64 KiB is not a JSON-size assumption. Define
`A=max(4096, queried_or_fallback_unit)` and
`alloc(n)=ceil(n/A)*A`.

JSON bounds are schema-derived. `max_json_bytes(schema, cardinalities,
known_strings)` walks the exact sorted-key canonical schema and counts ASCII
punctuation plus token maxima: each integer is at most 20 bytes, each binary64
token at most 32 bytes, each hash string is exactly 66 bytes including quotes,
and every known string/path uses its actual canonical escaped byte length.
Nullable fields use the larger of their numeric/object token and `null`. The
checked uint64 cardinality contract runs first; a 20-byte integer is therefore a
proved schema maximum rather than an assumption.

A sparse histogram pair needs at most 44 bytes including its following comma:
two unsigned 20-byte integers plus brackets/comma. For one frame, the maximum
number of distinct positive lane lengths is:

```text
K_lane_frame = min(16*W, floor((isqrt(1 + 128*H*W) - 1) / 2))
```

because `K` distinct lengths require at least `K*(K+1)/2` invalid lanes and the
frame has at most `16*H*W`. Per-frame touched-span cardinality is at most `W`.
Merged root cardinalities are at most `16*W` and `W`. Metadata/root frame-name
lists use exact cardinality `N`; metric fraction lists use `N`. These bounds,
not observed data, are inputs to `max_json_bytes`:

```text
stats_raw[i]    = max_json_bytes(FrameStats, K_lane_frame, W, frame_name[i])
manifest_raw[i] = max_json_bytes(FrameManifest, known payload paths for i)
metadata_raw    = max_json_bytes(Metadata, N, all frame names,
                                 all concrete prune-entry strings)
summary_raw     = max_json_bytes(RootSummary, 16*W, W, N, all frame names)
clamp_raw       = max_json_bytes(ClampSummary, N, all frame names)  # metric only
jsonl_raw       = sum(stats_raw[i] + 1 for i in source order)
J_frame         = max({0}, {stats_raw[i] + manifest_raw[i] for all i})
J_root          = max(metadata_raw, summary_raw,
                      clamp_raw when metric else 0)
encoding_input_manifest_raw = max_json_bytes(
    EncodingInputSequenceManifest,
    N,
    every resolved image relative path and PNG header,
)
final_video_manifest_raw = max_json_bytes(
    FinalVideoManifest,
    len(resolved_encoding_arguments),
    every normalized argument and final relative path,
)
```

The preflight helper constructs and tests a maximum-token schema skeleton for
every formula; an encoded object exceeding its bound is an internal error before
replacement. Let `P_set` and `R_set` be the disjoint frame sets from the lifecycle
matrix, `P=len(P_set)`, `R=len(R_set)`, and `M` be 1 only when target manifests
retain masks. Then:

```text
rgb_bound  = alloc(ceil(1.25 * H * W * 3))
mask_bound = alloc(ceil(1.25 * H * W))

pending_rgb   = P * 2 * rgb_bound
pending_masks = P * M * 4 * mask_bound
pending_json  = sum(alloc(stats_raw[i]) for i in P_set)
              + sum(alloc(manifest_raw[i]) for i in (P_set union R_set))

root_final = alloc(metadata_raw) + alloc(jsonl_raw) + alloc(summary_raw)
           + (alloc(clamp_raw) when metric else 0)
root_atomic_reserve = 2 * root_final

frame_atomic[i] = 2*rgb_bound + M*4*mask_bound
                + alloc(stats_raw[i]) + alloc(manifest_raw[i])  # i in P_set
manifest_atomic[i] = alloc(manifest_raw[i])                     # i in R_set
one_transaction_overlap = max({0}, all frame_atomic, all manifest_atomic)

pending_file_count = P*(4 + 4*M) + R
root_file_count = 3 + (1 when metric else 0)
atomic_file_count = max((4 + 4*M) if P>0 else 0, 1 if R>0 else 0)
filesystem_slack = A * (pending_file_count + root_file_count
                        + atomic_file_count)

required_bytes = pending_rgb + pending_masks + pending_json
               + root_atomic_reserve + one_transaction_overlap
               + filesystem_slack
```

The later final-encoding preflight is a separate transaction and deliberately
makes no claim that free space can hold the complete CRF-compressed video. There
is no deterministic "existing video reserve". It guarantees only manifest and
directory-entry publication space. On the job-root volume it requires and
physically allocates/fsyncs before FFmpeg:

```text
alloc(encoding_input_manifest_raw)
+ alloc(final_video_manifest_raw)
+ 4*A
```

The two `alloc` terms are the future atomic manifest temporaries; an existing
retained file already consumes blocks and is never counted as free. The `4*A`
term is an additional non-sparse, fsynced reservation file, not an unprotected
free-space estimate. Before FFmpeg, each manifest temporary is non-sparsely
filled and fsynced to its schema-derived maximum. After input revalidation or
video hashing, canonical bytes are streamed over the beginning of that same
extent, checked not to exceed the bound, fsynced, and truncated. Immediately
before each of the two manifest replacements, the reservation is shortened by
exactly `2*A`; that replacement and its following directory sync complete before
the next release. The empty reservation entry is removed only after both
publications are durable. Thus publication needs no new payload
blocks from ambient free space. Actual normalized FFmpeg arguments and all input
paths are resolved before this check.

Both assembled and direct encoding write only to a sibling temporary on the
final-output volume. FFmpeg ENOSPC/Windows error 112, interruption, validation
failure, or any error before publication removes that temporary and manifest
temporaries and leaves the old final video plus both old retained manifests
unchanged. Only a successfully closed and validated temporary may enter the
publication sequence. Output-file size is reported from the failed temporary,
but it is never presented as a preflight guarantee. When job root and output
share a volume, the manifest reservation is still tested independently from the
unknown growing video temporary; reclaim credit is never used to promise video
completion.

The 1.25 factor matches metric geometry. Allocation rounding and explicit
filesystem slack are both intentionally retained. Root reserve is doubled
because existing aggregates and replacement temporaries coexist. Existing valid
files consume disk and are not treated as free. Bytes from an entirely
incompatible stage may count as `reclaimable_bytes` only after a read-only audit
has enumerated exact files authorized for deletion; valid per-frame payloads
being atomically replaced are never counted as reclaimable.

`reclaimable_bytes` means uniquely releasable physical allocation, never the sum
of logical path sizes. Compute a separate ledger for every target volume and
never transfer reclaim credit between volumes. The audit runs under the acquired
job writer lock and uses `lstat` or non-following handle opens:

- on POSIX, allocation identity is `(st_dev, st_ino)` and allocated bytes are
  `st_blocks * 512`;
- on Windows, allocation identity is `(volume_serial, file_id)` and bytes are
  the filesystem allocation size from the opened handle, not `st_size`;
- each allocation identity contributes at most once, and only when its reported
  hard-link count equals the number of directory entries with that identity in
  the authorized deletion set after a complete job-tree audit;
- a link count larger than the authorized count, an identity reachable from a
  retained path, an external/unknown link, or failure to obtain stable identity,
  link-count, or allocation-size data contributes zero;
- symlinks and Windows reparse points are never followed. Only their own link
  object allocation may count, under the same reliable-identity rule;
- filesystems with clone/reflink, deduplication, compression, or other shared
  extents contribute zero unless the platform exposes a reliable unique-release
  allocation value for that identity.

Immediately before the first metadata write, re-open and revalidate every
credited identity, allocation size, link count, and authorized path. Any change
invalidates the entire audit and reruns preflight; it never merely subtracts a
delta. Directories themselves receive zero reclaim credit. These rules make the
existing crop-stage `os.link` optimization conservative rather than double
counting its stage paths.

Preflight requires both `current_free >= alloc(metadata_raw) + A` for the first
building-metadata transaction and
`current_free + reclaimable_bytes >= required_bytes` for the complete plan.
Both inequalities are evaluated independently on every volume containing a
planned temporary or final payload.

The mutation order is mandatory:

```text
read-only audit existing state
derive lifecycle transition, P_set, R_set, and exact reclaimable paths
read every native geometry/image header and compressed payload byte count
checked-u64 diagnostics cardinality preflight
compute JSON/image bounds and Fast/Quality host phase bounds
disk preflight using current_free + reclaimable_bytes
revalidate every credited physical allocation; restart audit on any change
then and only then:
    write building metadata
    invalidate downstream when P > 0
    delete audited incompatible payload/clamp sidecars
    render or migrate manifests
    consolidate aggregates
```

`_prepare_stereo_stage`, downstream reset, clamp deletion, and any payload
replacement must move after preflight. Insufficient space leaves the old stage
untouched and reports required, current free, and audited reclaimable bytes.
Re-evaluate if `P_set`, `R_set`, target mask policy, shape, names, geometry mode,
or allocation unit changes. A later ENOSPC/Windows error 112 reports the
persisted estimate, current free bytes, and failing path. With masks disabled,
stats/manifests remain through `complete`; final successful cleanup follows the
`payload_pruned` state instead of leaving dangling manifests.

Benchmarks include PNG encode/write time and actual diagnostic bytes. Fast
renderer-only p95 may regress at most 5 percent. Fast full-pipeline p95 with
intermediates disabled may regress at most 5 percent; with diagnostic masks
enabled it may regress at most 25 percent. Quality is gated below.

## Hash-bound Seven-frame Fixture

The canonical settings-payload hash is
`861ba59c027f57c62b94460b23906d6ddcb7c0fde50c96e002f72ce9544180da`.
The source-stage fingerprint is
`29ccafdddc83b6c408708b29c41e5d2fabc392028d7d4149a11bdb9b9fbc60c3`,
the source-frame fingerprint is
`af573cc60b900ab279466c11e53c9e64daca173b1818e30509d562e8485d78e4`,
and the canonical-stage fingerprint is
`bfc9dca3f7a4ee61816df04e97b1fc56b89d54a536cb5b434862883d0a7ec7fa`.

Every source is uint8 `[1080,1920,3]`; every canonical PNG is uint16
`[608,1080]`.

| Frame | Source SHA-256 | Canonical SHA-256 |
|---:|---|---|
| 89 | `cca5b9dd367ab23d4931c73ec98b1d091c431c69385e5d1077608f4ec3fd060b` | `66d524dd394d82ad9b96c1fda89b725b1b9ff954a25b26fcd0ea0eed8138f77c` |
| 111 | `6e5213f560e651495c8df6f98663b8d01190403ca55fa51d811415ce8131a11c` | `cf63208d7f948274ecdbd14445b19b557efe13cf329bfb35879a01bfc22d5281` |
| 171 | `6a3f46b3d952ab44bb6dcff85364e97b40e320168ed9c52272fd087f32b310cb` | `deb8512a101beee29fe08b53db2ac779ea864b47a1e2d21da355fafd22aa6fff` |
| 176 | `154969bd5ff228c94bfd5f1b0b0f7c66531d55758abcdbd03de53d363192d4bb` | `48e12f34b40345b81dcca7104e216cf5ae8a71614d8faed6a6a1f8061d55c1eb` |
| 231 | `be5f4a523a88386d0a9f781332a02cd6d9b716dc49e0504a052f91949f99a73b` | `1fcf5555f1679e564dd73aaa324ee0c873468e2f0ba6165ccbf5a7dfb3bb8d35` |
| 301 | `0d9182b0aba89afdfafd5adce9a7d2ca0db5742e8bc5b2db04d07f4f1884c932` | `296511484ffddee5b6dd74ad8bc7b51e7da2757a7dc4444ac08fee9ae9f2745b` |
| 401 | `fdf65df83cfe7d647f66b651eeb9cabe0a53c83578a66aad8f7b40662293ba3a` | `4cecaffe00821a84ff81f1422ebc2ef6b50269a0541eb4f23593a1fc80d20347` |

The canonical ordered manifest SHA-256 is
`ee9b8b6126667baa995e6c8055340e8494b38e4d6f9d65c64554d4389127c979`.
The verifier embeds the complete manifest and refuses shape, dtype, hash,
fingerprint, order, or settings mismatch.

Nearest-neighbour geometry is a verifier-only dependency-injected diagnostic.
It is not a setting, saved mode, resume identity, or production branch.

## Verification and Release Gates

### Settings, API, and Resume

1. Schema 1 through 4 migration yields Fast and limit 8; schema 5 requires both
   fields. Invalid modes, booleans, noninteger limits, out-of-range limits, and
   `processing_mode` are rejected.
2. Omitted new-job mode resolves by device and gate state exactly as specified;
   saved settings contain no `auto`. Resume omission preserves the migrated or
   persisted value, while an explicit override wins.
3. A matching migrated legacy Fast v3 stage remains reusable. Mode or
   background-Quality limit changes invalidate stereo and downstream only. A
   limit-only change in Fast or Quality none changes no RGB, diagnostics, or
   downstream identity and performs no rewrite.
4. Public `StereoRenderResult.__dataclass_fields__` remains compatible and its
   four masks have the existing dtype, shape, and semantics. The internal frame
   path proves those arrays are not allocated.
5. A diagnostics-triggered rerender always invalidates tracked downstream
   stages before writing, including when regenerated RGB hashes happen to match.
6. Background-Quality metadata fingerprints configured/scaled limits and both
   policy versions but not frame-dependent limits; each background frame
   manifest records both eyes' recomputed values. Quality none fixes all four
   repair-policy identity fields and every frame/root limit object to null.
7. Quality CPU and CUDA executions produce the same RGB/diagnostics identity and
   reuse each other's valid stage. Fast alone retains its old device-bearing
   metadata as a compatibility field; execution provenance never enters either
   Quality fingerprint.
8. The old zero- through three-positional `StereoRenderSettings` calls remain
   valid; the two new fields reject positional use. Direct `settings=None` is
   Fast on CPU/CUDA, while CLI/Web pass an explicit resolved mode. Relative
   Quality dispatch occurs before `build_relative_geometry`,
   `StereoSplatSettings`/`render_geometry` remain Fast-only, and metric Quality
   is not exposed as a new public primitive API.
9. Python/NumPy booleans and non-`Integral` limits are rejected by both public
   settings and `QualityStereoControls`; accepted NumPy integers normalize to
   Python `int`. The four discriminated plan variants reject every crossed
   primitive/settings/control combination before decoding.

### Geometry and Numeric Oracles

1. Smooth ramps and every pixel outside an edge band equal the exact current
   Torch bilinear baseline. Strength zero remains byte-identical for both eyes.
2. Vertical, diagonal, T-junction, and geometry-supported one-sample thin-line
   fixtures contain no cross-region intermediate geometry. Texture edges
   outside the band cannot create or move a region.
3. Relative and metric fixtures prove primitive dependency order, validity,
   pinhole equations, clamping, convergence, and independently recounted clamp
   statistics.
   Metric Quality receives native `MetricGeometryFrame` arrays, never calls the
   old bilinear `build_metric_geometry`, and rejects a full-resolution
   `StereoGeometryFrame`; metric Fast moves construction to the serial render
   thread but preserves the old numeric oracle exactly with no defensive-copy
   overlap.
4. An independent scalar binary64 oracle covers boundary clipping, all four
   corner masks, zero retained weight, one-ULP projected-lane changes, and exact
   metric weight reuse.
5. Adversarial union orders produce identical source-region IDs because IDs use
   minimum member index rather than the implementation's union root.
6. The four boundary-movement fixtures and multi-edge fixture match an
   independent scalar integer-geodesic oracle exactly. Production's one global
   indexed heap matches the scalar global queue; no stale duplicate exists,
   equal-cost owner decrease-key is covered, maximum live entries equal the band
   population, and injected capacity corruption fails typed. No independent
   tiled oracle or gate is claimed.
7. The one-eye offset builder matches the current full-map scalar arithmetic for
   left/right, int32 limits, signed half-lane boundaries, and one-ULP inputs. Peak
   live state is exactly one int32 map plus one float64 row; CUDA OOM retry
   preserves its allocation and bytes instead of rebuilding it.
8. Relative/metric native decode and owned geometry-builder oracles prove every
   array in the `G_i` formulas, no-copy construction, direct invalid-zero metric
   weighting, and release order. Fixtures cover `G_i<Q`, `G_i=Q`, and `G_i>Q`;
   capacity zero fails before mutation rather than clamping depth resolution.

### Visibility, Components, and Repair

1. Pass A records and Pass B invalid masks match by OR and popcount at
   full-frame, one-row, planned, and forced OOM-retry band heights. Record masks
   are nonzero, contiguous, disjoint, and preserve `far_side` under mirrored
   left/right fixtures.
   Equal-depth runs of odd and even length split exactly by per-lane distance,
   with the midpoint assigned left.
2. Pass B RGB and compact diagnostics are byte-identical across band heights,
   CPU/CUDA, and I/O worker counts. Every changed lane was invalid before fill.
3. A vertically full-height component proves cross-band planning without a
   dense fine grid. Horizontal lane-0/lane-15 and vertical disjoint-mask cases
   prove nonadjacent segments do not union; overlapping vertical masks do. A
   last-column/next-row fixture proves horizontal union cannot wrap rows, while
   adjacent same-pixel masks do union.
4. Different union-find strategies produce component IDs ordered by minimum
   canonical fine index. Separated same-pixel masks remain separate, while
   immediately adjacent same-region masks union exactly once.
5. A length-16 run starting at lane 8 reports lane count 16, touched span 2,
   and physical width 1.0; cap classification uses lane count only.
6. Mixed-region pixels are not pure proxies. A same-region pixel with one valid,
   deliberately foreground-contaminated lane may be context but is rejected by
   local, exemplar, and fallback donor selection. Clearance-radius boundaries
   are rejected as donors.
7. Local fills match the independent scalar oracle for both sides, odd/even
   equal-depth splits, target traversal, all five row positions, search limit,
   insufficient context, ineligible slots, exact BGR/first-difference scores,
   integer order, ordinal ties, and evaluation-slot exhaustion. An oriented
   gradient and asymmetric glyph prove donor order preserves direction and does
   not reflect content. Over-cap or failed-local runs reach exemplar or fallback
   instead of black output.
8. Exact pyramid tests cover odd ROI sizes, nonoverlapping half-open core
   interiors, partial last cores, clipped halos, unique ownership, read-only
   later-core targets, target-plus-barrier parents,
   bottom-up working colour, target-only nearest replication for all children,
   unfinished coarse targets, no patch clipping, update visibility, earlier-core
   context, and full-level backend assignment. Donor Sobel fixtures require the
   complete original-safe 9x9 support, reject a nominally legal 7x7 donor with a
   one-pixel adjacent foreground/barrier, and prove no donor patch-edge reflect.
   Target fixtures put barriers and unfinished provisional targets immediately
   outside a legal central 7x7 and prove their adjacent offsets contribute BGR
   but no gradient; processed non-barrier 3x3 support restores the exact gradient
   and expected donor selection.
9. Core, component, and eye budget exhaustion each take the unique specified
   fallback order. Candidate subsampling, 8,192-iteration termination, no-donor
   failure, and segment/scratch/index limits have exact unit expectations.
   The fixed 64 MiB record, 64 MiB graph, and 64 MiB shared repair arenas are
   allocated before mutation; fallback provisional queries precede repair-arena
   reset and exemplar.
   Injected preallocation, index-build, arena-reset, mid-component, and
   post-component `MemoryError` always fail with no committed frame; they never
   change backend selection or retain earlier exemplar results. Planning owns no
   dense repair map; the planar B allocation is demonstrably transferred to
   immutable `repair_bits` without increasing the measured planning peak.
10. A legal donor 129 through 256 1080p-equivalent pixels outside the exemplar
     ROI proves full-frame fallback lookup. A donor one pixel beyond the limit is
     rejected. Brute-force and implicit k-d queries agree across ties, degenerate
     coordinates, regions, and radius boundaries. Arena overflow and index-build
     failure are fatal. No synthesized value ever becomes a donor.
11. `background + quality` returns zero final unresolved lanes or an explicit
    error. `none` retains Revision 5 black-lane behavior and mask semantics.
12. OOM injection after every Pass A, Pass B, and Quality-none visibility band
    proves exact row-zero rollback. Pass A drops analysis/records/counters;
    Pass B preserves records, offset, and immutable compact maps but drops RGB
    and replay counters; none drops partial RGB/maps/histograms. Completed
    earlier-eye state survives, and an OOM at `h1` reports both heights without a
    third attempt or Quality downgrade.

### Diagnostics and Transactions

1. Quality background plan and Pass B counters independently agree on OR masks,
   popcounts, backend lanes, and final unresolved lanes. Quality none constructs
   no plan and reports the exact no-repair graph/budget availability with null
   component/budget values. Fast fill-helper and pre/post-band counters agree
   without constructing Quality regions or records; every Fast component count
   is null with the exact unavailable reason.
2. Coverage/repair masks, run measures/histograms, ratios, backend
   pixel/component counts,
   canonical JSON, positive zero, reserved bit 7, finite numbers, and hashes all
   match independent synthetic expectations.
3. Strict-key tests reject every missing, extra, integer-versus-binary64,
   nullable, noncanonical, bad path, self-fingerprint, payload-hash, and
   state-transition mutation in metadata, frame/encoding-input/final-video
   manifests, prune entries, JSONL, and summary. Scalar aggregation fixtures
   cover source-order last-bit differences, positive zero, empty p95, and
   histogram rank lookup. Every counter-bound multiplication/addition tests
   `U64_MAX-1`, `U64_MAX`, and overflow; consolidation repeats checked addition.
4. Fault injection after final-video close, file fsync, atomic publish,
   encoding-input/publication-manifest fsync/replace, prune metadata, and every
   cleanup step proves the exact recovery sequence. Missing/stale input or
   publication evidence always reencodes; two valid manifests after a crash
   permit prune without re-resolving `auto`; no state falsely claims reusable
   payload.
5. Parallel frame completion produces source-ordered, duplicate-free JSONL and
   the same ordered-manifest, JSONL, summary, metadata, and clamp-summary hashes
   as serial completion.
6. Relative stats require null metric projection. New Fast/Quality metric stats
   replace per-frame clamp sidecars, rebuild the exact compatibility summary,
   and preserve legacy Fast validation. Fault injection covers projection stats
   and the derived clamp summary.
7. Disk preflight covers every lifecycle row and exact `P_set`/`R_set`, maximum
   per-frame/root histogram cardinalities, long escaped frame names, long metric
   fraction lists, masks on/off, metric/relative aggregates, allocation-unit
   fallback, both retained-manifest reservations, and simultaneous old RGB plus
   all new temporaries. It explicitly makes no video-size promise. Direct and
   assembled FFmpeg ENOSPC tests leave an old final video and both old manifests
   byte-exact
   while removing sibling/reservation temporaries. Hard-link fixtures cover the
   existing no-op crop links, retained links, external-link counts,
   symlinks/reparse points, duplicate identities,
   allocation-size lookup failure, per-volume ledgers, and revalidation races.
   One-byte-below/equal/above tests run before a sentinel old stage and prove no
   mutation on failure. The verifier rejects mutation, wrong order, dtype,
   shape, fingerprint, or hash in any of the seven bound source/canonical
   fixtures before rendering.
8. The lifecycle matrix asserts exact `P`/`R` and invalidation behavior for
   same-policy damage, legacy one-frame damage, legacy metric-sidecar damage,
   mask false-to-true, mask true-to-false, and pruned resume. Final-video or
   either retained-manifest damage cannot reinterpret `payload_pruned` as
   reusable stereo. Cleanup tests change `INTERMEDIATE_DIRS` after prune commit
   and prove resume uses only contained, version-1 persisted paths; unknown
   cleanup versions delete nothing.
9. A pruned job with valid final video but damaged JSONL/summary reports
   `(true, false, false)` for final-video, historical-diagnostics, and stereo
   reuse. Inspection neither writes nor renders, the final video remains
   successful, and an explicit diagnostics/mask rebuild uses `P=N` rather than
   aggregate demotion.
10. Encoding-input fixtures prove direct mode authenticates the actually resolved
    06 or 07 left/right bytes and assembled mode the actual 99 sequence. RGB
    byte/order/header/path changes alter its fingerprint; mask/stats changes and
    true-to-false diagnostics migration do not. Audio tokens use full-file hashes.
    ffprobe fixtures reject extra/missing streams, dimensions, pixel format,
    codec, rational rate, frame count, audio-policy mismatch, and any full-decode
    error; nonempty garbage never publishes.

### Fast Compatibility

Run the complete Revision 5 independent discrete oracle, procedural fixtures,
CPU/CUDA comparison, banding, OOM retry, zero-strength, benchmark, and public
mask tests. Fast eye arrays and masks must be exactly equal, not merely visually
similar. Its one-eye relative/metric offset builders must equal the current two
full-map helpers before those helpers are retired from production Fast. Frame 89
must reproduce both pinned Fast eye hashes above.

### Real-fixture and Temporal Review

Render frames 89, 111, 171, 176, 231, 301, and 401 in Fast, Quality,
Quality-without-fill, and verifier-only nearest modes. Produce full-frame views,
400-percent crops around hair, feather, gun, and ribbon structures, both compact
diagnostic images, and a JSON report bound to candidate commit and input hashes.

The report must prove zero cross-region intermediate geometry in edge bands,
zero writes to valid lanes, zero donor-region violations, zero final unresolved
lanes, and only expected deterministic downstream transforms. Human review
rejects any named contour with a background stretch, soft halo wider than one
output pixel, nearest-style stair step, missing thin structure, or copied
foreground streak. Whole-frame PSNR/SSIM cannot override crop review.

A license-free sequence with a one-pixel foreground line moving by quarter-pixel
increments must move monotonically, preserve the line whenever geometry has a
region, and match serial/parallel output. Contiguous real-frame windows around
the seven fixtures receive flicker review. Failure returns to a separately
reviewed temporal design and does not authorize hidden optical flow.

### Performance and Capacity

The clean CUDA reference report must record GPU and driver, CPU model, physical
and logical core counts, installed RAM, OS, Python/Torch/CUDA/NumPy/OpenCV
versions, Torch intra/inter-op thread counts, OpenCV thread count, frame-worker
count, and every benchmark command. The current intended host reference is a
13th Gen Intel Core i9-13900K (24 cores/32 logical processors, 63.58 GiB RAM)
with RTX 4090; a changed host requires a newly recorded baseline rather than
being described only as an RTX 4090 run.

With five warmups and 30 measured frames on that clean CUDA environment:

- Quality renderer p95 is at most 5.0 seconds at 1920x1080 and 20.0 seconds at
  3840x2160, and at most 12 times matching Fast p95;
- a 4K synthetic component covering 25 percent of the image finishes within
  30.0 seconds, within budgets or with explicitly counted safe fallback;
- a 4K synthetic frame routing at least 25 percent of pixels to fallback builds
  its at-most-48 MiB index and completes exact queries within 30.0 seconds;
  query p95 visits at most 4,096 nodes and diagnostics report all visits;
- full-pipeline Quality p95 including diagnostics is at most 6.0 seconds at
  1080p and 24.0 seconds at 4K;
- GPU allocation plus 25 percent headroom fits 1,536 bytes per source pixel;
- Quality lifecycle capacity is exactly one; geometry, region solver, planning,
  Pass B, writer, consolidation, and public-wrapper measurements each stay below
  their calculated phase bound and 512 MiB. The indexed heap never exceeds one
  live entry per band pixel, and region scratch is absent during repair planning;
- Fast 1080p/4K runs at worker counts 1, 4, and 16, masks disabled/enabled, and
  native geometry ratios `G/Q` below, equal to, and above one. They verify each
  frame's compressed-byte/header-derived slot, calculated maximum capacity, and
  forced decoder plus active render plus writer overlap. Actual Python, NumPy,
  OpenCV, Torch, and allocator-resident memory remains within
  `fast_active_scratch + capacity*fast_slot_bytes` and 512 MiB;
- total stage-04 disk use and transaction peak do not exceed preflight reserve;
- Fast renderer and no-mask pipeline p95 regress at most 5 percent, and Fast
  pipeline p95 with masks enabled regresses at most 25 percent.

On the same host with renderer device CPU, five warmups and ten measured frames
must give Quality renderer p95 at most 120 seconds at 1080p and 480 seconds at
4K, with the same 512 MiB host limit and byte-identical CUDA output. CPU never
becomes the omitted-job Quality default, even when this reference gate passes.
The same completed Quality cache is reusable across CPU/CUDA because device
provenance is non-semantic.

The CUDA omitted-job default changes to Quality only after every numeric gate
and seven-frame human crop review passes. A failure cannot be hidden by raising
budgets, disabling diagnostics, lowering strength, or weakening RGB/depth
barriers without another reviewed canonical revision.

### Repository Gate

Black, configured flake8 including McCabe complexity 10, mypy, the full unit
suite, and at least 85 percent unit coverage must pass. No full-frame device
fine grid, silent Fast downgrade, unbounded repair allocation, or new neural
dependency is allowed. The candidate is not merged until the crop/report review
receives explicit user approval.

## Implementation Boundary

The implementation plan must begin with Task 0, before production renderer
changes. Prototype the exact indexed heap and exact implicit k-d lookup at 1080p
and 4K with edge-band populations 25, 50, and 100 percent plus a separate
25-percent-fallback case. Every run compares the independent scalar oracle and
records output equality, maximum live heap entries, visited-node p95, wall time,
and process RSS under the fixed arenas. Run both CPU and CUDA pipeline contexts;
the host algorithms remain semantically identical.

The same Task 0 also prototypes the no-copy relative/metric native decoders and
serial Fast geometry builders at `G/Q` below, equal to, and above one. It records
every live array, compressed input retention, allocator-resident peak, scalar
output equality, and capacity for workers 1, 4, and 16. A hidden constructor copy
or library workspace outside the fixed formula blocks production work just as a
heap/k-d gate failure does.

Task 0 first evaluates an implementation using only existing NumPy, Torch, and
OpenCV facilities. If it misses any performance or RSS gate, this specification
authorizes one project-owned C++/Torch host extension for indexed-heap and k-d
inner loops. It must ship as prebuilt artifacts for every supported
OS/Python/architecture combination, use Torch's already present C++/pybind11
toolchain, expose the same fixed dtypes/arenas/tie order/typed failures, and match
the scalar oracle byte for byte. Runtime JIT compilation, a new third-party
runtime dependency, an alternate RGB identity, relaxed exactness, unbounded
allocation, and silent Fast fallback remain forbidden. If neither existing-deps
nor prebuilt-extension path passes, explicit Quality is unavailable with an
actionable error and the implementation plan returns for architecture approval.

Implementation may change settings, CLI, Web controls, resume behavior, stereo
geometry, renderer internals, frame writer, diagnostics, tests, benchmark and
verifier scripts, and stereo documentation. Prefer focused Quality geometry,
coverage, and reconstruction modules over growing `stereo_renderer.py` into a
second monolith. Update at least `docs/ARCHITECTURE.md`, `docs/PARAMETERS.md`,
`docs/TROUBLESHOOTING.md`, and affected resume/performance documentation.

Do not modify depth inference, canonicalization, scene analysis, temporal
postprocessing, crop, distortion, upscaling, VR layout, or encoded media
semantics. Content-only encoding-input identity, captured executed FFmpeg
arguments, sibling-temporary encoding for both paths, exact container validation,
durable final-video publication, and versioned prune entries are explicitly
authorized transaction changes. No compressed-video size preflight guarantee is
authorized or claimed.
NumPy, Torch, and OpenCV are already production dependencies; no new external
runtime dependency is authorized.

## Rejected Alternatives

- **Nearest geometry:** confirms the cause but replaces halos with steps and
  increases true holes.
- **Full-image guided/bilateral filtering:** can transfer texture or line art
  into geometry where no source boundary exists.
- **Fill-cap-only changes:** did not affect 8/10-pixel experiments and exposed
  black cracks at 6 pixels.
- **Current boundary copy plus masks:** improves observability but repeats a
  potentially contaminated boundary colour.
- **Generic OpenCV inpaint:** cannot enforce source-region donor barriers.
- **Neural repair or matting:** adds downloads, nondeterminism, and a much wider
  validation surface before the proven causes are addressed.
- **Optical flow in Quality v1:** serializes frame work and can warp outlines;
  temporal state requires a separate reviewed design.

## Approval Criteria

Approval of this canonical specification accepts:

1. Exactly two modes: byte-compatible Fast and offline-first Quality.
2. Quality geometry modifies primitive interpolation only inside a
   geometry-supported band and preserves the exact Fast baseline elsewhere.
3. Quality repair uses banded analysis, a capped global segment plan, and banded
   replay rather than a dense full-frame fine grid.
4. Segment records preserve far-side intent and exact fine-lane connectivity;
   every fill writes only replay-confirmed invalid lanes.
5. Only fully covered, barrier-cleared same-region pixels may be copied; bounded
   exemplar and full-frame fallback follow the unique deterministic contracts.
6. Compact production diagnostics coexist with an unchanged allocating public
   mask API and producer-attested lane statistics.
7. Schema 1 through 4 jobs migrate to Fast; schema 5 preserves resolved intent.
   Gated CUDA jobs may default Quality, while CPU jobs default Fast.
8. No neural dependency, generic inpainting, or temporal state enters v1.
9. Quality v1 serializes one decoded/rendered/written frame lifecycle. Fast uses
   geometry-aware relative/metric slot bounds over `Q`, every `G_i`, compressed
   input bytes, JSON, and queue overlap. Either mode rejects a phase over 512 MiB
   rather than assuming `G<=Q` or silently reducing quality.
10. A retained content-only manifest authenticates the exact 06/07 or 99 image
    bytes consumed by FFmpeg, independent of diagnostics and mask policy. A
    durably fsynced manifest of the actually executed FFmpeg arguments binds that
    fingerprint after final-video publication and before `payload_pruned`;
    missing evidence forces reencode, never inference.
11. With intermediates disabled, valid historical aggregates may remain readable
    after pruning, but deleted stereo payload is never claimed reusable and later
    aggregate damage is reported as irrecoverable without invalidating a valid
    final video.
12. Direct public renderer omission remains Fast on every device; device-aware
    Quality defaulting belongs only to resolved CLI/Web jobs. Metric Quality uses
    explicit `QualityStereoControls` and a discriminated plan, while Quality none
    excludes the unused fill limit from semantic and diagnostics identity.
13. Prune recovery deletes only committed versioned relative paths, strict uint64
    cardinality is proved before mutation, and final encoding promises manifest
    space plus sibling-temporary recovery rather than an invented video-size
    reserve. Publication requires the exact ffprobe/full-decode validator.
14. Task 0 must prove the exact heap/k-d performance path. A project-owned
    prebuilt extension is permitted only under the fixed oracle and packaging
    constraints; it cannot weaken output or add a silent fallback.
15. All deterministic, resource, transaction, seven-frame visual, temporal, and
    Fast compatibility gates pass before implementation is considered releasable.
