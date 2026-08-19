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
| Stage and per-frame fill limits were conflated | Fingerprint configured/scaled limits plus policy versions; store geometry-dependent limits only in frame diagnostics. |
| Fast component counts had no graph | Mark Fast component counts explicitly unavailable and validate Fast lane/pixel counters without a Quality plan. |
| Diagnostics lacked a strict state machine and schemas | Define exact metadata, frame manifest, stats, JSONL, summary, hashing, building/complete, and aggregate-rebuild contracts. |
| Metric clamp sidecars were outside the transaction | Put new-render clamp statistics in committed frame stats and derive the legacy-compatible clamp summary from diagnostics. |
| Disk reserve omitted RGB transaction overlap | Replace the mask-only estimate with allocation-rounded final payload, aggregate, and one-frame atomic-overlap bounds. |
| Exemplar core ownership was ambiguous | Partition the target bbox into nonoverlapping half-open 384-pixel interiors; halos never own targets. |
| Horizontal union could wrap between rows | Require equal rows and consecutive columns, with a dedicated row-boundary fixture. |
| Fallback lookup had no capacity or complexity contract | Use a capped deterministic per-region implicit k-d index and add a high-fallback 4K stress gate. |
| Tiled geodesic verification had no equivalent algorithm | Require one global host priority queue and remove the unsupported tiled-equivalence claim. |

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
`occlusion_fill_max_px` accepts integers 1 through 32 and controls only local
Quality continuation. It is a 1080p-equivalent distance. For render height `H`:

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

## Persistence and Identity Contract

The only algorithm identities are:

```text
Fast RGB schema/algorithm: 1 / torch-horizontal-16x-zbuffer-v3
Quality RGB schema/algorithm: 2 / torch-horizontal-16x-rgb-geodesic-repair-v2
Diagnostics schema/algorithm: 1 / stereo-coverage-sidecar-v1
```

Fast output colour and legacy masks remain governed by Revision 5. Its
fingerprint retains the old render-setting shape and ignores the Quality-only
fill limit. Migrated legacy jobs may reuse an existing valid Fast v3 stage.
Resume replaces blanket legacy-schema invalidation with a semantic check for
that exact case.

Quality stage identity is constructed before rendering. It retains every
output-affecting current stage field, including geometry mode, ordered frame
names, render shape, `occlusion_fill`, renderer device type, encoding, upstream
geometry identity, and relative or metric projection settings. Its
Quality-specific portion contains exactly the resolved mode,
`configured_limit_1080p`, `scaled_safe_limit_px`,
`predicted_gap_policy="max-four-neighbour-eye-shift-v1"`,
`local_limit_formula="min-scaled-safe-predicted-plus2-v1"`, Quality RGB
algorithm identity, and source RGB guide fingerprint.

`scaled_safe_limit_px` is stage-constant because render shape is stage-constant.
Frame-dependent `predicted_gap_px` and `local_limit_px` are not stage fields;
both eyes record them in frame stats and the frame manifest repeats those values
as a transaction-level assertion. They are already determined by the upstream
geometry fingerprint, guide, settings, and policy versions.

Changing mode, the configured Quality limit, or Quality identity invalidates
stage 04 and tracked downstream stages, but not source, depth, canonical
disparity, stabilization, or metric geometry. A Fast limit-only change does not
invalidate a Fast stage.

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

### Native Metric Input Boundary

The current file pipeline bilinearly expands native metric primitives inside
`_decode_metric_work_item` before calling `render_geometry`. Quality must enter
before that irreversible step. Add this owned internal input without copying the
arrays already owned by `MetricGeometryFrame`:

```python
@dataclass(frozen=True)
class MetricStereoPrimitiveInput:
    metric: MetricGeometryFrame
    virtual_baseline_mm: np.float64
    convergence_distance_m: np.float64
    max_disparity_percent: np.float64
    retained_crop_width: int
```

`metric` supplies native float32 `inverse_depth`, bool `valid`, and float32
`focal_x_normalized`. Constructor validation is the existing metric scalar and
crop validation; the render shape comes from the uint8 source frame.

The internal decode/render union becomes:

```text
np.ndarray | StereoGeometryFrame | MetricStereoPrimitiveInput
```

- relative Fast and Quality continue receiving native canonical disparity;
- metric Fast calls current `build_metric_geometry` in the decoder and sends the
  resulting render-size `StereoGeometryFrame` plus `MetricProjectionStats` to
  the current `render_geometry` path;
- metric Quality sends `MetricStereoPrimitiveInput` with no precomputed stats to
  `render_metric_primitives_compact`, which performs region solving, one-sided
  primitive resampling, projection, clamping, and final stats in that order;
- `render_geometry(frame, StereoGeometryFrame, settings)` is Fast-only and
  raises when `settings.stereo_render_mode == "quality"`;
- metric Quality never silently accepts a render-size `StereoGeometryFrame`.

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

Production owns the complete host region, distance, predecessor/tie state, and
one global priority queue for the render frame. It is not an independent-tile
algorithm and has no fixed-halo approximation. Implementations may chunk array
storage, but every relaxation still enters the same global queue and follows
the same priority order; no tile may finalize a pixel independently. The host
budget already includes these full-frame arrays.

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
pure_bgr:       uint8  [H, W, 3], zero when pure_region_id is absent
```

`coverage_count` is the number of valid pre-fill lanes. A pixel is a pure proxy
for region `r` only when it has at least one valid lane and every valid lane's
winner belongs to `r`. Invalid lanes are ignored. Sum the valid lane colours in
ascending lane order using int32, divide each channel by `coverage_count`, and
round to nearest with ties to even. A pixel containing winners from two regions
is never a pure proxy, even if one region owns 15 of 16 lanes.

`pure_proxy` is context only. A pixel is a `safe_donor` for region `r` only
when it is a pure proxy, `coverage_count == 16`, and every in-frame pixel in its
clipped Chebyshev neighbourhood of radius
`max(1, floor(H / 1080 + 0.5))` also has coverage 16 and pure region `r`. Thus a
one-lane or boundary-adjacent proxy can influence context scoring but can never
be copied by local, exemplar, or fallback repair. The planner may materialize a
temporary boolean safe-donor mask inside its 64 MiB scratch budget; it is not a
fourth retained analysis array.

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
An unplanned record has zero `fill_bgr` and backend 0.

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
the filled 16-byte segment table, `repair_bits: uint8[H,W]`, deterministic
per-frame statistics, and measured budget counters.

### Pass B Replay

Pass B reruns the same banded packed visibility using the same host float64
offset maps. For each band it selects plan records by output-pixel range. The
union of their lane masks must equal Pass B invalid lanes exactly, and every
masked lane must still be invalid. Any mismatch is an internal deterministic
error.

Scatter each record's `fill_bgr` only into its masked invalid lanes. Keep every
valid winner colour unchanged. Execute the existing fixed balanced 16-lane
addition tree and ties-to-even output conversion in the original location, then
release the band. Pass B never uploads or materializes a full-frame fine grid.

This replay makes output independent of band height while retaining exact lane
coverage. One output pixel may use different background colours for different
unresolved lane runs without mixing their donor components.

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

#### Scalar Local-strip Oracle

For repair run `[s,e]` in row `y`, let `dir=-1` and `anchor_fine=s-1` for a left
far side, or `dir=+1` and `anchor_fine=e+1` for right. Let
`boundary_col=floor(anchor_fine/16)`. Enumerate unique touched target columns
from the boundary into the hole:

```text
left far side:  first_pixel, first_pixel+1, ..., last_pixel
right far side: last_pixel, last_pixel-1, ..., first_pixel
```

Call their count `P`. Target index `j` always receives donor index `j`; no
target order reversal is permitted later.

The actual boundary context is the exact sequence
`context_col[i] = boundary_col + i*dir` for `i=0,1,2` in row `y`. All three
coordinates must be in frame and be pure proxies of the repair run's region.
Do not skip an intervening non-proxy and do not shorten context. Otherwise local
repair is ineligible.

The bounded horizontal search is:

```text
search_limit_px = max(1, floor(64 * H / 1080 + 0.5))
row_deltas      = [0, -1, +1, -2, +2]
```

Enumerate candidate slots by `offset=1..search_limit_px` outermost and the five
`row_deltas` innermost. Slot `(offset,k)` has row
`cy=y+row_deltas[k]`, first donor column
`cx=boundary_col+dir*offset`, and ordinal
`(offset-1)*5+k`. Every slot consumes one of exactly
`5*search_limit_px` per-run evaluation slots, including an immediately rejected
out-of-frame slot; there is no pressure-dependent early termination.

A slot is eligible only when `cy` and every
`cx+j*dir`, `j=0..max(P,3)-1`, are in frame and are same-region `safe_donor`
pixels. Define `C[i,c]` as boundary `pure_bgr[y,context_col[i],c]` and `D[i,c]`
as donor `pure_bgr[cy,cx+i*dir,c]` for `i=0,1,2`, channels in B,G,R order.
Compute in this exact integer order:

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

Choose the smallest `(score, ordinal)`. Fill target column `j` from
`pure_bgr[cy,cx+j*dir]`; only that target column's repair-run lane mask receives
the colour. This supplies distinct donor pixels and never repeats the boundary
colour. If no slot is eligible, retain backend 0 and enter exemplar repair. A
standalone scalar implementation is the local-strip test oracle.

### Exemplar Input and ROI

The component repair function consumes a clipped pure proxy, safe-donor mask,
`pure_region_id`, target records, exact region ID, provisional target colours,
and `QualityRepairBudgets`. The full-frame planner computes provisional and
final fallback lookup; the component function never interprets absence from its
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

Candidate donor centres are enumerated in row-major order. Their complete 7x7
patch must lie inside the frame and working ROI and consist entirely of original
same-region `safe_donor` pixels.
For the target patch's known offsets, calculate:

```text
Y = (29*B + 150*G + 77*R + 128) >> 8
Gx = [-1 0 1; -2 0 2; -1 0 1] applied in int32
Gy = transpose(Gx) applied in int32
score = 2 * sum(BGR_L1)
        + sum(abs(Gx_target-Gx_donor) + abs(Gy_target-Gy_donor))
```

Gradient samples use reflect-101 at that level's working-ROI boundary and int64 score
accumulation in row-major sample/channel order. Target gradients read the
current working image, including the defined provisional values above, while
the known mask still controls which patch offsets contribute to the score. The
known-offset count is the same for every candidate in an iteration, so no
division occurs. Lowest score wins; a tie uses donor row then column.

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

Before any component runs, the outer full-frame planner indexes original safe
donors by source-region ID. It supplies provisional colours and resolves every
backend-0 segment after local/exemplar work from the nearest same-region safe
donor within this 1080p-equivalent distance:

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

The index must return the scalar-oracle donor exactly. It has one fixed 48 MiB
per-eye arena covering the uint32 donor array, region descriptors, construction
stack, and query stack. Build complexity is `O(N log N)` over indexed donors,
working memory is `O(N)`, expected query work is `O(log N)`, and exact worst case
remains `O(N)`; queries are never truncated because that would change output.
Diagnostics record indexed donor count, query count, total visited nodes,
maximum, and p95 visited nodes. Query p95 applies NumPy linear interpolation to
the source-order per-query visit counts and is `0.0` when there are no queries.
Wall-clock build time is benchmark telemetry, never hashed frame diagnostics.
The 4K fallback stress gate below constrains the practical case.

Before any component changes a record's backend or `fill_bgr`, allocate the
complete 64 MiB exemplar arena and complete 48 MiB fallback-index arena once.
The index builder and queries use only their arena after that point. If either
preallocation, index construction, or any later allocation raises
`MemoryError`, fail the frame and commit no RGB or diagnostics. Earlier planned
values are discarded. Runtime memory pressure never selects fallback and never
changes output under one RGB identity. Only the fixed core/component/eye
evaluation budgets may deterministically route records to backend 3.

If no safe donor exists within the limit or required index state exceeds its
fixed arena, propagate an actionable render error. There is no unconstrained
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
NumPy linear interpolation; an empty run set reports integer maxima 0 and float
p95 values `0.0`.

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
persisted PNG sidecars. For Quality, one counter is accumulated from final
segment records and a separate Pass B counter from actual scatter masks; lane
totals, per-backend totals, mask OR, popcount, and final unresolved counts must
agree before commit. Fast instead counts writes inside the existing fill helper
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
for every new Fast or Quality render regardless of mask retention.

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

### Metadata State Machine

`04_stereo_diagnostics/metadata.json` has exactly these keys:

```text
schema_version:                     1
algorithm_version:                  "stereo-coverage-sidecar-v1"
status:                             "building" | "complete" |
                                    "legacy_fast_unavailable"
rgb_stage_fingerprint:              string
source_guide_fingerprint:           string
frame_names:                        list[string]  # unique source-order stems
render_shape:                       [H, W]
render_mode:                        "fast" | "quality"
geometry_mode:                      "relative" | "metric"
mask_payloads_enabled:              bool
stats_schema_version:               1
frame_manifest_schema_version:      1
ordered_frame_manifest_fingerprint: string | null
frames_jsonl_sha256:                string | null
summary_sha256:                     string | null
metric_clamp_summary_sha256:        string | null
fingerprint:                        string
```

Before any frame transaction, atomically write `building` metadata with the
four aggregate hashes null and remove stale aggregate temporaries. After every
frame manifest validates, write JSONL, summary, and metric compatibility summary
to same-directory temporaries, replace them, then atomically write `complete`
metadata last with their hashes. Relative mode requires
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

An absent diagnostics root may become `legacy_fast_unavailable` only for a
saved schema 1 through 4 job with a matching reusable Fast v3 RGB stage. It
writes no frame directories, an empty JSONL, and a strict root summary with
`availability="legacy_fast_unavailable"`; ordered manifest hash is the hash of
canonical `[]`. Its complete-like metadata hashes those two aggregate files and
has `mask_payloads_enabled=false`. A legacy metric stage additionally hashes its
already validated old clamp summary; legacy relative uses null. It never
fabricates counts.

### Frame Statistics Schema

Each `stats.json` has exactly:

```text
schema_version:     1
frame_name:         string
render_mode:        "fast" | "quality"
geometry_mode:      "relative" | "metric"
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
`availability`, `local`, `exemplar`, and `fallback`. Quality availability is
`quality_segment_graph` and all three counts are integers. Fast availability is
`unavailable_fast_no_segment_graph` and all three values are null.

`quality_limits` is null for Fast. For Quality it has exactly
`max_neighbour_abs_q_jump_px`, `predicted_gap_px`, and `local_limit_px`.
`quality_budgets` is null for Fast. For Quality it has exactly:

```text
segment_record_count
segment_table_bytes
exemplar_evaluations
fallback_indexed_donor_count
fallback_query_count
fallback_visited_nodes_total
fallback_visited_nodes_max
fallback_visited_nodes_p95
```

### Frame Manifest and Transaction

Each `manifest.json` has exactly:

```text
schema_version:                1
algorithm_version:             "stereo-coverage-sidecar-v1"
frame_name:                    string
render_mode:                   "fast" | "quality"
geometry_mode:                 "relative" | "metric"
render_shape:                  [H, W]
rgb_stage_fingerprint:         string
mask_payloads_enabled:         bool
quality_limits:                null | {"left": QualityLimits,
                                       "right": QualityLimits}
payloads:                      PayloadMap
fingerprint:                   string
```

`QualityLimits` has exactly `max_neighbour_abs_q_jump_px`,
`predicted_gap_px`, and `local_limit_px`. `quality_limits` is null for Fast. For
Quality, its two objects must equal the corresponding `stats.json` eye values
exactly; this repetition lets the manifest assert the geometry-dependent limits
without placing a frame list in stage identity.

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
contain min/max for each of the three per-frame limit fields; they and budget
totals are null for Fast. Component availability and nullability must be the
same in every frame.

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
segment_record_count                 # sum
segment_table_bytes_max              # maximum eye/frame value
exemplar_evaluations                 # sum
fallback_indexed_donor_count         # sum
fallback_query_count                 # sum
fallback_visited_nodes_total         # sum
fallback_visited_nodes_max           # global maximum
fallback_visited_nodes_frame_p95_max # max of per-frame p95 values
```

`MetricProjectionAggregate` is required only for metric mode and has exactly
`valid_pixel_count`, `clamped_pixel_count`, `clamped_fraction`, ordered
`clamped_fractions`, `affected_frame_count`, `mean_clamped_fraction`, and
`max_clamped_fraction`. Global `clamped_fraction` divides summed clamped by
summed valid; mean is the arithmetic mean of ordered per-frame fractions.

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
derived clamp summary. With intermediates disabled, mask files are never
created; stats/manifests and aggregates remain. With intermediates enabled,
masks and manifest entries remain. No normal cleanup removes a
manifest-recorded file.

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

Capacity becomes mode-specific:

```text
STEREO_HOST_BUDGET            = 512 MiB
FAST_HOST_BYTES_PER_PIXEL     = 24
FAST_HOST_SLOT_OVERHEAD       = 1 MiB
QUALITY_PIPELINE_BYTES_PER_PIXEL = 48
QUALITY_PUBLIC_BYTES_PER_PIXEL   = 52
QUALITY_HOST_SLOT_OVERHEAD    = 64 MiB
```

Use the existing floor division formula with the selected per-mode values.
The generator uses the pipeline coefficient; the public `render()` wrapper uses
the public coefficient, whose extra four bytes per pixel are the four concrete
legacy boolean masks. At 4K, either Quality path admits exactly one lifecycle
slot. The 64 MiB overhead includes exemplar working state; source regions,
compact analysis, the capped 64 MiB segment table, output, diagnostics, global
geodesic arrays/queue, and the at-most-48 MiB fallback index in the
appropriate coefficient. The clean memory gate measures all stereo-owned host
allocations and requires them to remain within 512 MiB. It also verifies that
both fixed arenas are allocated before plan mutation and that left-eye planning
state is released before right-eye Pass A begins. Allocation failure is fatal,
never an alternate output path.

At 4K, one eye's dense Pass A analysis is exactly 8 bytes per pixel, or about
63.3 MiB. The full-resolution source-region map is 4 bytes per pixel, or about
31.6 MiB. Two completed eyes' coverage and repair arrays total another 4 bytes
per pixel, or about 31.6 MiB. The capped 64 MiB segment plan and 64 MiB exemplar
scratch are never live for both eyes simultaneously.

At 4K the public wrapper's four boolean masks add exactly 33,177,600 bytes
(31.64 MiB). Both compact-pipeline and public-wrapper peaks are measured
separately; neither may rely on allocator reuse hidden from the accounting.

### Disk and I/O

Preflight estimates the complete pending stage-04 transaction, not diagnostics
alone. Reuse the metric-stage allocation-unit query and 64 KiB fallback. Define:

```text
A = max(4096, target filesystem allocation unit)
alloc(n) = ceil(n / A) * A

rgb_file_bound  = alloc(ceil(1.25 * H * W * 3))
mask_file_bound = alloc(ceil(1.25 * H * W))
json_file_bound = alloc(64 KiB)

N = total source frame count
P = frame transactions which are missing or require replacement
M = 1 when mask payloads are enabled, otherwise 0

pending_final_rgb_reserve  = P * 2 * rgb_file_bound
pending_final_mask_reserve = P * M * 4 * mask_file_bound
pending_stats_manifest_reserve = P * 2 * json_file_bound

frames_jsonl_bound = alloc(N * 64 KiB)
root_file_count = 3 + (1 when geometry_mode == metric else 0)
root_aggregate_reserve = 2 * (
    frames_jsonl_bound + (root_file_count - 1) * json_file_bound
)

one_frame_atomic_rgb_overlap  = (2 * rgb_file_bound) if P > 0 else 0
one_frame_atomic_mask_overlap = (M * 4 * mask_file_bound) if P > 0 else 0
one_frame_atomic_json_overlap = (2 * json_file_bound) if P > 0 else 0

pending_file_count = P * (4 + M * 4)
aggregate_file_count = root_file_count
atomic_file_count = (4 + M * 4) if P > 0 else 0
filesystem_allocation_overhead = A * (
    pending_file_count + aggregate_file_count + atomic_file_count
)

required_bytes =
    pending_final_rgb_reserve
  + pending_final_mask_reserve
  + pending_stats_manifest_reserve
  + root_aggregate_reserve
  + one_frame_atomic_rgb_overlap
  + one_frame_atomic_mask_overlap
  + one_frame_atomic_json_overlap
  + filesystem_allocation_overhead
```

The factor 1.25 is the same conservative payload multiplier as metric geometry;
per-file allocation rounding and explicit filesystem slack are intentionally
both retained. Root reserve is doubled because an existing aggregate and its
replacement temporary coexist. The one-frame overlap separately covers old RGB,
masks, and JSON while every replacement temporary is pre-encoded. Existing
files already consume disk and are not treated as free.

If free space is below `required_bytes`, fail before rendering with required and
available bytes. Re-evaluate before a repaired-frame batch if `P`, mask policy,
shape, or allocation unit changes. A later ENOSPC/Windows error 112 reports the
persisted estimate, current free bytes, and failing path. With masks disabled,
stats/manifests and root aggregates still remain.

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
3. A matching migrated legacy Fast v3 stage remains reusable. Mode or Quality
   limit changes invalidate stereo and downstream only; a Fast limit-only
   change does not invalidate Fast RGB.
4. Public `StereoRenderResult.__dataclass_fields__` remains compatible and its
   four masks have the existing dtype, shape, and semantics. The internal frame
   path proves those arrays are not allocated.
5. A diagnostics-triggered rerender always invalidates tracked downstream
   stages before writing, including when regenerated RGB hashes happen to match.
6. Quality metadata fingerprints configured/scaled limits and both policy
   versions but not frame-dependent limits; each frame manifest records both
   eyes' recomputed values.

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
   `StereoGeometryFrame`; metric Fast preserves the old call path exactly.
4. An independent scalar binary64 oracle covers boundary clipping, all four
   corner masks, zero retained weight, one-ULP projected-lane changes, and exact
   metric weight reuse.
5. Adversarial union orders produce identical source-region IDs because IDs use
   minimum member index rather than the implementation's union root.
6. The four boundary-movement fixtures and multi-edge fixture match an
   independent scalar integer-geodesic oracle exactly. Production's one global
   queue matches the scalar global queue; no independent tiled oracle or gate is
   claimed.

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
   integer order, ordinal ties, and evaluation-slot exhaustion. Over-cap or
   failed-local runs reach exemplar or fallback instead of black output.
8. Exact pyramid tests cover odd ROI sizes, nonoverlapping half-open core
   interiors, partial last cores, clipped halos, unique ownership, read-only
   later-core targets, target-plus-barrier parents,
   bottom-up working colour, target-only nearest replication for all children,
   unfinished coarse targets, no patch clipping, update visibility, earlier-core
   context, and full-level backend assignment.
9. Core, component, and eye budget exhaustion each take the unique specified
   fallback order. Candidate subsampling, 8,192-iteration termination, no-donor
   failure, and segment/scratch/index limits have exact unit expectations.
   Injected preallocation, index-build, mid-component, and post-component
   `MemoryError` always fail with no committed frame; they never change backend
   selection or retain earlier exemplar results.
10. A legal donor 129 through 256 1080p-equivalent pixels outside the exemplar
     ROI proves full-frame fallback lookup. A donor one pixel beyond the limit is
     rejected. Brute-force and implicit k-d queries agree across ties, degenerate
     coordinates, regions, and radius boundaries. Arena overflow and index-build
     failure are fatal. No synthesized value ever becomes a donor.
11. `background + quality` returns zero final unresolved lanes or an explicit
    error. `none` retains Revision 5 black-lane behavior and mask semantics.

### Diagnostics and Transactions

1. Quality plan and Pass B counters independently agree on OR masks, popcounts,
   backend lanes, and final unresolved lanes. Fast fill-helper and pre/post-band
   counters agree without constructing Quality regions or records; every Fast
   component count is null with the exact unavailable reason.
2. Coverage/repair masks, run measures/histograms, ratios, backend
   pixel/component counts,
   canonical JSON, positive zero, reserved bit 7, finite numbers, and hashes all
   match independent synthetic expectations.
3. Strict-key tests reject every missing, extra, wrong-type, noncanonical, bad
   path, self-fingerprint, payload-hash, and state-transition mutation in
   metadata, stats, manifests, JSONL, and summary.
4. Fault injection after every frame and aggregate replacement leaves the
   correct `building` state and no false commit. Resume distinguishes legacy
   unavailable, rebuildable aggregate damage, and frame damage requiring RGB.
5. Parallel frame completion produces source-ordered, duplicate-free JSONL and
   the same ordered-manifest, JSONL, summary, metadata, and clamp-summary hashes
   as serial completion.
6. Relative stats require null metric projection. New Fast/Quality metric stats
   replace per-frame clamp sidecars, rebuild the exact compatibility summary,
   and preserve legacy Fast validation. Fault injection covers projection stats
   and the derived clamp summary.
7. Disk preflight covers missing and repaired frames, masks on/off, metric and
   relative aggregates, allocation-unit fallback, and simultaneous old RGB plus
   all new temporaries. It is exact one byte below, equal to, and above its
   reserve. The
   verifier rejects mutation, wrong order, dtype, shape, fingerprint, or hash in
   any of the seven bound source/canonical fixtures before rendering.

### Fast Compatibility

Run the complete Revision 5 independent discrete oracle, procedural fixtures,
CPU/CUDA comparison, banding, OOM retry, zero-strength, benchmark, and public
mask tests. Fast eye arrays and masks must be exactly equal, not merely visually
similar. Frame 89 must reproduce both pinned Fast eye hashes above.

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
- compact pipeline and public-wrapper host peaks separately fit 512 MiB;
- total stage-04 disk use and transaction peak do not exceed preflight reserve;
- Fast renderer and no-mask pipeline p95 regress at most 5 percent, and Fast
  pipeline p95 with masks enabled regresses at most 25 percent.

On the same host with renderer device CPU, five warmups and ten measured frames
must give Quality renderer p95 at most 120 seconds at 1080p and 480 seconds at
4K, with the same 512 MiB host limit and byte-identical CUDA output. CPU never
becomes the omitted-job Quality default, even when this reference gate passes.

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

Implementation may change settings, CLI, Web controls, resume behavior, stereo
geometry, renderer internals, frame writer, diagnostics, tests, benchmark and
verifier scripts, and stereo documentation. Prefer focused Quality geometry,
coverage, and reconstruction modules over growing `stereo_renderer.py` into a
second monolith. Update at least `docs/ARCHITECTURE.md`, `docs/PARAMETERS.md`,
`docs/TROUBLESHOOTING.md`, and affected resume/performance documentation.

Do not modify depth inference, canonicalization, scene analysis, temporal
postprocessing, crop, distortion, upscaling, VR layout, or encoding behavior.
NumPy, Torch, and OpenCV are already production dependencies; no new dependency
is authorized.

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
9. All deterministic, resource, transaction, seven-frame visual, temporal, and
   Fast compatibility gates pass before implementation is considered releasable.
