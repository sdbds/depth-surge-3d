# Stereo Quality Edge Reconstruction Revision 2

## Status

This revision closes the blocking and important findings from the PRO review of
`2026-08-19-stereo-quality-edge-reconstruction-design.md`. It is pending user
review before implementation planning. No production implementation is
authorized by this document alone.

The original design remains authoritative for its problem statement, diagnosis,
goals, non-goals, public Fast/Quality intent, and rejected alternatives. This
revision replaces its Quality data flow, metric resampling order, RGB boundary
solver, repair algorithm details, diagnostics persistence, resource budgets,
settings migration, fixture manifest, statistics definitions, and release
gates. Where the two documents differ, this revision controls.

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

## Revised End-to-end Architecture

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

### Why Source Regions Replace Eye-space Winner Components

The original specification asked for a global four-neighbour component over
fine-grid eye-space winners. That graph requires data which the banded renderer
deliberately releases. Revision 2 uses the already required Quality geometry
region as the background-component identity instead.

Source regions are constructed globally before either eye renders. Every valid
fine-lane winner can recover its full-frame source index from the existing
packed visibility key, then gather its uint32 source-region ID. A repair donor
must carry the same region ID as the far boundary selected for that unresolved
lane run. This prevents crossing a source depth edge without retaining a full
eye fine grid or merging an unbounded winner graph across render bands.

Target repair components are still global. Host planning creates nodes keyed by
`(output_pixel_index, source_region_id)` for sparse unresolved segments and
unions four-neighbour nodes with the same region ID. This union includes nodes
from adjacent render bands because it runs after all Pass A records are sorted.
A component may span the full image height without changing memory per valid
fine lane.

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

## Deterministic RGB-guided Region Solver

Revision 2 removes marker-controlled watershed and its floating, densely seeded
contract.

### Low-resolution Region IDs

For every primitive sample, derive its final one-eye displacement in output
pixels through the selected relative or metric formula. Two four-neighbour
samples are disconnected when either condition holds:

- their metric validity differs;
- their absolute one-eye displacement difference is at least `1.0` output
  pixel.

Union every other neighbour pair. Visit low-resolution samples in row-major
order, examine neighbours in up then left order, and assign final positive
uint32 region IDs by the row-major order of each union root. Zero is reserved
for unknown and `0xffffffff` is reserved for no region.

### Mapping and Edge Band

Map low-resolution sample centre `(iy, ix)` to render coordinates with the
same half-pixel convention as `align_corners=False`:

```text
fy = (iy + 0.5) * render_height / geometry_height - 0.5
fx = (ix + 0.5) * render_width  / geometry_width  - 0.5
py = clamp(floor(fy + 0.5), 0, render_height - 1)
px = clamp(floor(fx + 0.5), 0, render_width  - 1)
```

Nearest-label upsampling supplies the initial full-resolution region map. A
mapped boundary is any four-neighbour pair with different non-sentinel IDs. Its
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

Neighbour visitation order is up, left, right, down. Priority order is total
cost, then descending region rank, then ascending region ID, row, and column.
Region rank is the maximum float32 `near_score` bit pattern in that source
region after positive-zero normalization. A lower cost always wins. This makes
a flat-colour band reduce to a deterministic geometry-distance partition while
a real RGB discontinuity raises the cost of crossing that discontinuity. No
Sobel confidence threshold or floating Lab conversion participates in label
selection.

For Dijkstra boundary ties, choose the greater region rank and then lower region
ID. When one-sided primitive interpolation has zero retained spatial weight,
choose the matching low-resolution sample with lowest squared Euclidean distance
to `(fy, fx)`, then lowest source row and column.

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
is never a donor, even if one region owns 15 of 16 lanes.

Each contiguous invalid fine-lane run selects its far boundary through the
Revision 5 depth ordering. A one-sided boundary is legal only at the frame edge.
The run is split at output-pixel boundaries into records with this fixed
little-endian, unaligned 16-byte dtype:

```text
pixel_index: uint32
lane_mask:   uint16
region_id:   uint32
fill_bgr:    uint8[3]
backend:     uint8
reserved:    uint8[2]
```

`lane_mask` bit `i` corresponds to fine lane `i`. Backend values are 0
unplanned, 1 local strip, 2 exemplar, and 3 safe fallback. Reserved bytes are
zero and participate in plan hashing. Records are ordered by pixel index, least
set lane, then region ID. Multiple records may exist for one pixel when its
unresolved runs bind to different source regions.

The in-memory table is limited to 64 MiB per eye. Target components are admitted
whole in row-major component order. A component which would exceed the remaining
table or exemplar budget receives deterministic backend 3 planning instead of
partially entering exemplar repair. If even its compact fallback records cannot
fit, Quality raises `QualityRepairBudgetError`; it does not allocate an
unbounded table or silently use Fast.

### Global Host Planning

Sort Pass A records, build sparse `(pixel_index, region_id)` nodes, and union
four-neighbour nodes of the same region. Union roots and components use row-major
node order. This planning result is independent of Pass A band height.

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

A sparse target component whose maximum horizontal invalid-run width is no
greater than the local limit first tries strip fill. Donors must be pure proxy
pixels with the component's exact region ID. Candidate strips use the same row
or two rows above/below and extend away from the selected far boundary. Every
successive target output pixel uses the successive donor pixel; a boundary
colour is not repeated across the component.

Candidate context is the three nearest known pure pixels on the background side.
Define uint8 luma as `(29*B + 150*G + 77*R + 128) >> 8`. Candidate score is
int64 `2 * BGR_L1 + luma_first_difference_L1`. Compare the same context offsets
for every candidate. Lowest score wins; ties use absolute row delta, donor row,
then donor column. If no complete safe strip exists, retain backend 0 and enter
exemplar repair.

### Exemplar Input and ROI

The component repair function consumes only a clipped pure proxy, its
`pure_region_id`, the component target mask and segment records, the exact
region ID, and `QualityRepairBudgets`. It returns one uint8 BGR value and backend
for every segment or raises a typed budget/data error.

Expand the target bounding box by 128 actual output pixels and clip to the
frame. A repair core is at most 384x384 pixels. Larger boxes split into row-major
384x384 cores with a 64-pixel read-only halo, so each working ROI is at most
512x512. A target pixel belongs to the lowest row-major core containing it. A
core may read already completed earlier-core target colours as context but may
never use a synthesized target pixel as a donor.

### Exact Pyramid

Use full, half, and quarter levels, omitting a level whose shorter ROI side is
below 32 pixels. Construct each coarser cell from its clipped 2x2 children:

- it is a legal donor only when every existing child is a legal pure donor for
  the selected region;
- its donor colour is the per-channel integer mean of those children with
  ties-to-even rounding;
- it is a target when any child owned by the core is a target;
- it is a barrier when any child is outside the selected region or is neither a
  legal donor nor that component's target.

No synthetic padding is added. A 7x7 patch extending outside the clipped frame
or working ROI is ineligible. Completed coarse target colours initialize the
next level by nearest replication using child coordinate `(2*y, 2*x)`, but
finer target pixels remain unprocessed until a finer patch copies them.

Before the coarsest iteration, initialize every target colour from its nearest
legal pure proxy under the fallback distance and tie rules below, without
marking that target processed. This initialization is a scoring value only. At
finer levels it is replaced by the replicated coarser result. Consequently the
working image always has a defined uint8 value even where the processed mask is
false.

### Exact Patch Iteration

At each level, original legal donors start known and target pixels start
unprocessed. Recompute the four-neighbour target frontier after every patch
copy. Select the frontier centre whose clipped 7x7 patch has the greatest known
sample count; ties use row then column. Require at least eight known samples.

Candidate donor centres are enumerated in row-major order. Their complete 7x7
patch must consist only of original legal pure donors for the selected region.
For the target patch's known offsets, calculate:

```text
Y = (29*B + 150*G + 77*R + 128) >> 8
Gx = [-1 0 1; -2 0 2; -1 0 1] applied in int32
Gy = transpose(Gx) applied in int32
score = 2 * sum(BGR_L1)
        + sum(abs(Gx_target-Gx_donor) + abs(Gy_target-Gy_donor))
```

Gradient samples use reflect-101 at the clipped frame boundary and int64 score
accumulation in row-major sample/channel order. Target gradients read the
current working image, including the defined provisional values above, while
the known mask still controls which patch offsets contribute to the score. The
known-offset count is the same for every candidate in an iteration, so no
division occurs. Lowest score wins; a tie uses donor row then column.

Copy every currently unprocessed target pixel in the selected 7x7 patch from
the aligned donor patch. The update is immediately visible as target context,
but copied pixels never become donors. Recompute the frontier and repeat. A
level ends when every target is processed, 8,192 patch iterations have run, or
the donor-evaluation budget is exhausted.

Each 512x512 core may evaluate at most 2,000,000 donor patches across all
levels. When one iteration has more candidates than the remaining budget,
choose exactly `M` candidates from the `N` row-major candidates at indexes
`floor(k*N/M)` for `k=0..M-1`. These indexes are unique because `M <= N`.

### Fallback and Memory Failure

Unprocessed target segments use the nearest original pure proxy pixel with the
same region ID within this 1080p-equivalent distance:

```text
fallback_limit_px = max(1, floor(256 * H / 1080 + 0.5))
```

Distance is squared Euclidean; ties use donor row then column. Set backend 3.
No synthesized pixel is eligible. If no donor exists within the limit, fail the
frame explicitly rather than pulling foreground colour or leaving a black hole.

Exemplar scratch is limited to 64 MiB per eye. A host `MemoryError` first
releases exemplar scratch and applies the same component-level safe fallback.
If fallback allocation or lookup fails, propagate an actionable render error.
There is no unconstrained OpenCV inpaint path and no silent Fast downgrade.

## Compact Coverage and Statistics

Production stores only:

```text
coverage_count: uint8 [H,W]
repair_bits:    uint8 [H,W]
```

Boolean masks are zero-copy properties or temporary views. The fixed repair-bit
mapping is:

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

All pixel counts and ratios are per eye and frame. A pixel ratio denominator is
exactly `H*W`. A horizontal pre-fill hole run is a maximal contiguous invalid
fine-lane sequence in one row. Report its lane count and output-pixel width
`ceil(lane_count/16)`. Maximum and p95 operate over all nonempty runs, not one
maximum per row. P95 uses NumPy linear interpolation; an empty run set reports
integer maximum 0 and float p95 `0.0`.

Backend lane count is the number of lane bits assigned to that backend. Backend
pixel count is the number of pixels with at least one such bit. Backend
component count is the number of global sparse target components which used
that backend at least once. `final_unresolved_lane_count` is the sum of
unresolved lane bits after Pass B. Repair bit 6 is set for a pixel exactly when
its unresolved lane count is nonzero. The public legacy `hole_mask` remains
true only when all 16 lanes are finally unresolved.

Statistics contain only integers, finite nonnegative floats, strings, lists,
and dictionaries. Normalize negative floating zero to positive zero. Reject NaN
and infinity. Canonical bytes are ASCII JSON with sorted keys,
`separators=(",", ":")`, `ensure_ascii=True`, and `allow_nan=False`; SHA-256 of
those bytes is the statistics fingerprint.

## Diagnostics Stage and Transaction

Stereo RGB identity and diagnostics identity are separate:

```text
Fast RGB schema/algorithm: 1 / torch-horizontal-16x-zbuffer-v3
Quality RGB schema/algorithm: 2 / torch-horizontal-16x-rgb-geodesic-repair-v2
Diagnostics schema/algorithm: 1 / stereo-coverage-sidecar-v1
```

Fast RGB metadata and fingerprint remain byte-compatible with existing v3.
Diagnostics are rooted at:

```text
04_stereo_diagnostics/
    metadata.json
    frames/frame_000089/
        left_coverage.png
        left_repair.png
        right_coverage.png
        right_repair.png
        stats.json
        manifest.json
```

Coverage and repair PNGs are single-channel uint8 at render shape. They are
present only when `keep_intermediates=true`. `stats.json` is always present for
a newly rendered frame. `manifest.json` is the frame commit marker and is
written last. It records diagnostics identity, frame name, RGB dimensions, and
SHA-256 for both final eye PNGs, `stats.json`, and every enabled mask PNG.

For one frame, encode every file to a same-directory temporary path, replace
left RGB, right RGB, enabled masks, and stats in that order, then atomically
write the manifest. Any failure removes the manifest and every final file in
that frame transaction, including both RGB images. Resume treats a frame as
complete only when its manifest parses strictly and every recorded hash and
PNG header matches.

An absent diagnostics root is `legacy_fast_unavailable` only when the saved
settings source schema was 1 through 4 and a matching Fast v3 RGB stage already
exists. Migration atomically writes a diagnostics metadata marker with that
state. After that marker exists, or for any schema-v5 job, missing or invalid
new diagnostics are damage and cause per-frame regeneration. This distinction
does not alter the Fast RGB fingerprint.

After all frame manifests are valid, enumerate them in the exact source
`frame_names` order. Rebuild `stereo_coverage_frames.jsonl` and
`stereo_coverage_summary.json` from per-frame stats into temporary files and
atomically replace both root-level outputs. Writer threads never append JSONL.
Duplicate or missing frame names fail consolidation. The summary records the
ordered manifest hash and participates in the diagnostics fingerprint, not the
RGB fingerprint.

Stereo invalidation deletes `04_stereo_diagnostics` and both root summaries.
Normal intermediate cleanup deletes mask PNGs and per-frame diagnostic
directories only after ordered consolidation, while retaining the two root
summary files. Legacy-unavailable summaries state availability explicitly and
contain no fabricated counts.

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
) -> Literal["fast", "quality"]:
    ...
```

- new job, omitted override: `quality` after the release gates below pass;
- schema-v1-v4 resume, omitted override: migrated `fast`;
- schema-v5 resume, omitted override: persisted value;
- any resume with explicit override: validated override.

`--stereo-render-mode` and `--occlusion-fill-max-px` therefore use parser
default `None`, like the temporal postprocessor override. The fill-limit
resolver follows the same omission rules with new-job/migration default 8.
Web resume loads the persisted values before rendering controls and submits an
override only when the user changes them.

The implementation keeps the repository default Fast until all Quality
performance, memory, disk, and visual gates pass on the clean candidate. The
final gate commit changes the new-job default to Quality. If a gate fails, the
feature may remain an explicit Quality option, but it must not become the
default in that release.

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
QUALITY_HOST_BYTES_PER_PIXEL  = 48
QUALITY_HOST_SLOT_OVERHEAD    = 64 MiB
```

Use the existing floor division formula with the selected per-mode values.
At 4K, Quality admits exactly one lifecycle slot. The 64 MiB overhead includes
the exemplar working set; source regions, compact analysis, plans, output, and
diagnostics count in the per-pixel coefficient. The clean memory gate measures
all stereo-owned host allocations and requires them to remain within the
512 MiB budget. It also verifies that left-eye planning state is released before
right-eye Pass A begins.

At 4K, one eye's dense Pass A analysis is exactly 8 bytes per pixel, or about
63.3 MiB. The full-resolution source-region map is 4 bytes per pixel, or about
31.6 MiB. Two completed eyes' coverage and repair arrays total another 4 bytes
per pixel, or about 31.6 MiB. The capped 64 MiB segment plan and 64 MiB exemplar
scratch are never live for both eyes simultaneously.

### Disk and I/O

With masks enabled, diagnostics add four uint8 render-size images per frame.
Preflight reserves this conservative uncompressed amount before stage 04:

```text
diagnostic_mask_reserve = ceil((frame_count + 1) * H * W * 4 * 1.01)
```

Also reserve `4096 * frame_count + 1 MiB` for atomic stats/manifests and ordered
summaries. The additional frame accounts for the largest simultaneous mask
transaction; the 1 percent margin covers PNG/container overhead. If free space
is below the reserve, fail before rendering with the required and available byte
counts. With intermediates disabled, reserve only the stats/summary amount and
retain only the consolidated root files after cleanup.

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

## Revised Verification and Release Gates

The original geometry, reconstruction, Fast compatibility, real-crop, temporal,
formatting, type, complexity, and coverage gates remain, with these additions:

1. Pass A compact records and Pass B invalid masks match exactly at full-frame,
   one-row, normal planned, and forced OOM-retry band heights.
2. Pass B output and diagnostics are byte-identical across every tested band
   height and I/O worker count.
3. A vertically full-height target component proves global sparse union without
   retaining a full fine grid.
4. A partial pixel containing foreground and background winners proves that
   pure proxy generation rejects mixed donors and that separate lane runs bind
   to their own region IDs.
5. Relative and metric primitive-dependency tests prove exact Fast values outside
   bands and recomputed derived values inside them.
6. The four movement fixtures and the multiple-parallel-edge fixture match the
   independent integer-geodesic oracle exactly.
7. Pyramid scale, odd ROI, clipped frame edge, patch update order, budget
   exhaustion, tile ownership, MemoryError, fallback distance, and no-donor
   cases have exact unit expectations.
8. Transaction fault injection after every replacement step leaves no frame
   manifest and no partial RGB/mask/stat set. Resume distinguishes legacy
   unavailable from damaged new diagnostics.
9. Parallel completion order produces sorted, duplicate-free JSONL and the same
   summary hash as serial completion.
10. Statistics are independently recounted from lane masks and reject reserved
    bit 7, negative zero, NaN, and infinity.
11. Disk preflight is exact at one byte below/equal/above the required reserve.
12. The verifier rejects a mutation to any of the seven source or canonical
    files before rendering.

On the same clean RTX 4090 reference environment used by Revision 5, with five
warmups and 30 measured frames:

- Quality renderer p95 must be at most 5.0 seconds at 1920x1080 and 20.0 seconds
  at 3840x2160;
- Quality renderer p95 must also be at most 12 times the matching Fast p95;
- a 4K synthetic component occupying 25 percent of the image must finish within
  30.0 seconds, remain inside every repair budget, or use explicitly counted
  safe fallback;
- full-pipeline Quality p95 including diagnostics must be at most 6.0 seconds at
  1080p and 24.0 seconds at 4K;
- measured Quality GPU allocation plus 25 percent headroom must fit 1,536 bytes
  per source pixel;
- measured stereo-owned host memory must fit 512 MiB;
- actual diagnostic disk use must not exceed the preflight reserve.

The new-job default changes to Quality only after every numeric gate and the
seven-frame human crop review passes on the clean candidate. A failure cannot be
hidden by increasing budgets, disabling diagnostics, lowering stereo strength,
or weakening the RGB/depth barriers without another reviewed revision.

## Revised Approval Criteria

Approval of Revision 2 accepts these corrections:

1. Quality background reconstruction uses analysis, compact global planning,
   and visibility replay rather than retaining a full fine grid.
2. Global component identity comes from source geometry regions; every sparse
   unresolved lane segment binds to one such region.
3. Quality modifies primitive resampling only and preserves exact Fast baseline
   values outside edge bands.
4. Integer geodesic labeling replaces watershed and dense sample-centre markers.
5. Exemplar repair follows the exact bounded pure-function contract above and
   reports every fallback.
6. Coverage uses two dense uint8 arrays plus a capped sparse 16-byte plan, not
   seven resident boolean arrays.
7. Diagnostics use their own auditable transaction and resume identity without
   changing Fast v3 RGB identity.
8. Every saved schema v1-v4 migrates to Fast, and omitted resume flags preserve
   persisted intent.
9. Quality becomes the new-job default only after the fixed performance,
   memory, disk, deterministic, and visual gates pass.
10. All seven real fixture frames and every reported statistic are reproducibly
    hash-bound.
