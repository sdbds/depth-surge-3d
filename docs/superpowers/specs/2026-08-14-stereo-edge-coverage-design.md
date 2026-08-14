# Stereo Edge Coverage Design

## Status

Revision 3, updated on 2026-08-14 after a second independent review of Revision
2. Revision 1's two-layer coverage-budget model remains rejected. Revision 3
keeps the four-sample geometry but removes avoidable reduction passes, makes the
local copyrighted-sample check an explicit manual gate, and closes the
determinism, memory, batching, rounding, and fill-semantics gaps found in the
second review.

The document is pending user review. It authorizes no implementation until the
user approves it.

## Review Disposition

Revision 1 failed because scalar weight did not preserve where a contribution
occupied the target pixel. The same `0.5 foreground + 0.5 background` data can
mean complementary half-pixel coverage or complete overlap. Those cases require
different output but were indistinguishable in the proposed data structure.

This revision resolves the review findings as follows:

- Preserve horizontal occupancy as four explicit target samples per output
  pixel instead of one scalar coverage value.
- Resolve every source contribution at every covered target sample, so there is
  no fixed two-layer limit.
- Use strict depth ordering with a deterministic source-index tie-break. Remove
  the `0.25` depth tolerance and the `weight >= 0.5` primary-voter branch.
- Encode depth and source tie-break into one signed `int64` ordering key and use
  one integer `amax` reduction. Do not compare reduced floating-point depths or
  run a second source-index reduction.
- Keep uncovered target samples black when they cannot be filled. Never divide
  by known coverage and turn a partial silhouette opaque again.
- Fill from a discrete winning source sample, never a colour already mixed at
  an edge.
- Define reproducible procedural, determinism, resume, memory, and performance
  gates. Treat the copyrighted production sample as manual pre-merge evidence,
  not a unit test or CI dependency.
- Calculate threshold-sensitive geometry once per full frame and eye on the
  host, store signed `int32` offsets, and reuse their row slices across normal
  bands and OOM retry.
- State the intentional dark-edge behavior of `occlusion_fill=none`, exercise
  both fill modes, and define ties-to-even output rounding.
- Remove unused batched low-level splat input instead of silently dropping it.
- Limit byte-exact zero-strength behavior to the production `uint8` contract;
  do not invent a stronger historical contract for arbitrary floating dtypes.
- Stop claiming that the current resume report invalidates an encoded video;
  it tracks generated frame directories only.

## Linus Gate

### Is this a real problem?

Yes. Serrated contours are already present in persisted left- and right-eye
PNGs, before side-by-side assembly and video encoding. A representative frame
shows the defect while its full-pixel hole masks are empty. This is a renderer
visibility and sampling defect, not merely an encoder or hole-fill defect.

### Is there a simpler solution?

Yes. Use a fixed four-sample horizontal z-buffer. Projection is horizontal, so
two-dimensional supersampling is unnecessary. Four horizontal samples preserve
quarter-pixel occupancy, handle any number of overlapping depths with one
uniform rule, and remove the binary winner special cases.

An exact one-dimensional interval envelope is mathematically cleaner but
requires dynamic endpoint collection and a variable-length depth sweep per
target pixel. It is appropriate as a small scalar test oracle, not as the first
production implementation.

### What will this break?

Pixels at nonzero disparity, especially at depth discontinuities, intentionally
change. The internal splat result shape changes from output pixels to horizontal
subpixels. Existing stereo-stage outputs must not be reused. The internal
low-level batched `[N,H,W,3]` splat input is removed; the production renderer has
never used it and accepts one eye band at a time.

`occlusion_fill=none` also changes visibly at partially covered silhouettes. For
example, one valid foreground lane plus three unresolved lanes produces 25
percent foreground over black, whereas v1 normalized any positive accumulated
weight to an opaque foreground colour. A dark contour in this mode is therefore
an intentional consequence of returning RGB without an alpha channel or
invented background. `occlusion_fill=background` remains the normal bounded-fill
policy and has a separate anti-darkening gate below.

The following remain compatible: public settings, source and canonical stages,
output dimensions, output dtype, channel order, the six public
`StereoRenderResult` arrays, preview and file-backed entry points, and resume of
all stages upstream of stereo.

## Decision

Replace bilinear point splatting plus binary visibility with a fixed `4x`
horizontal subpixel z-buffer inside the existing Torch renderer.

```python
HORIZONTAL_SUBPIXELS = 4
```

This is an algorithm constant, not a setting. It is encoded in the stereo
algorithm version. Changing it requires another algorithm-version bump.

Do not change canonical disparity generation or resizing in this work. If the
reported-sample gate still fails after correct subpixel visibility, the
remaining error belongs to disparity-boundary reconstruction and requires a
separate design.

## Alternatives Considered

### Fixed 4x Horizontal Subpixel Z-buffer: Selected

Each source pixel projects an opaque unit-width horizontal footprint. Four
target sample points per output pixel retain occupancy position. Every covered
sample performs an ordinary nearest-depth selection, then the four winning
colours are averaged.

The work grows horizontally only. It naturally handles complementary coverage,
complete overlap, thin objects, and three or more depths without layer-specific
branches.

### Exact One-dimensional Interval Envelope: Oracle Only

Intersect every projected source interval with each output pixel, sort all
endpoints, choose the nearest contributor on every resulting segment, and
integrate segment colour exactly.

This is the reference definition. It is rejected for the first production path
because dynamic endpoint lists, segment sweeps, and unbounded overlap counts
would add more implementation and memory complexity than the observed defect
justifies.

### RGB-guided Disparity Reconstruction: Deferred

An edge-aware disparity upsampler may address a depth contour located on the
wrong RGB object. It cannot repair visibility that discards subpixel occupancy,
and combining both changes would make regressions impossible to attribute.

## Problem Statement

The current renderer projects a source pixel to two integer target columns with
bilinear weights. It then selects one depth winner, accepts contributions near
that winner, sums their colours, and divides by positive accumulated weight.

At a foreground boundary, a small foreground contribution can therefore become
a fully opaque foreground output pixel. Adjacent rows cross that binary decision
at different columns and produce a staircase. The renderer knows the fractional
horizontal position before reduction but discards it.

`stereo_strength` scales the visibility of this error. At width `1920`, strength
`1.25` spans 24 pixels of near-to-far binocular disparity and at the canonical
extremes moves one eye by up to 6 source pixels when convergence is `0.5`.
Lowering strength hides the symptom; it does not correct occupancy.

## Goals

- Preserve quarter-pixel horizontal occupancy through visibility resolution.
- Resolve arbitrary overlapping depth layers without a fixed layer count.
- Keep fully covered near surfaces fully occlusive.
- Preserve one-pixel source objects when they remain visible at a target sample.
- Reduce serrated contours without post-render blur or geometry changes.
- Keep one renderer for preview, single-image, CPU, CUDA, and file-backed paths.
- Preserve deterministic complete-row band rendering and bounded GPU memory.
- Invalidate only stereo and tracked downstream frame stages on resume.

## Non-goals

- Correcting an RGB/depth boundary that is geometrically misplaced.
- Learned, bilateral, or RGB-guided disparity upsampling.
- Temporal disparity stabilization.
- Learned view synthesis or wide-disocclusion inpainting.
- Adding antialiasing, quality, sample-count, or compatibility controls.
- Two-dimensional supersampling.
- Changing stereo strength, convergence, the public `occlusion_fill` choices, or
  their high-level policy (`none` versus bounded background extension). Moving
  fill to the fine grid and changing its seeds to discrete winners are intended
  algorithm changes.
- Adding encoded-video lifecycle tracking to the resume report.

## Unchanged Geometry

Canonical disparity remains finite `float32`, with `0=far` and `1=near`.

For canonical disparity `r`, width `W`, strength `s`, convergence `c`, and eye
sign `e` (`+1` left, `-1` right):

```text
d = (r - c) * W * s / 100
q = e * d / 2
u = x + q
```

`d` is total binocular disparity, `q` is source-to-eye displacement, and `u` is
the projected source-pixel centre. Larger `r` is always nearer. Contributions
outside the target frame are discarded, not clamped or reflected.

Canonical disparity is still resized once on the CPU to exact render height and
width with bilinear interpolation and `align_corners=False` before geometry is
computed. Pixel disparity is never resized after calculation.

## Subpixel Geometry

Let `S = HORIZONTAL_SUBPIXELS = 4`.

Source pixel centre `x` owns the half-open unit interval:

```text
[x - 1/2, x + 1/2)
```

After projection it owns:

```text
[u - 1/2, u + 1/2)
```

Fine-grid target sample `j`, where `0 <= j < W*S`, is located at:

```text
t(j) = -1/2 + (j + 1/2) / S
```

A projected source interval covers exactly the `S` consecutive sample indexes
beginning at:

```text
first_j = S*x + ceil(S*q - 1/2)
```

The half-open interval and `ceil` are the tie policy. No epsilon is applied.
Indexes `first_j + lane`, for `lane = 0..S-1`, are generated, and out-of-frame
indexes are discarded.

The resized canonical map already resides on the host. For both CPU and CUDA,
calculate `q` and `ceil(S*q - 1/2)` in host `float64` once per full frame and eye,
then store the resulting offset map as contiguous signed `int32`. Check the
range before narrowing. Under the current validated settings the conservative
bound is `abs(offset) <= ceil(S*W*strength/200 + 1)`, far inside `int32` even at
4K and strength `5.0`.

Use this evaluation order, where `canonical_host` denotes the resized host
values, with one array operation per line and no algebraic reassociation or
fast-math contraction:

```python
canonical64 = np.asarray(canonical_host, dtype=np.float64)
scale64 = np.float64(W) * np.float64(strength) / np.float64(200.0)
base64 = (canonical64 - np.float64(convergence)) * scale64
q64 = base64 if eye_is_left else -base64
fine_shift64 = q64 * np.float64(S)
offset = np.ceil(fine_shift64 - np.float64(0.5)).astype(np.int32)
```

The full-frame offset map is the geometry authority. Each row band receives a
slice of that map; changing band height or retrying after OOM must not recompute
geometry. An implementation may construct and release one eye map at a time or
retain both, but each eye map is computed only once for that frame.

Host `float64` is required because an algebraically rearranged or fused device
multiply/add in `(r-c)*scale` can differ by one ULP. At a fine-lane boundary that
single ULP can move `ceil` by one sample. CUDA must therefore not reconstruct
the offset from canonical values. The full-frame host pass and band-sliced
`int32` transfer count against the performance gate.

## Fine-grid Z-buffer

Every in-bounds source-to-sample contribution carries:

- a fine-grid target index;
- one packed ordering key derived from canonical disparity `r` and the
  deterministic full-frame linear source index.

Canonical disparity is finite, nonnegative `float32` in `[0,1]`. Canonicalize a
signed zero to positive zero before packing. For this range, the unsigned
IEEE-754 bit pattern has the same order as the numeric value. Let `depth_bits`
be a bit reinterpretation of `r`, not a numeric float-to-int cast, and require
every source index to fit in 32 bits:

```text
0 <= source_index < H*W < 2^32
tie_bits   = 0xFFFFFFFF - source_index
packed_key = (int64(depth_bits) << 32) | int64(tie_bits)
```

Because `r <= 1`, every valid key is nonnegative and below signed `int64`'s sign
bit. Greater disparity produces a greater key; equal disparity with a lower
source index also produces a greater key.

Initialize every fine-grid target key to `-1` and perform exactly one integer
`scatter_reduce_(amax)` over all in-bounds contributions. For every result other
than `-1`, decode the low 32 bits to recover the winning source index, gather its
source colour and canonical disparity once, and mark the sample valid. Leave a
`-1` sample invalid, black, and with disparity `-inf`.

There is no depth epsilon, primary voter, fractional colour normalization, or
fixed layer list. All contributors participate in the same z-buffer. A third or
later surface wins any fine sample not covered by a nearer surface.

The packed integer key is the only visibility reduction. A separate depth pass,
floating-point equality filter, source-index `amin`, and floating-point atomic
addition are not permitted. Equal-depth colour averaging is not permitted; the
packed source-index tie-break defines one result independent of contribution
order.

The internal result becomes:

```python
@dataclass(frozen=True)
class SubpixelSplatResult:
    colour: torch.Tensor       # float32 [H, W*S, 3]
    disparity: torch.Tensor    # float32 [H, W*S], -inf when invalid
    valid: torch.Tensor        # bool [H, W*S]
```

The current low-level `image`, `valid_mask`, `projected_disparity`, and
`accumulated_weight` fields are removed atomically. This type and the low-level
function are internal and are not package API.

The replacement low-level function accepts only unbatched colour `[H,W,3]` and
canonical `[H,W]`. Remove the existing rank-four compatibility path and replace
its batch-support test with an explicit rank-four rejection test. Public render
entry points, which already call the function one eye band at a time, do not
change.

## Background Fill

Background fill operates on invalid fine-grid samples before downsampling.

The existing output-pixel maximum gap is retained:

```text
max_gap_pixels = ceil(W * stereo_strength / 200) + 2
max_gap_samples = S * max_gap_pixels
```

For each invalid horizontal run no wider than `max_gap_samples`:

1. Find the nearest valid fine sample on the left and right.
2. When both exist, choose the lower canonical disparity because it is farther.
3. On equal disparity, choose the nearer horizontal sample; an exact distance
   tie chooses left.
4. Copy the selected winner colour and disparity into every sample in the run.
5. When only one candidate exists, fill only if the run touches that row's frame
   boundary. This deliberately preserves the existing bounded edge-extension
   behavior; it is a heuristic because content outside the source field of view
   is unknowable.
6. Runs wider than the maximum remain invalid and black.

The seed is always one discrete z-buffer winner, never a mixed output colour.
Foreground extension at a frame boundary is an explicitly accepted limitation,
not mislabeled as proven background reconstruction.

With `occlusion_fill=none`, skip all fill steps. Invalid fine samples stay black.

## Downsampling and Public Masks

After optional fill, reshape fine-grid colour from `[H,W*S,3]` to `[H,W,S,3]`.
For production `uint8` input, sum lanes in the fixed order
`((lane0 + lane1) + lane2) + lane3`, multiply by `0.25`, round once to nearest
with ties to even (`np.rint` semantics), and convert to source dtype. Invalid
lanes contribute black and are never divided away. For example, `127.5` rounds
to `128`, while `126.5` rounds to `126`.

This is the critical semantic correction: one known foreground lane plus three
unknown lanes remains 25 percent foreground over black when fill is disabled or
impossible. It never becomes opaque foreground through normalization.

`StereoRenderResult` retains its current six public arrays and shapes. Pixel
masks are derived as follows:

- `valid_mask[p]` is true when any of the four samples was valid before fill.
- `hole_mask[p]` is true when all four samples remain invalid after fill.

Partial occupancy is represented in output colour, not exposed as a new public
alpha array. This preserves the existing meaning that every `hole_mask` pixel is
fully black. Repository search shows these masks are renderer diagnostics and
test contracts; no downstream stage consumes them as alpha.

Output dimensions, channel order, and dtype remain unchanged. For production
`uint8` frames, `stereo_strength=0` must return both eyes byte-for-byte equal to
the source with all valid masks true and all hole masks false. Other accepted
integer and floating dtypes retain the current float32 rendering precision and
conversion behavior; no new byte-exact promise is made for them.

## Determinism

Determinism applies to the same source bytes, canonical bytes, settings,
renderer algorithm version, Torch version, and backend runtime.

- Geometry offsets for every backend are calculated once per full frame and eye
  on the host in float64, stored as int32, and sliced without recomputation.
- Visibility uses one packed-int64 max; contribution order cannot change the
  result.
- `uint8` lane sums are exactly representable in float32 and use a fixed order.
- Fill tie-breaks are total and explicit.
- Complete-row band height and OOM retry must not change any output byte or
  public mask.
- CPU and CUDA must produce byte-identical `uint8` images and masks for the
  fixed test corpus because they consume identical indexes and use no unordered
  floating-point sums.

Cross-machine reproducibility is required only when the recorded Torch and
backend runtime versions match. Floating-source images are compared within one
output dtype unit rather than byte-for-byte.

## Memory Bound

`GPU_TEMP_BUDGET = 256 MiB` remains the authority for complete-row band height.
Full-frame host geometry maps are outside that device budget but are included in
the benchmark's host-memory and transfer accounting.

Before implementation chooses the new `SPLAT_BYTES_PER_PIXEL`, it must document
a peak live-set table containing at least:

- source colour and canonical depth;
- host-provided `int32` target offsets after device transfer;
- expanded fine-grid target and source indexes;
- packed `int64` candidate keys and in-bounds masks;
- winning packed key, gathered depth and colour, and validity;
- background-fill cumulative indexes and candidate colours;
- framework sort or scatter workspace if allocated.

The review-time planning estimate is roughly `230 B/source-pixel` before
headroom. Use `300-400 B/source-pixel` for capacity planning until measurement
replaces it; this range is not permission to hard-code an unmeasured value. At
width `3840` under the `256 MiB` budget, `300 B` permits about 233 complete rows
and `400 B` about 174, or roughly 10-13 bands for a 2160-row frame. The current
`192 B` assumption permits about 364 rows and six bands, so launch and per-band
transfer overhead are a material part of the latency gate.

The final constant is measured per source pixel at `S=4`, rounded upward, and
includes at least 25 percent headroom over the largest observed live set. A test
that retains the old `192` without this accounting is a failure.

A 4K full-frame `int32` offset map is about `31.6 MiB` per eye; retaining both is
about `63.3 MiB` of host memory. A temporary full-frame float64 geometry array is
another `63.3 MiB` per eye while constructed. The implementation must document
whether eye maps are built serially or together and release float64 temporaries
before device rendering. In a normal render, band slicing transfers one int32
value per source pixel per eye rather than an int64 map; a failed OOM attempt may
retransfer only the retried rows.

The 4K renderer must remain bounded by reducing complete-row band height. An OOM
retry may halve band height once, as today, and must remain pixel-identical.

## Persistence and Resume

No settings schema or metadata shape changes are required. Keep:

```python
STEREO_STAGE_SCHEMA_VERSION = 1
```

Change only the renderer identity:

```python
STEREO_STAGE_ALGORITHM_VERSION = "torch-horizontal-4x-zbuffer-v2"
```

The existing fingerprint comparison then produces this behavior:

- source frames: preserve;
- scene data: preserve;
- raw depth: preserve;
- canonical disparity: preserve;
- v1 left/right stereo directories: invalidate and regenerate;
- tracked distortion, crop, upscale, and VR-frame directories: invalidate as
  downstream consequences.

The resume report currently ends at VR frames. It does not track, delete, or
atomically invalidate an already encoded video. This design makes no contrary
claim and does not expand resume scope. A resumed processing run must encode its
new final output from regenerated VR frames through the existing normal path.

## Verification

Unless a test states otherwise, RGB error metrics convert stored `uint8` channel
values to `float64` in `[0,1]`, take absolute error per channel, and aggregate
over the named pixels and all three channels. Percentiles use NumPy's linear
interpolation method. Colour tuples below are values in array channel order;
their human colour names are descriptive only.

### Independent Scalar Oracles

Tests use two implementations that import no production scatter, fill, or
reduction helper.

The discrete oracle applies the exact `S=4` sample positions, half-open interval
rule, strict z-buffer, source-index tie-break, fill policy, and ties-to-even lane
average with plain scalar CPU loops. It selects winners directly and must not
copy the production packed-key implementation. Production output and masks must
match it exactly for `uint8` fixtures.

The continuous oracle is used only on small fixtures with fill disabled. For
each target pixel it:

1. Intersects every projected source unit interval with that target pixel.
2. Collects and sorts all intersection endpoints.
3. Chooses the maximum canonical disparity on each endpoint segment using the
   same source-index tie-break.
4. Integrates winning colour by segment length; uncovered length contributes
   black.

This oracle distinguishes overlap from complementary coverage and handles any
number of layers. It defines geometric truth for measuring the approximation
introduced by four samples.

### Exact Unit Tests

Required tests include:

1. Two inputs with identical scalar `0.5 foreground + 0.5 background` weights
   but different overlap positions produce different correct outputs.
2. A three-layer case where the middle layer is fully hidden and the far layer
   occupies the remaining interval selects foreground plus far background.
3. A fully covering foreground suppresses a farther contribution even when
   their disparity difference is less than `0.25`.
4. A one-pixel foreground line survives when at least one target sample sees it
   and never expands beyond its sampled footprint.
5. Coplanar adjacent source pixels tile without overlap or gaps; exact ties use
   the lower source index.
6. Packed keys preserve numeric ordering at `0`, adjacent positive float32
   values, and `1`; equal-depth keys prefer source index zero over larger source
   indexes, and sentinel `-1` loses to every valid key.
7. Integer `scatter_reduce_(amax)` selects the same packed winner on supported
   CPU and CUDA runtimes. CUDA coverage is required when CUDA is available.
8. Integer, half-sample, and one-ULP-on-either-side footprint boundaries obey
   the half-open `ceil` policy.
9. `occlusion_fill=none` averages invalid lanes as black without normalization.
10. Internal background fill chooses the farther of two discrete winners and
    never copies a pre-mixed colour.
11. Single-sided fill occurs only for a bounded run touching a frame boundary.
12. A run over the maximum remains invalid; full-hole pixels are black.
13. Lane values that average to `127.5` and `126.5` round to `128` and `126`
    respectively, proving ties-to-even rather than half-away-from-zero.
14. Public `valid_mask` uses pre-fill `any`, and `hole_mask` uses post-fill
    `all-invalid`, including partial-coverage cases.
15. `uint8` strength zero is byte-identical to the source for both eyes.
16. Production unbatched validation, channel, and dtype contracts remain intact;
    rank-four colour or rank-three canonical input is rejected explicitly.
17. Full-frame-band, one-row-band, and OOM-retry outputs are byte-identical and
    use the same precomputed full-frame offset map.
18. CPU and CUDA match exactly for the complete fixed `uint8` corpus.

### Fixed Procedural Fixture

Generate a license-free fixture with these exact parameters:

- size: `256x128`, RGB `uint8`;
- background colour: `(224, 232, 240)`;
- foreground colour: `(24, 32, 48)`;
- foreground region: `x < 48 + floor(3*y/5)`;
- a second one-pixel line at `x = 180 - floor(y/4)`;
- canonical foreground/line: `0.9`;
- canonical background: `0.1`;
- convergence: `0.5`;
- occlusion fill modes: `none` and `background`;
- strengths: `0.5`, `1.25`, and `3.0`;
- both eyes.

For every mode, strength, and eye:

- production output and masks equal the independent discrete `S=4` oracle;
- the one-pixel line remains represented wherever the discrete oracle sees it.

For `occlusion_fill=none`:

- fully foreground and fully background pixels equal the continuous oracle
  within one `uint8` level;
- inside a four-pixel band around analytic boundaries, candidate RGB MAE versus
  the continuous oracle is no greater than 60 percent of the v1 renderer's MAE;
- candidate 95th-percentile boundary error is no greater than 75 percent of the
  v1 renderer's value;
- outside the boundary band, candidate MAE may exceed v1 by at most `1/255`.

For `occlusion_fill=background`, the fixture must leave every post-fill fine
lane valid. Have the independent oracle retain the discrete source colour chosen
for each lane. For each channel, require
`min(selected_lanes)-1 <= output <= max(selected_lanes)+1`. This explicitly
rejects black-lane darkening in background-fill mode.

The relative v1 comparison for `none` is a release gate, not a unit-test
dependency on old production code. Store frozen v1 expected metrics and fixture
hashes in the test after independently generating them from the parent revision.

### Manual Local Reported-sample Gate

This is one-time pre-merge and release evidence for the reported defect, not a
repository test. Implement it as `scripts/verify_stereo_edge_fixture.py`. It
must not live under `tests/`, be collected by `pytest`, or run in CI. The
copyrighted files and their derived crops are never committed.

Invoke it manually with a fixture root and output locations:

```text
python scripts/verify_stereo_edge_fixture.py --fixture-root <local-output-directory> --report-json <review-artifact.json> --crops-dir <review-crops-directory>
```

The following paths are relative to `--fixture-root`; the timestamped directory
name is not part of the contract. Missing files or hash mismatches make the
manual command exit nonzero with `fixture_unavailable`. That is a failed or
absent review artifact, never a skipped unit test and never a reason for CI to
fail on a machine that does not own the sample.

```text
source:
00_original_frames/frame_001213.png
sha256: 744938e0944ab76cd2074161e90f821776edbe0e0397889416e152760b757b3c

canonical uint16 PNG:
03_disparity_maps/frame_001213.png
sha256: b14ec2ac557d41e3020ed5df9faef038f22f19580eb35765f2b86550417f4e77

v1 left:
04_left_frames/frame_001213.png
sha256: 2db507b403329a9c3c4a8212c24f912bd63fb5d0b73355aa620cf3c6393dfd45

v1 right:
04_right_frames/frame_001213.png
sha256: e82c495b53c0615617650e1e4a3fdb3851bde98cddc0e563ddc31fb84b3dd599
```

Render at `1920x1080`, strength `1.25`, convergence `0.5`, and
`occlusion_fill=background`. Evaluate these half-open ROIs:

- left sleeve/hand: `[x0=280, y0=250, x1=720, y1=650]`;
- right sleeve: `[x0=1080, y0=140, x1=1540, y1=540]`;
- guitar/dress boundary: `[x0=640, y0=300, x1=1160, y1=920]`.

For each ROI, construct an independent `S=64` horizontal sample oracle with a
64-pixel source halo, the same strict z-buffer, and the same scaled fill policy.
Create an edge mask where the resized canonical `3x3` local range is at least
`0.02`, then dilate it by Chebyshev radius four.

For both eyes, require:

- candidate edge-mask MAE versus the `S=64` oracle is no greater than 70 percent
  of v1 edge-mask MAE;
- candidate edge-mask 95th-percentile error is no greater than 85 percent of v1;
- outside the edge mask, candidate MAE versus the oracle exceeds v1 by no more
  than `1/255`;
- output dimensions, stereo displacement direction, and zero-parallax plane are
  unchanged.

Also produce 400 percent nearest-neighbor comparison crops for human review.
The human check rejects a light/dark halo wider than one output pixel or loss of
the one-pixel guitar and hand details. It supplements the numeric gate and
cannot override a numeric failure.

The JSON report records the candidate commit and algorithm version, input
hashes, runtime versions, settings, every ROI metric, threshold result, overall
status, and crop paths. Attach the JSON and crops to implementation review. A
passing CI run without this attachment does not satisfy the manual gate; after
this v2 renderer decision is merged, the copyrighted sample is not treated as a
permanent reproducible regression test.

The originally supplied SBS screenshot has SHA-256
`f89b9fe66ed3c1af2ee2d8e17e5d499a3c0456162e7f2927dad46139d2805998`.
It documents the symptom only; without its source/canonical pair it is not an
algorithm oracle.

### Resume Tests

Create a complete v1 stereo metadata fixture with valid current upstream
fingerprints. Under the v2 constant, assert:

- stereo disposition is `invalidate` because the algorithm version differs;
- canonical and all upstream stages remain reusable;
- every tracked generated stage downstream of stereo is non-reusable;
- a v2 clean run and v2 resumed run produce byte-identical eye PNGs;
- the test makes no assertion that an existing encoded video is deleted.

### Performance and Memory Gate

Extend `scripts/benchmark_stereo_renderer.py`; the current average-only smooth
ramp benchmark is insufficient.

The benchmark must:

- record baseline and candidate git commit IDs, algorithm versions, GPU model,
  driver, Torch, CUDA, Python, resolution, settings, and sample count;
- run in separate fresh processes on the same machine;
- use five warmup frames and 30 measured frames;
- synchronize CUDA before and after every measured render;
- report median and linear-interpolated p95 latency, pipeline FPS, peak allocated
  CUDA bytes, and peak reserved CUDA bytes;
- include host geometry construction and offset-transfer time in end-to-end
  latency, and report those two candidate costs separately with transferred
  bytes so band overhead is visible;
- run both the existing smooth fixture and a fixed sharp multi-depth collision
  fixture at 1920x1080 and 3840x2160;
- measure the exact commit immediately preceding renderer implementation as the
  baseline and attach both JSON outputs to implementation review.

The candidate passes only when:

- median renderer latency is at most `2.5x` baseline for each fixture and
  resolution;
- p95 latency is at most `3.0x` baseline;
- no 4K run fails or exceeds the documented temporary-memory bound;
- measured peak live memory fits the new `SPLAT_BYTES_PER_PIXEL` with its 25
  percent headroom;
- no full-frame device-resident image, index, or subpixel buffer is introduced;
  the resized canonical map remains host-resident and the GPU owns row bands;
- host geometry calculation remains vectorized and adds no per-pixel Python
  loop.

Failure stops release. Do not silently reduce samples to two, add a quality
mode, or retain v1 behind a flag. Optimize the fixed algorithm or return for a
new design decision.

## Failure Policy

- If discrete-oracle tests fail, the implementation is wrong.
- If the continuous procedural gate fails, four samples are insufficient or
  visibility is wrong; do not tune the real-sample threshold.
- If procedural gates pass but the manual local reported-sample gate fails, the
  remaining defect is evidence for disparity-boundary reconstruction. Stop and
  write that separate design.
- If the local sample is unavailable or has a wrong hash, report that the manual
  release evidence is absent. Do not turn it into either a passing skip or a CI
  failure.
- If quality passes but performance or memory fails, optimize data layout before
  reconsidering the algorithm. Do not expose internal sample count to users.

## Required Documentation Changes After Implementation

- `docs/ARCHITECTURE.md`: document projected unit footprints, four horizontal
  samples, strict z-buffer, fill, and downsampling.
- `docs/PARAMETERS.md`: state that stereo strength scales geometry, not
  antialiasing quality, and that `occlusion_fill=none` composites unresolved
  subpixels over black because no alpha output is exposed.
- `docs/TROUBLESHOOTING.md`: distinguish renderer serration from a misplaced
  model depth boundary.
- Resume documentation: record the v1-to-v2 stereo-only algorithm invalidation.

## Rejected Alternatives

### Revision 1 Two-layer Coverage Budget

Rejected. Scalar weights do not retain occupancy position, two layers are not
enough, depth proximity does not prove surface identity, and normalization can
make partial coverage opaque again.

### Tune the Existing `0.5` and `0.25` Thresholds

Rejected. Threshold tuning moves the binary discontinuity without removing it.

### Lower Stereo Strength

Rejected as a fix. It weakens stereo geometry and hides rather than corrects the
sampling error.

### Blur Rendered Colour or Canonical Disparity

Rejected. Colour blur mixes surfaces after visibility loss; disparity blur
creates intermediate geometry and halos.

### 2x Horizontal Sampling

Rejected for the fixed algorithm because it represents only half-pixel
occupancy and leaves one binary transition inside every output pixel. The
quarter-pixel grid is the smallest selected quality target; the procedural gate
is allowed to reject it if that assumption is wrong.

### 8x Horizontal or 2D Sampling

Rejected as the first production path. Eight horizontal samples double the
selected candidate cost; 4x two-dimensional sampling processes 16 samples per
pixel despite vertical coordinates never changing.

### Adaptive Edge-only Sampling

Rejected. It needs a new edge classifier, creates quality discontinuities at
classifier boundaries, and adds branches before a fixed uniform method has been
shown insufficient.

### Compatibility Flag for v1

Rejected. Incorrect occupancy is not a user preference. Algorithm-versioned
resume already prevents accidental cache reuse.

## Implementation Boundary

An approved implementation may change only:

- `src/depth_surge_3d/rendering/forward_splat.py`;
- `src/depth_surge_3d/rendering/stereo_renderer.py`;
- `src/depth_surge_3d/processing/frames/stereo_generator.py` for the algorithm
  version;
- stereo/resume/benchmark tests and independent test-oracle helpers;
- `scripts/benchmark_stereo_renderer.py`;
- `scripts/verify_stereo_edge_fixture.py` for the opt-in local gate;
- the documentation listed above.

`src/depth_surge_3d/io/resume.py` should require no behavior change because it
already imports and compares the stereo algorithm version. Modify it only if a
new failing v1-to-v2 contract test proves that assumption false.

Depth inference, canonicalization, scene analysis, settings, CLI, Web controls,
distortion, crop, upscale, VR assembly, and video encoding are out of scope.

Implementation must preserve the repository quality gates: every function has
McCabe complexity at most 10 under the configured flake8 command, and the unit
suite retains at least 85 percent coverage under the configured
`pytest tests/unit --cov-fail-under=85` run. Split geometry construction, packed
key creation/reduction, winner gathering, fine-grid fill, downsampling, and mask
derivation into focused helpers; do not place the entire pipeline in one
exceptionally complex function or weaken either gate.

## Approval Criteria

Approve Revision 3 only if these decisions are acceptable:

1. Replace scalar two-layer coverage with a fixed four-sample horizontal
   z-buffer.
2. Encode strict canonical depth plus the lowest-source-index tie-break in one
   packed int64 key and perform one integer max reduction; no surface epsilon or
   float-equality pass remains.
3. Keep unresolved subpixels black instead of normalizing partial coverage to
   opacity, accepting the documented dark silhouette behavior when
   `occlusion_fill=none`.
4. Preserve the current bounded single-sided frame-edge fill as an explicit
   heuristic, while requiring two sides for internal runs.
5. Add no user setting or v1 compatibility path.
6. Preserve upstream resume data and invalidate stereo plus tracked downstream
   frame stages through the algorithm version only.
7. Keep the copyrighted production sample out of CI and require its standalone
   JSON/crop report as one-time manual review evidence.
8. Treat failure on that sample as a reason to stop and design disparity-boundary
   reconstruction, not as permission to add blur.
