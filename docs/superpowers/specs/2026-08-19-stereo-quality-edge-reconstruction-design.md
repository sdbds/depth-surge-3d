# Stereo Quality Edge Reconstruction Design

## Status

The sectioned design was approved in chat on 2026-08-19. This written
specification is committed for a separate user review before implementation
planning begins. Approval of this document authorizes planning, not merge.

PRO review found implementation-blocking ambiguities in the first written
revision. `2026-08-19-stereo-quality-edge-reconstruction-revision-2.md`
incorporates and supersedes its Quality data-flow and verification details. The
two documents must be reviewed together, with Revision 2 controlling on every
conflict.

This is one coherent stereo-stage revision. Geometry resampling, visibility
coverage, disocclusion reconstruction, diagnostics, settings, and resume
identity change together because each consumes the contract produced by the
previous step. Depth inference and every stage upstream of stereo remain out of
scope.

## Relationship to the Existing Renderer Contract

This specification extends and, where explicitly stated below, supersedes the
following documents:

- `docs/superpowers/specs/2026-08-14-stereo-edge-coverage-design.md`;
- `docs/superpowers/specs/2026-08-14-stereo-edge-coverage-revision-5.md`.

Revision 5 remains authoritative for the fixed 16-sample horizontal footprint,
packed strict z-buffer, deterministic source-index tie-break, balanced 16-lane
downsampling tree, complete-row banding, and the existing Fast pixel result.

The earlier prohibition on RGB-guided geometry reconstruction, a public quality
mode, and a second fill policy is superseded. That prohibition was correct for
the earlier serration defect, where increasing visibility sampling from 8 to 16
crossed the approved oracle thresholds. The newly reported defect survives the
16-sample renderer and has now been isolated to a different mechanism: low-
resolution geometry interpolation and background reconstruction at real
disocclusions.

## Reported Fixture and Root-Cause Evidence

The local fixture root is:

```text
H:\3dtest\1787051840_f908a5f038277cf447b8a6a9b5072311_20260818_191720
```

The supplied SBS screenshot matches `frame_000089.png` with 95 SIFT/RANSAC
inliers. Its mapped source region is approximately `x=1291..3594,
y=51..563` in the SBS output. The screenshot is evidence of the symptom, not a
committed test asset.

The immutable local inputs used during diagnosis are:

| Artifact | SHA-256 |
|---|---|
| settings JSON | `13b66e657877c8227b100f10265114f1a88162faff4f6a950d8b2d7234d41fbe` |
| source frame 89 | `cca5b9dd367ab23d4931c73ec98b1d091c431c69385e5d1077608f4ec3fd060b` |
| raw depth frame 89 | `5096c6dd730eb05e6bd5f9777c4541e8a0fbf7c816844296e0434499e26be387` |
| canonical disparity frame 89 | `66d524dd394d82ad9b96c1fda89b725b1b9ff954a25b26fcd0ea0eed8138f77c` |
| Fast left eye frame 89 | `9f2b60cbc1aee589ed00afa6d854eaaa009ae708acc62534a47dddcb68473bfd` |
| Fast right eye frame 89 | `be77f8fc1f8552c4d8240350d1e1a5c6f1b7d89efca059b260eed5490426534a` |
| supplied screenshot | `364e7590cc3ce8c5b724ddf5b319fd1ac7caaee8b74bc17a2b1eef2c9100122` |

The fixture uses 1920x1080 source and per-eye output, relative stereo geometry,
See-Through depth, `depth_resolution=1080`, `stereo_strength=1.5`,
`convergence=0.5`, and `occlusion_fill=background`. The stored raw depth and
canonical disparity are 1080x608 in width-height notation and are resized to
1920x1080 for rendering.

The diagnosis established all of the following:

- Re-rendering frame 89 through the current renderer reproduces both saved eye
  PNGs exactly, with maximum absolute channel difference zero.
- Stage 99 for frame 89 is an exact concatenation of the stage 04 eyes. Crop,
  VR assembly, and encoding are not the source of this artifact.
- With bilinear geometry and fill disabled, the left eye has 12,171 full-hole
  pixels (0.587 percent) and the right eye has 11,902 (0.574 percent). The
  maximum horizontal run is seven output pixels in both eyes.
- Current background fill changes 21,641 left-eye pixels (1.0436 percent) and
  22,214 right-eye pixels (1.0713 percent), then reports no full-hole pixels.
- Nearest-neighbor geometry increases the real maximum hole width to 11 pixels
  while removing much of the soft fringe and introducing visible stair steps.
  This is diagnostic evidence that bilinear interpolation creates intermediate
  disparities which bridge a real disocclusion.
- An 8- or 10-pixel cap changes no output pixels in frames 89, 111, 171, 176,
  231, 301, and 401 under the current bilinear path. A 6-pixel cap leaves black
  cracks. A cap cannot repair geometry which has already hidden the hole.
- The current fill copies one selected horizontal boundary sample across a
  complete run. It has no background-component barrier and is not general
  inpainting.
- Public masks are reduced after fill and are discarded by the frame writer, so
  current artifacts cannot report where repair actually happened.

The primary cause is therefore cross-edge bilinear geometry. Boundary-copy
fill is a secondary amplifier. Raw model alignment, source antialiasing, and
motion blur remain irreducible inputs and must not be mislabeled as renderer
bugs.

## Goals

- Preserve the current renderer as a selectable, byte-compatible Fast mode.
- Make Quality the default for newly created jobs while migrating existing v4
  jobs to Fast.
- Prevent interpolation from creating a geometry value between visibly
  separated foreground and background surfaces.
- Use RGB only to place a depth boundary already supported by geometry; never
  turn arbitrary line art or texture into new depth.
- Preserve smooth interpolation within one surface.
- Repair only genuine uncovered target samples and never overwrite a valid
  foreground winner.
- Prevent repair donors from crossing a foreground depth barrier.
- Replace repeated boundary-colour streaks with depth-restricted background
  continuation and exemplar synthesis.
- Preserve coverage and repair masks through writing, metadata, resume, and
  debugging.
- Keep source frames, scene data, raw depth, canonical disparity, and metric
  geometry reusable when only stereo render mode or fill limits change.
- Produce deterministic CPU/CUDA output independent of render band height and
  I/O worker count.

## Non-goals

- Recovering a thin object which has no distinct evidence in the input depth.
- Treating every anime outline as foreground geometry.
- Semantic segmentation, person matting, learned view synthesis, LaMa, or any
  other neural inpainting dependency.
- Optical-flow or sequential temporal stabilization in Quality v1.
- Changing depth model resolution, model calibration, stereo strength,
  convergence, metric camera equations, crop, distortion, upscaling, VR
  assembly, or video encoding.
- Blurring output colour or accepting lower stereo strength as a fix.
- Making the 16-lane sample count user-configurable.

## Public Settings Contract

The processing settings schema advances from 4 to 5 and adds:

```json
{
  "stereo_render_mode": "quality",
  "occlusion_fill_max_px": 8
}
```

`stereo_render_mode` accepts exactly `fast` and `quality`.
`occlusion_fill_max_px` accepts integer values from 1 through 32. The value is
a 1080p-equivalent pixel distance and applies only to local Quality background
continuation. For render height `H`, convert it with deterministic half-up
rounding:

```text
safe_limit_px = max(1, floor(occlusion_fill_max_px * H / 1080 + 0.5))
```

The existing `occlusion_fill` setting remains exactly `none` or `background`:

- `none` skips every reconstruction step and retains the Revision 5 black
  uncovered-lane behavior;
- `background + fast` uses the current bounded boundary-copy implementation;
- `background + quality` uses the reconstruction contract in this document.

The removed `processing_mode` name remains removed. It must not be revived or
accepted as an alias.

New explicit jobs which omit the new fields receive `quality` and `8` from
defaults. When parsing a schema-v4 saved job, migration injects `fast` and `8`
before normal validation. A schema-v5 saved job must contain both new fields;
their absence is an error rather than an implicit reinterpretation. Existing
v1-v3 migration behavior remains unchanged.

The Web UI exposes a two-value Fast/Quality segmented control and shows the
local fill limit as an advanced Quality setting. The CLI exposes
`--stereo-render-mode {fast,quality}` and
`--occlusion-fill-max-px 1..32`. Saved settings always write both values.

## Mode-specific Persistence Contract

Stage identity is resolved by mode rather than represented by one global
algorithm string:

```text
Fast schema:       1
Fast algorithm:    torch-horizontal-16x-zbuffer-v3
Quality schema:    2
Quality algorithm: torch-horizontal-16x-rgb-snap-exemplar-v1
```

Fast output colour, public valid masks, and public hole masks remain governed
by Revision 5. Fast fingerprints retain the old render-setting shape and ignore
`occlusion_fill_max_px`, which has no Fast behavior. A schema-v4 job migrated
to Fast may reuse an existing valid v3 stereo stage. Resume must replace the
current blanket `legacy renderer schema` decision with a semantic check that
allows exactly this v4-to-v5 Fast case. Older schemas remain subject to their
existing invalidation rules.

Quality fingerprints include:

- `stereo_render_mode`;
- the Quality stage, geometry-snap, and reconstruction algorithm versions;
- configured and effective local fill limits;
- the source RGB frame fingerprint used as the guide;
- every existing relative or metric projection setting and upstream geometry
  fingerprint.

Changing mode, the Quality fill limit, or a Quality algorithm version
invalidates stage 04 and every tracked downstream frame stage. It does not
invalidate source, scene, raw-depth, canonical-disparity, stabilized-disparity,
or metric-geometry stages. Switching back to Fast may reuse a matching Fast v3
stage only when that stage still exists.

Legacy reused Fast artifacts do not contain reconstructable per-lane
diagnostics. They remain reusable and explicitly report diagnostic availability
as `legacy_fast_unavailable`; the application must not invent masks from final
RGB. Every newly rendered Fast or Quality frame carries diagnostics.

## Architecture and Data Flow

The stereo path is separated into three responsibilities.

### Geometry Builder

The geometry builder consumes low-resolution relative disparity or metric
inverse depth, its existing calibration inputs, the render-shape source RGB
guide, and the selected mode. It produces the existing full-resolution
`StereoGeometryFrame` plus optional Quality edge diagnostics.

Fast calls the current bilinear implementation without changing evaluation
order or values. Quality calls the RGB-guided region algorithm below. Relative
and metric paths still produce the same `near_score`,
`total_disparity_fraction`, and `source_valid` meanings. Quality changes only
how low-resolution geometry is assigned to output pixels; it does not invent a
new calibration value.

### Visibility Renderer

The visibility renderer owns only projected unit footprints, 16-lane strict
z-buffer selection, and coverage. It does not decide which background texture
is plausible. It returns the fine-grid winner buffers required by the
reconstructor and a `StereoCoverage` record, then applies the existing balanced
tree after reconstruction.

### Occlusion Reconstructor

The reconstructor consumes fine-grid colour, winner depth, validity, projected
Quality barriers, and the selected fill policy. Fast dispatches to the current
fill helper. Quality performs bounded local continuation followed by restricted
exemplar repair. It may write only invalid lanes recorded by the visibility
renderer.

The frame writer receives both eye images and both coverage records. Masks are
not dropped at `_WriteItem` construction.

## Quality Geometry Upsampling

### Visible-edge Detection

Geometry thresholds are evaluated in final per-eye pixel displacement, not in
normalized depth. For total disparity fraction `d` and render width `W`, one
eye displacement is:

```text
q_px = 0.5 * W * d
```

Construct a four-neighbour graph over valid low-resolution geometry samples.
Block an edge when the absolute difference in `q_px` is at least exactly 1.0
output pixel. Connected samples after blocked edges are removed define surface
region labels. Metric invalid-depth policy remains the current infinite-
background policy before labels are constructed.

Strength zero creates no blocked edges and must retain byte-exact source output.
Gradual within-surface changes remain connected and use bilinear interpolation.

### Uncertainty Band and Markers

Map the low-resolution region labels to render coordinates with nearest-label
placement. A mapped boundary receives radius:

```text
edge_band_radius_px = ceil(max(render_width / geometry_width,
                               render_height / geometry_height)) + 1
```

Pixels outside this band keep their mapped region label as fixed markers.
Inside the band, the mapped centre of every low-resolution sample also remains
a marker so a one-sample-wide foreground component cannot be erased by marker
erosion. Every other band pixel starts unknown.

Run deterministic marker-based watershed on the original uint8 BGR source
frame. Watershed may move only boundaries inside the uncertainty band and may
select only an existing geometry region marker. It cannot create a region or
extend one beyond the band. A watershed boundary pixel is assigned to the
adjacent region with greatest `near_score`; an exact tie uses the lowest region
identifier.

Quality requires a three-channel uint8 BGR guide and raises an explicit type or
shape error otherwise. This matches the persisted pipeline frame contract and
avoids per-frame normalization changing the meaning of RGB confidence. Fast
retains its existing dtype behavior.

For each connected band component, calculate luma Sobel magnitude in `[0,1]`.
Accept its RGB-guided boundary only when its median boundary magnitude is both
at least `0.03125` and at least `1.25` times the component median. Otherwise use
the mapped nearest-region boundary for that component. These constants are
algorithm constants, not user settings; changing them requires a Quality
geometry-version change.

### One-sided Surface Interpolation

For each render pixel, calculate the four ordinary bilinear source indexes and
weights. In a smooth region, use all valid samples exactly as Fast does. In an
edge band, retain only source samples whose region label equals the selected
render label, renormalize the retained bilinear weights, and apply those same
weights to every continuous geometry field. If the retained weight is zero,
use the nearest low-resolution marker from that region with a deterministic
distance, row, then column tie-break.

This preserves a gradient within one surface while making the selected label
hard across a strong boundary. A strong band therefore cannot contain a
disparity strictly between its selected region's contributing samples and a
different region's samples.

The original source RGB is not recoloured or hardened. Antialiased boundary
colour stays in the source; only its visibility geometry receives one label.
Ambiguous watershed boundary pixels prefer the nearer surface to preserve thin
foreground continuity.

## Coverage Contract

Coverage originates at the 16-lane fine grid and is retained until writing.
For each eye, `StereoCoverage` exposes output-resolution arrays with these
meanings:

- `coverage_count`: uint8 count of lanes valid before fill, from 0 through 16;
- `prefill_partial_hole`: `0 < coverage_count < 16`;
- `prefill_full_hole`: `coverage_count == 0`;
- `locally_filled`: at least one invalid lane was filled locally;
- `residual_hole`: at least one lane remained invalid after local fill;
- `exemplar_filled`: at least one residual lane was filled by exemplar repair;
- `fallback_filled`: at least one residual lane required the safe fallback;
- `final_unresolved`: at least one lane remains invalid after all enabled
  reconstruction.

The existing public `valid_mask` remains pre-fill `any(valid lane)`. The
existing public `hole_mask` remains post-fill `all(lanes invalid)`. Those masks
are retained for API compatibility; `final_unresolved` is the stricter Quality
diagnostic and uses `any`.

No fill implementation may change a lane which was valid in the pre-fill
z-buffer result. This subset relation is a checked invariant, not a visual
expectation.

When `keep_intermediates=true`, each eye directory receives two lossless uint8
diagnostic PNGs per frame:

- coverage count, storing literal values 0 through 16;
- repair state, storing the seven boolean states above as documented bits.

New renders also write atomic per-frame JSON statistics and consolidate them
into root-level `stereo_coverage_frames.jsonl` and
`stereo_coverage_summary.json`. Per-eye statistics include pixel counts and
ratios for every state, maximum and p95 horizontal full-hole run width, fill
backend counts, and final unresolved-lane count. The root summary survives
normal intermediate cleanup. Mask PNGs do not survive when intermediates are
disabled. Legacy reused Fast stages use the explicit unavailable state defined
above.

## Quality Disocclusion Reconstruction

### Local Limit

Calculate the largest four-neighbour per-eye displacement jump in the current
full-resolution frame. The predicted bounded gap is:

```text
predicted_gap_px = ceil(max_neighbour_abs_q_jump_px) + 2
local_limit_px = min(predicted_gap_px, safe_limit_px)
local_limit_samples = 16 * local_limit_px
```

The `+2` retains the existing footprint guard. The setting limits only local
continuation. A larger component remains masked and proceeds to the dedicated
repair backend; it is never turned black merely because it exceeds the cap.

### Background Component Selection

For each invalid horizontal run, inspect its nearest valid left and right
boundaries. When both exist, select the farther winner using the renderer's
depth ordering. An equal-depth tie selects the nearest horizontal boundary and
then the left boundary. A one-sided candidate is eligible only when the run
touches that row's frame boundary.

Starting at the selected winner, construct its eye-space background component
through four-neighbour valid samples. Traversal cannot cross a neighbour pair
whose per-eye displacement differs by 1.0 pixel or more. Local and exemplar
donors must belong to this component. A candidate on the nearer side of the
occluding boundary is never eligible.

### Local Strip Continuation

Runs no wider than `local_limit_samples` first attempt deterministic strip
continuation. Candidate strips come from the selected background component,
extend away from its boundary, contain one distinct donor per filled sample,
and may search the same row plus two rows above and below. Score candidates
against the three known output pixels adjacent to the hole using equal-weight
Lab colour L1 and luma-gradient L1 error. Lowest score wins; ties use smallest
vertical distance, then row, then source column. If no complete safe strip
exists, leave the run in `residual_hole` rather than copying one foreground or
mixed boundary colour.

### Restricted Exemplar Repair

Residual lanes are represented by an output-resolution target mask where any
lane remains unresolved. Build an eye-space background proxy from valid winner
colours and the selected background component. Repair each connected target
component with deterministic three-level exemplar synthesis at quarter, half,
and full resolution; omit a pyramid level whose shorter ROI side is below 32
pixels.

At each level:

1. Use 7x7 target and donor patches.
2. Search only candidate centres from the selected background component inside
   the component bounding box expanded by 128 output pixels and clipped to the
   frame.
3. Reject a donor patch containing an unresolved sample, a nearer-depth sample,
   or a 1.0-pixel depth barrier.
4. Process target boundary patches by descending known-pixel count, with row
   and column as deterministic ties.
5. Score known target samples by Lab colour L1 plus `0.5` times luma-gradient
   L1. Normalize by the number of compared samples.
6. Copy only unknown proxy samples. Equal scores choose the lowest donor row
   and then column.

The reconstructed output-pixel background value is broadcast only into that
pixel's unresolved fine lanes. Original winning lanes remain untouched. The
balanced 16-lane reduction then preserves partial silhouette coverage instead
of replacing the whole output pixel.

If no legal exemplar exists, fill from the nearest eligible one-sided strip or
nearest safe background-component sample and set `fallback_filled`. If the
frame contains no eligible valid sample at all, return an explicit render error.
For normal `background + quality` pipeline input, `final_unresolved` must be
zero. Generic OpenCV Telea/Navier-Stokes inpainting is not an allowed primary or
fallback path because it cannot enforce the depth-component donor contract.

## Determinism, Memory, and Failure Policy

Fast retains every Revision 5 determinism and memory rule. Quality watershed,
region tie-breaks, candidate order, patch order, and exemplar ties are fully
specified above and must produce the same result for repeated runs, render band
heights, and I/O worker counts.

Quality geometry construction is host-side and tileable with the required
low-resolution-cell and edge-band halo. Tiled and untiled output must be exact.
The GPU still owns complete row bands rather than a full-frame fine grid. A CUDA
OOM halves the renderer band height and retries through the existing policy.
Quality code must not silently switch to Fast, relax a depth barrier, increase
the fill cap, or call unconstrained inpainting. A second OOM or a missing safe
background source terminates with an actionable error while retaining written
diagnostic state.

Quality v1 remains frame-independent so existing parallel frame scheduling is
preserved. Temporal stability is tested below. Optical flow is authorized only
by a separate reviewed design if those tests demonstrate a regression.

## Verification

### Settings and Resume Tests

- New explicit settings default to Quality and limit 8.
- Schema-v4 saved settings migrate to Fast and limit 8.
- Schema-v5 saved settings require both new fields.
- Invalid modes, non-integer limits, booleans, and values outside 1 through 32
  fail validation.
- `processing_mode` remains rejected.
- A valid migrated v4 Fast v3 stereo stage remains reusable.
- Quality/Fast switches and Quality limit changes invalidate stereo and tracked
  downstream stages only.
- A Fast limit change alone does not invalidate a Fast stage.
- Quality metadata binds the RGB guide and both Quality sub-algorithm versions.

### Geometry Unit Tests

- A smooth ramp is bit-identical to current bilinear geometry.
- A vertical two-surface edge, diagonal edge, T-junction, and one-sample thin
  line contain no cross-region intermediate disparity in a strong edge band.
- A high-contrast RGB edge moves an existing geometry boundary inside the
  uncertainty band.
- A texture edge outside the band cannot create or move geometry.
- A low-confidence RGB band uses the nearest-region fallback.
- A thin region retains at least one marker and is not erased.
- Relative and metric modes preserve their existing calibration, clamping,
  convergence, and invalid-depth behavior.
- Strength zero remains byte-identical for both eyes.
- Tiled and untiled Quality geometry match exactly.

### Reconstruction Unit Tests

- Every changed lane was invalid before fill.
- No local, exemplar, or fallback donor crosses a 1.0-pixel depth barrier.
- Local width uses the exact resolution scaling and minimum formula.
- A run over the local cap reaches exemplar repair rather than remaining black.
- Distinct strip pixels replace repeated single-boundary colour.
- Partial pixels retain valid foreground lanes and fill only missing lanes.
- Background mode reaches zero final unresolved lanes when a legal background
  donor exists.
- None mode retains the Revision 5 black-lane behavior and masks.
- Every coverage and state mask has the specified dtype, shape, and semantics.
- Statistics match independently counted synthetic masks.
- Exemplar results are identical across repeated runs and worker counts.

### Fast Compatibility Gate

Run the complete existing Revision 5 independent discrete oracle, CPU/CUDA,
banding, OOM retry, procedural fixture, benchmark, and zero-strength tests. Fast
eye arrays and masks must remain exactly equal, not merely visually similar.
Frame 89 must reproduce the two hash-pinned Fast eye PNGs above.

### Reported-fixture Quality Gate

Render frames 89, 111, 171, 176, 231, 301, and 401 from the local fixture in
Fast, Quality, Quality-without-fill, and nearest-neighbour diagnostic modes.
Produce full-frame comparisons, 400-percent crops around hair, feather, gun,
and ribbon structures, coverage images, repair-state images, and a JSON report
bound to the clean candidate commit and input hashes.

The report must prove:

- zero cross-region intermediate geometry values in classified strong bands;
- zero writes to pre-fill valid lanes;
- zero donor-component violations;
- zero final unresolved lanes for `background + quality`;
- no stage 06 or 99 pixel change beyond the expected deterministic transform of
  the new stage 04 eyes;
- measured Quality latency and peak CPU/CUDA memory at 1080p and 4K.

Human review rejects the candidate if a named contour retains background
stretch or a soft halo wider than one output pixel, if nearest-neighbour-style
stairs replace the halo, if a thin named structure disappears, or if repair
introduces a foreground-colour streak into the background. Numeric whole-frame
PSNR or SSIM cannot override this review because unchanged pixels dominate
those metrics.

### Temporal Gate

Use a license-free fixture containing a one-pixel foreground line translating
by quarter-pixel increments over a textured background. The selected region
boundary must move monotonically with the fixture, preserve the line whenever a
low-resolution marker exists, and produce identical results for serial and
parallel scheduling. Also render contiguous real-frame windows around the
named fixture frames for visual flicker review. Failure stops release and
returns for a temporal design; it does not authorize hidden optical flow.

### Repository and Release Gates

- Black, configured flake8 including McCabe complexity 10, mypy, and the full
  unit suite pass.
- Unit coverage remains at least 85 percent.
- Fast median latency regresses by no more than five percent on the existing
  1080p and 4K benchmark fixtures.
- Quality completes both resolutions without a full-frame device fine grid or
  silent fallback. Quality latency is reported but has no real-time ceiling.
- The candidate is not merged until the reported-fixture crops and JSON report
  receive explicit user review.

## Documentation and Implementation Boundary

Implementation may change the settings, CLI, Web form, resume contract, stereo
geometry, renderer, frame writer, tests, benchmark/verifier scripts, and stereo
documentation required by this feature. Focused new modules for Quality
geometry, coverage records, and reconstruction are preferred over expanding
`stereo_renderer.py` into a second pipeline.

Update at least:

- `docs/ARCHITECTURE.md`;
- `docs/PARAMETERS.md`;
- `docs/TROUBLESHOOTING.md`;
- resume and performance documentation affected by the mode-specific contract.

Do not modify depth-model inference, canonicalization algorithms, scene
analysis, temporal postprocessing, crop, distortion, upscaling, VR layout, or
video encoding behavior. No new third-party dependency is required; NumPy,
Torch, and OpenCV are already production dependencies.

## Rejected Alternatives

### Nearest-neighbour Geometry in Production

It confirms that bilinear mixing is causal but replaces halos with blocky steps
and increases true hole width without a quality reconstruction policy.

### Full-image Guided or Bilateral Filtering

It can transfer anime texture and line-art edges into depth even where geometry
contains no object boundary. Quality mode instead limits RGB influence to a
geometry-supported uncertainty band.

### Fill-cap-only Change

The 8- and 10-pixel experiments changed no sampled output under bilinear
geometry. A 6-pixel cap exposed black cracks. The cap is a guardrail after
geometry correction, not a standalone repair.

### Current Boundary Copy With Better Masks

Masks improve observability but do not prevent a single contaminated boundary
sample from becoming a horizontal colour streak.

### Generic OpenCV Inpainting

Telea and Navier-Stokes have no depth-component donor restriction and can pull
foreground colour directly into a disocclusion.

### Neural Matting or Video Inpainting

These methods may recover depth structures absent from the model, but they add
model downloads, nondeterminism, temporal failure modes, and a much larger
validation surface. They are not needed to test the established bilinear and
fill causes.

### Optical Flow in Quality v1

It would serialize or coordinate the current parallel frame path and may warp
anime outlines. Spatial determinism is implemented and measured first.

## Approval Criteria

Written-spec approval accepts all of these decisions:

1. Add Fast and Quality modes, with new jobs defaulting to Quality and schema-v4
   jobs migrating to Fast.
2. Keep Fast pixel output and eligible v4 Fast caches compatible with Revision
   5.
3. Restrict RGB guidance to existing geometry boundaries and use hard region
   assignment with one-sided interpolation.
4. Preserve all 16-lane visibility winners and permit repair writes only to
   uncovered lanes.
5. Use an 8-pixel 1080p-equivalent local cap by default, with larger holes sent
   to depth-restricted exemplar repair rather than left black.
6. Persist coverage and repair diagnostics, with an explicit unavailable state
   for reused legacy Fast artifacts.
7. Add no neural dependency or temporal state in Quality v1.
8. Require byte-exact Fast gates, synthetic structural tests, hash-bound real
   fixture evidence, temporal review, and explicit crop approval before merge.
