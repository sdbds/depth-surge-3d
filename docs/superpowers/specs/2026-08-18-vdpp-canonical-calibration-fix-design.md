# VDPP Canonical Calibration Fix

## Status

Approved direction, revised after five PRO reviews. The external whole-shot
fixture payload hash must be recaptured before implementation sign-off. This
document narrows and supersedes the v1 VDPP input and output range policy in
`2026-08-16-vdpp-temporal-postprocessor-design.md`. It does not change depth
estimation, base canonical disparity, or stereo rendering contracts.

## Problem

The released VDPP model predicts a non-negative residual and returns
`scaled_input + residual`. The current integration feeds scene-normalized
canonical disparity directly to the model and clips its uncalibrated output to
`[0, 1]`. Real output from the reported MoGe-2 sky shot reached `3.198`; an
existing stabilized artifact had mean `0.998`, so clipping destroyed depth
contrast and moved midpoint-coded sky pixels from `32768` to `65535`.

The upstream demo min/max-normalizes each input frame and then min/max-normalizes
the whole output video. The input normalization is part of the checkpoint's
actual operating domain. The final visualization normalization is not a valid
render contract here because it would force every shot to span `[0, 1]` and
silently redefine `stereo_strength`.

## Decisions And Invariants

1. Base `02_depth_raw` and `03_disparity_maps` files and fingerprints do not
   change.
2. VDPP keeps `0=far`, `1=near`; an accepted fit must have positive polarity.
3. Uint16 code `32768` is preserved by a **midpoint-code preservation
   heuristic**. It is not a validity mask and it is not a universal neutral or
   zero-parallax value; zero parallax depends on the configurable convergence.
4. The heuristic intentionally cannot distinguish MoGe invalid sky from a
   legitimate surface quantized to `32768`. Both are preserved bit-for-bit and
   excluded from normalization and fitting. This collision is the accepted
   compatibility cost of leaving the base canonical schema unchanged.
5. A true validity sidecar derived from raw depth would require a separate base
   schema, fingerprint, cache, and lifecycle design. In an uncached same-run
   pipeline raw depth is normally generated, but `DepthMapProcessor` removes its
   payloads before returning canonical files when `keep_intermediates=false`.
   Global canonical-cache restore, completed-job cleanup, and later resume may
   also have no raw payloads. Opportunistic raw-mask use would therefore make
   VDPP output depend on artifact availability.
6. Calibration is one affine transform per finalized shot, never per frame or
   per inference window.
7. OLS preserves the fit-pair mean before clipping and minimizes squared error to
   base disparity. It does **not** guarantee exact source variance or dynamic
   range; fixed quality gates reject weak fits.
8. Resident working memory remains bounded independently of shot length. The
   private float32 memmap and final PNG payloads remain disk-backed and scale
   with shot length.
9. Old saturated stabilized artifacts are incompatible and must be regenerated;
   valid base disparity remains reusable.

## Ownership

- `VDPPTemporalPostprocessor` continues to own only the pinned 32-frame,
  four-overlap model recurrence. Its output remains native-resolution float32
  with no canonical clipping.
- Job entrypoints own startup cleanup of stale VDPP private work. CLI, Web,
  `StereoProjector`, and `ProcessingOrchestrator` call one idempotent lightweight
  helper immediately after acquiring or first accepting the job writer lock and
  before resume audit/migration, model loading, or pipeline setup.
- `TemporalDepthStabilizer` owns base decoding, per-frame input normalization,
  pre-scan, stable statistics, shot fitting, quality gates, every active memmap
  lifecycle, encoded fallback, and progress.
- `StabilizedDepthStore.commit_shot()` accepts ordered native-shape `uint16`
  frames that are already final. The store hashes and atomically writes those
  exact values; it does not decode and re-encode them.

A semantic base fallback still commits a complete v2 stabilized artifact with
shot diagnostics. It never returns the base-path list or bypasses the requested
stabilized stage.

## Fixed Constants

The v2 producer uses these exact values:

```text
MIDPOINT_CODE = 32768
MODEL_MIDPOINT_VALUE = 0.5
STATS_TILE_PIXELS = 262144
MIN_PAIR_COUNT = 2
VARIANCE_EPSILON = 1e-12
PHYSICAL_BOUND_ULPS = 4
PHYSICAL_BOUND_REFERENCE_FLOOR = 1.0
MIN_POSITIVE_SCALE = 1e-8
MIN_CORRELATION = 0.50
MIN_POSTCLIP_CONTRAST_RATIO = 0.50
MAX_POSTCLIP_MEAN_DRIFT = 0.01
MAX_PRECLIP_OUT_OF_RANGE_FRACTION = 0.01
```

All variance values are population variances in decoded float64 canonical
units. `preclip_out_of_range_fraction` is the sum of the fractions below zero
and above one over fit-eligible pairs, before clipping.

The fixed gates are deliberately conservative. Eight read-only eight-frame
samples across the reported episode produced correlation and post-clip contrast
ratios of `0.5051..0.9734`, and pre-clip out-of-range fractions of
`0..0.009409`. A complete shot outside these bounds falls back to base rather
than publishing an uncertain stabilization.

## Bounded Pre-Scan

Before checkpoint resolution or model inference, scan one source PNG at a time
and retain only scalar counts:

- validate uint16 grayscale type, native shape, and frame order;
- count total pixels, exact midpoint-code pixels, and frames;
- count frames with at least two distinct non-midpoint uint16 codes;
- count individually flat frames: no non-midpoint pixels, or one unique
  non-midpoint code.

The pre-scan uses masked reductions with `where` over the one full-frame boolean
mask. It never materializes a full-frame boolean-compressed uint16 or float
selection.

If every pixel is midpoint-coded, commit a bit-exact copy with mode
`all_midpoint`. If no frame has non-midpoint range, commit a bit-exact copy with
mode `base_fallback` and reason `source_no_range`. Both decisions happen before
CUDA preflight or model execution.

An individually flat frame inside an otherwise non-degenerate shot is fed to
VDPP as all `0.5` so recurrence remains ordered, but it is excluded from the fit
and copied bit-for-bit during final commit. VDPP is not allowed to invent
structure for that frame.

## Inference Pass

For each pending non-degenerate shot:

1. Create one private float32 memmap shaped `[shot_length, H, W]`.
2. The loader allocates exactly one float32 host window shaped
   `[requested_length, H, W]` and fills it in place. For each source frame:
   - compare the original uint16 values to `MIDPOINT_CODE`;
   - for a ranged frame, determine `lo_code` and `hi_code` exactly in uint16 code
     space and min/max-normalize only non-midpoint pixels to `[0, 1]`;
   - write `MODEL_MIDPOINT_VALUE` at midpoint-code pixels;
   - for a flat frame, fill the entire model input with
     `MODEL_MIDPOINT_VALUE`.
3. Run the existing pinned recurrence unchanged.
4. For each final `(shot_local_index, raw_output)` emitted by `process_shot()`:
   - require finite values and exact native shape;
   - write the frame once at that index in the raw memmap;
   - reread the corresponding base uint16 PNG;
   - if the base frame has range, merge non-midpoint `(raw, base / 65535)` pairs
     into the shot statistics in fixed row-major tiles.

Only final emitted frames enter statistics. Four-frame recurrence overlaps are
not counted twice.

## Exact Arithmetic Contract

Input normalization is float32 arithmetic. For ranged non-midpoint pixels, the
implementation performs the equivalent of:

```text
source32 = source_u16.astype(float32)
source32 -= float32(lo_code)
source32 /= float32(hi_code - lo_code)
```

`lo_code` and `hi_code` are exact Python integers obtained from uint16 values.
Subtraction and division use float32 operands and float32 destinations. The
loader assigns uint16 values directly into its preallocated float32 output
slice, then subtracts and divides that slice in place. Extrema use `where` over
the full-frame midpoint mask; the loader must not create a second float32 source
frame, a float64 input frame/window, or a full-frame boolean-compressed array.

Fit pairs also have one conversion path. Raw `x` begins as the emitted float32
model output and is converted tile-wise to float64. Source `y` is first decoded
with `decode_canonical_png()` (`uint16 -> float32 / float32(65535)`), then
converted tile-wise to float64. Direct uint16-to-float64 division is not an
equivalent implementation.

Every fit, quality, and commit scan flattens frames in C row-major order and
visits exact half-open tiles
`[start:min(start + STATS_TILE_PIXELS, H * W))`. Source decoding calls
`decode_canonical_png()` only on the uint16 tile. The midpoint/eligibility mask
is built after slicing, and boolean compression is permitted only inside that
tile.

Both the quality pass and accepted commit pass call the same pure tile helper,
with no frame-sized alternative fast path:

```text
candidate_tile(raw_tile_f32, scale_f64, shift_f64):
    require raw_tile_f32.size <= STATS_TILE_PIXELS
    preclip64 = raw_tile_f32.astype(float64)
    preclip64 *= scale_f64
    preclip64 += shift_f64
    candidate32 = empty(raw_tile_f32.shape, dtype=float32)
    clip(preclip64, float64(0), float64(1), out=candidate32)
    return preclip64, candidate32
```

Pre-clip low/high counts inspect `preclip64`. Candidate mean and variance use
float64 Chan merges over the float32 values in `candidate32`. Final accepted
commit preallocates one full `candidate_frame32`, fills it tile by tile, calls
the existing `encode_canonical_png(candidate_frame32)` only after the frame is
complete, and then restores source midpoint-code pixels to uint16 `32768`.
These conversions and their ordering are part of the execution identity.

## Stable Statistics And Fit

Statistics use a Chan merge in float64, not raw sums followed by large-number
subtraction. The state is:

```text
count, mean_x, mean_y, M2_x, M2_y, C_xy
```

For state `A` and the next fixed row-major tile `B`, with `n = n_a + n_b`:

```text
delta_x = mean_x_b - mean_x_a
delta_y = mean_y_b - mean_y_a
mean_x = mean_x_a + delta_x * n_b / n
mean_y = mean_y_a + delta_y * n_b / n
M2_x = M2_x_a + M2_x_b + delta_x^2 * n_a * n_b / n
M2_y = M2_y_a + M2_y_b + delta_y^2 * n_a * n_b / n
C_xy = C_xy_a + C_xy_b + delta_x * delta_y * n_a * n_b / n
```

Derived values are:

```text
variance_x = M2_x / count
variance_y = M2_y / count
covariance = C_xy / count
correlation = C_xy / sqrt(M2_x * M2_y)
scale = covariance / variance_x
shift = mean_y - scale * mean_x
```

Finite-precision Chan merges can leave a physically bounded result just outside
its mathematical range. For example, the same Chan merge formula produces
`source_std = 0.5000000000000001` for 19 zeroes followed by 19 ones with
three-value test tiles, and a valid correlation can become
`1.0000000000000002`. Those values are not semantic input failures.

The shared calibration module therefore owns these exact primitives:

```text
canonicalize_float(value):
    require a finite built-in float
    if value == 0.0: return 0.0
    return value

normalize_bounded_stat(value, low, high=None):
    value = canonicalize_float(value)
    reference = max(
        PHYSICAL_BOUND_REFERENCE_FLOOR,
        abs(low),
        abs(high) if high is not None else 0.0,
    )
    slack = PHYSICAL_BOUND_ULPS * math.ulp(reference)
    if value < low:
        if low - value <= slack: return canonicalize_float(float(low))
        raise NumericalContractError
    if high is not None and value > high:
        if value - high <= slack: return canonicalize_float(float(high))
        raise NumericalContractError
    return value
```

The `PHYSICAL_BOUND_REFERENCE_FLOOR` deliberately defines ULP distance at
canonical-unit scale, even for a zero lower bound; `math.ulp(0.0)` is not a
useful tolerance for normalized statistics. `canonicalize_float` converts every
negative zero to positive zero, including unbounded moments and fitted values;
Python JSON writes `-0.0` and `0.0` differently even though ordinary float
equality treats them as equal. The producer canonicalizes before every decision
and persists only canonical values. A reader compares finite floats by exact
`float.hex()` representation, not ordinary equality, so a self-hashed
non-canonical one-ULP excursion or negative zero is rejected rather than silently
rewritten. A finite excursion beyond the fixed slack is a hard
numerical-contract failure, not `base_fallback`.

Apply the primitive to correlation `[-1, 1]`, source/candidate mean `[0, 1]`,
source/candidate population variance `[0, 0.25]`, source/candidate standard
deviation `[0, 0.5]`, raw variance/standard deviation `[0, +inf)`, both pre-clip
fractions and their sum `[0, 1]`, mean drift `[0, 1]`, and contrast ratio
`[0, +inf)`. Raw mean, covariance, scale, and shift have no physical bound, but
still pass through `canonicalize_float` before any dependent calculation.
`midpoint_fraction` and every other non-null diagnostic float use the same
positive-zero rule.

`source_variance` and `raw_variance` are the normalized population variances
actually compared with `VARIANCE_EPSILON`; both are persisted. Their standard
deviations are then computed from those normalized variances and normalized
again. Correlation is normalized before its gate. This makes the writer's
variance branch exactly reproducible from the manifest instead of asking a
validator to reconstruct it through `source_std ** 2` or `raw_std ** 2`.

### Canonical Derived Diagnostics

The manifest must describe one internally consistent calculation, not merely a
set of individually plausible floats. After counts and all available
re-derivation inputs have been canonicalized, one shared pure function evaluates
derived values in this exact order:

```text
expected_midpoint_fraction = canonicalize_float(midpoint_count / shot_pixels)

expected_source_std = normalize_bounded_stat(
    math.sqrt(source_variance), 0.0, 0.5
)
expected_raw_std = normalize_bounded_stat(
    math.sqrt(raw_variance), 0.0
)

expected_correlation = normalize_bounded_stat(
    covariance / math.sqrt(raw_variance * source_variance),
    -1.0,
    1.0,
)
expected_scale = canonicalize_float(covariance / raw_variance)
expected_shift = canonicalize_float(
    source_mean - expected_scale * raw_mean
)

expected_contrast = normalize_bounded_stat(
    candidate_std / expected_source_std,
    0.0,
)
expected_drift = normalize_bounded_stat(
    abs(candidate_mean - source_mean),
    0.0,
    1.0,
)
expected_preclip_total = normalize_bounded_stat(
    preclip_low_fraction + preclip_high_fraction,
    0.0,
    1.0,
)
```

The re-derivation inputs are counts, source/raw mean and population variance,
covariance, candidate mean/std, and the two pre-clip fractions, subject to the
nullability matrix below. Multiplication occurs before `sqrt`; multiplication
occurs before subtraction in `expected_shift`. Implementations may not replace
these expressions with algebraically equivalent paths such as
`covariance / (raw_std * source_std)`.

The producer uses these expected values for fallback ordering and every quality
gate, and persists those same values. The writer canonicalizes each supplied
derived value and requires its `float.hex()` to match the expected value before
hashing; it does not silently accept a second floating path. The read-only
validator independently re-derives the required fields from stored canonical
inputs and applies the same exact representation comparison to
`midpoint_fraction`, `source_std`, `raw_std`, `correlation`, `scale`, `shift`,
`postclip_contrast_ratio`, and `postclip_mean_drift`.

Derivation stops at the first unavailable or non-finite prerequisite according
to the reason/nullability matrix. In particular, variance fallbacks do not
derive fit fields; a non-finite scale or shift requires `nonfinite_fit`, retains
only the finite derived correlation, and stores scale/shift as null. A finite
physical-bound excursion beyond the ULP allowance remains a hard numerical
contract failure.

This proves algebraic coherence of the persisted tuple, not that its raw moments
came from a now-deleted model-output memmap. Proving raw provenance would require
retaining or separately hashing that private payload and is outside this repair;
the independent final-PNG verifier below covers the durable source/output
contract instead.

Fallback is selected before fitting when `count < MIN_PAIR_COUNT`, or either
variance is at most `VARIANCE_EPSILON`. After fitting, non-finite parameters or
`scale < MIN_POSITIVE_SCALE` select base fallback. `scale == 1e-8` proceeds to
the remaining quality gates; `scale < 1e-8`, including zero and negative scale,
uses reason `scale_below_minimum`.

A non-finite model output is a hard model failure. Finite model output followed
by non-finite aggregate statistics or fitted parameters is a semantic
`base_fallback`, because the source frames remain valid but the calibration is
not identifiable.

## Quality Pass And Commit Pass

The affine cannot be accepted from OLS alone. Before writing final PNGs, scan
the raw memmap and source uint16 frames a second time in fixed tiles. Candidate
mean and variance use the same fixed-tile float64 Chan merge as the fit pass,
never `sum`/`sum-of-squares` subtraction. Over the same fit-eligible pairs,
compute:

```text
preclip64, candidate32 = candidate_tile(raw_tile_f32, scale64, shift64)
preclip_low_fraction
preclip_high_fraction
candidate_mean
candidate_std
postclip_contrast_ratio = candidate_std / source_std
postclip_mean_drift = abs(candidate_mean - source_mean)
```

Candidate variance is normalized to `[0, 0.25]` before its square root. The two
pre-clip fractions are normalized separately; their binary64 sum is normalized
again and that exact result is used by both the gate and diagnostics validator.
Mean, standard deviation, contrast, and drift are likewise normalized before
the following comparisons. No gate compares an unpersisted reconstruction of a
different statistic.

Use OLS only when every condition holds:

```text
correlation >= 0.50
postclip_contrast_ratio >= 0.50
postclip_mean_drift <= 0.01
preclip_low_fraction + preclip_high_fraction <= 0.01
```

Otherwise the entire shot uses `base_fallback`. This quality scan is separate
from the commit scan so no candidate PNG is written before the fallback decision
is final.

The final commit pass yields already encoded uint16 frames:

- accepted ranged frames: apply the shot affine, clip to `[0, 1]`, encode, then
  restore every source `MIDPOINT_CODE` pixel to exact code `32768`;
- individually flat frames: copy the original uint16 frame;
- any shot fallback: copy every original uint16 frame.

Fallback is therefore pixel-level bit-exact, not decode/encode equivalent. The
shot manifest is published only after every encoded frame is atomically written
and hashed.

## Calibration Diagnostics

Shot manifest schema becomes version 2 and includes this self-hashed payload:

```json
{
  "calibration": {
    "mode": "ols | base_fallback | all_midpoint",
    "pair_count": 0,
    "midpoint_count": 0,
    "midpoint_fraction": 0.0,
    "flat_frame_count": 0,
    "source_mean": null,
    "source_variance": null,
    "source_std": null,
    "raw_mean": null,
    "raw_variance": null,
    "raw_std": null,
    "covariance": null,
    "correlation": null,
    "scale": null,
    "shift": null,
    "candidate_mean": null,
    "candidate_std": null,
    "postclip_contrast_ratio": null,
    "postclip_mean_drift": null,
    "preclip_low_fraction": null,
    "preclip_high_fraction": null,
    "fallback_reason": null
  }
}
```

Unavailable numeric values are JSON `null`, never NaN or infinity. When several
conditions fail, the first applicable reason in this exact order is persisted:

```text
source_no_range
too_few_pairs
nonfinite_statistics
source_variance
raw_variance
nonfinite_fit
scale_below_minimum
correlation
contrast
mean_drift
preclip_out_of_range
```

`all_midpoint` is a calibration mode rather than a fallback reason. Boundaries
are accepted: count `2`, scale `1e-8`, correlation `0.50`, contrast `0.50`, mean
drift `0.01`, and out-of-range fraction `0.01` proceed. Diagnostics participate
in the shot manifest fingerprint but not the stage semantic fingerprint.
`nonfinite_statistics` covers either fit-pass or quality-pass Chan state and
outranks every later reason even when it is discovered during the quality scan.

### Strict Diagnostics Schema

Self-hash is not schema validation. A pure standard-library shared contract in
`core/vdpp_calibration.py` owns the constants, exact key set, enums, nullability
matrix, and two explicit entry points:

```text
canonicalize_vdpp_calibration_diagnostics(
    calibration: object,
    *,
    shot_length: int,
    native_shape: tuple[int, int],
) -> dict[str, JSONScalar]

validate_vdpp_calibration_diagnostics(
    calibration: object,
    *,
    shot_length: int,
    native_shape: tuple[int, int],
) -> dict[str, JSONScalar]
```

The writer entry point applies bounded-stat normalization, validates all other
relationships, and returns the canonical object that is hashed and persisted.
The read-only validator applies the same calculation but rejects if any stored
value would change, preventing multiple serialized representations of the same
physical boundary. `commit_shot()` always persists the canonicalizer's returned
object and then runs the read-only validator as an assertion of the writer/read
contract.

The validator rejects missing and unknown keys. `pair_count`, `midpoint_count`,
and `flat_frame_count` must be non-bool Python integers in range. Every non-null
numeric diagnostic must be a finite Python float, not a bool, integer, NumPy
scalar, NaN, or infinity. It also enforces:

- `shot_length` and both native dimensions are non-bool positive Python
  integers;
- `shot_pixels = shot_length * H * W` without overflow ambiguity;
- `0 <= midpoint_count <= shot_pixels` and
  `0 <= pair_count <= shot_pixels - midpoint_count`;
- `0 <= flat_frame_count <= shot_length`;
- `midpoint_fraction` is exactly the binary64 result of
  `canonicalize_float(midpoint_count / shot_pixels)`; Python JSON round-trip
  preserves that exact float, so no tolerance or decimal rounding is allowed;
- every persisted non-null float is positive zero when numerically zero, and
  exact derived-field comparisons use `float.hex()` so `-0.0` cannot pass as
  `0.0`;
- every fraction is canonical under `normalize_bounded_stat(..., 0, 1)`, and the
  normalized binary64 sum of pre-clip low plus high is at most `1`;
- correlation is canonical in `[-1, 1]`; source/candidate means are canonical in
  `[0, 1]`; source variance is canonical in `[0, 0.25]`;
  source/candidate standard deviations are canonical in `[0, 0.5]`; raw
  variance/standard deviation and contrast ratio are canonically non-negative;
  mean drift is canonical in `[0, 1]`;
- source/raw standard deviation, correlation, scale, shift, contrast, and mean
  drift exactly equal the canonical derived-diagnostics graph above whenever
  their nullability group is required; the variance fallback checks use the
  persisted variance values directly, never squared standard deviations;
- mode, fallback reason, persisted threshold boundaries, and reason priority are
  mutually consistent.

For nullability, define moments `M` as source/raw mean, population variance, and
standard deviation plus covariance; fit `F` as correlation, scale, and shift;
and quality `Q` as candidate mean/std, contrast, drift, and both pre-clip
fractions. The exact matrix is:

| Mode / reason | M | F | Q |
| --- | --- | --- | --- |
| `all_midpoint` | null | null | null |
| fallback `source_no_range` | null | null | null |
| fallback `too_few_pairs` | null | null | null |
| fallback `nonfinite_statistics` | null | null | null |
| fallback `source_variance` | required | null | null |
| fallback `raw_variance` | required | null | null |
| fallback `nonfinite_fit` | required | correlation only | null |
| fallback `scale_below_minimum` | required | required | null |
| fallback `correlation` | required | required | null |
| fallback `contrast` | required | required | required |
| fallback `mean_drift` | required | required | required |
| fallback `preclip_out_of_range` | required | required | required |
| `ols` | required | required | required |

`mode="ols"` requires null fallback reason and all accepted gates.
`mode="base_fallback"` requires exactly one listed reason and the causal
condition for that reason while all earlier finite conditions pass.
`mode="all_midpoint"` requires null reason, zero pairs, midpoint count equal to
all shot pixels, midpoint fraction `1.0`, and every frame flat.
`source_no_range` requires zero pairs and every frame flat. For
`nonfinite_statistics` all derived diagnostics are deliberately normalized to
null even if the failure occurred during the quality pass. A
`source_variance` reason requires persisted `source_variance <= 1e-12`; a
`raw_variance` reason requires persisted `source_variance > 1e-12` and
`raw_variance <= 1e-12`. Equal-to-threshold values fall back.

The validator runs in all four trust boundaries:

1. `StabilizedDepthStore.commit_shot()` before constructing or hashing a
   manifest;
2. the shared read-only shot-record audit used by `StabilizedDepthStore` before
   accepting a completed-shot record;
3. `io.resume._validate_stabilized_stage()` before reporting a building stage as
   resumable;
4. complete render-disparity validation before accepting a v2 shot manifest.

Every stabilized JSON write uses `json.dumps(..., allow_nan=False)`. Read paths
still validate finite diagnostics explicitly because Python's parser accepts
non-standard NaN/Infinity tokens by default. A valid self-hash never substitutes
for this schema validation.

### Shared Shot-Record Audit And Resume

The current storage audit and complete render validator duplicate shot-manifest
validation, while resume preflight catches complete-validation failure and
blindly labels every `status="building"` artifact resumable. V2 replaces those
paths with one read-only helper in `core/render_disparity.py`:

```text
audit_stabilized_shot_records(
    root: Path,
    *,
    metadata: dict[str, Any],
    frame_names: list[str],
    shot_plan: list[dict[str, int]],
    native_shape: tuple[int, int],
) -> StabilizedShotRecordAudit
```

The result identifies reusable, invalid, and pending shot IDs and carries the
normalized payload records needed for a complete artifact fingerprint. The
helper validates every declared completed record: sorted unique in-range shot
IDs, fixed manifest path, manifest bytes and SHA-256, self-hash, schema/range,
strict calibration diagnostics, ordered file list, PNG header/native shape,
every payload hash, and shot payload fingerprint.

A malformed `completed_shots` container, non-dict record, boolean/non-integer
ID, duplicate, unsorted ID, or ID outside the shot plan is a structural stage
error and raises. A missing record is pending. A known shot whose manifest,
diagnostics, or payload fails validation is invalid and must be regenerated;
other valid shots remain reusable.

`StabilizedDepthStore.audit()` maps a structural error to `reset_required` and
otherwise consumes the helper's three ID sets. Complete render validation uses
the same result and additionally requires every planned shot reusable, no
invalid or pending shots, and matching stage payload/artifact fingerprints.

`io.resume._validate_stabilized_stage()` no longer obtains partial semantics by
catching an error from the complete-only validator. After metadata, semantic,
state, and execution-plan checks, a building artifact calls the shared audit.
A structural failure reports `invalidate`; otherwise it reports `resume` with
the exact counts of record-valid, invalid-to-regenerate, and pending shots. Its
message states that record validity is provisional until the later CUDA runtime
identity check; the runtime-aware store audit remains authoritative for whether
record-valid shots can actually be mixed into this invocation. Thus preflight
does not claim invalid diagnostics are reusable, while preserving the intended
per-shot recovery behavior.

## Work Files, Locking, And Cleanup

Private files live only under
`03_disparity_stabilized/.vdpp-work/shot_<id>.raw.f32.mmap`; cleanup never scans
or deletes outside that named directory. The production pipeline already owns a
job-level `JobWriterLock`; there is no independent stage writer lock.

A new top-level lightweight helper avoids the eager imports under the
`processing` package. Its only runtime dependencies are the standard library and
the pure `core.constants` module:

```text
depth_surge_3d/vdpp_work.py

cleanup_vdpp_private_work(
    output_root: Path,
    job_lock: JobWriterLock,
) -> None
```

`JobWriterLock` is imported only under `TYPE_CHECKING`; runtime validation uses
its `is_acquired` and `output_dir` attributes. The helper imports no Torch,
model, NumPy, OpenCV, `processing`, or `io` package. It derives the stabilized
root from resolved `output_root` and
`INTERMEDIATE_DIRS["disparity_stabilized"]`; callers cannot supply an arbitrary
stage path and cleanup does not depend on the `directories` mapping.

The helper requires an acquired lock whose resolved output root exactly matches
the argument. Before following either derived child, `lstat`-based checks reject
symbolic links, junctions, and any platform reparse point for both the stabilized
root and `.vdpp-work`; when present, both must be ordinary directories. It
performs a validation pass before deletion: every work entry must be a
non-reparse regular file whose entire name matches
`^shot_[0-9]+[.]raw[.]f32[.]mmap$`. An unknown file, directory, or special entry
is retained and aborts cleanup without deleting any owned file. After successful
validation it unlinks the known files and removes the now-empty work directory.
Any cleanup failure is a hard job error rather than a silently archived or
retained multi-gigabyte file.

The cleanup invariant is: after a job lock is acquired or first accepted,
cleanup is the first job-private workspace mutation, before resume report/audit,
migration, model loading, or pipeline setup. Calls are explicit and idempotent:

- CLI resume calls it immediately after `JobWriterLock.acquire()` and before
  `build_resume_report()`.
- Web `process_video_async()` calls it immediately after lock acquisition and
  before resume report construction.
- `StereoProjector.execute_video()` calls it immediately after acquiring or
  validating a supplied lock and before `_ensure_model_loaded()`.
- `ProcessingOrchestrator.process()` calls it immediately after acquiring or
  validating a supplied lock and before `_setup_processing()`; this covers
  direct API users.

A fresh-interpreter import test for `depth_surge_3d.vdpp_work` proves that its
import closure does not add Torch to `sys.modules`. Later stereo rendering may
independently import Torch and is not part of this cleanup contract.

Ordering is:

1. Under the acquired job writer lock, the entrypoint removes stale private work
   before resume migration, model loading, or any render-source branch.
2. The selected path performs read-only base, scene, and lightweight stabilized
   audits.
3. If the lightweight artifact is complete, return without CUDA checks.
4. For incomplete work, require an available CUDA device, collect its runtime
   identity, and perform the runtime-aware
   audit to identify pending or invalid shots.
5. Run the bounded pre-scan over those shots. This fixes degenerate decisions
   before any model allocation and identifies which pending shots need a raw
   memmap.
6. Preflight disk for pending final PNGs and the largest pending non-degenerate
   shot.
7. Resolve the checkpoint only when at least one pending shot is non-degenerate,
   then load/preflight the model, prepare the store, and mutate pending shot
   state. If all pending shots are semantic copies, prepare and commit them
   without constructing VDPP.

An incomplete requested VDPP stage remains CUDA-only, including when every
pending shot becomes a copy-only semantic fallback. Such a run requires CUDA
availability and CUDA runtime identity, but it does not resolve the checkpoint
or construct VDPP when no non-degenerate shot remains. A complete stabilized
cache remains reusable without CUDA.

Every memmap owner uses `try/finally`. Before unlink on Windows it flushes the
mapping, releases all slices/views, explicitly closes the underlying mapping,
and then unlinks. A cleanup failure is reported but cannot replace an already
active model, cancellation, decode, or disk exception. With no active exception,
cleanup failure is surfaced.

## Disk And Memory Bounds

For pending work, define:

```text
frame_u16_bytes = H * W * 2
conservative_png_bound = ceil(frame_u16_bytes * 1.10)
pending_frames = frames in pending/invalid shots after runtime-aware audit
largest_pending_shot = max pending non-degenerate shot length, or 0

required_bytes = ceil(1.10 * (
    pending_frames * conservative_png_bound
    + largest_pending_shot * H * W * 4
    + conservative_png_bound
    + 1 MiB metadata/manifest allowance
))
```

The second term is the largest raw float32 memmap and the third is one atomic PNG
temporary. Reusable shot payloads are not charged again.

The conservative host-resident array bound is:

```text
normalized input window:          32 * H * W * 4
source uint16 frame:               1 * H * W * 2
midpoint boolean mask:             1 * H * W * 1
native raw output:                 1 * H * W * 4
candidate frame32:                 1 * H * W * 4
canonical encoder conservative:   1 * H * W * (3 * 4 + 2)
all tile scratch combined:         at most 8 * STATS_TILE_PIXELS * 8 bytes
```

The encoder allowance covers up to three full float32 temporaries plus the
uint16 result in addition to `candidate_frame32`. Source-decoded float32,
float64 affine values, candidate tiles, eligibility masks, and boolean-compressed
fit values all belong to the fixed tile-scratch allowance; none is a full-frame
array. No owned float64 host array may contain more than
`STATS_TILE_PIXELS` elements.

The loader fills the normalized host window in place; an unnormalized float32
window must not coexist with it. GPU memory retains the existing pinned VDPP
recurrence bound. Tests instrument owned arrays and prove that a longer shot
changes disk size, not peak resident array shape.

## Cache Identity

The stabilized producer algorithm becomes `vdpp-canonical-shot-v2`. Its
execution plan records all behavior-changing strings and constants.

The values are imported from the shared `core/vdpp_calibration.py` contract by
both execution-plan construction and runtime calibration; they are not repeated
as independent literals. The persisted plan is:

```text
input_normalization = "per-frame-minmax-excluding-midpoint-code-v2"
input_arithmetic = "u16-extrema-float32-subtract-divide-v1"
tile_iteration = "c-row-major-fixed-262144-v1"
midpoint_code_policy = "preserve-u16-32768-heuristic-v2"
calibration_fit = "positive-ols-chan-v1"
degenerate_policy = "prescan-flat-frame-copy-u16-v1"
fit_quality_policy = "corr-contrast-mean-drift-v1"
clip_policy = "preclip-fraction-gate-then-clip-v1"
base_fallback_policy = "copy-source-u16-v1"
fit_pair_conversion = "raw-f32-source-decode-f32-then-f64-v1"
candidate_arithmetic = "raw-f32-to-f64-affine-clip-cast-f32-v1"
candidate_statistics = "float64-chan-over-candidate-f32-v1"
encoding_policy = "canonical-encoder-then-restore-midpoint-v1"
calibration_precision = "float32-input-float64-fit-float32-candidate-v2"
physical_bound_policy = "snap-outward-canonical-unit-ulps-v1"
signed_zero_policy = "canonical-positive-zero-all-diagnostic-floats-v1"
diagnostic_float_equality = "finite-binary64-float-hex-v1"
derived_diagnostics_policy = "recompute-from-canonical-persisted-moments-v1"
variance_diagnostics = "persist-compared-population-variance-v1"
calibration_diagnostics_schema = "strict-exact-keys-derived-positive-zero-v4"
fallback_reason_order = [
  "source_no_range",
  "too_few_pairs",
  "nonfinite_statistics",
  "source_variance",
  "raw_variance",
  "nonfinite_fit",
  "scale_below_minimum",
  "correlation",
  "contrast",
  "mean_drift",
  "preclip_out_of_range"
]
midpoint_code = 32768
model_midpoint_value = 0.5
stats_tile_pixels = 262144
min_pair_count = 2
variance_epsilon = 1e-12
physical_bound_ulps = 4
physical_bound_reference_floor = 1.0
min_positive_scale = 1e-8
min_correlation = 0.50
min_postclip_contrast_ratio = 0.50
max_postclip_mean_drift = 0.01
max_preclip_out_of_range_fraction = 0.01
```

Runtime identity additionally records `numpy_version` and `opencv_version`.
Partial shots generated by a different NumPy/OpenCV runtime are not mixed. The
algorithm, execution plan, and runtime changes invalidate v1 stabilized
artifacts without changing any depth-model setting or base canonical cache key.

## Verification

Automated tests must first fail against v1 behavior and then prove:

1. Per-frame non-midpoint min/max normalization matches the pinned upstream
   input policy while midpoint-code pixels receive model value `0.5`; float32
   last-bit fixtures reject a float64-first implementation.
2. A legitimate source pixel quantized to `32768` demonstrates the documented
   heuristic collision: it is counted and preserved, never claimed as validity.
3. Chan tile statistics match a direct float64 reference on low-variance and
   long synthetic data without cancellation-induced gate changes. Fit and
   candidate-statistics cases immediately below, equal to, and immediately above
   every fixed threshold prove the inclusive boundary contract and reason order.
   Physical-bound fixtures cover one ULP inside, exact boundary, each outward
   distance through four canonical-unit ULPs, and the first value beyond the
   allowance. The 19-zero/19-one three-value-tile standard deviation and an
   above-one correlation roundoff case snap to their exact bounds; farther
   excursions hard-fail. Writer fixtures feed negative zero through bounded,
   unbounded, and derived fields and prove that only positive zero is persisted.
4. A deterministic fake VDPP with a large positive residual produces an
   accepted, non-saturated affine result when quality gates pass.
5. `count < 2`, source/raw variance immediately below, equal to, and above
   `1e-12`, too-small or negative scale, non-finite fit, and every quality-gate
   failure select a bit-exact whole-shot base fallback with the expected stable
   reason. The manifest persists the exact compared variances, and validator
   decisions do not reconstruct them from standard deviation.
6. Non-finite model output, decode, shape, ordering, cancellation, and disk
   faults remain hard failures; cleanup never hides the primary exception.
7. Individually flat frames in a normal shot are model-fed as `0.5`, excluded
   from statistics, and copied bit-for-bit in an otherwise accepted shot.
8. A 61-frame synthetic shot proves overlap frames enter statistics once,
   midpoint preservation crosses the 32-frame boundary, and one affine applies
   without a boundary jump. Existing recurrence-equivalence tests remain
   unchanged.
9. Instrumentation proves the host bound above, memmaps close before Windows
   unlink, no owned float64 host array exceeds `STATS_TILE_PIXELS`, and disk
   preflight uses pending shots plus the largest pending memmap.
10. CLI, Web, `StereoProjector`, and direct orchestrator paths remove stale work
    immediately after lock acquisition/acceptance. In particular, an invalid v1
    stabilized stage with archive migration loses `.vdpp-work` before migration,
    and `legacy_v1` never receives that directory. Complete/preselected cache
    hits and `temporal_postprocessor=off` derive cleanup from the output root,
    not the stage-directory mapping. Symlink/junction, unknown-entry, wrong-lock,
    idempotence, and fresh-interpreter no-Torch-import cases are covered.
11. Independently tampering with `correlation`, `scale`, `shift`,
    `postclip_contrast_ratio`, or `postclip_mean_drift` rejects the supplied
    object at commit. Recomputing every manifest/metadata hash after any one of
    those changes still invalidates partial audit, resume preflight reuse, and
    complete render validation. The same read paths reject a legal `0.0` changed
    to `-0.0`, a deleted calibration/variance, unknown key, NaN/Infinity,
    non-canonical bounded value, or inconsistent mode/reason. All stabilized
    JSON writers use `allow_nan=False`.
12. With matching runtime identity, a building artifact with valid,
    invalid-diagnostics, and missing shot records produces the same three-way
    shot classification in shared audit, resume preflight, and store audit.
    Structural completed-record corruption requires a reset; resume text reports
    invalid shots as regeneration work and marks valid records provisional until
    runtime identity is checked.
13. Shot diagnostics are self-hashed and strictly validated; threshold,
    physical-bound, signed-zero, derived-diagnostic, variance-persistence,
    NumPy, and OpenCV identity changes invalidate or prevent mixed partial resume
    as specified.
14. A complete v1 artifact and a partial v1 artifact are invalidated; the base
    canonical artifact is preserved; complete and partial v2 OLS or fallback
    artifacts validate and resume according to their runtime identity.

The implementation must commit `scripts/verify_vdpp_calibration.py`; it drives
the reported real artifact as a local, non-CI quality gate:

```text
job basename:
  1786983915_BanG Dream Its MyGO S01E03-[1080p][BDRIP][AV1.OPUS]_20260818_002515
base canonical fingerprint:
  8292b1291fe6c552fc3843deadbbf5657efe5a1d734f48f1ef3389dc4253cf75
source raw fingerprint:
  b7529c0a7e348ab599ef1446015d210022603e89d3ff960ae419f7aa50f8cc65
scene manifest fingerprint:
  08defe1eee63a921191df2db93e1c71637548233df30e051c194e5e911f3967a
frame_000500.png SHA-256:
  cc19d33a8dbceee4056ac0e141980228e335e6d3fb6c930a42c7f185791cecb8
frame 500 shot range, zero-based:
  [207, 562)
shot length:
  355
native shape:
  [608, 1080]
ordered source frame-name range:
  frame_000208.png .. frame_000562.png
ordered source payload fingerprint:
  PENDING_RECAPTURE
frame 500 non-midpoint source mean/std:
  0.05490560770408996 / 0.027351981535192124
frame 500 midpoint count/fraction at 608x1080:
  522561 / 0.7958104897660818
```

The ordered source payload fingerprint is exactly:

```text
canonical_json_hash([
  {"name": frame_name, "sha256": sha256(file_bytes)},
  ... in shot order ...
])
```

The previously inspected external job directory is no longer present, so its
354 additional hashes cannot be reconstructed from the metadata or the recorded
frame-500 hash. The placeholder must be replaced from a restored or regenerated
byte-identical artifact before implementation sign-off; inventing a value would
defeat the gate. Recapture rechecks all 355 ordered file hashes, frame-500 hash,
base/raw/scene fingerprints, shot range, native shape, source mean/std, and
midpoint count/fraction as one indivisible fixture record. If regenerated bytes
differ, every field is captured as a new fixture; only replacing the payload
fingerprint is forbidden.

While the fixture contains `PENDING_RECAPTURE`, the verifier exits non-zero
before CUDA/model loading and implementation sign-off remains blocked. There is
no flag to skip or weaken this fixture identity check.

The verifier hashes all 355 source inputs and rejects a payload-fingerprint
mismatch before loading VDPP. It validates the resulting manifest and
diagnostics through the shared strict audit, including diagnostic correlation at
least `0.50`, contrast at least `0.50`, mean drift at most `0.01`, and pre-clip
out-of-range fraction at most `0.01`.

That is not the quality verdict. After commit, the script independently reads
every base and stabilized uint16 PNG in `[207, 562)`, checks order and native
shape, and recomputes from final PNG values rather than diagnostic aggregates.
Over the same fit-eligible population -- non-midpoint pixels from source frames
with non-midpoint range -- it reports:

```text
actual_pair_count
actual_midpoint_count
actual_midpoint_fraction
actual_flat_frame_count
actual_source_mean
actual_source_variance
actual_source_std
actual_output_mean
actual_output_std
actual_output_to_source_correlation
actual_output_contrast_ratio = actual_output_std / actual_source_std
actual_output_mean_drift = abs(actual_output_mean - actual_source_mean)
actual_endpoint_counts = output code-0/code-65535 counts and fractions
```

The verifier implements its own fixed-order float64 Chan accumulator over the
decoded final PNGs; it does not import the producer's fit, candidate, or
diagnostic accumulator. Its independent implementation nevertheless follows the
specified source decode, C-row-major tile boundaries, Chan merge order, and
positive-zero/bounded-stat canonicalization. Therefore actual pair, midpoint,
and flat-frame counts must exactly equal their diagnostics; actual midpoint
fraction and actual source mean, population variance, and standard deviation
must have the same `float.hex()` values as `midpoint_fraction`, `source_mean`,
`source_variance`, and `source_std`. These are exact source-contract checks, not
quality tolerances.

The verifier requires actual correlation at least `0.50`, actual contrast ratio
in `[0.50, 1.05]`, and actual mean drift at most
`0.01 + PNG_QUANTIZATION_ALLOWANCE`, where the fixed conservative allowance is
exactly `1.0 / 65535.0`. It also requires the final actual output mean and
standard deviation to match diagnostic `candidate_mean` and `candidate_std`
within `PNG_QUANTIZATION_ALLOWANCE + 4 * math.ulp(1.0)`.

Every one of the 355 source midpoint-code locations, not only those in frame
500, must remain exact output code `32768`. The accepted whole-shot output must
be non-degenerate and not pixel-identical to base. Pre-clip fractions remain
diagnostics-only because clipping makes them unrecoverable from PNGs; post-clip
mean, standard deviation, correlation, contrast, drift, and endpoint counts are
always measured independently. Base and raw fingerprints must remain unchanged.
The copyrighted episode is not added to the repository; synthetic CI fixtures
cover the same final-PNG verification logic.
