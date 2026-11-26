# Technical Architecture

## Pipeline

Depth Surge 3D converts monocular video through restartable, fingerprinted
stages:

```text
source video
  -> 00_original_frames
  -> 01_scene_data (candidate cuts, final segments, samples, bounds)
  -> 02_depth_raw (native model output and representation metadata)
  -> global barrier: all raw inference complete
  -> selected Stage 3:
       relative: 03_disparity_maps (canonical relative disparity)
         -> optional 03_disparity_stabilized (VDPP, one transaction per final shot), or
       03_metric_geometry (metric inverse depth, validity, focal data)
  -> 04_left_frames + 04_right_frames (forward-splat DIBR)
  -> optional projection, crop, and upscale stages
  -> 99_vr_frames
  -> encoded video with optional source audio
```

The frame, scene, raw-depth, canonical, and stereo stages have separate
metadata. `00_original_frames/metadata.json` anchors the exact encoded PNG
bytes to the source-video SHA-256 after extraction completes. A stage is reused
only when its schema, frame manifest, settings, model identity, and upstream
fingerprints match.

VDPP is a derived artifact stage, not an estimator backend. It consumes the
same canonical uint16 relative disparity produced by V2, V3, See-Through, and
relative MoGe-2.
The fixed adapter uses 32-frame windows, four retained observations, stride 28,
FP32, and the upstream affine continuation rule. A window never crosses a final
scene boundary. The depth-model owner is released before VDPP construction,
and VDPP is released before stereo rendering, so both neural models are never
resident together.

## Depth Contract

Every estimator returns a `DepthBatch` containing float32 values and one
explicit `DepthRepresentation`:

- `METRIC_DEPTH`: positive physical distance; converted with a safe reciprocal.
- `INVERSE_DEPTH`: near-is-large score; passed through.
- `RELATIVE_DEPTH`: affine-relative distance; converted with linear negation.

Non-finite values are invalid. Non-positive values are invalid only for metric
physical distance. Estimators never perform per-frame min/max normalization.

The backend registry is the single owner of backend variants, capabilities,
factories, availability, and report identity. A `DepthBatch` may also carry a
typed `PinholeCameraBatch`; MoGe-2 supplies one normalized horizontal focal
value per frame. Existing raw schema v2 depth-only data remains readable. New
raw schema v3 writes exact depth payload membership plus camera-model metadata;
`pinhole_fx` frames atomically store depth and a scalar float32 focal value.

Raw output is stored at model-native resolution. The first inference chunk
selects `float16` or `float32` storage for the whole raw directory and records
the choice. A preflight estimate checks disk space before long inference.
Remote DA3 and See-Through repositories are resolved to one immutable Hub
snapshot before loading; that resolved artifact identity is part of the raw
fingerprint.

Framewise estimators that explicitly advertise exact repeated-batch semantics
retain at most one completed input batch and its typed `DepthBatch`. A following
batch may reuse that result only when its batch shape, dtype, encoded source
sizes, and every decoded BGR value are identical. The whole batch, including
its position order and camera values, is reused; short tail batches never reuse
a full batch. Encountering an already completed chunk during resume clears this
transient state. This keeps model batch context and stored values unchanged
while avoiding repeated inference for consecutive holds or duplicated frames.

### See-Through Inference Overhead

The opaque CUDA runner keeps scheduler timestep indices on the CPU and copies
the timestep vector once for the UNet. DDIM coefficient tables live on the CPU;
passing CUDA indices to them caused 22 scalar device-to-host reads per four-step
batch. The regression test now observes zero such reads without changing the
scheduler, timestep values, noise seed, precision, or denoising step count.

For uint8 video frames, resize runs on BGR first, then RGB channels are copied
directly into one preallocated inference-size batch. This avoids full-source
RGB copies and the final stack allocation. Non-uint8 inputs retain the original
RGB-before-resize and stack path, preserving both SIMD rounding and OpenCV's
returned dtype/byte order rather than assuming it matches the input array.
The preprocessing fingerprint is unchanged: this is an exact-output
optimization, not a new depth algorithm.

The September 4, 2026 local comparison against `a19e973` used RTX 4090,
PyTorch 2.13.0+cu130, BF16, four steps, batch four, and the pinned See-Through
snapshot `aa7a892f83ff68d7b09186a405ba08d5d33f770f`. It sampled 48 real frames
across four parts of the 473-frame source, from 1920x1080 to native 1080x608.
Three alternating A/B pairs measured estimator calls after loading and warmup:

| Measurement | Baseline | Candidate |
| --- | ---: | ---: |
| Median depth estimation for 48 frames | 17.113 s | 16.199 s |
| Preprocessing peak allocation, four 1080p frames | 21,982,480 bytes | 11,820,024 bytes |

Estimator latency fell 5.34%; every measured float32 depth array had identical
SHA-256 `4be99653bb79ff427f9cd94bbc89b8337f3d8383183e72ebcb80ec8bf2a04b93`.
Independent tail batches of one, two, and three frames also matched exactly.
All 473 source frames produced identical resized RGB pixels. These measurements
exclude extraction, model loading, storage, stereo rendering, and final video
encoding; they do not establish a 5.34% reduction in complete-task time.
The local harness and reports are retained under
`build/see-through-depth-profile-20260904/` (`compare.py`, `comparison.json`).

Two acceleration probes were deliberately not enabled. Capturing UNet and VAE
calls with CUDA Graphs preserved exact output but reduced warm estimator time
only about 4%, with additional startup and state-management cost. Compiling only
the UNet with `torch.compile` also saved about 4% warm, but the first 16-frame
call took 284.365 seconds and depth differences reached 0.19921875 on a [0,1]
scale. Neither result justifies a default backend switch. TensorRT was not part
of those probes. CUDA-event samples put UNet near 44% and combined
VAE encode/decode near 44% of warm batch wall time, so UNet-only acceleration
must not be presented as acceleration of the entire depth stage.

The subsequent RGB-upload change keeps video bytes as uint8 until they reach
CUDA, then normalizes on-device for FP16/BF16 VAE inputs. All 256 byte values
round exactly as the old CPU-FP32-then-cast path in both dtypes. FP32, CPU,
and non-uint8 inputs retain the original normalization order. Negative-stride
inputs are materialized as contiguous bytes before upload. No model precision,
depth fingerprint, scheduler setting, or retention policy changes.

For a four-frame 1080x608 RGB batch, 32 alternating microbenchmark samples
(first four discarded) measured BF16 normalization/upload at 34.763 ms versus
1.751 ms; FP16 measured 34.665 ms versus 1.802 ms. This is only the input step.
Four full 48-frame A/B pairs all preserved exact depth output, but concurrent
external GPU graphics/compute load invalidated the latency comparison. The
owned benchmark was stopped before remaining validation completed; no complete-
task or additional estimator speedup is claimed from those interrupted runs.
Observations and a rerunnable
harness are retained as `rgb-transfer-observations.json` and `compare_rgb.py`
in the same local profile directory. The earlier 5.34% comparison above covers
only the earlier resize/scheduler changes, not this RGB-upload addition.

A later replay after the interfering process had exited completed four
alternating A/B pairs on the same 48 real frames. Both paths included the
resize/scheduler changes; only RGB normalization/upload differed. Median
estimator latency was 16.136950 seconds for CPU float32 staging and 15.676624
seconds for uint8 GPU upload, a 2.85% reduction in this comparison. All eight
measured float32 depth arrays, both warmups, and independent tail batches of
one, two, and three frames matched exactly. The 48-frame SHA-256 remained
`4be99653bb79ff427f9cd94bbc89b8337f3d8383183e72ebcb80ec8bf2a04b93`.
`rgb-transfer-comparison.json` and `rgb-transfer-idle-comparison.log` contain
the completed run. This is an additional isolated estimator measurement,
not a complete-task improvement or an additive claim with the earlier 5.34%.

A VAE channels-last probe was also rejected: 16-frame eager calls measured
5.260/5.265 seconds, versus 6.007/6.063 seconds with the changed layout. It was
slower and failed exact equivalence (maximum depth difference 0.234375). This
does not establish behavior on every device, but provides no reason to change
the default on the measured RTX 4090.

Before source prefetch was implemented, `_infer_raw_chunk` loaded and stacked
PNGs, invoked the estimator, then completed `RawDepthStore.write_batch` before
any next batch started. Historical CPU-only 48-frame probes under shared system
load measured PNG load/stack at
1.855/2.949/3.558 seconds and float16 NPZ compression to memory at
2.097/3.050/3.158 seconds (4,303,366 encoded bytes). These are not idle-machine
end-to-end timings and exclude NPZ disk writes. That observation motivated the
current bounded prefetch: `_load_raw_chunk_array` can decode the next batch
while `_infer_raw_chunk` performs inference and ordered raw writes on the caller.
The current limits and failure behavior are documented under Memory And I/O.

The September 8 overhead trials retained the previous Quality device offsets
in both arms. An initial combined range-check/inference-context candidate
matched all outputs in eight complete 48-frame jobs and six complete 195-frame
jobs, including warmups and a three-frame tail. Whole-job timing varied, and
the longer run did not establish standalone inference-context acceleration.
That context change was therefore deferred; opaque inference still uses
`torch.no_grad()`. The range-check-only candidate reduced helper time and
temporary allocation, but a fresh 48-frame ABBA comparison did not demonstrate
a complete-job benefit: medians were 52.369 seconds before and 53.401 seconds
after, excluding separately measured model loading. All six final-media and
diagnostic comparisons matched exactly, with equal measured CUDA allocator
peaks. This does not prove a causal slowdown on a shared workstation, but it
fails the acceptance threshold for a speed optimization; the range-check
candidate was also withdrawn. Neither candidate changes the production
default, model settings, precision, or cache fingerprints. Detailed evidence
and rejected CUDA Graph, VAE grouping/layout, and decoder-thread probes are in
`build/raw-write-profile-20260908/results.md`.

### TensorRT Feasibility Probe

A subsequent September 4 probe installed Torch-TensorRT 2.13.0 and TensorRT
11.0.0.114 into `build/tensorrt-probe-20260904/packages`, not the application's
environment. It used the same checkpoint, BF16, four denoising steps, batch
four, and native 1080x608 resolution. No quantization or model-setting change
was made. TF32 was disabled in the TensorRT builder.

The initial JIT dry run reported full converter coverage: UNet 3222/3222 ops,
VAE encoder 206/206, and VAE decoder 254/254. Separate AOT runs then built and
executed actual full-component engines, followed by a combined run of all
three engines. All tested engine outputs were finite and repeatable. Saved
UNet, encoder, and decoder artifacts each reproduced their pre-save component
output exactly after reload; this is not equivalence to native PyTorch.

The combined 16-frame sample covered four positions in the real video. Native
estimator calls bracketing the experiment took 5.455 and 5.383 seconds; four
warm TensorRT calls took 3.459, 3.401, 3.410, and 3.420 seconds. These are
feasibility observations, not a controlled sustained A/B or complete-task
benchmark. Report files retain `latency_claim_valid=false`. PyTorch allocator
peaks also exclude TensorRT allocations and must not be reported as total VRAM
savings. The first observed component builds took 217.134 seconds for UNet,
65.825 seconds for the encoder, and 43.342 seconds for the decoder. In the
combined run, reloading the saved UNet alone took 23.209 seconds; startup and
artifact handling therefore need separate measurement from warm inference.

Native outputs matched before and after every experiment, but the combined
TensorRT depth output did not: maximum absolute difference 0.453125, mean
0.001117783, and p99 0.01171875 on the [0,1] depth scale. These are differences,
not percentages of lost quality. A downstream diagnostic used the original
scene bounds, unchanged float16 raw storage and uint16 canonical encoding,
and the production Fast offset functions at 1920x1080. On those 16 frames,
the largest left-eye sample-offset change was 6.875 pixels; the largest
per-frame p99 was 0.3125 pixels. This diagnostic holds scene bounds fixed and
does not test recalibration, final rendered images, Quality geometry, or
temporal stability. It neither proves a visible regression nor certifies
quality preservation.

TensorRT remains an isolated experiment, not an application backend. The
throwaway harness, per-component and combined JSON reports, saved sample
depths, and geometry diagnostics are retained under
`build/tensorrt-probe-20260904/`. A backend change requires representative final
stereo/temporal quality validation and an end-to-end timing comparison; the
current results do not satisfy that gate.

## Relative And Metric Geometry

Relative mode remains the default and derives `03_disparity_maps`. Experimental
metric-camera mode derives `03_metric_geometry`. These are independent Stage-3
directories: only the selected missing stage is generated, and a valid inactive
stage is preserved. The metric store holds native-resolution inverse depth,
depth validity, and normalized horizontal focal data. When intermediates are
retained, it also holds one viewable uint8 depth-preview PNG per frame; these
PNGs are diagnostics and are not projection inputs.

Both paths produce a common `StereoGeometryFrame` containing near-score,
total-disparity fraction, and source-validity arrays. The common stereo
renderer therefore does not branch on depth-model identity. Metric projection
uses the virtual capture-camera baseline, focal estimate, depth, and one
convergence distance. `virtual_baseline_mm` is not viewer IPD.

The metric formula is evaluated in pre-crop source coordinates, transformed by
the exact retained center-crop width, and clamped as total left-to-right
disparity in retained final-output coordinates. It is then converted back to
renderer coordinates. The common center crop and axis-aligned resize run later,
so changing final output width does not alter the capped projection fraction.
Unclamped inverse depth remains the z-order key.

Every metric geometry build resolves and persists one deterministic
clip-global automatic convergence value from valid positive metric samples,
even when the active render requests an explicit distance. This finalized
metadata is a barrier: metric stereo and live preview do not start before it
exists. Baseline, explicit-versus-auto convergence, disparity cap, and render
width are stereo settings rather than metric-stage identity.

MoGe's predicted mask is depth confidence, not source-image alpha. Metric
projection excludes masked pixels from convergence and clamp statistics, gives
them the lowest visibility rank, and projects them on the infinite-background
plane. Their source colors therefore remain available to fill the background
instead of being replaced with black.

## Scene Canonicalization

Scene cuts are analyzed from source-frame luma histograms before depth
canonicalization. Candidate manifests carry `status: candidate`; only a
`status: final` manifest may drive canonical output.

After all raw frames exist, deterministic depth samples are pooled per candidate
segment. Adjacent segments are merged left-to-right, repeatedly until stable.
Merged samples are the union of their child sample sets. Final 2nd/98th
percentiles are then computed once for each final segment.

Canonicalization is a pure function of raw depth, representation, final scene
ID, and final bounds. Output is relative disparity in `[0, 1]`, where `1` is
near. Empty and flat scenes produce `0.5`. Canonical maps use `uint16` PNG and
store exact upstream fingerprints in `metadata.json`.

## Stereo Renderer

Canonical relative disparity is resized to the render target with unchanged
host bilinear interpolation. Both full-frame eye offset maps use the legacy
float64 operation order and signed int32 fine-lane storage. Eligible relative
Fast and Quality CUDA rendering builds them directly on device; other paths
use the host:

```text
scale64 = (float64(target_width) * float64(stereo_strength)) / float64(200)
shift64 = ((float64(r) - float64(convergence)) * scale64) * float64(16)
left_offset  = ceil(shift64 - float64(0.5))
right_offset = ceil(-shift64 - float64(0.5))
```

Each source pixel represents one projected horizontal unit footprint. The
renderer preserves its occupancy with 16 fixed samples: source column `x`
writes lanes `16*x + offset + {0,...,15}`. Every fine lane has one strict
z-buffer winner. The ordering key packs nonnegative float32 disparity bits and
the inverted full-frame source index into one int64 value:

```text
key = (float32_bits(r) << 32) | (0xffffffff - source_index)
```

One integer `amax` therefore selects the largest disparity and, for exact depth
ties, the lowest source index. There is no epsilon visibility window,
floating-point atomic sum, or fixed layer count. CPU and CUDA reduction order
cannot change the winner.

Invalid splat pixels are represented by explicit masks, never inferred from
color. `background` occlusion fill runs on the fine grid and extends the farther
of the two discrete boundary winners across bounded horizontal gaps. Equal
depth uses distance and then the left boundary as tie-breaks. `none` leaves
unresolved lanes black. Sixteen lanes are combined with a fixed balanced
addition tree, multiplied by `0.0625`, and converted with ties-to-even rounding.
The public valid mask means at least one lane was covered before fill; the
public hole mask means all 16 lanes remain unresolved after fill.

Quality geometry selects stable low-resolution regions from the partition's
known dense label cardinality in the prebuilt native extension, avoiding the
former NumPy maximum and histogram passes. The same extension computes stable
samples, stable surface components, and interior support. Surface connectivity
is exactly the original region identifier plus the stable-sample bit. Its union
forest always attaches the larger root to the smaller root, so the root remains
the canonical row-major pixel without a separate canonical-key vector. Equal
parents are rejected before root lookup, jump filtering shares one raster scan,
and surface materialization shares the support-count scan. The fused path also
avoids the former float64 `region * 2` synthetic-displacement raster and repeat
masked label histograms. It does not spawn intra-frame workers; frame-level
concurrency remains the only parallel layer. The separate Python operations
remain as a differential-test oracle.

The fused stable-surface builder releases its initial-region counters before
allocating surface counters. After the surface scan it also releases the
union-find and root-to-region arrays before allocating fallback flags. This
avoids retaining successive phases' workspaces together when nearly every
native pixel forms a separate region, without changing traversal or outputs.

Geometry resampling first searches exact Chebyshev rings in the source raster,
up to 16 pixels by default. After each ring, a conservative lower bound proves
whether every unseen pixel is farther than the current best sample; equal
distances retain the canonical row-then-column tie-break. Queries that cannot be
proved locally fall back to the existing exact native k-d tree, but that tree is
built only from stable samples belonging to unresolved query regions. Local
query batches use deterministic block-parallel native workers and debit the
visit budget in query order. The hybrid path changes only search diagnostics,
not selected samples or rendered pixels.

Nearest queries retain four input vectors and two owned uint32 result vectors,
totalling `32Q` bytes for `Q` edge queries. Native calls and gathers use chunks
of at most `min(65536, max(1, Q // 8))` queries. Their `38B` temporary bound fits
the existing `40Q` query reservation plus at most 30 fixed bytes from the runtime
allowance. All local queries precede all k-d fallbacks, and output primitives
are written only after every query succeeds. Visit totals and maxima aggregate
without retaining a full uint64 result vector or overflowing uint32 sums.
Construction-only low-resolution fields are released before querying. The
allowed-region mask and native builder's seen flags are charged separately
from the persistent index, then released before fallback queries. For `N`
render pixels and `P` native pixels, nearest-phase admission is conservatively
`61N + 19P + 5` bytes for relative geometry and `69N + 28P + 5` for metric
geometry, plus the unchanged runtime and thread allowances. The absolute host
cap remains 512 MiB; these are phase bounds, not whole-process measurements.

Nearest-label region upsampling precomputes the exact half-pixel source row and
column mappings once per call, then copies deterministic output-row partitions
under the shared native worker cap. Sparse region seeds apply the stable-pixel
and stable-surface filters before leaving native code, avoiding a writable
full-frame copy and NumPy lookup raster. Geometry also takes mutable ownership
of the native geodesic owner raster and freezes it with the other final outputs,
rather than copying that raster between the geodesic and matte stages.

Geometry integrity checks hash the C-contiguous region buffer directly rather
than materializing an additional Python `bytes` object. Both before/after
checks remain in place and retain the same C-order byte digest, including for
non-contiguous input after normalization. This removes an 8,294,400-byte copy
per check for a 1920x1080 uint32 raster without changing the memory admission
budget or renderer identity. The buffer must not be mutated concurrently with
hashing; current geometry construction owns these region maps.

Relative Quality updates the final edge-band disparity in one native raster
pass rather than issuing one NumPy gather and scatter per output row. The final
maximum four-neighbour displacement jump is reduced directly from the binary64
raster by deterministic row-partitioned native workers, avoiding two full-frame
temporary difference arrays. Foreground-matte expansion still recomputes its
derived displacement afterward: the compact sparse winner records intentionally
carry output primitives but treat the first displacement raster only as the
expansion eligibility input.

Foreground-matte winner selection is limited to the already frozen edge band,
which covers every four-connected boundary in the final geodesic region map.
The native loop checks candidate destinations in the same up, down, left, right
order and leaves all non-candidates unexpanded. It also performs the finite-value
and canonical-region validation once; Python verifies only layouts and
writeability instead of rescanning the same full-frame values.

## Memory And I/O

CUDA Fast scatter first attempts two bounded Triton kernels. The first computes
the 16 packed winner candidates per source pixel directly into the winner
buffer with integer atomic max; the second decodes and gathers the winners
without expanded target, candidate, or decoded-index tensors. A Triton import
or compilation failure disables this path for that CUDA device and preserves
the byte-identical PyTorch scatter fallback.

CUDA Quality visibility reuses the same packed winner reduction, but its
default compact path does not gather the general fine-grid disparity, validity,
or winner-index planes. A Quality-specific Triton kernel reads the 16 winners
for each output pixel, accumulates the exact uint8 BGR sum, multiplies by the
binary-exact `1/16`, and emits only the downsampled float32 BGR value plus one
packed missing-lane mask. The existing native histogram and background repair
consume that mask unchanged. Triton unavailability or compilation failure
falls back to the general byte-identical Quality band path.

Quality background repair prepares its CPU Telea input in one native call.
The call validates coverage once, uses a `17 x 256` uint8 lookup table for the
frozen ties-to-even colour normalization, constructs the directional exclusion
mask with row sliding windows and a three-row vertical buffer, and emits repair
bits, the missing-pixel mask, all counts, and the exact nonzero mask bounds in
the same traversal. Masks whose padded bounds occupy less than 90 percent of the
frame run OpenCV Telea only on that region. The padding is `radius + 1`, which
preserves the full-frame Telea result byte for byte; larger masks retain the
full-frame path. Four-connected component semantics and final proxy composition
are unchanged.

Both paths use deterministic per-pixel memory bounds. On an RTX 4090, the
Triton live set measured `855.819` temporary bytes per band source pixel during
a complete 4K render. Its configured `1088 B/source-pixel` retains 27.6 percent
measured headroom and permits 64 complete rows, or 34 bands per eye, under the
256 MiB baseline budget. Default Fast CUDA rendering may grow that budget only
enough to keep an eye at or below 17 bands, subject to a 512 MiB cap and one
eighth of currently free VRAM. A 4K Triton render therefore uses 128-row bands
when memory is available, while 1080p remains at the 256 MiB baseline. Explicit
budgets, CPU rendering, low-free-memory cases, and failed CUDA memory queries
remain unchanged. The PyTorch fallback retains the conservative
`1280 B/source-pixel` bound because it still materializes expanded indexes,
masks, and packed candidates.

Public Fast rendering leaves the first CUDA OOM exception scope before releasing
memory and retrying. Failed-band tensors retained by that traceback are therefore
released before the retry starts. The band-size policy, single retry, retry rows,
and final error reporting remain unchanged.

Relative Fast compact preparation retains only an owned float32 near-score
raster and its all-valid source mask. It does not construct the common float64
disparity plane, which the legacy relative-offset formula never reads. The
owned near-score tensor and its NumPy view share storage rather than retaining
a second full-raster copy. Input and control validation, bilinear resizing,
float64 offset operation order, and half-lane rounding are unchanged. Public
common geometry, metric preparation, and Quality retain their existing paths.

Fast CUDA rendering attempts one bounded preload of the source image, near
score and source-valid mask. For relative compact frames, one optional Triton
kernel uses the already uploaded near score to produce both int32 eye maps.
It keeps each original unfused float64 operation; runtime control scalars travel
as binary64 bit patterns rather than Triton's default float32 scalar arguments.
There is no offset readback, host int32 map, or second upload in the supported
path. The kernel consumes only the original CPU resize of validated `[0, 1]`
scores. With at most `2**31 - 1` pixels and validated controls, a conservative
two-unit score/convergence distance bounds the fine shift below the width;
both rounded offsets and even masked tail indexes fit int32. Unsupported sizes
or layouts retain the checked host builder instead of weakening validation.

Public Fast relative and all metric paths retain host float64 offset construction, and
all host float64 scratch is released when it returns. The relative compact
fallback builds one host pair lazily if Triton is unavailable, compilation
fails, or the full-frame preload fails. Its holder retains only CPU inputs and
optional CPU outputs, never CUDA tensors. Compilation failure disables only
this helper on that device; the known Triton driver OOM spelling reaches the
existing Torch OOM recovery, and other runtime errors still propagate.

Both eyes reuse the same resident inputs and offsets. For a 4K uint8 relative
frame this static set remains about 126.6 MiB, separate from the scatter scratch
budget. A failed preload drops partial allocations and uses host-sliced bands;
a band OOM retries with the original baseline-derived smaller height and no
full-frame preload. The first failure's exception scope ends before memory
release and retry, so traceback frames cannot keep resident CUDA planes alive.
CPU rendering always uses the band-sliced path. The two paths remain exact.

Without explicit profiling, `fast_offset_build` measures CPU construction or
CUDA dispatch, not asynchronous kernel completion; completion is synchronized
there only when `profile=True`. Complete-render and complete-task timings,
rather than isolated stage counters, determine the measured speed benefit.

Background fill and downsampling still complete before the next band, final
eye outputs remain host arrays, and no full-frame expanded index, winner, or
16-lane buffer is retained on the device. Fast CUDA first attempts value-only
nearest-boundary scans in two optional Triton kernels. Inclusive maximum and
minimum scans run in tiles of at most 2,048 fine lanes; a second kernel carries
each tile's extrema across the row. Both int32 boundary maps and the original
`-1` / row-width sentinels remain exact, without Torch's unused arg-index maps
or row-flip buffers. Rows wider than 65,536 fine lanes, unsupported tensors, and
unavailable or failed scan compilation retain the original Torch
`cummax`/`cummin` path. Scan failure state is independent of splat, fill, and
output acceleration; allocation OOM, including the pinned NVIDIA driver's
ordinary RuntimeError spelling, reaches the existing smaller-band retry.

For `H` band rows and fine width `F`, scan outputs occupy `8HF` bytes, the shared
column vector `4F`, and temporary tile summaries `8H * ceil(F / 2048)` bytes.
Summaries are at most 32 KiB for the default 128-row bands in the supported
width range. Actual RTX 4090 launch metadata at 1080p, 4K, and maximum supported
row width reports no driver-local register spills or global scratch for either
kernel. This avoids the hidden local-memory traffic of a rejected whole-row
scan candidate; it is not a bound on all CUDA-driver or process memory.

One optional Triton fill kernel then fuses candidate disparity reads, left/right
selection, gap classification, RGB gather, and hole/fill mask publication.
Unsupported inputs and compilation failures fall back to the byte-identical
Torch operations; fill-kernel failure state is isolated from the other kernels.
The pinned Triton driver's exact OOM RuntimeError is normalized to Torch's OOM
type so the existing smaller-band retry runs. Other runtime errors propagate,
and neither kind is cached as permanent fill-kernel unavailability.
Fast CUDA lane coverage is
recovered exactly from the packed 16-bit missing mask. Hole-run histogram scans
begin synchronously and switch to one native background worker only when a
measured band is expensive enough to hide useful work. At most two packed bands
may be pending, their native updates remain serialized, and the renderer joins
the worker before exposing complete frame diagnostics. Each eye also drains
pending updates before `HoleRunDiagnostics` makes its histogram arrays read-only;
waiting only at frame exit allowed a delayed native task to encounter frozen
buffers. The same executor can serve both eyes. CPU and native-unavailable paths
keep the synchronous analyzer.

For uint8 CUDA input with the native analyzer, each band combines the RGB output,
packed missing mask, repair flags, and three int64 counters into one host download
instead of four. The payload remains exactly `6P+24` bytes for `P` output pixels:
counters first preserve int64 and uint16 alignment even for odd pixel counts.
Rounding, clipping, masks, counters, and native analysis are unchanged; other
dtypes and fallback paths retain their original conversions. A queued mask view
keeps its entire packet alive, bounded by two pending tasks plus the current band.
The packet is not retained after its native task and render-band consumers finish.

In the preceding download-batching round, before output fusion, a warmed ABBA
test on Windows, CPython 3.11.11, Torch 2.13.0+cu130, and an RTX 4090 used
192 different cached real source/depth frames at 1920x1080, default
`profile=False`, and RGB24 NVENC output at 3840x1080. Both variants included the
required per-eye completion barrier. Median time from cached inputs through
encoding, independent validation, publication, cleanup, and terminal audit fell
from 99.17 s to 96.17 s (3.0 percent); compact rendering fell from 50.53 s to
46.93 s (7.1 percent). Downloads fell from 13,824 to 3,456 with identical byte
counts. All streamed RGB, encoded MP4, decoded video, and aggregate diagnostic
identities matched, and all three independent validation passes remained enabled.
Depth inference and input extraction are outside these timings. A final 48-frame
ABBA repetition also preserved all identities and reduced median completion time
from 31.72 s to 31.00 s (2.3 percent), with compact rendering 6.9 percent faster.
An earlier 48-frame trial of the pre-fix candidate was 5.2 percent slower. These
are sample-specific observations, not a guaranteed speedup for every clip.

For that download-batching round, the measured CUDA allocation peak was unchanged
at 206,225,920 bytes for the 192-frame trial and 472,781,312 bytes for a separate
4K single-frame probe. The latter used a nearest-upscaled real image with
unchanged native depth, not a native-4K video benchmark. Its outputs also matched,
and its largest packet was
2,949,144 bytes, within the existing default static-plus-scratch CUDA bound.
Process-memory samples at 20 ms intervals are observational, not a hard host
peak-memory proof.

Fast uint8/native CUDA bands can produce the same packet layout directly with
two optional Triton output kernels. The first reduces the 16 RGB samples using
the original adjacent-pair float32 addition tree, rounds ties to even, and
writes the packed missing mask and repair flags without the separate Torch
output intermediates. The second sums bounded per-block integer counts into
the three int64 header fields. Non-finite reduced channels are still counted
and rejected after native lane analysis, preserving failure precedence. Neither
projection, background-fill selection, output precision, nor diagnostics change.

The renderer requests complete repair bytes: the existing 16-bit validity bitmap
adds `0x02` for no coverage, `0x01` for partial coverage, and no coverage flag for
full coverage. The first output kernel combines these with the existing filled
and unresolved flags; the host copies the result without rescanning coverage or
merging the flags again. The low-level helper defaults to legacy repair bytes
unless explicitly requested. Each band that actually falls back to the Torch
producer retains the original CPU coverage checks and merge, including frames
with both fused and unfused bands.

For a band that actually produced complete fused repair flags, the host no longer
needs coverage before processing the next band. The existing native histogram
worker also writes that band's coverage view, avoiding the foreground lookup
and its temporary array. The first native scan stays synchronous; a measured
cost of at least 0.5 ms can enable the single worker for deferred coverage,
while bands needing immediate coverage retain the 2 ms threshold and lookup.
An explicitly supplied threshold overrides either default. The contract is chosen
from each actual payload, not from assumed kernel availability, so mixed fallback
frames remain correct. Pending work retains both packet and coverage views;
the two-item bound and per-eye completion barrier are unchanged. OOM retries
and foreground failures still drain and shut down the old queue first.

The count partials are limited to 8,192 blocks and 96 KiB. Larger custom bands,
unsupported tensors, and unavailable or failed Triton output compilation retain
the existing Torch producer and identical packet decoder. CUDA allocation OOM,
including the pinned NVIDIA driver's exact ordinary RuntimeError spelling,
reaches the existing smaller-band retry; unrelated runtime errors propagate.
Output-kernel failure does not disable splat or fill acceleration. CPU, other
source dtypes, native-unavailable rendering, and Quality do not opt into this
fused output path. The packet size,
single-download contract, pending native-work bound, and per-eye completion
barrier are unchanged.

Separate warmed output-fusion ABBA trials on the same Windows / RTX 4090 setup
measured cached-input-through-terminal-audit medians of 29.22 s versus 28.02 s
for 48 real 1080p frames (4.12 percent less), and 89.14 s versus 88.49 s for
192 frames (0.73 percent less). The longer trial's gain was smaller than its
between-run variation; an earlier 24-frame render-only trial was 8.92 percent
slower. These observations do not establish a general long-clip speedup and
exclude extraction, depth inference, fixture setup, and kernel warmup. Both
full-task trials preserved every RGB/MP4/decoded-video/diagnostic identity,
download size, and all three independent validation passes. A separate trace
counted 834 versus 294 GPU kernel launches for one 1080p frame; trace durations
are not used as performance evidence. CUDA allocated peaks fell from
206,225,920 to 189,391,360 bytes at 1080p, and from 472,781,312 to 441,078,272
bytes in the same nearest-upscaled 4K single-frame check described above.

With that fused GPU output unchanged on both sides, removing unused relative
Fast host geometry reduced warmed ABBA cached-input-through-terminal-audit
medians from 28.51 s to 26.59 s for 48 real 1080p frames (6.74 percent less),
and from 88.13 s to 84.61 s for 192 frames (4.00 percent less). Compact rendering
fell from 9.69 s to 8.51 s and from 46.48 s to 42.43 s, respectively. Every
streamed RGB, encoded MP4, decoded-video, and diagnostic identity matched;
download counts and byte totals, all three independent validation passes, and
authenticated cleanup were unchanged. These separate within-round comparisons
exclude extraction, depth inference, fixture setup, and warmup; they are not
a guaranteed whole-video speedup. Earlier exploratory timings preceded the
restored scalar-control revalidation and are not final-candidate evidence.

At compact-core entry, the retained host geometry and offset payloads fall from
25 to 13 bytes per source pixel, deduplicated by backing storage: 51,840,000 to
26,956,800 bytes at 1080p and 207,360,000 to 107,827,200 bytes in the same
nearest-upscaled 4K probe. This counts neither source RGB nor native depth,
freed preparation scratch, allocator caches, or unrelated process memory; it
is not a process RSS or peak-memory measurement. CUDA allocated peaks remain
189,391,360 bytes in both complete-task trials and 441,078,272 bytes in the
4K single-frame probe. A 240-case differential check preserved all outputs and
existing error outcomes; the original offset rounding and failure paths remain
covered separately by preparation regression tests.

With host preparation and all other kernels shared, the tiled value-only scan
change reduced warmed ABBA cached-input-through-terminal-audit medians from
27.19 s to 26.61 s for 48 real 1080p frames (2.14 percent less), and from
83.80 s to 81.00 s for 192 frames (3.34 percent less) on the same Windows /
RTX 4090 setup. Compact rendering fell from 8.77 s to 8.31 s and from 41.66 s
to 37.95 s, respectively. Both variants received complete-pipeline warmup;
extraction, depth inference, fixture setup, and warmup remain outside timing.
These are modest sample-specific gains, not a guaranteed whole-video speedup.
All RGB, MP4, decoded-video, and diagnostic identities matched, with all three
independent validations, download sizes, authenticated cleanup, and measured
CUDA allocation peaks unchanged. The separate 4K output/resource check also
matched the original path and its existing budget.

A separate single-frame trace counted 294 versus 186 GPU kernel launches; trace
times are not speed evidence. An isolated scan ABBA measured 34.1 percent less
wall time and 79,814,656 versus 33,692,672 bytes of incremental Torch allocation,
not overall rendering or process-memory reductions. Earlier whole-row scan
trials are superseded: their kernel metadata exposed driver-local spills, and
the first full-task baseline had an anomalous unrelated CPU offset-build time.
Neither their larger apparent speedup nor Torch-only memory counts establish
the final candidate's performance or memory behavior.

Moving complete coverage repair flags into the existing output producer reduced
warmed ABBA cached-input-through-terminal-audit medians from 26.15 s to 25.23 s
for 48 real 1080p frames (3.53 percent less), and from 57.22 s to 52.99 s for
192 frames (7.40 percent less) on the same Windows / RTX 4090 setup. Compact
rendering fell from 8.52 s to 7.09 s and from 31.17 s to 27.26 s, respectively.
These are separate within-round comparisons, not speedups inferred by comparing
absolute timings across rounds. Extraction, depth inference, fixture setup,
and warmup are excluded. The 192-frame repeat used complete 192-frame warmup
jobs for both variants; an earlier trial had large changes in unchanged
consolidation and finalization phases and is not the final speed estimate.

Both trials preserved every RGB, MP4, decoded-video, and diagnostic identity,
all three independent validations, authenticated cleanup, download sizes, and
measured CUDA allocation peaks. The separate nearest-upscaled 4K output and
allocation check also matched the baseline. A one-frame trace still counts
186 GPU kernel launches, and actual output-kernel metadata at masked tails,
1080p/4K bands, and the maximum supported pixel count shows unchanged registers,
shared memory, and allocation increments, with no driver-local spills or global
scratch. A 240-case differential check preserved all outputs and existing errors.
The gains remain sample-specific; no quality settings or image arithmetic change.

Deferring fused-band coverage to the existing native worker was measured in a
later, separate warmed ABBA comparison on the same Windows / RTX 4090 setup.
For 48 real cached 1080p frames, observed completion medians were 24.36 s versus
23.21 s (4.74 percent less), with compact rendering 7.42 s versus 6.80 s
(8.45 percent less). The shorter trial still included some finalization variance.
A 192-frame repeat measured 51.34 s versus 47.26 s (7.94 percent less), with
compact rendering 27.20 s versus 23.07 s (15.18 percent less); unchanged
consolidation and finalization phases remained comparable in that repeat.
Both sizes received complete same-size warmup jobs for both variants.

The first 192-frame series is retained but not used as the final speed estimate:
its apparent total reduction was -0.64 percent while unchanged offset building,
decoding, consolidation and finalization abruptly became much faster midway
through the series. No production code changed for the repeat. All trials kept
identical RGB, MP4, decoded streams and diagnostics, three independent complete
validations, authenticated cleanup, download counts and packet sizes. CUDA
allocation peaks were unchanged within the comparisons: 189,391,360 bytes for
the complete 1080p jobs and 440,660,480 bytes in a separate nearest-upscaled 4K
single-frame probe. The latter also verified zero foreground coverage lookups
in the fused candidate and the original one-worker, two-pending-item bound.
Extraction, depth inference, fixture setup and warmup remain outside the task
timings; these measurements do not promise the same whole-video speedup.

Resident relative CUDA offsets were measured in a separate warmed ABBA trial
on the same Windows / RTX 4090 setup. For 192 real cached 1080p frames, completion
medians fell from 51.04 s to 47.58 s (6.77 percent less); compact rendering fell
from 25.11 s to 21.63 s (13.88 percent less). The 48-frame trial is also retained:
rendering improved from 6.27 s to 5.70 s (9.09 percent less), but completion rose
from 20.40 s to 20.98 s (2.83 percent more) with variation in unchanged finishing
phases. Neither that negative result nor the fixed completion work is excluded
from the report. Both sizes received complete same-size warmup for both variants.
All RGB, MP4, decoded-stream and diagnostic identities matched, with all three
independent complete validations and authenticated cleanup retained. Extraction,
depth inference, fixture setup and warmup remain outside these timings.

A separate resource check removed exactly two offset uploads per frame:
16,588,800 bytes at 1080p and 66,355,200 bytes for nearest-upscaled 4K RGB with
unchanged native depth. Both host offset outputs also disappear on the supported
path. One CUDA kernel launch is added per frame; download counts and byte totals
are unchanged. CUDA allocation peaks matched the baseline at 189,391,360 and
440,660,480 bytes, respectively. The new kernel uses no driver-local spills,
shared memory or global scratch in the measured tail/1080p/4K configurations.
Its two logical outputs total eight bytes per pixel; actual allocator block
rounding is recorded separately, not assumed to be only 512-byte alignment.
Thirty-two different nonzero control pairs reuse one compiled kernel, and 256
numerical differential cases matched 18,223,872 int32 outputs exactly. These are
sample-specific rendering/resource observations, not whole-video guarantees.

Quality relative CUDA rendering reuses that unfused float64 offset kernel with
a single-eye specialization. It allocates one int32 plane, consumes it directly
in the existing splat, and releases it before building the other eye. The
already-preloaded near score is reused; no offset readback, host offset map, or
offset upload is needed on this path. A conservative host range check preserves
the int32 proof for resized scores in `[-1, 2]`; unsupported controls, layouts,
sizes or ranges retain the checked native builder. CPU and metric rendering are
unchanged. Missing Triton, a failed preload, or a compilation failure also uses
the host path. Single-eye compilation failures have their own per-device disable
state and cannot disable the paired Fast helper. Offset-allocation OOM recovery
unwinds the failed call before releasing memory and retrying the host builder.

A separate local warmed ABBA trial of this Quality change measured 59.50 s
versus 57.88 s (2.72 percent less) for 48 real 1080p frames from fresh extraction
through depth inference, encoding, publication, cleanup, terminal audit and
model release. Model loading was measured separately and is excluded; including
those warmed loads gives 66.32 s versus 65.07 s (1.89 percent less). Both
adjacent measured pairs were faster; only two runs per variant were measured.
Canonical depth PNGs, every streamed RGB frame, MP4 bytes, decoded audio/video,
summary diagnostics and per-frame diagnostics matched exactly. The model
artifact, settings and frozen input were unchanged, all three independent
validations remained enabled, and measured CUDA allocation peaks matched at
7,268,528,128 bytes. The 48-frame and 360-frame cached-input trials observed
1.15 and 3.15 percent less completion time, respectively, but had appreciable
unchanged-stage variance and are not stable speed estimates. These shared-machine
observations do not establish cold-start or general whole-video gains. The
isolated harness, input identities and all results are retained under
`build/quality-device-offsets-20260906/`.

Stereo decode and output work is bounded by both `stereo_io_workers` and a host
byte budget. End-to-end throughput measurements include decode, render, image
encoding, and writes; renderer median, p95, mean, host geometry, and offset-transfer
costs are reported explicitly.

Quality geodesic propagation uses a four-way indexed heap with contiguous cached
distance, rank, and owner keys. Initial candidates are compacted before bottom-up
heap construction; hole-based sifts move each displaced entry once and refresh
the complete key on every decrease. Ordering remains distance ascending, rank
descending, owner ascending, then pixel ascending. No movement costs, precision,
seeds, or region-selection rules change. The existing `heap_pixel_bytes` scratch
field charges both vector capacities: 4 bytes for the pixel plus 16 bytes for its
cached priority per band pixel. The region-phase preflight grows from `52P+18N`
to `68P+18N`, before runtime and thread allowances; the existing matte phase
already has that bound, so the maximum admission charge does not increase.

On Windows, CPython 3.11.11, and an RTX 4090, a warmed ABBA comparison over 48
uniformly spaced frames from the 473-frame 1920x1080 Quality sample measured
35.66 s versus 34.11 s for the complete stereo-stage call, including preflight,
decode, render, retained PNG writes, and diagnostics completion. The overlapping
pipeline portion measured 27.90 s versus 26.70 s. All 336 RGB, mask, and per-frame
statistics file hashes matched. Both variants retained three lifecycle slots,
three decoders, three writers, and four PNG encoders, with the same 520,946,176
byte admission charge. The cached queue does increase live regional scratch.
Whole-stage 20 ms memory sampling observed peak private-commit deltas below
440 MiB from each warmed baseline, but sampling is not a hard peak-memory proof.
These timings describe this stereo-stage sample, not depth estimation or final video
encoding. The isolated eight-frame geodesic probe improved by 23.5 percent;
the observed complete stereo-stage reduction was only 4.3 percent.

Geodesic relaxation now skips RGB loads and edge-cost calculation when the
neighbor's known distance is strictly below `source_distance + base_cost`.
Equality still reaches the original owner/rank tie-breaker. Pruning is disabled
whenever adding the maximum possible edge cost could overflow uint64, preserving
the original overflow checks even for already-settled back edges. No additional
buffers or admission charges are introduced. An eight-real-frame warmed probe
reduced the sum of per-frame solve medians from 0.944 s to 0.866 s (8.2 percent),
with all arrays and queue statistics identical. A separate 48-frame transient
stereo-stage ABBA comparison preserved all 96 streamed RGB hashes and aggregate
diagnostics, but its overlapping pipeline median moved only from 23.46 s to
23.36 s, within run-to-run variation. This is a measured solver elapsed-time
reduction, not evidence of a substantial whole-stage or whole-video speedup. The large
overflow regression requires `DEPTH_SURGE_RUN_LARGE_NATIVE_TESTS=1` because its
raw-solver fixture needs about 704 MiB outside the public render budget.

The file-backed Quality writer encodes its two RGB rasters and, when retained,
four diagnostic masks through one shared bounded PNG executor. Every enabled
payload finishes encoding before the frame transaction creates sibling
temporaries; publication order, manifest-last durability, OpenCV compression
settings, and output bytes therefore remain unchanged. Capacity selection caps
the executor at four workers and charges 4 MiB per worker against the 512 MiB
host budget: at most 2 MiB of attested stack reserve plus thread/TLS runtime
slack. The existing pipeline workers retain their separate 2 MiB charge, and
all simultaneous encoded payloads are already charged in the Quality writer
slot. The existing 16 MiB writer runtime allowance is pressure-tested with
high-entropy RGB and retained mask outputs; final PNG sizes alone do not bound
native codec allocation peaks. Revalidate this allowance when changing the
codec runtime or worker cap. The cap follows whole-pipeline timings rather than
the encoding-only optimum, so encoding does not crowd out decode and rendering.
Fast and transient output do not create this executor.

Per-frame Fast and Quality diagnostics count their seven repair-state bits in
one native uint8 raster pass with seven uint64 counters, rather than allocating
one temporary bit plane per state. Fast's local-filled pixel count reuses that
same result. No additional native workers or image-sized scratch are created.
Missing or older native extensions, noncontiguous arrays, ndarray subclasses,
and equivalent non-singleton dtypes retain the NumPy counting path.
Dense uint64 histograms serialize through bounded 4,096-bin nonzero scans instead
of Python visits to every zero bin. Both canonical output entry points emit the
same ordered integer pairs, including counts above the float64 exact-integer
range; temporary indexes occupy at most 32 KiB on a 64-bit host. Array subclasses
keep scalar access semantics, and replay reads current bins rather than cached
pairs. No diagnostic fields, checks, or retained payloads are removed.

A separate diagnostics-only ABBA comparison on the same Windows / RTX 4090
setup used the current renderer and rebuilt native binary on both sides, with
only the three previous Python statistics/serialization functions restored for
the baseline. For 192 distinct cached real 1920x1080 source/depth frames and
3840x1080 RGB24 NVENC output, median cached-input-through-cleanup time fell from
97.65 s to 88.89 s (9.0 percent); the stereo stage fell from 76.21 s to 67.37 s
(11.6 percent). A 48-frame ABBA measured 20.73 s versus 19.41 s (6.4 percent).
All streamed pixels, MP4 bytes, decoded frames, and aggregate diagnostics
matched. All three independent container validations and authenticated cleanup
remained enabled; CUDA download calls, bytes, packet sizes, and the measured
206,225,920-byte allocation peak were unchanged. These within-trial timings do
not include source extraction or depth inference and must not be compared
directly with previous rounds' absolute times as an isolated speedup.

The shared diagnostics change also preserved the 48-frame Quality output in
an ABBA run: median completion time was 44.84 s versus 43.51 s (3.0 percent).
The existing exact repeated-input cache remained enabled on both sides, with
47 new renders supplying all 48 output frames. Quality retained two lifecycle
slots, all three validations, identical CUDA transfers, and the same measured
74,692,608-byte allocation peak. This does not change Quality's geometry or
repair algorithms.

Diagnostics consolidation retains every payload hash, schema check, and uint64
overflow check. Sparse histogram pairs still replay from one bounded frame
buffer instead of becoming Python lists. Their integer tokens use a compiled
ASCII scanner and a fixed uint64 limit; exact Python integers avoid generic
integral-protocol dispatch while NumPy scalars and integer subclasses retain
their normalization path. JSON-native metadata and histogram-stripped frame
statistics use the standard JSON encoder for exact canonical-byte validation,
without traversing the histogram-aware serializer. The small-object decoder
remains separate from the general entry point, and duplicate keys,
noncanonical spellings, and nonfinite values are still rejected.

For one 473-frame 1920x1080 retained-mask Quality sample on Windows with
CPython 3.11.11, a warmed baseline/candidate/candidate/baseline comparison of
`complete_streaming_diagnostics_stage` measured median wall times of 23.81 s
before and 18.09 s after this change. This includes payload auditing and
aggregate publication; summary construction alone went from 4.37 s to 2.02 s.
Metadata, JSONL, and summary SHA-256 identities matched in every run. These are
diagnostics completion timings, not whole-video or whole-stereo-stage speedups.

Diagnostic frame-name serialization replays declared records from the bounded
input manifest, validating its canonical rows, order, count, byte length and
SHA-256 without reopening every source PNG or NPZ just to obtain a name.
Explicit provider audits still hash actual payload contents. The final audit
runs after completion metadata is staged, synced and replay-checked, immediately
before atomic publication; rejection cleans only the owned temporary and does
not publish a complete root. A no-work reuse also audits source contents after
aggregate revalidation, before returning success. Provider I/O and format
failures retain typed input-change handling across read-only retries; MemoryError
and stereo-control budget failures are not reclassified. No frame-name list is retained.

Relative Fast and Quality jobs use a transient final-output path only when direct VR
encoding is enabled, intermediates are not retained, upscaling is disabled,
schema-5 publication is active without a recovery reservation, and distortion
plus crop can be reproduced exactly in memory. Rendered pairs are released in
source order through the existing bounded pipeline, transformed and assembled
with the same pixel operations as the file-backed stages, converted from BGR8
to immutable RGB24 bytes, and written to FFmpeg through `rawvideo`. The explicit
pixel format, dimensions and frame rate precede the single image input. This
removes the PNG compression/decompression round trip without changing final
codec, preset, CRF/CQ, chroma sampling, frame rate, or audio selection. This path
never writes
or rereads `04_left_frames`, `04_right_frames`, or cropped RGB PNGs. Unsupported
settings, retained intermediates, and metric geometry continue through the
file-backed pipeline. Recovery first completes a fully authenticated streamed
sibling using its persisted input format rather than the new-session default;
all other recoverable states use the file-backed pipeline.

Raw streams use `encoding-input-sequence-v4` / schema 3. Each closed `raw_frame`
identity records RGB24 dimensions, exactly `3 * width * height` bytes, and the
SHA-256 of the same immutable payload handed to the bounded pipe. Nominal `.rgb`
paths identify ordered frames; they are not files and do not broaden the PNG-only
file-backed provider. Invocation and completed-output evidence are mandatory.
The bounded reader cross-checks raw identities, dimensions, input option scope,
and transport. Existing PNG v2/v3 manifests and v3 interrupted publications
remain readable, but cannot be relabeled as raw evidence or vice versa.

Transient stereo diagnostics use one ordered JSONL extent and publish the same
aggregate summary and completion metadata without per-frame diagnostics files.
FFmpeg stdin, progress, and diagnostics are drained with fixed-size buffers;
pipe backpressure propagates to the stereo lifecycle permits, so memory remains
bounded by configured pipeline capacity rather than frame count. Both Fast and
Quality capacity selection charge the declared distortion-map, transform, VR
assembly, and pipe-copy peak against the 512 MiB host budget. The new producer
pipe peak is `9 * final_pixels`: assembled BGR, converted RGB, and immutable
bytes, plus any live distortion outputs/maps. Conversion temporaries for
noncontiguous views end before the immutable copy is made; they do not add a
fourth simultaneous frame buffer. Codec-internal memory is not part of this
producer buffer bound. If no lifecycle slot remains, the streamed reservation
is discarded and the job resumes through
the file-backed path.

A Windows / RTX 4090 ABBA run over 48 cached 1920x1080 source/depth frames, with
3840x1080 side-by-side NVENC output and no distortion, measured Quality at
58.58 s with PNG versus 55.69 s with RGB24 (4.9 percent less elapsed time).
This includes stereo rendering, encoding, validation, publication, authorized
cleanup, and the terminal audit, but excludes extraction and depth estimation.
The overlapping Quality pipeline alone measured 29.68 s versus 26.58 s. Fast
measured 43.86 s versus 43.75 s, effectively unchanged: its PNG writer was
already overlapped by rendering. Its frame transport work nevertheless fell
from 8.85 s to 1.85 s. Do not describe this local reduction as a whole-job gain.
Both modes preserved all assembled-frame hashes, complete MP4 bytes, decoded
streams, and aggregate diagnostics. Fast retained eight lifecycle slots and
Quality two; no worker survived the joins. The consumer frame-buffer bound fell
from 47,716,352 to 37,324,800 bytes without increasing the shared budget.

Final-container validation retains both FFprobe's decoded frame counts and the
independent complete FFmpeg decode, including strict stream metadata, audio,
diagnostic, timeout, cancellation and terminal-progress checks. The FFprobe
video decoder uses `min(8, cpu_count or 1, expected_frame_count)` threads rather
than its [single-thread default](https://github.com/FFmpeg/FFmpeg/blob/n8.0/libavcodec/options_table.h).
Audio decoder options and all encoding options are unchanged. The explicit
ceiling avoids unconstrained decoder auto-threading;
it is a thread limit, not a total child-process memory bound. Publication and
recovery use the same validator without caching or skipping any validation pass.

A subsequent Windows / i9-13900K / RTX 4090 ABBA comparison, with RGB24 enabled
on both sides and the same 48-frame NVENC fixture, measured Fast at 42.68 s
before versus 38.69 s after bounded probe parallelism (9.35 percent less), and
Quality at 56.56 s versus 53.66 s (5.12 percent less). Each job retained all
three final-container validation calls; their combined FFprobe count time fell
from about 4.90 s to 1.24 s. Complete MP4 bytes, decoded frames, assembled-frame
hashes and diagnostics matched. These are cached-input stereo-through-cleanup
timings, not extraction/depth-inference or whole-video timings. Separate
child-process observations on the 3840x1080 HEVC fixture measured approximately
110 MB working set with one video decoder thread, 230 MB with eight, and
369 MB with sixteen/auto; sixteen/auto did not improve probe time meaningfully.
Those observations do not establish a universal decoder memory bound.

The input-manifest reader scans JSON string/object delimiters within its existing
64 KiB input blocks, carrying quote, escape and object-depth state across short
reads. It hashes consumed spans rather than individual bytes; lookahead and
reservation padding never enter the raw or self fingerprint. The 1 MiB token
limit and strict canonical-value decoder remain unchanged, as do the closed
PNG/raw schemas, no-follow acquisition and every publication/recovery check.
This is a bounded lexical optimization, not a manifest cache or permission to
reuse evidence across file mutations.

With bounded probe threading enabled on both sides, an isolated reader-only
ABBA on the same Windows 48-frame fixture measured Fast at 39.76 s before
versus 35.34 s after batching (11.14 percent less), and Quality at 51.79 s
versus 48.83 s (5.71 percent less). Finalization alone fell from 13.86 s to
10.64 s for Fast and from 14.90 s to 12.19 s for Quality. Every run retained
23 input-manifest projections during publication, another 138 during
finalization, and all three container validations. Assembled pixels, complete
MP4 bytes, decoded streams and diagnostics matched. These are cached-input
stereo-through-cleanup measurements, not extraction/depth-inference timings.
The same 138 parses measured in isolation fell from 4.66 s to 1.43 s; this
69 percent local reduction must not be reported as whole-video acceleration.

The shared canonical JSON codec avoids a second Unicode-scalar traversal when
re-encoding a freshly parsed tree. Exact ASCII `str` values also bypass
per-character checks. After exact `bytes` pass ASCII decoding, the standard JSON
decoder cannot produce non-ASCII strings without a `\u` escape, so inputs without
that escape need no scalar traversal. Unicode escapes and byte/string subclasses
retain the applicable validation path. This proof depends on the fixed standard
decoder and internal hooks; custom decoder hooks would require re-evaluation.
Duplicate-key rejection, canonical bytes, byte limits, and error behavior remain
unchanged. Public encoding still validates on every call; there is no object or
file-evidence cache across calls or recovery boundaries.

An isolated codec-only ABBA comparison with the batched reader and bounded probe
threads enabled on both sides measured Fast at 37.26 s before versus 34.51 s
after (7.37 percent less), and Quality at 48.20 s versus 45.16 s (6.30 percent
less). This uses the same Windows 48-frame cached-input stereo-through-cleanup
fixture, not extraction or depth inference. All assembled pixels, complete MP4
bytes, decoded streams and diagnostics matched. Every job retained the same
23 publication and 138 finalization manifest projections, three container
validations, and authenticated cleanup. A 15,000-case differential probe also
preserved encoded bytes, decoded values, and error types/messages.

Preallocated reservation zero checks use native byte counting for exact `bytes`.
Frame classification scans the suffix at its existing offset without copying
the padding; an empty suffix or any nonzero byte still prevents a complete-full
classification. Byte subclasses retain their previous iteration and slicing
semantics. Terminal-source parsing keeps its NUL boundary and partial-frame
checks, and no file acquisition, identity check or recovery phase is skipped.

A padding-only ABBA change on the same Windows 48-frame cached-input fixture
measured Fast at 31.98 s before versus 30.85 s after (3.53 percent less), and
Quality at 41.00 s versus 38.95 s (5.01 percent less). Fast publication/cleanup
padding-check time alone fell from 1.16 s to 0.018 s, with every check retained.
Pixels, MP4 bytes, decoded streams and diagnostics matched; all three container
validations and 23/138 publication/finalization manifest projections remained.
Extraction and depth inference are excluded. A separate 220-check sample of
528,384-byte extents measured 8.35 s versus 0.118 s; one complete-frame check's
traced temporary allocation fell from 525,358 to 254 bytes. Those local numbers
are not whole-job speedups or bounds on total pipeline memory. A 20,000-case
differential probe preserved frame states, terminal settings bytes and errors.

Windows handle-identity queries lazily reuse their ctypes structure types and
the bound `GetFileInformationByHandleEx` signature. The single-entry cache holds
only ABI declarations, never handles, file identities, attributes, or result
buffers. Each call allocates three fresh structures and performs all three
native queries, retaining current object-kind, reparse-point, and link-count
checks. Binding failures remain retryable; thread-local last-error handling and
the pre-query platform guard are unchanged. This shared I/O optimization leaves
the POSIX path, image arithmetic, and publication/cleanup validation rules alone.

With only this binding setup varied, separate warmed Fast ABBA trials on the
same Windows / RTX 4090 setup reduced cached-input-through-terminal-audit
medians from 24.63 s to 22.40 s for 48 real 1080p frames (9.05 percent less),
and from 73.94 s to 69.71 s for 192 frames (5.72 percent less). Both variants
received complete same-size pipeline warmup. All RGB, MP4, decoded-stream,
and diagnostic identities matched; all three independent container validations,
download sizes, authenticated cleanup, and CUDA allocation peaks were unchanged.
These are within-round sample observations, not a general whole-video speedup;
extraction, depth inference, fixture setup, and warmup are outside the timing.

Windows relative reservation-file and acquired-root opens also lazily reuse
their own ctypes ABI declarations. Two independent single-entry caches retain
the original structure layouts and native signatures, including each path's
existing status-block field types. Every acquisition still creates fresh status,
name and object-attribute buffers and performs the same native opens, identity
queries, permission checks and closes. No paths, handles, results or publication
evidence are cached. Initialization failures remain retryable; simultaneous cold
calls may construct equivalent independent bindings rather than sharing mutable
call buffers. The POSIX branches are unchanged.

A frozen-code comparison varying only these two acquisition setups used two
complete same-size warmups followed by eight Fast ABBA/BAAB jobs. For 48 real
1080p cached frames, total median time fell from 21.427 s to 21.096 s
(1.55 percent less); finalization fell from 7.502 s to 6.937 s (7.53 percent
less). Pixels, complete MP4 bytes, decoded streams, diagnostics, download sizes,
all three container validations and authenticated cleanup matched. Separate
profiled runs retained exactly the same identity-check, acquisition and
close-wrapper counts; native call statements were unchanged. A four-job
192-frame comparison is retained but is not a representative
speed estimate because its unmodified phases varied substantially. The paired
Quality check establishes compatibility, not a Quality speedup. These fixtures
do not retain audio. Timings exclude extraction, depth inference, fixture setup
and warmup.

Untimed 250-acquisition probes on each path found 750 newly retained ctypes
pointer types before this change and none after warmup with the caches. Traced
Python memory retained after collection fell from approximately 4.4 MB to at
most 1,136 bytes, and native handle counts were unchanged. These are local
allocation observations, not whole-task memory bounds. Isolated acquisition
timings improved by about 45 percent; that number is not the total-job gain.

Encoding-manifest name and identity arrays now group strict JSON decoding and
hash updates over values already present in the reader's 64-KiB buffer. Small
complete arrays take one batch; larger arrays use complete buffered prefixes,
with at most one prefix attempt per loaded buffer. Every scalar, closed schema,
canonical spelling, path, frame sequence, shape and cardinality check remains.
Unsuccessful speculation, including deeply nested JSON, returns to the original
token path without consuming bytes or updating hashes. This preserves the first
error and its original consumption boundary. No manifest, file identity or
publication evidence is cached, and no additional reads are issued by batching.

A frozen-parser comparison on the same Windows / RTX 4090 setup used complete
same-size warmups before the measured Fast jobs. Eight 48-frame ABBA/BAAB jobs
reduced the total median from 20.423 s to 20.051 s (1.82 percent less). Four
360-frame ABBA jobs, representing 15 seconds at 24 fps, measured 113.452 s versus
105.230 s (7.25 percent less); finalization fell from 22.114 s to 17.223 s.
The unchanged renderer also varied in the longer trial, so the entire observed
total reduction cannot be attributed to this parser change. These are
within-round cached-input-to-terminal-audit observations, not guaranteed
whole-video gains. Extraction, depth inference, fixture setup and warmup are
excluded, and audio retention is disabled. Pixels, complete MP4 bytes, decoded
streams, diagnostics, download sizes, CUDA allocation peaks, all three container
validations and authenticated cleanup matched. The separate paired Quality run
establishes compatibility only, not a Quality speedup.

Untimed attribution retained all original manifest-parse, acquisition,
native-identity-query, recovery, validation and close-wrapper counts. A
90,000-frame synthetic manifest required 639 strict JSON calls instead of
180,002, with the same 318 reads and a maximum 64-KiB read request. This trades
a bounded batch of temporary Python objects for fewer codec and hash-update
calls: the longest traced parser probe used about 0.8 MiB more peak Python
allocation. Neither the byte-buffer limit nor these local allocation
observations are whole-process memory bounds. Differential checks also preserve
error types, messages, loaded bytes, read requests and both digest boundaries.

The legacy in-memory depth API rejects projected outputs over 512 MiB. The main
video pipeline remains file-backed.

Framewise raw-depth generation prefetches at most one subsequent input batch
on a single decoder thread. PNG headers bound the additional decoded BGR arrays
and their contiguous stack to 256 MiB; unknown headers or oversized batches use
the serial path. This is a source-buffer budget, not a process-RSS or decoder
scratch-space guarantee. A background `MemoryError` retries on the caller
and disables further prefetch for that raw stage. Decoded per-frame arrays are
released before inference rather than being retained alongside the stack.

Inference, native-output validation, dtype promotion, raw writes, and metadata
commits remain ordered on the caller. Complete resume chunks are not decoded
and still reset consecutive-batch reuse. The loader is joined on every exit,
including interruption; a later decode failure cannot hide an earlier inference
or write failure. No raw-to-Stage-3 barrier, batch grouping, precision, model
setting, or VDA temporal-window rule changes. VDA continues to use its existing
sequence path without this framewise prefetch.

Scene sampling groups integer scene IDs once instead of scanning the entire
clip for each scene. First-seen scene order, source-frame order, the exact
linspace/rint/unique sample-32 rule, and persisted frame-index arrays are
unchanged. Metric convergence uses a linear selection mask to restore global
source order, while noninteger inputs retain the previous NumPy matching path.
Depth conversion, spatial sampling, percentile arithmetic and artifact versions
are unchanged. The grouping trades O(N) temporary indexes for O(N) traversal
instead of O(N * scene_count) repeated scans.

Runtime timing uses a monotonic clock. The completion banner is sampled after
successful terminal media audit, rather than before finalization. The
orchestrator's in-memory `process_elapsed_seconds` measures its current
`process()` invocation through model release and owned-lock teardown, on success
or failure; callers must interpret it alongside the outcome. It excludes
upstream model loading, caller preparation, and earlier resume attempts. The
banner precedes teardown and therefore has a narrower boundary. Persisted
timing contracts are unchanged: schema 5 freezes duration from `terminal_at`,
and legacy metadata retains its pre-finalization snapshot. Neither persisted
field is a substitute for the new invocation measurement.

A local 48-frame, 1920x1080 See-Through comparison held the existing estimator,
model artifact, batch grouping, resolution, precision and retention settings
fixed. After complete warmups of both paths, three alternating measured pairs
reduced the `generate_depth_map_files` median from 22.815 s to 19.691 s
(13.69 percent less); raw-stage medians were 21.378 s and 18.380 s. All 101
generated artifacts matched by SHA-256 in every warmup and measured run,
including raw NPZs, canonical PNGs, scene samples and metadata. This scope
excludes source extraction, model loading, stereo rendering, encoding and final
audit. Unrelated GPU activity was present on the shared workstation; these are
local warmed observations, not a whole-video performance guarantee. The
isolated harness and JSON results are retained under
`build/raw-prefetch-profile-20260905/`.

A separate five-round CPU-only index microbenchmark used 60,000 integer frame
IDs across 600 scenes of 100 frames. Median selection fell from 2.975 s to
0.083 s for contiguous scenes and from 2.634 s to 0.059 s for interleaved IDs;
ordered index lists matched exactly on every execution. These numbers cover
only scene-index selection, not depth loading, sampling arithmetic or GPU work,
and do not measure peak memory.

Raw NPZ writes use standard lossless DEFLATE level 1 with the same NPY member
bytes, ZIP64 support and atomic replacement/temporary-file cleanup contract.
Existing higher-compression files remain readable and can coexist with new
files during resume. The `npz_deflate` compression family and artifact schema
are unchanged; compressed container bytes may differ, but stored precision,
camera values and metadata identities do not. Lower compression trades disk
space for less CPU work; disk preflight continues to use uncompressed bounds.

Depth normalization reuses private score and quantization buffers without
mutating caller arrays. Gathered subtraction/division, clipping, percentile
sampling and rounding order remain unchanged. Scalar quantization retains its
original NumPy expression to preserve scalar promotion behavior. Differential
tests compare float bytes and encoded uint16 values, including special values,
strided/read-only inputs and quantization boundaries. For 1,048,576 float32
elements, traced peak allocations fell from 17,826,792 to 13,632,392 bytes for
canonicalization and from 8,389,044 to 6,291,768 bytes for encoding. These are
local operation allocations, not whole-process memory bounds.

A second local 48-frame See-Through comparison retained the previous prefetch
and scene-index optimizations in both arms. Complete warmups followed by three
alternating measured pairs reduced the depth-generation median from 19.410 s
to 18.717 s (3.57 percent less). Raw NPZ storage increased from 2,972,303 to
3,985,256 bytes (34.08 percent); all stage artifacts together increased from
6,932,204 to 7,945,157 bytes (14.61 percent), excluding source frames. Every
raw NPY member matched byte-for-byte and every other artifact, including all
48 canonical PNGs and scene/metadata files, matched by SHA-256 in all runs.
This measures `generate_depth_map_files`, not extraction, model loading, stereo
rendering, encoding or final audit. Shared GPU activity and observed per-run
variation limit extrapolation; this is not a whole-video speed guarantee or
an additive claim with the earlier benchmark. The harness and JSON results
are retained under `build/raw-lossless-profile-20260905/`.

Metric-camera disk preflight covers both missing raw payloads and the complete
metric store. The native output estimate is checked before inference and the
preflight is repeated after the first stored batch if its measured shape or
selected dtype differs. The metric-stage bound does not assume compression:

```text
A              = max(4096, filesystem allocation unit)
payload_bound  = sum(5 * height * width)
metadata_bound = max(16 MiB, A * frame_count)
atomic_overlap = ceil(1.25 * largest_frame_payload) + A
required_bytes = ceil(1.25 * payload_bound) + metadata_bound + atomic_overlap
```

The five bytes per pixel are float32 inverse depth plus Boolean validity. A
64-KiB allocation unit is used when the filesystem value cannot be queried.
Preflight cannot reserve capacity against other writers. Post-preflight
`ENOSPC` removes only the current temporary write, keeps prior committed frames
and writing metadata resumable, withholds completion metadata, and reports the
preflight bound, current free bytes, and failing path.

### Canonical Metadata Reads

Quality metadata is still revalidated at each original transaction boundary.
`_ExactCanonicalReader` satisfies already-buffered reads without intermediate
chunk assembly and consumes ordinary string runs directly from its existing
64 KiB buffer. Quotes, escapes, nesting/scalar limits, strict ASCII canonical
decoding, and raw/self fingerprints are unchanged. Runs stop at the first
possible byte/string budget violation; refill and truncated-read handling
retain the original path. No earlier validation verdict is cached or trusted.

A September 8 follow-up for single-video workloads compared six complete
360-frame jobs, including warmups. Raw/canonical/frame pixels, decoded audio
and video, MP4 bytes, and both diagnostic files matched exactly. Measured
production finalization medians decreased from 25.571 to 21.554 seconds, with
both adjacent pairs improving, while complete-process medians were 323.981
versus 332.539 seconds and unchanged inference/rendering timings varied. This
is a verified reduction in metadata/finalization overhead, not an established
whole-generation speedup. The synthetic 10,000-record parser fixture measures
metadata scaling only, not a generated 10,000-frame video. Evidence and timing
boundaries are recorded in `build/long-video-profile-20260908/results.md`.

## Resume And Migration

Resume decisions are deterministic and reported before mutation. The selected
settings file is fixed for the whole operation. Web resume resolves only the
source path and SHA-256 recorded by that file. Source frames require the saved
source-video SHA-256, their extraction-time content fingerprint, a contiguous
frame manifest, matching dimensions, and readable payloads. Raw reuse
additionally requires the exact fingerprint of the loaded estimator and
validates every persisted NPZ. Canonical and stereo PNGs are decoded and checked
for their declared dtype and shape before reuse.

Settings schema v4 combines the MoGe geometry settings with
`temporal_postprocessor`. V1-v3 jobs migrate upward with a missing temporal mode
set to `off`; v4 is parsed strictly, and future schemas fail before any job
mutation. Resume distinguishes an omitted override from an explicit `off`.

Every writer takes a fail-fast OS lock before the authoritative audit and holds
it through migration, model execution, downstream writes, and cleanup. A
complete stabilized artifact is validated from stored identities, shot
manifests, and per-file SHA-256 values before any CUDA or model probe. It can
therefore render on a machine without CUDA, the VDPP checkpoint, or the
historical base estimator. Incomplete VDPP work resumes at shot granularity;
an interrupted long shot is recomputed from its beginning.

A depth or stereo schema change does not by itself invalidate valid
`00_original_frames`. Raw model changes invalidate raw depth and every
downstream stage. Canonical changes invalidate canonical disparity and
downstream stages. Render-setting changes invalidate stereo output and
downstream stages.

Relative and metric Stage-3 fingerprints are independent. A mode switch
selects the matching stage, preserves a compatible inactive stage, and
invalidates stereo and downstream output without interpreting one geometry
format as the other. Metric stereo fingerprints also include crop width, SAR,
baseline, requested/effective convergence, and disparity cap.

For `keep_intermediates=false`, raw frame NPZ files are removed only at the
validated selected-Stage-3 barrier. A later render or encode failure keeps that
completed Stage 3 for resume. Full successful-finalization cleanup removes all
intermediate payloads, including payloads from both Stage-3 directories;
cleanup does not run after a failed job.

The 16-sample renderer changes the stereo algorithm identity from v1 to v3.
Resuming v1 metadata preserves source, scene, raw-depth, and canonical stages,
then regenerates stereo and every tracked downstream frame stage. Encoded video
files are outside that frame-stage invalidation and are not implicitly deleted.

Old generated directories are archived under `legacy_v1/` by default.
Non-interactive runs never delete them implicitly; deletion requires
`--migrate-legacy delete`. Removed keys found in an on-disk settings file are
listed and stripped during migration, while the same keys supplied explicitly
are validation errors. Web migration is deferred to the processing thread until
the model fingerprint is available. Failed archive/settings transactions move
stages back to their original paths. Explicit deletion commits after stage
movement and settings migration; staging cleanup is post-commit garbage
collection, so an interrupted cleanup is reported without an impossible partial
rollback.

## Main Components

- `depth_surge_3d.py`: CLI, validation, and resume entry point.
- `app.py`: Flask and Socket.IO Web UI.
- `processing/frames/depth_processor.py`: scene, raw-depth, canonical, and cache
  orchestration.
- `inference/depth/backend_registry.py`: backend variants, capabilities,
  construction, availability, and report identity.
- `processing/frames/metric_geometry.py`: restartable metric Stage 3, disk
  bound, and clip-global convergence.
- `processing/frames/depth_normalizer.py`: representation conversion and pure
  scene canonicalization.
- `processing/frames/stereo_generator.py`: bounded stereo I/O pipeline.
- `processing/video/video_encoder.py`: file-backed and bounded raw RGB24/legacy PNG
  encoding, schema-5 reservation, validation, and final publication.
- `processing/video/streamed_vr_manifest.py`: exact streamed RGB24/legacy PNG input evidence.
- `processing/frames/temporal_stabilizer.py`: VDPP artifact audit, shot resume,
  disk/device preflight, and file-backed generation.
- `inference/depth/vdpp_temporal_postprocessor.py`: exact bounded 32/4 neural
  recurrence and explicit model lifecycle.
- `core/render_disparity.py`: strict base/stabilized producer validation.
- `rendering/forward_splat.py`: packed 16-lane z-buffer for one row band.
- `rendering/stereo_renderer.py`: host geometry, fine-grid fill, downsampling,
  and bounded eye rendering.
- `rendering/stereo_geometry.py`: common relative and metric renderer input.
- `io/resume.py`: stage validation, reports, and legacy migration.
