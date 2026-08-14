# Stereo Edge Coverage Production Revision 5

## Status

Approved by the user for production implementation on 2026-08-14 after review
of the specification and experimental crops. Approval does not authorize merge
until every final release gate in this document passes.

This is a narrow amendment to
`docs/superpowers/specs/2026-08-14-stereo-edge-coverage-design.md`. It
supersedes that document's sample-count, downsampling-tree, memory-bound,
algorithm-identity, and associated verification clauses. Every other Revision
4 contract remains incorporated by reference.

Revision 4 selected a fixed `S=8` horizontal z-buffer. Its clean implementation
at `5eebae98ad1baa75b5a8273dd66d8401c176695b` passed the repository,
independent-oracle, procedural, determinism, resume, and memory gates. The
hash-pinned production sample then failed five of six unchanged p95 gates even
though all six edge-MAE gates passed. Revision 4's own Failure Policy therefore
authorized a higher-sample sensitivity experiment rather than threshold
tuning.

The user approved a bounded `S=16` experiment. The clean prototype at
`f4fe687042277cb1d9dbbcaad548acb188aaf6a9` passed every automated quality,
CUDA-memory, and latency gate. It remains isolated on
`codex/stereo-edge-s16-experiment`, was never merged to `main`, and creates no
production contract. Its only purpose is to supply evidence for this revision.

The six 400-percent experimental comparison crops passed Codex visual inspection
and subsequent user review. The immutable numeric report remains
`numeric_pass_human_review_required` because it was generated before that user
decision; the approval is recorded after the report rather than rewriting its
evidence. New production-commit crops still require final review before merge.

## Linus Gate

### Is this still the same real problem?

Yes. The serration exists in persisted eye PNGs before VR assembly or video
encoding. Revision 4 fixed the visibility data structure, but eight sample
points leave a one-level quantized tail error on the real contour. This is a
sampling-resolution failure inside the selected representation, not evidence
for changing stereo strength, blurring colour, or reconstructing canonical
depth.

### Is there a smaller change?

Yes: keep the Revision 4 algorithm and change its fixed horizontal resolution
from 8 to 16. Independent `S=16` and `S=32` sensitivity renders both crossed the
unchanged production thresholds. The real Torch `S=16` prototype then proved
that the smallest successful count also fits the existing quality, memory, and
performance budgets. No second renderer, adaptive classifier, or public quality
setting is needed.

### What does it cost?

At 4K under the normal 256 MiB temporary budget, the measured implementation
uses 54-row bands: 40 bands per eye and 80 total. It transfers 63.3 MiB of
precomputed int32 offsets per frame and spends a median of about 0.32 seconds on
host geometry in the measured 4K cases. The worst measured renderer median was
`1.9314x` v1, below the approved `2.5x` limit. This is acceptable evidence for
`S=16`; it is not evidence that `S=32` would fit.

## Experimental Evidence

All evidence below was produced from clean subject commits on one RTX 4090 with
driver `620.02`, Python `3.11.11`, Torch `2.13.0+cu130`, and CUDA runtime `13.0`.
The artifacts are outside Git because the production fixture is copyrighted and
the benchmark workspaces are large.

### Repository Gate

The experimental commit passed Black, flake8 including McCabe complexity 10,
mypy, and all 938 unit tests. Coverage was 91.33 percent, above the repository's
85-percent floor.

### Hash-pinned Production Quality

The source, canonical map, v1 eye images, ROIs, edge masks, `S=64` oracle,
aggregation rules, and thresholds are byte-for-byte the Revision 4 gate. Only
the rendered sample count changed; the experimental identity changed for audit
purposes but did not affect pixels.

| ROI | Eye | `S=16` / v1 edge MAE | Candidate p95 levels | v1 p95 levels | Result |
|---|---|---:|---:|---:|---|
| left sleeve/hand | left | 0.135375 | 0 | 1 | pass |
| left sleeve/hand | right | 0.180292 | 0 | 1 | pass |
| right sleeve | left | 0.189330 | 0 | 1 | pass |
| right sleeve | right | 0.130890 | 0 | 2 | pass |
| guitar/dress boundary | left | 0.275045 | 0 | 1 | pass |
| guitar/dress boundary | right | 0.217516 | 0 | 1 | pass |

All six ratios are below `0.70`; every p95 ratio is zero and therefore below
`0.85`; every outside-edge regression and structural check passed. The numeric
report was generated from clean commit `f4fe687` with strength `1.25`,
convergence `0.5`, and `occlusion_fill=background`.

### CUDA Live Memory

The first three rows force the complete requested band into one render pass.
The last row is the selected 54-row band observed during a complete 4K render
under the normal 256 MiB budget.

| Source band | Peak allocated | Peak reserved | Allocated/source pixel | With 25% headroom |
|---|---:|---:|---:|---:|
| `1024x128` | 111,083,520 | 163,577,856 | 847.500 | 1,059.375 |
| `1920x128` | 208,347,136 | 287,309,824 | 847.767 | 1,059.708 |
| `3840x64` | 208,470,016 | 287,309,824 | 848.267 | 1,060.333 |
| `3840x54` in full 4K | 177,278,464 | 226,492,416 | 854.931 | 1,068.664 |

The complete `3840x2160` render produced both full-size host outputs with 54-row
bands, 80 eye-bands total, and zero OOM retries. Full-frame host geometry and
offset-transfer accounting were both 66,355,200 bytes. The geometry value is
the retained pair of int32 maps, not a process-RSS peak. Temporary float64
geometry arrays are released before device rendering; production must preserve
that ownership boundary. No full-frame device image, index, or fine-grid buffer
was introduced.

Set the production planning constant to:

```python
SPLAT_BYTES_PER_PIXEL = 1280
```

This is a measured bound, not the old linear extrapolation. The largest
observed live allocation plus the required 25-percent headroom is
`1,068.664 B/source-pixel`; 1,280 leaves additional allocator and shape margin.
At width 3840 it deterministically selects 54 complete rows under 256 MiB.

### Performance Against v1

The committed experimental benchmark harness ran the exact v1 baseline
`6bba9ee1e3c3a5df91f4a0b81458584661822b34` and the clean `S=16` prototype in
separate fresh processes. Each case used five warmups and 30 synchronized
measurements. Harness hashes, settings, GPU, driver, and runtime matched.

| Resolution | Fixture | v1 median | `S=16` median | Median ratio | p95 ratio |
|---|---|---:|---:|---:|---:|
| 1920x1080 | smooth | 0.262499 s | 0.338028 s | 1.287729 | 1.358538 |
| 1920x1080 | collision | 0.238959 s | 0.379248 s | 1.587081 | 1.601581 |
| 3840x2160 | smooth | 0.953109 s | 1.510354 s | 1.584661 | 1.581578 |
| 3840x2160 | collision | 0.774839 s | 1.496471 s | 1.931330 | 1.838817 |

Every median ratio is below `2.5`; every linear-interpolated p95 ratio is below
`3.0`; both 4K fixtures completed. Pipeline throughput is recorded but is not
used to hide renderer latency.

## Production Decision

On approval, replace Revision 4's fixed `S=8` decision with one fixed
16-sample horizontal z-buffer:

```python
HORIZONTAL_SUBPIXELS = 16
SPLAT_BYTES_PER_PIXEL = 1280
STEREO_STAGE_ALGORITHM_VERSION = "torch-horizontal-16x-zbuffer-v3"
```

These are algorithm constants, not settings. The `v3` identity deliberately
does not reuse Revision 4's `torch-horizontal-8x-zbuffer-v2` identity or the
prototype's `-experiment` identity. The schema remains:

```python
STEREO_STAGE_SCHEMA_VERSION = 1
```

Revision 4 remains authoritative for all non-sample-count behavior:

- source pixels project opaque unit-width horizontal intervals;
- half-open sample occupancy uses the host float64 `ceil` geometry rule;
- per-eye offsets are computed once per full frame, narrowed to int32, and
  sliced into bands without CUDA recomputation;
- visibility is one signed int64 packed-key `amax`, encoding strict nonnegative
  float32 depth and the lowest source-index tie-break;
- winner colour and depth are gathered from the discrete source, with no float
  equality pass, epsilon, atomic float sum, or two-layer special case;
- background fill operates on fine-grid winners before downsampling;
- unresolved lanes remain black for `occlusion_fill=none`;
- public pre-fill valid and post-fill hole masks retain Revision 4 semantics;
- low-level batched splat input remains removed;
- public settings, result arrays, dimensions, dtype, and channel order do not
  change.

The 16 lane colours must be reduced by one fixed balanced tree: adjacent lanes
to eight pairs, adjacent pairs to four quads, adjacent quads to two octets, and
the two octets to one total. Multiply once by exactly `0.0625`, then apply
ties-to-even output conversion. Do not call an unordered reduction and do not
normalize by valid-lane count.

The independent discrete oracle uses exactly 16 sample points. The continuous
small-fixture oracle and the manual `S=64` production oracle do not change.

## Persistence

The v3 algorithm identity invalidates existing stereo eye directories and all
tracked downstream frame stages while preserving source frames, scene data, raw
depth, and canonical disparity. This applies to v1 output and to any local v2
experimental artifact. No settings migration is required.

The resume report still ends at generated VR frames. It does not claim to
delete or atomically invalidate an already encoded video. A resumed production
run must encode from the regenerated downstream frames through the existing
normal path.

## Production Implementation Boundary

After explicit Revision 5 approval, create a new production worktree and
`codex/` branch from the then-current `main`. Do not merge the experimental
branch or relabel `f4fe687` as production. Port the reviewed final diff so the
production history makes the approval boundary visible.

Use TDD again on the production branch:

1. Port tests and independent reference expectations for fixed `S=16` and the
   production v3 identity; demonstrate failure against v1.
2. Port only the approved renderer, algorithm-version, test, verifier,
   benchmark, and documentation changes.
3. Remove the `-experiment` identity; do not add a runtime sample selector.
4. Run the complete repository and release evidence before merge.

The allowed file boundary remains Revision 4's boundary:

- `src/depth_surge_3d/rendering/forward_splat.py`;
- `src/depth_surge_3d/rendering/stereo_renderer.py`;
- `src/depth_surge_3d/processing/frames/stereo_generator.py`;
- stereo, resume, verifier, benchmark, and independent-oracle tests;
- `scripts/benchmark_stereo_renderer.py`;
- `scripts/verify_stereo_edge_fixture.py`;
- `docs/superpowers/specs/2026-08-14-stereo-edge-coverage-revision-5.md`;
- `docs/ARCHITECTURE.md`, `docs/PARAMETERS.md`,
  `docs/TROUBLESHOOTING.md`, and relevant resume documentation.

Do not change depth inference, canonicalization, scene analysis, public
settings, CLI/Web controls, distortion, crop, upscale, VR assembly, or video
encoding. Preserve McCabe complexity at most 10 and unit coverage at least 85
percent.

## Final Release Gates

Experimental evidence selects the design but does not substitute for evidence
from the final production commit. Before merge, rerun all of the following from
a clean committed production candidate:

1. Black, flake8, mypy, and `pytest tests/unit --cov-fail-under=85`.
2. Exact production/discrete-oracle, procedural, CPU/CUDA, banding, OOM-retry,
   resume, and zero-strength tests with fixed `S=16`.
3. The unchanged hash-pinned production fixture, producing a new JSON report
   and all six 400-percent crops tied to the production v3 commit.
4. User visual review of all six crops. Numeric success cannot override a halo
   wider than one output pixel or loss of the named guitar and hand details.
5. The three forced-band CUDA probes and one complete 4K normal-budget probe.
   Require 25-percent live-set headroom within `1280 B/source-pixel`, successful
   full-size outputs, and no second OOM.
6. Fresh-process v1 and production-candidate benchmarks using one committed
   harness, five warmups, 30 measured frames, both fixtures, and both
   resolutions. Require every median ratio at most `2.5` and every p95 ratio at
   most `3.0`.

Any failure stops merge. Do not lower `S`, increase thresholds, increase
`SPLAT_BYTES_PER_PIXEL`, add a quality mode, or retain v1 behind a flag without
a new reviewed revision. If a future real-sample gate fails after exact-oracle
tests pass, compare fixed `S=32` and `S=64` visibility before blaming canonical
depth placement.

## Audit Artifacts

The bounded experiment produced these immutable review inputs:

| Artifact | SHA-256 |
|---|---|
| `reported-sample-s16.json` | `128a6846b13ed0a90e9213042309c59cf06e7614573f3f349e6eb7bf77b2fa85` |
| `memory-s16.json` | `f021fa89824f3f39ce2955cd3b78509905df1dbd25c1a9ecf7164cf3aea710b8` |
| `benchmark-baseline.json` | `d1aed277f6d3fe0f9b589bfdd8b5f8db48d81ca24969d3b9e1c24627c0890e71` |
| `benchmark-s16.json` | `35b81e819f43650eadff7d16995584c553ae85972200db6fc61d810d6c4717a5` |
| `benchmark-comparison-s16.json` | `e6ed423e3b1908c7fdff4312f194e1b952fcb4ec8469fe582f0d4b28f7697ca8` |
| `experiment-decision-s16.json` | `01fec3921422385d065f1655b5d1a623132c86ee87176383b7176bcdf380f07d` |

They live under
`E:\Code\depth-surge-3d-review\stereo-edge-s16-experiment`. The six crop paths
are recorded in `reported-sample-s16.json`. None of these local artifacts belongs
in unit tests or CI.

## Approval Criteria

Approve Revision 5 only if all of these decisions are acceptable:

1. Replace the unshipped Revision 4 `S=8` choice with fixed production `S=16`.
2. Keep the packed visibility, host geometry, fill, mask, and public API design
   unchanged.
3. Adopt measured `SPLAT_BYTES_PER_PIXEL = 1280`, accepting 54-row 4K bands and
   80 eye-band launches under the normal budget.
4. Use production identity `torch-horizontal-16x-zbuffer-v3` and invalidate old
   stereo/downstream frame artifacts through the existing resume contract.
5. Add no public sample-count or quality setting and no v1 compatibility path.
6. Treat `f4fe687` and its reports as design evidence only, never as a production
   commit or release report.
7. Require a new clean production report, memory run, benchmark pair, and user
   crop review before merge; do not weaken any Revision 4 threshold.

Approval authorizes production implementation in a new worktree. It does not
authorize merge until every final release gate passes.
