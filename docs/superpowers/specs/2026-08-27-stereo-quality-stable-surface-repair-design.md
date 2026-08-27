# Stereo Quality Stable-Surface Repair Design

**Status:** implemented; awaiting integration

**Supersedes:** the Quality geometry-edge and background-repair behavior in
`2026-08-19-stereo-quality-edge-reconstruction-canonical.md`. Fast mode,
metric projection, 16-lane visibility, and diagnostic transaction semantics
remain unchanged unless this document says otherwise.

## Goal

Reduce fine-structure tearing and foreground-colour leakage at disocclusions,
while replacing the v9 second splat and global donor planner with a bounded
single-splat repair. Keep exactly the public `fast` and `quality` modes.

## Root Cause

The low-resolution region builder can connect a depth ramp transitively even
though its endpoints differ by many output pixels. Bilinear geometry then
preserves intermediate edge depths. The v9 background path attempts to repair
the resulting holes after the contour has already torn, and pays for a second
splat plus a global donor search.

Even after stable-surface reconstruction, an antialiased outline can straddle
the selected RGB edge. The outer mixed pixel then remains on the background
surface and becomes a detached duplicate when the foreground moves. This is a
geometry matte defect, not an encoder defect.

Separately, the previous NVENC command selected H.265, `p7`, and `hq` without a
rate-control or quality target. On the reported fixture it encoded at about
2.16 Mbit/s and exposed large colour bands and blocks that were not present in
the rendered PNGs.

## Stable-Surface Geometry

For a native geometry raster with `N` samples, calculate:

```text
area_threshold = 1                                  when N < 1024
area_threshold = max(2, round_half_up(16*N/656640)) otherwise
```

`656640` is the 1080-depth reference raster (`1080 x 608`). Regions below the
area threshold are uncertainty, not eligible surfaces. If this would remove
every surface, retain every nonempty region rather than inventing geometry.

Within otherwise eligible regions, reject both endpoints of every four-neighbor
one-eye displacement jump greater than or equal to `0.25` output pixels. Split
the remaining support into disconnected stable components. This prevents a
transitively connected ramp from surviving under one region label.

Upsample the stable component labels by nearest neighbor. The geodesic band is
the union of the four-connected stable-surface boundary and the dilated
uncertainty support. Only stable labels seed the RGB geodesic solve. Production
movement cost is:

```text
4 + 8 * max_abs_BGR_difference
```

The smaller base term makes crossing a real RGB edge more expensive than
following it. The native API retains `256 + 8 * difference` as its default for
historical callers; production v10 passes `4` and `8` explicitly.

Index cross-interior native samples for every stable component. If erosion
would leave a component empty, index all of its stable samples. Snap every
full-resolution edge-band primitive to the nearest indexed member of the
selected stable component. Outside the band, retain the existing bilinear
baseline.

After primitive reconstruction, expand the nearer stable surface by exactly one
four-connected output pixel only where the selected neighboring region differs
and the one-eye displacement jump is at least `0.25` pixels. Copy all geometry
primitives and validity from that neighbor. This keeps the outer antialiasing
pixel attached to the foreground without globally dilating small depth noise or
adding a user-visible anime mode.

## Background Proxy Repair

Run the existing 16-lane z-buffer splat once per eye and retain its pre-repair
coverage count. Accumulate row-bounded lane-hole histograms in the prebuilt
native extension while each band is resident. For `occlusion_fill=none`, output
remains the unfilled single-splat result.

For `occlusion_fill=background`:

1. Normalize every nonempty pre-repair pixel by `16 / coverage` to form a
   visible-colour proxy.
2. Put pixels with coverage at most 8 in the true inpaint core. Higher-coverage
   partial pixels retain their visible contribution and do not enlarge the
   expensive Telea domain.
3. Exclude donors on the foreground side up to the effective local limit, two
   pixels on the far side, and one pixel vertically. Run OpenCV Telea with a
   radius from 1 through 3.
4. Composite only the missing-lane fraction:

```text
output = pre_repair + proxy * (16 - coverage) / 16
```

Pixels with coverage 16 remain byte-identical. If the inpaint mask covers the
whole image, use the source frame as the bounded fallback. The configured
`occlusion_fill_max_px` still limits the local radius and donor exclusion; it
does not authorize the old global planner or a second splat.

Repair diagnostics retain the original coverage and hole-run evidence. Missing
lanes are reported through the existing fallback counters. Donor-index and
visit counters are zero because v10 builds no global donor index.

## Final Encoding

When the resolved encoder is `hevc_nvenc`, use this explicit quality contract:

```text
-c:v hevc_nvenc -pix_fmt yuv420p -preset p7 -tune hq
-rc vbr -cq 18 -b:v 0
```

Software encoders retain CRF 18. The final publication command fingerprint
binds these arguments, so a resumed publication cannot silently reuse the old
unspecified NVENC quality policy.

## Performance Contract

The production background path must not allocate `QualityFixedArenas`, call
`plan_quality_repairs`, build a safe-donor k-d index, or replay a second splat.
The one-pixel matte step is linear in output pixels and uses bounded dense
scratch.

On the reference RTX 4090 host, the seven selected 1920x1080 frames must each
complete the two-eye core render in less than 5.0 seconds. A fresh native build
and cold selected-frame run measured 2.56 through 4.58 seconds, with a median of
3.93 seconds. The older v9 path was approximately 8.9 seconds per selected
frame on the same host. Durable diagnostics on slow storage are measured
separately. These numbers are deliberately the reproducible cold-run evidence,
not the faster timings observed while iterating with warm process state.
The selected-frame measurements are retained in
`../reports/2026-08-27-stereo-quality-v10-selected-frames.json`.

## Identity And Compatibility

Quality output identity becomes
`torch-horizontal-16x-rgb-geodesic-repair-v10`. Fast identity remains
`torch-horizontal-16x-zbuffer-v3`. A resume whose Quality RGB stage records v9
must rerender rather than reuse RGB or diagnostics.

New CUDA jobs resolve to Quality; CPU jobs resolve to Fast; explicit mode
selection remains authoritative. No third mode is added.

## Acceptance Gates

- Synthetic transition slivers and transitive ramps resolve to stable surface
  primitives rather than intermediate depth.
- A large displacement edge expands the nearer matte by one pixel; small
  displacement differences do not expand.
- Complete-coverage pixels remain byte-identical, and partial pixels retain
  their valid subpixel contribution.
- Background production performs one splat per eye and never enters the v9
  planner or fixed arenas.
- NVENC publication carries the exact CQ18 command contract.
- Quality, diagnostics, resume, encoder, CLI, and Web tests pass.
- Frames `000089`, `000111`, `000171`, `000176`, `000231`, `000301`, and
  `000401` receive focused pre-encode visual review. Compression QA and DIBR QA
  remain separate evidence.

No algorithm can reconstruct occluded texture absent from the source. The
release gate therefore rejects wide stretching, detached duplicate contours,
missing thin structures, and temporal flicker, but does not claim exact hidden
background recovery.
