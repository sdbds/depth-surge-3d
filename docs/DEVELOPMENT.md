# Development Guide

## VDPP Boundaries

VDPP is integrated at the canonical relative-disparity artifact boundary. Do
not add backend-specific branches to V2, V3, See-Through, or relative MoGe-2. The production
constants are fixed by the released checkpoint: batch 1, window 32, overlap 4,
stride 28, `downsize=True`, FP32, and upstream affine continuation.

The minimal upstream source is vendored under
`src/depth_surge_3d/_vendor/vdpp`. `UPSTREAM.json` records release v1.0,
revision `73cc2b4dc6b3b5cfb2e37f51e452461e03fe26f5`, original paths, and file
digests. Mechanical import changes are documented in `NOTICE.md`; the upstream
Apache-2.0 license must remain in distributions.

The released v1.0 checkpoint has two zero-valued `shift_head.0.*` tensors that
the pinned public class does not declare or use. The adapter registers the
inert 1x1 compatibility slot before strict state-dict loading and records
`released-zero-shift-head-v1` in model identity. Do not drop keys, use
`strict=False`, or wire this slot into forward inference.

When updating the vendor:

1. Select an immutable upstream revision and checkpoint.
2. Copy only inference-required files and make only package-relative import
   changes.
3. Regenerate every original and vendored SHA-256 in `UPSTREAM.json`.
4. Bump the VDPP algorithm/port identity so existing stabilized caches cannot
   claim equivalence.
5. Re-run vendor, checkpoint, upstream-equivalence, storage, resume, and wheel
   tests.

## Verification

The normal suite uses deterministic fake forwards and never downloads the
116,485,370-byte checkpoint. Important focused commands are:

```powershell
python -m pytest tests/unit/test_vdpp_vendor.py `
  tests/unit/test_vdpp_artifact.py `
  tests/unit/test_vdpp_temporal_postprocessor.py `
  tests/unit/test_temporal_storage.py `
  tests/unit/test_temporal_stabilizer.py -q
```

Build the wheel and source distribution with `uv build`, then inspect that the
VDPP Python source, `LICENSE`, `NOTICE.md`, and `UPSTREAM.json` are present in
both. Demo files, assets, external Depth Anything copies, and downloaded model
artifacts must remain absent.

## Quality Gate

`scripts/evaluate_vdpp_quality.py` evaluates a versioned JSON manifest. Each
run points to a digest-pinned NPZ containing `[T,H,W]` arrays named `baseline`,
`vdpp`, and `ground_truth`, plus optional `valid_mask`. Inputs must already be
aligned to the benchmark's metric-depth scale.

The evaluator pins paper equations 9-10 as mean valid TGSE plus standard AbsRel
and delta-1. It uses seeds `[0,1,2]`, takes the median across repeats for each
sequence, then the unweighted median across sequences. A backend passes only
when unrounded values satisfy:

```text
TGSE_vdpp   <= 0.99 * TGSE_base
AbsRel_vdpp <= 1.02 * AbsRel_base
delta1_vdpp >= delta1_base - 0.02
```

Run it with:

```powershell
python scripts/evaluate_vdpp_quality.py benchmarks/vdpp/evaluation_manifest.json `
  --output benchmarks/vdpp/report.json
```

The harness does not make VDPP recommended by itself. Representative DA3 mono,
DA3 metric, V2 control, and blinded See-Through/anime review data still need to
be checked in and reviewed before removing the Experimental label.
