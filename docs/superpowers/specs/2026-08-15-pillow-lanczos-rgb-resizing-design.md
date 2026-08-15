# Pillow Lanczos RGB Resizing Design

## Status

The user approved this design on 2026-08-15 after comparing bounded float32
resampling behavior in Pillow and OpenCV. Canonical disparity resampling remains
bilinear because Lanczos and bicubic kernels can ring beyond the input range.

## Problem

The shared `resize_image` utility currently delegates ordinary output-frame
resizing to OpenCV `INTER_CUBIC`. The requested output-image quality policy is
Pillow Lanczos instead. Live previews have a different goal: they should be
cheap, stable reductions for browser transport rather than production-quality
output resizes.

## Accepted Design

### Production RGB and grayscale frames

- Keep the existing `resize_image(image, target_width, target_height)` entry
  point so all current production-frame callers change together.
- Implement it with `PIL.Image.fromarray`, `Image.Resampling.LANCZOS`, and a
  NumPy result.
- Preserve array channel order. BGR arrays loaded by OpenCV remain BGR because
  spatial resampling operates independently on each channel.
- Preserve the input `uint8` dtype, expected output shape, writable storage,
  and C-contiguous layout.
- Remove the OpenCV interpolation argument from the helper. No repository
  caller supplies it, and accepting OpenCV enum integers as Pillow enums would
  silently map to different algorithms.
- Remove or replace stale constants that claim ordinary resizing uses
  `cv2.INTER_CUBIC`.

### Live previews

- Keep preview resizing in OpenCV.
- Pass `interpolation=cv2.INTER_AREA` explicitly in both the file-backed and
  array-backed preview paths.
- Keep existing preview dimensions, throttling, PNG encoding, payload limits,
  and Socket.IO behavior unchanged.

### Resume and stage-cache invalidation

- Change `CROP_STAGE_ALGORITHM_VERSION` from `vr-crop-v1` to `vr-crop-v2`
  because fisheye-aware crop output is resized through `resize_image`.
- Change `VR_STAGE_ALGORITHM_VERSION` from `vr-layout-v1` to `vr-layout-v2`
  because VR assembly resizes mismatched eye images through `resize_image`.
- These version changes prevent completed manifests from reusing Bicubic PNGs
  after the production resize policy changes to Pillow Lanczos.

### Canonical depth

- Keep `StereoRenderer._resize_canonical` unchanged: Torch bilinear with
  `align_corners=False`.
- Do not add clipping or normalization.
- Do not alter depth-model input resizing, AI-upscaler internals, fisheye
  remapping, or forward-splat interpolation.

## Direct FFmpeg Boundary

The opt-in direct VR encoding path remains unchanged. Its FFmpeg
`bicubic+accurate_rnd` filter is not OpenCV, does not call `resize_image`, and
avoids materializing an assembled frame sequence. Forcing this path through
Pillow would contradict its performance contract and is outside the approved
scope.

## Error Handling

Pillow conversion and resize errors propagate as they do from the current
OpenCV helper. Target-dimension validation is not expanded in this change.
Production callers continue to supply decoded `uint8` image arrays and positive
dimensions.

## Verification

- Add a deterministic unit test comparing `resize_image` pixel-for-pixel with
  Pillow Lanczos for a structured `uint8` fixture.
- Retain and run the existing RGB, grayscale, upscale, and aspect-ratio tests.
- Add preview tests that observe `cv2.INTER_AREA` in both preview entry points.
- Add version-contract tests for the crop and VR assembly stages so old v1
  manifests cannot remain valid after the resize algorithm changes.
- Add a depth-contract regression test proving canonical resize output remains
  the existing Torch bilinear result.
- Run focused tests first, then the complete unit-test suite and configured
  formatting or lint checks for changed Python files.

## Non-goals

- Replacing OpenCV as the project's decoder, encoder, or fisheye-remap library.
- Changing canonical disparity interpolation.
- Changing Real-ESRGAN network interpolation.
- Changing See-Through model RGB preprocessing.
- Changing the direct FFmpeg encoder's resize filter.
- Claiming Pillow Lanczos and OpenCV Lanczos are pixel-identical.
