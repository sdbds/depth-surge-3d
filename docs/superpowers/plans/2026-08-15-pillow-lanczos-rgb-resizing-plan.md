# Pillow Lanczos RGB Resizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use Pillow Lanczos for production RGB/grayscale frame resizing, use
OpenCV `INTER_AREA` for live previews, and prove canonical disparity remains on
its existing Torch bilinear path.

**Architecture:** Keep the existing `resize_image` boundary so production frame
callers inherit one policy change. Preview transport stays inside `app.py` and
only makes its OpenCV downsampling mode explicit. Canonical disparity receives
a characterization test but no production-code change.

**Tech Stack:** Python 3.10-3.12, NumPy, Pillow >= 10.0.0, OpenCV, PyTorch,
pytest.

## Global Constraints

- Production decoded RGB/BGR and grayscale frames use Pillow
  `Image.Resampling.LANCZOS` through `resize_image`.
- Preserve `uint8` dtype, channel order, output shape, writable storage, and
  C-contiguous layout.
- Both Web preview entry points use OpenCV `cv2.INTER_AREA` explicitly.
- Canonical disparity remains Torch bilinear with `align_corners=False` and no
  clipping or normalization.
- Do not change depth-model preprocessing, Real-ESRGAN internals, fisheye
  remapping, forward splatting, or direct FFmpeg `bicubic+accurate_rnd`.
- Pillow is already a project dependency; do not edit dependency manifests.

---

## File Map

- `src/depth_surge_3d/utils/imaging/image_processing.py`: production image
  resize implementation.
- `src/depth_surge_3d/core/constants.py`: remove the unused stale OpenCV
  interpolation description.
- `src/depth_surge_3d/processing/frames/distortion_processor.py`: invalidate
  cached crop outputs produced with the old resize policy.
- `src/depth_surge_3d/processing/frames/vr_assembler.py`: invalidate cached VR
  layouts produced with the old resize policy.
- `app.py`: file-backed and array-backed preview resizes.
- `tests/unit/test_image_processing.py`: Pillow Lanczos behavior and ndarray
  contract.
- `tests/unit/test_direct_vr_progress.py`: preview interpolation behavior.
- `tests/unit/test_stereo_renderer.py`: unchanged canonical-depth interpolation
  contract.
- `tests/unit/test_stage_manifest.py`: old resize-stage manifest invalidation.

### Task 1: Lock The Existing Canonical Depth Contract

**Files:**
- Modify: `tests/unit/test_stereo_renderer.py`

**Interfaces:**
- Consumes: `StereoRenderer._resize_canonical(canonical: np.ndarray, render_shape: tuple[int, int]) -> torch.Tensor`.
- Produces: a regression guard for Torch bilinear output and bounded disparity.

- [ ] **Step 1: Add the characterization test**

```python
def test_canonical_resize_remains_bilinear_and_bounded() -> None:
    canonical = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    expected = torch.nn.functional.interpolate(
        torch.from_numpy(canonical).view(1, 1, 2, 2),
        size=(7, 9),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    actual = StereoRenderer._resize_canonical(canonical, (7, 9))

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert actual.min().item() >= 0.0
    assert actual.max().item() <= 1.0
```

- [ ] **Step 2: Run the characterization test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_stereo_renderer.py::test_canonical_resize_remains_bilinear_and_bounded -v
```

Expected: PASS against the pre-change implementation. This is deliberately a
behavior lock, not a failing feature test; no depth production code changes.

- [ ] **Step 3: Commit the depth behavior guard**

```powershell
git add tests/unit/test_stereo_renderer.py
git commit -m "test: lock bilinear canonical resizing"
```

### Task 2: Replace Production Bicubic Resizing With Pillow Lanczos

**Files:**
- Modify: `tests/unit/test_image_processing.py`
- Modify: `tests/unit/test_stage_manifest.py`
- Modify: `src/depth_surge_3d/utils/imaging/image_processing.py:10-35`
- Modify: `src/depth_surge_3d/core/constants.py:242-244`
- Modify: `src/depth_surge_3d/processing/frames/distortion_processor.py:37-38`
- Modify: `src/depth_surge_3d/processing/frames/vr_assembler.py:30-31`

**Interfaces:**
- Consumes: decoded `uint8` arrays shaped `[H,W]` or `[H,W,3]`.
- Produces: `resize_image(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray` with Pillow Lanczos pixels and owned C-contiguous storage.

- [ ] **Step 1: Write the failing Pillow parity test**

Add `from PIL import Image` and this test to `TestResizeImage`:

```python
def test_resize_matches_pillow_lanczos_and_returns_owned_array(self):
    image = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
    expected = np.array(
        Image.fromarray(image).resize((13, 11), Image.Resampling.LANCZOS),
        copy=True,
    )

    resized = resize_image(image, 13, 11)

    np.testing.assert_array_equal(resized, expected)
    assert resized.dtype == image.dtype
    assert resized.flags.c_contiguous
    assert resized.flags.writeable
```

- [ ] **Step 2: Write the failing cache-version regression test**

Add `import pytest` and the production stage modules to
`tests/unit/test_stage_manifest.py`:

```python
from src.depth_surge_3d.processing.frames import distortion_processor, vr_assembler
```

Then add:

```python
@pytest.mark.parametrize(
    ("stage", "legacy_version", "current_version"),
    [
        ("crop", "vr-crop-v1", distortion_processor.CROP_STAGE_ALGORITHM_VERSION),
        ("vr_assembly", "vr-layout-v1", vr_assembler.VR_STAGE_ALGORITHM_VERSION),
    ],
)
def test_resize_policy_versions_invalidate_v1_manifests(
    tmp_path,
    stage,
    legacy_version,
    current_version,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "output"
    output.mkdir()
    _write_rgb(source)
    _write_rgb(output / source.name)
    legacy_identity = build_stage_identity(
        stage=stage,
        algorithm_version=legacy_version,
        frame_names=[source.name],
        source_files=[source],
        settings={},
    )
    assert complete_stage(legacy_identity, (output,), shape=(4, 6, 3))
    current_identity = build_stage_identity(
        stage=stage,
        algorithm_version=current_version,
        frame_names=[source.name],
        source_files=[source],
        settings={},
    )

    assert not stage_is_reusable(current_identity, (output,))
```

- [ ] **Step 3: Run the tests and verify the old implementation fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_image_processing.py::TestResizeImage::test_resize_matches_pillow_lanczos_and_returns_owned_array tests/unit/test_stage_manifest.py::test_resize_policy_versions_invalidate_v1_manifests -v
```

Expected: the Pillow parity case FAILS because the helper uses OpenCV
`INTER_CUBIC`, and both manifest cases FAIL because the production stage
versions still equal their v1 legacy values.

- [ ] **Step 4: Implement Pillow Lanczos and invalidate old stage caches**

Import Pillow and replace the existing resize body and OpenCV interpolation
parameter:

```python
from PIL import Image


def resize_image(
    image: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Resize a uint8 image with Pillow Lanczos resampling."""

    pil_image = Image.fromarray(np.ascontiguousarray(image))
    resized = pil_image.resize(
        (target_width, target_height),
        resample=Image.Resampling.LANCZOS,
    )
    return np.array(resized, dtype=image.dtype, copy=True, order="C")
```

Remove the unused `DEFAULT_INTERPOLATION = "cv2.INTER_CUBIC"` constant. Keep
the module's `cv2` import because fisheye remapping still uses OpenCV. Change:

```python
CROP_STAGE_ALGORITHM_VERSION = "vr-crop-v2"
VR_STAGE_ALGORITHM_VERSION = "vr-layout-v2"
```

These bumps make the current identities differ from manifests written for
Bicubic output.

- [ ] **Step 5: Run image-processing and manifest tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_image_processing.py tests/unit/test_stage_manifest.py -v
```

Expected: all tests PASS, including RGB, grayscale, upscale, aspect-ratio, and
the new exact Pillow Lanczos and v1 manifest invalidation comparisons.

- [ ] **Step 6: Commit production resizing and cache invalidation**

```powershell
git add tests/unit/test_image_processing.py tests/unit/test_stage_manifest.py src/depth_surge_3d/utils/imaging/image_processing.py src/depth_surge_3d/core/constants.py src/depth_surge_3d/processing/frames/distortion_processor.py src/depth_surge_3d/processing/frames/vr_assembler.py
git commit -m "feat: use Pillow Lanczos for frame resizing"
```

### Task 3: Make Preview Downsampling Use OpenCV INTER_AREA

**Files:**
- Modify: `tests/unit/test_direct_vr_progress.py`
- Modify: `app.py:516-520`
- Modify: `app.py:595-599`

**Interfaces:**
- Consumes: `ProgressCallback.send_preview_frame(...)` and
  `ProgressCallback.send_preview_frame_from_array(...)`.
- Produces: unchanged preview payloads whose resize calls explicitly select
  `cv2.INTER_AREA`.

- [ ] **Step 1: Write the failing parameterized preview test**

Add `import numpy as np` and `import pytest`, then add:

```python
@pytest.mark.parametrize("source_kind", ["file", "array"])
def test_preview_downsampling_uses_inter_area(tmp_path, source_kind):
    callback = web_app.ProgressCallback(
        "test-session",
        total_frames=1,
        preview_update_interval=0,
    )
    callback.preview_downscale_width = 6
    frame = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    original_resize = web_app.cv2.resize

    with (
        patch.object(web_app.cv2, "resize", wraps=original_resize) as resize_spy,
        patch.object(web_app.socketio, "emit"),
    ):
        if source_kind == "file":
            frame_path = tmp_path / "frame.png"
            assert web_app.cv2.imwrite(str(frame_path), frame)
            callback.send_preview_frame(frame_path, "stereo_left", 1)
        else:
            callback.send_preview_frame_from_array(frame, "stereo_left", 1)

    assert resize_spy.called
    assert resize_spy.call_args.kwargs.get("interpolation") == web_app.cv2.INTER_AREA
```

- [ ] **Step 2: Run the preview test and verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_direct_vr_progress.py::test_preview_downsampling_uses_inter_area -v
```

Expected: two FAIL results because both current calls omit the interpolation
keyword and therefore use OpenCV's default `INTER_LINEAR`.

- [ ] **Step 3: Add INTER_AREA to both preview calls**

Change each call to:

```python
frame_small = cv2.resize(
    frame,
    (new_width, new_height),
    interpolation=cv2.INTER_AREA,
)
```

Use `frame_array` instead of `frame` in the array-backed method. Do not change
preview sizing, encoding, throttling, or transport behavior.

- [ ] **Step 4: Run preview and neighboring app tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_direct_vr_progress.py tests/unit/test_see_through_entrypoints.py tests/unit/test_video_encoder.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit preview interpolation**

```powershell
git add app.py tests/unit/test_direct_vr_progress.py
git commit -m "feat: use area resampling for previews"
```

### Task 4: Verify Scope And Regression Safety

**Files:**
- Verify all files changed since design commit `ac76226`.

**Interfaces:**
- Consumes: the three preceding commits.
- Produces: fresh test, format, and scope evidence.

- [ ] **Step 1: Verify focused behavior together**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_image_processing.py tests/unit/test_direct_vr_progress.py tests/unit/test_stereo_renderer.py tests/unit/test_stage_manifest.py -q
```

Expected: all focused tests PASS.

- [ ] **Step 2: Run the complete unit suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit -q
```

Expected: all unit tests PASS.

- [ ] **Step 3: Check formatting and patch hygiene**

Run:

```powershell
.\.venv\Scripts\python.exe -m black --check app.py src/depth_surge_3d/utils/imaging/image_processing.py src/depth_surge_3d/core/constants.py src/depth_surge_3d/processing/frames/distortion_processor.py src/depth_surge_3d/processing/frames/vr_assembler.py tests/unit/test_image_processing.py tests/unit/test_direct_vr_progress.py tests/unit/test_stereo_renderer.py tests/unit/test_stage_manifest.py
git diff --check ac76226..HEAD
```

Expected: Black reports no changes needed and Git reports no whitespace errors.

- [ ] **Step 4: Audit excluded paths and final diff**

Run:

```powershell
git diff ac76226..HEAD -- src/depth_surge_3d/rendering/stereo_renderer.py src/depth_surge_3d/inference/depth src/depth_surge_3d/inference/upscaling src/depth_surge_3d/processing/video/video_encoder.py
git status --short
git log --oneline ac76226..HEAD
```

Expected: no production diff in depth, upscaling, renderer, or direct FFmpeg
paths; status only shows the user's pre-existing unrelated working-tree changes;
the log contains the scoped test and implementation commits.
