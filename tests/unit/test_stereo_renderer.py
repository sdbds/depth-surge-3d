from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from src.depth_surge_3d.rendering.forward_splat import (
    HORIZONTAL_SUBPIXELS,
    SubpixelSplatResult,
)
from src.depth_surge_3d.rendering.stereo_renderer import (
    GPU_TEMP_BUDGET,
    SPLAT_BYTES_PER_PIXEL,
    StereoRenderResult,
    StereoRenderer,
    StereoRenderSettings,
    _convert_image_band,
    _downsample_subpixel_band,
    _fill_background_band,
    calculate_band_height,
    calculate_eye_sample_offsets,
)


def _subpixel_result(
    colour: torch.Tensor,
    valid: torch.Tensor,
    disparity: torch.Tensor,
) -> SubpixelSplatResult:
    return SubpixelSplatResult(colour=colour, disparity=disparity, valid=valid)


def _empty_render_result(frame: np.ndarray) -> StereoRenderResult:
    height, width = frame.shape[:2]
    image = np.zeros_like(frame)
    mask = np.zeros((height, width), dtype=np.bool_)
    return StereoRenderResult(
        left_image=image.copy(),
        right_image=image.copy(),
        left_valid_mask=mask.copy(),
        right_valid_mask=mask.copy(),
        left_hole_mask=~mask,
        right_hole_mask=~mask,
    )


def test_band_height_uses_measured_sixteen_sample_bound() -> None:
    expected = GPU_TEMP_BUDGET // (3840 * SPLAT_BYTES_PER_PIXEL)

    assert SPLAT_BYTES_PER_PIXEL == 1280
    assert calculate_band_height(3840, 2160) == expected
    assert calculate_band_height(16, 3) == 3


def test_renderer_rejects_full_frame_source_indexes_that_do_not_fit_uint32() -> None:
    source = np.broadcast_to(
        np.zeros((1, 1, 3), dtype=np.uint8),
        (65536, 65536, 3),
    )
    canonical = np.broadcast_to(
        np.zeros((1, 1), dtype=np.float32),
        (65536, 65536),
    )

    with pytest.raises(ValueError, match="source indexes must fit in 32 bits"):
        StereoRenderer(device="cpu").render(source, canonical)


def test_full_frame_eye_offsets_use_host_float64_and_int32_storage() -> None:
    canonical = torch.full((1, 100), 0.5, dtype=torch.float32)
    canonical[0, 0] = 0.0
    canonical[0, 1] = 1.0

    left, right = calculate_eye_sample_offsets(
        canonical,
        StereoRenderSettings(stereo_strength=5.0, convergence=0.5),
    )

    assert left.dtype == np.int32
    assert right.dtype == np.int32
    assert left.flags.c_contiguous and right.flags.c_contiguous
    extreme_offset = HORIZONTAL_SUBPIXELS * 5 // 4
    assert left[0, :3].tolist() == [-extreme_offset, extreme_offset, 0]
    assert right[0, :3].tolist() == [extreme_offset, -extreme_offset, 0]


def test_half_lane_boundaries_obey_ceil_without_epsilon() -> None:
    boundary = np.float32(2.0 / HORIZONTAL_SUBPIXELS)
    below = np.nextafter(boundary, np.float32(0.0))
    above = np.nextafter(boundary, np.float32(1.0))
    canonical = torch.full((1, 100), 0.5, dtype=torch.float32)
    canonical[0, :3] = torch.tensor([below, boundary, above], dtype=torch.float32)

    left, _right = calculate_eye_sample_offsets(
        canonical,
        StereoRenderSettings(stereo_strength=0.5, convergence=0.0),
    )

    assert left[0, :3].tolist() == [0, 0, 1]


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


def test_geometry_maps_are_built_once_and_reused_after_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((5, 4, 3), dtype=np.uint8)
    renderer = StereoRenderer(
        device="cpu",
        temporary_budget_bytes=4 * SPLAT_BYTES_PER_PIXEL * 4,
    )
    geometry_calls = 0
    attempts: list[tuple[int, int]] = []

    def fake_geometry(
        canonical: torch.Tensor,
        settings: StereoRenderSettings,
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal geometry_calls
        del settings
        geometry_calls += 1
        offsets = np.zeros(canonical.shape, dtype=np.int32)
        return offsets, offsets.copy()

    def fake_render(
        source: np.ndarray,
        resized_canonical: torch.Tensor,
        eye_offsets: tuple[np.ndarray, np.ndarray],
        settings: StereoRenderSettings,
        band_height: int,
    ) -> StereoRenderResult:
        del resized_canonical, settings
        attempts.append((band_height, id(eye_offsets)))
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError("simulated")
        return _empty_render_result(source)

    monkeypatch.setattr(
        "src.depth_surge_3d.rendering.stereo_renderer.calculate_eye_sample_offsets",
        fake_geometry,
    )
    monkeypatch.setattr(renderer, "_render_with_band_height", fake_render)

    renderer.render(frame, np.full((1, 1), 0.5, dtype=np.float32))

    assert geometry_calls == 1
    assert [height for height, _identity in attempts] == [4, 2]
    assert attempts[0][1] == attempts[1][1]


def test_real_oom_retry_is_byte_identical_to_direct_retry_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = np.random.default_rng(29)
    frame = generator.integers(0, 256, size=(5, 11, 3), dtype=np.uint8)
    canonical = generator.random((5, 11), dtype=np.float32)
    settings = StereoRenderSettings(stereo_strength=3.0, occlusion_fill="background")
    retry_budget = 11 * SPLAT_BYTES_PER_PIXEL * 2
    expected = StereoRenderer(
        device="cpu",
        temporary_budget_bytes=retry_budget,
    ).render(frame, canonical, settings)
    renderer = StereoRenderer(
        device="cpu",
        temporary_budget_bytes=11 * SPLAT_BYTES_PER_PIXEL * 4,
    )
    original = renderer._render_with_band_height
    attempted_heights: list[int] = []

    def fail_first_attempt(
        source: np.ndarray,
        resized_canonical: torch.Tensor,
        eye_offsets: tuple[np.ndarray, np.ndarray],
        render_settings: StereoRenderSettings,
        band_height: int,
    ) -> StereoRenderResult:
        attempted_heights.append(band_height)
        if len(attempted_heights) == 1:
            raise torch.cuda.OutOfMemoryError("simulated")
        return original(
            source,
            resized_canonical,
            eye_offsets,
            render_settings,
            band_height,
        )

    monkeypatch.setattr(renderer, "_render_with_band_height", fail_first_attempt)

    actual = renderer.render(frame, canonical, settings)

    assert attempted_heights == [4, 2]
    for field in StereoRenderResult.__dataclass_fields__:
        assert np.array_equal(getattr(actual, field), getattr(expected, field))


def test_renderer_slices_full_frame_offsets_and_uses_global_source_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[torch.Tensor, int]] = []

    def fake_splat(
        image: torch.Tensor,
        canonical: torch.Tensor,
        sample_offsets: torch.Tensor,
        *,
        source_index_offset: int,
    ) -> SubpixelSplatResult:
        calls.append((sample_offsets.detach().cpu(), source_index_offset))
        colour = image.to(torch.float32).repeat_interleave(HORIZONTAL_SUBPIXELS, dim=1)
        disparity = canonical.repeat_interleave(HORIZONTAL_SUBPIXELS, dim=1)
        valid = torch.ones(disparity.shape, dtype=torch.bool, device=image.device)
        return _subpixel_result(colour, valid, disparity)

    monkeypatch.setattr(
        "src.depth_surge_3d.rendering.stereo_renderer.forward_splat_band",
        fake_splat,
    )
    width = 5
    renderer = StereoRenderer(
        device="cpu",
        temporary_budget_bytes=width * SPLAT_BYTES_PER_PIXEL * 2,
    )
    frame = np.arange(5 * width * 3, dtype=np.uint8).reshape(5, width, 3)

    result = renderer.render(
        frame,
        np.full((5, width), 0.5, dtype=np.float32),
        StereoRenderSettings(stereo_strength=0.0, occlusion_fill="none"),
    )

    assert [(value.shape[0], offset) for value, offset in calls] == [
        (2, 0),
        (2, 10),
        (1, 20),
        (2, 0),
        (2, 10),
        (1, 20),
    ]
    assert all(value.dtype == torch.int32 for value, _offset in calls)
    assert np.array_equal(result.left_image, frame)
    assert np.array_equal(result.right_image, frame)


def test_background_fill_prefers_farther_discrete_candidate() -> None:
    colour = torch.tensor(
        [[[200.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 100.0]]]
    )
    valid = torch.tensor([[True, False, False, True]])
    disparity = torch.tensor([[1.0, -torch.inf, -torch.inf, 0.0]])

    filled, hole_mask = _fill_background_band(
        _subpixel_result(colour, valid, disparity),
        max_gap_samples=2,
    )

    torch.testing.assert_close(filled[0, 1], colour[0, 3])
    torch.testing.assert_close(filled[0, 2], colour[0, 3])
    assert not hole_mask.any()


def test_background_fill_uses_distance_then_left_for_equal_depth() -> None:
    colour = torch.tensor(
        [[[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 20.0, 0.0]]]
    )
    valid = torch.tensor([[True, False, False, False, True]])
    disparity = torch.tensor([[0.5, -torch.inf, -torch.inf, -torch.inf, 0.5]])

    filled, hole_mask = _fill_background_band(
        _subpixel_result(colour, valid, disparity),
        max_gap_samples=3,
    )

    torch.testing.assert_close(filled[0, 1], colour[0, 0])
    torch.testing.assert_close(filled[0, 2], colour[0, 0])
    torch.testing.assert_close(filled[0, 3], colour[0, 4])
    assert not hole_mask.any()


def test_background_fill_extends_only_bounded_frame_edge_runs() -> None:
    colour = torch.zeros((2, 6, 3), dtype=torch.float32)
    colour[0, 2] = torch.tensor([10.0, 20.0, 30.0])
    colour[1, 3] = torch.tensor([40.0, 50.0, 60.0])
    valid = torch.tensor(
        [
            [False, False, True, False, False, False],
            [False, False, False, True, False, False],
        ]
    )
    disparity = torch.where(valid, torch.tensor(0.25), torch.tensor(-torch.inf))

    filled, holes = _fill_background_band(
        _subpixel_result(colour, valid, disparity),
        max_gap_samples=2,
    )

    assert torch.equal(filled[0, :2], colour[0, 2].expand(2, 3))
    assert holes[0, 3:].all()
    assert holes[1, :3].all()
    assert torch.equal(filled[1, 4:], colour[1, 3].expand(2, 3))


def test_downsampling_uses_fixed_lane_average_and_ties_to_even() -> None:
    zero = [[0.0, 0.0, 0.0]] * 8
    first = zero + [[255.0, 255.0, 255.0]] * 8
    second = zero + [[253.0, 253.0, 253.0]] * 8
    fine = torch.tensor([[*first, *second]])

    averaged = _downsample_subpixel_band(fine)
    converted = _convert_image_band(averaged, np.dtype(np.uint8))

    assert converted[0, :, 0].tolist() == [128, 126]


def test_none_darkens_partial_coverage_while_background_fills_it() -> None:
    frame = np.full((1, 20, 3), 100, dtype=np.uint8)
    canonical = np.ones((1, 20), dtype=np.float32)
    renderer = StereoRenderer(device="cpu")

    none = renderer.render(
        frame,
        canonical,
        StereoRenderSettings(stereo_strength=5.0, convergence=0.0, occlusion_fill="none"),
    )
    background = renderer.render(
        frame,
        canonical,
        StereoRenderSettings(
            stereo_strength=5.0,
            convergence=0.0,
            occlusion_fill="background",
        ),
    )

    assert none.left_image[0, 0, 0] == 50
    assert none.left_valid_mask[0, 0]
    assert not none.left_hole_mask[0, 0]
    assert background.left_image[0, 0, 0] == 100
    assert not background.left_hole_mask.any()


def test_uint8_strength_zero_is_byte_exact_for_both_eyes() -> None:
    generator = np.random.default_rng(3)
    frame = generator.integers(0, 256, size=(4, 13, 3), dtype=np.uint8)
    canonical = generator.random((2, 7), dtype=np.float32)

    result = StereoRenderer(device="cpu").render(
        frame,
        canonical,
        StereoRenderSettings(stereo_strength=0.0, occlusion_fill="none"),
    )

    assert np.array_equal(result.left_image, frame)
    assert np.array_equal(result.right_image, frame)
    assert result.left_valid_mask.all() and result.right_valid_mask.all()
    assert not result.left_hole_mask.any() and not result.right_hole_mask.any()


def test_row_band_height_does_not_change_any_output_byte_or_mask() -> None:
    generator = np.random.default_rng(13)
    frame = generator.integers(0, 256, size=(6, 17, 3), dtype=np.uint8)
    canonical = generator.random((6, 17), dtype=np.float32)
    settings = StereoRenderSettings(stereo_strength=5.0, occlusion_fill="background")
    full_band = StereoRenderer(device="cpu")
    one_row = StereoRenderer(
        device="cpu",
        temporary_budget_bytes=17 * SPLAT_BYTES_PER_PIXEL,
    )

    expected = full_band.render(frame, canonical, settings)
    actual = one_row.render(frame, canonical, settings)

    for field in StereoRenderResult.__dataclass_fields__:
        assert np.array_equal(getattr(actual, field), getattr(expected, field))


def test_second_cuda_oom_reports_frame_and_both_attempted_heights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((5, 4, 3), dtype=np.uint8)
    renderer = StereoRenderer(
        device="cpu",
        temporary_budget_bytes=4 * SPLAT_BYTES_PER_PIXEL * 4,
    )

    def always_oom(*_args, **_kwargs) -> StereoRenderResult:
        raise torch.cuda.OutOfMemoryError("simulated")

    monkeypatch.setattr(renderer, "_render_with_band_height", always_oom)

    with pytest.raises(RuntimeError, match=r"4x5.*band heights 4 and 2"):
        renderer.render(frame, np.full((1, 1), 0.5, dtype=np.float32))


def test_renderer_catches_the_cuda_oom_type_supported_by_torch_2_0() -> None:
    text = inspect.getsource(StereoRenderer.render)

    assert "except torch.cuda.OutOfMemoryError" in text
    assert "except torch.OutOfMemoryError" not in text


@pytest.mark.parametrize("strength", [0.0, 5.0])
def test_strength_boundaries_are_valid(strength: float) -> None:
    settings = StereoRenderSettings(stereo_strength=strength)

    assert 0.0 <= settings.stereo_strength <= 5.0


@pytest.mark.parametrize(
    "values",
    [
        {"stereo_strength": -0.1},
        {"stereo_strength": 5.1},
        {"convergence": -0.1},
        {"convergence": 1.1},
        {"occlusion_fill": "telea"},
    ],
)
def test_render_settings_reject_invalid_values(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        StereoRenderSettings(**values)


def test_renderer_rejects_nonfinite_or_out_of_range_canonical() -> None:
    renderer = StereoRenderer(device="cpu")
    frame = np.zeros((1, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="finite"):
        renderer.render(frame, np.array([[0.5, np.nan]], dtype=np.float32))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        renderer.render(frame, np.array([[0.5, 1.1]], dtype=np.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_renderer_cpu_cuda_parity_is_byte_exact() -> None:
    generator = np.random.default_rng(17)
    frame = generator.integers(0, 256, size=(5, 17, 3), dtype=np.uint8)
    canonical = generator.random((5, 17), dtype=np.float32)
    settings = StereoRenderSettings(stereo_strength=4.0, occlusion_fill="background")

    cpu = StereoRenderer(device="cpu").render(frame, canonical, settings)
    cuda = StereoRenderer(device="cuda").render(frame, canonical, settings)

    for field in StereoRenderResult.__dataclass_fields__:
        assert np.array_equal(getattr(cuda, field), getattr(cpu, field))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_renderer_peak_cuda_live_set_fits_configured_bytes_with_headroom() -> None:
    height, width = 128, 1024
    renderer = StereoRenderer(
        device="cuda",
        temporary_budget_bytes=height * width * SPLAT_BYTES_PER_PIXEL,
    )
    renderer.render(
        np.zeros((1, width, 3), dtype=np.uint8),
        np.full((1, width), 0.5, dtype=np.float32),
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    initial_bytes = torch.cuda.memory_allocated()

    generator = np.random.default_rng(23)
    frame = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    canonical = generator.random((height, width), dtype=np.float32)
    result = renderer.render(
        frame,
        canonical,
        StereoRenderSettings(stereo_strength=5.0, occlusion_fill="background"),
    )
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated() - initial_bytes

    assert result.left_image.shape == frame.shape
    assert peak_bytes * 1.25 <= height * width * SPLAT_BYTES_PER_PIXEL
