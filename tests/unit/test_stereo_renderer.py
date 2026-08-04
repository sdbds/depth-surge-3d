from __future__ import annotations

import numpy as np
import pytest
import torch

from src.depth_surge_3d.rendering.forward_splat import SplatBandResult
from src.depth_surge_3d.rendering.stereo_renderer import (
    GPU_TEMP_BUDGET,
    SPLAT_BYTES_PER_PIXEL,
    StereoRenderResult,
    StereoRenderer,
    StereoRenderSettings,
    _fill_background_band,
    calculate_band_height,
)


def _splat_result(
    image: torch.Tensor,
    valid: torch.Tensor,
    projected: torch.Tensor,
) -> SplatBandResult:
    return SplatBandResult(
        image=image,
        valid_mask=valid,
        projected_disparity=projected,
        accumulated_weight=valid.to(dtype=torch.float32),
    )


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


def test_band_height_uses_fixed_budget_and_int64_aware_estimate() -> None:
    expected = GPU_TEMP_BUDGET // (3840 * SPLAT_BYTES_PER_PIXEL)

    assert SPLAT_BYTES_PER_PIXEL == 192
    assert calculate_band_height(3840, 2160) == expected
    assert calculate_band_height(16, 3) == 3


def test_renderer_resizes_canonical_before_target_width_disparity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[torch.Tensor] = []

    def fake_splat(
        image: torch.Tensor,
        disparity: torch.Tensor,
        eye_sign: int,
    ) -> SplatBandResult:
        del eye_sign
        captured.append(disparity.detach().cpu())
        valid = torch.ones(disparity.shape, dtype=torch.bool, device=image.device)
        return _splat_result(image.to(torch.float32), valid, disparity)

    monkeypatch.setattr(
        "src.depth_surge_3d.rendering.stereo_renderer.forward_splat_band",
        fake_splat,
    )
    frame = np.zeros((2, 8, 3), dtype=np.uint8)
    native_canonical = np.ones((1, 1), dtype=np.float32)
    renderer = StereoRenderer(device="cpu")

    renderer.render(
        frame,
        native_canonical,
        StereoRenderSettings(stereo_strength=2.0, convergence=0.5),
    )

    expected_disparity = torch.full((2, 8), 0.08, dtype=torch.float32)
    assert len(captured) == 2
    torch.testing.assert_close(captured[0], expected_disparity)
    torch.testing.assert_close(captured[1], expected_disparity)


def test_renderer_uses_bilinear_canonical_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[torch.Tensor] = []

    def fake_splat(
        image: torch.Tensor,
        disparity: torch.Tensor,
        eye_sign: int,
    ) -> SplatBandResult:
        del eye_sign
        captured.append(disparity.detach().cpu())
        valid = torch.ones(disparity.shape, dtype=torch.bool, device=image.device)
        return _splat_result(image.to(torch.float32), valid, disparity)

    monkeypatch.setattr(
        "src.depth_surge_3d.rendering.stereo_renderer.forward_splat_band",
        fake_splat,
    )

    StereoRenderer(device="cpu").render(
        np.zeros((1, 4, 3), dtype=np.uint8),
        np.array([[0.0, 1.0]], dtype=np.float32),
        StereoRenderSettings(stereo_strength=5.0, convergence=0.0),
    )

    expected = torch.tensor([[0.0, 0.05, 0.15, 0.2]], dtype=torch.float32)
    torch.testing.assert_close(captured[0], expected)


def test_renderer_processes_complete_bands_and_eyes_sequentially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, str]] = []

    def fake_splat(
        image: torch.Tensor,
        disparity: torch.Tensor,
        eye_sign: int,
    ) -> SplatBandResult:
        calls.append((eye_sign, image.shape[0], image.device.type))
        valid = torch.ones(disparity.shape, dtype=torch.bool, device=image.device)
        return _splat_result(image.to(torch.float32), valid, disparity)

    monkeypatch.setattr(
        "src.depth_surge_3d.rendering.stereo_renderer.forward_splat_band",
        fake_splat,
    )
    width = 5
    budget = width * SPLAT_BYTES_PER_PIXEL * 2
    renderer = StereoRenderer(device="cpu", temporary_budget_bytes=budget)

    result = renderer.render(
        np.zeros((5, width, 3), dtype=np.uint8),
        np.full((5, width), 0.5, dtype=np.float32),
        StereoRenderSettings(occlusion_fill="none"),
    )

    assert calls == [
        (1, 2, "cpu"),
        (1, 2, "cpu"),
        (1, 1, "cpu"),
        (-1, 2, "cpu"),
        (-1, 2, "cpu"),
        (-1, 1, "cpu"),
    ]
    assert isinstance(result.left_image, np.ndarray)
    assert isinstance(result.right_image, np.ndarray)
    assert not hasattr(result, "projected_disparity")


def test_background_fill_prefers_farther_candidate_over_nearer_distance() -> None:
    image = torch.tensor([[[200.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 100.0]]])
    valid = torch.tensor([[True, False, False, True]])
    projected = torch.tensor([[1.0, 0.0, 0.0, -1.0]])

    filled, hole_mask = _fill_background_band(
        _splat_result(image, valid, projected),
        max_gap_width=2,
    )

    torch.testing.assert_close(filled[0, 1], image[0, 3])
    torch.testing.assert_close(filled[0, 2], image[0, 3])
    assert not hole_mask.any()


def test_background_fill_uses_distance_only_when_depths_tie() -> None:
    image = torch.tensor(
        [
            [
                [10.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 20.0, 0.0],
            ]
        ]
    )
    valid = torch.tensor([[True, False, False, False, True]])
    projected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])

    filled, hole_mask = _fill_background_band(
        _splat_result(image, valid, projected),
        max_gap_width=3,
    )

    torch.testing.assert_close(filled[0, 1], image[0, 0])
    torch.testing.assert_close(filled[0, 2], image[0, 0])
    torch.testing.assert_close(filled[0, 3], image[0, 4])
    assert not hole_mask.any()


def test_background_fill_leaves_wide_and_fully_invalid_runs_as_holes() -> None:
    image = torch.zeros((2, 5, 3), dtype=torch.float32)
    image[0, 0] = torch.tensor([10.0, 0.0, 0.0])
    image[0, 4] = torch.tensor([0.0, 20.0, 0.0])
    valid = torch.tensor([[True, False, False, False, True], [False, False, False, False, False]])
    projected = torch.zeros((2, 5), dtype=torch.float32)

    filled, hole_mask = _fill_background_band(
        _splat_result(image, valid, projected),
        max_gap_width=2,
    )

    assert torch.equal(filled, image)
    assert hole_mask[0, 1:4].all()
    assert hole_mask[1].all()


def test_valid_black_pixels_are_not_holes_and_output_restores_input_dtype() -> None:
    frame = np.zeros((3, 7, 3), dtype=np.uint8)
    canonical = np.full((3, 7), 0.5, dtype=np.float32)

    result = StereoRenderer(device="cpu").render(
        frame,
        canonical,
        StereoRenderSettings(occlusion_fill="background"),
    )

    assert result.left_image.dtype == np.uint8
    assert result.right_image.dtype == np.uint8
    assert np.array_equal(result.left_image, frame)
    assert np.array_equal(result.right_image, frame)
    assert result.left_valid_mask.all()
    assert result.right_valid_mask.all()
    assert not result.left_hole_mask.any()
    assert not result.right_hole_mask.any()


def test_occlusion_fill_none_preserves_explicit_black_holes() -> None:
    frame = np.full((1, 100, 3), 127, dtype=np.uint8)
    canonical = np.ones((1, 100), dtype=np.float32)

    result = StereoRenderer(device="cpu").render(
        frame,
        canonical,
        StereoRenderSettings(
            stereo_strength=5.0,
            convergence=0.0,
            occlusion_fill="none",
        ),
    )

    assert result.left_hole_mask.any()
    assert np.all(result.left_image[result.left_hole_mask] == 0)


def test_row_band_height_does_not_change_rendered_bytes() -> None:
    generator = np.random.default_rng(13)
    frame = generator.integers(0, 256, size=(6, 17, 3), dtype=np.uint8)
    canonical = np.linspace(0.0, 1.0, 17, dtype=np.float32)[None, :]
    settings = StereoRenderSettings(stereo_strength=5.0, occlusion_fill="background")
    full_band = StereoRenderer(device="cpu")
    one_row = StereoRenderer(
        device="cpu",
        temporary_budget_bytes=17 * SPLAT_BYTES_PER_PIXEL,
    )

    expected = full_band.render(frame, canonical, settings)
    actual = one_row.render(frame, canonical, settings)

    assert np.array_equal(actual.left_image, expected.left_image)
    assert np.array_equal(actual.right_image, expected.right_image)
    assert np.array_equal(actual.left_valid_mask, expected.left_valid_mask)
    assert np.array_equal(actual.right_valid_mask, expected.right_valid_mask)
    assert np.array_equal(actual.left_hole_mask, expected.left_hole_mask)
    assert np.array_equal(actual.right_hole_mask, expected.right_hole_mask)


def test_cuda_oom_retries_once_with_half_band_height(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((5, 4, 3), dtype=np.uint8)
    renderer = StereoRenderer(
        device="cpu",
        temporary_budget_bytes=4 * SPLAT_BYTES_PER_PIXEL * 4,
    )
    attempts: list[int] = []

    def fake_render(
        source: np.ndarray,
        resized_canonical: torch.Tensor,
        settings: StereoRenderSettings,
        band_height: int,
    ) -> StereoRenderResult:
        del resized_canonical, settings
        attempts.append(band_height)
        if len(attempts) == 1:
            raise torch.OutOfMemoryError("simulated")
        return _empty_render_result(source)

    monkeypatch.setattr(renderer, "_render_with_band_height", fake_render)

    renderer.render(frame, np.full((1, 1), 0.5, dtype=np.float32))

    assert attempts == [4, 2]


def test_second_cuda_oom_reports_frame_and_both_attempted_heights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((5, 4, 3), dtype=np.uint8)
    renderer = StereoRenderer(
        device="cpu",
        temporary_budget_bytes=4 * SPLAT_BYTES_PER_PIXEL * 4,
    )

    def always_oom(
        source: np.ndarray,
        resized_canonical: torch.Tensor,
        settings: StereoRenderSettings,
        band_height: int,
    ) -> StereoRenderResult:
        del source, resized_canonical, settings, band_height
        raise torch.OutOfMemoryError("simulated")

    monkeypatch.setattr(renderer, "_render_with_band_height", always_oom)

    with pytest.raises(
        RuntimeError,
        match=r"4x5.*band heights 4 and 2",
    ):
        renderer.render(frame, np.full((1, 1), 0.5, dtype=np.float32))


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
def test_renderer_cpu_cuda_parity() -> None:
    frame = np.arange(5 * 11 * 3, dtype=np.uint8).reshape(5, 11, 3)
    canonical = np.linspace(0.0, 1.0, 11, dtype=np.float32)[None, :]
    settings = StereoRenderSettings(stereo_strength=4.0, occlusion_fill="background")

    cpu = StereoRenderer(device="cpu").render(frame, canonical, settings)
    cuda = StereoRenderer(device="cuda").render(frame, canonical, settings)

    assert np.array_equal(cuda.left_image, cpu.left_image)
    assert np.array_equal(cuda.right_image, cpu.right_image)
    assert np.array_equal(cuda.left_valid_mask, cpu.left_valid_mask)
    assert np.array_equal(cuda.right_valid_mask, cpu.right_valid_mask)
    assert np.array_equal(cuda.left_hole_mask, cpu.left_hole_mask)
    assert np.array_equal(cuda.right_hole_mask, cpu.right_hole_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_renderer_peak_cuda_memory_fits_fixed_band_budget() -> None:
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
    baseline = torch.cuda.memory_allocated()

    generator = np.random.default_rng(23)
    frame = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    canonical = generator.random((height, width), dtype=np.float32)
    result = renderer.render(
        frame,
        canonical,
        StereoRenderSettings(stereo_strength=5.0, occlusion_fill="background"),
    )
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated() - baseline

    assert result.left_image.shape == frame.shape
    assert peak_bytes <= height * width * SPLAT_BYTES_PER_PIXEL
