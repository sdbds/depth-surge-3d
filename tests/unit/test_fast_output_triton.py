"""Fused Fast output retains the legacy packet and balanced RGB reduction."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.depth_surge_3d.rendering import fast_output_triton as output_module
from src.depth_surge_3d.rendering import stereo_renderer as renderer_module


requires_triton_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or output_module.triton is None,
    reason="Triton CUDA is unavailable",
)


@requires_triton_cuda
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("width", [1, 7, 129])
def test_fused_fast_output_matches_legacy_packet(background, width):
    rng = np.random.default_rng(20260831 + width)
    height = 3
    colour = torch.from_numpy(
        rng.integers(0, 256, size=(height, width * 16, 3)).astype(np.float32)
    ).cuda()
    prefill = torch.from_numpy(rng.random((height, width, 16)) > 0.6).cuda()
    filled = (~prefill) & torch.from_numpy(rng.random((height, width, 16)) > 0.5).cuda()
    unresolved = (~prefill) & (~filled)
    reduced = renderer_module._downsample_subpixel_band(colour)
    repair = renderer_module._fast_repair_lane_flags(filled, unresolved, background=background)
    missing = renderer_module._pack_missing_lane_masks(
        prefill, renderer_module._lane_mask_weights(torch.device("cuda"))
    )
    counts = torch.stack(
        (
            torch.count_nonzero(filled),
            torch.count_nonzero(unresolved),
            torch.count_nonzero(~torch.isfinite(reduced)),
        )
    )
    expected = torch.cat(
        (
            counts.view(torch.uint8),
            missing.view(torch.uint8).reshape(-1),
            reduced.round().clamp(0, 255).to(torch.uint8).reshape(-1),
            repair.reshape(-1),
        )
    ).cpu()

    payload = output_module.triton_fast_band_payload(
        colour, prefill, filled, unresolved, background=background
    )

    assert payload is not None, "Supported Fast CUDA bands must use the fused output kernel"
    assert payload.dtype == torch.uint8 and payload.is_contiguous()
    assert payload.numel() == 24 + height * width * 6
    assert torch.equal(payload.cpu(), expected)


@requires_triton_cuda
def test_fused_fast_output_retains_balanced_float32_addition():
    values = np.zeros((1, 16, 3), dtype=np.float32)
    values[0, :4, 0] = [2**24, 1, -(2**24), 23]
    values[0, :4, 1] = [2**24, -(2**24), 1, 23]
    values[0, :, 2] = 2.5
    colour = torch.from_numpy(values).cuda()
    prefill = torch.ones((1, 1, 16), dtype=torch.bool, device="cuda")
    empty = torch.zeros_like(prefill)

    payload = output_module.triton_fast_band_payload(colour, prefill, empty, empty, background=True)

    assert payload is not None, "Supported float32 bands must retain balanced reduction"
    # Reassociating the first channel to 24 would change the rounded byte from 1 to 2.
    assert payload[26:29].cpu().tolist() == [1, 2, 2]
    assert torch.equal(payload[:24].cpu(), torch.zeros(24, dtype=torch.uint8))


@requires_triton_cuda
@pytest.mark.parametrize("fill", ["none", "background"])
def test_fast_renderer_fuses_band_output_without_torch_intermediates(monkeypatch, fill):
    if renderer_module._native_lane_analyzer() is None:
        pytest.skip("Native lane analysis is unavailable")
    rng = np.random.default_rng(20260831)
    frame = rng.integers(0, 256, size=(7, 31, 3), dtype=np.uint8)
    canonical = rng.random((5, 17), dtype=np.float32)
    settings = renderer_module.StereoRenderSettings(
        stereo_strength=4.25, convergence=0.37, occlusion_fill=fill
    )
    expected = renderer_module.StereoRenderer(device="cpu").render_relative_fast_compact(
        frame, canonical, settings
    )

    def unexpected_intermediate(*_args, **_kwargs):
        raise AssertionError("Fused CUDA output must not materialize Torch output intermediates")

    monkeypatch.setattr(renderer_module, "_downsample_subpixel_band", unexpected_intermediate)
    monkeypatch.setattr(renderer_module, "_fast_repair_lane_flags", unexpected_intermediate)
    renderer = renderer_module.StereoRenderer(
        device="cuda", temporary_budget_bytes=31 * renderer_module.TRITON_SPLAT_BYTES_PER_PIXEL * 2
    )
    actual = renderer.render_relative_fast_compact(frame, canonical, settings)

    for name in (
        "left_image",
        "right_image",
        "left_coverage_count",
        "right_coverage_count",
        "left_repair_bits",
        "right_repair_bits",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for name in ("left_fast_diagnostics", "right_fast_diagnostics"):
        actual_diagnostics = getattr(actual, name)
        expected_diagnostics = getattr(expected, name)
        assert (
            actual_diagnostics.local_filled_lane_count
            == expected_diagnostics.local_filled_lane_count
        )
        assert (
            actual_diagnostics.final_unresolved_lane_count
            == expected_diagnostics.final_unresolved_lane_count
        )
        np.testing.assert_array_equal(
            actual_diagnostics.hole_runs.lane_count_histogram,
            expected_diagnostics.hole_runs.lane_count_histogram,
        )
        np.testing.assert_array_equal(
            actual_diagnostics.hole_runs.touched_pixel_span_histogram,
            expected_diagnostics.hole_runs.touched_pixel_span_histogram,
        )


def _constant_bands(height=2, width=17, device="cuda"):
    colour = torch.full((height, width * 16, 3), 127.5, device=device)
    prefill = torch.ones((height, width, 16), dtype=torch.bool, device=device)
    filled = torch.zeros_like(prefill)
    unresolved = torch.zeros_like(prefill)
    return colour, prefill, filled, unresolved


@requires_triton_cuda
def test_fused_fast_output_rounds_every_uint8_sample_mean():
    totals = np.arange(255 * 16 + 1, dtype=np.int32)
    lanes = np.clip(totals[:, None] - 255 * np.arange(16), 0, 255).astype(np.float32)
    colours = np.repeat(lanes.reshape(1, -1, 1), 3, axis=2)
    colour = torch.from_numpy(colours).cuda()
    prefill = torch.ones((1, totals.size, 16), dtype=torch.bool, device="cuda")
    empty = torch.zeros_like(prefill)

    payload = output_module.triton_fast_band_payload(colour, prefill, empty, empty, background=True)

    assert payload is not None
    image, missing, repair, counts = renderer_module._download_fast_cuda_payload(
        payload, (1, totals.size, 3), (1, totals.size)
    )
    expected = np.repeat(np.rint(totals / 16.0).astype(np.uint8)[:, None], 3, axis=1)
    np.testing.assert_array_equal(image[0], expected)
    assert not missing.any() and not repair.any() and not counts.any()


@requires_triton_cuda
def test_fused_fast_output_preserves_every_lane_bitmap():
    masks = np.arange(65536, dtype=np.uint16)
    bits = (masks[:, None] & (1 << np.arange(16, dtype=np.uint16))) != 0
    colour = torch.full((1, masks.size * 16, 3), 127.5, device="cuda")
    prefill = torch.from_numpy(bits.reshape(1, masks.size, 16)).cuda()
    filled = ~prefill
    holes = torch.zeros_like(filled)

    payload = output_module.triton_fast_band_payload(
        colour, prefill, filled, holes, background=True
    )

    assert payload is not None
    image, missing, repair, counts = renderer_module._download_fast_cuda_payload(
        payload, (1, masks.size, 3), (1, masks.size)
    )
    np.testing.assert_array_equal(missing[0], masks ^ np.uint16(65535))
    np.testing.assert_array_equal(repair[0], np.where(masks == 65535, 0, 4).astype(np.uint8))
    assert np.all(image == 128)
    assert counts.tolist() == [524288, 0, 0]


@requires_triton_cuda
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), np.finfo(np.float32).max]
)
def test_fused_fast_output_counts_nonfinite_reduced_channels(value):
    colour, prefill, filled, unresolved = _constant_bands(height=1, width=129)
    colour[0, -16:, 1] = float(value)
    colour[0, :16, 2] = float(value)
    before = colour.clone()

    payload = output_module.triton_fast_band_payload(
        colour, prefill, filled, unresolved, background=False
    )

    assert payload is not None
    assert payload[:24].view(torch.int64).cpu().tolist() == [0, 0, 2]
    torch.testing.assert_close(colour, before, rtol=0, atol=0, equal_nan=True)
    assert prefill.all() and not filled.any() and not unresolved.any()


@requires_triton_cuda
def test_fused_fast_counts_reduce_as_int64_without_float_rounding():
    host_counts = np.arange(35 * 3, dtype=np.int32).reshape(35, 3) + np.int32(2**30)
    counts = torch.from_numpy(host_counts).cuda()
    packet = torch.full((41,), 0xA5, dtype=torch.uint8, device="cuda")

    output_module._write_lane_counts_kernel[(3,)](counts, packet, 35, REDUCTION_SIZE=64)

    expected = host_counts.sum(axis=0, dtype=np.int64)
    np.testing.assert_array_equal(packet[:24].view(torch.int64).cpu().numpy(), expected)
    assert packet[24:].cpu().tolist() == [0xA5] * 17
    np.testing.assert_array_equal(counts.cpu().numpy(), host_counts)


def test_fused_fast_output_returns_none_for_cpu():
    assert (
        output_module.triton_fast_band_payload(*_constant_bands(device="cpu"), background=True)
        is None
    )


@requires_triton_cuda
@pytest.mark.parametrize(
    "case",
    [
        "colour_dtype",
        "colour_channels",
        "fine_width",
        "lane_count",
        "empty",
        "prefill_dtype",
        "filled_dtype",
        "unresolved_dtype",
        "mask_shape",
        "host_mask",
        "colour_strides",
        "prefill_strides",
        "filled_strides",
        "unresolved_strides",
    ],
)
def test_fused_fast_output_rejects_unsupported_inputs_before_allocating(monkeypatch, case):
    colour, prefill, filled, unresolved = _constant_bands()
    dtype_cases = {
        "colour_dtype": ("colour", torch.float64),
        "prefill_dtype": ("prefill", torch.uint8),
        "filled_dtype": ("filled", torch.int32),
        "unresolved_dtype": ("unresolved", torch.float32),
    }
    values = dict(colour=colour, prefill=prefill, filled=filled, unresolved=unresolved)
    if case in dtype_cases:
        name, dtype = dtype_cases[case]
        values[name] = values[name].to(dtype)
        colour, prefill, filled, unresolved = values.values()
    elif case == "colour_channels":
        colour = torch.empty((*colour.shape[:2], 4), device="cuda")
    elif case == "fine_width":
        colour = torch.empty((2, colour.shape[1] - 1, 3), device="cuda")
    elif case == "lane_count":
        prefill = torch.ones((2, 17, 15), dtype=torch.bool, device="cuda")
    elif case == "empty":
        colour, prefill, filled, unresolved = _constant_bands(height=0)
    elif case == "mask_shape":
        unresolved = torch.zeros((2, 16, 16), dtype=torch.bool, device="cuda")
    elif case == "host_mask":
        filled = filled.cpu()
    else:
        key = case.removesuffix("_strides")
        values[key] = values[key].repeat_interleave(2, dim=0)[::2]
        colour, prefill, filled, unresolved = (values[key] for key in values)

    def unexpected_allocation(*_args, **_kwargs):
        raise AssertionError("Unsupported inputs must fall back before allocating")

    monkeypatch.setattr(torch, "empty", unexpected_allocation)
    assert (
        output_module.triton_fast_band_payload(colour, prefill, filled, unresolved, background=True)
        is None
    )


@requires_triton_cuda
def test_fused_fast_output_bounds_partial_count_storage(monkeypatch):
    bands = _constant_bands(height=1, width=129)
    monkeypatch.setattr(output_module, "_MAX_COUNT_BLOCKS", 1)

    def unexpected_allocation(*_args, **_kwargs):
        raise AssertionError("Bands beyond the reduction budget must use Torch fallback")

    monkeypatch.setattr(torch, "empty", unexpected_allocation)
    assert output_module.triton_fast_band_payload(*bands, background=True) is None


@requires_triton_cuda
@pytest.mark.parametrize(
    "failed_kernel", ["_fast_band_payload_kernel", "_write_lane_counts_kernel"]
)
def test_fused_fast_output_compile_failure_keeps_other_acceleration(monkeypatch, failed_kernel):
    from src.depth_surge_3d.rendering import fast_fill_triton, fast_splat_triton

    bands = _constant_bands()
    monkeypatch.setattr(output_module, "_DISABLED_CUDA_DEVICES", set())

    class FailingKernel:
        def __getitem__(self, _grid):
            def fail(*_args, **_kwargs):
                raise output_module.triton.errors.TritonError("injected output compile failure")

            return fail

    monkeypatch.setattr(output_module, failed_kernel, FailingKernel())
    assert output_module.triton_fast_band_payload(*bands, background=True) is None
    assert not output_module.triton_output_available(bands[0].device)
    assert fast_fill_triton.triton_fill_available(bands[0].device)
    assert fast_splat_triton.triton_splat_available(bands[0].device)


@requires_triton_cuda
@pytest.mark.parametrize("allocation", ["payload", "partials"])
def test_fused_fast_output_allocation_oom_is_not_swallowed(monkeypatch, allocation):
    bands = _constant_bands()
    original_empty = torch.empty
    failing_dtype = torch.uint8 if allocation == "payload" else torch.int32

    def fail(*args, **kwargs):
        if kwargs.get("dtype") == failing_dtype:
            raise torch.cuda.OutOfMemoryError("injected output OOM")
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fail)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="injected output OOM"):
        output_module.triton_fast_band_payload(*bands, background=True)
    assert output_module.triton_output_available(bands[0].device)


@requires_triton_cuda
@pytest.mark.parametrize("allocation", ["payload", "partials"])
def test_fused_fast_allocation_oom_retries_smaller_bands(monkeypatch, allocation):
    if renderer_module._native_lane_analyzer() is None:
        pytest.skip("Native lane analysis is unavailable")
    rng = np.random.default_rng(20260831)
    frame = rng.integers(0, 256, size=(7, 31, 3), dtype=np.uint8)
    canonical = rng.random((5, 17), dtype=np.float32)
    settings = renderer_module.StereoRenderSettings(stereo_strength=5.0, convergence=0.2)
    expected = renderer_module.StereoRenderer(device="cpu").render_relative_fast_compact(
        frame, canonical, settings
    )
    original_empty = torch.empty
    original_payload = output_module.triton_fast_band_payload
    failing_dtype = torch.uint8 if allocation == "payload" else torch.int32
    injected = False
    in_output = False
    band_heights = []

    def fail_once(*args, **kwargs):
        nonlocal injected
        if in_output and not injected and kwargs.get("dtype") == failing_dtype:
            injected = True
            raise torch.cuda.OutOfMemoryError("injected fused output allocation failure")
        return original_empty(*args, **kwargs)

    def observe_band(colour, *args, **kwargs):
        nonlocal in_output
        band_heights.append(colour.shape[0])
        in_output = True
        try:
            return original_payload(colour, *args, **kwargs)
        finally:
            in_output = False

    monkeypatch.setattr(torch, "empty", fail_once)
    monkeypatch.setattr(output_module, "triton_fast_band_payload", observe_band)
    renderer = renderer_module.StereoRenderer(
        device="cuda", temporary_budget_bytes=31 * renderer_module.TRITON_SPLAT_BYTES_PER_PIXEL * 2
    )
    actual = renderer.render_relative_fast_compact(frame, canonical, settings)

    assert injected and band_heights == [2] + [1] * 14
    for name in (
        "left_image",
        "right_image",
        "left_coverage_count",
        "right_coverage_count",
        "left_repair_bits",
        "right_repair_bits",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for name in ("left_fast_diagnostics", "right_fast_diagnostics"):
        actual_diagnostics = getattr(actual, name)
        expected_diagnostics = getattr(expected, name)
        assert (
            actual_diagnostics.local_filled_lane_count
            == expected_diagnostics.local_filled_lane_count
        )
        assert (
            actual_diagnostics.final_unresolved_lane_count
            == expected_diagnostics.final_unresolved_lane_count
        )
        np.testing.assert_array_equal(
            actual_diagnostics.hole_runs.lane_count_histogram,
            expected_diagnostics.hole_runs.lane_count_histogram,
        )
        np.testing.assert_array_equal(
            actual_diagnostics.hole_runs.touched_pixel_span_histogram,
            expected_diagnostics.hole_runs.touched_pixel_span_histogram,
        )
