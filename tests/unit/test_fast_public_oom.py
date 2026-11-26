"""Public Fast retry must not retain the failed attempt's CUDA tensors."""

import gc
import weakref

import numpy as np
import pytest
import torch

from src.depth_surge_3d.rendering import stereo_renderer as rendering


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("failed_band", [1, 3])
def test_public_cuda_oom_drops_failed_band_tensors_before_releasing(monkeypatch, failed_band):
    if not rendering._triton_splat_available(torch.device("cuda")):
        pytest.skip("Triton Fast path is required")
    gc.collect()
    initial_cuda_bytes = torch.cuda.memory_allocated()
    rng = np.random.default_rng(20260831)
    frame = rng.integers(0, 256, (37, 73, 3), dtype=np.uint8)
    depth = rng.random((37, 73), dtype=np.float32)
    settings = rendering.StereoRenderSettings(stereo_strength=4.25, convergence=0.37)
    expected = rendering.StereoRenderer(device="cpu").render(frame, depth, settings)
    monkeypatch.setattr(rendering, "GPU_TEMP_BUDGET", 73 * 1088 * 2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (8 * 1024**3, 16 * 1024**3))
    renderer = rendering.StereoRenderer(device="cuda")
    original_splat = rendering.forward_splat_band
    original_render = renderer._render_with_band_height
    original_release = renderer._release_after_oom
    failed_refs, attempts, releases = [], [], []
    calls = 0

    def fail_after_real_splat(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original_splat(*args, **kwargs)
        if calls == failed_band:
            failed_refs.extend(
                weakref.ref(value)
                for value in (
                    *args[:3],
                    kwargs["source_valid"],
                    result.colour,
                    result.disparity,
                    result.valid,
                )
            )
            raise torch.cuda.OutOfMemoryError("injected after real CUDA band allocations")
        return result

    def observe_attempt(*args, **kwargs):
        attempts.append(args[4])
        return original_render(*args, **kwargs)

    def release_without_failed_tensors():
        gc.collect()
        assert len(failed_refs) == 7
        assert all(value() is None for value in failed_refs), "failed traceback retains CUDA bands"
        releases.append(True)
        original_release()

    monkeypatch.setattr(rendering, "forward_splat_band", fail_after_real_splat)
    monkeypatch.setattr(renderer, "_render_with_band_height", observe_attempt)
    monkeypatch.setattr(renderer, "_release_after_oom", release_without_failed_tensors)

    actual = renderer.render(frame, depth, settings)

    assert attempts == [10, 5] and releases == [True]
    assert calls == failed_band + 16
    for field in rendering.StereoRenderResult.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= initial_cuda_bytes
