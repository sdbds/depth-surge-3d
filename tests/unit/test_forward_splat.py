"""CPU reference tests for depth-aware horizontal forward splatting."""

from __future__ import annotations

import pytest
import torch

from src.depth_surge_3d.rendering.forward_splat import forward_splat_band


def test_constant_disparity_uses_correct_left_and_right_sign() -> None:
    image = torch.zeros((1, 7, 3), dtype=torch.float32)
    image[0, 3] = 1.0
    disparity = torch.full((1, 7), 2.0, dtype=torch.float32)

    left = forward_splat_band(image, disparity, eye_sign=1)
    right = forward_splat_band(image, disparity, eye_sign=-1)

    assert int(torch.argmax(left.image[0, :, 0])) == 4
    assert int(torch.argmax(right.image[0, :, 0])) == 2


def test_fractional_projection_preserves_bilinear_coverage() -> None:
    image = torch.zeros((1, 5, 3), dtype=torch.float32)
    image[0, 2] = 1.0
    disparity = torch.ones((1, 5), dtype=torch.float32)

    result = forward_splat_band(image, disparity, eye_sign=1)

    assert result.image[0, 2, 0].item() == pytest.approx(0.5)
    assert result.image[0, 3, 0].item() == pytest.approx(0.5)


def test_out_of_frame_contributions_become_explicit_holes() -> None:
    image = torch.ones((1, 4, 3), dtype=torch.float32)
    disparity = torch.full((1, 4), 100.0, dtype=torch.float32)

    result = forward_splat_band(image, disparity, eye_sign=1)

    assert not result.valid_mask.any()
    assert torch.count_nonzero(result.image) == 0


@pytest.mark.parametrize(
    ("eye_sign", "near_x", "far_x"),
    [(1, 0, 2), (-1, 2, 0)],
)
def test_both_eyes_use_total_disparity_as_the_near_z_key(
    eye_sign: int, near_x: int, far_x: int
) -> None:
    image = torch.zeros((1, 3, 3), dtype=torch.float32)
    image[0, near_x] = torch.tensor([1.0, 0.0, 0.0])
    image[0, far_x] = torch.tensor([0.0, 0.0, 1.0])
    disparity = torch.full((1, 3), 100.0, dtype=torch.float32)
    disparity[0, near_x] = 4.0
    disparity[0, far_x] = 0.0

    result = forward_splat_band(image, disparity, eye_sign=eye_sign)

    target_x = 2 if eye_sign == 1 else 0
    assert result.image[0, target_x, 0].item() == pytest.approx(1.0)
    assert result.image[0, target_x, 2].item() == pytest.approx(0.0)


def test_low_weight_foreground_tail_antialiases_instead_of_owning_z_buffer() -> None:
    image = torch.zeros((1, 12, 3), dtype=torch.float32)
    image[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    image[0, 11] = torch.tensor([0.0, 0.0, 1.0])
    disparity = torch.full((1, 12), 100.0, dtype=torch.float32)
    disparity[0, 0] = 20.02
    disparity[0, 11] = 0.0

    result = forward_splat_band(image, disparity, eye_sign=1)

    assert result.valid_mask[0, 11]
    assert result.image[0, 11, 0].item() == pytest.approx(0.01 / 1.01, abs=1e-4)
    assert result.image[0, 11, 2].item() == pytest.approx(1.0 / 1.01, abs=1e-4)


def test_target_without_primary_voter_falls_back_to_all_contributions() -> None:
    image = torch.zeros((1, 2, 3), dtype=torch.float32)
    image[0, 0] = torch.tensor([0.2, 0.4, 0.6])
    disparity = torch.tensor([[1.2, 100.0]], dtype=torch.float32)

    result = forward_splat_band(image, disparity, eye_sign=1)

    assert result.valid_mask[0, 0]
    torch.testing.assert_close(result.image[0, 0], image[0, 0])


def test_twenty_pixel_full_range_ramp_has_no_internal_holes() -> None:
    image = torch.ones((1, 20, 3), dtype=torch.float32)
    disparity = torch.linspace(0.0, 36.48, 20, dtype=torch.float32).unsqueeze(0)

    result = forward_splat_band(image, disparity, eye_sign=1)

    assert result.valid_mask[0].all()


def test_batched_input_preserves_batch_shape_and_independence() -> None:
    image = torch.zeros((2, 1, 5, 3), dtype=torch.float32)
    image[0, 0, 2] = 1.0
    image[1, 0, 3] = 2.0
    disparity = torch.zeros((2, 1, 5), dtype=torch.float32)

    result = forward_splat_band(image, disparity, eye_sign=1)

    assert result.image.shape == image.shape
    assert result.valid_mask.shape == disparity.shape
    torch.testing.assert_close(result.image, image)


def test_nonfinite_disparity_is_rejected() -> None:
    image = torch.zeros((1, 2, 3), dtype=torch.float32)
    disparity = torch.tensor([[0.0, float("nan")]], dtype=torch.float32)

    with pytest.raises(ValueError, match="finite"):
        forward_splat_band(image, disparity, eye_sign=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_matches_cpu_reference() -> None:
    generator = torch.Generator().manual_seed(7)
    image = torch.rand((2, 5, 11, 3), generator=generator)
    disparity = torch.rand((2, 5, 11), generator=generator) * 6.0 - 3.0

    cpu = forward_splat_band(image, disparity, eye_sign=-1)
    cuda = forward_splat_band(image.cuda(), disparity.cuda(), eye_sign=-1)

    torch.testing.assert_close(cuda.image.cpu(), cpu.image, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cuda.projected_disparity.cpu(), cpu.projected_disparity)
    assert torch.equal(cuda.valid_mask.cpu(), cpu.valid_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_collisions_are_byte_deterministic() -> None:
    generator = torch.Generator().manual_seed(11)
    image = torch.rand((4, 64, 129, 3), generator=generator).cuda()
    disparity = (torch.rand((4, 64, 129), generator=generator) * 20.0 - 10.0).cuda()

    first = forward_splat_band(image, disparity, eye_sign=1)
    second = forward_splat_band(image, disparity, eye_sign=1)

    assert torch.equal(first.image, second.image)
    assert torch.equal(first.projected_disparity, second.projected_disparity)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_peak_memory_fits_splat_budget() -> None:
    height, width = 128, 1024
    forward_splat_band(
        torch.zeros((1, 8, 3), device="cuda"),
        torch.zeros((1, 8), device="cuda"),
        eye_sign=1,
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    initial_bytes = torch.cuda.memory_allocated()

    image = torch.rand((height, width, 3), device="cuda")
    disparity = torch.rand((height, width), device="cuda")
    disparity.mul_(20.0).sub_(10.0)
    result = forward_splat_band(image, disparity, eye_sign=1)
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated() - initial_bytes

    assert result.image.shape == image.shape
    assert peak_bytes <= height * width * 192
