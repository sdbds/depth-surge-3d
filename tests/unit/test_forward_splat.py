"""Exact tests for the sixteen-sample packed-key horizontal z-buffer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.depth_surge_3d.rendering.forward_splat import (
    HORIZONTAL_SUBPIXELS,
    _decode_source_index,
    _pack_depth_source_key,
    forward_splat_band,
)


def _offsets(values: list[list[int]], *, device: str = "cpu") -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=device)


def test_zero_offsets_tile_each_source_pixel_into_sixteen_fine_samples() -> None:
    image = torch.tensor([[[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]]])
    canonical = torch.full((1, 3), 0.5, dtype=torch.float32)

    result = forward_splat_band(image, canonical, _offsets([[0, 0, 0]]))

    assert HORIZONTAL_SUBPIXELS == 16
    expected = image.repeat_interleave(HORIZONTAL_SUBPIXELS, dim=1)
    torch.testing.assert_close(result.colour, expected)
    torch.testing.assert_close(
        result.disparity,
        canonical.repeat_interleave(HORIZONTAL_SUBPIXELS, dim=1),
    )
    assert result.valid.all()


def test_positive_offset_discards_only_out_of_frame_fine_samples() -> None:
    image = torch.tensor([[[7.0, 8.0, 9.0]]])
    canonical = torch.tensor([[0.75]], dtype=torch.float32)

    result = forward_splat_band(image, canonical, _offsets([[1]]))

    assert result.valid.tolist() == [[False] + [True] * (HORIZONTAL_SUBPIXELS - 1)]
    torch.testing.assert_close(
        result.colour[0, 1:],
        image[0, 0].expand(HORIZONTAL_SUBPIXELS - 1, 3),
    )
    assert torch.count_nonzero(result.colour[0, 0]) == 0
    assert torch.isneginf(result.disparity[0, 0])


def test_nearer_depth_wins_every_colliding_fine_sample() -> None:
    image = torch.tensor([[[255.0, 0.0, 0.0], [0.0, 0.0, 255.0]]])
    canonical = torch.tensor([[0.9, 0.1]], dtype=torch.float32)

    full = HORIZONTAL_SUBPIXELS
    result = forward_splat_band(image, canonical, _offsets([[full, 0]]))

    assert not result.valid[0, :full].any()
    assert result.valid[0, full:].all()
    torch.testing.assert_close(result.colour[0, full:], image[0, 0].expand(full, 3))
    torch.testing.assert_close(result.disparity[0, full:], torch.full((full,), 0.9))


def test_equal_depth_collision_uses_lowest_full_frame_source_index() -> None:
    image = torch.tensor([[[1.0, 2.0, 3.0], [9.0, 8.0, 7.0]]])
    canonical = torch.full((1, 2), 0.5, dtype=torch.float32)

    result = forward_splat_band(
        image,
        canonical,
        _offsets([[HORIZONTAL_SUBPIXELS, 0]]),
        source_index_offset=10,
    )

    torch.testing.assert_close(
        result.colour[0, HORIZONTAL_SUBPIXELS:],
        image[0, 0].expand(HORIZONTAL_SUBPIXELS, 3),
    )


def test_three_colliding_layers_use_one_uniform_depth_rule() -> None:
    image = torch.tensor([[[90.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 10.0]]])
    canonical = torch.tensor([[0.9, 0.5, 0.1]], dtype=torch.float32)

    full = HORIZONTAL_SUBPIXELS
    result = forward_splat_band(image, canonical, _offsets([[full, 0, -full]]))

    torch.testing.assert_close(result.colour[0, full : 2 * full], image[0, 0].expand(full, 3))


def test_complementary_and_overlapping_half_coverage_remain_distinguishable() -> None:
    image = torch.tensor([[[100.0, 0.0, 0.0], [0.0, 0.0, 100.0]]])
    canonical = torch.tensor([[0.9, 0.1]], dtype=torch.float32)

    full = HORIZONTAL_SUBPIXELS
    half = full // 2
    complementary = forward_splat_band(image, canonical, _offsets([[-half, -half]]))
    overlapping = forward_splat_band(image, canonical, _offsets([[-half, -(full + half)]]))

    complementary_pixel = complementary.colour[0, :full].sum(dim=0) / full
    overlapping_pixel = overlapping.colour[0, :full].sum(dim=0) / full
    torch.testing.assert_close(complementary_pixel, torch.tensor([50.0, 0.0, 50.0]))
    torch.testing.assert_close(overlapping_pixel, torch.tensor([50.0, 0.0, 0.0]))


def test_fully_covering_near_surface_wins_without_depth_epsilon() -> None:
    image = torch.tensor([[[200.0, 0.0, 0.0], [0.0, 0.0, 200.0]]])
    canonical = torch.tensor([[0.5, 0.49]], dtype=torch.float32)

    full = HORIZONTAL_SUBPIXELS
    result = forward_splat_band(image, canonical, _offsets([[full, 0]]))

    torch.testing.assert_close(result.colour[0, full:], image[0, 0].expand(full, 3))


def test_packed_key_orders_depth_then_inverted_source_index() -> None:
    next_positive = np.nextafter(np.float32(0.0), np.float32(1.0))
    depth = torch.tensor(
        [0.0, -0.0, next_positive, 0.5, 0.5, 1.0],
        dtype=torch.float32,
    )
    source = torch.tensor([7, 7, 7, 9, 2, 7], dtype=torch.int64)

    keys = _pack_depth_source_key(depth, source)

    assert keys[0] == keys[1]
    assert keys[2] > keys[1]
    assert keys[4] > keys[3]
    assert keys[5] > keys[4]
    assert (keys >= 0).all()
    assert _decode_source_index(keys[3:5]).tolist() == [9, 2]


def test_rank_four_batch_input_is_rejected_explicitly() -> None:
    image = torch.zeros((2, 1, 3, 3), dtype=torch.float32)
    canonical = torch.zeros((2, 1, 3), dtype=torch.float32)
    offsets = torch.zeros((2, 1, 3), dtype=torch.int32)

    with pytest.raises(ValueError, match="unbatched"):
        forward_splat_band(image, canonical, offsets)


@pytest.mark.parametrize(
    ("image", "canonical", "offsets", "message"),
    [
        (
            torch.zeros((1, 2, 4)),
            torch.zeros((1, 2)),
            torch.zeros((1, 2), dtype=torch.int32),
            "three channels",
        ),
        (
            torch.zeros((1, 2, 3)),
            torch.zeros((1, 3)),
            torch.zeros((1, 2), dtype=torch.int32),
            "match",
        ),
        (
            torch.zeros((1, 2, 3)),
            torch.zeros((1, 2)),
            torch.zeros((1, 2), dtype=torch.int64),
            "int32",
        ),
    ],
)
def test_invalid_shapes_and_offset_dtype_are_rejected(
    image: torch.Tensor,
    canonical: torch.Tensor,
    offsets: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        forward_splat_band(image, canonical, offsets)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -0.1, 1.1])
def test_nonfinite_or_out_of_range_canonical_is_rejected(bad_value: float) -> None:
    image = torch.zeros((1, 2, 3), dtype=torch.float32)
    canonical = torch.tensor([[0.0, bad_value]], dtype=torch.float32)

    with pytest.raises(ValueError, match="Canonical"):
        forward_splat_band(image, canonical, _offsets([[0, 0]]))


def test_source_index_must_fit_in_low_32_bits() -> None:
    image = torch.zeros((1, 2, 3), dtype=torch.float32)
    canonical = torch.zeros((1, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="32 bits"):
        forward_splat_band(
            image,
            canonical,
            _offsets([[0, 0]]),
            source_index_offset=2**32 - 1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_matches_cpu_byte_exactly() -> None:
    generator = torch.Generator().manual_seed(7)
    image = torch.randint(0, 256, (5, 11, 3), generator=generator, dtype=torch.uint8)
    canonical = torch.rand((5, 11), generator=generator, dtype=torch.float32)
    offsets = torch.randint(-5, 6, (5, 11), generator=generator, dtype=torch.int32)

    cpu = forward_splat_band(image, canonical, offsets, source_index_offset=121)
    cuda = forward_splat_band(
        image.cuda(),
        canonical.cuda(),
        offsets.cuda(),
        source_index_offset=121,
    )

    assert torch.equal(cuda.colour.cpu(), cpu.colour)
    assert torch.equal(cuda.disparity.cpu(), cpu.disparity)
    assert torch.equal(cuda.valid.cpu(), cpu.valid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_packed_collisions_are_repeatable() -> None:
    generator = torch.Generator().manual_seed(11)
    image = torch.randint(0, 256, (64, 129, 3), generator=generator, dtype=torch.uint8).cuda()
    canonical = torch.rand((64, 129), generator=generator).cuda()
    offsets = torch.randint(-12, 13, (64, 129), generator=generator, dtype=torch.int32).cuda()

    first = forward_splat_band(image, canonical, offsets)
    second = forward_splat_band(image, canonical, offsets)

    assert torch.equal(first.colour, second.colour)
    assert torch.equal(first.disparity, second.disparity)
    assert torch.equal(first.valid, second.valid)
