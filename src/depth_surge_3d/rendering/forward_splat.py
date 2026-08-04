"""Depth-aware horizontal bilinear forward splatting with explicit coverage."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SplatBandResult:
    """One rendered row band and the geometry needed by background filling."""

    image: torch.Tensor
    valid_mask: torch.Tensor
    projected_disparity: torch.Tensor
    accumulated_weight: torch.Tensor


def _as_batched(
    image: torch.Tensor, disparity: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    unbatched = image.ndim == 3
    if unbatched:
        image = image.unsqueeze(0)
        disparity = disparity.unsqueeze(0)
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError("Image must have shape [H,W,3] or [N,H,W,3]")
    if disparity.ndim != 3 or disparity.shape != image.shape[:-1]:
        raise ValueError("Disparity must match image shape [N,H,W]")
    if image.device != disparity.device:
        raise ValueError("Image and disparity must use the same device")
    return image.to(dtype=torch.float32), disparity.to(dtype=torch.float32), unbatched


def _scatter_amax(
    indexes: torch.Tensor,
    values: torch.Tensor,
    include: torch.Tensor,
    target_count: int,
) -> torch.Tensor:
    """Take a finite max; unlike floating-point addition, order cannot change it."""

    output = torch.full(
        (target_count,),
        -torch.inf,
        dtype=torch.float32,
        device=values.device,
    )
    candidates = values.masked_fill(~include, -torch.inf)
    output.scatter_reduce_(
        0,
        indexes,
        candidates,
        reduce="amax",
        include_self=True,
    )
    return output


def _write_segment_sum(
    output: torch.Tensor,
    sorted_values: torch.Tensor,
    unique_indexes: torch.Tensor,
    counts: torch.Tensor,
    valid_segments: torch.Tensor,
) -> None:
    """Write a fixed-order scalar segment sum into a preallocated target."""

    reduced = torch.segment_reduce(sorted_values, "sum", lengths=counts)
    output[unique_indexes[valid_segments]] = reduced[valid_segments]


def forward_splat_band(
    image: torch.Tensor,
    disparity: torch.Tensor,
    eye_sign: int,
) -> SplatBandResult:
    """Project complete rows horizontally using total disparity as a shared z-key."""

    if eye_sign not in {-1, 1}:
        raise ValueError("eye_sign must be +1 for left or -1 for right")
    source, total_disparity, unbatched = _as_batched(image, disparity)
    if not torch.isfinite(source).all() or not torch.isfinite(total_disparity).all():
        raise ValueError("Image and disparity tensors must be finite")

    batch_size, height, width, _channels = source.shape
    device = source.device
    source_x = torch.arange(width, dtype=torch.float32, device=device).view(1, 1, width)
    target_x = source_x + total_disparity * (float(eye_sign) * 0.5)
    floor_x = torch.floor(target_x).to(dtype=torch.int64)
    fraction = target_x - floor_x.to(dtype=torch.float32)

    target_columns = torch.stack((floor_x, floor_x + 1), dim=-1)
    weights = torch.stack((1.0 - fraction, fraction), dim=-1)
    batch_offsets = (
        torch.arange(batch_size, dtype=torch.int64, device=device).view(batch_size, 1, 1)
        * height
        * width
    )
    row_offsets = torch.arange(height, dtype=torch.int64, device=device).view(1, height, 1) * width
    linear_indexes = batch_offsets.unsqueeze(-1) + row_offsets.unsqueeze(-1) + target_columns
    in_bounds = (target_columns >= 0) & (target_columns < width)
    contributes = in_bounds & (weights > 0.0)
    safe_indexes = linear_indexes.clamp(0, batch_size * height * width - 1).reshape(-1)

    flat_weights = weights.expand(batch_size, height, width, 2).reshape(-1)
    flat_disparity = total_disparity.unsqueeze(-1).expand(-1, -1, -1, 2).reshape(-1)
    flat_contributes = contributes.expand(batch_size, height, width, 2).reshape(-1)
    target_count = batch_size * height * width

    all_z = _scatter_amax(
        safe_indexes,
        flat_disparity,
        flat_contributes,
        target_count,
    )
    primary_voters = flat_contributes & (flat_weights >= 0.5)
    primary_z = _scatter_amax(
        safe_indexes,
        flat_disparity,
        primary_voters,
        target_count,
    )
    winning_z = torch.where(torch.isfinite(primary_z), primary_z, all_z)
    visible = flat_contributes & (flat_disparity >= winning_z[safe_indexes] - 0.25)
    visible_weights = torch.where(visible, flat_weights, 0.0)
    del (
        source_x,
        target_x,
        floor_x,
        fraction,
        target_columns,
        weights,
        batch_offsets,
        row_offsets,
        linear_indexes,
        in_bounds,
        contributes,
        flat_weights,
        flat_contributes,
        all_z,
        primary_voters,
        primary_z,
        winning_z,
    )

    # Reuse one stable segment layout and reduce scalar channels separately.
    # This keeps deterministic sums inside the renderer's row-band budget.
    safe_indexes.masked_fill_(~visible, target_count)
    del visible
    order = torch.argsort(safe_indexes, stable=True)
    sorted_indexes = safe_indexes[order]
    unique_indexes, counts = torch.unique_consecutive(
        sorted_indexes,
        return_counts=True,
    )
    valid_segments = unique_indexes < target_count
    del sorted_indexes, safe_indexes

    sorted_weights = visible_weights[order]
    del visible_weights
    accumulated_weight = torch.zeros(target_count, dtype=torch.float32, device=device)
    _write_segment_sum(
        accumulated_weight,
        sorted_weights,
        unique_indexes,
        counts,
        valid_segments,
    )

    sorted_values = flat_disparity[order]
    sorted_values.mul_(sorted_weights)
    accumulated_disparity = torch.zeros(
        target_count,
        dtype=torch.float32,
        device=device,
    )
    _write_segment_sum(
        accumulated_disparity,
        sorted_values,
        unique_indexes,
        counts,
        valid_segments,
    )
    del flat_disparity, sorted_values, total_disparity

    accumulated_color = torch.zeros(
        (target_count, 3),
        dtype=torch.float32,
        device=device,
    )
    for channel in range(3):
        contribution_channel = source[..., channel].unsqueeze(-1).expand(-1, -1, -1, 2).reshape(-1)
        sorted_values = contribution_channel[order]
        sorted_values.mul_(sorted_weights)
        _write_segment_sum(
            accumulated_color[:, channel],
            sorted_values,
            unique_indexes,
            counts,
            valid_segments,
        )
        del contribution_channel, sorted_values

    del order, unique_indexes, counts, valid_segments, sorted_weights

    valid = accumulated_weight > 0.0
    safe_denominator = torch.where(valid, accumulated_weight, 1.0)
    rendered = accumulated_color / safe_denominator.unsqueeze(1)
    projected = accumulated_disparity / safe_denominator
    rendered = torch.where(valid.unsqueeze(1), rendered, 0.0)
    projected = torch.where(valid, projected, 0.0)

    rendered = rendered.reshape(batch_size, height, width, 3)
    valid = valid.reshape(batch_size, height, width)
    projected = projected.reshape(batch_size, height, width)
    accumulated_weight = accumulated_weight.reshape(batch_size, height, width)
    if unbatched:
        rendered = rendered.squeeze(0)
        valid = valid.squeeze(0)
        projected = projected.squeeze(0)
        accumulated_weight = accumulated_weight.squeeze(0)
    return SplatBandResult(
        image=rendered,
        valid_mask=valid,
        projected_disparity=projected,
        accumulated_weight=accumulated_weight,
    )
