"""Independent scalar references for horizontal stereo coverage tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


SAMPLES = 16


@dataclass(frozen=True)
class ReferenceEyeResult:
    image: np.ndarray
    valid_mask: np.ndarray
    hole_mask: np.ndarray
    fine_colour: np.ndarray
    fine_valid: np.ndarray


def _source_offset(
    value: np.float32,
    *,
    width: int,
    strength: float,
    convergence: float,
    eye_sign: int,
) -> int:
    scale = np.float64(width) * np.float64(strength) / np.float64(200.0)
    base = (np.float64(value) - np.float64(convergence)) * scale
    q = base if eye_sign == 1 else -base
    fine_shift = q * np.float64(SAMPLES)
    return math.ceil(float(fine_shift - np.float64(0.5)))


def _choose_fill_source(
    left: int,
    right: int,
    column: int,
    depth: np.ndarray,
) -> int:
    if left < 0:
        return right
    if right >= depth.shape[0]:
        return left
    if depth[left] < depth[right]:
        return left
    if depth[right] < depth[left]:
        return right
    return left if column - left <= right - column else right


def _fill_reference_row(
    colour: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    max_gap_samples: int,
) -> None:
    width = valid.shape[0]
    column = 0
    while column < width:
        if valid[column]:
            column += 1
            continue
        start = column
        while column < width and not valid[column]:
            column += 1
        end = column
        if end - start > max_gap_samples:
            continue
        left = start - 1
        right = end
        if left < 0 and right >= width:
            continue
        if left >= 0 and right < width or start == 0 or end == width:
            for target in range(start, end):
                source = _choose_fill_source(left, right, target, depth)
                colour[target] = colour[source]
                depth[target] = depth[source]
                valid[target] = True


def _convert_reference(colour: np.ndarray, dtype: np.dtype) -> np.ndarray:
    lanes = colour.reshape(colour.shape[0], colour.shape[1] // SAMPLES, SAMPLES, 3)
    pairs = lanes[:, :, 0::2] + lanes[:, :, 1::2]
    quads = pairs[:, :, 0::2] + pairs[:, :, 1::2]
    octets = quads[:, :, 0::2] + quads[:, :, 1::2]
    total = octets[:, :, 0] + octets[:, :, 1]
    averaged = total * np.float32(0.0625)
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        averaged = np.clip(np.rint(averaged), limits.min, limits.max)
    return averaged.astype(dtype, copy=False)


def render_discrete_reference(
    frame: np.ndarray,
    canonical: np.ndarray,
    *,
    strength: float,
    convergence: float,
    eye_sign: int,
    occlusion_fill: str,
) -> ReferenceEyeResult:
    """Render one eye with scalar loops and direct depth/source comparisons."""

    source = np.asarray(frame)
    depth_source = np.asarray(canonical, dtype=np.float32)
    height, width = source.shape[:2]
    fine_width = width * SAMPLES
    fine_colour = np.zeros((height, fine_width, 3), dtype=np.float32)
    fine_depth = np.full((height, fine_width), -np.inf, dtype=np.float32)
    fine_source = np.full((height, fine_width), -1, dtype=np.int64)
    prefill_valid = np.zeros((height, fine_width), dtype=np.bool_)

    for row in range(height):
        for source_column in range(width):
            source_index = row * width + source_column
            source_depth = depth_source[row, source_column]
            offset = _source_offset(
                source_depth,
                width=width,
                strength=strength,
                convergence=convergence,
                eye_sign=eye_sign,
            )
            first = source_column * SAMPLES + offset
            for target in range(first, first + SAMPLES):
                if not 0 <= target < fine_width:
                    continue
                old_depth = fine_depth[row, target]
                old_source = fine_source[row, target]
                wins = source_depth > old_depth or (
                    source_depth == old_depth and source_index < old_source
                )
                if wins:
                    fine_colour[row, target] = source[row, source_column]
                    fine_depth[row, target] = source_depth
                    fine_source[row, target] = source_index
                    prefill_valid[row, target] = True

    postfill_valid = prefill_valid.copy()
    if occlusion_fill == "background":
        max_gap = SAMPLES * (math.ceil(width * strength / 200.0) + 2)
        for row in range(height):
            _fill_reference_row(
                fine_colour[row],
                fine_depth[row],
                postfill_valid[row],
                max_gap,
            )
    elif occlusion_fill != "none":
        raise ValueError("Unknown occlusion fill mode")

    fine_colour[~postfill_valid] = 0.0
    image = _convert_reference(fine_colour, source.dtype)
    valid_mask = prefill_valid.reshape(height, width, SAMPLES).any(axis=2)
    hole_mask = ~postfill_valid.reshape(height, width, SAMPLES).any(axis=2)
    return ReferenceEyeResult(
        image=image,
        valid_mask=valid_mask,
        hole_mask=hole_mask,
        fine_colour=fine_colour,
        fine_valid=postfill_valid,
    )


def _target_contributors(
    canonical_row: np.ndarray,
    *,
    strength: float,
    convergence: float,
    eye_sign: int,
) -> list[list[tuple[float, float, np.float32, int]]]:
    width = canonical_row.shape[0]
    result: list[list[tuple[float, float, np.float32, int]]] = [[] for _ in range(width)]
    scale = np.float64(width) * np.float64(strength) / np.float64(200.0)
    for source_column, source_depth in enumerate(canonical_row):
        base = (np.float64(source_depth) - np.float64(convergence)) * scale
        q = base if eye_sign == 1 else -base
        start = float(np.float64(source_column) + q - np.float64(0.5))
        end = start + 1.0
        first_pixel = math.floor(start + 0.5)
        last_pixel = math.ceil(end + 0.5) - 1
        for target in range(max(0, first_pixel), min(width - 1, last_pixel) + 1):
            result[target].append((start, end, source_depth, source_column))
    return result


def _integrate_target(
    contributors: list[tuple[float, float, np.float32, int]],
    frame_row: np.ndarray,
    target: int,
) -> np.ndarray:
    pixel_start = target - 0.5
    pixel_end = target + 0.5
    endpoints = {pixel_start, pixel_end}
    for start, end, _depth, _source in contributors:
        endpoints.add(max(pixel_start, start))
        endpoints.add(min(pixel_end, end))
    ordered = sorted(value for value in endpoints if pixel_start <= value <= pixel_end)
    colour = np.zeros(3, dtype=np.float64)
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        midpoint = (start + end) * 0.5
        active = [item for item in contributors if item[0] <= midpoint < item[1]]
        if not active:
            continue
        winner = min(active, key=lambda item: (-float(item[2]), item[3]))
        colour += np.asarray(frame_row[winner[3]], dtype=np.float64) * (end - start)
    return colour


def render_continuous_reference(
    frame: np.ndarray,
    canonical: np.ndarray,
    *,
    strength: float,
    convergence: float,
    eye_sign: int,
) -> np.ndarray:
    """Integrate exact projected interval visibility with uncovered length black."""

    source = np.asarray(frame)
    depth_source = np.asarray(canonical, dtype=np.float32)
    height, width = source.shape[:2]
    output = np.zeros(source.shape, dtype=np.float64)
    for row in range(height):
        targets = _target_contributors(
            depth_source[row],
            strength=strength,
            convergence=convergence,
            eye_sign=eye_sign,
        )
        for target, contributors in enumerate(targets):
            output[row, target] = _integrate_target(contributors, source[row], target)
    if np.issubdtype(source.dtype, np.integer):
        limits = np.iinfo(source.dtype)
        output = np.clip(np.rint(output), limits.min, limits.max)
    return output.astype(source.dtype, copy=False)
