"""Deterministic integer RGB geodesic region solver prototype."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator

import numpy as np

from .indexed_heap import IndexedRegionHeap, QualityRegionQueueBudgetError


_NEIGHBOURS = ((-1, 0), (0, -1), (0, 1), (1, 0))
_UINT64_MAX = int(np.iinfo(np.uint64).max)
_NO_REGION = np.uint32(0xFFFFFFFF)


@dataclass(frozen=True)
class LowResolutionRegionResult:
    """Canonical low-resolution labels and their deterministic tie ranks."""

    region_map: np.ndarray
    canonical_keys: np.ndarray
    region_rank_bits: np.ndarray


@dataclass(frozen=True)
class GeodesicSolveResult:
    """Task 0 output including scratch diagnostics used by the gate."""

    region_map: np.ndarray
    distance: np.ndarray
    heap_position: np.ndarray
    max_live_entries: int
    settled_count: int
    queue_insertions: int
    queue_decrease_keys: int
    queue_pops: int
    heap_pixel_bytes: int
    heap_position_bytes: int
    distance_bytes: int


def build_low_resolution_regions(
    one_eye_displacement_px: np.ndarray,
    metric_validity: np.ndarray,
    near_score: np.ndarray,
) -> LowResolutionRegionResult:
    """Build stable four-neighbour regions from final one-eye displacement."""

    _validate_low_resolution_inputs(
        one_eye_displacement_px,
        metric_validity,
        near_score,
    )
    height, width = one_eye_displacement_px.shape
    sample_count = height * width
    if sample_count + 1 > int(np.iinfo(np.uint32).max):
        raise ValueError("low-resolution sample count exceeds uint32 contract")

    parent, canonical_key = _union_connected_low_resolution_samples(
        one_eye_displacement_px,
        metric_validity,
    )
    return _materialize_low_resolution_regions(
        parent,
        canonical_key,
        near_score,
        height,
        width,
    )


def _union_connected_low_resolution_samples(
    one_eye_displacement_px: np.ndarray,
    metric_validity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = one_eye_displacement_px.shape
    parent = np.arange(height * width, dtype=np.uint32)
    canonical_key = np.arange(height * width, dtype=np.uint32)
    for y in range(height):
        for x in range(width):
            pixel = y * width + x
            if y and _low_resolution_samples_connect(
                one_eye_displacement_px,
                metric_validity,
                y,
                x,
                y - 1,
                x,
            ):
                _union_low_resolution(parent, canonical_key, pixel, pixel - width)
            if x and _low_resolution_samples_connect(
                one_eye_displacement_px,
                metric_validity,
                y,
                x,
                y,
                x - 1,
            ):
                _union_low_resolution(parent, canonical_key, pixel, pixel - 1)
    return parent, canonical_key


def _materialize_low_resolution_regions(
    parent: np.ndarray,
    canonical_key: np.ndarray,
    near_score: np.ndarray,
    height: int,
    width: int,
) -> LowResolutionRegionResult:
    sample_count = height * width
    for pixel in range(sample_count):
        parent[pixel] = np.uint32(_find_low_resolution_root(parent, pixel))

    region_count = sum(int(parent[pixel]) == pixel for pixel in range(sample_count))
    region_for_root = np.zeros(sample_count, dtype=np.uint32)
    canonical_keys = np.zeros(region_count + 1, dtype=np.uint32)
    next_region = 0
    for pixel in range(sample_count):
        if int(parent[pixel]) != pixel:
            continue
        next_region += 1
        region_for_root[pixel] = np.uint32(next_region)
        canonical_keys[next_region] = canonical_key[pixel]

    region_map = np.empty((height, width), dtype=np.uint32)
    normalized_score = np.array(near_score, copy=True, order="C")
    normalized_score[normalized_score == np.float32(0.0)] = np.float32(0.0)
    score_bits = normalized_score.view(np.uint32)
    region_rank_bits = np.zeros(region_count + 1, dtype=np.uint32)
    for pixel in range(sample_count):
        region = int(region_for_root[int(parent[pixel])])
        region_map.flat[pixel] = np.uint32(region)
        if int(score_bits.flat[pixel]) > int(region_rank_bits[region]):
            region_rank_bits[region] = score_bits.flat[pixel]

    for values in (region_map, canonical_keys, region_rank_bits):
        values.setflags(write=False)
    return LowResolutionRegionResult(
        region_map=region_map,
        canonical_keys=canonical_keys,
        region_rank_bits=region_rank_bits,
    )


def nearest_label_upsample(
    low_resolution_regions: np.ndarray,
    *,
    render_height: int,
    render_width: int,
) -> np.ndarray:
    """Upsample labels with the frozen unclipped half-pixel coordinate oracle."""

    _validate_region_map(low_resolution_regions)
    destination_height = _validate_positive_dimension("render_height", render_height)
    destination_width = _validate_positive_dimension("render_width", render_width)
    source_height, source_width = low_resolution_regions.shape
    output = np.empty((destination_height, destination_width), dtype=np.uint32)
    for y in range(destination_height):
        source_y = _nearest_source_index(y, source_height, destination_height)
        for x in range(destination_width):
            source_x = _nearest_source_index(x, source_width, destination_width)
            output[y, x] = low_resolution_regions[source_y, source_x]
    output.setflags(write=False)
    return output


def map_low_resolution_centres(
    *,
    geometry_height: int,
    geometry_width: int,
    render_height: int,
    render_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map every source centre to the nearest render pixel in binary64 order."""

    source_height = _validate_positive_dimension("geometry_height", geometry_height)
    source_width = _validate_positive_dimension("geometry_width", geometry_width)
    destination_height = _validate_positive_dimension("render_height", render_height)
    destination_width = _validate_positive_dimension("render_width", render_width)
    rows = _mapped_centres_one_axis(source_height, destination_height)
    columns = _mapped_centres_one_axis(source_width, destination_width)
    rows.setflags(write=False)
    columns.setflags(write=False)
    return rows, columns


def edge_band_radius(
    *,
    geometry_height: int,
    geometry_width: int,
    render_height: int,
    render_width: int,
) -> int:
    """Return the canonical mapped-boundary dilation radius."""

    source_height = _validate_positive_dimension("geometry_height", geometry_height)
    source_width = _validate_positive_dimension("geometry_width", geometry_width)
    destination_height = _validate_positive_dimension("render_height", render_height)
    destination_width = _validate_positive_dimension("render_width", render_width)
    scale_x = float(destination_width) / float(source_width)
    scale_y = float(destination_height) / float(source_height)
    return int(math.ceil(max(scale_x, scale_y))) + 1


def build_four_connected_edge_band(
    mapped_regions: np.ndarray,
    *,
    radius: int,
) -> np.ndarray:
    """Mark every valid two-sided label boundary and cross-dilate it."""

    _validate_region_map(mapped_regions)
    dilation_radius = _validate_nonnegative_integer("radius", radius)
    height, width = mapped_regions.shape
    boundary = np.zeros((height, width), dtype=np.bool_)
    for y in range(height):
        for x in range(width):
            region = mapped_regions[y, x]
            if not _is_region(region):
                continue
            if x + 1 < width and _different_regions(region, mapped_regions[y, x + 1]):
                boundary[y, x] = True
                boundary[y, x + 1] = True
            if y + 1 < height and _different_regions(region, mapped_regions[y + 1, x]):
                boundary[y, x] = True
                boundary[y + 1, x] = True
    for _ in range(dilation_radius):
        boundary = _cross_dilate(boundary)
    boundary.setflags(write=False)
    return boundary


def cross_skeleton(fragment: np.ndarray) -> np.ndarray:
    """Return the exact repeated cross-opening morphological skeleton."""

    _validate_bool_raster("fragment", fragment)
    current = np.array(fragment, copy=True, order="C")
    skeleton = np.zeros(fragment.shape, dtype=np.bool_)
    while np.any(current):
        eroded = _cross_erode(current)
        opened = _cross_dilate(eroded)
        skeleton |= current & ~opened
        current = eroded
    skeleton.setflags(write=False)
    return skeleton


def build_sparse_region_seeds(
    mapped_regions: np.ndarray,
    edge_band: np.ndarray,
    *,
    radius: int,
) -> np.ndarray:
    """Keep immutable outside markers and deterministic sparse band seeds."""

    _validate_region_map(mapped_regions)
    shape = (int(mapped_regions.shape[0]), int(mapped_regions.shape[1]))
    _validate_bool_raster("edge_band", edge_band, shape=shape)
    erosion_radius = _validate_nonnegative_integer("radius", radius)
    seeds = np.array(mapped_regions, copy=True, order="C")
    seeds[edge_band] = np.uint32(0)
    fragment_map, fragment_regions, fragment_bounds, fragment_count = _label_region_fragments(
        mapped_regions, edge_band
    )
    for fragment_id in range(1, fragment_count + 1):
        y0, y1, x0, x1 = (int(value) for value in fragment_bounds[fragment_id])
        fragment = fragment_map[y0 : y1 + 1, x0 : x1 + 1] == np.uint32(fragment_id)
        eroded = fragment
        for _ in range(erosion_radius):
            eroded = _cross_erode(eroded)
        selected = eroded if np.any(eroded) else cross_skeleton(fragment)
        if not np.any(selected):
            first = int(np.flatnonzero(fragment)[0])
            selected = np.zeros(fragment.shape, dtype=np.bool_)
            selected.flat[first] = True
        seed_view = seeds[y0 : y1 + 1, x0 : x1 + 1]
        seed_view[selected] = fragment_regions[fragment_id]
    seeds.setflags(write=False)
    return seeds


def integer_bgr_movement_cost(left: np.ndarray, right: np.ndarray) -> int:
    """Return the frozen uint64 adjacent-pixel traversal cost."""

    _validate_bgr_pixel("left", left)
    _validate_bgr_pixel("right", right)
    maximum = 0
    for channel in range(3):
        maximum = max(maximum, abs(int(left[channel]) - int(right[channel])))
    return 256 + 8 * maximum


def solve_geodesic_regions(
    guide: np.ndarray,
    band_mask: np.ndarray,
    seed_region_map: np.ndarray,
    region_rank_bits: np.ndarray,
) -> GeodesicSolveResult:
    """Assign every band pixel with one global indexed four-neighbour heap."""

    _validate_solver_inputs(guide, band_mask, seed_region_map, region_rank_bits)
    height, width = band_mask.shape
    owner = np.array(seed_region_map, copy=True, order="C")
    distance = np.full((height, width), np.uint64(_UINT64_MAX), dtype=np.uint64)
    band_count = int(np.count_nonzero(band_mask))
    if band_count == 0:
        position = np.full((height, width), -1, dtype=np.int32)
        return _freeze_result(
            owner,
            distance,
            position,
            max_live_entries=0,
            settled_count=0,
            queue_insertions=0,
            queue_decrease_keys=0,
            queue_pops=0,
            heap_pixel_bytes=0,
        )

    heap = IndexedRegionHeap(
        distance,
        owner,
        region_rank_bits,
        capacity=band_count,
    )
    _queue_band_seeds(band_mask, owner, distance, heap)
    _queue_outside_boundary_candidates(guide, band_mask, owner, distance, heap)
    settled_count = _propagate_regions(guide, band_mask, owner, distance, heap)
    if np.any(band_mask & (owner == 0)):
        raise ValueError("edge-band component has no seed")
    return _freeze_result(
        owner,
        distance,
        heap.position,
        heap.max_live_entries,
        settled_count,
        heap.insertions,
        heap.decrease_keys,
        heap.pop_count,
        heap.heap_pixel.nbytes,
    )


def _validate_bgr_pixel(name: str, values: np.ndarray) -> None:
    if not isinstance(values, np.ndarray) or values.dtype != np.uint8 or values.shape != (3,):
        raise TypeError(f"{name} must be one uint8 BGR pixel")


def _validate_low_resolution_inputs(
    one_eye_displacement_px: np.ndarray,
    metric_validity: np.ndarray,
    near_score: np.ndarray,
) -> None:
    if (
        not isinstance(one_eye_displacement_px, np.ndarray)
        or one_eye_displacement_px.dtype != np.float64
        or one_eye_displacement_px.ndim != 2
        or not one_eye_displacement_px.flags.c_contiguous
    ):
        raise TypeError("one_eye_displacement_px must be a C-contiguous float64 raster")
    shape = (
        int(one_eye_displacement_px.shape[0]),
        int(one_eye_displacement_px.shape[1]),
    )
    if not shape[0] or not shape[1]:
        raise ValueError("low-resolution rasters must be non-empty")
    _validate_bool_raster("metric_validity", metric_validity, shape=shape)
    if (
        not isinstance(near_score, np.ndarray)
        or near_score.dtype != np.float32
        or near_score.shape != shape
        or not near_score.flags.c_contiguous
    ):
        raise TypeError("near_score must be a matching C-contiguous float32 raster")


def _validate_region_map(region_map: np.ndarray) -> None:
    if (
        not isinstance(region_map, np.ndarray)
        or region_map.dtype != np.uint32
        or region_map.ndim != 2
        or not region_map.flags.c_contiguous
    ):
        raise TypeError("region map must be a C-contiguous uint32 raster")
    if not region_map.shape[0] or not region_map.shape[1]:
        raise ValueError("region map must be non-empty")


def _validate_bool_raster(
    name: str,
    values: np.ndarray,
    *,
    shape: tuple[int, int] | None = None,
) -> None:
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.bool_
        or values.ndim != 2
        or (shape is not None and values.shape != shape)
        or not values.flags.c_contiguous
    ):
        qualifier = "matching " if shape is not None else ""
        raise TypeError(f"{name} must be a {qualifier}C-contiguous bool raster")


def _validate_positive_dimension(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _low_resolution_samples_connect(
    displacement: np.ndarray,
    validity: np.ndarray,
    y: int,
    x: int,
    neighbour_y: int,
    neighbour_x: int,
) -> bool:
    if bool(validity[y, x]) != bool(validity[neighbour_y, neighbour_x]):
        return False
    difference = abs(float(displacement[y, x]) - float(displacement[neighbour_y, neighbour_x]))
    return difference < 1.0


def _find_low_resolution_root(parent: np.ndarray, pixel: int) -> int:
    root = pixel
    while int(parent[root]) != root:
        root = int(parent[root])
    while int(parent[pixel]) != pixel:
        next_pixel = int(parent[pixel])
        parent[pixel] = np.uint32(root)
        pixel = next_pixel
    return root


def _union_low_resolution(
    parent: np.ndarray,
    canonical_key: np.ndarray,
    left: int,
    right: int,
) -> None:
    left_root = _find_low_resolution_root(parent, left)
    right_root = _find_low_resolution_root(parent, right)
    if left_root == right_root:
        return
    left_key = int(canonical_key[left_root])
    right_key = int(canonical_key[right_root])
    if (right_key, right_root) < (left_key, left_root):
        left_root, right_root = right_root, left_root
        left_key, right_key = right_key, left_key
    parent[right_root] = np.uint32(left_root)
    canonical_key[left_root] = np.uint32(min(left_key, right_key))


def _nearest_source_index(
    destination_index: int,
    source_size: int,
    destination_size: int,
) -> int:
    coordinate = (
        ((float(destination_index) + 0.5) * float(source_size)) / float(destination_size)
    ) - 0.5
    nearest = math.floor(coordinate + 0.5)
    return min(max(nearest, 0), source_size - 1)


def _mapped_centres_one_axis(source_size: int, destination_size: int) -> np.ndarray:
    result = np.empty(source_size, dtype=np.uint32)
    for source_index in range(source_size):
        coordinate = (
            ((float(source_index) + 0.5) * float(destination_size)) / float(source_size)
        ) - 0.5
        nearest = math.floor(coordinate + 0.5)
        result[source_index] = np.uint32(min(max(nearest, 0), destination_size - 1))
    return result


def _is_region(value: np.uint32) -> bool:
    return value != np.uint32(0) and value != _NO_REGION


def _different_regions(left: np.uint32, right: np.uint32) -> bool:
    return _is_region(right) and left != right


def _cross_dilate(mask: np.ndarray) -> np.ndarray:
    output = np.array(mask, copy=True, order="C")
    output[1:, :] |= mask[:-1, :]
    output[:-1, :] |= mask[1:, :]
    output[:, 1:] |= mask[:, :-1]
    output[:, :-1] |= mask[:, 1:]
    return output


def _cross_erode(mask: np.ndarray) -> np.ndarray:
    output = np.array(mask, copy=True, order="C")
    output[0, :] = False
    output[-1, :] = False
    output[:, 0] = False
    output[:, -1] = False
    if mask.shape[0] > 2:
        output[1:-1, :] &= mask[:-2, :] & mask[2:, :]
    if mask.shape[1] > 2:
        output[:, 1:-1] &= mask[:, :-2] & mask[:, 2:]
    return output


def _label_region_fragments(
    region_map: np.ndarray,
    band_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    fragment_map = np.zeros(band_mask.shape, dtype=np.uint32)
    pixel_count = int(np.count_nonzero(band_mask))
    fragment_regions = np.zeros(pixel_count + 1, dtype=np.uint32)
    fragment_bounds = np.zeros((pixel_count + 1, 4), dtype=np.uint32)
    if pixel_count == 0:
        return fragment_map, fragment_regions, fragment_bounds, 0
    queue = np.empty(pixel_count, dtype=np.uint32)
    height, width = band_mask.shape
    fragment_count = 0
    for start in range(band_mask.size):
        region = region_map.flat[start]
        if (
            not bool(band_mask.flat[start])
            or int(fragment_map.flat[start]) != 0
            or not _is_region(region)
        ):
            continue
        fragment_count += 1
        head = 0
        tail = 1
        queue[0] = np.uint32(start)
        fragment_map.flat[start] = np.uint32(fragment_count)
        fragment_regions[fragment_count] = region
        start_y, start_x = divmod(start, width)
        fragment_bounds[fragment_count] = np.asarray(
            [start_y, start_y, start_x, start_x],
            dtype=np.uint32,
        )
        while head < tail:
            pixel = int(queue[head])
            head += 1
            y, x = divmod(pixel, width)
            bounds = fragment_bounds[fragment_count]
            bounds[0] = np.uint32(min(int(bounds[0]), y))
            bounds[1] = np.uint32(max(int(bounds[1]), y))
            bounds[2] = np.uint32(min(int(bounds[2]), x))
            bounds[3] = np.uint32(max(int(bounds[3]), x))
            for ny, nx in _neighbour_coordinates(y, x, height, width):
                neighbour = ny * width + nx
                if not bool(band_mask.flat[neighbour]):
                    continue
                if int(fragment_map.flat[neighbour]) != 0:
                    continue
                if region_map.flat[neighbour] != region:
                    continue
                fragment_map.flat[neighbour] = np.uint32(fragment_count)
                queue[tail] = np.uint32(neighbour)
                tail += 1
    return fragment_map, fragment_regions, fragment_bounds, fragment_count


def _validate_solver_inputs(
    guide: np.ndarray,
    band_mask: np.ndarray,
    seed_region_map: np.ndarray,
    region_rank_bits: np.ndarray,
) -> None:
    if (
        not isinstance(guide, np.ndarray)
        or guide.dtype != np.uint8
        or guide.ndim != 3
        or guide.shape[-1] != 3
        or not guide.flags.c_contiguous
    ):
        raise TypeError("guide must be a C-contiguous three-channel uint8 BGR raster")
    shape = (int(guide.shape[0]), int(guide.shape[1]))
    if (
        not isinstance(band_mask, np.ndarray)
        or band_mask.dtype != np.bool_
        or band_mask.shape != shape
        or not band_mask.flags.c_contiguous
    ):
        raise TypeError("band_mask must be a matching C-contiguous bool raster")
    if (
        not isinstance(seed_region_map, np.ndarray)
        or seed_region_map.dtype != np.uint32
        or seed_region_map.shape != shape
        or not seed_region_map.flags.c_contiguous
    ):
        raise TypeError("seed_region_map must be a matching C-contiguous uint32 raster")
    if (
        not isinstance(region_rank_bits, np.ndarray)
        or region_rank_bits.dtype != np.uint32
        or region_rank_bits.ndim != 1
        or region_rank_bits.size < 2
        or not region_rank_bits.flags.c_contiguous
    ):
        raise TypeError("region_rank_bits must be a C-contiguous uint32 vector")
    maximum_region = int(seed_region_map.max(initial=np.uint32(0)))
    if maximum_region >= region_rank_bits.size:
        raise ValueError("seed region exceeds region_rank_bits")


def _queue_band_seeds(
    band_mask: np.ndarray,
    owner: np.ndarray,
    distance: np.ndarray,
    heap: IndexedRegionHeap,
) -> None:
    for pixel in range(band_mask.size):
        if bool(band_mask.flat[pixel]) and int(owner.flat[pixel]) > 0:
            distance.flat[pixel] = np.uint64(0)
            heap.push_or_decrease(pixel)


def _queue_outside_boundary_candidates(
    guide: np.ndarray,
    band_mask: np.ndarray,
    owner: np.ndarray,
    distance: np.ndarray,
    heap: IndexedRegionHeap,
) -> None:
    height, width = band_mask.shape
    for y in range(height):
        for x in range(width):
            if not band_mask[y, x] or owner[y, x] != 0:
                continue
            for ny, nx in _neighbour_coordinates(y, x, height, width):
                region = int(owner[ny, nx])
                if band_mask[ny, nx] or region == 0:
                    continue
                _relax(
                    y * width + x,
                    integer_bgr_movement_cost(guide[ny, nx], guide[y, x]),
                    region,
                    distance,
                    owner,
                    heap,
                )


def _propagate_regions(
    guide: np.ndarray,
    band_mask: np.ndarray,
    owner: np.ndarray,
    distance: np.ndarray,
    heap: IndexedRegionHeap,
) -> int:
    height, width = band_mask.shape
    settled_count = 0
    while len(heap):
        pixel = heap.pop()
        settled_count += 1
        y, x = divmod(pixel, width)
        source_distance = int(distance[y, x])
        source_owner = int(owner[y, x])
        for ny, nx in _neighbour_coordinates(y, x, height, width):
            if not band_mask[ny, nx]:
                continue
            movement = integer_bgr_movement_cost(guide[y, x], guide[ny, nx])
            if source_distance > _UINT64_MAX - movement:
                raise QualityRegionQueueBudgetError("geodesic distance overflow")
            _relax(
                ny * width + nx,
                source_distance + movement,
                source_owner,
                distance,
                owner,
                heap,
            )
    return settled_count


def _relax(
    pixel: int,
    candidate_distance: int,
    candidate_owner: int,
    distance: np.ndarray,
    owner: np.ndarray,
    heap: IndexedRegionHeap,
) -> None:
    current_owner = int(owner.flat[pixel])
    if not _candidate_is_better(
        candidate_distance,
        candidate_owner,
        int(distance.flat[pixel]),
        current_owner,
        heap.region_rank_bits,
    ):
        return
    distance.flat[pixel] = np.uint64(candidate_distance)
    owner.flat[pixel] = np.uint32(candidate_owner)
    heap.push_or_decrease(pixel)


def _candidate_is_better(
    candidate_distance: int,
    candidate_owner: int,
    current_distance: int,
    current_owner: int,
    ranks: np.ndarray,
) -> bool:
    if current_owner == 0 or candidate_distance != current_distance:
        return current_owner == 0 or candidate_distance < current_distance
    candidate_rank = int(ranks[candidate_owner])
    current_rank = int(ranks[current_owner])
    if candidate_rank != current_rank:
        return candidate_rank > current_rank
    return candidate_owner < current_owner


def _neighbour_coordinates(
    y: int,
    x: int,
    height: int,
    width: int,
) -> Iterator[tuple[int, int]]:
    for dy, dx in _NEIGHBOURS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width:
            yield ny, nx


def _freeze_result(
    owner: np.ndarray,
    distance: np.ndarray,
    position: np.ndarray,
    max_live_entries: int,
    settled_count: int,
    queue_insertions: int,
    queue_decrease_keys: int,
    queue_pops: int,
    heap_pixel_bytes: int,
) -> GeodesicSolveResult:
    for values in (owner, distance, position):
        values.setflags(write=False)
    return GeodesicSolveResult(
        owner,
        distance,
        position,
        max_live_entries,
        settled_count,
        queue_insertions,
        queue_decrease_keys,
        queue_pops,
        heap_pixel_bytes,
        position.nbytes,
        distance.nbytes,
    )
