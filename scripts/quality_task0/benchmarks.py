"""Scale matrices and isolated RSS workers for the stereo Quality Task 0 gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import platform as platform_module
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Literal, cast

import numpy as np

from . import (
    content_reader,
    encoding_provider,
    fallback,
    implicit_kd,
    local_strip,
    metric_npz,
    native_kd,
    native_region_solver,
    region_solver,
    repair_records,
    reporting,
    stereo_control,
)


HOST_RSS_CAP_BYTES = 512 * 1024 * 1024
CONTROL_RSS_CAP_BYTES = 4 * 1024 * 1024
ENCODING_RSS_CAP_BYTES = 8 * 1024 * 1024
_RESOLUTIONS = {"1080p": (1920, 1080), "4k": (3840, 2160)}
_GEOMETRY_RATIOS = ("0.5", "1", "2", "4")
_EDGE_POPULATIONS = (25, 50, 100)
_GATE_NAMES = (
    "geometry-scale",
    "region-scale",
    "fallback-scale",
    "content-scale",
    "control-scale",
    "encoding-scale",
    "fault-matrix",
    "runtime-matrix",
    "real-fixture",
)


class BenchmarkError(RuntimeError):
    """Base Task 0 scale-gate error."""


class BenchmarkCapacityError(BenchmarkError):
    """A full case cannot begin inside the fixed host budget."""


class BenchmarkExecutionError(BenchmarkError):
    """An isolated worker failed or emitted invalid evidence."""


def _context_token(context: str) -> str:
    if context == "cpu":
        return "cpu"
    if context.startswith("cuda:"):
        fields = context.split(":", 2)
        if len(fields) == 3 and fields[1].isdigit():
            return f"cuda{fields[1]}"
    raise ValueError(f"invalid benchmark device context: {context}")


@dataclass(frozen=True)
class BenchmarkCase:
    run_id: str
    gate: str
    case: str
    device_context: str
    operation: str
    parameters: tuple[tuple[str, object], ...]
    fingerprint: str

    def _base_object(self) -> dict:
        return {
            "run_id": self.run_id,
            "gate": self.gate,
            "case": self.case,
            "device_context": self.device_context,
            "operation": self.operation,
            "parameters": dict(self.parameters),
        }

    def to_object(self) -> dict:
        return {**self._base_object(), "fingerprint": self.fingerprint}

    def parameter(self, name: str) -> object:
        values = dict(self.parameters)
        if name not in values:
            raise KeyError(name)
        return values[name]

    def integer_parameter(self, name: str) -> int:
        value = self.parameter(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise BenchmarkExecutionError(f"benchmark parameter is not an integer: {name}")
        return value

    def string_parameter(self, name: str) -> str:
        value = self.parameter(name)
        if not isinstance(value, str):
            raise BenchmarkExecutionError(f"benchmark parameter is not a string: {name}")
        return value

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        gate: str,
        case: str,
        device_context: str,
        operation: str,
        parameters: dict,
    ) -> BenchmarkCase:
        if not parameters or all(isinstance(key, str) for key in parameters):
            ordered = tuple(sorted(parameters.items()))
        else:
            raise ValueError("benchmark parameters require string keys")
        provisional = cls(
            run_id=run_id,
            gate=gate,
            case=case,
            device_context=device_context,
            operation=operation,
            parameters=ordered,
            fingerprint="",
        )
        reporting.GateRunDescriptor.create(
            run_id=run_id,
            gate=gate,
            case=case,
            device_context=device_context,
            status="passed",
            wall_time_ns=0,
            peak_rss_bytes=0,
            rss_cap_bytes=1,
            arena_bytes=0,
            visit_total=0,
            visit_max=0,
            visit_p95=0,
            equality_sha256="00" * 32,
            facts={},
        )
        return cls(
            **{
                **provisional.__dict__,
                "fingerprint": hashlib.sha256(
                    reporting.canonical_json_bytes(provisional._base_object())
                ).hexdigest(),
            }
        )

    @classmethod
    def from_object(cls, value: object) -> BenchmarkCase:
        expected = {
            "run_id",
            "gate",
            "case",
            "device_context",
            "operation",
            "parameters",
            "fingerprint",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise BenchmarkExecutionError("benchmark case schema is invalid")
        if not isinstance(value["parameters"], dict):
            raise BenchmarkExecutionError("benchmark parameters are not an object")
        case = cls.create(
            run_id=value["run_id"],
            gate=value["gate"],
            case=value["case"],
            device_context=value["device_context"],
            operation=value["operation"],
            parameters=value["parameters"],
        )
        if case.fingerprint != value["fingerprint"]:
            raise BenchmarkExecutionError("benchmark case fingerprint is invalid")
        return case


@dataclass(frozen=True)
class WorkerResult:
    status: reporting.RunStatus
    arena_bytes: int
    visit_total: int
    visit_max: int
    visit_p95: int
    equality_sha256: str
    facts: dict
    baseline_rss_bytes: int = 0
    phase_wall_time_ns: int = 0
    worker_process_id: int = 0

    def to_object(self) -> dict:
        return {
            "status": self.status,
            "arena_bytes": self.arena_bytes,
            "visit_total": self.visit_total,
            "visit_max": self.visit_max,
            "visit_p95": self.visit_p95,
            "equality_sha256": self.equality_sha256,
            "facts": self.facts,
            "baseline_rss_bytes": self.baseline_rss_bytes,
            "phase_wall_time_ns": self.phase_wall_time_ns,
            "worker_process_id": self.worker_process_id,
        }

    @classmethod
    def from_object(cls, value: object) -> WorkerResult:
        expected = {
            "status",
            "arena_bytes",
            "visit_total",
            "visit_max",
            "visit_p95",
            "equality_sha256",
            "facts",
            "baseline_rss_bytes",
            "phase_wall_time_ns",
            "worker_process_id",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise BenchmarkExecutionError("benchmark worker result schema is invalid")
        if value["status"] not in {"passed", "failed"} or not isinstance(value["facts"], dict):
            raise BenchmarkExecutionError("benchmark worker status/facts are invalid")
        for key in (
            "arena_bytes",
            "visit_total",
            "visit_max",
            "visit_p95",
            "baseline_rss_bytes",
            "phase_wall_time_ns",
            "worker_process_id",
        ):
            reporting.checked_u64(value[key], key)
        if not isinstance(value["equality_sha256"], str) or len(value["equality_sha256"]) != 64:
            raise BenchmarkExecutionError("benchmark equality hash is invalid")
        return cls(
            status=cast(reporting.RunStatus, value["status"]),
            arena_bytes=value["arena_bytes"],
            visit_total=value["visit_total"],
            visit_max=value["visit_max"],
            visit_p95=value["visit_p95"],
            equality_sha256=value["equality_sha256"],
            facts=value["facts"],
            baseline_rss_bytes=value["baseline_rss_bytes"],
            phase_wall_time_ns=value["phase_wall_time_ns"],
            worker_process_id=value["worker_process_id"],
        )


@dataclass(frozen=True)
class GeometryMemoryEstimate:
    geometry_pixels: int
    query_pixels: int
    index_bytes: int
    query_bytes: int
    predicted_phase_rss_bytes: int


def available_gate_names() -> tuple[str, ...]:
    return _GATE_NAMES


def _ratio_shape(width: int, height: int, ratio: str) -> tuple[int, int]:
    if ratio == "0.5":
        return height, width // 2
    if ratio == "1":
        return height, width
    if ratio == "2":
        return height * 2, width
    if ratio == "4":
        return height * 2, width * 2
    raise ValueError("geometry ratio is outside the frozen matrix")


def build_geometry_cases(device_contexts: tuple[str, ...]) -> tuple[BenchmarkCase, ...]:
    cases: list[BenchmarkCase] = []
    for context in device_contexts:
        token = _context_token(context)
        for resolution, (width, height) in _RESOLUTIONS.items():
            for ratio in _GEOMETRY_RATIOS:
                ratio_token = ratio.replace(".", "_")
                for percent in _EDGE_POPULATIONS:
                    cases.append(
                        BenchmarkCase.create(
                            run_id=(f"geometry.{token}.{resolution}.r{ratio_token}.edge{percent}"),
                            gate="geometry-scale",
                            case=f"{resolution}-r{ratio_token}-edge{percent}",
                            device_context=context,
                            operation="geometry",
                            parameters={
                                "width": width,
                                "height": height,
                                "resolution": resolution,
                                "geometry_ratio": ratio,
                                "edge_percent": percent,
                            },
                        )
                    )
    return tuple(cases)


def _simple_case(
    *,
    run_id: str,
    gate: str,
    case: str,
    operation: str,
    parameters: dict,
    context: str = "cpu",
) -> BenchmarkCase:
    return BenchmarkCase.create(
        run_id=run_id,
        gate=gate,
        case=case,
        device_context=context,
        operation=operation,
        parameters=parameters,
    )


def build_case_registry(
    *,
    device_contexts: tuple[str, ...],
    fixture_settings: Path,
) -> dict[str, BenchmarkCase]:
    cases = list(build_geometry_cases(device_contexts))
    for context in device_contexts:
        token = _context_token(context)
        for resolution, (width, height) in _RESOLUTIONS.items():
            for percent in _EDGE_POPULATIONS:
                cases.append(
                    _simple_case(
                        run_id=f"region.{token}.{resolution}.edge{percent}",
                        gate="region-scale",
                        case=f"{resolution}-edge{percent}",
                        operation="region",
                        context=context,
                        parameters={
                            "width": width,
                            "height": height,
                            "resolution": resolution,
                            "edge_percent": percent,
                        },
                    )
                )
        cases.append(
            _simple_case(
                run_id=f"fallback.{token}.4k.edge25",
                gate="fallback-scale",
                case="4k-edge25",
                operation="fallback",
                context=context,
                parameters={"width": 3840, "height": 2160, "edge_percent": 25},
            )
        )
    for resolution, (width, height) in _RESOLUTIONS.items():
        for ratio in _GEOMETRY_RATIOS:
            ratio_token = ratio.replace(".", "_")
            cases.append(
                _simple_case(
                    run_id=f"content.cpu.{resolution}.r{ratio_token}",
                    gate="content-scale",
                    case=f"{resolution}-r{ratio_token}",
                    operation="content",
                    parameters={
                        "width": width,
                        "height": height,
                        "resolution": resolution,
                        "geometry_ratio": ratio,
                    },
                )
            )
    for frame_count in (1, 10_000, 1_000_000, 4_194_304):
        cases.append(
            _simple_case(
                run_id=f"control.cpu.n{frame_count}",
                gate="control-scale",
                case=f"n{frame_count}",
                operation="control",
                parameters={"frame_count": frame_count},
            )
        )
        for mode in ("direct", "assembled"):
            cases.append(
                _simple_case(
                    run_id=f"encoding.cpu.{mode}.n{frame_count}",
                    gate="encoding-scale",
                    case=f"{mode}-n{frame_count}",
                    operation="encoding",
                    parameters={"frame_count": frame_count, "mode": mode},
                )
            )
    cases.extend(
        (
            _simple_case(
                run_id="fault.cpu.aggregate",
                gate="fault-matrix",
                case="aggregate",
                operation="faults",
                parameters={},
            ),
            _simple_case(
                run_id="runtime.cpu.matrix",
                gate="runtime-matrix",
                case="matrix",
                operation="runtime",
                parameters={},
            ),
            _simple_case(
                run_id="fixture.cpu.reported",
                gate="real-fixture",
                case="reported",
                operation="fixture",
                parameters={"settings_path": str(Path(fixture_settings))},
            ),
        )
    )
    registry = {case.run_id: case for case in cases}
    if len(registry) != len(cases):
        raise ValueError("benchmark registry has duplicate run ids")
    return registry


def required_run_ids(
    selected_gates: tuple[str, ...],
    registry: dict[str, BenchmarkCase],
) -> tuple[str, ...]:
    unknown = set(selected_gates).difference(_GATE_NAMES)
    if unknown:
        raise reporting.GateSelectionError(f"unknown benchmark gates: {sorted(unknown)}")
    required = tuple(
        sorted(run_id for run_id, case in registry.items() if case.gate in selected_gates)
    )
    if not required:
        raise reporting.IncompleteTask0ReportError("selected gates have no registered runs")
    return required


def estimate_geometry_case_bytes(
    *,
    width: int,
    height: int,
    geometry_ratio: str,
    edge_percent: int,
) -> GeometryMemoryEstimate:
    geometry_height, geometry_width = _ratio_shape(width, height, geometry_ratio)
    geometry_pixels = geometry_height * geometry_width
    query_pixels = max(1, width * height * edge_percent // 100)
    index_bytes = 4 * geometry_pixels + 8
    query_bytes = 37 * query_pixels
    build_scratch = 4 * geometry_pixels
    fill_scratch = 8 * 1024 * 1024
    fixed_headroom = 24 * 1024 * 1024
    predicted = (
        4 * geometry_pixels
        + index_bytes
        + query_bytes
        + build_scratch
        + fill_scratch
        + fixed_headroom
    )
    return GeometryMemoryEstimate(
        geometry_pixels=geometry_pixels,
        query_pixels=query_pixels,
        index_bytes=index_bytes,
        query_bytes=query_bytes,
        predicted_phase_rss_bytes=predicted,
    )


def require_geometry_capacity(estimate: GeometryMemoryEstimate) -> None:
    if estimate.predicted_phase_rss_bytes > HOST_RSS_CAP_BYTES:
        raise BenchmarkCapacityError(
            "geometry scale case exceeds the 512 MiB host cap at pre-mutation admission"
        )


def visit_p95_nearest_rank(visits: np.ndarray) -> int:
    if not isinstance(visits, np.ndarray) or visits.dtype != np.uint32 or visits.ndim != 1:
        raise TypeError("visit vector must be a one-dimensional uint32 array")
    if visits.size == 0:
        raise ValueError("visit vector is empty")
    histogram = np.bincount(visits)
    rank = math.ceil(0.95 * int(visits.size))
    cumulative = 0
    for value, count in enumerate(histogram):
        cumulative += int(count)
        if cumulative >= rank:
            return value
    raise AssertionError("visit histogram did not reach its population")


def full_grid_nearest_sample_indices(
    target_y: np.ndarray,
    target_x: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    if (
        target_y.dtype != np.float64
        or target_x.dtype != np.float64
        or target_y.ndim != 1
        or target_x.shape != target_y.shape
    ):
        raise TypeError("full-grid targets must be matching one-dimensional float64 arrays")
    if height <= 0 or width <= 0 or height * width > implicit_kd.UINT32_MAX:
        raise ValueError("full-grid shape is outside the uint32 sample-index contract")
    rows = np.ceil(target_y - np.float64(0.5)).astype(np.int64)
    columns = np.ceil(target_x - np.float64(0.5)).astype(np.int64)
    np.clip(rows, 0, height - 1, out=rows)
    np.clip(columns, 0, width - 1, out=columns)
    return (rows * width + columns).astype(np.uint32)


def _hash_arrays(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.uint64).tobytes())
        digest.update(memoryview(cast(Any, contiguous)).cast("B"))
    return digest.hexdigest()


def _geometry_worker(case: BenchmarkCase) -> WorkerResult:
    width = case.integer_parameter("width")
    height = case.integer_parameter("height")
    ratio = case.string_parameter("geometry_ratio")
    edge_percent = case.integer_parameter("edge_percent")
    estimate = estimate_geometry_case_bytes(
        width=width,
        height=height,
        geometry_ratio=ratio,
        edge_percent=edge_percent,
    )
    try:
        require_geometry_capacity(estimate)
    except BenchmarkCapacityError:
        facts = {
            "capacity_rejected": True,
            "pre_mutation": True,
            "predicted_phase_rss_bytes": estimate.predicted_phase_rss_bytes,
            "geometry_pixels": estimate.geometry_pixels,
            "query_pixels": estimate.query_pixels,
            "oracle": "full-grid-closed-form-v1",
        }
        return WorkerResult(
            status="passed",
            arena_bytes=estimate.index_bytes,
            visit_total=0,
            visit_max=0,
            visit_p95=0,
            equality_sha256=hashlib.sha256(reporting.canonical_json_bytes(facts)).hexdigest(),
            facts=facts,
        )
    geometry_height, geometry_width = _ratio_shape(width, height, ratio)
    region_ids = np.ones((geometry_height, geometry_width), dtype=np.uint32)
    index = native_kd.build_native_implicit_region_kd(region_ids, region_count=1)
    query_count = estimate.query_pixels
    query_regions = np.ones(query_count, dtype=np.uint32)
    target_y = np.empty(query_count, dtype=np.float64)
    target_x = np.empty(query_count, dtype=np.float64)
    chunk_size = 1_048_576
    for start in range(0, query_count, chunk_size):
        stop = min(query_count, start + chunk_size)
        linear = np.arange(start, stop, dtype=np.uint64)
        rows = linear // np.uint64(width)
        columns = linear - rows * np.uint64(width)
        target_y[start:stop] = (rows.astype(np.float64) + np.float64(0.5)) * np.float64(
            geometry_height / height
        ) - np.float64(0.5)
        target_x[start:stop] = (columns.astype(np.float64) + np.float64(0.5)) * np.float64(
            geometry_width / width
        ) - np.float64(0.5)
    budget = implicit_kd.VisitBudget(limit=implicit_kd.UINT64_MAX)
    result = native_kd.query_native_geometry_batch(
        index,
        query_regions,
        target_y,
        target_x,
        budget=budget,
    )
    sample_positions = np.unique(
        np.linspace(0, query_count - 1, min(query_count, 4096), dtype=np.int64)
    )
    expected_samples = full_grid_nearest_sample_indices(
        target_y[sample_positions],
        target_x[sample_positions],
        height=geometry_height,
        width=geometry_width,
    )
    if not np.array_equal(result.sample_index[sample_positions], expected_samples):
        raise BenchmarkExecutionError("native geometry differs from the full-grid closed form")
    visits = result.visited_nodes
    visit_total = int(visits.sum(dtype=np.uint64))
    if visit_total != budget.consumed:
        raise BenchmarkExecutionError("geometry visit accounting is incomplete")
    return WorkerResult(
        status="passed",
        arena_bytes=index.index_bytes,
        visit_total=visit_total,
        visit_max=int(visits.max(initial=np.uint32(0))),
        visit_p95=visit_p95_nearest_rank(visits),
        equality_sha256=_hash_arrays(
            result.found,
            result.sample_index,
            result.distance2,
            result.visited_nodes,
        ),
        facts={
            "capacity_rejected": False,
            "geometry_pixels": estimate.geometry_pixels,
            "query_pixels": query_count,
            "index_bytes": index.index_bytes,
            "index_bound_bytes": 8 * estimate.geometry_pixels + 4,
            "oracle": "full-grid-closed-form-v1",
            "native_build": native_kd.native_build_info(),
        },
    )


def _fault_worker() -> WorkerResult:
    checks: list[str] = []
    try:
        estimate = estimate_geometry_case_bytes(
            width=3840,
            height=2160,
            geometry_ratio="4",
            edge_percent=100,
        )
        require_geometry_capacity(estimate)
    except BenchmarkCapacityError:
        checks.append("geometry-pre-mutation-cap")
    budget = implicit_kd.VisitBudget(limit=0)
    index = native_kd.build_native_implicit_region_kd(
        np.ones((1, 1), dtype=np.uint32), region_count=1
    )
    try:
        native_kd.query_native_geometry_batch(
            index,
            np.ones(1, dtype=np.uint32),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            budget=budget,
        )
    except implicit_kd.QualityGeometryQueryBudgetError:
        checks.append("geometry-next-node-fatal")
    if checks != ["geometry-pre-mutation-cap", "geometry-next-node-fatal"]:
        raise BenchmarkExecutionError("fault matrix did not reach every expected disposition")
    digest = hashlib.sha256(reporting.canonical_json_bytes(checks)).hexdigest()
    return WorkerResult("passed", index.index_bytes, 0, 0, 0, digest, {"checks": checks})


def _region_fixture(
    width: int,
    height: int,
    edge_percent: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    band_width = max(2, width * edge_percent // 100)
    start = (width - band_width) // 2
    stop = start + band_width
    guide = np.zeros((height, width, 3), dtype=np.uint8)
    edge = (start + stop) // 2
    guide[:, edge:] = np.uint8(255)
    band = np.zeros((height, width), dtype=np.bool_)
    band[:, start:stop] = True
    seeds = np.zeros((height, width), dtype=np.uint32)
    seeds[:, start] = np.uint32(1)
    seeds[:, stop - 1] = np.uint32(2)
    ranks = np.asarray([0, 20, 10], dtype=np.uint32)
    return guide, band, seeds, ranks


def _assert_small_region_oracle() -> str:
    guide, band, seeds, ranks = _region_fixture(23, 17, 75)
    expected = region_solver.solve_geodesic_regions(guide, band, seeds, ranks)
    actual = native_region_solver.solve_native_geodesic_regions(guide, band, seeds, ranks)
    for left, right in (
        (actual.region_map, expected.region_map),
        (actual.distance, expected.distance),
        (actual.heap_position, expected.heap_position),
    ):
        if not np.array_equal(left, right):
            raise BenchmarkExecutionError("native region solver differs from scalar Dijkstra")
    return _hash_arrays(actual.region_map, actual.distance, actual.heap_position)


def _region_worker(case: BenchmarkCase) -> WorkerResult:
    width = case.integer_parameter("width")
    height = case.integer_parameter("height")
    edge_percent = case.integer_parameter("edge_percent")
    guide, band, seeds, ranks = _region_fixture(width, height, edge_percent)
    result = native_region_solver.solve_native_geodesic_regions(guide, band, seeds, ranks)
    band_pixels = int(np.count_nonzero(band))
    if result.settled_count != band_pixels or result.max_live_entries > band_pixels:
        raise BenchmarkExecutionError("native region queue violates its one-entry bound")
    oracle_hash = _assert_small_region_oracle()
    equality = hashlib.sha256()
    equality.update(bytes.fromhex(oracle_hash))
    equality.update(bytes.fromhex(_hash_arrays(result.region_map, result.distance)))
    arena = result.heap_pixel_bytes + result.heap_position_bytes + result.distance_bytes
    return WorkerResult(
        status="passed",
        arena_bytes=arena,
        visit_total=0,
        visit_max=0,
        visit_p95=0,
        equality_sha256=equality.hexdigest(),
        facts={
            "band_pixels": band_pixels,
            "max_live_entries": result.max_live_entries,
            "settled_count": result.settled_count,
            "queue_insertions": result.queue_insertions,
            "queue_decrease_keys": result.queue_decrease_keys,
            "queue_pops": result.queue_pops,
            "heap_pixel_bytes": result.heap_pixel_bytes,
            "heap_position_bytes": result.heap_position_bytes,
            "distance_bytes": result.distance_bytes,
            "oracle": "scalar-four-neighbour-dijkstra-v1",
            "oracle_hash": oracle_hash,
        },
    )


def fallback_stripe_geometry(*, edge_percent: int, safe_radius: int) -> tuple[int, int]:
    if edge_percent <= 0 or edge_percent >= 100 or safe_radius < 0:
        raise ValueError("fallback stripe inputs are outside their closed range")
    divisor = math.gcd(100, edge_percent)
    period_parts = 100 // divisor
    hole_parts = edge_percent // divisor
    donor_parts = period_parts - hole_parts
    scale = max(1, math.ceil((2 * safe_radius + 1) / donor_parts))
    return period_parts * scale, hole_parts * scale


def _fallback_worker(case: BenchmarkCase) -> WorkerResult:
    width = case.integer_parameter("width")
    height = case.integer_parameter("height")
    edge_percent = case.integer_parameter("edge_percent")
    safe_radius = max(1, math.floor(height / 1080.0 + 0.5))
    stripe_period, hole_width = fallback_stripe_geometry(
        edge_percent=edge_percent,
        safe_radius=safe_radius,
    )
    hole_columns = np.arange(width) % stripe_period < hole_width
    coverage = np.full((height, width), 16, dtype=np.uint8)
    regions = np.ones((height, width), dtype=np.uint32)
    coverage[:, hole_columns] = np.uint8(0)
    regions[:, hole_columns] = np.uint32(implicit_kd.UINT32_MAX)
    values = np.arange(width, dtype=np.uint32)[None, :]
    planes = tuple(
        np.ascontiguousarray(
            np.broadcast_to(((values + channel * 67) % 256).astype(np.uint8), (height, width))
        )
        for channel in range(3)
    )
    bgr = cast(tuple[np.ndarray, np.ndarray, np.ndarray], planes)
    analysis = local_strip.QualityVisibilityAnalysis(coverage, regions, bgr)
    columns = np.flatnonzero(hole_columns).astype(np.uint32)
    pixels = (
        np.arange(height, dtype=np.uint32)[:, None] * np.uint32(width) + columns[None, :]
    ).reshape(-1)
    records = np.zeros(pixels.size, dtype=repair_records.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = pixels
    records["lane_mask"] = np.uint16(0xFFFF)
    records["region_id"] = np.uint32(1)
    budgets = local_strip.QualityRepairBudgets()
    result = fallback.precompute_native_fallbacks(
        records,
        analysis,
        render_shape=(height, width),
        region_count=1,
        budgets=budgets,
    )
    visits = result.visited_nodes
    total = int(visits.sum(dtype=np.uint64))
    if total != budgets.repair_fallback_visits_consumed:
        raise BenchmarkExecutionError("fallback visit accounting is incomplete")
    if result.query_count != pixels.size:
        raise BenchmarkExecutionError("fallback query population differs from repair records")
    return WorkerResult(
        status="passed",
        arena_bytes=result.index_bytes + visits.nbytes + result.records.nbytes,
        visit_total=total,
        visit_max=int(visits.max(initial=np.uint32(0))),
        visit_p95=visit_p95_nearest_rank(visits),
        equality_sha256=_hash_arrays(result.records, visits),
        facts={
            "query_count": result.query_count,
            "query_fraction": result.query_count / (width * height),
            "indexed_donor_count": result.indexed_donor_count,
            "index_bytes": result.index_bytes,
            "index_cap_bytes": fallback.QUALITY_REPAIR_FALLBACK_INDEX_BYTES,
            "visit_bytes": visits.nbytes,
            "visit_cap_bytes": fallback.QUALITY_REPAIR_FALLBACK_VISITS_BYTES,
            "retained_safe_donor_mask_bytes": result.retained_safe_donor_mask_bytes,
            "oracle": "same-region-safe-donor-native-v1",
        },
    )


def _content_worker(case: BenchmarkCase) -> WorkerResult:
    width = case.integer_parameter("width")
    height = case.integer_parameter("height")
    ratio = case.string_parameter("geometry_ratio")
    geometry_height, geometry_width = _ratio_shape(width, height, ratio)
    shape = (geometry_height, geometry_width)
    with tempfile.TemporaryDirectory(prefix="quality-task0-content-") as directory:
        path = Path(directory) / "frame_000000.npz"
        inverse = np.ones(shape, dtype=np.float32)
        valid = np.ones(shape, dtype=np.bool_)
        write_trace = metric_npz.write_metric_npz(
            path,
            inverse,
            valid,
            np.float32(1.0),
        )
        del inverse, valid
        gc.collect()
        frame = content_reader.audit_metric_payload(path, expected_shape=shape)
        identity = frame.identity()
        owned_bytes = frame.owned_payload_bytes
        facts = {
            "geometry_pixels": geometry_height * geometry_width,
            "owned_payload_bytes": owned_bytes,
            "expected_owned_payload_bytes": 5 * geometry_height * geometry_width,
            "archive_bytes": write_trace.archive_bytes,
            "logical_bound_bytes": write_trace.logical_bound_bytes,
            "maximum_input_chunk_bytes": write_trace.maximum_input_chunk_bytes,
            "maximum_output_block_bytes": write_trace.maximum_output_block_bytes,
            "complete_raw_member_copies": write_trace.complete_raw_member_copies,
            "complete_compressed_member_copies": write_trace.complete_compressed_member_copies,
            "maximum_stream_bytes": frame.maximum_stream_bytes,
            "stream_workspace_peak_bytes": frame.stream_workspace_peak_bytes,
            "value_scratch_peak_bytes": frame.value_scratch_peak_bytes,
            "archive_form": identity.archive_form,
            "released_before_next_frame": True,
        }
        if owned_bytes != facts["expected_owned_payload_bytes"]:
            raise BenchmarkExecutionError("metric audit does not own exactly 5*G bytes")
        equality = hashlib.sha256(
            reporting.canonical_json_bytes(
                {
                    "sha256": identity.sha256,
                    "shape": list(identity.native_shape),
                    "focal_bits": identity.focal_x_normalized_float32_bits,
                }
            )
        ).hexdigest()
        frame.release()
        if not frame.released or frame.owned_payload_bytes != 0:
            raise BenchmarkExecutionError("metric frame did not release its owned arrays")
    return WorkerResult(
        "passed",
        int(facts["owned_payload_bytes"]),
        0,
        0,
        0,
        equality,
        facts,
    )


def _stereo_provider(frame_count: int) -> stereo_control.ReplayableStereoProvider:
    generation = stereo_control.StereoProviderGeneration(
        audit_generation_id="01" * 16,
        input_manifest_fingerprint="02" * 32,
        frame_count=frame_count,
    )

    def replay():
        for index in range(frame_count):
            stem = stereo_control.canonical_frame_stem(index)
            filename = f"{stem}.png"
            yield stereo_control.StereoProviderItem(
                audit_generation_id=generation.audit_generation_id,
                input_manifest_fingerprint=generation.input_manifest_fingerprint,
                source_index=index,
                frame_name=stem,
                guide_path=Path("00_original_frames") / filename,
                geometry_path=Path("02_depth_raw") / filename,
                left_path=Path("04_left_frames") / filename,
                right_path=Path("04_right_frames") / filename,
                guide_sha256="03" * 32,
                geometry_sha256="04" * 32,
            )

    return stereo_control.ReplayableStereoProvider(generation, replay)


def _control_worker(case: BenchmarkCase) -> WorkerResult:
    frame_count = case.integer_parameter("frame_count")
    provider = _stereo_provider(frame_count)
    plan = stereo_control.audit_stereo_provider(
        provider,
        lambda item: (
            stereo_control.StereoAction.RENDER
            if item.source_index % 3 != 0
            else stereo_control.StereoAction.MIGRATE
        ),
    )
    stereo_control.validate_stereo_provider_replay(
        provider,
        plan,
        mutation_started=False,
    )
    fast = {
        str(workers): stereo_control.select_fast_pipeline_capacity(
            configured_workers=workers,
            pending_render_count=plan.render_count,
            fast_active_scratch=32 * 1024 * 1024,
            fast_slot_bytes=8 * 1024 * 1024,
        ).__dict__
        for workers in (1, 4, 16)
    }
    quality_capacity = stereo_control.quality_pipeline_capacity(plan.render_count)
    facts = {
        "frame_count": frame_count,
        "packed_action_bytes": plan.actions.storage_nbytes,
        "persistent_control_bytes": plan.persistent_control_bytes,
        "render_count": plan.render_count,
        "migrate_count": plan.migrate_count,
        "quality_capacity": list(quality_capacity),
        "fast_capacities": fast,
        "retained_item_lists": 0,
    }
    return WorkerResult(
        "passed",
        plan.persistent_control_bytes,
        0,
        0,
        0,
        hashlib.sha256(reporting.canonical_json_bytes(facts)).hexdigest(),
        facts,
    )


def _deterministic_encoding_provider(
    frame_count: int,
    mode: str,
) -> encoding_provider.EncodingSequenceProvider:
    generation_id = "05" * 16
    encoding_mode: Literal["direct_stereo", "assembled_vr"]
    left: PurePosixPath | None
    right: PurePosixPath | None
    frames: PurePosixPath | None
    if mode == "direct":
        encoding_mode = "direct_stereo"
        left = PurePosixPath("07_left_upscaled")
        right = PurePosixPath("07_right_upscaled")
        frames = None
    elif mode == "assembled":
        encoding_mode = "assembled_vr"
        left = right = None
        frames = PurePosixPath("99_vr_frames")
    else:
        raise BenchmarkExecutionError("encoding benchmark mode is invalid")
    last_index = frame_count - 1
    generation = encoding_provider._generation_object(
        audit_generation_id=generation_id,
        mode=encoding_mode,
        frame_count=frame_count,
        first_index=0,
        last_index=last_index,
        left_parent=left,
        right_parent=right,
        frames_parent=frames,
    )
    return encoding_provider.EncodingSequenceProvider(
        mode=encoding_mode,
        frame_count=frame_count,
        first_index=0,
        last_index=last_index,
        left_parent=left,
        right_parent=right,
        frames_parent=frames,
        audit_generation_id=generation_id,
        generation_fingerprint=hashlib.sha256(
            encoding_provider.canonical_json_bytes(generation)
        ).hexdigest(),
    ).validated()


def _encoding_worker(case: BenchmarkCase) -> WorkerResult:
    frame_count = case.integer_parameter("frame_count")
    mode = case.string_parameter("mode")
    provider = _deterministic_encoding_provider(frame_count, mode)
    audit = encoding_provider.audit_encoding_provider(provider)
    encoding_provider.validate_encoding_provider(provider, audit, launch_started=False)
    headers: tuple[tuple[int, int], ...]
    vr_format: Literal["side_by_side", "over_under"] = "side_by_side"
    if mode == "direct":
        headers = ((1920, 1080), (1920, 1080))
    else:
        headers = ((3840, 1080),)
    argv = encoding_provider.build_image_encoding_argv(
        provider,
        ffmpeg="ffmpeg",
        fps="30/1",
        vr_format=vr_format,
        per_eye_size=(1920, 1080),
        first_headers=headers,
        output_path=".depth-surge-final-v4-" + "06" * 16 + ".tmp.mp4",
    )
    peaks = encoding_provider.encoding_control_phase_peaks()
    facts = {
        "frame_count": frame_count,
        "mode": mode,
        "provider_slots": len(provider.__slots__),
        "retained_item_lists": 0,
        "generation_fingerprint": provider.generation_fingerprint,
        "item_stream_sha256": audit.item_stream_sha256,
        "argv_sha256": hashlib.sha256(reporting.canonical_json_bytes(list(argv))).hexdigest(),
        "phase_peaks": peaks,
        "maximum_phase_bytes": max(peaks.values()),
        "control_cap_bytes": encoding_provider.FINAL_ENCODING_CONTROL_BUDGET,
    }
    return WorkerResult(
        "passed",
        max(peaks.values()),
        0,
        0,
        0,
        hashlib.sha256(reporting.canonical_json_bytes(facts)).hexdigest(),
        facts,
    )


def _bounded_version(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise BenchmarkExecutionError(f"runtime command failed: {command[0]}") from error
    if len(completed.stdout) > 1024 * 1024:
        raise BenchmarkExecutionError(f"runtime command output is unbounded: {command[0]}")
    return completed.stdout.decode("utf-8", errors="strict").splitlines()[0]


def _installed_memory_bytes() -> int:
    if os.name == "nt":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
        return int(status.ullTotalPhys)
    sysconf = getattr(os, "sysconf")
    page_size = sysconf("SC_PAGE_SIZE")
    pages = sysconf("SC_PHYS_PAGES")
    return int(page_size * pages)


def _runtime_worker() -> WorkerResult:
    import cv2
    import torch

    cuda_available = bool(torch.cuda.is_available())
    facts = {
        "cpu_model": platform_module.processor() or platform_module.machine(),
        "physical_memory_bytes": _installed_memory_bytes(),
        "logical_core_count": os.cpu_count() or 1,
        "python": platform_module.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "ffmpeg": _bounded_version(["ffmpeg", "-version"]),
        "ffprobe": _bounded_version(["ffprobe", "-version"]),
        "native_build": native_kd.native_build_info(),
        "platform": platform_module.platform(),
    }
    return WorkerResult(
        "passed",
        0,
        0,
        0,
        0,
        hashlib.sha256(reporting.canonical_json_bytes(facts)).hexdigest(),
        facts,
    )


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _frame_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    count = 0
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file(follow_symlinks=False) and re.fullmatch(
                r"frame_[0-9]+\.png", entry.name
            ):
                count += 1
    return count


def _fixture_worker(case: BenchmarkCase) -> WorkerResult:
    settings_path = Path(case.string_parameter("settings_path"))
    raw = settings_path.read_bytes()
    if len(raw) > 512 * 1024:
        raise BenchmarkExecutionError("fixture settings exceed the audit cap")

    def unique(pairs: list[tuple[str, object]]) -> dict:
        value: dict = {}
        for key, item in pairs:
            if key in value:
                raise BenchmarkExecutionError(f"fixture settings duplicate key: {key}")
            value[key] = item
        return value

    try:
        settings = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
        metadata = settings["metadata"]
        video = settings["video_properties"]
        processing = settings["processing_settings"]
        runtime = settings["runtime_info"]
        source_path = Path(metadata["source_video"])
        final_path = Path(runtime.get("final_output", runtime.get("final_output_relative_path")))
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise BenchmarkExecutionError("fixture settings schema is unusable") from error
    if not source_path.is_file() or not final_path.is_file():
        raise BenchmarkExecutionError("fixture source or final video is missing")
    source_hash, source_bytes = _sha256_file(source_path)
    final_hash, final_bytes = _sha256_file(final_path)
    root = settings_path.parent
    frame_counts = {
        name: _frame_count(root / name)
        for name in ("00_original_frames", "04_left_frames", "04_right_frames")
    }
    facts = {
        "settings_path": str(settings_path),
        "settings_sha256": hashlib.sha256(raw).hexdigest(),
        "settings_schema_version": metadata.get("settings_schema_version"),
        "processing_status": metadata.get("processing_status"),
        "source_sha256": source_hash,
        "source_bytes": source_bytes,
        "final_sha256": final_hash,
        "final_bytes": final_bytes,
        "width": video["width"],
        "height": video["height"],
        "frame_count": video["frame_count"],
        "stereo_strength": processing["stereo_strength"],
        "occlusion_fill": processing["occlusion_fill"],
        "frame_counts": frame_counts,
        "decoy_settings_ignored": True,
    }
    return WorkerResult(
        "passed",
        0,
        0,
        0,
        0,
        hashlib.sha256(reporting.canonical_json_bytes(facts)).hexdigest(),
        facts,
    )


def _prepare_worker_case(case: BenchmarkCase) -> None:
    if case.operation != "runtime" and not case.device_context.startswith("cuda:"):
        return
    import cv2  # noqa: F401
    import torch

    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.get_device_name(0)
    torch.get_num_threads()
    torch.get_num_interop_threads()
    cv2.getNumThreads()


def _execute_prepared_worker_case(
    case: BenchmarkCase,
    *,
    baseline_rss_bytes: int,
) -> WorkerResult:
    operations: dict[str, Callable[[BenchmarkCase], WorkerResult]] = {
        "geometry": _geometry_worker,
        "region": _region_worker,
        "fallback": _fallback_worker,
        "content": _content_worker,
        "control": _control_worker,
        "encoding": _encoding_worker,
        "runtime": lambda _case: _runtime_worker(),
        "fixture": _fixture_worker,
        "faults": lambda _case: _fault_worker(),
    }
    if case.operation in operations:
        started = time.perf_counter_ns()
        result = operations[case.operation](case)
        return replace(
            result,
            baseline_rss_bytes=baseline_rss_bytes,
            phase_wall_time_ns=time.perf_counter_ns() - started,
            worker_process_id=os.getpid(),
        )
    raise BenchmarkExecutionError(f"benchmark operation is not implemented: {case.operation}")


def execute_worker_case(case: BenchmarkCase) -> WorkerResult:
    _prepare_worker_case(case)
    gc.collect()
    return _execute_prepared_worker_case(
        case,
        baseline_rss_bytes=reporting.read_process_rss_bytes(os.getpid()),
    )


def _load_case_file(path: Path) -> BenchmarkCase:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkExecutionError("worker case JSON is malformed") from error
    if reporting.canonical_json_bytes(value) != raw:
        raise BenchmarkExecutionError("worker case JSON is not canonical")
    return BenchmarkCase.from_object(value)


def _await_go_handshake(
    path: Path,
    *,
    deadline_ns: int,
    now_ns: Callable[[], int] = time.perf_counter_ns,
    pause: Callable[[float], None] = time.sleep,
) -> None:
    while True:
        try:
            payload = path.read_bytes()
        except (FileNotFoundError, PermissionError) as error:
            if now_ns() >= deadline_ns:
                raise BenchmarkExecutionError("benchmark parent handshake timed out") from error
            pause(0.001)
            continue
        if payload != b"go":
            raise BenchmarkExecutionError("benchmark parent handshake is invalid")
        return


def run_case_subprocess(  # noqa: C901
    case: BenchmarkCase,
    *,
    case_file: Path | None = None,
) -> reporting.GateRunDescriptor:
    root = Path(__file__).resolve().parents[2]
    owned_temp = tempfile.TemporaryDirectory(prefix="quality-task0-control-")
    control_root = Path(owned_temp.name)
    if case_file is None:
        case_file = control_root / "case.json"
        case_file.write_bytes(reporting.canonical_json_bytes(case.to_object()))
    ready_file = control_root / "ready.json"
    go_file = control_root / "go"
    started = time.perf_counter_ns()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "scripts.quality_task0.benchmarks",
            "--worker-case",
            str(case_file),
            "--ready-file",
            str(ready_file),
            "--go-file",
            str(go_file),
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    peak_rss = 0
    process_tree_baseline_rss = 0
    try:
        ready_deadline = time.perf_counter_ns() + 30_000_000_000
        while not ready_file.is_file() and process.poll() is None:
            if time.perf_counter_ns() >= ready_deadline:
                raise BenchmarkExecutionError("benchmark worker baseline handshake timed out")
            time.sleep(0.005)
        if not ready_file.is_file():
            stdout, stderr = process.communicate(timeout=5)
            detail = stderr.decode("utf-8", errors="replace")[-4096:]
            raise BenchmarkExecutionError(f"benchmark worker failed before readiness: {detail}")
        ready_raw = ready_file.read_bytes()
        try:
            ready_value = json.loads(ready_raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BenchmarkExecutionError("benchmark worker readiness is malformed") from error
        if (
            reporting.canonical_json_bytes(ready_value) != ready_raw
            or not isinstance(ready_value, dict)
            or set(ready_value) != {"baseline_rss_bytes", "worker_process_id"}
        ):
            raise BenchmarkExecutionError("benchmark worker readiness schema is invalid")
        reporting.checked_u64(ready_value["baseline_rss_bytes"], "worker baseline RSS")
        reporting.checked_u64(ready_value["worker_process_id"], "worker process id")
        for _ in range(3):
            process_tree_baseline_rss = max(
                process_tree_baseline_rss,
                reporting.read_process_tree_rss_bytes(process.pid),
            )
            time.sleep(0.002)
        go_temporary = go_file.with_name(f".{go_file.name}.tmp")
        go_temporary.write_bytes(b"go")
        os.replace(go_temporary, go_file)
        peak_rss = process_tree_baseline_rss
        while process.poll() is None:
            try:
                peak_rss = max(
                    peak_rss,
                    reporting.read_process_tree_rss_bytes(process.pid),
                )
            except (OSError, reporting.ReportError):
                pass
            time.sleep(0.005)
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        owned_temp.cleanup()
    wall_time = time.perf_counter_ns() - started
    if len(stdout) > 1024 * 1024 or len(stderr) > 1024 * 1024:
        raise BenchmarkExecutionError("benchmark worker output exceeds one MiB")
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[-4096:]
        raise BenchmarkExecutionError(f"benchmark worker failed: {detail}")
    try:
        value = json.loads(stdout.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkExecutionError("benchmark worker output is malformed") from error
    if reporting.canonical_json_bytes(value) != stdout:
        raise BenchmarkExecutionError("benchmark worker output is not canonical")
    result = WorkerResult.from_object(value)
    if (
        result.baseline_rss_bytes != ready_value["baseline_rss_bytes"]
        or result.worker_process_id != ready_value["worker_process_id"]
    ):
        raise BenchmarkExecutionError("worker result differs from readiness identity")
    return finalize_worker_descriptor(
        case,
        result,
        process_wall_time_ns=wall_time,
        process_tree_peak_rss_bytes=peak_rss,
        process_tree_baseline_rss_bytes=process_tree_baseline_rss,
    )


def finalize_worker_descriptor(
    case: BenchmarkCase,
    result: WorkerResult,
    *,
    process_wall_time_ns: int,
    process_tree_peak_rss_bytes: int,
    process_tree_baseline_rss_bytes: int,
) -> reporting.GateRunDescriptor:
    baseline = result.baseline_rss_bytes
    if baseline <= 0 or result.worker_process_id <= 0:
        raise BenchmarkExecutionError("worker did not record a positive process baseline identity")
    if process_tree_baseline_rss_bytes < baseline:
        raise BenchmarkExecutionError("process-tree baseline is below the worker baseline")
    if process_tree_peak_rss_bytes < process_tree_baseline_rss_bytes:
        raise BenchmarkExecutionError("process-tree RSS peak is below its measured baseline")
    if process_wall_time_ns < result.phase_wall_time_ns:
        raise BenchmarkExecutionError(
            "worker phase wall time is outside the process wall time: "
            f"phase={result.phase_wall_time_ns}, process={process_wall_time_ns}"
        )
    phase_peak_rss_bytes = process_tree_peak_rss_bytes - process_tree_baseline_rss_bytes
    if case.operation == "control":
        rss_cap_bytes = CONTROL_RSS_CAP_BYTES
    elif case.operation == "encoding":
        rss_cap_bytes = ENCODING_RSS_CAP_BYTES
    else:
        rss_cap_bytes = HOST_RSS_CAP_BYTES
    cap_exceeded = phase_peak_rss_bytes > rss_cap_bytes
    arena_cap_exceeded = result.arena_bytes > rss_cap_bytes
    wall_cap_ns = 30_000_000_000 if case.operation in {"geometry", "region", "fallback"} else None
    wall_cap_exceeded = wall_cap_ns is not None and result.phase_wall_time_ns > wall_cap_ns
    facts = {
        **result.facts,
        "process_wall_time_ns": process_wall_time_ns,
        "process_baseline_rss_bytes": baseline,
        "process_tree_baseline_rss_bytes": process_tree_baseline_rss_bytes,
        "launcher_peak_rss_bytes": process_tree_baseline_rss_bytes - baseline,
        "process_tree_peak_rss_bytes": process_tree_peak_rss_bytes,
        "rss_measurement": "process-tree-peak-minus-baseline-v1",
        "rss_cap_exceeded": cap_exceeded,
        "arena_cap_exceeded": arena_cap_exceeded,
        "wall_cap_ns": wall_cap_ns,
        "wall_cap_exceeded": wall_cap_exceeded,
    }
    status: reporting.RunStatus = (
        "failed" if cap_exceeded or arena_cap_exceeded or wall_cap_exceeded else result.status
    )
    return reporting.GateRunDescriptor.create(
        run_id=case.run_id,
        gate=case.gate,
        case=case.case,
        device_context=case.device_context,
        status=status,
        wall_time_ns=result.phase_wall_time_ns,
        peak_rss_bytes=phase_peak_rss_bytes,
        rss_cap_bytes=rss_cap_bytes,
        arena_bytes=result.arena_bytes,
        visit_total=result.visit_total,
        visit_max=result.visit_max,
        visit_p95=result.visit_p95,
        equality_sha256=result.equality_sha256,
        facts=facts,
    )


def assemble_gate_report(
    *,
    selected_gates: tuple[str, ...],
    registry: dict[str, BenchmarkCase],
    source: reporting.SourceIdentity,
    environment: reporting.EnvironmentIdentity,
    run: Callable[[BenchmarkCase], reporting.GateRunDescriptor] = run_case_subprocess,
) -> reporting.Task0Report:
    required = required_run_ids(selected_gates, registry)
    runs = tuple(run(registry[run_id]) for run_id in required)
    return reporting.Task0Report.create(
        source=source,
        environment=environment,
        required_run_ids=required,
        runs=runs,
        gate_decision="prebuilt_extension",
    )


def _default_device_contexts() -> tuple[str, ...]:
    try:
        import torch
    except ImportError:
        return ("cpu",)
    if not torch.cuda.is_available():
        return ("cpu",)
    return (
        "cpu",
        reporting.device_context_label("cuda", index=0, name=torch.cuda.get_device_name(0)),
    )


def run_task0_gate_suite(
    selected_gates: tuple[str, ...],
    *,
    report_path: Path | None = None,
) -> reporting.Task0Report:
    root = Path(__file__).resolve().parents[2]
    fixture = Path(
        os.environ.get(
            "DEPTH_SURGE_TASK0_FIXTURE_SETTINGS",
            r"H:\3dtest\1787051840_f908a5f038277cf447b8a6a9b5072311_20260818_191720\f908a5f038277cf447b8a6a9b5072311_1787051942-settings.json",
        )
    )
    registry = build_case_registry(
        device_contexts=_default_device_contexts(), fixture_settings=fixture
    )
    excluded_paths = {"docs/superpowers/reports/2026-08-23-stereo-quality-task0.json"}
    if report_path is not None:
        resolved_report = (
            (root / report_path).resolve()
            if not report_path.is_absolute()
            else report_path.resolve()
        )
        try:
            excluded_paths.add(resolved_report.relative_to(root).as_posix())
        except ValueError:
            pass
    source = reporting.capture_source_identity(
        root,
        excluded_relative_paths=tuple(sorted(excluded_paths)),
    )
    return assemble_gate_report(
        selected_gates=selected_gates,
        registry=registry,
        source=source,
        environment=reporting.capture_environment_identity(),
    )


def _worker_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-case", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--go-file", type=Path, required=True)
    arguments = parser.parse_args(argv)
    case = _load_case_file(arguments.worker_case)
    _prepare_worker_case(case)
    gc.collect()
    baseline_rss_bytes = reporting.read_process_rss_bytes(os.getpid())
    ready = reporting.canonical_json_bytes(
        {
            "baseline_rss_bytes": baseline_rss_bytes,
            "worker_process_id": os.getpid(),
        }
    )
    ready_temporary = arguments.ready_file.with_name(f".{arguments.ready_file.name}.tmp")
    ready_temporary.write_bytes(ready)
    os.replace(ready_temporary, arguments.ready_file)
    go_deadline = time.perf_counter_ns() + 30_000_000_000
    _await_go_handshake(arguments.go_file, deadline_ns=go_deadline)
    result = _execute_prepared_worker_case(
        case,
        baseline_rss_bytes=baseline_rss_bytes,
    )
    sys.stdout.buffer.write(reporting.canonical_json_bytes(result.to_object()))
    return 0


if __name__ == "__main__":
    sys.exit(_worker_main(sys.argv[1:]))
