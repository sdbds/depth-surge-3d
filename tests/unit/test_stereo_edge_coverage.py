"""Independent acceptance gates for horizontal subpixel edge coverage."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from src.depth_surge_3d.rendering.stereo_renderer import (
    StereoRenderer,
    StereoRenderSettings,
)
from tests.unit.stereo_edge_reference import (
    SAMPLES,
    render_continuous_reference,
    render_discrete_reference,
)


FRAME_SHA256 = "f99eddf3c8bd24dd14d3fb9bb4f3da11fd4f19cc992f7677c2858743f967c758"
CANONICAL_SHA256 = "5a46810d7489422d11bb6262e64ef40f6427b233db7b8ff3f0775b57e266da65"
V1_NONE_METRICS = {
    (0.5, 1): (0.019099491648511255, 0.22352941176470587, 0.0011149557862360634),
    (0.5, -1): (0.03166303558460421, 0.2313725490196078, 0.0),
    (1.25, 1): (0.08155410312273058, 0.5607843137254902, 0.0028066128412149167),
    (1.25, -1): (0.10319535221496005, 0.580392156862745, 0.0),
    (3.0, 1): (0.09687726942628903, 0.47058823529411764, 0.0023562366122919756),
    (3.0, -1): (0.10806100217864924, 0.48627450980392156, 0.0),
}


def _procedural_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width, height = 256, 128
    background = np.array([224, 232, 240], dtype=np.uint8)
    foreground_colour = np.array([24, 32, 48], dtype=np.uint8)
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = background
    rows = np.arange(height)[:, None]
    columns = np.arange(width)[None, :]
    foreground = columns < 48 + (3 * rows) // 5
    line = columns == 180 - rows // 4
    frame[foreground | line] = foreground_colour
    canonical = np.full((height, width), 0.1, dtype=np.float32)
    canonical[foreground | line] = np.float32(0.9)

    edge_mask = np.zeros((height, width), dtype=np.bool_)
    for row in range(height):
        for centre in (48 + (3 * row) // 5, 180 - row // 4):
            edge_mask[row, max(0, centre - 4) : min(width, centre + 5)] = True
    return frame, canonical, edge_mask


def _eye_arrays(result, eye_sign: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = "left" if eye_sign == 1 else "right"
    return (
        getattr(result, f"{prefix}_image"),
        getattr(result, f"{prefix}_valid_mask"),
        getattr(result, f"{prefix}_hole_mask"),
    )


def _errors(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.abs(actual.astype(np.float64) - expected.astype(np.float64)) / 255.0


def test_procedural_fixture_bytes_are_frozen() -> None:
    frame, canonical, _edge_mask = _procedural_fixture()

    assert hashlib.sha256(frame.tobytes()).hexdigest() == FRAME_SHA256
    assert hashlib.sha256(canonical.tobytes()).hexdigest() == CANONICAL_SHA256


@pytest.mark.parametrize("strength", [0.5, 1.25, 3.0])
@pytest.mark.parametrize("fill", ["none", "background"])
@pytest.mark.parametrize("eye_sign", [1, -1])
def test_production_matches_independent_discrete_oracle(
    strength: float,
    fill: str,
    eye_sign: int,
) -> None:
    frame, canonical, _edge_mask = _procedural_fixture()
    settings = StereoRenderSettings(
        stereo_strength=strength,
        convergence=0.5,
        occlusion_fill=fill,
    )

    result = StereoRenderer(device="cpu").render(frame, canonical, settings)
    actual_image, actual_valid, actual_hole = _eye_arrays(result, eye_sign)
    reference = render_discrete_reference(
        frame,
        canonical,
        strength=strength,
        convergence=0.5,
        eye_sign=eye_sign,
        occlusion_fill=fill,
    )

    assert np.array_equal(actual_image, reference.image)
    assert np.array_equal(actual_valid, reference.valid_mask)
    assert np.array_equal(actual_hole, reference.hole_mask)
    if fill == "background":
        assert reference.fine_valid.all()
        lanes = reference.fine_colour.reshape(128, 256, SAMPLES, 3)
        assert np.all(actual_image.astype(np.int16) >= lanes.min(axis=2) - 1)
        assert np.all(actual_image.astype(np.int16) <= lanes.max(axis=2) + 1)


@pytest.mark.parametrize("strength", [0.5, 1.25, 3.0])
@pytest.mark.parametrize("eye_sign", [1, -1])
def test_none_mode_beats_frozen_v1_boundary_metrics(
    strength: float,
    eye_sign: int,
) -> None:
    frame, canonical, edge_mask = _procedural_fixture()
    settings = StereoRenderSettings(
        stereo_strength=strength,
        convergence=0.5,
        occlusion_fill="none",
    )
    result = StereoRenderer(device="cpu").render(frame, canonical, settings)
    actual, _valid, _hole = _eye_arrays(result, eye_sign)
    continuous = render_continuous_reference(
        frame,
        canonical,
        strength=strength,
        convergence=0.5,
        eye_sign=eye_sign,
    )
    error = _errors(actual, continuous)
    edge_error = error[edge_mask].reshape(-1)
    outside_error = error[~edge_mask].reshape(-1)
    mae = float(edge_error.mean())
    p95 = float(np.percentile(edge_error, 95, method="linear"))
    outside_mae = float(outside_error.mean())
    v1_mae, v1_p95, v1_outside = V1_NONE_METRICS[(strength, eye_sign)]

    assert mae <= 0.60 * v1_mae
    assert p95 <= 0.75 * v1_p95
    assert outside_mae <= v1_outside + 1.0 / 255.0

    foreground = np.array([24, 32, 48], dtype=np.uint8)
    background = np.array([224, 232, 240], dtype=np.uint8)
    solid = np.all(continuous == foreground, axis=2) | np.all(continuous == background, axis=2)
    assert np.max(np.abs(actual[solid].astype(np.int16) - continuous[solid].astype(np.int16))) <= 1


def test_continuous_oracle_integrates_half_pixel_translation() -> None:
    frame = np.full((1, 20, 3), 100, dtype=np.uint8)
    canonical = np.ones((1, 20), dtype=np.float32)

    continuous = render_continuous_reference(
        frame,
        canonical,
        strength=5.0,
        convergence=0.0,
        eye_sign=1,
    )

    assert continuous[0, 0, 0] == 50
    assert np.all(continuous[0, 1:, 0] == 100)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("strength", [0.5, 1.25, 3.0])
@pytest.mark.parametrize("fill", ["none", "background"])
def test_complete_fixed_uint8_corpus_is_repeatable_and_cpu_cuda_exact(
    strength: float,
    fill: str,
) -> None:
    frame, canonical, _edge_mask = _procedural_fixture()
    settings = StereoRenderSettings(
        stereo_strength=strength,
        convergence=0.5,
        occlusion_fill=fill,
    )

    cpu = StereoRenderer(device="cpu").render(frame, canonical, settings)
    cuda_first = StereoRenderer(device="cuda").render(frame, canonical, settings)
    cuda_second = StereoRenderer(device="cuda").render(frame, canonical, settings)

    for field in cpu.__dataclass_fields__:
        expected = getattr(cpu, field)
        assert np.array_equal(getattr(cuda_first, field), expected)
        assert np.array_equal(getattr(cuda_second, field), expected)
