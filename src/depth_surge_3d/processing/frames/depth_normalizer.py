"""Stateless conversion from estimator depth to canonical relative disparity."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ...inference.depth.types import DepthBatch, DepthRepresentation


@dataclass(frozen=True)
class SceneDepthBounds:
    """Fixed disparity-score percentiles for one finalized scene."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError("Scene depth bounds must be finite")
        if self.low > self.high:
            raise ValueError("Scene depth bound low must not exceed high")


def depth_to_score(
    values: np.ndarray, representation: DepthRepresentation
) -> tuple[np.ndarray, np.ndarray]:
    """Return a near-increasing score and its finite/physical validity mask."""

    source = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(source)
    score = np.zeros(source.shape, dtype=np.float32)

    if representation is DepthRepresentation.METRIC_DEPTH:
        valid = finite & (source > 0.0)
        np.divide(np.float32(1.0), source, out=score, where=valid)
    elif representation is DepthRepresentation.INVERSE_DEPTH:
        valid = finite
        score[valid] = source[valid]
    elif representation is DepthRepresentation.RELATIVE_DEPTH:
        valid = finite
        score[valid] = -source[valid]
    else:
        raise TypeError(f"Unsupported depth representation: {representation!r}")

    return score, valid


def canonicalize_depth(
    values: np.ndarray,
    representation: DepthRepresentation,
    bounds: SceneDepthBounds,
) -> np.ndarray:
    """Map raw values through immutable scene bounds to `0=far, 1=near`."""

    score, valid = depth_to_score(values, representation)
    canonical = np.full(score.shape, np.float32(0.5), dtype=np.float32)
    if bounds.low == bounds.high:
        return canonical

    scaled = (score[valid] - np.float32(bounds.low)) / np.float32(bounds.high - bounds.low)
    canonical[valid] = np.clip(scaled, 0.0, 1.0).astype(np.float32, copy=False)
    return canonical


def canonicalize_single_scene(batch: DepthBatch) -> np.ndarray:
    """Canonicalize one estimator batch as a single immutable scene."""

    if not isinstance(batch, DepthBatch):
        raise TypeError("Single-scene canonicalization requires a DepthBatch")
    score, valid = depth_to_score(batch.values, batch.representation)
    samples = score[valid]
    if samples.size == 0:
        bounds = SceneDepthBounds(0.0, 0.0)
    else:
        bounds = SceneDepthBounds(
            float(np.percentile(samples, 2)),
            float(np.percentile(samples, 98)),
        )
    return canonicalize_depth(batch.values, batch.representation, bounds)


def encode_canonical_png(values: np.ndarray) -> np.ndarray:
    """Encode canonical float values deterministically as uint16 pixels."""

    canonical = np.asarray(values, dtype=np.float32)
    if not np.isfinite(canonical).all():
        raise ValueError("Canonical disparity must be finite before encoding")
    return np.rint(np.clip(canonical, 0.0, 1.0) * np.float32(65535.0)).astype(np.uint16)


def decode_canonical_png(values: np.ndarray) -> np.ndarray:
    """Decode canonical uint16 pixels to float32."""

    encoded = np.asarray(values)
    if encoded.dtype != np.uint16:
        raise TypeError("Canonical disparity storage must use uint16")
    return encoded.astype(np.float32) / np.float32(65535.0)
