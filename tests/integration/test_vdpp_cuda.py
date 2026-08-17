"""Opt-in real-checkpoint smoke test for the pinned VDPP CUDA path."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.depth_surge_3d.inference.depth.vdpp_temporal_postprocessor import (
    VDPPTemporalPostprocessor,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("DEPTH_SURGE_RUN_VDPP_CUDA") != "1",
    reason="set DEPTH_SURGE_RUN_VDPP_CUDA=1 for the real checkpoint smoke test",
)


def test_real_checkpoint_processes_and_releases_one_frame() -> None:
    assert torch.cuda.is_available(), "VDPP CUDA smoke was requested but CUDA is unavailable"
    checkpoint = Path(os.environ.get("DEPTH_SURGE_VDPP_CHECKPOINT", "models/VDPP/vdpp.pth"))
    assert checkpoint.is_file(), f"VDPP CUDA smoke checkpoint is missing: {checkpoint}"
    source = np.linspace(0.0, 1.0, 32 * 48, dtype=np.float32).reshape(1, 32, 48)
    adapter = VDPPTemporalPostprocessor(checkpoint_path=checkpoint, device="cuda")
    try:
        outputs = list(adapter.process_shot(1, lambda start, end: source[start:end].copy()))
        preflight = adapter.preflight(60, (32, 48))
        torch.cuda.synchronize()
    finally:
        adapter.release()

    assert len(outputs) == 1
    assert outputs[0][0] == 0
    assert outputs[0][1].shape == (32, 48)
    assert np.isfinite(outputs[0][1]).all()
    assert preflight["preflight_window_lengths"] == [32, 32]
    assert preflight["preflight_max_memory_allocated"] > 0
    assert preflight["preflight_max_memory_reserved"] > 0
    assert adapter.retained_frame_count == 0
