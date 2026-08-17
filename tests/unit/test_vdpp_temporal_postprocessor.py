"""Bounded pinned-VDPP recurrence adapter tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.depth_surge_3d._vendor.vdpp.vdpp_model import VDPP
from src.depth_surge_3d.inference.depth.vdpp_temporal_postprocessor import (
    VDPPTemporalPostprocessor,
)


class _FakeModel:
    def __init__(self, mode: str = "marker") -> None:
        self.mode = mode
        self.calls: list[tuple[int, int, int]] = []

    def __call__(self, frames: torch.Tensor, *, downsize: bool) -> torch.Tensor:
        assert downsize is True
        self.calls.append((frames.shape[1], frames.shape[2], frames.shape[3]))
        if self.mode == "constant":
            return torch.ones_like(frames)
        local = torch.arange(frames.shape[1], device=frames.device, dtype=frames.dtype).view(
            1, -1, 1, 1
        )
        return frames * torch.tensor(1.25, dtype=frames.dtype, device=frames.device) + local / 50


class _UpstreamFake:
    infer_overlap_size = 4

    def __init__(self, model: _FakeModel) -> None:
        self.model = model

    def __call__(self, frames: torch.Tensor, *, downsize: bool) -> torch.Tensor:
        return self.model(frames, downsize=downsize)


def _source(count: int, shape: tuple[int, int] = (15, 17)) -> np.ndarray:
    height, width = shape
    rows = np.linspace(0.0, 0.4, height, dtype=np.float32)[:, None]
    cols = np.linspace(0.0, 0.3, width, dtype=np.float32)[None, :]
    return np.stack([rows + cols + index / 500 for index in range(count)]).astype(np.float32)


@pytest.mark.parametrize(
    ("length", "requests"),
    [
        (1, [(0, 1)]),
        (25, [(0, 25)]),
        (31, [(0, 31)]),
        (32, [(0, 32)]),
        (33, [(0, 32), (28, 33)]),
        (60, [(0, 32), (28, 60)]),
        (61, [(0, 32), (28, 60), (56, 61)]),
        (100, [(0, 32), (28, 60), (56, 88), (84, 100)]),
    ],
)
def test_fixed_windows_have_no_tail_padding_and_emit_each_frame_once(
    length: int,
    requests: list[tuple[int, int]],
) -> None:
    source = _source(length, (14, 14))
    observed: list[tuple[int, int]] = []
    adapter = VDPPTemporalPostprocessor(model=_FakeModel(), device="cpu")

    outputs = list(
        adapter.process_shot(
            length,
            lambda start, end: observed.append((start, end)) or source[start:end].copy(),
        )
    )

    assert observed == requests
    assert [index for index, _values in outputs] == list(range(length))
    assert all(values.shape == (14, 14) for _index, values in outputs)
    assert adapter.retained_frame_count == 0


@pytest.mark.parametrize("length", [33, 60, 61])
def test_bounded_adapter_matches_pinned_upstream_continuation_math(length: int) -> None:
    source = _source(length)
    model = _FakeModel()
    adapter = VDPPTemporalPostprocessor(model=model, device="cpu")

    actual = np.stack(
        [values for _index, values in adapter.process_shot(length, lambda s, e: source[s:e])]
    )
    upstream_model = _FakeModel()
    expected = VDPP.infer_video_depth(
        _UpstreamFake(upstream_model),
        torch.from_numpy(source).unsqueeze(0),
        infer_frame=32,
        downsize=True,
    ).squeeze(0)

    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-6)


def test_zero_determinant_keeps_upstream_zero_scale_and_shift() -> None:
    source = np.ones((33, 14, 14), dtype=np.float32)
    adapter = VDPPTemporalPostprocessor(model=_FakeModel("constant"), device="cpu")

    output = np.stack(
        [values for _index, values in adapter.process_shot(33, lambda s, e: source[s:e])]
    )

    np.testing.assert_array_equal(output[:32], np.ones_like(output[:32]))
    np.testing.assert_array_equal(output[32], np.zeros_like(output[32]))


def test_loader_failure_clears_retained_device_state() -> None:
    source = _source(33, (14, 14))
    adapter = VDPPTemporalPostprocessor(model=_FakeModel(), device="cpu")

    def loader(start: int, end: int) -> np.ndarray:
        if start:
            raise OSError("decode failed")
        return source[start:end]

    with pytest.raises(OSError, match="decode failed"):
        list(adapter.process_shot(33, loader))

    assert adapter.retained_frame_count == 0


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (lambda data: data[:2], "exactly"),
        (lambda data: data.astype(np.float64), "float32"),
        (lambda data: data[:, :, :, None], "shape"),
        (lambda data: np.full_like(data, np.nan), "finite"),
    ],
)
def test_loader_contract_is_strict(bad, message: str) -> None:
    source = _source(3, (14, 14))
    adapter = VDPPTemporalPostprocessor(model=_FakeModel(), device="cpu")

    with pytest.raises((TypeError, ValueError), match=message):
        list(adapter.process_shot(3, lambda _start, _end: bad(source)))


def test_execution_plan_comes_from_the_same_fixed_forward_contract() -> None:
    adapter = VDPPTemporalPostprocessor(model=_FakeModel(), device="cpu")

    plan = adapter.execution_plan((15, 17))

    assert plan["window_size"] == 32
    assert plan["overlap"] == 4
    assert plan["stride"] == 28
    assert plan["downsize"] is True
    assert plan["precision"] == "fp32"
    assert plan["tail_padding"] is False
    assert plan["padded_input_shape"] == [28, 28]
    assert plan["working_shape"] == [224, 224]


def test_checkpoint_load_is_strict_and_never_uses_unsafe_pickle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "vdpp.pth"
    checkpoint.write_bytes(b"fixture")
    calls: list[dict] = []

    class Loadable(_FakeModel):
        def load_state_dict(self, state, *, strict: bool):
            assert state == {"weight": torch.tensor(1)}
            assert strict is True

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    def fake_load(path, **kwargs):
        calls.append({"path": path, **kwargs})
        return {"weight": torch.tensor(1)}

    monkeypatch.setattr(torch, "load", fake_load)
    adapter = VDPPTemporalPostprocessor(
        checkpoint_path=checkpoint,
        device="cuda",
        model_factory=Loadable,
        allow_unavailable_cuda_for_tests=True,
    )

    assert calls == [{"path": checkpoint, "map_location": "cpu", "weights_only": True}]
    adapter.release()
    adapter.release()
