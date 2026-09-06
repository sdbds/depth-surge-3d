"""Scene-index traversal budgets and legacy sample-order equivalence."""

from collections.abc import Sequence

import numpy as np
import pytest

from src.depth_surge_3d.processing.frames import metric_geometry, scene_analyzer


class _CountedSceneIds(Sequence):
    def __init__(self, values):
        self.values = values
        self.visits = 0

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        value = self.values[index]
        self.visits += 1
        return value


def _legacy_scene_selection(scene_ids):
    selected = {}
    for scene_id in dict.fromkeys(scene_ids):
        indexes = [index for index, value in enumerate(scene_ids) if value == scene_id]
        count = min(len(indexes), 32)
        positions = np.unique(np.rint(np.linspace(0, len(indexes) - 1, count)).astype(np.int64))
        selected[scene_id] = [indexes[int(position)] for position in positions]
    return selected


def _legacy_convergence_selection(scene_ids):
    selected = []
    scene_array = np.asarray(scene_ids)
    for scene_id in dict.fromkeys(int(value) for value in scene_ids):
        indexes = np.flatnonzero(scene_array == scene_id)
        if len(indexes) > 32:
            positions = np.unique(np.rint(np.linspace(0, len(indexes) - 1, 32)).astype(np.int64))
            indexes = indexes[positions]
        selected.extend(int(value) for value in indexes)
    return tuple(sorted(selected))


def test_scene_selection_visits_source_ids_once_with_many_interleaved_scenes():
    select = getattr(scene_analyzer, "select_scene_frame_indexes", None)
    assert callable(select), "Scene sampling needs a shared linear-time index selector"
    scene_ids = _CountedSceneIds(list(range(200)) * 40 + [203])

    selected = select(scene_ids)

    assert selected == _legacy_scene_selection(scene_ids.values)
    assert list(selected) == list(range(200)) + [203]
    assert scene_ids.visits <= len(scene_ids)


@pytest.mark.parametrize(
    "scene_ids, expected",
    [([], {}), ([9], {9: [0]}), ([8, 3, 8, -1, 3], {8: [0, 2], 3: [1, 4], -1: [3]})],
)
def test_scene_selection_keeps_first_scene_order_and_noncontiguous_members(scene_ids, expected):
    select = getattr(scene_analyzer, "select_scene_frame_indexes", None)
    assert callable(select), "Scene sampling needs a shared linear-time index selector"
    actual = select(scene_ids)
    assert actual == expected
    assert list(actual) == list(expected)


@pytest.mark.parametrize("length", [31, 32, 33, 63, 64, 65, 127, 100_003])
def test_scene_selection_matches_legacy_sample32_rounding_and_tail(length):
    select = getattr(scene_analyzer, "select_scene_frame_indexes", None)
    assert callable(select), "Scene sampling needs a shared linear-time index selector"
    scene_ids = [7] * length + [2, 7, 2, 99]
    assert select(scene_ids) == _legacy_scene_selection(scene_ids)


def test_convergence_selection_does_not_scan_whole_clip_once_per_scene(monkeypatch):
    scene_ids = list(range(200)) * 40 + [203]
    original = np.flatnonzero
    examined = 0

    def counted_flatnonzero(values):
        nonlocal examined
        examined += np.size(values)
        return original(values)

    expected = _legacy_convergence_selection(scene_ids)
    monkeypatch.setattr(metric_geometry.np, "flatnonzero", counted_flatnonzero)

    actual = metric_geometry.select_convergence_frame_indexes(scene_ids)

    assert actual == expected
    assert examined <= 2 * len(scene_ids)


@pytest.mark.parametrize(
    "scene_ids",
    [
        [],
        [5, 1, 5, -1],
        [8] * 33 + [3] * 65 + [8, 99],
        np.array([3, 1, 3], dtype=np.int8),
        np.array([2**64 - 1, 0, 2**64 - 1], dtype=np.uint64),
        [True, False, True],
        [1.0, 1.5, 2.0, 2.5],
        ["1", "2", "1"],
        np.array(["1", 1, 2.5, 2], dtype=object),
        np.array([2**64, 2**64 + 1, 2**64], dtype=object),
    ],
)
def test_convergence_selection_preserves_typed_sequence_matching(scene_ids):
    assert metric_geometry.select_convergence_frame_indexes(scene_ids) == (
        _legacy_convergence_selection(scene_ids)
    )


@pytest.mark.parametrize("scene_ids", [[float("nan")], [float("inf")], ["scene"], [None]])
def test_convergence_selection_preserves_conversion_errors(scene_ids):
    with pytest.raises((TypeError, ValueError, OverflowError)) as legacy_error:
        _legacy_convergence_selection(scene_ids)
    with pytest.raises(type(legacy_error.value)):
        metric_geometry.select_convergence_frame_indexes(scene_ids)
