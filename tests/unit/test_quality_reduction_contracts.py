"""A persisted aggregate must never silently combine unlike frame diagnostics."""

from dataclasses import replace

import numpy as np
import pytest

from src.depth_surge_3d.rendering.quality import diagnostics, renderer, types
from src.depth_surge_3d.rendering.stereo_renderer import StereoRenderer, StereoRenderSettings


@pytest.fixture(scope="module")
def compact_frame():
    return renderer.render_relative_quality_compact(
        renderer=StereoRenderer(device="cpu"),
        frame=np.full((2, 32, 3), (1, 81, 163), np.uint8),
        canonical=np.ones((1, 8), np.float32),
        settings=StereoRenderSettings(
            stereo_render_mode="quality",
            stereo_strength=5,
            convergence=0,
            occlusion_fill="background",
        ),
    )


def frame_stats(compact, name):
    return diagnostics.build_quality_frame_stats(
        compact, frame_name=name, geometry_mode="relative", occlusion_fill="background"
    )


def summarize(frames, **options):
    args = dict(
        frame_names=("A", "B"),
        render_shape=(2, 32),
        render_mode="quality",
        geometry_mode="relative",
        occlusion_fill="background",
        ordered_frame_manifest_fingerprint="a" * 64,
        frames_jsonl_sha256="b" * 64,
    )
    args.update(options)
    return diagnostics.build_diagnostics_summary(iter(frames), **args)


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("mixed-graph", "component availability"),
        ("mixed-limits", "limit availability"),
        ("mixed-budgets", "budget availability"),
        ("invalid-budget", "budget availability"),
        ("dense-width", "dense histogram width"),
        ("bucket", "bucket"),
        ("unattested-nearest", "geometry-nearest"),
        ("unexpected-metric", "relative diagnostics"),
        ("missing-metric", "metric diagnostics"),
    ],
)
def test_failed_frame_reduction_is_explicit_and_retry_with_valid_frames_is_exact(
    compact_frame, fault, reason
):
    first, second = (frame_stats(compact_frame, name) for name in ("A", "B"))
    eye = second["eyes"]["right"]
    options = {}
    if fault == "mixed-graph":
        eye["backend_component_counts"]["availability"] = "unavailable_fast_no_segment_graph"
    elif fault == "mixed-limits":
        eye["quality_limits"] = None
    elif fault == "mixed-budgets":
        eye["quality_budgets"]["availability"] = "unavailable_quality_none_no_repair_plan"
    elif fault == "invalid-budget":
        for frame in (first, second):
            frame["eyes"]["right"]["quality_budgets"]["availability"] = "unknown"
    elif fault == "dense-width":
        eye["hole_runs"]["lane_count_histogram"] = types.DenseHistogramPairs(
            np.zeros(16, np.uint64)
        )
    elif fault == "bucket":
        eye["hole_runs"]["lane_count_histogram"] = [[-1, 1]]
    elif fault == "unattested-nearest":
        second["geometry_nearest"] = None
    elif fault == "unexpected-metric":
        second["metric_projection"] = dict(
            valid_pixel_count=64, clamped_pixel_count=0, clamped_fraction=0.0
        )
    else:
        options["geometry_mode"] = "metric"
    with pytest.raises(ValueError, match=reason):
        summarize((first, second), **options)
    clean = [frame_stats(compact_frame, name) for name in ("A", "B")]
    retried = summarize(clean)
    assert retried["eyes"]["right"]["pixel_count"] == 128
    assert (
        retried["eyes"]["right"]["backend_lane_counts"]["fallback"]
        == 2 * clean[0]["eyes"]["right"]["backend_lane_counts"]["fallback"]
    )


@pytest.mark.parametrize(
    ("names", "frames", "options", "reason"),
    [
        ((), (), {}, "nonempty"),
        (("A",), ("A", "B"), {}, "exceed"),
        (("A", "B"), ("A",), {}, "ended"),
        (("A",), ("A",), {"frame_name_replay": lambda: iter(("A",))}, "exactly one"),
        (None, ("A",), {"frame_name_replay": lambda: iter(("A",)), "frame_count": 0}, "positive"),
        (None, ("A",), {"frame_name_replay": lambda: iter(("A",)), "frame_count": 2}, "ended"),
        (("A",), ("A",), {"render_shape": (0, 32)}, "dimensions"),
    ],
)
def test_reducer_rejects_missing_duplicate_or_inconsistent_sequence_authority(
    compact_frame, names, frames, options, reason
):
    with pytest.raises(ValueError, match=reason):
        summarize(
            (frame_stats(compact_frame, name) for name in frames), frame_names=names, **options
        )


def test_frame_stats_require_attested_fast_counters_and_matching_streamed_width(compact_frame):
    args = dict(frame_name="frame", geometry_mode="relative", occlusion_fill="background")
    with pytest.raises(ValueError, match="producer-attested"):
        diagnostics.build_quality_frame_stats(compact_frame, **args, render_mode="fast")
    with pytest.raises(ValueError, match="render_mode"):
        diagnostics.build_quality_frame_stats(compact_frame, **args, render_mode="codec")
    with pytest.raises(ValueError, match="geometry_mode"):
        diagnostics.build_quality_frame_stats(compact_frame, **{**args, "geometry_mode": "camera"})
    incompatible = types.HoleRunDiagnostics(np.zeros(64, np.uint64), np.zeros(4, np.uint64))
    with pytest.raises(ValueError, match="histogram width"):
        diagnostics.build_quality_frame_stats(
            replace(compact_frame, right_hole_runs=incompatible), **args
        )
    stats = diagnostics.build_quality_frame_stats(
        replace(compact_frame, geometry_nearest=None), **args
    )
    assert stats["geometry_nearest"] == {
        "indexed_sample_count": 0,
        "query_count": 0,
        "visited_nodes_total": 0,
        "visited_nodes_max": 0,
    }
