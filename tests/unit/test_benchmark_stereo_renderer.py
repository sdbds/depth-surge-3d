"""Smoke coverage for the stereo renderer review benchmark."""

from scripts.benchmark_stereo_renderer import _synthetic_inputs, benchmark_resolution


def test_collision_fixture_contains_sharp_depth_boundaries() -> None:
    frame, canonical = _synthetic_inputs(32, 18, "collision")

    assert frame.shape == (18, 32, 3)
    assert canonical.shape == (18, 32)
    assert len(set(canonical.reshape(-1).tolist())) >= 3
    assert (canonical[:, 1:] != canonical[:, :-1]).any()


def test_benchmark_smoke_reports_review_schema_and_pipeline_metrics(tmp_path) -> None:
    result = benchmark_resolution(
        width=32,
        height=18,
        frame_count=1,
        warmup_count=1,
        fixture="collision",
        device="cpu",
        workers=1,
        workspace=tmp_path,
    )

    assert result["resolution"] == "32x18"
    assert result["fixture"] == "collision"
    assert result["warmup_frames"] == 1
    assert result["measured_frames"] == 1
    assert len(result["git_commit"]) == 40
    assert isinstance(result["git_dirty"], bool)
    assert result["git_diff_sha256"] is None or len(result["git_diff_sha256"]) == 64
    assert len(result["benchmark_harness_sha256"]) == 64
    assert result["algorithm_version"] == "torch-horizontal-16x-zbuffer-v3"
    assert result["horizontal_samples"] == 16
    assert result["settings"] == {
        "convergence": 0.5,
        "occlusion_fill": "background",
        "stereo_strength": 2.0,
    }
    assert result["python_version"]
    assert result["torch_version"]
    assert result["renderer_latency_median_seconds"] > 0.0
    assert result["renderer_latency_p95_seconds"] > 0.0
    assert result["renderer_mean_ms_per_frame"] > 0.0
    assert "gpu_render_ms_per_frame" not in result
    assert result["renderer_fps"] > 0.0
    assert result["host_geometry_seconds_median"] >= 0.0
    assert result["host_geometry_bytes_per_frame"] == 2 * 32 * 18 * 4
    assert result["offset_transfer_seconds_median"] >= 0.0
    assert result["offset_transfer_bytes_per_frame"] == 2 * 32 * 18 * 4
    assert result["pipeline_wall_seconds"] > 0.0
    assert result["pipeline_fps"] > 0.0
    assert result["peak_cuda_allocated_bytes"] == 0
    assert result["peak_cuda_reserved_bytes"] == 0
    assert 0.0 <= result["writer_utilization_percent"] <= 100.0
    assert result["queue_wait_seconds"] >= 0.0
    assert result["permit_wait_seconds"] >= 0.0
