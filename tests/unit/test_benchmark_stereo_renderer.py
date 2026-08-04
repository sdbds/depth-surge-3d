"""Smoke coverage for the non-gating stereo renderer benchmark."""

from scripts.benchmark_stereo_renderer import benchmark_resolution


def test_benchmark_smoke_reports_renderer_and_pipeline_metrics(tmp_path) -> None:
    result = benchmark_resolution(
        width=32,
        height=18,
        frame_count=1,
        device="cpu",
        workers=1,
        workspace=tmp_path,
    )

    assert result["resolution"] == "32x18"
    assert result["frames"] == 1
    assert result["renderer_wall_seconds"] > 0.0
    assert result["renderer_fps"] > 0.0
    assert result["pipeline_wall_seconds"] > 0.0
    assert result["pipeline_fps"] > 0.0
    assert result["gpu_render_ms_per_frame"] is None
    assert result["peak_cuda_bytes"] == 0
    assert 0.0 <= result["writer_utilization_percent"] <= 100.0
    assert result["queue_wait_seconds"] >= 0.0
    assert result["permit_wait_seconds"] >= 0.0
