"""Unit and real-FFmpeg tests for zero-PNG VideoFrameSource streaming decode (P4)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.io.operations import (
    get_video_properties,
    save_processing_settings,
    verify_ffmpeg_installation,
)
from src.depth_surge_3d.io.resume import build_resume_report
from src.depth_surge_3d.processing.frames.depth_processor import DepthMapProcessor
from src.depth_surge_3d.processing.frames.scene_analyzer import (
    _luma_histogram,
    analyze_scenes_from_histograms,
)
from src.depth_surge_3d.processing.frames.stereo_generator import StereoPairGenerator
from src.depth_surge_3d.processing.frames.video_frame_source import build_video_frame_source
from src.depth_surge_3d.processing.orchestration.pipeline_orchestrator import ProcessingOrchestrator
from src.depth_surge_3d.processing.video.video_encoder import VideoEncoder
from src.depth_surge_3d.inference.depth.types import DepthBatch, DepthRepresentation


def _make_synthetic_mp4(path: Path, *, num_frames: int = 16, width: int = 64, height: int = 48):
    """Create a deterministic synthetic MP4 with a hard scene cut at frame 8."""

    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        "24",
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "12",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for idx in range(num_frames):
        val = 30 if idx < 8 else 220
        frame = np.full((height, width, 3), [val, val, val], dtype=np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    rc = proc.wait(timeout=15.0)
    assert rc == 0, proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""


class _DummyDepthEstimator:
    model_name = "dummy-stream-v1"
    model_version = "1"
    exact_input_batch_reuse = True

    def estimate_depth_batch(self, frames: np.ndarray, **_kwargs) -> DepthBatch:
        count, height, width = frames.shape[:3]
        values = np.linspace(0.2, 0.8, height * width, dtype=np.float32).reshape(1, height, width)
        batch = np.repeat(values, count, axis=0)
        return DepthBatch(values=batch, representation=DepthRepresentation.RELATIVE_DEPTH)


@pytest.mark.skipif(not verify_ffmpeg_installation(), reason="FFmpeg required")
def test_video_frame_source_zero_png_parity_concurrency_and_scene_cuts(tmp_path: Path) -> None:
    video = tmp_path / "cut_clip.mp4"
    _make_synthetic_mp4(video, num_frames=16, width=64, height=48)
    props = get_video_properties(str(video))
    settings = {"scene_detection": True, "scene_cut_threshold": 0.55, "min_scene_frames": 4}
    frames_dir = tmp_path / "00_original_frames"

    source = build_video_frame_source(
        video,
        frames_dir,
        props,
        settings,
        compute_histograms=True,
    )
    assert source is not None
    assert source.frame_count == 16
    assert (source.width, source.height) == (64, 48)
    assert not frames_dir.exists() or list(frames_dir.glob("*.png")) == []

    # Scene cuts from precomputed streaming histograms match PNG scene analysis!
    stream_scene_dir = tmp_path / "scene_stream"
    stream_manifest = analyze_scenes_from_histograms(
        source.frame_names,
        source.luma_histograms,
        stream_scene_dir,
        enabled=True,
        threshold=0.55,
        min_frames=4,
    )
    assert stream_manifest["candidate_cuts"] == [8]

    # Sliding-window overlap (V2 pattern) + backward restart beyond window
    with source.open_reader(window_size=6, consume_once=False) as reader:
        f0 = reader.get_frame(0)
        f7 = reader.get_frame(7)
        f8 = reader.get_frame(8)
        assert f0.shape == (48, 64, 3) and f0.dtype == np.uint8
        assert float(f7.mean()) < 90.0
        assert float(f8.mean()) > 170.0
        # Within sliding window (8 - 6 = 2 -> index 4 is cached)
        f4 = reader.get_frame(4)
        assert float(f4.mean()) < 90.0
        # Behind sliding window (index 0 evicted -> clean FFmpeg restart)
        f0_again = reader.get_frame(0)
        np.testing.assert_array_equal(f0_again, f0)

    # Concurrent out-of-order consume_once (StereoPairGenerator 10-worker pattern)
    with source.open_reader(window_size=16, consume_once=True) as reader:
        order = [2, 0, 1, 5, 3, 4, 9, 6, 8, 7, 11, 10, 15, 12, 13, 14]
        with ThreadPoolExecutor(max_workers=4) as pool:
            fetched = list(pool.map(reader.get_frame, order))
        by_idx = dict(zip(order, fetched, strict=True))
        for idx in range(8):
            assert float(by_idx[idx].mean()) < 90.0
        for idx in range(8, 16):
            assert float(by_idx[idx].mean()) > 170.0

    with source.open_reader(window_size=2, consume_once=True) as reader:
        np.testing.assert_array_equal(reader.get_frame(15), by_idx[15])
        assert len(reader._cache) <= 2
        np.testing.assert_array_equal(reader.get_frame(0), by_idx[0])

    # Existing PNGs disable streaming source so resume reuses disk frames
    frames_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frames_dir / "frame_000001.png"), np.zeros((48, 64, 3), dtype=np.uint8))
    encoder = VideoEncoder()
    assert (
        encoder.prepare_streaming_frame_source(
            str(video), {"base": tmp_path, "frames": frames_dir}, props, settings
        )
        is None
    )


@pytest.mark.skipif(not verify_ffmpeg_installation(), reason="FFmpeg required")
@pytest.mark.parametrize("bt709_tags", [False, True])
def test_streaming_histograms_match_extracted_pngs_at_bin_edges(
    tmp_path: Path, bt709_tags: bool
) -> None:
    video = tmp_path / "luma.mp4"
    pixels = 64 * 48
    chroma = bytes([128]) * (pixels // 2)
    samples = b"".join((bytes([y]) * pixels + chroma) * 8 for y in (106, 108))
    samples += np.random.default_rng(7).integers(0, 256, pixels * 3 // 2, dtype=np.uint8).tobytes()
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            "64x48",
            "-r",
            "24",
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-colorspace",
            "smpte170m",
            "-color_trc",
            "smpte170m",
            "-color_primaries",
            "smpte170m",
            *(
                [
                    "-bsf:v",
                    "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
                ]
                if bt709_tags
                else []
            ),
            str(video),
        ],
        input=samples,
        check=True,
    )
    props = get_video_properties(str(video))
    source = build_video_frame_source(video, tmp_path / "stream", props, {})
    files = VideoEncoder().extract_frames(str(video), {"base": tmp_path}, props, {})
    assert source is not None and source.luma_histograms is not None
    with source.open_reader() as reader:
        for index in (0, 8, 16):
            np.testing.assert_array_equal(
                source.luma_histograms[index], _luma_histogram(files[index])
            )
            np.testing.assert_array_equal(reader.get_frame(index), cv2.imread(str(files[index])))


@pytest.mark.skipif(not verify_ffmpeg_installation(), reason="FFmpeg required")
def test_depth_and_stereo_stages_run_end_to_end_with_zero_source_pngs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "e2e_clip.mp4"
    _make_synthetic_mp4(video, num_frames=10, width=64, height=48)
    props = get_video_properties(str(video))
    job_root = tmp_path / "job"
    frames_dir = job_root / "00_original_frames"
    dirs = {
        "base": job_root,
        "frames": frames_dir,
        "depth_raw": job_root / "02_depth_raw",
        "disparity_maps": job_root / "03_disparity_maps",
        "scene_analysis": job_root / "01_scene_analysis",
        "metric_geometry": job_root / "03_metric_geometry",
        "left_frames": job_root / "04_left_frames",
        "right_frames": job_root / "04_right_frames",
    }
    settings = {
        "keep_intermediates": False,
        "scene_detection": True,
        "scene_cut_threshold": 0.55,
        "min_scene_frames": 4,
        "stereo_geometry_mode": "relative",
        "stereo_render_mode": "fast",
        "occlusion_fill": "background",
        "stereo_strength": 2.0,
        "convergence": 0.5,
    }
    encoder = VideoEncoder()
    source = encoder.prepare_streaming_frame_source(str(video), dirs, props, settings)
    assert source is not None
    virtual_files = source.virtual_frame_files()

    depth_proc = DepthMapProcessor(_DummyDepthEstimator())
    disp_files = depth_proc.generate_depth_map_files(
        virtual_files,
        settings,
        dirs,
        progress_tracker=None,
        frame_source=source,
    )
    assert disp_files is not None and len(disp_files) == 10
    assert not frames_dir.exists() or list(frames_dir.glob("*.png")) == []

    save_processing_settings(job_root, "stream", settings, props, str(video), transactional_v5=True)
    report = build_resume_report(job_root, settings, source_video=video)
    assert report.stage("frames").disposition == "preserve"
    assert report.stage("scene_data").disposition == "preserve"
    assert report.stage("disparity_maps").disposition == "preserve"
    assert not report.migration_paths

    seen: list[str] = []
    stereo_gen = StereoPairGenerator()
    resume = ProcessingOrchestrator(
        None,
        stereo_gen,
        None,
        None,
        None,
        encoder,
        preselected_depth_files=disp_files,
        preselected_render_source="base",
    )

    def render_pairs(directories, settings, manifest, *_args):
        assert manifest.render_shape == (48, 64)
        assert manifest.frame_source.source_fingerprint == source.source_fingerprint
        return stereo_gen.create_stereo_pairs_from_manifest(
            manifest,
            directories,
            settings,
            frame_consumer=lambda name, l_img, r_img: seen.append(name),
        )

    monkeypatch.setattr(resume, "_execute_remaining_steps", render_pairs)
    monkeypatch.setattr(f"{source.__module__}._scan_video_stream", lambda *args, **kwargs: None)
    assert resume._execute_pipeline(str(video), job_root, dirs, props, settings)
    assert len(seen) == 10
    assert not frames_dir.exists() or list(frames_dir.glob("*.png")) == []
