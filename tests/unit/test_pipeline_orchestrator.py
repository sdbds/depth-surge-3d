"""End-to-end ProcessingOrchestrator unit tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.depth_surge_3d.core.settings import validate_settings
from src.depth_surge_3d.io.operations import load_processing_settings
from src.depth_surge_3d.processing.frames.vr_assembler import DirectStereoSource
from src.depth_surge_3d.processing.orchestration.pipeline_orchestrator import ProcessingOrchestrator


class _FakeEncoder:
    def __init__(self) -> None:
        self.streamed_frames: list[str] = []
        self.direct_called = False
        self.assembled_called = False

    def extract_frames(
        self, video_path: str, directories: dict[str, Path], _props: dict, _settings: dict
    ) -> list[Path]:
        frames_dir = directories["frames"]
        frames_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for i in (1, 2):
            p = frames_dir / f"frame_{i:06d}.png"
            cv2.imwrite(str(p), np.full((16, 16, 3), 100, dtype=np.uint8))
            out.append(p)
        return out

    def create_video_from_stereo_source(
        self, source: DirectStereoSource, output_dir: Path, *_args: Any, **_kwargs: Any
    ) -> bool:
        self.direct_called = True
        (output_dir / "clip_sbs_1080p.mp4").write_bytes(b"fake-mp4")
        return True

    def create_video(self, vr_dir: Path, output_dir: Path, *_args: Any, **_kwargs: Any) -> bool:
        self.assembled_called = True
        (output_dir / "clip_sbs_1080p.mp4").write_bytes(b"fake-mp4")
        return True

    def open_streamed_vr_session(self, *, output_dir: Path, frame_count: int, **_kwargs: Any):
        parent = self

        class _Session:
            def write(self, name: str, _frame: np.ndarray) -> None:
                parent.streamed_frames.append(name)

            def finish_and_publish(self) -> bool:
                (output_dir / "streamed.mp4").write_bytes(b"fake-streamed")
                return len(parent.streamed_frames) == frame_count

            def abort(self) -> None:
                pass

            def discard_for_file_backed_fallback(self) -> None:
                pass

        return _Session()


class _FakeDepth:
    def generate_depth_map_files(
        self, frame_files: list[Path], _settings: dict, directories: dict[str, Path], _pt=None
    ) -> list[Path]:
        disp_dir = directories["disparity_maps"]
        disp_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for f in frame_files:
            p = disp_dir / f.name
            cv2.imwrite(str(p), np.full((8, 8), 32768, dtype=np.uint16))
            out.append(p)
        (disp_dir / "metadata.json").write_text(
            json.dumps({"encoding_scale": 1.0, "fingerprint": "a" * 64}), encoding="utf-8"
        )
        return out


class _FakeStereo:
    def __init__(self) -> None:
        self.last_metric_clamp_summary = None

    def create_stereo_pairs_from_manifest(
        self, manifest, directories, _settings, _pt=None, *, frame_consumer=None, **_kw
    ) -> bool:
        if frame_consumer is not None:
            for rec in manifest.records:
                img = np.zeros((16, 16, 3), dtype=np.uint8)
                frame_consumer(rec.frame_name, img, img)
            return True
        for key in ("left_frames", "right_frames"):
            d = directories[key]
            d.mkdir(parents=True, exist_ok=True)
            for rec in manifest.records:
                cv2.imwrite(str(d / f"{rec.frame_name}.png"), np.zeros((16, 16, 3), dtype=np.uint8))
        return True


class _FakeDistortion:
    def prepare_stream_transform(self, *, source_shape, settings):
        return lambda left, right: (left, right)

    def apply_distortion_and_crop(
        self, left_files, right_files, directories, _settings, _pt=None
    ) -> bool:
        for key, files in (("left_cropped", left_files), ("right_cropped", right_files)):
            d = directories[key]
            d.mkdir(parents=True, exist_ok=True)
            for f in files:
                cv2.imwrite(str(d / f.name), np.zeros((16, 16, 3), dtype=np.uint8))
        return True

    def crop_frames(self, directories, _settings, _pt=None, _n=0) -> bool:
        return self.apply_distortion_and_crop(
            sorted(directories["left_frames"].glob("*.png")),
            sorted(directories["right_frames"].glob("*.png")),
            directories,
            _settings,
        )


class _FakeUpscaler:
    def apply_upscaling(self, *_args: Any) -> bool:
        return True


class _FakeAssembler:
    def resolve_direct_vr_source(
        self, directories: dict[str, Path], _settings: dict, num_frames: int
    ) -> DirectStereoSource:
        return DirectStereoSource(
            left_parent=directories["left_cropped"],
            right_parent=directories["right_cropped"],
            first_index=1,
            frame_count=num_frames,
        )

    def assemble_vr_frames(
        self, directories: dict[str, Path], _settings: dict, _pt=None, _n=0
    ) -> bool:
        vr_dir = directories["vr_frames"]
        vr_dir.mkdir(parents=True, exist_ok=True)
        for i in (1, 2):
            cv2.imwrite(str(vr_dir / f"frame_{i:06d}.png"), np.zeros((16, 32, 3), dtype=np.uint8))
        return True


def test_orchestrator_file_backed_and_cleanup(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-bytes")
    out = tmp_path / "job"
    encoder = _FakeEncoder()
    orch = ProcessingOrchestrator(
        depth_processor=_FakeDepth(),
        stereo_generator=_FakeStereo(),
        distortion_processor=_FakeDistortion(),
        upscaler=_FakeUpscaler(),
        vr_assembler=_FakeAssembler(),
        video_encoder=encoder,
    )
    settings = validate_settings(
        {
            "vr_format": "side_by_side",
            "vr_resolution": "custom:16x16",
            "per_eye_width": 16,
            "per_eye_height": 16,
            "vr_output_width": 32,
            "vr_output_height": 16,
            "direct_vr_encode": True,
            "keep_intermediates": False,
            "apply_distortion": False,
            "crop_factor": 1.0,
        },
        source="explicit",
    )
    props = {
        "width": 16,
        "height": 16,
        "fps": 30.0,
        "frame_count": 2,
        "duration": 2 / 30.0,
        "codec": 0,
        "sample_aspect_ratio": "1:1",
        "sample_aspect_ratio_numerator": 1,
        "sample_aspect_ratio_denominator": 1,
    }
    ok = orch.process(video, out, props, settings)
    assert ok is True
    assert encoder.streamed_frames == ["frame_000001", "frame_000002"]
    assert not (out / "00_original_frames").exists()
    saved = load_processing_settings(out / "processing-settings-v5.json")
    assert saved is not None
    assert saved["metadata"]["processing_status"] == "completed"
    assert saved["metadata"]["cleanup_status"] == "complete"
