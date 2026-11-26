"""
Video processor with temporal consistency.

Thin orchestrator that delegates to specialized processor modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ...io.job_lock import JobWriterLock
from ..frames.depth_processor import DepthMapProcessor
from ..frames.distortion_processor import DistortionProcessor
from ..frames.frame_upscaler import FrameUpscalerProcessor
from ..frames.stereo_generator import StereoPairGenerator
from ..frames.vr_assembler import VRFrameAssembler
from ..video.video_encoder import VideoEncoder
from .pipeline_orchestrator import ProcessingOrchestrator


class VideoProcessor:
    """Thin orchestrator delegating to specialized video processing stages."""

    def __init__(
        self,
        depth_estimator: Any,
        verbose: bool = False,
        release_depth_model: Callable[[], None] | None = None,
        temporal_stabilizer: Any | None = None,
        preselected_depth_files: list[Path] | None = None,
        preselected_render_source: str | None = None,
        **_ignored: object,
    ) -> None:
        self.depth_estimator = depth_estimator
        self.verbose = verbose
        if release_depth_model is None and depth_estimator is not None:
            release_depth_model = depth_estimator.unload_model

        self.depth_processor = (
            DepthMapProcessor(depth_estimator, verbose=verbose)
            if depth_estimator is not None
            else None
        )
        self.stereo_generator = StereoPairGenerator(verbose=verbose)
        self.distortion_processor = DistortionProcessor(verbose=verbose)
        self.upscaler = FrameUpscalerProcessor(verbose=verbose)
        self.vr_assembler = VRFrameAssembler(verbose=verbose)
        self.video_encoder = VideoEncoder(verbose=verbose)

        self.orchestrator = ProcessingOrchestrator(
            depth_processor=self.depth_processor,
            stereo_generator=self.stereo_generator,
            distortion_processor=self.distortion_processor,
            upscaler=self.upscaler,
            vr_assembler=self.vr_assembler,
            video_encoder=self.video_encoder,
            verbose=verbose,
            release_depth_model=release_depth_model,
            temporal_stabilizer=temporal_stabilizer,
            preselected_depth_files=preselected_depth_files,
            preselected_render_source=preselected_render_source,
        )

    def process(
        self,
        video_path: str,
        output_dir: str,
        video_properties: dict[str, Any],
        settings: dict[str, Any],
        progress_callback: Any = None,
        job_lock: JobWriterLock | None = None,
    ) -> bool:
        return self.orchestrator.process(
            Path(video_path),
            Path(output_dir),
            video_properties,
            settings,
            progress_tracker=progress_callback,
            job_lock=job_lock,
        )
