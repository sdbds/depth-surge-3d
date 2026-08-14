"""
Video processor with temporal consistency.

This module implements video processing using Video-Depth-Anything
for temporal consistency across video frames.

REFACTORED: This is now a thin orchestrator that delegates to specialized
processor modules for improved maintainability and testability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..frames.depth_processor import DepthMapProcessor
from ..frames.stereo_generator import StereoPairGenerator
from ..frames.distortion_processor import DistortionProcessor
from ..frames.frame_upscaler import FrameUpscalerProcessor
from ..frames.vr_assembler import VRFrameAssembler
from ..video.video_encoder import VideoEncoder
from .pipeline_orchestrator import ProcessingOrchestrator


class VideoProcessor:
    """
    Handles video processing with temporal consistency.

    Uses Video-Depth-Anything model to process entire videos with
    temporal consistency for superior depth estimation quality.

    This is a thin orchestrator that delegates to specialized processors:
    - DepthMapProcessor: Depth map generation with caching
    - StereoPairGenerator: Stereo pair creation
    - DistortionProcessor: Fisheye distortion and cropping
    - FrameUpscalerProcessor: AI upscaling
    - VRFrameAssembler: VR frame assembly
    - VideoEncoder: Video encoding with FFmpeg
    - ProcessingOrchestrator: Pipeline coordination
    """

    def __init__(self, depth_estimator: Any, verbose: bool = False):
        """
        Initialize video processor with specialized modules.

        Args:
            depth_estimator: Depth estimation model instance (VideoDepthEstimator or VideoDepthEstimatorDA3)
            verbose: Enable verbose output
        """
        self.depth_estimator = depth_estimator
        self.verbose = verbose

        # Initialize specialized processor modules
        self.depth_processor = DepthMapProcessor(depth_estimator, verbose=verbose)
        self.stereo_generator = StereoPairGenerator(verbose=verbose)
        self.distortion_processor = DistortionProcessor(verbose=verbose)
        self.upscaler = FrameUpscalerProcessor(verbose=verbose)
        self.vr_assembler = VRFrameAssembler(verbose=verbose)
        self.video_encoder = VideoEncoder(verbose=verbose)

        # Initialize pipeline orchestrator with all processors
        self.orchestrator = ProcessingOrchestrator(
            depth_processor=self.depth_processor,
            stereo_generator=self.stereo_generator,
            distortion_processor=self.distortion_processor,
            upscaler=self.upscaler,
            vr_assembler=self.vr_assembler,
            video_encoder=self.video_encoder,
            verbose=verbose,
        )

    def process(
        self,
        video_path: str,
        output_dir: str,
        video_properties: dict[str, Any],
        settings: dict[str, Any],
        progress_callback=None,
    ) -> bool:
        """
        Process video in batch mode with temporal consistency.

        Args:
            video_path: Path to input video
            output_dir: Output directory path
            video_properties: Video metadata (frame_count, fps, etc.)
            settings: Processing settings
            progress_callback: Optional progress callback for web UI

        Returns:
            True if processing completed successfully

        Side effects:
            - Delegates to ProcessingOrchestrator which executes full pipeline
            - Creates output directory structure
            - Generates intermediate and final output files
        """
        return self.orchestrator.process(
            Path(video_path),
            Path(output_dir),
            video_properties,
            settings,
            progress_tracker=progress_callback,
        )
