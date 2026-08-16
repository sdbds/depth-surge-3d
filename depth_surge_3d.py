#!/usr/bin/env python3
"""
Depth Surge 3D - Convert 2D videos to immersive 3D VR format using AI depth estimation.

This is the main entry point using the new modular architecture.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from depth_surge_3d.rendering import create_stereo_projector  # noqa: E402
from depth_surge_3d.core.constants import (  # noqa: E402
    DEFAULT_SETTINGS,
    FISHEYE_PROJECTIONS,
    VALIDATION_RANGES,
)
from depth_surge_3d.core.settings import validate_settings  # noqa: E402
from depth_surge_3d.utils import (  # noqa: E402
    get_available_resolutions,
    warning as console_warning,
)
from depth_surge_3d.utils.domain.resolution import get_resolution_dimensions  # noqa: E402
from depth_surge_3d.io.operations import (  # noqa: E402
    validate_video_file,
    can_resume_processing,
    load_processing_settings,
)
from depth_surge_3d.io.resume import (  # noqa: E402
    apply_legacy_migration,
    build_resume_report,
    resolve_resume_depth_model_version,
)
from depth_surge_3d.processing.frames.depth_storage import (  # noqa: E402
    build_current_model_fingerprint,
)
from depth_surge_3d.inference.depth.backend_registry import (  # noqa: E402
    backend_availability,
    get_backend_spec,
    list_backend_specs,
    normalize_model_size,
)


def _print_resume_report(report) -> None:
    """Print the deterministic stage decisions before migration."""
    print("Resume stage report:")
    for stage in report.stages:
        print(
            f"  - {stage.name}: {stage.disposition} " f"({stage.size_bytes} bytes) - {stage.reason}"
        )
    if report.removed_settings:
        print("  - Removed legacy settings: " + ", ".join(report.removed_settings))


def _parse_vr_resolution(value: str) -> str:
    if value == "auto":
        return value
    try:
        get_resolution_dimensions(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def _build_processing_settings(args: argparse.Namespace) -> dict[str, object]:
    """Translate CLI names once, then hand one validated object to the pipeline."""
    spec = get_backend_spec(args.depth_model_version)
    model_size = normalize_model_size(
        args.depth_model_version,
        model_path=args.model,
        model_size=args.model_size,
    )
    use_metric = (
        True
        if args.depth_model_version == "moge2"
        else bool(args.metric) and spec.capabilities.metric_depth
    )
    return validate_settings(
        {
            "vr_format": args.format,
            "vr_resolution": args.vr_resolution,
            "stereo_strength": args.stereo_strength,
            "convergence": args.convergence,
            "occlusion_fill": args.occlusion_fill,
            "scene_detection": args.scene_detection,
            "scene_cut_threshold": args.scene_cut_threshold,
            "min_scene_frames": args.min_scene_frames,
            "raw_storage_dtype": args.raw_storage_dtype,
            "stereo_io_workers": args.stereo_io_workers,
            "migrate_legacy": args.migrate_legacy,
            "start_time": args.start,
            "end_time": args.end,
            "apply_distortion": not args.no_distortion,
            "fisheye_projection": args.fisheye_projection,
            "fisheye_fov": args.fisheye_fov,
            "crop_factor": args.crop_factor,
            "fisheye_crop_factor": args.fisheye_crop_factor,
            "preserve_audio": not args.no_audio,
            "keep_intermediates": not args.no_intermediates,
            "target_fps": args.target_fps,
            "experimental_frame_interpolation": args.experimental_frame_interpolation,
            "upscale_model": args.upscale_model,
            "verbose": args.verbose,
            "depth_model_version": args.depth_model_version,
            "model_size": model_size,
            "model_path": args.model,
            "depth_resolution": args.depth_resolution,
            "use_metric_depth": use_metric,
            "device": args.device,
        },
        source="explicit",
    )


def _backend_is_available(backend_id: str) -> bool:
    availability = backend_availability(backend_id)
    if availability.available:
        return True
    print(f"Error: {availability.reason}")
    if availability.install_command:
        print(f"Install with: {availability.install_command}")
    return False


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert 2D videos to immersive 3D VR format using AI depth estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4                                    # Basic conversion
  %(prog)s video.mp4 --vr-resolution 16x9-4k          # High quality 4K
  %(prog)s video.mp4 --vr-resolution custom:2560x1080 # Custom resolution
  %(prog)s --resume ./output/my_video_output/          # Resume previous job
  %(prog)s --list-resolutions                          # Show available resolutions

Note: Always uses Video-Depth-Anything for temporal consistency across frames.
        """,
    )

    # Input/output arguments
    parser.add_argument("input_video", nargs="?", help="Input video file path")

    parser.add_argument(
        "--output-dir",
        "-o",
        default=DEFAULT_SETTINGS["output_dir"],
        help="Output directory (default: %(default)s)",
    )

    # Resume functionality
    parser.add_argument(
        "--resume", metavar="DIRECTORY", help="Resume processing from an existing output directory"
    )

    # VR settings
    parser.add_argument(
        "--vr-resolution",
        type=_parse_vr_resolution,
        default=DEFAULT_SETTINGS["vr_resolution"],
        help="VR output resolution per eye (default: %(default)s)",
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=["side_by_side", "over_under"],
        default=DEFAULT_SETTINGS["vr_format"],
        help="VR output format (default: %(default)s)",
    )

    # Time range
    parser.add_argument("--start", help="Start time (format: HH:MM:SS or seconds)")

    parser.add_argument("--end", help="End time (format: HH:MM:SS or seconds)")

    # Depth and stereo parameters
    parser.add_argument(
        "--stereo-strength",
        type=float,
        default=DEFAULT_SETTINGS["stereo_strength"],
        help="Total horizontal disparity as a percentage of frame width (default: %(default)s)",
    )
    parser.add_argument(
        "--convergence",
        type=float,
        default=DEFAULT_SETTINGS["convergence"],
        help="Canonical depth placed at zero parallax (default: %(default)s)",
    )
    parser.add_argument(
        "--occlusion-fill",
        choices=["none", "background"],
        default=DEFAULT_SETTINGS["occlusion_fill"],
        help="Disocclusion handling mode (default: %(default)s)",
    )
    parser.add_argument(
        "--scene-detection",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SETTINGS["scene_detection"],
        help="Enable deterministic scene segmentation (default: %(default)s)",
    )
    parser.add_argument(
        "--scene-cut-threshold",
        type=float,
        default=DEFAULT_SETTINGS["scene_cut_threshold"],
        help="Luma-histogram scene cut threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--min-scene-frames",
        type=int,
        default=DEFAULT_SETTINGS["min_scene_frames"],
        help="Minimum candidate scene length in frames (default: %(default)s)",
    )
    parser.add_argument(
        "--raw-storage-dtype",
        choices=["auto", "float16", "float32"],
        default=DEFAULT_SETTINGS["raw_storage_dtype"],
        help="Raw native depth storage type (default: %(default)s)",
    )
    parser.add_argument(
        "--stereo-io-workers",
        type=int,
        default=DEFAULT_SETTINGS["stereo_io_workers"],
        help="Stereo decode and output worker count (default: %(default)s)",
    )
    parser.add_argument(
        "--migrate-legacy",
        choices=["archive", "delete"],
        default=DEFAULT_SETTINGS["migrate_legacy"],
        help="How resume handles legacy generated stages (default: %(default)s)",
    )

    # Distortion and projection
    parser.add_argument(
        "--fisheye-projection",
        choices=FISHEYE_PROJECTIONS,
        default=DEFAULT_SETTINGS["fisheye_projection"],
        help=f'Fisheye projection type (default: {DEFAULT_SETTINGS["fisheye_projection"]})',
    )
    parser.add_argument(
        "--fisheye-fov",
        type=float,
        default=DEFAULT_SETTINGS["fisheye_fov"],
        help=f'Fisheye field of view in degrees (default: {DEFAULT_SETTINGS["fisheye_fov"]})',
    )
    parser.add_argument(
        "--no-distortion",
        action="store_true",
        help="Disable fisheye distortion (keeps rectilinear projection)",
    )

    # Quality and processing options
    parser.add_argument(
        "--crop-factor",
        type=float,
        default=DEFAULT_SETTINGS["crop_factor"],
        help=f'Center crop factor (1.0 = no crop, 0.5 = crop to half) (default: {DEFAULT_SETTINGS["crop_factor"]})',
    )
    parser.add_argument(
        "--fisheye-crop-factor",
        type=float,
        default=DEFAULT_SETTINGS["fisheye_crop_factor"],
        help=f'Fisheye crop factor (default: {DEFAULT_SETTINGS["fisheye_crop_factor"]})',
    )
    # Model and device
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        help=("Path to model file (V2), model name (V3), or Hugging Face repository (See-Through)"),
    )
    model_group.add_argument(
        "--model-size",
        choices=["vits", "vitb", "vitl"],
        default=None,
    )
    parser.add_argument(
        "--depth-model-version",
        choices=[spec.backend_id for spec in list_backend_specs()],
        default="v2",
        help="Registered depth model backend (default: %(default)s)",
    )
    parser.add_argument("--depth-resolution", default="auto")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Processing device (default: auto)",
    )
    parser.add_argument(
        "--metric",
        action="store_true",
        help="Use metric depth model (outputs real depth values in meters)",
    )

    # Output options
    parser.add_argument("--no-audio", action="store_true", help="Do not preserve audio in output")
    parser.add_argument(
        "--no-intermediates", action="store_true", help="Do not keep intermediate processing files"
    )
    parser.add_argument("--target-fps", type=int, help="Target output FPS (default: match source)")

    # Experimental features
    parser.add_argument(
        "--experimental-frame-interpolation",
        action="store_true",
        help="EXPERIMENTAL: Double FPS using motion interpolation. WARNING: May produce artifacts, wobbling, or poor quality. Recommended for artistic experimentation only.",
    )
    parser.add_argument(
        "--upscale-model",
        choices=["none", "x2", "x4", "x4-conservative"],
        default="none",
        help="AI upscaling model (Real-ESRGAN). Options: none (disabled, default), x2 (2x fast), x4 (4x best quality), x4-conservative (4x without GAN artifacts). Significantly increases processing time (+2-3x) and VRAM usage (+2-4GB).",
    )

    # Information and debugging
    parser.add_argument(
        "--list-resolutions", action="store_true", help="List all available VR resolution options"
    )
    parser.add_argument("--model-info", action="store_true", help="Show model information and exit")
    parser.add_argument(
        "--cache-info", action="store_true", help="Show depth map cache statistics and exit"
    )
    parser.add_argument("--cache-clear", action="store_true", help="Clear depth map cache and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    return parser


def validate_arguments(args) -> dict[str, object] | None:  # noqa: C901
    """Validate CLI arguments and return the normalized processing settings."""

    # Handle resume mode
    if args.resume:
        if not Path(args.resume).exists():
            print(f"Error: Resume directory does not exist: {args.resume}")
            return None
        return {}  # Resume settings come from the persisted job metadata.

    # Regular mode validations
    if not args.input_video:
        print("Error: Input video is required when not resuming")
        return None

    # Validate input video
    if not validate_video_file(args.input_video):
        print(f"Error: Invalid or unsupported video file: {args.input_video}")
        return None

    if not _backend_is_available(args.depth_model_version):
        return None

    try:
        processing_settings = _build_processing_settings(args)
    except ValueError as exc:
        print(f"Error: {exc}")
        return None

    if (
        args.fisheye_fov < VALIDATION_RANGES["fisheye_fov"][0]
        or args.fisheye_fov > VALIDATION_RANGES["fisheye_fov"][1]
    ):
        print(
            f"Error: FOV must be between {VALIDATION_RANGES['fisheye_fov'][0]} and {VALIDATION_RANGES['fisheye_fov'][1]} degrees"
        )
        return None

    if (
        args.crop_factor < VALIDATION_RANGES["crop_factor"][0]
        or args.crop_factor > VALIDATION_RANGES["crop_factor"][1]
    ):
        print(
            f"Error: Crop factor must be between {VALIDATION_RANGES['crop_factor'][0]} and {VALIDATION_RANGES['crop_factor'][1]}"
        )
        return None

    if args.target_fps and (
        args.target_fps < VALIDATION_RANGES["target_fps"][0]
        or args.target_fps > VALIDATION_RANGES["target_fps"][1]
    ):
        print(
            f"Error: Target FPS must be between {VALIDATION_RANGES['target_fps'][0]} and {VALIDATION_RANGES['target_fps'][1]}"
        )
        return None

    return processing_settings


def list_available_resolutions():
    """List all available VR resolution options."""
    print("Available VR Resolution Options:")
    print("=" * 40)

    resolutions = get_available_resolutions()

    for category, items in resolutions.items():
        if items:  # Only show categories with items
            print(f"\n{category.replace('_', ' ').title()}:")
            for item in items:
                print(f"  {item['name']:<15} - {item['description']}")

    print("\nCustom Resolution:")
    print("  custom:WxH      - Custom resolution (e.g., custom:1920x1080)")
    print("\nAuto Detection:")
    print("  auto            - Automatically detect optimal resolution")


def main():  # noqa: C901
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle special commands
    if args.list_resolutions:
        list_available_resolutions()
        return 0

    if args.cache_info:
        from depth_surge_3d.utils.domain.depth_cache import get_cache_size, get_cache_dir

        cache_entries, cache_size_bytes = get_cache_size()
        cache_size_mb = cache_size_bytes / (1024 * 1024)
        cache_dir = get_cache_dir()
        print("Depth Map Cache Information")
        print("=" * 40)
        print(f"Cache directory: {cache_dir}")
        print(f"Cached videos:   {cache_entries}")
        print(f"Total size:      {cache_size_mb:.1f} MB")
        if cache_entries > 0:
            print(f"\nAverage:         {cache_size_mb / cache_entries:.1f} MB per video")
        print("\nCache speeds up re-processing with different stereo/VR settings.")
        print("To clear cache: depth_surge_3d.py --cache-clear")
        return 0

    if args.cache_clear:
        from depth_surge_3d.utils.domain.depth_cache import clear_cache

        count = clear_cache()
        print(f"Cleared {count} cached video(s) from depth map cache")
        return 0

    # Handle resume mode
    if args.resume:
        print(f"Checking resume capability for: {args.resume}")
        resume_info = can_resume_processing(Path(args.resume))

        if not resume_info["can_resume"]:
            print("Cannot resume processing:")
            for rec in resume_info["recommendations"]:
                print(f"  - {rec}")
            return 1

        print("Can resume processing:")
        print(f"  - Batch: {resume_info['batch_name']}")
        print(f"  - Status: {resume_info['status']}")
        if resume_info["progress_info"]:
            progress = resume_info["progress_info"]
            print(f"  - Progress: {progress['frames_processed']} frames processed")

        for rec in resume_info["recommendations"]:
            print(f"  - {rec}")

        # Load settings from the settings file
        settings_data = load_processing_settings(resume_info["settings_file"])
        if not settings_data:
            print("Could not load settings file")
            return 1

        # Extract video path and settings
        video_path = settings_data["metadata"]["source_video"]
        processing_settings = validate_settings(
            settings_data["processing_settings"],
            source="legacy_disk",
        )
        processing_settings["migrate_legacy"] = args.migrate_legacy
        processing_settings["verbose"] = args.verbose
        if any(
            value == "--raw-storage-dtype" or value.startswith("--raw-storage-dtype=")
            for value in sys.argv[1:]
        ):
            processing_settings["raw_storage_dtype"] = args.raw_storage_dtype
        processing_settings["depth_model_version"] = resolve_resume_depth_model_version(
            processing_settings,
            Path(args.resume),
            default="v2",
        )
        resume_spec = get_backend_spec(processing_settings["depth_model_version"])
        processing_settings["model_size"] = normalize_model_size(
            processing_settings["depth_model_version"],
            model_path=processing_settings.get("model_path"),
            model_size=processing_settings.get("model_size"),
        )
        processing_settings["depth_resolution"] = processing_settings.get(
            "depth_resolution", "auto"
        )
        processing_settings["use_metric_depth"] = (
            True
            if processing_settings["depth_model_version"] == "moge2"
            else bool(processing_settings.get("use_metric_depth", False))
            and resume_spec.capabilities.metric_depth
        )

        if not _backend_is_available(processing_settings["depth_model_version"]):
            return 1

        projector = create_stereo_projector(
            model_path=processing_settings.get("model_path"),
            device=processing_settings.get("device", "auto"),
            metric=bool(processing_settings.get("use_metric_depth", False)),
            depth_model_version=processing_settings.get("depth_model_version", "v2"),
            model_size=processing_settings.get("model_size"),
        )
        if not projector.load_model():
            print("Could not load depth estimation model")
            return 1
        model_fingerprint = build_current_model_fingerprint(
            projector.depth_estimator,
            processing_settings,
        )

        try:
            resume_report = build_resume_report(
                Path(args.resume).resolve(),
                processing_settings,
                source_video=video_path,
                model_fingerprint=model_fingerprint,
                settings_file=resume_info["settings_file"],
            )
            _print_resume_report(resume_report)
            apply_legacy_migration(resume_report, args.migrate_legacy)
            processing_settings = resume_report.migrated_settings
        except (OSError, TypeError, ValueError) as exc:
            print(f"Cannot migrate resume data: {exc}")
            return 1

        print("Resuming processing...")
        print(f"Input: {video_path}")
        print(f"Output: {args.resume}")

        success = projector.process_video(
            video_path=video_path,
            output_dir=args.resume,
            settings=processing_settings,
        )

        if success:
            print("Resume processing completed successfully!")
            return 0
        else:
            print("Resume processing failed. Check error messages above.")
            return 1

    # Validate arguments for normal processing
    processing_settings = validate_arguments(args)
    if processing_settings is None:
        return 1

    # Create stereo projector
    try:
        projector = create_stereo_projector(
            processing_settings.get("model_path"),
            processing_settings.get("device", "auto"),
            bool(processing_settings.get("use_metric_depth", False)),
            processing_settings.get("depth_model_version", "v2"),
            model_size=processing_settings.get("model_size"),
        )

        if args.model_info:
            info = projector.get_model_info()
            print("Model Information:")
            print("=" * 20)
            for key, value in info.items():
                print(f"{key}: {value}")
            return 0

        # Process video
        model_name = get_backend_spec(args.depth_model_version).display_name
        print("Starting Depth Surge 3D processing...")
        print(f"Input: {args.input_video}")
        print(f"Output: {args.output_dir} (batch subdirectory will be created)")
        print(f"Model: {model_name}")
        print(f"Format: {args.format}")
        print(f"Resolution: {args.vr_resolution}")

        # Show warning for experimental features
        if args.experimental_frame_interpolation:
            print(console_warning("WARNING: Experimental frame interpolation enabled!"))
            print("   This feature may produce artifacts, wobbling, or poor quality.")
            print("   Recommended for artistic experimentation only.")
            print()

        # Create batch-specific output directory
        if args.resume:
            # Use existing directory for resume
            batch_output_dir = args.resume
        else:
            # Create new batch directory
            from datetime import datetime

            video_name = Path(args.input_video).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_output_dir = Path(args.output_dir) / f"{video_name}_{timestamp}"

        success = projector.process_video(
            video_path=args.input_video,
            output_dir=str(batch_output_dir),
            settings=processing_settings,
        )

        if success:
            print("Processing completed successfully!")
            print(f"Output saved to: {batch_output_dir}")
            return 0
        else:
            print("Processing failed. Check error messages above.")
            return 1

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1
    finally:
        # Clean up
        try:
            if "projector" in locals():
                projector.unload_model()
        except Exception:  # noqa: E722
            pass


if __name__ == "__main__":
    sys.exit(main())
