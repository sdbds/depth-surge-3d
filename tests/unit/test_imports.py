"""Test that all entry points can import correctly."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest


def _load_cli_module():
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root / "src"))
    try:
        return importlib.import_module("depth_surge_3d.cli")
    finally:
        sys.path.pop(0)


class TestEntryPointImports:
    """Test that all entry point files can import successfully."""

    def test_cli_imports(self):
        """Test that the packaged CLI imports without running it."""
        assert callable(_load_cli_module().main)

    def test_legacy_root_launcher_help(self, tmp_path):
        project_root = Path(__file__).parent.parent.parent
        result = subprocess.run(
            [sys.executable, str(project_root / "depth_surge_3d.py"), "--help"],
            capture_output=True,
            check=False,
            cwd=tmp_path,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert "Convert 2D videos to immersive 3D VR format" in result.stdout

    def test_legacy_root_launcher_does_not_duplicate_cli_logic(self):
        project_root = Path(__file__).parent.parent.parent
        launcher = (project_root / "depth_surge_3d.py").read_text(encoding="utf-8")

        assert "from depth_surge_3d.cli import main" in launcher
        assert "def main(" not in launcher
        assert "def create_argument_parser(" not in launcher

    def test_cli_exposes_only_final_stereo_controls(self):
        parser = _load_cli_module().create_argument_parser()

        args = parser.parse_args(
            [
                "video.mp4",
                "--stereo-strength",
                "3.25",
                "--convergence",
                "0.4",
                "--occlusion-fill",
                "none",
                "--no-scene-detection",
                "--scene-cut-threshold",
                "0.7",
                "--min-scene-frames",
                "12",
                "--raw-storage-dtype",
                "float32",
                "--stereo-io-workers",
                "3",
                "--migrate-legacy",
                "delete",
            ]
        )

        assert args.stereo_strength == 3.25
        assert args.convergence == 0.4
        assert args.occlusion_fill == "none"
        assert args.scene_detection is False
        assert args.scene_cut_threshold == 0.7
        assert args.min_scene_frames == 12
        assert args.raw_storage_dtype == "float32"
        assert args.stereo_io_workers == 3
        assert args.migrate_legacy == "delete"

    def test_cli_accepts_custom_vr_resolution(self):
        parser = _load_cli_module().create_argument_parser()

        args = parser.parse_args(["video.mp4", "--vr-resolution", "custom:2560x1080"])

        assert args.vr_resolution == "custom:2560x1080"

    def test_cli_builds_one_complete_settings_object(self):
        module = _load_cli_module()
        args = module.create_argument_parser().parse_args(
            ["video.mp4", "--upscale-model", "x4", "--verbose"]
        )

        settings = module._build_processing_settings(args)

        assert settings["upscale_model"] == "x4"
        assert settings["verbose"] is True
        assert settings["target_fps"] is None

    def test_cli_validation_returns_the_settings_it_builds(self, monkeypatch):
        module = _load_cli_module()
        args = module.create_argument_parser().parse_args(
            ["video.mp4", "--upscale-model", "x4", "--verbose"]
        )
        monkeypatch.setattr(module, "validate_video_file", lambda _path: True)

        settings = module.validate_arguments(args)

        assert isinstance(settings, dict)
        assert settings["upscale_model"] == "x4"
        assert settings["verbose"] is True

    @pytest.mark.parametrize(
        "resolution",
        ["custom", "custom:0x1080", "custom:2560", "custom:axb"],
    )
    def test_cli_rejects_invalid_custom_vr_resolution(self, resolution):
        parser = _load_cli_module().create_argument_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["video.mp4", "--vr-resolution", resolution])

    @pytest.mark.parametrize(
        "removed_option",
        ["--baseline", "--focal-length", "--hole-fill-quality"],
    )
    def test_cli_rejects_removed_stereo_controls(self, removed_option):
        parser = _load_cli_module().create_argument_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["video.mp4", removed_option, "1"])

    def test_app_imports(self):
        """Test that web UI (app.py) imports work."""
        # Add project root to path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root / "src"))

        # Test all imports used in app.py
        from src.depth_surge_3d.utils.system.console import warning
        from depth_surge_3d.processing import VideoProcessor
        from depth_surge_3d.rendering import create_stereo_projector
        from depth_surge_3d.utils.domain.resolution import (
            get_resolution_dimensions,
            calculate_vr_output_dimensions,
            auto_detect_resolution,
        )
        from depth_surge_3d.utils.path_utils import sanitize_filename

        assert callable(warning)
        assert VideoProcessor is not None
        assert callable(create_stereo_projector)
        assert callable(get_resolution_dimensions)
        assert callable(calculate_vr_output_dimensions)
        assert callable(auto_detect_resolution)
        assert callable(sanitize_filename)

    def test_app_uses_only_the_canonical_package_namespace(self):
        project_root = Path(__file__).parent.parent.parent
        tree = ast.parse((project_root / "app.py").read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }

        assert not any(module.startswith("src.depth_surge_3d") for module in imported_modules)
        assert "depth_surge_3d.processing.batch_processor" not in imported_modules

    def test_all_public_modules_importable(self):
        """Test that all public modules can be imported."""
        # Add project root to path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root / "src"))

        # Test all major module imports
        from depth_surge_3d import __version__
        from depth_surge_3d.core import constants
        from depth_surge_3d.inference import (
            create_video_depth_estimator,
            create_video_depth_estimator_da3,
            create_upscaler,
        )
        from depth_surge_3d.io import operations
        from depth_surge_3d.processing import (  # noqa: F401
            VideoProcessor,
            ProcessingOrchestrator,
            DepthMapProcessor,
            StereoPairGenerator,
            DistortionProcessor,
            FrameUpscalerProcessor,
            VRFrameAssembler,
            VideoEncoder,
        )
        from depth_surge_3d.rendering import create_stereo_projector
        from depth_surge_3d.utils import (  # noqa: F401
            batch_analysis,
            path_utils,
        )
        from depth_surge_3d.utils.domain import (  # noqa: F401
            depth_cache,
            resolution,
        )
        from depth_surge_3d.utils.imaging import image_processing  # noqa: F401
        from depth_surge_3d.utils.system import (  # noqa: F401
            check_cuda,
            console,
            vram_manager,
        )

        # Basic sanity checks
        assert __version__ is not None
        assert constants is not None
        assert callable(create_video_depth_estimator)
        assert callable(create_video_depth_estimator_da3)
        assert callable(create_upscaler)
        assert operations is not None
        assert VideoProcessor is not None
        assert ProcessingOrchestrator is not None
        assert callable(create_stereo_projector)


class TestNoOrphanedImports:
    """Test that no code uses old import paths."""

    def test_no_old_console_imports(self):
        """Verify no code uses old utils.console path."""
        project_root = Path(__file__).parent.parent.parent

        # Check source files
        bad_imports = []
        for py_file in project_root.glob("**/*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            content = py_file.read_text(encoding="utf-8")
            # Check for old import pattern (but not the new one)
            if "utils.console" in content and "utils.system.console" not in content:
                bad_imports.append(str(py_file.relative_to(project_root)))

        assert bad_imports == [], f"Found old console imports in: {bad_imports}"

    def test_dead_progress_module_is_removed(self):
        project_root = Path(__file__).parent.parent.parent

        assert not (project_root / "src/depth_surge_3d/utils/domain/progress.py").exists()
        assert "from .progress import" not in (
            project_root / "src/depth_surge_3d/utils/domain/__init__.py"
        ).read_text(encoding="utf-8")
        assert "ProgressTracker" not in (
            project_root / "src/depth_surge_3d/utils/__init__.py"
        ).read_text(encoding="utf-8")

    def test_no_old_resolution_imports(self):
        """Verify no code uses old utils.resolution path."""
        project_root = Path(__file__).parent.parent.parent

        bad_imports = []
        for py_file in project_root.glob("**/*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            content = py_file.read_text(encoding="utf-8")
            # Check for old import pattern (but not the new one)
            if "utils.resolution" in content and "utils.domain.resolution" not in content:
                bad_imports.append(str(py_file.relative_to(project_root)))

        assert bad_imports == [], f"Found old resolution imports in: {bad_imports}"

    def test_no_old_depth_cache_imports(self):
        """Verify no code uses old utils.depth_cache path."""
        project_root = Path(__file__).parent.parent.parent

        bad_imports = []
        for py_file in project_root.glob("**/*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            content = py_file.read_text(encoding="utf-8")
            # Check for old import pattern (but not the new one)
            if "utils.depth_cache" in content and "utils.domain.depth_cache" not in content:
                bad_imports.append(str(py_file.relative_to(project_root)))

        assert bad_imports == [], f"Found old depth_cache imports in: {bad_imports}"

    def test_no_old_video_processor_imports(self):
        """Verify no code uses old processing.video_processor path."""
        project_root = Path(__file__).parent.parent.parent

        bad_imports = []
        for py_file in project_root.glob("**/*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
            if "processing/orchestration/video_processor.py" in str(py_file):
                continue  # Skip the actual module file
            if "test_imports.py" in str(py_file):
                continue  # Skip this test file itself

            content = py_file.read_text(encoding="utf-8")
            # Check for old import pattern
            if "processing.video_processor import VideoProcessor" in content:
                bad_imports.append(str(py_file.relative_to(project_root)))

        assert bad_imports == [], f"Found old imports in: {bad_imports}"
