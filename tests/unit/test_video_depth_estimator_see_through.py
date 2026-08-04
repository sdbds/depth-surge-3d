"""Tests for the experimental See-Through Marigold depth estimator."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


def _module():
    return importlib.import_module(
        "src.depth_surge_3d.inference.depth.video_depth_estimator_see_through"
    )


def _make_vendor_tree(root: Path) -> Path:
    marker = (
        root
        / "module"
        / "see_through"
        / "vendor"
        / "modules"
        / "marigold"
        / "marigold_depth_pipeline.py"
    )
    marker.parent.mkdir(parents=True)
    marker.write_text("# test marker\n", encoding="utf-8")
    return root


class TestSeeThroughDepthEstimator:
    def test_defaults_target_the_see_through_marigold_model(self):
        module = _module()

        estimator = module.SeeThroughDepthEstimator(device="cpu")

        assert estimator.repo_id == "24yearsold/seethroughv0.0.1_marigold"
        assert estimator.processing_resolution == 768
        assert estimator.denoising_steps == 4
        assert estimator.seed == 1026
        assert estimator.max_batch_size == 1
        assert estimator.model is None

    def test_resolve_source_root_prefers_explicit_path(self, tmp_path):
        module = _module()
        explicit_root = _make_vendor_tree(tmp_path / "qinglong-captions")

        resolved = module.resolve_qinglong_captions_root(explicit_root)

        assert resolved == explicit_root.resolve()

    def test_resolve_source_root_uses_environment_override(self, tmp_path, monkeypatch):
        module = _module()
        env_root = _make_vendor_tree(tmp_path / "from-env")
        monkeypatch.setenv("QINGLONG_CAPTIONS_ROOT", str(env_root))

        resolved = module.resolve_qinglong_captions_root()

        assert resolved == env_root.resolve()

    def test_load_model_passes_runtime_and_local_cache_to_loader(self, tmp_path):
        module = _module()
        source_root = _make_vendor_tree(tmp_path / "qinglong-captions")
        cache_dir = source_root / "huggingface" / "hub"
        cache_dir.mkdir(parents=True)
        pipeline = MagicMock()
        loader = MagicMock(return_value=pipeline)

        estimator = module.SeeThroughDepthEstimator(
            device="cpu",
            source_root=source_root,
            pipeline_loader=loader,
        )

        with patch.object(module, "install_diffusers_flash_attention", return_value=1) as install:
            assert estimator.load_model() is True

        assert estimator.model is pipeline
        install.assert_called_once_with()
        loader.assert_called_once_with(
            repo_id="24yearsold/seethroughv0.0.1_marigold",
            source_root=source_root.resolve(),
            cache_dir=cache_dir.resolve(),
            device="cpu",
            dtype=torch.float32,
        )

    def test_estimate_depth_batch_uses_full_frame_layer_contract_and_fixed_seed(self):
        module = _module()
        calls = []

        class FakePipeline:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                image = kwargs["img_list"][0]
                value = image[0, 0, 0] / 255.0
                depth = torch.full((1, image.shape[0], image.shape[1]), value)
                return SimpleNamespace(depth_tensor=depth)

        estimator = module.SeeThroughDepthEstimator(device="cpu")
        estimator.model = FakePipeline()
        frames = np.array(
            [
                np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8),
                np.full((2, 3, 3), [40, 50, 60], dtype=np.uint8),
            ]
        )

        batch = estimator.estimate_depth_batch(frames, input_size=1080)

        assert batch.values.shape == (2, 768, 768)
        assert batch.values.dtype == np.float32
        assert batch.values[0, 0, 0] == pytest.approx(30 / 255.0)
        assert batch.values[1, 0, 0] == pytest.approx(60 / 255.0)
        assert [call["img_list"][0].shape for call in calls] == [(768, 768, 4)] * 2
        assert [call["img_list"][0][0, 0].tolist() for call in calls] == [
            [30, 20, 10, 255],
            [60, 50, 40, 255],
        ]
        assert all("input_image" not in call for call in calls)
        assert [call["denoising_steps"] for call in calls] == [4, 4]
        assert [call["generator"].initial_seed() for call in calls] == [1026, 1026]
        assert all(call["match_input_res"] is True for call in calls)
        assert all(call["color_map"] is None for call in calls)
        assert all(call["show_progress_bar"] is False for call in calls)

    def test_estimate_depth_batch_requires_loaded_model(self):
        module = _module()
        estimator = module.SeeThroughDepthEstimator(device="cpu")

        with pytest.raises(RuntimeError, match="Model not loaded"):
            estimator.estimate_depth_batch(np.zeros((1, 2, 2, 3), dtype=np.uint8))

    def test_factory_uses_default_repo_when_model_path_is_none(self):
        module = _module()

        estimator = module.create_see_through_depth_estimator(None, device="cpu", metric=True)

        assert isinstance(estimator, module.SeeThroughDepthEstimator)
        assert estimator.repo_id == module.DEFAULT_SEE_THROUGH_REPO
        assert estimator.metric is False


def test_depth_processor_honors_estimator_batch_limit():
    from src.depth_surge_3d.processing.frames.depth_processor import DepthMapProcessor

    estimator = SimpleNamespace(
        model_type="see_through",
        max_batch_size=1,
        get_model_size=lambda: "large",
    )
    processor = DepthMapProcessor(estimator)

    with (
        patch(
            "src.depth_surge_3d.processing.frames.depth_processor.get_vram_info",
            return_value={"total": 24.0, "available": 20.0},
        ),
        patch(
            "src.depth_surge_3d.processing.frames.depth_processor.calculate_optimal_chunk_size",
            return_value=24,
        ),
    ):
        chunk_size, input_size = processor._determine_chunk_params(1920, 1080, "auto")

    assert input_size == 1080
    assert chunk_size == 1
