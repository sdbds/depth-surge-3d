"""Tests for the experimental See-Through Marigold depth estimator."""

from __future__ import annotations

import importlib
import tracemalloc
from pathlib import Path
from types import ModuleType, SimpleNamespace
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RGB transfer regression")
def test_rgb_upload_preserves_all_byte_values_without_host_float_expansion(dtype, reverse):
    from torch.utils._python_dispatch import TorchDispatchMode

    module = _module()
    values = np.arange(256, dtype=np.uint8).reshape(1, 16, 16, 1).repeat(3, axis=-1)
    if reverse:
        values = values[:, :, ::-1]
    original = values.copy()
    expected = torch.from_numpy(values.astype(np.float32) / np.float32(255.0))
    expected = expected.permute(0, 3, 1, 2).to(device="cuda", dtype=dtype) * 2.0 - 1.0
    uploads = []

    class Uploads(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if (
                func == torch.ops.aten._to_copy.default
                and args[0].device.type == "cpu"
                and args[0].ndim == 4
                and torch.device(kwargs.get("device", "cpu")).type == "cuda"
            ):
                uploads.append((args[0].dtype, args[0].numel()))
            return func(*args, **kwargs)

    with Uploads():
        actual = module._prepare_rgb_tensor(values, device="cuda", dtype=dtype)

    assert torch.equal(actual, expected)
    assert actual.stride() == expected.stride()
    np.testing.assert_array_equal(values, original)
    assert uploads == [(torch.uint8, values.size)]


@pytest.mark.parametrize("input_dtype", [np.uint8, np.float32, np.float64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rgb_preparation_preserves_cpu_and_non_byte_numerics(input_dtype, dtype):
    module = _module()
    values = np.linspace(0, 255, 96).reshape(2, 4, 4, 3).astype(input_dtype)
    expected = torch.from_numpy(values.astype(np.float32) / np.float32(255.0))
    expected = expected.permute(0, 3, 1, 2).to(dtype=dtype) * 2.0 - 1.0

    actual = module._prepare_rgb_tensor(values, device="cpu", dtype=dtype)

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FP32 compatibility")
def test_rgb_preparation_keeps_cpu_rounding_for_cuda_fp32():
    module = _module()
    values = np.arange(256, dtype=np.uint8).reshape(1, 16, 16, 1).repeat(3, axis=-1)
    expected = torch.from_numpy(values.astype(np.float32) / np.float32(255.0))
    expected = expected.permute(0, 3, 1, 2).to(device="cuda") * 2.0 - 1.0

    actual = module._prepare_rgb_tensor(values, device="cuda", dtype=torch.float32)

    assert torch.equal(actual, expected)


class TestSeeThroughDepthEstimator:
    def test_external_source_location_is_explicit_and_relocatable(self, tmp_path, monkeypatch):
        module = _module()
        root = _make_vendor_tree(tmp_path / "checkout")
        monkeypatch.setenv(module.QINGLONG_CAPTIONS_ENV, str(root))
        assert module.resolve_qinglong_captions_root() == root
        invalid = tmp_path / "missing"
        with pytest.raises(FileNotFoundError, match="source not found"):
            module.resolve_qinglong_captions_root(invalid)
        monkeypatch.setenv(module.QINGLONG_CAPTIONS_ENV, str(invalid))
        with pytest.raises(FileNotFoundError, match="is missing"):
            module.resolve_qinglong_captions_root()
        assert module.resolve_qinglong_captions_root(root) == root

    @pytest.mark.parametrize(
        "error", [ImportError("vendor dependency"), OSError("weights disk offline")]
    )
    def test_failed_loader_cannot_advertise_a_ready_backend(self, tmp_path, error, capsys):
        root = _make_vendor_tree(tmp_path / "checkout")
        pipeline = MagicMock(side_effect=error)
        estimator = _module().SeeThroughDepthEstimator(
            device="cpu", source_root=root, pipeline_loader=pipeline
        )
        assert not estimator.load_model()
        assert str(error) in capsys.readouterr().out
        assert not estimator.get_model_info()["loaded"]
        pipeline.assert_called_once()
        with pytest.raises(RuntimeError, match="not loaded"):
            estimator.estimate_depth_batch(np.zeros((1, 8, 8, 3), dtype=np.uint8))

    @pytest.mark.parametrize(
        "output,reason",
        [
            (SimpleNamespace(), "no depth map"),
            (SimpleNamespace(depth_np=np.zeros((1, 3, 8, 8))), "depth shape"),
        ],
    )
    def test_non_depth_output_is_not_a_valid_cpu_batch(self, output, reason):
        estimator = _module().SeeThroughDepthEstimator(device="cpu", processing_resolution=16)
        estimator.model = lambda **_: output
        with pytest.raises(RuntimeError, match=reason):
            estimator.estimate_depth_batch(np.zeros((1, 8, 8, 3), dtype=np.uint8))
        estimator.unload_model()
        assert not estimator.get_model_info()["loaded"]

    def test_defaults_target_the_see_through_marigold_model(self):
        module = _module()

        estimator = module.SeeThroughDepthEstimator(device="cpu")

        assert estimator.repo_id == "24yearsold/seethroughv0.0.1_marigold"
        assert estimator.processing_resolution == 768
        assert estimator.denoising_steps == 4
        assert estimator.seed == 1026
        assert estimator.max_batch_size == 1
        assert estimator.exact_input_batch_reuse is True
        assert estimator.model is None
        assert estimator.inference_precision == "float32"

    @pytest.mark.parametrize(
        ("bf16_supported", "precision"),
        [(True, "bfloat16"), (False, "float16")],
    )
    def test_cuda_inference_precision_matches_loader_dtype(self, bf16_supported, precision):
        module = _module()

        with (
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(total_memory=8 * 1024**3),
            ),
            patch.object(torch.cuda, "is_bf16_supported", return_value=bf16_supported),
        ):
            estimator = module.SeeThroughDepthEstimator(device="cuda")
            assert estimator.inference_precision == precision

    def test_cuda_uses_bounded_independent_frame_batches(self):
        module = _module()

        with patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(total_memory=24 * 1024**3),
        ):
            estimator = module.SeeThroughDepthEstimator(device="cuda")

        assert estimator.max_batch_size == 4

    def test_native_output_shape_preserves_768_square_and_other_aspect_ratios(self):
        module = _module()
        estimator = module.SeeThroughDepthEstimator(
            device="cpu",
            processing_resolution=768,
        )

        assert estimator.estimate_output_shape(1920, 1080, 768) == (768, 768)
        assert estimator.estimate_output_shape(3840, 2160, 512) == (288, 512)
        assert estimator.estimate_output_shape(1920, 1080, 1080) == (608, 1080)
        assert estimator.estimate_output_shape(1080, 1920, 1080) == (1080, 608)

    @pytest.mark.parametrize(
        "dtype",
        [
            np.uint8,
            np.uint16,
            np.float32,
            np.float64,
            np.dtype(np.uint16).newbyteorder("S"),
            np.dtype(np.float32).newbyteorder("S"),
            np.dtype(np.float64).newbyteorder("S"),
        ],
    )
    @pytest.mark.parametrize("output_shape", [(16, 24), (80, 96), (48, 48)])
    @pytest.mark.parametrize("readonly", [False, True])
    def test_rgb_resize_preserves_pixels_dtype_and_noncontiguous_inputs(
        self, dtype, output_shape, readonly
    ):
        import cv2

        module = _module()
        rng = np.random.default_rng(42)
        frames = rng.integers(0, 256, size=(3, 37, 122, 3), dtype=np.uint8).astype(dtype)
        frames = frames[:, :, ::2]
        if readonly:
            frames.setflags(write=False)
        original = frames.copy()
        height, width = output_shape
        interpolation = (
            cv2.INTER_AREA if max(frames.shape[1:3]) > max(output_shape) else cv2.INTER_LINEAR
        )
        expected = np.stack(
            [
                cv2.resize(
                    np.ascontiguousarray(frame[..., ::-1]),
                    (width, height),
                    interpolation=interpolation,
                )
                for frame in frames
            ]
        )

        actual = module.SeeThroughDepthEstimator._resize_rgb_frames(frames, height, width)

        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(frames, original)
        assert actual.dtype == expected.dtype
        assert actual.flags.c_contiguous

    def test_rgb_downsize_bounds_temporary_memory_by_output_not_source_resolution(self):
        import cv2  # noqa: F401

        module = _module()
        frames = np.zeros((4, 640, 1080, 3), dtype=np.uint8)
        tracing = tracemalloc.is_tracing()
        if not tracing:
            tracemalloc.start()
        try:
            baseline, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            result = module.SeeThroughDepthEstimator._resize_rgb_frames(frames, 160, 272)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            if not tracing:
                tracemalloc.stop()

        # One result batch and small resized scratch frames, not a full-size RGB copy.
        assert peak - baseline < 3 * result.nbytes

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
        pipeline._depth_surge_artifact_identity = "hf:owner/model@resolved"
        loader = MagicMock(return_value=pipeline)

        estimator = module.SeeThroughDepthEstimator(
            device="cpu",
            source_root=source_root,
            pipeline_loader=loader,
        )

        with patch(
            "src.depth_surge_3d.inference.depth.attention_backend."
            "install_diffusers_flash_attention",
            return_value=1,
        ) as install:
            assert estimator.load_model() is True

        assert estimator.model is pipeline
        install.assert_not_called()
        loader.assert_called_once_with(
            repo_id="24yearsold/seethroughv0.0.1_marigold",
            source_root=source_root.resolve(),
            cache_dir=cache_dir.resolve(),
            device="cpu",
            dtype=torch.float32,
        )
        assert estimator.get_model_info()["artifact_identity"] == ("hf:owner/model@resolved")

    def test_cuda_estimation_batches_independent_frames_without_changing_the_seed_contract(self):
        module = _module()
        batch_calls = []

        def batch_runner(*, pipeline, rgb_frames, denoising_steps, seed):
            del pipeline
            batch_calls.append((len(rgb_frames), denoising_steps, seed))
            return np.stack(
                [
                    np.full(rgb.shape[:2], rgb[0, 0, 0] / 255.0, dtype=np.float32)
                    for rgb in rgb_frames
                ]
            )

        class SequentialPipeline:
            def __call__(self, **kwargs):
                image = kwargs["img_list"][0]
                return SimpleNamespace(depth_np=np.zeros(image.shape[:2], dtype=np.float32))

        estimator = module.SeeThroughDepthEstimator(device="cuda")
        estimator.max_batch_size = 2
        estimator.batch_runner = batch_runner
        estimator.model = SequentialPipeline()
        frames = np.array(
            [
                np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8),
                np.full((2, 3, 3), [40, 50, 60], dtype=np.uint8),
                np.full((2, 3, 3), [70, 80, 90], dtype=np.uint8),
            ]
        )

        batch = estimator.estimate_depth_batch(frames, input_size=24)

        assert batch_calls == [(2, 4, 1026), (1, 4, 1026)]
        assert batch.values.shape == (3, 16, 24)
        np.testing.assert_allclose(batch.values[:, 0, 0], [30 / 255, 60 / 255, 90 / 255])

    def test_default_loader_pins_both_components_to_one_snapshot(self, tmp_path, monkeypatch):
        module = _module()
        source_root = _make_vendor_tree(tmp_path / "qinglong-captions")
        cache_dir = source_root / "huggingface" / "hub"
        snapshot = tmp_path / "hub" / "snapshots" / ("b" * 40)
        snapshot.mkdir(parents=True)
        identity = f"hf:owner/model@{'b' * 40}"
        resolve = MagicMock(return_value=(str(snapshot), identity))
        monkeypatch.setattr(module, "resolve_hf_snapshot", resolve, raising=False)

        layer_module = ModuleType("layerdiff3d")
        marigold_module = ModuleType("marigold")
        unet_type = MagicMock()
        pipeline_type = MagicMock()
        unet = MagicMock()
        pipeline = MagicMock()
        unet_type.from_pretrained.return_value = unet
        pipeline_type.from_pretrained.return_value = pipeline
        layer_module.UNetFrameConditionModel = unet_type
        marigold_module.MarigoldDepthPipeline = pipeline_type

        with patch.dict(
            "sys.modules",
            {
                "module.see_through.vendor.modules.layerdiffuse.layerdiff3d": layer_module,
                "module.see_through.vendor.modules.marigold": marigold_module,
            },
        ):
            result = module._load_see_through_pipeline(
                repo_id="owner/model",
                source_root=source_root,
                cache_dir=cache_dir,
                device="cpu",
                dtype=torch.float32,
            )

        resolve.assert_called_once_with("owner/model", cache_dir=cache_dir)
        unet_type.from_pretrained.assert_called_once_with(
            str(snapshot),
            subfolder="unet",
            cache_dir=str(cache_dir),
            torch_dtype=torch.float32,
        )
        assert pipeline_type.from_pretrained.call_args.args[0] == str(snapshot)
        assert result._depth_surge_artifact_identity == identity

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

        assert batch.values.shape == (2, 720, 1080)
        assert batch.values.dtype == np.float32
        assert batch.values[0, 0, 0] == pytest.approx(30 / 255.0)
        assert batch.values[1, 0, 0] == pytest.approx(60 / 255.0)
        assert [call["img_list"][0].shape for call in calls] == [(720, 1080, 4)] * 2
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

    def test_estimate_depth_batch_keeps_native_768_square_mode(self):
        module = _module()

        class FakePipeline:
            def __call__(self, **kwargs):
                image = kwargs["img_list"][0]
                depth = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
                return SimpleNamespace(depth_np=depth)

        estimator = module.SeeThroughDepthEstimator(device="cpu")
        estimator.model = FakePipeline()
        frames = np.zeros((1, 360, 640, 3), dtype=np.uint8)

        batch = estimator.estimate_depth_batch(frames, input_size=768)

        assert batch.values.shape == (1, 768, 768)

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


def test_opaque_batch_matches_individual_denoising_with_the_same_seed():
    from diffusers import DDIMScheduler

    calls = []

    class TinyUNet:
        device = torch.device("cpu")
        dtype = torch.float32

        def __call__(self, sample, timestep, encoder_hidden_states):
            assert sample.shape[1:3] == (1, 12)
            torch.testing.assert_close(sample[:, :, :4], sample[:, :, 4:8])
            assert encoder_hidden_states.shape[0] == sample.shape[0]
            calls.append(int(timestep))
            return SimpleNamespace(sample=sample[:, :, -4:] * 0.1 + sample[:, :, :4] * 0.2)

    pipeline = SimpleNamespace(
        vae=SimpleNamespace(device="cpu", dtype=torch.float32),
        unet=TinyUNet(),
        scheduler=DDIMScheduler(num_train_timesteps=16),
        empty_text_embed=None,
        encode_rgb=lambda x: torch.cat((x, x[:, :1]), dim=1),
        decode_depth=lambda x: x.mean(dim=1, keepdim=True),
    )
    pipeline.encode_empty_text = lambda: setattr(pipeline, "empty_text_embed", torch.zeros(1, 2, 3))

    def denoise(frames):
        return _module()._run_opaque_marigold_batch(
            pipeline=pipeline, rgb_frames=frames, denoising_steps=4, seed=21
        )

    images = np.stack((np.zeros((8, 8, 3), np.uint8), np.full((8, 8, 3), 255, np.uint8)))
    result = denoise(images)
    assert calls == [12, 8, 4, 0]
    np.testing.assert_allclose(result, np.concatenate([denoise(x[None]) for x in images]))
    assert result.min() >= 0 and result.max() <= 1
    assert not np.array_equal(result[0], result[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA synchronization regression")
def test_opaque_denoising_does_not_read_cuda_scalars_for_cpu_scheduler_indices():
    from torch.utils._python_dispatch import TorchDispatchMode

    diffusers = pytest.importorskip("diffusers")
    module = _module()
    unet_timesteps = []

    class ScalarReads(TorchDispatchMode):
        cuda_reads = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten._local_scalar_dense.default and args[0].is_cuda:
                self.cuda_reads += 1
            return func(*args, **(kwargs or {}))

    class TinyUNet:
        device = torch.device("cuda")
        dtype = torch.float32

        def __call__(self, sample, timestep, encoder_hidden_states):
            assert timestep.is_cuda
            unet_timesteps.append(timestep.clone())
            return SimpleNamespace(sample=sample[:, :, -4:] * 0.1)

    pipeline = SimpleNamespace(
        vae=SimpleNamespace(device=torch.device("cuda"), dtype=torch.float32),
        unet=TinyUNet(),
        scheduler=diffusers.DDIMScheduler(num_train_timesteps=16),
        empty_text_embed=torch.zeros((1, 2, 3), device="cuda"),
        encode_rgb=lambda values: torch.cat((values, values[:, :1]), dim=1),
        decode_depth=lambda values: values.mean(dim=1, keepdim=True),
    )
    with ScalarReads() as reads:
        result = module._run_opaque_marigold_batch(
            pipeline=pipeline,
            rgb_frames=np.zeros((2, 8, 8, 3), dtype=np.uint8),
            denoising_steps=4,
            seed=1026,
        )

    assert result.shape == (2, 8, 8)
    np.testing.assert_array_equal(result[0], result[1])
    assert torch.stack(unet_timesteps).cpu().tolist() == [12, 8, 4, 0]
    assert reads.cuda_reads == 0
