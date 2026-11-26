"""Check the shipped VDPP tensor graph on CPU, without weights, downloads or CUDA."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.depth_surge_3d._vendor.vdpp.dinov2 import DinoVisionTransformer
from src.depth_surge_3d._vendor.vdpp.dinov2_layers import attention as dino_attention
from src.depth_surge_3d._vendor.vdpp.dpt import DPTHead
from src.depth_surge_3d._vendor.vdpp.dpt_temporal import DPTHeadTemporal
from src.depth_surge_3d._vendor.vdpp.motion_module import motion_module
from src.depth_surge_3d._vendor.vdpp.motion_module.attention import CrossAttention
from src.depth_surge_3d._vendor.vdpp.vdpp_model import VDPP
from src.depth_surge_3d._vendor.vdpp.util.transform import NormalizeImage, PrepareForNet, Resize
from src.depth_surge_3d._vendor.vdpp_utils.normal_utils import normal_vector


@pytest.fixture(scope="module")
def reference_model():
    # Select the real PyTorch attention kernels even on hosts with a CUDA xFormers build.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with pytest.MonkeyPatch.context() as backend, torch.random.fork_rng(devices=[]):
            backend.setattr(dino_attention, "XFORMERS_AVAILABLE", False)
            backend.setattr(motion_module, "XFORMERS_AVAILABLE", False)
            torch.random.default_generator.manual_seed(183)
            yield VDPP(features=32, out_channels=[32] * 4, num_frames=8).eval()
    finally:
        torch.set_num_threads(threads)


@torch.inference_mode()
def test_overlap_and_resize_match_independently_aligned_windows(reference_model):
    # A five-frame window has four overlapping frames, so six inputs require two calls.
    clip = torch.linspace(0.05, 0.95, 6 * 25 * 39).reshape(1, 6, 25, 39)
    padded = F.interpolate(clip, size=(28, 42), mode="bilinear", align_corners=True)
    a = reference_model(padded[:, :5], downsize=False)
    b = reference_model(padded[:, 1:], downsize=False)
    samples = b[0, :4].numpy().ravel()
    scale, offset = np.linalg.lstsq(
        np.column_stack((samples, np.ones_like(samples))), a[0, -4:].numpy().ravel(), rcond=None
    )[0]
    expected = torch.cat((a, b[:, -1:] * float(scale) + float(offset)), dim=1)
    expected = F.interpolate(expected, size=clip.shape[-2:], mode="bilinear", align_corners=True)
    actual = reference_model.infer_video_depth(clip, infer_frame=5, downsize=False)
    assert actual.shape == clip.shape and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


@torch.inference_mode()
def test_single_frame_keeps_the_spatial_ramp_through_the_release_resize_path(reference_model):
    clip = torch.linspace(0.1, 0.8, 28 * 42).reshape(1, 1, 28, 42)
    # With the initialized unit scale head VDPP adds a nonnegative residual.
    output = reference_model.infer_video_depth(clip, downsize=True)
    assert output.shape == clip.shape and torch.isfinite(output).all()
    assert torch.all(output >= clip - 1e-6)
    torch.testing.assert_close(output, reference_model(clip, downsize=True))


@pytest.mark.parametrize("chunked", [False, True])
@torch.inference_mode()
def test_dino_patch_grid_and_class_token_match_block_outputs(chunked):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(927)
        model = DinoVisionTransformer(
            img_size=28,
            patch_size=14,
            embed_dim=32,
            depth=2,
            num_heads=4,
            block_chunks=2 if chunked else 0,
            num_register_tokens=2,
        ).eval()
        image = torch.randn(1, 3, 28, 42)
        features = model.forward_features(image)
        ((grid, cls),) = model.get_intermediate_layers(
            image, n=1, reshape=True, return_class_token=True
        )
        assert grid.shape == (1, 32, 2, 3)
        torch.testing.assert_close(grid.flatten(2).transpose(1, 2), features["x_norm_patchtokens"])
        torch.testing.assert_close(cls, features["x_norm_clstoken"])
        torch.testing.assert_close(model(image), cls)


@torch.inference_mode()
def test_surface_normals_preserve_downward_and_rightward_slope():
    y, x = torch.meshgrid(torch.arange(5.0), torch.arange(7.0), indexing="ij")
    images = torch.stack((x * 2 + y, torch.zeros_like(x))).reshape(1, 2, 1, 5, 7)
    normals = normal_vector(images)
    expected = F.normalize(torch.tensor([2.0, 1.0, 1.0]), dim=0)
    torch.testing.assert_close(
        normals[0, 0, :, 1:-1, 1:-1], expected[:, None, None].expand(3, 3, 5)
    )
    torch.testing.assert_close(normals[0, 1, 2], torch.ones(5, 7))
    torch.testing.assert_close(normals.square().sum(2), torch.ones(1, 2, 5, 7))


@pytest.mark.parametrize("with_class_token", [False, True])
@torch.inference_mode()
def test_zero_initialized_temporal_head_matches_static_depth_for_any_microbatch(
    reference_model, with_class_token
):
    options = dict(
        in_channels=16, features=32, out_channels=[32] * 4, use_clstoken=with_class_token
    )
    spatial = DPTHead(**options).eval()
    spatial.scratch.output_conv2[2].bias.fill_(1.0)
    temporal = DPTHeadTemporal(**options, num_frames=4).eval()
    loaded = temporal.load_state_dict(spatial.state_dict(), strict=False)
    assert loaded.missing_keys and all(k.startswith("motion_modules.") for k in loaded.missing_keys)
    assert not loaded.unexpected_keys
    features = [(torch.randn(4, 6, 16), torch.randn(4, 16)) for _ in range(4)]
    expected = spatial(features, 2, 3)
    assert expected.shape == (4, 1, 28, 42) and expected.std() > 0
    for microbatch in (2, 4):
        output, history = temporal(features, 2, 3, 4, micro_batch_size=microbatch)
        torch.testing.assert_close(output, expected)
        assert history and all(item.shape[1] == 4 for item in history)


@pytest.mark.parametrize("extra_context", [False, True])
@torch.inference_mode()
def test_attention_slices_match_pytorch_scaled_dot_product_reference(extra_context):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(44)
        layer = CrossAttention(
            8,
            heads=2,
            dim_head=4,
            added_kv_proj_dim=8 if extra_context else None,
            norm_num_groups=2,
            upcast_attention=True,
            upcast_softmax=True,
        ).eval()
        x, context = torch.randn(2, 3, 8), torch.randn(2, 2 if extra_context else 3, 8)
        tokens = layer.group_norm(x.transpose(1, 2)).transpose(1, 2)

        def heads(t):
            return t.reshape(2, -1, 2, 4).transpose(1, 2)

        q = heads(layer.to_q(tokens))
        if extra_context:
            k = heads(torch.cat((layer.add_k_proj(context), layer.to_k(tokens)), dim=1))
            v = heads(torch.cat((layer.add_v_proj(context), layer.to_v(tokens)), dim=1))
            mask = torch.tensor([0.0, -100.0]).expand(2, 1, 2)
            full_mask = F.pad(mask, (0, 3)).unsqueeze(1)
        else:
            k, v = heads(layer.to_k(context)), heads(layer.to_v(context))
            mask = torch.tensor([0.0, 0.0, -100.0]).expand(4, 1, 3)
            full_mask = mask.reshape(2, 2, 1, 3)
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=full_mask)
        expected = layer.to_out[0](expected.transpose(1, 2).reshape_as(x))
        for size in (None, 1):
            layer.set_attention_slice(size)
            torch.testing.assert_close(layer(x, context, mask), expected)


@torch.inference_mode()
def test_cached_temporal_attention_last_frame_matches_full_prefix(monkeypatch):
    monkeypatch.setattr(motion_module, "XFORMERS_AVAILABLE", False)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(417)
        layer = motion_module.TemporalAttention(
            query_dim=16, heads=2, dim_head=8, temporal_max_len=4
        ).eval()
        frames = torch.randn(3, 4, 16)
        _, prefix = layer(frames[:2], video_length=2)
        last, added = layer(frames[2:], video_length=1, cached_hidden_states=prefix)
        full, history = layer(frames, video_length=3)
        torch.testing.assert_close(last, full[-1:])
        torch.testing.assert_close(torch.cat([prefix, added], dim=1), history)


@pytest.mark.parametrize(
    "method,shape", [("lower_bound", (56, 112)), ("upper_bound", (28, 56)), ("minimal", (28, 56))]
)
@pytest.mark.parametrize("portrait", [False, True])
def test_preprocessing_preserves_patch_aspect_and_label_values(method, shape, portrait):
    rows, cols = (80, 40) if portrait else (40, 80)
    shape = shape[::-1] if portrait else shape
    labels = np.ones((rows, cols), np.uint8)
    labels[:, : cols // 2] = 0
    rgb = np.repeat(labels[..., None], 3, axis=-1).astype(np.float32)
    sample = {"image": rgb, "depth": labels.astype(np.float32) * 10, "mask": labels.astype(bool)}
    resized = Resize(56, 56, keep_aspect_ratio=True, resize_method=method, ensure_multiple_of=14)(
        sample
    )
    # Pixel labels must use nearest-neighbour even when RGB is area-resampled.
    np.testing.assert_array_equal(resized["depth"], resized["mask"] * 10)
    assert set(np.unique(resized["mask"])) == {0, 1}
    ready = PrepareForNet()(
        NormalizeImage(np.array([0.2, 0.4, 0.6]), np.array([0.1, 0.2, 0.3]))(resized)
    )
    assert ready["image"].shape == (3, *shape)
    assert all(v.dtype == np.float32 and v.flags.c_contiguous for v in ready.values())
    np.testing.assert_allclose(ready["image"][:, 0, 0], [-2, -2, -2])
    np.testing.assert_allclose(ready["image"][:, -1, -1], [8, 3, 4 / 3], rtol=1e-6)
