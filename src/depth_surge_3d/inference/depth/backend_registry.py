"""Central registry for the supported depth-estimation backends."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, cast

from ...core.constants import DEFAULT_DA3_MODEL, MODEL_PATHS, MODEL_PATHS_METRIC
from .video_depth_estimator import create_video_depth_estimator
from .video_depth_estimator_da3 import create_video_depth_estimator_da3
from .video_depth_estimator_see_through import (
    DEFAULT_SEE_THROUGH_REPO,
    create_see_through_depth_estimator,
)


StereoGeometryMode = Literal["relative", "metric_camera"]


@dataclass(frozen=True)
class BackendCapabilities:
    metric_depth: bool
    pinhole_fx: bool
    stereo_geometry_modes: frozenset[StereoGeometryMode]


@dataclass(frozen=True)
class ModelVariantSpec:
    setting: str
    display_name: str
    backend_value: str | None = None
    repo_id: str | None = None
    revision: str | None = None
    parameters_millions: int | None = None


@dataclass(frozen=True)
class BackendAvailability:
    available: bool
    reason: str | None = None
    install_command: str | None = None


@dataclass(frozen=True)
class EstimatorRequest:
    model_path: str | None
    model_size: str | None
    device: str
    metric: bool
    temporal_window_overlap: int


@dataclass(frozen=True)
class DepthBackendSpec:
    backend_id: str
    display_name: str
    default_model_size: str
    variants: Mapping[str, ModelVariantSpec]
    capabilities: BackendCapabilities
    factory: Callable[[EstimatorRequest], Any]
    availability_probe: Callable[[], BackendAvailability]


_RELATIVE_ONLY = frozenset({"relative"})


def _available() -> BackendAvailability:
    return BackendAvailability(available=True)


def _moge_availability() -> BackendAvailability:
    try:
        installed = importlib.util.find_spec("moge.model.v2") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        installed = False
    if installed:
        return BackendAvailability(available=True)
    return BackendAvailability(
        available=False,
        reason="MoGe-2 optional dependency is not installed",
        install_command="uv sync --extra moge2",
    )


def _variants(*variants: ModelVariantSpec) -> Mapping[str, ModelVariantSpec]:
    return MappingProxyType({variant.setting: variant for variant in variants})


def _create_v2(request: EstimatorRequest) -> Any:
    variant = resolve_model_variant("v2", request.model_size)
    paths = MODEL_PATHS_METRIC if request.metric else MODEL_PATHS
    model_path = request.model_path or paths[variant.setting]
    return create_video_depth_estimator(
        model_path,
        request.device,
        request.metric,
        request.temporal_window_overlap,
    )


def _create_v3(request: EstimatorRequest) -> Any:
    variant = resolve_model_variant("v3", request.model_size)
    model_name = request.model_path or cast(str, variant.backend_value)
    return create_video_depth_estimator_da3(model_name, request.device, request.metric)


def _create_see_through(request: EstimatorRequest) -> Any:
    return create_see_through_depth_estimator(
        request.model_path or DEFAULT_SEE_THROUGH_REPO,
        request.device,
        False,
    )


def _create_moge2(request: EstimatorRequest) -> Any:
    """Defer the optional adapter import until MoGe-2 is selected."""
    availability = _moge_availability()
    if not availability.available:
        raise RuntimeError(availability.reason)
    variant = resolve_model_variant("moge2", request.model_size)
    from .video_depth_estimator_moge2 import create_video_depth_estimator_moge2

    return create_video_depth_estimator_moge2(
        model_size=variant.setting,
        model_path=request.model_path,
        repo_id=variant.repo_id,
        revision=variant.revision,
        device=request.device,
    )


_BACKEND_SPECS = (
    DepthBackendSpec(
        backend_id="v2",
        display_name="Video-Depth-Anything V2",
        default_model_size="vitl",
        variants=_variants(
            ModelVariantSpec("vits", "Small"),
            ModelVariantSpec("vitb", "Base"),
            ModelVariantSpec("vitl", "Large"),
        ),
        capabilities=BackendCapabilities(True, False, _RELATIVE_ONLY),
        factory=_create_v2,
        availability_probe=_available,
    ),
    DepthBackendSpec(
        backend_id="v3",
        display_name="Depth Anything V3",
        default_model_size="vitl",
        variants=_variants(
            ModelVariantSpec("vits", "Small", backend_value="small"),
            ModelVariantSpec("vitb", "Base", backend_value="base"),
            ModelVariantSpec("vitl", "Large", backend_value=DEFAULT_DA3_MODEL),
        ),
        capabilities=BackendCapabilities(True, False, _RELATIVE_ONLY),
        factory=_create_v3,
        availability_probe=_available,
    ),
    DepthBackendSpec(
        backend_id="see_through",
        display_name="See-Through Marigold",
        default_model_size="vitl",
        variants=_variants(ModelVariantSpec("vitl", "See-Through")),
        capabilities=BackendCapabilities(False, False, _RELATIVE_ONLY),
        factory=_create_see_through,
        availability_probe=_available,
    ),
    DepthBackendSpec(
        backend_id="moge2",
        display_name="MoGe-2",
        default_model_size="vitb",
        variants=_variants(
            ModelVariantSpec(
                "vits",
                "Small",
                repo_id="Ruicheng/moge-2-vits-normal",
                revision="679230677b4d282c6f304189a93e98e14f085902",
                parameters_millions=35,
            ),
            ModelVariantSpec(
                "vitb",
                "Base",
                repo_id="Ruicheng/moge-2-vitb-normal",
                revision="54ad3a693e61907ea4633d13dec6ee682fa09419",
                parameters_millions=104,
            ),
            ModelVariantSpec(
                "vitl",
                "Large",
                repo_id="Ruicheng/moge-2-vitl",
                revision="39c4d5e957afe587e04eec59dc2bcc3be5ecd968",
                parameters_millions=326,
            ),
        ),
        capabilities=BackendCapabilities(
            metric_depth=True,
            pinhole_fx=True,
            stereo_geometry_modes=frozenset({"relative", "metric_camera"}),
        ),
        factory=_create_moge2,
        availability_probe=_moge_availability,
    ),
)
_BACKEND_SPECS_BY_ID = MappingProxyType({spec.backend_id: spec for spec in _BACKEND_SPECS})


def get_backend_spec(backend_id: str) -> DepthBackendSpec:
    """Return one registered backend, rejecting unknown IDs explicitly."""
    try:
        return _BACKEND_SPECS_BY_ID[backend_id]
    except KeyError as exc:
        raise ValueError(f"Unknown depth backend: {backend_id}") from exc


def list_backend_specs() -> tuple[DepthBackendSpec, ...]:
    """List registered backends in their stable presentation order."""
    return _BACKEND_SPECS


def backend_availability(backend_id: str) -> BackendAvailability:
    """Report whether the selected backend can be constructed in this environment."""
    return get_backend_spec(backend_id).availability_probe()


def resolve_model_variant(backend_id: str, model_size: str | None) -> ModelVariantSpec:
    """Resolve an optional model-size override against a backend's variants."""
    spec = get_backend_spec(backend_id)
    selected_size = model_size or spec.default_model_size
    try:
        return spec.variants[selected_size]
    except KeyError as exc:
        raise ValueError(f"Unknown model size for {backend_id}: {selected_size}") from exc


def create_registered_depth_estimator(backend_id: str, request: EstimatorRequest) -> Any:
    """Construct the selected estimator after validating its backend ID."""
    return get_backend_spec(backend_id).factory(request)
