"""Depth estimation inference modules.

Video depth estimators for temporal consistency.
"""

from .video_depth_estimator import (
    VideoDepthEstimator,
    create_video_depth_estimator,
)
from .video_depth_estimator_da3 import create_video_depth_estimator_da3
from .video_depth_estimator_see_through import (
    SeeThroughDepthEstimator,
    create_see_through_depth_estimator,
)
from .video_depth_estimator_moge2 import (
    VideoDepthEstimatorMoGe2,
    create_video_depth_estimator_moge2,
)
from .types import DepthBatch, DepthRepresentation, PinholeCameraBatch
from .backend_registry import (
    BackendAvailability,
    BackendCapabilities,
    DepthBackendSpec,
    EstimatorRequest,
    ModelVariantSpec,
    backend_availability,
    create_registered_depth_estimator,
    get_backend_spec,
    list_backend_specs,
    resolve_model_variant,
)

__all__ = [
    "VideoDepthEstimator",
    "create_video_depth_estimator",
    "create_video_depth_estimator_da3",
    "SeeThroughDepthEstimator",
    "create_see_through_depth_estimator",
    "VideoDepthEstimatorMoGe2",
    "create_video_depth_estimator_moge2",
    "DepthBatch",
    "DepthRepresentation",
    "PinholeCameraBatch",
    "BackendAvailability",
    "BackendCapabilities",
    "DepthBackendSpec",
    "EstimatorRequest",
    "ModelVariantSpec",
    "backend_availability",
    "create_registered_depth_estimator",
    "get_backend_spec",
    "list_backend_specs",
    "resolve_model_variant",
]
