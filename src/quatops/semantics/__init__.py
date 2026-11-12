"""Deprecated compatibility path; use ``quatops.experimental.semantics`` instead."""

from functools import partial as _partial

from .._experimental_compat import (
    list_attributes as _list_attributes,
)
from .._experimental_compat import (
    resolve_attribute as _resolve_attribute,
)
from .._experimental_compat import (
    warn_compatibility as _warn_compatibility,
)
from ..experimental.semantics import (
    AEROSPACE_BODY_FRD,
    CANONICAL_BODY_FRAME,
    CANONICAL_BODY_RFU,
    CANONICAL_WORLD_FRAME,
    CANONICAL_WORLD_RFU,
    ECEF_FRAME,
    ECEF_WORLD,
    OPENXR_SPACE_RUB,
    ROS_BODY_FLU,
    ROS_CAMERA_OPTICAL,
    WORLD_ENU,
    WORLD_NED,
    CoordinateFrame,
    ECEFBodyOrientation,
    FrameConvention,
    FrameCorrespondence,
    FramedOrientation,
    FramedOrientationBatch,
    FrameRole,
    PhysicalOrientation,
    PhysicalOrientationBatch,
    SemanticQuaternionMap,
)

__all__ = [
    "AEROSPACE_BODY_FRD",
    "CANONICAL_BODY_FRAME",
    "CANONICAL_BODY_RFU",
    "CANONICAL_WORLD_FRAME",
    "CANONICAL_WORLD_RFU",
    "ECEF_FRAME",
    "ECEF_WORLD",
    "OPENXR_SPACE_RUB",
    "ROS_BODY_FLU",
    "ROS_CAMERA_OPTICAL",
    "WORLD_ENU",
    "WORLD_NED",
    "CoordinateFrame",
    "ECEFBodyOrientation",
    "FrameConvention",
    "FrameCorrespondence",
    "FrameRole",
    "FramedOrientation",
    "FramedOrientationBatch",
    "PhysicalOrientation",
    "PhysicalOrientationBatch",
    "SemanticQuaternionMap",
]

_warn_compatibility(__name__, "quatops.experimental.semantics")
__getattr__ = _partial(_resolve_attribute, "quatops.experimental.semantics")
__dir__ = _partial(_list_attributes, globals(), "quatops.experimental.semantics")
