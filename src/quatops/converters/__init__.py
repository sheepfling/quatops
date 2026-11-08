"""Deprecated compatibility path; use ``quatops.experimental.converters`` instead."""

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
from ..experimental.converters import (
    AEROSPACE_BODY_FRD,
    CANONICAL_BODY_FRAME,
    CANONICAL_BODY_RFU,
    CANONICAL_WORLD_FRAME,
    CANONICAL_WORLD_RFU,
    CONVERTERS,
    ECEF_FRAME,
    ECEF_WORLD,
    OPENXR_SPACE_RUB,
    ROS_BODY_FLU,
    ROS_CAMERA_OPTICAL,
    TARGET_DECLARATIONS,
    WORLD_ENU,
    WORLD_NED,
    ConverterProfile,
    CoordinateFrame,
    ECEFBodyOrientation,
    FrameConvention,
    FrameCorrespondence,
    FramedOrientation,
    FramedOrientationBatch,
    FrameRole,
    OrientationConverter,
    PhysicalOrientation,
    PhysicalOrientationBatch,
    SemanticQuaternionMap,
    TargetDeclaration,
    TargetOrientation,
    TargetOrientationBatch,
    accepted_converters,
    convert_orientation_batch_from_target,
    convert_orientation_batch_to_target,
    convert_orientation_from_target,
    convert_orientation_to_target,
    get_converter,
    get_target_declaration,
)

__all__ = [
    "AEROSPACE_BODY_FRD",
    "CANONICAL_BODY_FRAME",
    "CANONICAL_BODY_RFU",
    "CANONICAL_WORLD_FRAME",
    "CANONICAL_WORLD_RFU",
    "CONVERTERS",
    "ECEF_FRAME",
    "ECEF_WORLD",
    "OPENXR_SPACE_RUB",
    "ROS_BODY_FLU",
    "ROS_CAMERA_OPTICAL",
    "TARGET_DECLARATIONS",
    "WORLD_ENU",
    "WORLD_NED",
    "ConverterProfile",
    "CoordinateFrame",
    "ECEFBodyOrientation",
    "FrameConvention",
    "FrameCorrespondence",
    "FrameRole",
    "FramedOrientation",
    "FramedOrientationBatch",
    "OrientationConverter",
    "PhysicalOrientation",
    "PhysicalOrientationBatch",
    "SemanticQuaternionMap",
    "TargetDeclaration",
    "TargetOrientation",
    "TargetOrientationBatch",
    "accepted_converters",
    "convert_orientation_batch_from_target",
    "convert_orientation_batch_to_target",
    "convert_orientation_from_target",
    "convert_orientation_to_target",
    "get_converter",
    "get_target_declaration",
]

_warn_compatibility(__name__, "quatops.experimental.converters")
__getattr__ = _partial(_resolve_attribute, "quatops.experimental.converters")
__dir__ = _partial(_list_attributes, globals(), "quatops.experimental.converters")
