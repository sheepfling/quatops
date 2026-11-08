"""Frame-aware conversion to declared target representations.

The scalar front has no mandatory optional dependencies. NumPy batch methods
load NumPy only when called. Target profiles document the evidence scope and
never choose coordinate axes for the caller. Physical frame semantics live in
``quatops.semantics``; these imports remain available for compatibility.
"""

from ..semantics import (
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
from .gateway import (
    OrientationConverter,
    convert_orientation_batch_from_target,
    convert_orientation_batch_to_target,
    convert_orientation_from_target,
    convert_orientation_to_target,
)
from .targets import (
    CONVERTERS,
    TARGET_DECLARATIONS,
    ConverterProfile,
    TargetDeclaration,
    TargetOrientation,
    TargetOrientationBatch,
    accepted_converters,
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
