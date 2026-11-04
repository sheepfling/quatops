"""Backward-compatible imports for frame semantics now in ``quatops.semantics``."""

from ..semantics import (
    CANONICAL_BODY_FRAME,
    CANONICAL_WORLD_FRAME,
    ECEF_FRAME,
    CoordinateFrame,
    ECEFBodyOrientation,
    FrameCorrespondence,
    FramedOrientation,
    FramedOrientationBatch,
    FrameRole,
    PhysicalOrientation,
    PhysicalOrientationBatch,
    SemanticQuaternionMap,
)

__all__ = [
    "CANONICAL_BODY_FRAME",
    "CANONICAL_WORLD_FRAME",
    "ECEF_FRAME",
    "CoordinateFrame",
    "ECEFBodyOrientation",
    "FrameCorrespondence",
    "FrameRole",
    "FramedOrientation",
    "FramedOrientationBatch",
    "PhysicalOrientation",
    "PhysicalOrientationBatch",
    "SemanticQuaternionMap",
]
