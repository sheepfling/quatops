"""Frame-aware conversion to declared engine, transport, and library layouts.

The scalar front has no mandatory optional dependencies. NumPy batch methods
load NumPy only when called. Target profiles document the evidence scope and
never choose coordinate axes for the caller.
"""

from .frames import (
    CANONICAL_BODY_FRAME,
    ECEF_FRAME,
    CoordinateFrame,
    ECEFBodyOrientation,
    FrameCorrespondence,
    FramedOrientation,
    FramedOrientationBatch,
    SemanticQuaternionMap,
)
from .targets import (
    CONVERTERS,
    TARGET_DECLARATIONS,
    ConverterProfile,
    TargetDeclaration,
    TargetOrientation,
    TargetOrientationBatch,
    get_converter,
    get_target_declaration,
)

__all__ = [
    "CANONICAL_BODY_FRAME",
    "CONVERTERS",
    "ECEF_FRAME",
    "TARGET_DECLARATIONS",
    "ConverterProfile",
    "CoordinateFrame",
    "ECEFBodyOrientation",
    "FrameCorrespondence",
    "FramedOrientation",
    "FramedOrientationBatch",
    "SemanticQuaternionMap",
    "TargetDeclaration",
    "TargetOrientation",
    "TargetOrientationBatch",
    "get_converter",
    "get_target_declaration",
]
