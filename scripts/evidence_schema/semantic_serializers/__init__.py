"""Target-specific semantic adapters for the shared evidence normalizer."""

from __future__ import annotations

from typing import Callable

from scripts.converter_evidence import ConverterEvidencePackage
from scripts.evidence_schema.converter_captures import (
    CesiumContract,
    EigenContract,
    MuJoCoContract,
    OpenCVContract,
    OpenUsdContract,
    Ros2Contract,
    SpiceyPyContract,
    ThreeGltfContract,
)
from scripts.evidence_schema.crystallized import SemanticSerialization
from scripts.evidence_schema.semantic_serializers import (
    cesium,
    eigen,
    gltf,
    mujoco,
    opencv,
    openusd,
    ros2,
    spiceypy,
)
from scripts.evidence_schema.semantic_serializers.context import (
    SemanticSerializationContext,
)

_Serializer = Callable[
    [ConverterEvidencePackage, SemanticSerializationContext], SemanticSerialization
]

_SERIALIZERS: dict[type, _Serializer] = {
    EigenContract: eigen.serialize,
    MuJoCoContract: mujoco.serialize,
    OpenCVContract: opencv.serialize,
    Ros2Contract: ros2.serialize,
    SpiceyPyContract: spiceypy.serialize,
    CesiumContract: cesium.serialize,
    ThreeGltfContract: gltf.serialize,
    OpenUsdContract: openusd.serialize,
}


def serialize_target_semantics(
    package: ConverterEvidencePackage,
    context: SemanticSerializationContext,
) -> SemanticSerialization:
    """Run the serializer registered for one validated target contract type."""
    serializer = _SERIALIZERS.get(type(package.contract))
    if serializer is None:
        raise TypeError(f"no semantic serializer for {type(package.contract).__name__}")
    return serializer(package, context)
