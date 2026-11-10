"""Semantic serialization for glTF/Three.js node-transform experiments."""

from __future__ import annotations

import re

from scripts.converter_evidence import ConverterEvidencePackage
from scripts.evidence_schema.converter_captures import (
    ThreeGltfCapture,
    ThreeGltfContract,
    ThreeGltfInputs,
)
from scripts.evidence_schema.crystallized import (
    FrameSemantics,
    SceneNodeWitness,
    SceneTransportContract,
    SceneTransportSerialization,
    canonical_source_frame_semantics,
)
from scripts.evidence_schema.semantic_serializers._contract_helpers import (
    coefficient_order,
    coordinate_frame,
    signed_axis_map,
)
from scripts.evidence_schema.semantic_serializers._witness_helpers import (
    validate_canonical_input_digest,
)
from scripts.evidence_schema.semantic_serializers.context import (
    SemanticSerializationContext,
)


def serialize(
    package: ConverterEvidencePackage, context: SemanticSerializationContext
) -> SceneTransportSerialization:
    """Map glTF node-local and hierarchy observations to scene witnesses."""
    if not isinstance(package.contract, ThreeGltfContract):
        raise TypeError("glTF serializer received a non-glTF contract")
    if not isinstance(package.target_capture, ThreeGltfCapture):
        raise TypeError("glTF serializer received a non-glTF capture")
    if not isinstance(package.inputs, ThreeGltfInputs):
        raise TypeError("glTF serializer requires typed glTF inputs")
    _validate_canonical_inputs(package.inputs, context)
    return SceneTransportSerialization(
        layer="scene-transport",
        contract=serialize_gltf_contract(package),
        witnesses=serialize_gltf_witnesses(package.target_capture, package.inputs),
    )


def serialize_gltf_contract(
    package: ConverterEvidencePackage,
) -> SceneTransportContract:
    """Map glTF loader and caller-assigned probe-frame semantics."""
    if not isinstance(package.contract, ThreeGltfContract):
        raise TypeError("expected a typed glTF contract")
    contract = package.contract
    canonical = contract.canonical_frame
    probe = contract.probe_frame
    order_match = re.search(
        r"GLTF\s+(WXYZ|XYZW)", contract.serialized_storage, re.IGNORECASE
    )
    version_match = re.search(r"glTF\s+(\d+(?:\.\d+)*)", contract.target)
    if order_match is None or version_match is None:
        raise ValueError(
            "glTF contract lacks a declared storage order or format version"
        )
    return SceneTransportContract(
        kind="scene_transport",
        format_name="glTF",
        format_version=version_match.group(1),
        loader_name=package.target.name,
        loader_version=package.target.version,
        canonical_frame=coordinate_frame(
            "QuatOps canonical frame", canonical.handedness, canonical.model_dump()
        ),
        canonical_frame_semantics=canonical_source_frame_semantics(),
        target_frame=coordinate_frame(
            "glTF caller-assigned probe frame", probe.handedness, probe.model_dump()
        ),
        target_frame_semantics=FrameSemantics(
            role="numeric_probe",
            authority="fixture_only",
            statement=contract.frame_role_note,
        ),
        target_frame_role="caller_assigned_probe",
        destination_axes_from_source=signed_axis_map(
            contract.destination_axes_from_source
        ),
        quaternion_order=coefficient_order(order_match.group(1)),
        matrix_action=contract.transform_semantics,
        hierarchy_semantics=contract.transform_semantics,
        scope=[
            "serialized node quaternion and translation readback",
            "native local basis and mixed-vector action",
            "parent-child world transform composition",
            "world point action and inverse point round trip",
        ],
        excludes=contract.excludes,
    )


def _validate_canonical_inputs(
    inputs: ThreeGltfInputs, context: SemanticSerializationContext
) -> None:
    validate_canonical_input_digest(
        inputs.canonical_input_sha256, context, target="glTF"
    )
    known_values = {
        tuple(row.coefficients_wxyz) for row in context.canonical_inputs.orientations
    }
    if any(tuple(row.canonical_wxyz) not in known_values for row in inputs.nodes):
        raise ValueError("glTF node input differs from shared canonical orientations")


def serialize_gltf_witnesses(
    capture: ThreeGltfCapture, inputs: ThreeGltfInputs
) -> list[SceneNodeWitness]:
    input_by_name = {row.node: row for row in inputs.nodes}
    witnesses: list[SceneNodeWitness] = []
    for observation in capture.observations:
        input_name = observation.node.removeprefix("QuatOps").lower()
        item = input_by_name[input_name]
        output = observation.outputs
        witnesses.append(
            SceneNodeWitness(
                kind="scene_node",
                node_id=observation.node,
                source_coefficients_wxyz=item.canonical_wxyz,
                serialized_coefficients=item.gltf_xyzw,
                parent_id="QuatOpsParent" if input_name == "child" else None,
                local_coefficients=output.local_quaternion_xyzw.values,
                coefficient_order="xyzw",
                local_translation_xyz=item.gltf_translation_xyz,
                probe_point_local_xyz=(
                    inputs.gltf_probe_vectors_xyz[-1]
                    if output.world_probe_point_xyz is not None
                    else None
                ),
                local_basis_images_xyz=output.local_basis_images_xyz.values,
                world_basis_images_xyz=(
                    output.world_basis_images_xyz.values
                    if output.world_basis_images_xyz is not None
                    else None
                ),
                world_probe_point_xyz=(
                    output.world_probe_point_xyz.values
                    if output.world_probe_point_xyz is not None
                    else None
                ),
                roundtrip_probe_point_xyz=(
                    output.roundtrip_probe_point_xyz.values
                    if output.roundtrip_probe_point_xyz is not None
                    else None
                ),
            )
        )
    return witnesses
