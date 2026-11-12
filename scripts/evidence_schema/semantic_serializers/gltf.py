"""Semantic serialization for glTF/Three.js node-transform experiments."""

from __future__ import annotations

from typing import Literal

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
    derive_signed_axis_map,
    frame_axes,
    frame_signature,
    map_hamilton_quaternion_wxyz,
    map_vector_components,
    signed_axis_map,
    validate_mapped_vectors,
)
from scripts.evidence_schema.semantic_serializers._witness_helpers import (
    canonical_probe_vectors,
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
    _validate_canonical_inputs(package.contract, package.inputs, context)
    _validate_observations(package.target_capture)
    return SceneTransportSerialization(
        layer="scene-transport",
        contract=serialize_gltf_contract(package),
        witnesses=serialize_gltf_witnesses(
            package.target_capture,
            package.inputs,
            coefficient_order(package.contract.serialized_storage_order),
        ),
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
    if (package.target.name, package.target.version) != (
        contract.target.name,
        contract.target.version,
    ):
        raise ValueError("glTF manifest target differs from semantic contract")
    return SceneTransportContract(
        kind="scene_transport",
        format_name=contract.format_name,
        format_version=contract.format_version,
        loader_name=contract.target.name,
        loader_version=contract.target.version,
        orientation_field="node.rotation",
        meters_per_unit=contract.meters_per_unit,
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
        quaternion_order=coefficient_order(contract.serialized_storage_order),
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
    contract: ThreeGltfContract,
    inputs: ThreeGltfInputs,
    context: SemanticSerializationContext,
) -> None:
    validate_canonical_input_digest(
        inputs.canonical_input_sha256, context, target="glTF"
    )
    known_values = {
        tuple(row.coefficients_wxyz) for row in context.canonical_inputs.orientations
    }
    input_frames = inputs.frame_contract
    if (
        input_frames.canonical.handedness != contract.canonical_frame.handedness
        or frame_axes(input_frames.canonical) != frame_axes(contract.canonical_frame)
        or frame_axes(input_frames.canonical)
        != frame_axes(context.canonical_inputs.source_frame)
        or context.canonical_inputs.source_frame.handedness != "right-handed"
    ):
        raise ValueError("glTF canonical frame differs from shared or target contract")
    if (
        contract.algebra != context.canonical_inputs.contract.algebra
        or contract.rotation_action != context.canonical_inputs.contract.rotation_action
        or contract.source_storage_order.upper()
        != context.canonical_inputs.contract.coefficient_order.upper()
    ):
        raise ValueError(
            "glTF quaternion contract differs from shared canonical contract"
        )
    if frame_signature(input_frames.gltf_probe_frame) != frame_signature(
        contract.probe_frame
    ):
        raise ValueError("glTF input probe frame differs from target contract")
    if (
        input_frames.destination_axes_from_canonical
        != contract.destination_axes_from_source
    ):
        raise ValueError("glTF input axis map differs from target contract")
    derived_map = derive_signed_axis_map(
        frame_axes(contract.canonical_frame), frame_axes(contract.probe_frame)
    )
    if contract.destination_axes_from_source != derived_map:
        raise ValueError("glTF frame meanings disagree with the declared axis map")
    expected_quaternion_semantics = (
        f"{contract.algebra} {contract.rotation_action}; "
        "glTF rotation property stores "
        f"{contract.serialized_storage_order}"
    )
    if input_frames.quaternion != expected_quaternion_semantics:
        raise ValueError("glTF input quaternion semantics differ from contract")
    expected_nodes = {"parent", "child"}
    input_node_names = [row.node for row in inputs.nodes]
    if (
        len(set(input_node_names)) != len(input_node_names)
        or set(input_node_names) != expected_nodes
    ):
        raise ValueError("glTF inputs must contain one parent and one child node")
    source_probes = canonical_probe_vectors(context)
    if inputs.probe_vectors_xyz != source_probes:
        raise ValueError("glTF source probes differ from shared inputs")
    validate_mapped_vectors(
        inputs.gltf_probe_vectors_xyz,
        source_probes,
        contract.destination_axes_from_source,
        context="glTF probe vectors",
    )
    for row in inputs.nodes:
        if tuple(row.canonical_wxyz) not in known_values:
            raise ValueError(
                "glTF node input differs from shared canonical orientations"
            )
        expected_quaternion_wxyz = map_hamilton_quaternion_wxyz(
            row.canonical_wxyz, contract.destination_axes_from_source
        )
        expected_quaternion_xyzw = (
            expected_quaternion_wxyz[1:] + expected_quaternion_wxyz[:1]
        )
        if row.gltf_xyzw != expected_quaternion_xyzw:
            raise ValueError(f"glTF {row.node} quaternion differs from its axis map")
        expected_translation = map_vector_components(
            row.canonical_translation_xyz, contract.destination_axes_from_source
        )
        if row.gltf_translation_xyz != expected_translation:
            raise ValueError(f"glTF {row.node} translation differs from its axis map")


def _validate_observations(capture: ThreeGltfCapture) -> None:
    expected_entry_points = {
        "GLTFLoader.parse",
        "Object3D.updateMatrixWorld",
        "Object3D.localToWorld",
        "Matrix4.invert",
    }
    direct_entry_points = (expected_entry_points - {"Object3D.localToWorld"}) | {
        "Vector3.applyMatrix4"
    }
    if len(set(capture.native_entry_points)) != len(capture.native_entry_points) or set(
        capture.native_entry_points
    ) not in (expected_entry_points, direct_entry_points):
        raise ValueError(
            "glTF native entry points differ from the pinned capture scope"
        )
    observation_names = [row.node for row in capture.observations]
    if len(set(observation_names)) != len(observation_names) or set(
        observation_names
    ) != {"QuatOpsParent", "QuatOpsChild"}:
        raise ValueError("glTF capture must contain one parent and one child node")


def serialize_gltf_witnesses(
    capture: ThreeGltfCapture,
    inputs: ThreeGltfInputs,
    native_order: Literal["wxyz", "xyzw"],
) -> list[SceneNodeWitness]:
    input_by_name = {row.node: row for row in inputs.nodes}
    capture_to_input = {"QuatOpsParent": "parent", "QuatOpsChild": "child"}
    witnesses: list[SceneNodeWitness] = []
    for observation in capture.observations:
        input_name = capture_to_input[observation.node]
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
                coefficient_order=native_order,
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
