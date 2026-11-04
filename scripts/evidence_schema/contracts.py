"""Build shared contracts from validated target-specific declarations."""

from __future__ import annotations

import re
from typing import Literal

from scripts.converter_evidence import ConverterEvidencePackage
from scripts.evidence_schema.crystallized import (
    CoordinateFrame,
    EvidenceContract,
    FrameAxisMapping,
    FrameChainContract,
    GeographicContract,
    QuaternionBoundaryContract,
    RotationVectorBoundaryContract,
    SceneTransportContract,
)
from scripts.evidence_schema.policies import tolerances_for_target
from scripts.evidence_schema.targets import (
    CesiumContract,
    CesiumInputs,
    EigenContract,
    MuJoCoContract,
    OpenCVContract,
    OpenUsdContract,
    Ros2Capture,
    Ros2Contract,
    SpiceyPyContract,
    ThreeGltfContract,
)


def _order(value: str) -> Literal["wxyz", "xyzw"]:
    normalized = value.lower().replace(" ", "")
    if normalized == "wxyz":
        return "wxyz"
    if normalized == "xyzw":
        return "xyzw"
    raise ValueError(f"unsupported declared quaternion order: {value!r}")


def _handedness(value: str) -> Literal["right-handed", "left-handed"]:
    normalized = value.lower()
    if normalized in {"right", "right-handed"}:
        return "right-handed"
    if normalized in {"left", "left-handed"}:
        return "left-handed"
    raise ValueError(f"unsupported declared handedness: {value!r}")


def _frame(name: str, handedness: str, axes: object) -> CoordinateFrame:
    if not isinstance(axes, dict):
        raise TypeError(f"frame {name!r} must declare its positive axes")
    return CoordinateFrame(
        name=name,
        handedness=_handedness(handedness),
        positive_x=str(axes["positive_x"]),
        positive_y=str(axes["positive_y"]),
        positive_z=str(axes["positive_z"]),
    )


def _axis_map(values: list[str]) -> list[str]:
    if len(values) != 3 or any(
        value not in {"x", "-x", "y", "-y", "z", "-z"} for value in values
    ):
        raise ValueError(f"invalid declared signed-axis map: {values!r}")
    if len({value.lstrip("-") for value in values}) != 3:
        raise ValueError(f"axis map must be a signed permutation: {values!r}")
    return values


def _canonical_frame(contract) -> CoordinateFrame:
    canonical = contract.canonical
    return _frame(
        "QuatOps canonical frame",
        canonical.handedness,
        canonical.frame.model_dump(),
    )


def _probe_frame(contract) -> CoordinateFrame:
    canonical = contract.canonical
    probe = contract.target_boundary.probe_frame_assignment
    canonical_axes = canonical.frame.model_dump()
    probe_axes = probe.model_dump(exclude={"meaning"})
    if any(canonical_axes[key] != probe_axes[key] for key in canonical_axes):
        raise ValueError("target probe frame differs from its declared canonical frame")
    return _frame(
        f"{contract.target.name} explicit numeric probe frame",
        canonical.handedness,
        probe_axes,
    )


def _canonical_order(contract) -> Literal["wxyz", "xyzw"]:
    order = _order(contract.canonical.coefficient_order)
    if contract.canonical.algebra.lower() != "hamilton":
        raise ValueError("converter contract does not declare Hamilton algebra")
    if not contract.canonical.rotation_action.lower().startswith("active"):
        raise ValueError("converter contract does not declare active rotation action")
    if _handedness(contract.canonical.handedness) != "right-handed":
        raise ValueError("this converter product requires an explicit RH source frame")
    return order


def _native_order(boundary) -> Literal["wxyz", "xyzw"]:
    declared_order = getattr(boundary, "coefficient_order", None)
    if isinstance(declared_order, str):
        return _order(declared_order)

    constructor = getattr(boundary, "constructor_order", None)
    if isinstance(constructor, str):
        components = "".join(re.findall(r"\b([wxyz])\b", constructor, re.IGNORECASE))
        if components:
            return _order(components)
    raise ValueError(
        "target contract does not explicitly declare native coefficient order"
    )


def _direct_quaternion_contract(contract) -> QuaternionBoundaryContract:
    boundary = contract.target_boundary
    canonical_order = _canonical_order(contract)
    frame_map = getattr(boundary, "source_to_target_coordinate_map", None)
    if frame_map is None:
        frame_map = boundary.coordinate_map
    if frame_map is None:
        raise ValueError("direct quaternion contract has no declared coordinate map")
    native_order = _native_order(boundary)
    conversion = boundary.canonical_to_native
    inverse = boundary.native_to_canonical
    if not conversion or not inverse:
        raise ValueError("direct quaternion contract must declare both directions")
    return QuaternionBoundaryContract(
        kind="quaternion_boundary",
        canonical_algebra="Hamilton",
        canonical_order=canonical_order,
        rotation_action="active",
        source_frame=_canonical_frame(contract),
        destination_frame=_probe_frame(contract),
        destination_axes_from_source=_axis_map(frame_map),
        native_algebra="Hamilton",
        native_order=native_order,
        native_storage_order=(
            _order(boundary.coeffs_access_order)
            if getattr(boundary, "coeffs_access_order", None)
            else None
        ),
        native_vector_action=(
            getattr(boundary, "matrix_action", None)
            or getattr(boundary, "observed_quat2mat_vector_action", None)
            or getattr(boundary, "observed_q2m_vector_action", None)
            or "native quaternion action captured on the declared coordinate basis"
        ),
        conversion_rule=conversion,
        inverse_rule=inverse,
        input_domain=contract.canonical.normalization,
        tolerances=tolerances_for_target(contract.target.name),
        scope=[
            "canonical-to-native coefficient boundary",
            "native quaternion action on three basis vectors",
            "native-created orientation for reverse conversion",
            "both orders of a noncommuting rotation product",
        ],
        excludes=contract.excludes,
    )


def _rotation_vector_contract(contract) -> RotationVectorBoundaryContract:
    canonical_order = _canonical_order(contract)
    boundary = contract.target_boundary
    if not boundary.rotation_vector_from_canonical or not boundary.native_to_canonical:
        raise ValueError("rotation-vector contract must declare both conversion paths")
    return RotationVectorBoundaryContract(
        kind="rotation_vector_boundary",
        canonical_algebra="Hamilton",
        canonical_order=canonical_order,
        rotation_action="active",
        source_frame=_canonical_frame(contract),
        destination_frame=_probe_frame(contract),
        destination_axes_from_source=_axis_map(boundary.coordinate_map),
        angle_units="radians",
        native_representation=boundary.representation,
        native_matrix_action=boundary.native_matrix_action,
        conversion_rule=boundary.rotation_vector_from_canonical,
        inverse_rule=boundary.native_to_canonical,
        tolerances=tolerances_for_target(contract.target.name),
        scope=[
            "quaternion-to-rotation-vector boundary",
            "native basis and selected mixed-vector actions",
            "native matrix-to-rotation-vector round trip on a mixed orientation",
            "both orders of a noncommuting matrix product",
        ],
        excludes=contract.excludes,
    )


def _ros_contract(package: ConverterEvidencePackage) -> FrameChainContract:
    contract = package.contract
    if not isinstance(contract, Ros2Contract):
        raise TypeError("expected a typed ROS 2 contract")
    capture = package.target_capture
    if not isinstance(capture, Ros2Capture):
        raise TypeError("expected a typed ROS 2 capture")
    forward = capture.native_inputs
    return FrameChainContract(
        kind="frame_chain",
        quaternion_order=_order(contract.rotation.native_order),
        vector_action=contract.rotation.action,
        lookup_direction=contract.frame_direction,
        frame_mappings=[
            FrameAxisMapping(
                frame=forward.target_frame,
                axes_from_canonical=_axis_map(
                    contract.frames.ros_world_from_canonical_world
                ),
            ),
            FrameAxisMapping(
                frame="base",
                axes_from_canonical=_axis_map(
                    contract.frames.ros_base_from_canonical_base
                ),
            ),
            FrameAxisMapping(
                frame=forward.source_frame,
                axes_from_canonical=_axis_map(
                    contract.frames.ros_sensor_optical_from_canonical_sensor
                ),
            ),
        ],
        scope=[
            "named-frame transform direction",
            "two-edge noncommuting frame chain",
            "forward and reverse lookup",
            "message-field round trip",
            "native point action",
        ],
        excludes=contract.exclusions,
    )


def _cesium_contract(package: ConverterEvidencePackage) -> GeographicContract:
    contract = package.contract
    if not isinstance(contract, CesiumContract) or not isinstance(
        package.inputs, CesiumInputs
    ):
        raise TypeError("expected typed Cesium contract and inputs")
    frame_inputs = package.inputs.frame_contract
    storage = contract.storage.upper()
    native_order_match = re.search(r"CESIUM\s+(WXYZ|XYZW)", storage)
    if native_order_match is None:
        raise ValueError("Cesium storage declaration lacks a native component order")
    canonical = frame_inputs.canonical_local
    local = contract.local_frame
    return GeographicContract(
        kind="geographic",
        canonical_frame=_frame(
            "QuatOps canonical local frame",
            canonical.handedness,
            canonical.model_dump(),
        ),
        local_frame=_frame("Cesium local ENU", local.handedness, local.model_dump()),
        local_axes_from_canonical=_axis_map(contract.source_map),
        global_frame_name="ECEF",
        geodetic_datum=contract.geodesy,
        forward_expression=contract.forward,
        inverse_expression=contract.inverse,
        quaternion_order=_order(native_order_match.group(1)),
        scope=[
            "local quaternion placement at multiple geodetic origins",
            "native basis and mixed-vector actions",
            "ECEF point and direction action",
            "inverse point and orientation recovery",
            "antipodal quaternion input",
        ],
        excludes=contract.excludes,
    )


def _scene_contract(package: ConverterEvidencePackage) -> SceneTransportContract:
    contract = package.contract
    if isinstance(contract, ThreeGltfContract):
        canonical = contract.canonical_frame
        asset = contract.asset_frame
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
            canonical_frame=_frame(
                "QuatOps canonical frame", canonical.handedness, canonical.model_dump()
            ),
            asset_frame=_frame(
                "glTF asset frame", asset.handedness, asset.model_dump()
            ),
            destination_axes_from_source=_axis_map(
                contract.destination_axes_from_source
            ),
            quaternion_order=_order(order_match.group(1)),
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
    if isinstance(contract, OpenUsdContract):
        order_match = re.search(
            r"(WXYZ|XYZW)", contract.serialized_orientation, re.IGNORECASE
        )
        if order_match is None:
            raise ValueError("OpenUSD contract lacks a declared serialized order")
        frame_match = re.fullmatch(
            r"(right-handed|left-handed);\s*\+X\s+([^,]+),\s*\+Y\s+([^,]+),\s*\+Z\s+(.+)",
            contract.source.frame,
            re.IGNORECASE,
        )
        if frame_match is None:
            raise ValueError("OpenUSD source frame is not in the declared axis format")
        stage = contract.stage
        return SceneTransportContract(
            kind="scene_transport",
            format_name="OpenUSD USDA",
            format_version=package.target.version,
            loader_name=package.target.name,
            loader_version=package.target.version,
            canonical_frame=_frame(
                "QuatOps canonical frame",
                frame_match.group(1),
                {
                    "positive_x": frame_match.group(2),
                    "positive_y": frame_match.group(3),
                    "positive_z": frame_match.group(4),
                },
            ),
            asset_frame=_frame(
                "OpenUSD stage frame", stage.handedness, stage.model_dump()
            ),
            destination_axes_from_source=_axis_map(
                contract.destination_axes_from_source
            ),
            quaternion_order=_order(order_match.group(1)),
            matrix_action=contract.matrix_action,
            hierarchy_semantics=contract.matrix_action,
            scope=[
                "serialized orient operation readback",
                "local and world transform-stack matrices",
                "parent-child composition and resetXformStack",
                "world point action and inverse point round trip",
            ],
            excludes=contract.excludes,
        )
    raise TypeError("expected a typed scene transport contract")


def crystallize_contract(package: ConverterEvidencePackage) -> EvidenceContract:
    """Normalize a target contract without inferring frame semantics."""
    contract = package.contract
    if isinstance(contract, OpenCVContract):
        return _rotation_vector_contract(contract)
    if isinstance(contract, (EigenContract, MuJoCoContract, SpiceyPyContract)):
        return _direct_quaternion_contract(contract)
    if isinstance(contract, Ros2Contract):
        return _ros_contract(package)
    if isinstance(contract, CesiumContract):
        return _cesium_contract(package)
    if isinstance(contract, (ThreeGltfContract, OpenUsdContract)):
        return _scene_contract(package)
    raise TypeError(f"unsupported target contract {type(contract).__name__}")
