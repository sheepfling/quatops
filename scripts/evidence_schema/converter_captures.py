"""Typed native observation schemas for supported converter capture targets."""

# Pydantic annotations must remain importable on the supported Python 3.9 floor.
# ruff: noqa: UP045

from __future__ import annotations

import math
from typing import Annotated, Literal, Optional, Union

from pydantic import ConfigDict, Field, model_validator

from scripts.evidence_schema.primitives import (
    NATIVE_RETAINED_CASE_IDS,
    SHA256,
    Matrix3,
    Matrix4,
    NativeCaptureModel,
    Quaternion4,
    Scalar,
    Vector3,
)

BasisActions = Annotated[list[Vector3], Field(min_length=3, max_length=3)]
NumberList9 = Annotated[list[Scalar], Field(min_length=9, max_length=9)]


class TargetIdentity(NativeCaptureModel):
    name: str
    version: Optional[str] = None
    package_version: Optional[str] = None
    engine_version: Optional[str] = None
    wrapper_version: Optional[str] = None
    distribution_version: Optional[str] = None
    toolkit_version: Optional[str] = None
    header: Optional[str] = None
    interface: Optional[str] = None
    distribution: Optional[str] = None
    native_scalar_dtype: Optional[str] = None
    native_scalar_profiles: Optional[list[str]] = None
    version_macros: Optional[list[int]] = None


class ContractTargetIdentity(NativeCaptureModel):
    name: str
    version: Optional[str] = None
    distro: Optional[str] = None
    tf2_version: Optional[str] = None
    native_scalar_dtype: Optional[str] = None
    native_scalar_bytes: Optional[int] = None
    scalar_profiles: Optional[list[str]] = None
    entry_points: Optional[list[str]] = None
    interface: Optional[str] = None
    header: Optional[str] = None
    api_version: Optional[str] = None
    distribution: Optional[str] = None
    openusd_version_tuple: Optional[list[int]] = None


class CanonicalAxesContract(NativeCaptureModel):
    positive_x: str
    positive_y: str
    positive_z: str


class ContractFrame(CanonicalAxesContract):
    handedness: str


class CanonicalRotationContract(NativeCaptureModel):
    algebra: str
    coefficient_order: str
    rotation_action: str
    matrix_action: str
    handedness: str
    normalization: str
    frame: CanonicalAxesContract


class ProbeFrameAssignment(CanonicalAxesContract):
    meaning: str


class BoundaryContractBase(NativeCaptureModel):
    native_to_canonical: str
    probe_frame_assignment: ProbeFrameAssignment


class LibraryContractBase(NativeCaptureModel):
    schema_version: Literal["1.0"]
    contract_id: str
    target: ContractTargetIdentity
    canonical: CanonicalRotationContract
    scope: list[str]
    excludes: list[str]


class EigenBoundaryContract(BoundaryContractBase):
    constructor_order: str
    coeffs_access_order: str
    matrix_action: str
    observed_product: str
    source_to_target_coordinate_map: Annotated[
        list[str], Field(min_length=3, max_length=3)
    ]
    canonical_to_native: str
    native_matrix_storage: str


class DirectQuaternionBoundaryContract(BoundaryContractBase):
    coefficient_order: str
    observed_product: str
    source_to_target_coordinate_map: Annotated[
        list[str], Field(min_length=3, max_length=3)
    ]
    native_matrix_layout: str
    canonical_to_native: str
    reverse_matrix_route: str


class MuJoCoBoundaryContract(DirectQuaternionBoundaryContract):
    observed_quat2mat_vector_action: str


class OpenCVBoundaryContract(BoundaryContractBase):
    representation: str
    rotation_vector_from_canonical: str
    native_matrix_action: str
    coordinate_map: Annotated[list[str], Field(min_length=3, max_length=3)]
    matrix_layout: str


class SpiceyPyBoundaryContract(DirectQuaternionBoundaryContract):
    observed_q2m_vector_action: str


class EigenContract(LibraryContractBase):
    target_boundary: EigenBoundaryContract


class MuJoCoContract(LibraryContractBase):
    target_boundary: MuJoCoBoundaryContract


class OpenCVContract(LibraryContractBase):
    target_boundary: OpenCVBoundaryContract


class SpiceyPyContract(LibraryContractBase):
    target_boundary: SpiceyPyBoundaryContract


class RosFramesContract(NativeCaptureModel):
    canonical_axes: ContractFrame
    ros_world_from_canonical_world: Annotated[
        list[str], Field(min_length=3, max_length=3)
    ]
    ros_base_from_canonical_base: Annotated[
        list[str], Field(min_length=3, max_length=3)
    ]
    ros_sensor_optical_from_canonical_sensor: Annotated[
        list[str], Field(min_length=3, max_length=3)
    ]
    interpretation: str


class RosRotationContract(NativeCaptureModel):
    algebra: str
    native_order: str
    action: str
    units: str


class Ros2Contract(NativeCaptureModel):
    schema_version: Optional[Literal["1.0"]] = None
    contract_id: str
    target: ContractTargetIdentity
    native_api: list[str]
    rotation: RosRotationContract
    frames: RosFramesContract
    frame_direction: str
    conversion: str
    inverse_conversion: str
    input_domain: str
    scope: list[str]
    exclusions: list[str]
    native_witnesses: Optional[str] = None


class GeographicFrameContract(ContractFrame):
    pass


class CesiumContract(NativeCaptureModel):
    schema_version: Optional[Literal["1.0"]] = None
    contract_id: str
    target: str
    algebra: str
    storage: str
    local_frame: GeographicFrameContract
    source_map: Annotated[list[str], Field(min_length=3, max_length=3)]
    native_api: list[str]
    forward: str
    inverse: str
    geodesy: str
    normalization: str
    scope: list[str]
    excludes: list[str]


class SceneFrameContract(ContractFrame):
    forward: Optional[str] = None


class ThreeGltfContract(NativeCaptureModel):
    schema_version: Optional[Literal["1.0"]] = None
    contract_id: str
    target: str
    algebra: str
    source_storage: str
    serialized_storage: str
    canonical_frame: SceneFrameContract
    probe_frame: SceneFrameContract
    frame_role_note: str
    destination_axes_from_source: Annotated[
        list[str], Field(min_length=3, max_length=3)
    ]
    transform_semantics: str
    scope: list[str]
    excludes: list[str]


class OpenUsdSourceContract(NativeCaptureModel):
    algebra: str
    storage: str
    frame: str


class OpenUsdStageContract(NativeCaptureModel):
    handedness: str
    upAxis: str
    positive_x: str
    positive_y: str
    positive_z: str
    forward: str
    metersPerUnit: float


class OpenUsdContract(NativeCaptureModel):
    schema_version: Optional[Literal["1.0"]] = None
    contract_id: str
    target: str
    source: OpenUsdSourceContract
    stage: OpenUsdStageContract
    destination_axes_from_source: Annotated[
        list[str], Field(min_length=3, max_length=3)
    ]
    serialized_orientation: str
    matrix_action: str
    native_numeric_type: str
    scope: list[str]
    excludes: list[str]


TargetContract = Union[
    EigenContract,
    MuJoCoContract,
    OpenCVContract,
    SpiceyPyContract,
    Ros2Contract,
    CesiumContract,
    ThreeGltfContract,
    OpenUsdContract,
]


class NativeConstructedInput(NativeCaptureModel):
    case_id: Optional[str] = None
    axis_xyz: Vector3
    angle_radians: Scalar


class CompositionInput(NativeCaptureModel):
    left_case_id: str
    right_case_id: str
    product_order: Optional[str] = None


class DirectLibraryInputBase(NativeCaptureModel):
    schema_version: Literal["1.0"]
    contract_id: str
    native_constructed_orientation: NativeConstructedInput
    shared_canonical_inputs: str
    retained_case_ids: list[str]


class EigenInputs(DirectLibraryInputBase):
    scalar_profiles: list[Literal["float32", "float64"]]
    composition: CompositionInput


class MuJoCoInputs(DirectLibraryInputBase):
    pass


class OpenCVInputs(DirectLibraryInputBase):
    precision_profiles: list[Literal["float32", "float64"]]


class SpiceyPyInputs(DirectLibraryInputBase):
    @model_validator(mode="after")
    def validate_native_constructor(self) -> SpiceyPyInputs:
        constructor = self.native_constructed_orientation
        if constructor.case_id != SPICEYPY_CONSTRUCTOR_CASE_ID:
            raise ValueError(
                f"SpiceyPy constructor case_id must be {SPICEYPY_CONSTRUCTOR_CASE_ID!r}"
            )
        if tuple(self.retained_case_ids) != SPICEYPY_CAPTURE_INPUT_CASE_IDS:
            raise ValueError(
                "SpiceyPy inputs must retain the complete ordered recipe selection"
            )
        if not all(math.isfinite(value) for value in constructor.axis_xyz):
            raise ValueError(
                "SpiceyPy constructor axis must contain only finite values"
            )
        if not math.isfinite(constructor.angle_radians):
            raise ValueError("SpiceyPy constructor angle must be finite")
        return self


class QuaternionValues(NativeCaptureModel):
    wxyz: Quaternion4
    coeffs_xyzw: Optional[Quaternion4] = None


class EigenMatrix(NativeCaptureModel):
    rows: Matrix3
    column_major_storage: NumberList9


class EigenOrientation(NativeCaptureModel):
    case_id: str
    quaternion: QuaternionValues
    rotation_matrix: Optional[EigenMatrix] = None
    native_quaternion_basis_actions_xyz: BasisActions
    native_quaternion_probe_action_xyz: Optional[Vector3] = None
    matrix_to_quaternion: Optional[QuaternionValues] = None


class EigenConstructedOrientation(EigenOrientation):
    constructor: str


class EigenComposition(NativeCaptureModel):
    case_id: str
    lhs_case_id: str
    rhs_case_id: str
    product_order: str
    quaternion: QuaternionValues
    native_quaternion_basis_actions_xyz: BasisActions


class EigenPrecisionProfile(NativeCaptureModel):
    dtype: Literal["float32", "float64"]
    scalar_bytes: Literal[4, 8]
    orientations: list[EigenOrientation]
    native_created: list[EigenConstructedOrientation]
    composition: list[EigenComposition]


class EigenObservations(NativeCaptureModel):
    profiles: list[EigenPrecisionProfile]


class EigenCapture(NativeCaptureModel):
    schema_version: Literal["1.0"]
    evidence_kind: Literal["native_runtime_selected_converter_witnesses"]
    target: TargetIdentity
    observations: EigenObservations


class MuJoCoOrientation(NativeCaptureModel):
    case_id: str
    native_input_wxyz: Quaternion4
    native_mju_rotVecQuat_basis_xyz: BasisActions
    native_mju_rotVecQuat_probe_xyz: Optional[Vector3] = None
    native_constructor: Optional[str] = None
    native_lhs_wxyz: Optional[Quaternion4] = None
    native_rhs_wxyz: Optional[Quaternion4] = None
    product_order: Optional[str] = None


class MuJoCoObservations(NativeCaptureModel):
    canonical_to_native: list[MuJoCoOrientation]
    native_created: list[MuJoCoOrientation]
    composition: list[MuJoCoOrientation]


class MuJoCoCapture(NativeCaptureModel):
    schema_version: Literal["1.0"]
    evidence_kind: Literal["native_runtime_selected_converter_witnesses"]
    target: TargetIdentity
    observations: MuJoCoObservations


class OpenCVOrientation(NativeCaptureModel):
    case_id: str
    dtype: Literal["float32", "float64"]
    native_input_rotation_vector_xyz_radians: Vector3
    native_gemm_basis_actions_xyz: BasisActions
    native_gemm_probe_action_xyz: Optional[Vector3] = None
    native_matrix_to_rotation_vector_xyz_radians: Vector3


class OpenCVObservations(NativeCaptureModel):
    canonical_to_native: list[OpenCVOrientation]
    native_created: list[OpenCVOrientation]
    composition: list[OpenCVComposition]


class OpenCVComposition(NativeCaptureModel):
    case_id: str
    dtype: Literal["float32", "float64"]
    native_lhs_case_id: str
    native_rhs_case_id: str
    product_order: str
    native_gemm_basis_actions_xyz: BasisActions
    native_matrix_to_rotation_vector_xyz_radians: Vector3


class OpenCVCapture(NativeCaptureModel):
    schema_version: Literal["1.0"]
    evidence_kind: Literal["native_runtime_selected_converter_witnesses"]
    target: TargetIdentity
    observations: OpenCVObservations


SPICEYPY_CAPTURE_INPUT_CASE_IDS = (
    "identity",
    "quarter_x",
    "quarter_y",
    "quarter_z",
    "mixed_a",
    "mixed_b",
    "antipode_mixed_a",
    "nonprincipal_halfturn",
)
SPICEYPY_RETAINED_CASE_IDS = NATIVE_RETAINED_CASE_IDS
SPICEYPY_CONSTRUCTOR_CASE_ID = "native_axis_angle_mixed"
SPICEYPY_COMPOSITION_CASE_IDS = (
    "mixed_a_times_mixed_b",
    "mixed_b_times_mixed_a",
)


class SpiceyPyOrientation(NativeCaptureModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )

    case_id: str
    native_input_wxyz: Quaternion4
    native_mxv_basis_xyz: BasisActions
    native_mxv_probe_xyz: Optional[Vector3] = None
    native_lhs_wxyz: Optional[Quaternion4] = None
    native_rhs_wxyz: Optional[Quaternion4] = None
    product_order: Optional[str] = None


class SpiceyPyConstructedOrientation(NativeCaptureModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )

    case_id: str
    native_constructor: str
    native_m2q_wxyz: Quaternion4
    native_mxv_basis_xyz: BasisActions


class SpiceyPyObservations(NativeCaptureModel):
    canonical_to_native: list[SpiceyPyOrientation]
    native_created: list[SpiceyPyConstructedOrientation]
    composition: list[SpiceyPyOrientation]

    @model_validator(mode="after")
    def validate_complete_recipe(self) -> SpiceyPyObservations:
        canonical_ids = tuple(row.case_id for row in self.canonical_to_native)
        if canonical_ids != SPICEYPY_RETAINED_CASE_IDS:
            raise ValueError(
                "SpiceyPy capture must contain the complete ordered canonical selection"
            )
        constructed_ids = tuple(row.case_id for row in self.native_created)
        if constructed_ids != (SPICEYPY_CONSTRUCTOR_CASE_ID,):
            raise ValueError(
                "SpiceyPy capture must contain its one selected constructed orientation"
            )
        composition_ids = tuple(row.case_id for row in self.composition)
        if composition_ids != SPICEYPY_COMPOSITION_CASE_IDS:
            raise ValueError(
                "SpiceyPy capture must contain both ordered composition products"
            )
        for row in self.composition:
            if (
                row.native_lhs_wxyz is None
                or row.native_rhs_wxyz is None
                or row.product_order != "lhs * rhs; rhs acts first"
            ):
                raise ValueError(
                    f"SpiceyPy product {row.case_id!r} lacks its ordered operands"
                )
        return self


class SpiceyPyCapture(NativeCaptureModel):
    schema_version: Literal["1.0"]
    evidence_kind: Literal["native_runtime_selected_converter_witnesses"]
    target: TargetIdentity
    observations: SpiceyPyObservations


class TaggedVector3(NativeCaptureModel):
    values: Vector3
    dtype: Literal["binary64", "float32", "float64"]


class TaggedQuaternion(NativeCaptureModel):
    values: Quaternion4
    dtype: Literal["binary64", "float32", "float64"]


class TaggedMatrix4(NativeCaptureModel):
    values: Matrix4
    dtype: Literal["binary64", "float32", "float64"]


class TaggedVectors3(NativeCaptureModel):
    values: list[Vector3]
    dtype: Literal["binary64", "float32", "float64"]


class RosTransform(NativeCaptureModel):
    quaternion_xyzw: TaggedQuaternion
    translation_xyz: TaggedVector3


class RosTransformMessage(NativeCaptureModel):
    child_frame: str
    parent_frame: str
    rotation_xyzw: TaggedQuaternion
    stamp_nanosec: int
    stamp_sec: int
    translation_xyz: TaggedVector3


class RosInputs(NativeCaptureModel):
    base_from_sensor: RosTransform
    probe_in_source_xyz: TaggedVector3
    source_frame: str
    stamp_sec: int
    target_frame: str
    world_from_base: RosTransform


class RosOutputs(NativeCaptureModel):
    lookup: RosTransform
    lookup_message: Optional[RosTransformMessage] = None
    message_roundtrip: Optional[RosTransform] = None
    probe_in_target_xyz: Optional[TaggedVector3] = None


class RosObservation(NativeCaptureModel):
    operation_id: str
    native_outputs: RosOutputs


class Ros2Capture(NativeCaptureModel):
    schema_version: Literal["1.0"]
    evidence_kind: Literal["native_ros2_tf2_buffer_core_capture"]
    selected_from_full_observation_count: int
    native_inputs: RosInputs
    observations: Annotated[list[RosObservation], Field(min_length=2, max_length=2)]


class Binary64Vector3(NativeCaptureModel):
    values: Vector3
    dtype: Literal["binary64"]


class Binary64Quaternion(NativeCaptureModel):
    values: Quaternion4
    dtype: Literal["binary64"]


class CesiumNumericVectors3(NativeCaptureModel):
    values: list[Vector3]
    dtype: Literal["binary64"]


class CesiumOutputs(NativeCaptureModel):
    local_basis_and_probe_images: Optional[CesiumNumericVectors3] = None
    model_to_ecef_quaternion_xyzw: Binary64Quaternion
    local_quaternion_recovered_from_ecef_xyzw: Binary64Quaternion
    world_points_xyz_m: CesiumNumericVectors3
    world_directions_xyz: CesiumNumericVectors3
    point_roundtrip_local_xyz_m: CesiumNumericVectors3
    ecef_origin_xyz_m: Binary64Vector3


class CesiumObservation(NativeCaptureModel):
    case_id: str
    outputs: CesiumOutputs


class CesiumCapture(NativeCaptureModel):
    schema_version: Literal["1.0"]
    evidence_kind: Literal["native_cesium_quatops_converter_capture"]
    observations: Annotated[list[CesiumObservation], Field(min_length=5, max_length=5)]


class CesiumInputCase(NativeCaptureModel):
    case_id: str
    canonical_orientation_case: str
    location_case: str
    canonical_wxyz: Quaternion4
    quatops_cesium_local_xyzw: Quaternion4
    local_translation_xyz_m: Vector3
    longitude_degrees: Scalar
    latitude_degrees: Scalar
    height_m: Scalar


class CesiumInputFrame(NativeCaptureModel):
    handedness: Literal["right"]
    positive_x: str
    positive_y: str
    positive_z: str
    forward: Optional[str] = None
    upAxis: Optional[str] = None


class CesiumInputFrameContract(NativeCaptureModel):
    canonical_local: CesiumInputFrame
    cesium_local: CesiumInputFrame
    map_canonical_to_cesium_enu: Annotated[list[str], Field(min_length=3, max_length=3)]
    destination_local_api: str
    world_placement: str


class CesiumInputs(NativeCaptureModel):
    schema_version: Literal["1.0"]
    kind: Literal["quatops_generated_converter_inputs"]
    canonical_input_sha256: SHA256
    geographic_input_sha256: SHA256
    frame_contract: CesiumInputFrameContract
    probe_vectors_xyz: Annotated[list[Vector3], Field(min_length=4, max_length=4)]
    cases: list[CesiumInputCase]


class ThreeNumericBasisImages(NativeCaptureModel):
    values: Annotated[list[Vector3], Field(min_length=3, max_length=3)]
    dtype: Literal["binary64"]


class ThreeOutputs(NativeCaptureModel):
    local_quaternion_xyzw: Binary64Quaternion
    local_basis_images_xyz: ThreeNumericBasisImages
    world_basis_images_xyz: Optional[ThreeNumericBasisImages] = None
    world_probe_point_xyz: Optional[Binary64Vector3] = None
    roundtrip_probe_point_xyz: Optional[Binary64Vector3] = None


class ThreeNodeObservation(NativeCaptureModel):
    node: str
    outputs: ThreeOutputs


class ThreeGltfCapture(NativeCaptureModel):
    schema_version: Literal["1.0"]
    evidence_kind: Literal["native_threejs_gltf_converter_capture"]
    native_entry_points: list[str]
    observations: Annotated[
        list[ThreeNodeObservation], Field(min_length=2, max_length=2)
    ]


class ThreeInputNode(NativeCaptureModel):
    node: str
    canonical_wxyz: Quaternion4
    gltf_xyzw: Quaternion4
    canonical_translation_xyz: Vector3
    gltf_translation_xyz: Vector3


class ThreeGltfInputs(NativeCaptureModel):
    schema_version: Literal["1.0"]
    kind: Literal["quatops_generated_gltf_converter_input"]
    canonical_input_sha256: SHA256
    frame_contract: ThreeGltfInputFrameContract
    probe_vectors_xyz: Annotated[list[Vector3], Field(min_length=4, max_length=4)]
    gltf_probe_vectors_xyz: Annotated[list[Vector3], Field(min_length=4, max_length=4)]
    nodes: Annotated[list[ThreeInputNode], Field(min_length=2, max_length=2)]


class ThreeGltfInputFrameContract(NativeCaptureModel):
    canonical: CesiumInputFrame
    gltf_probe_frame: CesiumInputFrame
    destination_axes_from_canonical: Annotated[
        list[str], Field(min_length=3, max_length=3)
    ]
    quaternion: str
    translation: str
    scope: str


class OpenUsdOutputs(NativeCaptureModel):
    orient_op_wxyz: Quaternion4
    resets_xform_stack: bool
    world_basis_images_xyz: Annotated[list[Vector3], Field(min_length=3, max_length=3)]
    world_probe_point_xyz: Vector3
    roundtrip_probe_point_xyz: Optional[Vector3] = None


class OpenUsdNodeObservation(NativeCaptureModel):
    node: str
    prim_path: str
    outputs: OpenUsdOutputs


class OpenUsdCapture(NativeCaptureModel):
    schema_version: Literal["1.0"]
    evidence_kind: Literal["native_openusd_quatops_converter_capture"]
    native_entry_points: list[str]
    ordered_xform_ops: list[str]
    observations: Annotated[
        list[OpenUsdNodeObservation], Field(min_length=3, max_length=3)
    ]


class OpenUsdInputNode(NativeCaptureModel):
    node: str
    canonical_case_id: str
    canonical_wxyz: Quaternion4
    usd_wxyz: Quaternion4
    canonical_translation_xyz: Vector3
    usd_translation_xyz: Vector3


class OpenUsdInputs(NativeCaptureModel):
    schema_version: Literal["1.0"]
    kind: Literal["quatops_generated_openusd_converter_input"]
    canonical_input_sha256: SHA256
    frame_contract: OpenUsdInputFrameContract
    probe_vectors_xyz: Annotated[list[Vector3], Field(min_length=4, max_length=4)]
    usd_probe_vectors_xyz: Annotated[list[Vector3], Field(min_length=4, max_length=4)]
    nodes: Annotated[list[OpenUsdInputNode], Field(min_length=3, max_length=3)]
    stack_contract: OpenUsdStackContract


class OpenUsdInputFrameContract(NativeCaptureModel):
    canonical: CesiumInputFrame
    usd_stage: CesiumInputFrame
    destination_axes_from_source: Annotated[
        list[str], Field(min_length=3, max_length=3)
    ]
    quaternion: str
    matrix_action: str


class OpenUsdStackContract(NativeCaptureModel):
    parent_ops: list[str]
    child_ops: list[str]
    reset_ops: list[str]


TargetInputs = Union[
    EigenInputs,
    MuJoCoInputs,
    OpenCVInputs,
    SpiceyPyInputs,
    CesiumInputs,
    ThreeGltfInputs,
    OpenUsdInputs,
]


TargetCapture = Union[
    EigenCapture,
    MuJoCoCapture,
    OpenCVCapture,
    SpiceyPyCapture,
    Ros2Capture,
    CesiumCapture,
    ThreeGltfCapture,
    OpenUsdCapture,
]
