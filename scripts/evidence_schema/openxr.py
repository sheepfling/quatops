"""Strict finite-number records for the captured OpenXR test-runtime chain."""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import Field, FiniteFloat, model_validator
from typing_extensions import TypeAlias

from scripts.evidence_schema import primitives
from scripts.evidence_schema.primitives import (
    NativeCaptureModel,
)

# Keep the earlier OpenXR schema import surface while the shared helper lives
# with the other scalar primitives.
finite_max_abs = primitives.finite_max_abs

ObservationKey: TypeAlias = Literal["A_from_B", "B_from_C", "A_from_C"]
InputKey: TypeAlias = Literal["A", "B", "C"]
PoseVector = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
PoseQuaternion = Annotated[list[FiniteFloat], Field(min_length=4, max_length=4)]


class OpenXRPose(NativeCaptureModel):
    orientation_xyzw: PoseQuaternion
    position_xyz_m: PoseVector


class LocatedOpenXRPose(NativeCaptureModel):
    target: Literal["A", "B", "C"]
    base: Literal["A", "B", "C"]
    location_flags: Literal[15]
    pose: OpenXRPose


class OpenXRTestRuntimeIdentity(NativeCaptureModel):
    product: Literal["Khronos OpenXR-SDK-Source test_runtime"]
    headless_extension: Literal["XR_MND_headless"]
    headless_extension_advertised: Literal[True]
    hardware_runtime: Literal[False]


class OpenXRTestRuntimeCapture(NativeCaptureModel):
    # Native stdout predates the retained fixture envelope; if supplied, pin it.
    # Pydantic 2 on Python 3.9 cannot resolve PEP 604 forward references.
    schema_version: Optional[Literal["1.0"]] = None  # noqa: UP045
    evidence_kind: Literal["native_openxr_test_runtime_frame_chain"]
    runtime: OpenXRTestRuntimeIdentity
    api_version: Annotated[list[int], Field(min_length=3, max_length=3)]
    locate_time_xr_time: int
    reference_space_type: Literal["LOCAL", "VIEW", "LOCAL_FLOOR"]
    inputs_pose_in_reference_space: dict[InputKey, OpenXRPose]
    observations: dict[ObservationKey, LocatedOpenXRPose]
    quatops_executed: Literal[False]


class OpenXRVerifierSummary(NativeCaptureModel):
    evidence_kind: Literal["derived_reference"]
    reference: Literal["SciPy Rotation"]
    scipy_version: str
    numpy_version: str
    pydantic_version: str
    quatops_executed: Literal[False]
    tolerance: FiniteFloat
    native_formula_max_errors: dict[ObservationKey, FiniteFloat]
    chain_rotation_matrix_max_abs_error: FiniteFloat
    chain_translation_max_abs_error_m: FiniteFloat
    chain_point_max_abs_error_m: FiniteFloat
    result: Literal["pass"]


class OpenXRRelationCase(NativeCaptureModel):
    """One declared target-in-base relation exercised by the fixture."""

    observation: ObservationKey
    target: InputKey
    base: InputKey


class OpenXRReferenceSpaceContract(NativeCaptureModel):
    reference_space_type: Literal["LOCAL"]
    handedness: Literal["right-handed"]
    positive_x: Literal["right"]
    positive_y: Literal["up"]
    positive_z: Literal["back"]
    origin: Literal["runtime-defined"]


class OpenXRSemanticConfig(NativeCaptureModel):
    """Experiment-owned OpenXR facts, separate from capture provenance."""

    schema_version: Literal["1.0"]
    evidence_status: Literal["declared_not_native_accepted"]
    product: Literal["Khronos OpenXR SDK and SDK-Source test_runtime"]
    version: Literal["1.1.65"]
    api_version: Annotated[list[int], Field(min_length=3, max_length=3)]
    quaternion_order: Literal["xyzw"]
    quaternion_precision: Literal["binary32"]
    position_units: Literal["metres"]
    validation_tolerance: FiniteFloat
    orientation_applied_before_translation: Literal[True]
    locate_returns_target_pose_in_base: Literal[True]
    reference_space: OpenXRReferenceSpaceContract
    input_spaces: Annotated[list[InputKey], Field(min_length=3, max_length=3)]
    relation_cases: Annotated[
        list[OpenXRRelationCase], Field(min_length=3, max_length=3)
    ]

    @model_validator(mode="after")
    def relation_cases_are_consistent(self) -> OpenXRSemanticConfig:
        if len(set(self.input_spaces)) != len(self.input_spaces):
            raise ValueError("OpenXR semantic config repeats an input space")
        if len({case.observation for case in self.relation_cases}) != len(
            self.relation_cases
        ):
            raise ValueError("OpenXR semantic config repeats a relation case")
        input_spaces = set(self.input_spaces)
        for case in self.relation_cases:
            if case.base not in input_spaces or case.target not in input_spaces:
                raise ValueError("OpenXR relation refers to an undeclared input space")
            if case.observation != f"{case.base}_from_{case.target}":
                raise ValueError("OpenXR relation key disagrees with target/base order")
        return self
