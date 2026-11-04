"""Strict finite-number records for the captured OpenXR test-runtime chain."""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import Field, FiniteFloat
from typing_extensions import TypeAlias

from scripts.evidence_schema import primitives
from scripts.evidence_schema.primitives import NativeCaptureModel

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
    reference_space_type: Literal["LOCAL"]
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
