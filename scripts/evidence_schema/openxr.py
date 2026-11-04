"""Strict finite-number records for the captured OpenXR test-runtime chain."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, FiniteFloat

from scripts.evidence_schema.primitives import NativeCaptureModel

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
    schema_version: Literal["1.0"] | None = None
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


def finite_max_abs(values: Iterable[float], label: str) -> float:
    """Return a finite maximum absolute error; reject empty or non-finite inputs."""
    finite_values = [float(value) for value in values]
    if not finite_values:
        raise ValueError(f"{label} is empty")
    for value in finite_values:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite value: {value}")
    maximum = max(abs(value) for value in finite_values)
    return maximum
