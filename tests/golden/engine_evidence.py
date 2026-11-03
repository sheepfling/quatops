"""Typed semantic models for upstream engine-convention evidence."""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

Vector3 = tuple[float, float, float]
QuaternionXYZW = tuple[float, float, float, float]
RotationMatrix3 = tuple[Vector3, Vector3, Vector3]
EulerOrder = Literal["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]


class EngineEulerCase(BaseModel):
    """One published Euler-convention case and its expected rotations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    engine: str
    evidence_kind: Literal["upstream_static_fixture"]
    input_degrees_xyz: Vector3
    api_input_unit: Literal["radians"]
    engine_order: EulerOrder
    intrinsic: bool
    expected_quaternion_xyzw: QuaternionXYZW
    expected_matrix_rows: RotationMatrix3
    quaternion_atol: float
    matrix_atol: float
    upstream_quaternion_tolerance: Optional[float]
    source_urls: tuple[str, ...]


class EngineEulerEvidence(BaseModel):
    """Fixture metadata that distinguishes published data from runtime captures."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    research_date: date
    evidence_kind: Literal["upstream_static_fixtures_not_local_native_captures"]
    quaternion_order: Literal["xyzw"]
    matrix_layout: str
    input_note: str
    precision_note: str
    comparison_note: str
    native_engines_executed: tuple[str, ...]
    unreal_expected_values: None
    cases: tuple[EngineEulerCase, ...]
