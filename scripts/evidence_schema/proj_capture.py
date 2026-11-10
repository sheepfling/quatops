"""Strict schema for PROJ's retained geographic-frame observations."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from scripts.evidence_schema.primitives import BasisImages, NativeCaptureModel, Vector3


class ProjCaptureContract(NativeCaptureModel):
    """Exact units and transform direction used by the PROJ experiment."""

    units: Literal["geographic degrees; Cartesian metres"]
    quaternion_order: None
    matrix_action: Literal["column vectors"]
    frame_direction: Literal["forward ECEF -> ENU; inverse ENU -> ECEF"]


class ProjNativeInputs(NativeCaptureModel):
    """Inputs are resolved by case ID in the referenced geographic corpus."""


class ProjNumericVector3(NativeCaptureModel):
    """One shape-checked binary64 vector stored in the raw capture format."""

    values: Vector3
    dtype: Literal["binary64"]
    shape: Annotated[list[Literal[3]], Field(min_length=1, max_length=1)]


class ProjNumericBasisImages(NativeCaptureModel):
    """Three shape-checked binary64 endpoint vectors."""

    values: BasisImages
    dtype: Literal["binary64"]
    shape: Annotated[list[Literal[3]], Field(min_length=2, max_length=2)]


class ProjNativeOutputs(NativeCaptureModel):
    """PROJ ECEF origin, ENU endpoints, and inverse point readback."""

    origin_ecef_m: ProjNumericVector3
    enu_basis_endpoints_ecef_m: ProjNumericBasisImages
    probe_endpoint_ecef_m: ProjNumericVector3
    probe_roundtrip_local_xyz_m: ProjNumericVector3


class ProjObservation(NativeCaptureModel):
    """Fully typed raw observation emitted by the pinned PROJ capture."""

    case_id: str
    operation_id: Literal["geographic-enu-ecef"]
    native_entry_point: Literal[
        "pyproj.Transformer.from_crs(4979,4978); from_pipeline(topocentric); transform"
    ]
    contract: ProjCaptureContract
    native_inputs: ProjNativeInputs
    native_outputs: ProjNativeOutputs
    status: Literal["ok"]
    diagnostic: None
    schema_version: Literal["1.0"]
