"""Executable bindings from documented roles to explicit frame definitions."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, radians, sin
from numbers import Real
from operator import index
from typing import Literal

from ..engines.basis import BasisMap, OrthogonalBasisMap, SignedAxisMap
from .conventions import ECEF_FRAME
from .frames import CoordinateFrame
from .roles import SEMANTIC_CONTRACTS, AlignmentPurpose, SemanticAuthority

AlignmentProvider = Literal["application", "runtime"]
AlignmentSource = Literal[
    "application_declaration", "calibration", "survey", "runtime_relation"
]
OpenXRReferenceSpaceType = Literal["view", "local", "local_floor", "stage"]
_OPENXR_BINDABLE_REFERENCE_SPACES = {
    "openxr.view-space": frozenset({"view"}),
    "openxr.reference-space": frozenset({"local", "local_floor"}),
}
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_XR_SPACE_LOCATION_ORIENTATION_VALID_BIT = 0x00000001
_XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT = 0x00000004


@dataclass(frozen=True)
class FrameAlignment:
    """Typed alignment of a role's local axes into a named reference frame."""

    reference_frame: CoordinateFrame
    basis_to_reference: BasisMap
    purpose: AlignmentPurpose
    provider: AlignmentProvider
    alignment_source: AlignmentSource
    alignment_evidence_reference: str
    space_context: OpenXRSpaceContext | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reference_frame, CoordinateFrame):
            raise TypeError("reference_frame must be a CoordinateFrame")
        if not isinstance(self.basis_to_reference, (OrthogonalBasisMap, SignedAxisMap)):
            raise TypeError("basis_to_reference must be an orthogonal BasisMap")
        if self.purpose not in (
            "scene_alignment",
            "body_alignment",
            "sensor_mount",
            "model_alignment",
            "space_alignment",
        ):
            raise ValueError("unsupported frame alignment purpose")
        if self.reference_frame.handedness != "right":
            raise ValueError("alignment reference frame must be right-handed")
        if self.reference_frame.reference_convention is None:
            raise ValueError(
                "alignment reference frame must declare reference_convention"
            )
        identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        if self.reference_frame.basis_to_reference.matrix != identity:
            raise ValueError(
                "alignment reference frame must use identity basis in its reference"
            )
        if self.alignment_source not in (
            "application_declaration",
            "calibration",
            "survey",
            "runtime_relation",
        ):
            raise ValueError("unsupported physical alignment source")
        if (
            not isinstance(self.alignment_evidence_reference, str)
            or not self.alignment_evidence_reference.strip()
        ):
            raise ValueError("alignment evidence reference must be non-empty")
        if self.provider not in ("application", "runtime"):
            raise ValueError("provider must be 'application' or 'runtime'")
        if self.provider == "runtime" and self.alignment_source != "runtime_relation":
            raise ValueError(
                "runtime-provided alignment requires runtime_relation source"
            )
        if (
            self.provider == "application"
            and self.alignment_source == "runtime_relation"
        ):
            raise ValueError(
                "application-provided alignment cannot claim a runtime relation"
            )
        if self.space_context is not None and not isinstance(
            self.space_context, OpenXRSpaceContext
        ):
            raise TypeError("space_context must be an OpenXRSpaceContext")


@dataclass(frozen=True)
class OpenXRSpaceContext:
    """Session-scoped identity for one time-specific OpenXR space relation.

    Space IDs are opaque application labels for the actual XrSpace handles;
    they are not substitutes for reference-space type names such as LOCAL.
    """

    space_id: str
    base_space_id: str
    session_id: str
    time_ns: int
    runtime_id: str
    reference_space_type: OpenXRReferenceSpaceType
    location_flags: int

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                self.space_id,
                self.base_space_id,
                self.session_id,
                self.runtime_id,
            )
        ):
            raise ValueError(
                "OpenXR space IDs, session ID, and runtime ID must be non-empty"
            )
        if self.reference_space_type not in (
            "view",
            "local",
            "local_floor",
            "stage",
        ):
            raise ValueError("unsupported OpenXR reference-space type")
        if isinstance(self.time_ns, bool):
            raise TypeError("OpenXR time_ns must be a signed 64-bit integer")
        try:
            time_ns = index(self.time_ns)
        except TypeError as exc:
            raise TypeError("OpenXR time_ns must be a signed 64-bit integer") from exc
        if not _INT64_MIN <= time_ns <= _INT64_MAX:
            raise ValueError("OpenXR time_ns must fit the XrTime signed 64-bit range")
        object.__setattr__(self, "time_ns", time_ns)
        if isinstance(self.location_flags, bool) or not isinstance(
            self.location_flags, int
        ):
            raise TypeError("OpenXR location_flags must be an integer bitmask")
        if not 0 <= self.location_flags < (1 << 64):
            raise ValueError("OpenXR location_flags must fit the unsigned 64-bit range")
        if not self.orientation_valid:
            raise ValueError(
                "OpenXR orientation binding requires XR_SPACE_LOCATION_ORIENTATION_VALID_BIT"
            )

    @property
    def orientation_valid(self) -> bool:
        """Whether the runtime marked the orientation in this location valid."""
        return bool(self.location_flags & _XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)

    @property
    def orientation_tracked(self) -> bool:
        """Whether the runtime marked the orientation as actively tracked."""
        return bool(self.location_flags & _XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT)


@dataclass(frozen=True)
class GeodeticOrigin:
    """Geodetic origin used for a local ENU frame, in degrees and meters."""

    longitude_degrees: float
    latitude_degrees: float
    height_meters: float

    def __post_init__(self) -> None:
        values = (self.longitude_degrees, self.latitude_degrees, self.height_meters)
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
            for value in values
        ):
            raise ValueError("geodetic origin values must be finite")
        if not -180.0 <= self.longitude_degrees <= 180.0:
            raise ValueError("longitude must be within [-180, 180] degrees")
        if not -90.0 <= self.latitude_degrees <= 90.0:
            raise ValueError("latitude must be within [-90, 90] degrees")


@dataclass(frozen=True)
class Ellipsoid:
    """Ellipsoid dimensions required by a geographic frame placement."""

    name: str
    semi_major_axis_meters: float
    semi_minor_axis_meters: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ellipsoid name must be non-empty")
        axes = (self.semi_major_axis_meters, self.semi_minor_axis_meters)
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
            for value in axes
        ):
            raise ValueError("ellipsoid dimensions must be finite")
        if not 0.0 < self.semi_minor_axis_meters <= self.semi_major_axis_meters:
            raise ValueError("ellipsoid axes must satisfy 0 < b <= a")


@dataclass(frozen=True)
class SemanticFrameBinding:
    """Resolved frame plus its authority, source, and supplied context."""

    contract_id: str
    authority: SemanticAuthority
    coordinate_frame: CoordinateFrame
    evidence_reference: str
    target_family: str | None = None
    alignment: FrameAlignment | None = None
    reference_frame: CoordinateFrame | None = None
    origin: GeodeticOrigin | None = None
    ellipsoid: Ellipsoid | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_frame, CoordinateFrame):
            raise TypeError("coordinate_frame must be a CoordinateFrame")
        if (
            not isinstance(self.evidence_reference, str)
            or not self.evidence_reference.strip()
        ):
            raise ValueError("evidence_reference must be non-empty")
        contract = _contract(self.contract_id)
        if self.authority != contract.authority:
            raise ValueError("binding authority does not match its semantic contract")
        if self.target_family != contract.target_family:
            raise ValueError(
                "binding target family does not match its semantic contract"
            )
        if self.evidence_reference != (contract.source_url or contract.contract_id):
            raise ValueError(
                "binding evidence reference does not match its semantic contract"
            )
        frame = self.coordinate_frame
        if contract.binding_kind == "fixed_axes":
            if not isinstance(self.alignment, FrameAlignment):
                raise TypeError("fixed-axis bindings require a FrameAlignment")
            if self.origin is not None or self.ellipsoid is not None:
                raise ValueError(
                    "fixed-axis bindings require alignment and no geographic context"
                )
            if self.reference_frame is not None:
                if not isinstance(self.reference_frame, CoordinateFrame):
                    raise TypeError("reference_frame must be a CoordinateFrame")
                if not self.reference_frame.has_identical_definition(
                    self.alignment.reference_frame
                ):
                    raise ValueError(
                        "binding reference frame conflicts with its alignment"
                    )
            object.__setattr__(self, "reference_frame", self.alignment.reference_frame)
            if (
                frame.axes != contract.positive_axes
                or frame.handedness != contract.handedness
                or frame.role != contract.frame_role
            ):
                raise ValueError(
                    "bound frame axes, handedness, or role conflict with contract"
                )
            if frame.reference != self.alignment.reference_frame.reference:
                raise ValueError("bound frame reference conflicts with alignment")
            if (
                frame.reference_convention
                != self.alignment.reference_frame.reference_convention
            ):
                raise ValueError("bound frame convention conflicts with alignment")
            if (
                frame.basis_to_reference.matrix
                != self.alignment.basis_to_reference.matrix
            ):
                raise ValueError("bound frame basis conflicts with alignment")
            if self.alignment.purpose != contract.alignment_purpose:
                raise ValueError("binding alignment purpose conflicts with contract")
            expected_provider = (
                "runtime"
                if contract.alignment_purpose == "space_alignment"
                else "application"
            )
            if self.alignment.provider != expected_provider:
                raise ValueError("binding alignment provider conflicts with contract")
            if contract.contract_id in _OPENXR_BINDABLE_REFERENCE_SPACES:
                context = self.alignment.space_context
                if context is None:
                    raise ValueError(
                        "OpenXR reference-space bindings require space context"
                    )
                if (
                    context.reference_space_type
                    not in _OPENXR_BINDABLE_REFERENCE_SPACES[contract.contract_id]
                ):
                    raise ValueError(
                        "OpenXR reference-space type does not match the selected role"
                    )
                if context.base_space_id != self.alignment.reference_frame.name:
                    raise ValueError(
                        "OpenXR base space must match the alignment reference frame"
                    )
            elif self.alignment.space_context is not None:
                raise ValueError(
                    "OpenXR space context is only valid for OpenXR bindings"
                )
        elif contract.binding_kind == "geographic_enu":
            if not isinstance(self.origin, GeodeticOrigin) or not isinstance(
                self.ellipsoid, Ellipsoid
            ):
                raise TypeError(
                    "geographic bindings require typed origin and ellipsoid"
                )
            if not isinstance(self.reference_frame, CoordinateFrame):
                raise TypeError(
                    "geographic bindings require their full reference frame"
                )
            if self.alignment is not None:
                raise ValueError(
                    "geographic bindings require origin and ellipsoid context"
                )
            if (
                frame.axes != ("east", "north", "up")
                or frame.handedness != "right"
                or frame.role != "world"
                or frame.reference_convention != ECEF_FRAME.reference_convention
            ):
                raise ValueError(
                    "geographic ENU frame must declare right-handed world ENU axes"
                )
            if (
                self.reference_frame.handedness != "right"
                or self.reference_frame.axes != ECEF_FRAME.axes
                or self.reference_frame.reference_convention
                != ECEF_FRAME.reference_convention
                or self.reference_frame.basis_to_reference.matrix
                != ((1, 0, 0), (0, 1, 0), (0, 0, 1))
                or frame.reference != self.reference_frame.reference
                or frame.reference_convention
                != self.reference_frame.reference_convention
            ):
                raise ValueError(
                    "geographic binding reference must be the declared ECEF frame"
                )
            if (
                frame.basis_to_reference.matrix
                != _enu_to_ecef_basis(self.origin, self.ellipsoid).matrix
            ):
                raise ValueError("geographic ENU basis does not match its typed origin")
        else:
            raise ValueError(
                f"semantic binding kind {contract.binding_kind!r} has no executable binder"
            )


def _contract(contract_id: str):
    try:
        return SEMANTIC_CONTRACTS[contract_id]
    except KeyError as exc:
        raise KeyError(f"unknown semantic role {contract_id!r}") from exc


def _enu_to_ecef_basis(
    origin: GeodeticOrigin, ellipsoid: Ellipsoid
) -> OrthogonalBasisMap:
    longitude = radians(origin.longitude_degrees)
    latitude = radians(origin.latitude_degrees)
    sin_lon, cos_lon = sin(longitude), cos(longitude)
    sin_lat, cos_lat = sin(latitude), cos(latitude)
    semi_major = ellipsoid.semi_major_axis_meters
    semi_minor = ellipsoid.semi_minor_axis_meters
    eccentricity_squared = 1.0 - (semi_minor / semi_major) ** 2
    prime_vertical = semi_major / (1.0 - eccentricity_squared * sin_lat**2) ** 0.5
    origin_x = (prime_vertical + origin.height_meters) * cos_lat * cos_lon
    origin_y = (prime_vertical + origin.height_meters) * cos_lat * sin_lon
    origin_z = (
        prime_vertical * (1.0 - eccentricity_squared) + origin.height_meters
    ) * sin_lat
    normal = (
        origin_x / semi_major**2,
        origin_y / semi_major**2,
        origin_z / semi_minor**2,
    )
    normal_length = sum(component**2 for component in normal) ** 0.5
    if not isfinite(normal_length) or normal_length <= 1e-15:
        raise ValueError("Cesium local ENU frame is undefined at the ellipsoid center")
    up = tuple(component / normal_length for component in normal)
    east = (-sin_lon, cos_lon, 0.0)
    north = (
        up[1] * east[2] - up[2] * east[1],
        up[2] * east[0] - up[0] * east[2],
        up[0] * east[1] - up[1] * east[0],
    )
    return OrthogonalBasisMap(
        (
            (east[0], north[0], up[0]),
            (east[1], north[1], up[1]),
            (east[2], north[2], up[2]),
        )
    )


def bind_semantic_role(
    contract_id: str,
    *,
    name: str,
    alignment: FrameAlignment,
) -> SemanticFrameBinding:
    """Bind a fixed-axis target/standard role using an explicit alignment.

    The authoritative source supplies the frame's axis meanings and
    handedness. The caller/runtime supplies its orientation in the named
    reference. No target-world or model alignment is inferred.
    """
    if not isinstance(alignment, FrameAlignment):
        raise TypeError("alignment must be a FrameAlignment")
    contract = _contract(contract_id)
    if contract.binding_kind != "fixed_axes":
        raise ValueError(
            f"semantic role {contract_id!r} needs its role-specific typed context"
        )
    if alignment.purpose != contract.alignment_purpose:
        raise ValueError(
            f"role {contract_id!r} requires alignment purpose "
            f"{contract.alignment_purpose!r}"
        )
    expected_provider: AlignmentProvider = (
        "runtime" if alignment.purpose == "space_alignment" else "application"
    )
    if alignment.provider != expected_provider:
        raise ValueError(
            f"role {contract_id!r} requires {expected_provider}-supplied alignment"
        )
    if alignment.purpose == "space_alignment":
        if alignment.space_context is None:
            raise ValueError(
                "OpenXR reference-space binding requires typed space context"
            )
        if contract_id not in _OPENXR_BINDABLE_REFERENCE_SPACES:
            raise ValueError(
                "only documented OpenXR reference-space roles can be bound"
            )
        if (
            alignment.space_context.reference_space_type
            not in _OPENXR_BINDABLE_REFERENCE_SPACES[contract_id]
        ):
            raise ValueError(
                "OpenXR reference-space type does not match the selected role"
            )
        if alignment.space_context.base_space_id != alignment.reference_frame.name:
            raise ValueError(
                "OpenXR base space must match the alignment reference frame"
            )
    elif alignment.space_context is not None:
        raise ValueError("OpenXR space context is only valid for a space alignment")
    assert contract.positive_axes is not None and contract.handedness is not None
    frame = CoordinateFrame(
        name=name,
        reference=alignment.reference_frame.reference,
        axes=contract.positive_axes,
        basis_to_reference=alignment.basis_to_reference,
        handedness=contract.handedness,
        role=contract.frame_role,
        reference_convention=alignment.reference_frame.reference_convention,
    )
    return SemanticFrameBinding(
        contract_id=contract.contract_id,
        authority=contract.authority,
        coordinate_frame=frame,
        evidence_reference=contract.source_url or contract.contract_id,
        target_family=contract.target_family,
        alignment=alignment,
    )


def bind_cesium_enu_frame(
    *,
    name: str,
    origin: GeodeticOrigin,
    ellipsoid: Ellipsoid,
    ecef_reference: CoordinateFrame = ECEF_FRAME,
) -> SemanticFrameBinding:
    """Resolve Cesium's local ENU axes into ECEF at a typed geodetic origin."""
    contract = _contract("cesium.enu-to-fixed")
    if not isinstance(origin, GeodeticOrigin):
        raise TypeError("origin must be a GeodeticOrigin")
    if not isinstance(ellipsoid, Ellipsoid):
        raise TypeError("ellipsoid must be an Ellipsoid")
    if not isinstance(ecef_reference, CoordinateFrame):
        raise TypeError("ecef_reference must be a CoordinateFrame")
    if (
        ecef_reference.handedness != "right"
        or ecef_reference.reference_convention is None
    ):
        raise ValueError("ECEF reference must be right-handed and convention-labelled")
    if (
        ecef_reference.reference_convention != ECEF_FRAME.reference_convention
        or ecef_reference.axes != ECEF_FRAME.axes
    ):
        raise ValueError("ECEF reference must use the QuatOps ECEF convention")
    if ecef_reference.basis_to_reference.matrix != ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        raise ValueError("ECEF reference must use identity basis")

    enu_to_ecef = _enu_to_ecef_basis(origin, ellipsoid)
    frame = CoordinateFrame(
        name=name,
        reference=ecef_reference.reference,
        axes=("east", "north", "up"),
        basis_to_reference=enu_to_ecef,
        handedness="right",
        role="world",
        reference_convention=ecef_reference.reference_convention,
    )
    return SemanticFrameBinding(
        contract_id=contract.contract_id,
        authority=contract.authority,
        coordinate_frame=frame,
        evidence_reference=contract.source_url or contract.contract_id,
        target_family=contract.target_family,
        origin=origin,
        ellipsoid=ellipsoid,
        reference_frame=ecef_reference,
    )
