"""Declared target representation profiles and a shared conversion front.

Profiles describe one concrete native boundary. They do not infer frame axes,
and their evidence status limits the claim that may be made for that target.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Literal, cast

from ...python import hamilton as _hamilton
from ..semantics.frames import (
    CoordinateFrame,
    FramedOrientation,
    FramedOrientationBatch,
    SemanticQuaternionMap,
)

Representation = Literal["quaternion", "rotation_vector"]
ComponentOrder = Literal["wxyz", "xyzw"]
EvidenceStatus = Literal[
    "accepted_converter",
    "declared_contract",
    "supporting_oracle_only",
]
TargetRole = Literal[
    "orientation_converter",
    "frame_provider",
    "format_validator",
    "loader_observation",
]


@dataclass(frozen=True)
class TargetOrientation:
    """A target value bound to its converter and complete target frames.

    Construct a value received from a native API with the explicit frame
    definitions that give its coefficients meaning.
    """

    converter_id: str
    values: tuple[float, ...]
    representation: Representation
    component_order: ComponentOrder | None
    from_frame: CoordinateFrame
    to_frame: CoordinateFrame

    def __post_init__(self) -> None:
        if self.representation not in ("quaternion", "rotation_vector"):
            raise ValueError("unsupported target orientation representation")
        values = tuple(float(value) for value in self.values)
        expected = 4 if self.representation == "quaternion" else 3
        if len(values) != expected:
            raise ValueError(f"{self.representation} must contain {expected} values")
        if any(not isfinite(value) for value in values):
            raise ValueError("orientation values must be finite")
        if self.representation == "quaternion" and self.component_order not in (
            "wxyz",
            "xyzw",
        ):
            raise ValueError("quaternion target values require a component order")
        if (
            self.representation == "rotation_vector"
            and self.component_order is not None
        ):
            raise ValueError("rotation vectors do not have quaternion component order")
        if not self.converter_id:
            raise ValueError("converter_id must not be empty")
        if not isinstance(self.from_frame, CoordinateFrame):
            raise TypeError("from_frame must be a CoordinateFrame definition")
        if not isinstance(self.to_frame, CoordinateFrame):
            raise TypeError("to_frame must be a CoordinateFrame definition")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class TargetOrientationBatch:
    """A target array with frame definitions and component layout attached."""

    converter_id: str
    values: object
    representation: Representation
    component_order: ComponentOrder | None
    from_frame: CoordinateFrame
    to_frame: CoordinateFrame
    component_axis: int

    def __post_init__(self) -> None:
        if not self.converter_id:
            raise ValueError("target batch must retain a converter ID")
        if not isinstance(self.from_frame, CoordinateFrame):
            raise TypeError("from_frame must be a CoordinateFrame definition")
        if not isinstance(self.to_frame, CoordinateFrame):
            raise TypeError("to_frame must be a CoordinateFrame definition")
        if isinstance(self.component_axis, bool):
            raise TypeError("component_axis must be an integer")
        try:
            component_axis = operator.index(self.component_axis)
        except TypeError as exc:
            raise TypeError("component_axis must be an integer") from exc
        object.__setattr__(self, "component_axis", component_axis)


@dataclass(frozen=True)
class ConverterProfile:
    """One explicit target representation usable through the scalar front."""

    converter_id: str
    target: str
    api: str
    version: str
    representation: Representation
    component_order: ComponentOrder | None
    evidence_status: EvidenceStatus
    scope: str

    def __post_init__(self) -> None:
        if not all(
            (self.converter_id, self.target, self.api, self.version, self.scope)
        ):
            raise ValueError("converter identity, API, version, and scope are required")
        if self.representation == "quaternion":
            if self.component_order not in ("wxyz", "xyzw"):
                raise ValueError("quaternion profiles require WXYZ or XYZW storage")
        elif self.component_order is not None:
            raise ValueError("non-quaternion profiles must omit component_order")
        if self.representation not in ("quaternion", "rotation_vector"):
            raise ValueError("unsupported target orientation representation")
        if self.evidence_status not in (
            "accepted_converter",
            "declared_contract",
            "supporting_oracle_only",
        ):
            raise ValueError("unsupported evidence status")

    def to_target(
        self,
        orientation: FramedOrientation,
        frame_map: SemanticQuaternionMap,
    ) -> TargetOrientation:
        """Convert a labelled canonical orientation into this target layout."""
        mapped = frame_map.to_destination(orientation)
        if self.representation == "rotation_vector":
            values = _hamilton.to_rotation_vector(mapped.quaternion_wxyz)
        elif self.component_order == "xyzw":
            w, x, y, z = mapped.quaternion_wxyz
            values = (x, y, z, w)
        else:
            values = mapped.quaternion_wxyz
        return TargetOrientation(
            converter_id=self.converter_id,
            values=values,
            representation=self.representation,
            component_order=self.component_order,
            from_frame=frame_map.body.destination,
            to_frame=frame_map.world.destination,
        )

    def from_target(
        self,
        orientation: TargetOrientation,
        frame_map: SemanticQuaternionMap,
    ) -> FramedOrientation:
        """Convert a labelled target value back to canonical Hamilton WXYZ."""
        if not isinstance(orientation, TargetOrientation):
            raise TypeError("orientation must be a TargetOrientation")
        if orientation.converter_id != self.converter_id:
            raise ValueError("target orientation belongs to a different converter")
        if (
            orientation.representation != self.representation
            or orientation.component_order != self.component_order
        ):
            raise ValueError("target orientation layout does not match this converter")
        if not orientation.from_frame.has_identical_definition(
            frame_map.body.destination
        ) or not orientation.to_frame.has_identical_definition(
            frame_map.world.destination
        ):
            raise ValueError(
                "target orientation frame definitions do not match this converter"
            )
        frame_map.validate_destination_labels(
            orientation.from_frame.name, orientation.to_frame.name
        )
        if self.representation == "rotation_vector":
            quaternion_wxyz = _hamilton.from_rotation_vector(orientation.values)
        elif self.component_order == "xyzw":
            x, y, z, w = orientation.values
            quaternion_wxyz = (w, x, y, z)
        else:
            quaternion_wxyz = orientation.values
        destination = FramedOrientation(
            cast(tuple[float, float, float, float], tuple(quaternion_wxyz)),
            from_frame=orientation.from_frame.name,
            to_frame=orientation.to_frame.name,
        )
        return frame_map.from_destination(destination)

    def to_target_array(
        self,
        orientations: FramedOrientationBatch,
        frame_map: SemanticQuaternionMap,
        component_axis: int = -1,
    ) -> TargetOrientationBatch:
        """Convert a NumPy batch through the same profile and frame contract."""
        from .numpy import to_target_array

        return to_target_array(self, orientations, frame_map, component_axis)

    def from_target_array(
        self,
        orientations: TargetOrientationBatch,
        frame_map: SemanticQuaternionMap,
        component_axis: int | None = None,
    ) -> FramedOrientationBatch:
        """Convert a target NumPy batch back through this profile."""
        from .numpy import from_target_array

        return from_target_array(self, orientations, frame_map, component_axis)


@dataclass(frozen=True)
class TargetDeclaration:
    """Inventory entry distinguishing converters from related evidence targets."""

    target_id: str
    target: str
    api: str
    version: str
    role: TargetRole
    state: str
    scope: str
    converter_id: str | None = None


def _declaration(profile: ConverterProfile) -> TargetDeclaration:
    state = {
        "accepted_converter": "accepted_converter",
        "declared_contract": "declared_not_native_accepted",
        "supporting_oracle_only": "supporting_oracle_only",
    }[profile.evidence_status]
    return TargetDeclaration(
        target_id=profile.converter_id,
        target=profile.target,
        api=profile.api,
        version=profile.version,
        role="orientation_converter",
        state=state,
        scope=profile.scope,
        converter_id=profile.converter_id,
    )


def _profile(
    converter_id: str,
    target: str,
    api: str,
    version: str,
    order: ComponentOrder | None,
    status: EvidenceStatus,
    scope: str,
    representation: Representation = "quaternion",
) -> ConverterProfile:
    return ConverterProfile(
        converter_id=converter_id,
        target=target,
        api=api,
        version=version,
        representation=representation,
        component_order=order,
        evidence_status=status,
        scope=scope,
    )


_PROFILES = {
    profile.converter_id: profile
    for profile in (
        _profile(
            "unreal.fquat",
            "Unreal",
            "TQuat<T>",
            "5.7.4",
            "xyzw",
            "accepted_converter",
            "Native quaternion action; float32 and float64.",
        ),
        _profile(
            "unity.quaternion",
            "Unity",
            "UnityEngine.Quaternion",
            "6000.5.2f1",
            "xyzw",
            "accepted_converter",
            "Native quaternion action; float32.",
        ),
        _profile(
            "godot.quaternion",
            "Godot",
            "Quaternion",
            "4.7 / 4.8-dev1",
            "xyzw",
            "accepted_converter",
            "Native quaternion action for the captured frame maps; float32.",
        ),
        _profile(
            "blender.mathutils.quaternion",
            "Blender",
            "mathutils.Quaternion",
            "5.1.2",
            "wxyz",
            "accepted_converter",
            "Quaternion field conversion; mesh axes and importer behavior remain separate.",
        ),
        _profile(
            "cesium.quaternion",
            "CesiumJS",
            "Quaternion",
            "1.124.0",
            "xyzw",
            "accepted_converter",
            "Quaternion field conversion; ENU/ECEF placement requires an origin-specific frame map.",
        ),
        _profile(
            "gltf.rotation",
            "glTF",
            "node.rotation",
            "2.0",
            "xyzw",
            "accepted_converter",
            "Quaternion field conversion for the accepted static Three.js loader proof.",
        ),
        _profile(
            "threejs.quaternion",
            "Three.js",
            "Quaternion",
            "0.187.0",
            "xyzw",
            "accepted_converter",
            "Quaternion field conversion; scene hierarchy serialization remains a transport operation.",
        ),
        _profile(
            "openusd.gfquatd",
            "OpenUSD",
            "GfQuatd orient",
            "26.11",
            "wxyz",
            "accepted_converter",
            "Static orient-op quaternion field conversion.",
        ),
        _profile(
            "ros2.geometry-msgs-quaternion",
            "ROS 2",
            "geometry_msgs/Quaternion",
            "Jazzy / 5.3.8",
            "xyzw",
            "accepted_converter",
            "Quaternion message field; frame IDs, timestamps, and translation remain explicit.",
        ),
        _profile(
            "spiceypy.cspice-q",
            "SpiceyPy / CSPICE",
            "q2m / m2q quaternion",
            "8.3.0 / N0067",
            "wxyz",
            "accepted_converter",
            "Direct numeric quaternion boundary in the captured probe frame.",
        ),
        _profile(
            "mujoco.quaternion",
            "MuJoCo",
            "mju quaternion APIs",
            "3.17.0",
            "wxyz",
            "accepted_converter",
            "Direct quaternion API; model and body frames remain caller supplied.",
        ),
        _profile(
            "opencv.rodrigues",
            "OpenCV",
            "Rodrigues rotation vector",
            "4.13.0",
            None,
            "accepted_converter",
            "Rotation-vector boundary in radians; camera extrinsics are outside scope.",
            "rotation_vector",
        ),
        _profile(
            "eigen.quaternion-constructor",
            "Eigen",
            "Quaternion(w,x,y,z)",
            "5.0.1",
            "wxyz",
            "accepted_converter",
            "Constructor coefficient boundary; application frames remain caller supplied.",
        ),
        _profile(
            "eigen.quaternion-coeffs",
            "Eigen",
            "Quaternion::coeffs()",
            "5.0.1",
            "xyzw",
            "accepted_converter",
            "Coefficient accessor boundary; application frames remain caller supplied.",
        ),
        _profile(
            "openxr.xrquaternionf",
            "OpenXR",
            "XrQuaternionf in XrPosef",
            "1.1.65",
            "xyzw",
            "declared_contract",
            "Spec and Khronos test-runtime contract only; no vendor-runtime or hardware acceptance.",
        ),
        _profile(
            "sophus.so3-params",
            "Sophus",
            "SO3::params()",
            "1.24.6",
            "xyzw",
            "supporting_oracle_only",
            "Captured SO(3) representation; not an accepted application-frame converter.",
        ),
    )
}

CONVERTERS: Mapping[str, ConverterProfile] = MappingProxyType(_PROFILES)
"""Immutable converter declarations keyed by concrete native boundary."""

_ACCEPTED_CONVERTERS: Mapping[str, ConverterProfile] = MappingProxyType(
    {
        converter_id: profile
        for converter_id, profile in _PROFILES.items()
        if profile.evidence_status == "accepted_converter"
    }
)


def accepted_converters() -> Mapping[str, ConverterProfile]:
    """Return only profiles whose captured evidence accepts converter use.

    Use :func:`get_converter` to inspect declared or supporting profiles when
    their narrower evidence state is appropriate for the application.
    """
    return _ACCEPTED_CONVERTERS


_TARGET_DECLARATIONS = {
    declaration.target_id: declaration
    for declaration in (_declaration(profile) for profile in _PROFILES.values())
}
_TARGET_DECLARATIONS.update(
    {
        "proj.9.5.1": TargetDeclaration(
            target_id="proj.9.5.1",
            target="PROJ",
            api="WGS84 coordinate transforms",
            version="9.5.1",
            role="frame_provider",
            state="supporting_frame_evidence_only",
            scope="Coordinate-frame evidence; PROJ exposes no quaternion boundary here.",
        ),
        "khronos.gltf-validator": TargetDeclaration(
            target_id="khronos.gltf-validator",
            target="Khronos glTF Validator",
            api="glTF structural validation",
            version="2.0.0-dev.3.10",
            role="format_validator",
            state="format_validation_only",
            scope="Checks asset validity; does not provide a rotation-action oracle.",
        ),
        "blender.gltf-import": TargetDeclaration(
            target_id="blender.gltf-import",
            target="Blender glTF importer",
            api="glTF scene and animation import",
            version="5.1.2",
            role="loader_observation",
            state="loader_observation_only",
            scope="Importer behavior evidence; distinct from mathutils.Quaternion conversion.",
        ),
        "vtk.gltf-loader": TargetDeclaration(
            target_id="vtk.gltf-loader",
            target="VTK glTF loader",
            api="glTF scene loading",
            version="9.6.2",
            role="loader_observation",
            state="loader_observation_only",
            scope="Loader evidence only; no QuatOps-generated converter acceptance claim.",
        ),
        "trimesh.gltf-loader": TargetDeclaration(
            target_id="trimesh.gltf-loader",
            target="Trimesh glTF loader",
            api="glTF scene loading",
            version="4.11.1",
            role="loader_observation",
            state="loader_observation_only",
            scope="Loader evidence only; no QuatOps-generated converter acceptance claim.",
        ),
    }
)
TARGET_DECLARATIONS: Mapping[str, TargetDeclaration] = MappingProxyType(
    _TARGET_DECLARATIONS
)
"""Converter and evidence-target index, including explicitly non-converter roles."""


def get_converter(converter_id: str) -> ConverterProfile:
    """Return a declared profile by ID, or raise ``KeyError`` if unknown."""
    return CONVERTERS[converter_id]


def get_target_declaration(target_id: str) -> TargetDeclaration:
    """Return any converter/evidence target declaration by ID."""
    return TARGET_DECLARATIONS[target_id]
