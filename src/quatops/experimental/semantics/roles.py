"""Authoritative frame-role declarations, separate from quaternion profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from .frames import FrameRole, Handedness

SemanticAuthority = Literal[
    "target_defined",
    "standard_defined",
    "application_declared",
    "representation_only",
    "fixture_only",
    "unknown_or_conflicting",
]
SemanticSourceKind = Literal[
    "target_documentation", "formal_standard", "application_contract"
]
SemanticBindingKind = Literal[
    "fixed_axes", "geographic_enu", "relational", "application_frames"
]
AlignmentPurpose = Literal[
    "scene_alignment",
    "body_alignment",
    "sensor_mount",
    "model_alignment",
    "space_alignment",
]


@dataclass(frozen=True)
class SemanticContract:
    """Versioned authority for one physical frame role or relation."""

    contract_id: str
    authority: SemanticAuthority
    role: str
    source_kind: SemanticSourceKind
    binding_kind: SemanticBindingKind
    source_url: str | None = None
    source_version: str | None = None
    source_section: str | None = None
    target_family: str | None = None
    positive_axes: tuple[str, str, str] | None = None
    handedness: Handedness | None = None
    frame_role: FrameRole = "generic"
    alignment_purpose: AlignmentPurpose | None = None
    application_source: str | None = None
    source_claim: str | None = None
    assurance_scope: str | None = None
    exclusions: tuple[str, ...] = ()
    converter_ids: tuple[str, ...] = ()
    execution_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.contract_id, str)
            or not self.contract_id.strip()
            or not isinstance(self.role, str)
            or not self.role.strip()
        ):
            raise ValueError("semantic contracts require an ID and a role")
        if self.authority not in (
            "target_defined",
            "standard_defined",
            "application_declared",
            "representation_only",
            "fixture_only",
            "unknown_or_conflicting",
        ):
            raise ValueError("unsupported semantic authority")
        if self.binding_kind not in (
            "fixed_axes",
            "geographic_enu",
            "relational",
            "application_frames",
        ):
            raise ValueError("unsupported semantic binding kind")
        if self.source_kind not in (
            "target_documentation",
            "formal_standard",
            "application_contract",
        ):
            raise ValueError("unsupported semantic source kind")
        if self.frame_role not in ("world", "body", "sensor", "generic"):
            raise ValueError("unsupported semantic frame role")
        if (
            self.authority == "target_defined"
            and self.source_kind != "target_documentation"
        ):
            raise ValueError("target-defined semantics require target documentation")
        if (
            self.authority == "standard_defined"
            and self.source_kind != "formal_standard"
        ):
            raise ValueError("standard-defined semantics require a formal standard")
        if (
            self.authority == "application_declared"
            and self.source_kind != "application_contract"
        ):
            raise ValueError(
                "application-declared semantics require an application contract"
            )
        if self.authority == "application_declared" and not self.application_source:
            raise ValueError("application-declared contracts require an API source")
        if self.authority in ("target_defined", "standard_defined") and not all(
            (self.source_url, self.source_version, self.source_section)
        ):
            raise ValueError(
                "authoritative semantic contracts require a pinned source URL, "
                "version, and section"
            )
        if self.authority in ("target_defined", "standard_defined") and not (
            self.source_url and self.source_url.startswith("https://")
        ):
            raise ValueError("authoritative source URL must use HTTPS")
        if self.authority in ("target_defined", "standard_defined") and not all(
            (self.source_claim, self.assurance_scope)
        ):
            raise ValueError(
                "authoritative contracts require a reviewed source claim and scope"
            )
        if not isinstance(self.exclusions, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.exclusions
        ):
            raise ValueError("semantic contract exclusions must be non-empty strings")
        if not isinstance(self.converter_ids, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.converter_ids
        ):
            raise ValueError(
                "semantic contract converter IDs must be non-empty strings"
            )
        if not isinstance(self.execution_evidence_ids, tuple) or any(
            not isinstance(item, str) or not item.strip()
            for item in self.execution_evidence_ids
        ):
            raise ValueError("semantic contract evidence IDs must be non-empty strings")
        if self.binding_kind == "fixed_axes":
            if self.positive_axes is None or self.handedness is None:
                raise ValueError("fixed-axis contracts require axes and handedness")
            if len(self.positive_axes) != 3 or any(
                not isinstance(axis, str) or not axis.strip()
                for axis in self.positive_axes
            ):
                raise ValueError(
                    "fixed-axis contracts require three named positive axes"
                )
            if len({axis.strip().casefold() for axis in self.positive_axes}) != 3:
                raise ValueError("fixed-axis meanings must be distinct")
            if self.handedness not in ("right", "left"):
                raise ValueError("fixed-axis contract handedness must be right or left")
            derived_handedness = _axis_triad_handedness(self.positive_axes)
            if derived_handedness != self.handedness:
                raise ValueError(
                    "fixed-axis handedness conflicts with its named axis directions"
                )
            if self.alignment_purpose is None:
                raise ValueError("fixed-axis contracts require an alignment purpose")
            if self.alignment_purpose not in (
                "scene_alignment",
                "body_alignment",
                "sensor_mount",
                "model_alignment",
                "space_alignment",
            ):
                raise ValueError("unsupported fixed-axis alignment purpose")
        elif (
            self.positive_axes is not None
            or self.handedness is not None
            or self.alignment_purpose is not None
        ):
            raise ValueError(
                "non-fixed-axis contracts cannot declare fixed-axis fields"
            )


def _axis_triad_handedness(axes: tuple[str, str, str]) -> Handedness:
    """Derive parity for registered cardinal directions in canonical RFU axes."""
    direction = {
        "right": (1, 0, 0),
        "left": (-1, 0, 0),
        "forward": (0, 1, 0),
        "back": (0, -1, 0),
        "up": (0, 0, 1),
        "down": (0, 0, -1),
    }
    try:
        columns = tuple(direction[axis.strip().casefold()] for axis in axes)
    except KeyError as exc:
        raise ValueError(
            "fixed-axis roles must use recognized cardinal axis directions"
        ) from exc
    a, b, c = columns
    determinant = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    if abs(determinant) != 1:
        raise ValueError("fixed-axis directions must form an orthogonal triad")
    return "right" if determinant > 0 else "left"


_ROLE_ITEMS = (
    SemanticContract(
        "unreal.world-axes",
        "target_defined",
        "Unreal world axes",
        "target_documentation",
        "fixed_axes",
        "https://dev.epicgames.com/documentation/en-us/unreal-engine/coordinate-system-and-spaces-in-unreal-engine?application_version=5.7",
        "5.7",
        "Coordinate System; World Space",
        "Unreal",
        ("forward", "right", "up"),
        "left",
        "world",
        "scene_alignment",
        source_claim="Unreal Editor axes are +X forward, +Y right, +Z up; the editor coordinate system is left-handed.",
        assurance_scope="Unreal Editor world-axis role for the documented 5.7 coordinate-space contract.",
        exclusions=(
            "project-specific coordinate-space overrides",
            "asset or mesh local-forward axis",
            "geographic alignment",
        ),
        converter_ids=("unreal.fquat",),
        execution_evidence_ids=("unreal.native.5.7.4",),
    ),
    SemanticContract(
        "unity.world-axes",
        "target_defined",
        "Unity world axes",
        "target_documentation",
        "fixed_axes",
        "https://docs.unity.com/en-us/engine/6000.5/manual/scripting/programming-math/unity-engine-math/class-quaternion",
        "6000.5",
        "Coordinate handedness",
        "Unity",
        ("right", "up", "forward"),
        "left",
        "world",
        "scene_alignment",
        source_claim="Unity's documented left-handed coordinate system uses +X right, +Y up, +Z forward.",
        assurance_scope="Unity world-coordinate convention documented for the 6000.5 quaternion API.",
        exclusions=(
            "imported model local axes",
            "project-specific scene alignment",
            "geographic alignment",
        ),
        converter_ids=("unity.quaternion",),
        execution_evidence_ids=("unity.native.6000.5.2f1",),
    ),
    SemanticContract(
        "godot.camera-axes",
        "target_defined",
        "Godot Camera3D local axes",
        "target_documentation",
        "fixed_axes",
        "https://docs.godotengine.org/en/4.7/tutorials/assets_pipeline/importing_3d_scenes/model_export_considerations.html",
        "4.7",
        "3D asset direction conventions",
        "Godot",
        ("right", "up", "back"),
        "right",
        "sensor",
        "sensor_mount",
        source_claim="Godot 4.7 documents right-handed 3D coordinates with +Y up and -Z as the camera-forward direction.",
        assurance_scope="Godot Camera3D local sensor-axis role, interpreted with the engine's documented 3D coordinates.",
        exclusions=(
            "arbitrary Node3D world alignment",
            "asset-forward conventions",
            "camera mount calibration",
        ),
        converter_ids=("godot.quaternion",),
        execution_evidence_ids=("godot.native.4.7", "godot.native.4.8-dev1"),
    ),
    SemanticContract(
        "ros.rep103.body-flu",
        "standard_defined",
        "ROS body FLU frame",
        "formal_standard",
        "fixed_axes",
        "https://github.com/openrobotics/reps/blob/104970e7fb6f9a5bac2c0ec820d09ab443e38150/_posts/rep-0103.md",
        "REP-103",
        "Axis Orientation",
        "ROS 2",
        ("forward", "left", "up"),
        "right",
        "body",
        "body_alignment",
        source_claim="REP-103 recommends right-handed body frames with +X forward, +Y left, +Z up; it permits documented exceptions.",
        assurance_scope="ROS conventional body FLU frame under REP-103, not a rule that every ROS frame complies.",
        exclusions=(
            "per-robot body-frame compliance",
            "sensor mount calibration",
            "nonconforming application frames",
        ),
        converter_ids=("ros2.geometry-msgs-quaternion",),
        execution_evidence_ids=("ros2.tf2.0.36.23",),
    ),
    SemanticContract(
        "ros.rep103.optical",
        "standard_defined",
        "ROS camera optical frame",
        "formal_standard",
        "fixed_axes",
        "https://github.com/openrobotics/reps/blob/104970e7fb6f9a5bac2c0ec820d09ab443e38150/_posts/rep-0103.md",
        "REP-103",
        "Suffix Frames",
        "ROS 2",
        ("right", "down", "forward"),
        "right",
        "sensor",
        "sensor_mount",
        source_claim="REP-103 defines camera optical axes as +Z forward, +X right, +Y down.",
        assurance_scope="ROS camera optical frame convention under REP-103 suffix-frame guidance.",
        exclusions=(
            "camera-to-body extrinsics",
            "sensor mounting calibration",
            "nonconforming application frames",
        ),
        converter_ids=("ros2.geometry-msgs-quaternion",),
        execution_evidence_ids=("ros2.tf2.0.36.23",),
    ),
    SemanticContract(
        "cesium.enu-to-fixed",
        "target_defined",
        "Cesium local ENU frame",
        "target_documentation",
        "geographic_enu",
        "https://cesium.com/downloads/cesiumjs/releases/1.124/Build/Documentation/Transforms.html#eastNorthUpToFixedFrame",
        "1.124.0",
        "Transforms.eastNorthUpToFixedFrame",
        "CesiumJS",
        frame_role="world",
        source_claim="Cesium defines local x=east, y=north, z=ellipsoid-surface normal at the supplied origin, mapped to the selected ellipsoid's fixed frame.",
        assurance_scope="Cesium local ENU-to-ellipsoid-fixed basis at an explicit origin and ellipsoid.",
        exclusions=(
            "body-forward alignment",
            "inertial-to-fixed conversion",
            "time-dependent Earth orientation",
            "position or pose translation",
        ),
        converter_ids=("cesium.quaternion",),
        execution_evidence_ids=("cesium.native.1.124.0",),
    ),
    SemanticContract(
        "gltf.asset-front",
        "standard_defined",
        "glTF oriented asset frame",
        "formal_standard",
        "fixed_axes",
        "https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html",
        "2.0",
        "3.4 Coordinate System and Units",
        "glTF",
        ("left", "up", "forward"),
        "right",
        "body",
        "model_alignment",
        source_claim="glTF 2.0 defines +Y up, +Z forward, and -X right, and says the front of an oriented asset faces +Z.",
        assurance_scope="glTF oriented-asset coordinate convention and static node rotation field.",
        exclusions=(
            "proof that arbitrary mesh geometry was authored facing +Z",
            "application model alignment",
            "animation interpolation",
        ),
        converter_ids=("gltf.rotation",),
        execution_evidence_ids=("gltf.threejs.loader.0.187.0",),
    ),
    SemanticContract(
        "gltf.node-local",
        "standard_defined",
        "glTF node-local to parent relation",
        "formal_standard",
        "relational",
        "https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html",
        "2.0",
        "3.5.2 Nodes and Hierarchy; 3.5.3 Transformations; 5.25.6 node.rotation",
        "glTF",
        source_claim="glTF node transforms are local to the node's parent; the node rotation is a unit quaternion stored XYZW.",
        assurance_scope="Static glTF node-local rotation field and hierarchy behavior in the pinned loader fixture.",
        exclusions=(
            "general scene-export/import semantics",
            "animation curves",
            "application model alignment",
        ),
        converter_ids=("gltf.rotation",),
        execution_evidence_ids=("gltf.threejs.loader.0.187.0",),
    ),
    SemanticContract(
        "openxr.reference-space",
        "standard_defined",
        "OpenXR LOCAL or LOCAL_FLOOR reference-space axes",
        "formal_standard",
        "fixed_axes",
        "https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrReferenceSpaceType.html",
        "1.1.65",
        "XR_REFERENCE_SPACE_TYPE_LOCAL; XR_REFERENCE_SPACE_TYPE_LOCAL_FLOOR",
        "OpenXR",
        ("right", "up", "back"),
        "right",
        "world",
        "space_alignment",
        source_claim="OpenXR 1.1 LOCAL and LOCAL_FLOOR use +X right, +Y up, -Z forward; their origins are runtime-defined and can be revised.",
        assurance_scope="Named LOCAL or LOCAL_FLOOR reference-space orientation at the supplied runtime, base-space, and XrTime context.",
        exclusions=(
            "geodetic alignment",
            "vendor-runtime tracking quality",
            "STAGE or arbitrary action spaces",
            "validity without orientation-valid flag",
        ),
        converter_ids=("openxr.xrquaternionf",),
        execution_evidence_ids=("openxr.test-runtime.1.1.65",),
    ),
    SemanticContract(
        "openxr.view-space",
        "standard_defined",
        "OpenXR VIEW reference-space axes",
        "formal_standard",
        "fixed_axes",
        "https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrReferenceSpaceType.html",
        "1.1.65",
        "XR_REFERENCE_SPACE_TYPE_VIEW",
        "OpenXR",
        ("right", "up", "back"),
        "right",
        "sensor",
        "space_alignment",
        source_claim="OpenXR 1.1 VIEW uses +X right, +Y up, -Z forward and tracks the viewer without gravity alignment.",
        assurance_scope="Named VIEW-space orientation at the supplied runtime, base-space, and XrTime context.",
        exclusions=(
            "gravity-aligned orientation",
            "vendor-runtime tracking quality",
            "invalid orientation locations",
        ),
        converter_ids=("openxr.xrquaternionf",),
        execution_evidence_ids=("openxr.test-runtime.1.1.65",),
    ),
    SemanticContract(
        "quatops.parent-child-orientation",
        "application_declared",
        "Directed parent-from-child orientation",
        "application_contract",
        "application_frames",
        application_source="docs/semantics.md",
        source_claim="QuatOps defines q_parent_from_child as the active rotation mapping child coordinates into parent coordinates.",
        assurance_scope="User-declared parent and child coordinate frames for a physical orientation.",
        exclusions=(
            "automatic discovery of frame axes",
            "physical calibration",
            "target-specific native behavior",
        ),
    ),
)
SEMANTIC_CONTRACTS: Mapping[str, SemanticContract] = MappingProxyType(
    {contract.contract_id: contract for contract in _ROLE_ITEMS}
)
