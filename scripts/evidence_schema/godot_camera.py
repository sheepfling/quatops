"""Native Camera3D node observations, separate from quaternion-only captures."""

import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, FiniteFloat

from quatops.scalar import rotate
from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.primitives import NativeCaptureModel
from scripts.evidence_schema.utilities import sha256_file

Vector3 = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
Quaternion = Annotated[list[FiniteFloat], Field(min_length=4, max_length=4)]


class CameraAxes(NativeCaptureModel):
    right: Vector3
    up: Vector3
    forward: Vector3


class CameraCase(NativeCaptureModel):
    id: Literal["0_0", "1_2", "2_1"]
    parent_xyzw: Quaternion
    camera_xyzw: Quaternion
    local_axes: CameraAxes
    global_axes: CameraAxes
    camera_axes: CameraAxes
    center_ray: Vector3
    global_origin: Vector3
    camera_origin: Vector3


class GodotCameraCapture(NativeCaptureModel):
    schema_version: Literal[1]
    target: Literal["Godot Camera3D"]
    version: Literal["4.7-stable (official)", "4.8-dev1 (official)"]
    build_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    platform: Literal["Windows"]
    cases: Annotated[list[CameraCase], Field(min_length=3, max_length=3)]


class CameraManifest(NativeCaptureModel):
    schema_version: Literal[1]
    target: Literal["Godot Camera3D"]
    version: Literal["4.7-stable (official)", "4.8-dev1 (official)"]
    build_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    platform: Literal["Windows"]
    capture_source: Literal["experiments/godot/camera3d/capture.gd"]
    capture_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    semantic_contract: Literal["experiments/godot/camera3d/semantic_contract.json"]
    semantic_contract_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    witnesses_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    executable_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    native_run_date: str
    execution: str


class CameraContract(NativeCaptureModel):
    schema_version: Literal[1]
    contract_id: Literal["godot-camera3d-node-action-v1"]
    versions: list[Literal["4.7-stable", "4.8-dev1"]]
    frame_authority: Literal["application_declared"]
    world_axes: list[Literal["right", "up", "backward"]]
    camera_local_axes: list[Literal["right", "up", "backward"]]
    forward: Literal["negative_z"]
    coefficient_order: Literal["xyzw"]
    action: Literal["active"]
    composition: Literal["parent * camera; camera acts first"]
    parent_translation: Vector3
    camera_translation: Vector3
    scope: list[str]
    excludes: list[str]


def _close(actual: list[float], expected: list[float]) -> None:
    if len(actual) != len(expected) or any(
        not math.isclose(a, b, abs_tol=2e-6, rel_tol=0)
        for a, b in zip(actual, expected)
    ):
        raise ValueError("Camera3D native action differs from declared contract")


def _action(q: list[float], v: list[float]) -> list[float]:
    return list(rotate((q[3], q[0], q[1], q[2]), (v[0], v[1], v[2])))


def load_camera_capture(root: Path, path: Path) -> GodotCameraCapture:
    """Validate target types, provenance, inputs and native camera action."""
    manifest = CameraManifest.model_validate(
        read_evidence_document(path.parent / "provenance.json"), strict=True
    )
    cp = root / manifest.semantic_contract
    contract = CameraContract.model_validate(read_evidence_document(cp), strict=True)
    for file, digest in [
        (path, manifest.witnesses_sha256),
        (root / manifest.capture_source, manifest.capture_sha256),
        (cp, manifest.semantic_contract_sha256),
    ]:
        if sha256_file(file) != digest:
            raise ValueError("Camera3D provenance digest differs")
    capture = GodotCameraCapture.model_validate(
        read_evidence_document(path), strict=True
    )
    if (
        capture.version != manifest.version
        or capture.build_hash != manifest.build_hash
        or capture.platform != manifest.platform
        or capture.version.split(" ")[0] not in contract.versions
    ):
        raise ValueError("Camera3D target identity differs from provenance")
    if (
        contract.world_axes != ["right", "up", "backward"]
        or contract.camera_local_axes != contract.world_axes
    ):
        raise ValueError("Camera3D assigned frame axes differ")
    if contract.parent_translation != [
        1.25,
        -2.5,
        4.0,
    ] or contract.camera_translation != [-0.5, 1.75, 2.25]:
        raise ValueError("Camera3D translation assignment differs")
    if [c.id for c in capture.cases] != ["0_0", "1_2", "2_1"]:
        raise ValueError("Camera3D retained cases differ")
    rotations = [
        [0.0, 0.0, 0.0, 1.0],
        [0.2, 0.4, 0.4, 0.8],
        [-0.3, 0.6164414002968976, -0.2, 0.7],
    ]
    for c in capture.cases:
        pi, ci = (int(i) for i in c.id.split("_"))
        _close(c.parent_xyzw, rotations[pi])
        _close(c.camera_xyzw, rotations[ci])
        for name, axis in [
            ("right", [1.0, 0.0, 0.0]),
            ("up", [0.0, 1.0, 0.0]),
            ("forward", [0.0, 0.0, -1.0]),
        ]:
            local = _action(c.camera_xyzw, axis)
            world = _action(c.parent_xyzw, local)
            _close(getattr(c.local_axes, name), local)
            _close(getattr(c.global_axes, name), world)
            _close(getattr(c.camera_axes, name), world)
        _close(
            c.center_ray,
            _action(c.parent_xyzw, _action(c.camera_xyzw, [0.0, 0.0, -1.0])),
        )
        translation = _action(c.parent_xyzw, contract.camera_translation)
        origin = [a + b for a, b in zip(contract.parent_translation, translation)]
        _close(c.global_origin, origin)
        _close(c.camera_origin, origin)
    return capture
