"""Validate the bounded authored-mesh loader witness, without inferring asset axes."""

from __future__ import annotations

import base64
import json
import math
import struct
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import Field

from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.primitives import (
    Quaternion4,
    StrictEvidenceModel,
    Vector3,
)
from scripts.evidence_schema.utilities import sha256_file


class MeshVertex(StrictEvidenceModel):
    index: Literal[0, 1, 2, 3]
    local_xyz: Vector3
    parent_xyz: Vector3
    world_xyz: Vector3


class AuthoredMeshCapture(StrictEvidenceModel):
    schema_version: Literal["1.0"]
    evidence_kind: Literal["native_threejs_authored_mesh_capture"]
    node: Literal["OrientedMesh"]
    parent: Literal["FixtureParent"]
    geometry_type: Literal["BufferGeometry"]
    indices: Annotated[list[int], Field(min_length=12, max_length=12)]
    local_quaternion_xyzw: Quaternion4
    parent_quaternion_xyzw: Quaternion4
    vertices: Annotated[list[MeshVertex], Field(min_length=4, max_length=4)]


class MeshTarget(StrictEvidenceModel):
    name: Literal["Three.js GLTFLoader"]
    version: Literal["0.187.0"]
    revision: Literal["187"]


class MeshExecution(StrictEvidenceModel):
    was_run: Literal[True]
    utc: str
    node: str
    platform: str
    architecture: str


class MeshProvenance(StrictEvidenceModel):
    asset_sha256: str
    capture_sha256: str
    package_lock_sha256: str
    witnesses_sha256: str


class AuthoredMeshManifest(StrictEvidenceModel):
    schema_version: Literal["1.0"]
    evidence_kind: Literal["native_threejs_authored_mesh_capture"]
    target: MeshTarget
    execution: MeshExecution
    scope: Literal[
        "Authored asymmetric mesh vertices transformed through static parent-child hierarchy"
    ]
    authority: Literal["application_declared"]
    excludes: list[str]
    provenance: MeshProvenance


class _NodePose(StrictEvidenceModel):
    rotation: Quaternion4
    translation: Vector3


def _matrix(q: list[float]) -> NDArray[np.float64]:
    x, y, z, w = q
    if not math.isclose(sum(value * value for value in q), 1.0, abs_tol=2e-14):
        raise ValueError("authored mesh rotation is not unit length")
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _equal(actual: ArrayLike, expected: ArrayLike, label: str) -> None:
    if not np.allclose(actual, expected, rtol=0, atol=2e-14):
        raise ValueError(f"oriented mesh {label} differs from authored geometry/action")


def validate_oriented_mesh(root: Path, witness_path: Path) -> tuple[str, set[str]]:
    """Validate native geometry, scoped authorship, provenance and hierarchy action.

    Returns only the actual authored-mesh witness, never general asset alignment.
    """
    experiment = root / "experiments/transports/gltf/2.0.0/oriented_mesh"
    capture = AuthoredMeshCapture.model_validate(
        read_evidence_document(witness_path), strict=True
    )
    manifest = AuthoredMeshManifest.model_validate(
        read_evidence_document(witness_path.parent / "provenance.json"), strict=True
    )
    paths = {
        "asset_sha256": experiment / "oriented_mesh.gltf",
        "capture_sha256": experiment / "capture.mjs",
        "witnesses_sha256": witness_path,
        "package_lock_sha256": root
        / "experiments/transports/gltf/2.0.0/threejs/0.187.0/package-lock.json",
    }
    for key, path in paths.items():
        if sha256_file(path) != getattr(manifest.provenance, key):
            raise ValueError(f"oriented mesh {key} differs from provenance")
    if "Arbitrary mesh-front inference" not in manifest.excludes:
        raise ValueError("oriented mesh cannot infer arbitrary asset alignment")
    # This is a bounded fixture validator, not a generic glTF parser. Check the
    # exact author declarations and binary positions before interpreting output.
    asset = read_evidence_document(paths["asset_sha256"])
    expected_structure = {
        "asset": {
            "version": "2.0",
            "generator": "QuatOps explicit oriented-mesh fixture",
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "meshes": [
            {
                "name": "AsymmetricTetrahedron",
                "primitives": [
                    {"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}
                ],
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 48, "target": 34962},
            {"buffer": 0, "byteOffset": 48, "byteLength": 24, "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
                "min": [-3.0, 0.0, 0.0],
                "max": [0.0, 5.0, 2.0],
            },
            {"bufferView": 1, "componentType": 5123, "count": 12, "type": "SCALAR"},
        ],
    }
    if set(asset) != set(expected_structure) | {"nodes", "buffers"}:
        raise ValueError("oriented mesh asset fields differ")
    # Canonical JSON comparison also rejects bool-as-index (True == 1 in Python).
    actual_structure = {key: asset[key] for key in expected_structure}
    if json.dumps(actual_structure, sort_keys=True) != json.dumps(
        expected_structure, sort_keys=True
    ):
        raise ValueError("oriented mesh accessor/primitive/scene structure differs")
    nodes = asset.get("nodes")
    buffers = asset.get("buffers")
    if not isinstance(nodes, list) or len(nodes) != 2 or not isinstance(buffers, list):
        raise ValueError("oriented mesh asset structure differs")
    parent, child = nodes
    if not isinstance(parent, dict) or not isinstance(child, dict):
        raise TypeError("oriented mesh node structure differs")
    if (
        set(parent) != {"name", "translation", "rotation", "children"}
        or set(child) != {"name", "translation", "rotation", "mesh", "extras"}
        or parent["name"] != capture.parent
        or child["name"] != capture.node
    ):
        raise ValueError("oriented mesh node identity or transform fields differ")
    expected_assignment = {
        "frame_authority": "application_declared",
        "front_vertex": 1,
        "right_vertex": 2,
        "up_vertex": 3,
        "origin_vertex": 0,
        "forward": "+Z",
        "right": "-X",
        "up": "+Y",
    }
    if json.dumps(child.get("extras"), sort_keys=True) != json.dumps(
        expected_assignment, sort_keys=True
    ):
        raise ValueError("oriented mesh author alignment differs")
    if (
        json.dumps(parent.get("children")) != "[1]"
        or json.dumps(child.get("mesh")) != "0"
    ):
        raise ValueError("oriented mesh hierarchy differs")
    if len(buffers) != 1 or not isinstance(buffers[0], dict):
        raise ValueError("oriented mesh binary buffer differs")
    if (
        set(buffers[0]) != {"byteLength", "uri"}
        or type(buffers[0]["byteLength"]) is not int
        or buffers[0]["byteLength"] != 72
    ):
        raise ValueError("oriented mesh buffer declaration differs")
    uri = buffers[0].get("uri")
    if not isinstance(uri, str):
        raise ValueError("oriented mesh embedded buffer is missing")
    media_type, separator, encoded = uri.partition(",")
    if media_type != "data:application/octet-stream;base64" or not separator:
        raise ValueError("oriented mesh embedded buffer is missing")
    binary = base64.b64decode(encoded, validate=True)
    if len(binary) != 72:
        raise ValueError("oriented mesh binary length differs")
    expected_vertices = np.asarray([[0, 0, 0], [0, 0, 2], [-3, 0, 0], [0, 5, 0]])
    _equal(
        np.asarray(struct.unpack("<12f", binary[:48])).reshape(4, 3),
        expected_vertices,
        "vertices",
    )
    if list(struct.unpack("<12H", binary[48:])) != capture.indices:
        raise ValueError("oriented mesh topology differs")
    parent_pose = _NodePose.model_validate(
        {"rotation": parent.get("rotation"), "translation": parent.get("translation")}
    )
    child_pose = _NodePose.model_validate(
        {"rotation": child.get("rotation"), "translation": child.get("translation")}
    )
    _equal(parent_pose.rotation, capture.parent_quaternion_xyzw, "parent quaternion")
    _equal(child_pose.rotation, capture.local_quaternion_xyzw, "child quaternion")
    rp, rc = (
        _matrix(capture.parent_quaternion_xyzw),
        _matrix(capture.local_quaternion_xyzw),
    )
    if np.allclose(rp @ rc, rc @ rp):
        raise ValueError("oriented mesh must witness noncommuting composition")
    if [row.index for row in capture.vertices] != list(range(4)):
        raise ValueError("oriented mesh vertex identities differ")
    for row in capture.vertices:
        _equal(row.local_xyz, expected_vertices[row.index], "local position")
        parent_point = rc @ expected_vertices[row.index] + np.asarray(
            child_pose.translation
        )
        world_point = rp @ parent_point + np.asarray(parent_pose.translation)
        _equal(row.parent_xyz, parent_point, "parent action")
        _equal(row.world_xyz, world_point, "world action")
    return manifest.target.version, {capture.node}
