"""Pinned native loader proof for one explicitly authored oriented mesh."""

from __future__ import annotations

import base64
import hashlib
import json
import struct

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quatops.experimental.engines import QuaternionFrameMap, SignedAxisMap
from tests.support.paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
EXPERIMENT = ROOT / "experiments/transports/gltf/2.0.0/oriented_mesh"
FIXTURE = ROOT / "tests/golden/transports/gltf/2.0.0/threejs/0.187.0/oriented_mesh"
ASSET = json.loads((EXPERIMENT / "oriented_mesh.gltf").read_text(encoding="utf-8"))
CAPTURE = json.loads((FIXTURE / "witnesses.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((FIXTURE / "provenance.json").read_text(encoding="utf-8"))


def test_native_mesh_capture_identity_and_source_hashes():
    assert MANIFEST["target"] == {
        "name": "Three.js GLTFLoader",
        "version": "0.187.0",
        "revision": "187",
    }
    assert MANIFEST["execution"]["was_run"] is True
    assert MANIFEST["authority"] == "application_declared"
    assert "Arbitrary mesh-front inference" in MANIFEST["excludes"]
    paths = {
        "asset_sha256": EXPERIMENT / "oriented_mesh.gltf",
        "capture_sha256": EXPERIMENT / "capture.mjs",
        "witnesses_sha256": FIXTURE / "witnesses.json",
        "package_lock_sha256": ROOT
        / "experiments/transports/gltf/2.0.0/threejs/0.187.0/package-lock.json",
    }
    for key, path in paths.items():
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest() == MANIFEST["provenance"][key]
        )
    assert CAPTURE["node"] == "OrientedMesh"
    assert CAPTURE["parent"] == "FixtureParent"
    assert CAPTURE["geometry_type"] == "BufferGeometry"
    assert len(CAPTURE["vertices"]) == 4
    assert [v["index"] for v in CAPTURE["vertices"]] == list(range(4))


def test_authored_asymmetric_geometry_survives_real_loader():
    child = ASSET["nodes"][1]
    assert child["extras"] == {
        "frame_authority": "application_declared",
        "front_vertex": 1,
        "right_vertex": 2,
        "up_vertex": 3,
        "origin_vertex": 0,
        "forward": "+Z",
        "right": "-X",
        "up": "+Y",
    }
    payload = base64.b64decode(ASSET["buffers"][0]["uri"].split(",", 1)[1])
    positions = np.asarray(struct.unpack("<12f", payload[:48])).reshape(4, 3)
    np.testing.assert_array_equal(
        positions, [[0, 0, 0], [0, 0, 2], [-3, 0, 0], [0, 5, 0]]
    )
    np.testing.assert_array_equal(
        [v["local_xyz"] for v in CAPTURE["vertices"]], positions
    )
    assert CAPTURE["indices"] == list(struct.unpack("<12H", payload[48:]))
    # Four unequal geometric witnesses prevent a symmetric mesh masking a flipped front.
    assert np.linalg.det(np.stack([positions[2], positions[3], positions[1]])) != 0


@pytest.mark.parametrize("index", range(4))
def test_native_vertex_actions_match_independent_parent_child_oracle(index):
    parent, child = ASSET["nodes"]
    rp = Rotation.from_quat(parent["rotation"]).as_matrix()
    rc = Rotation.from_quat(child["rotation"]).as_matrix()
    assert not np.allclose(rp @ rc, rc @ rp)
    observation = CAPTURE["vertices"][index]
    local = np.asarray(observation["local_xyz"])
    expected_parent = rc @ local + child["translation"]
    expected_world = rp @ expected_parent + parent["translation"]
    np.testing.assert_allclose(
        observation["parent_xyz"], expected_parent, rtol=0, atol=2e-14
    )
    np.testing.assert_allclose(
        observation["world_xyz"], expected_world, rtol=0, atol=2e-14
    )
    recovered = rc.T @ (
        rp.T @ (expected_world - parent["translation"]) - child["translation"]
    )
    np.testing.assert_allclose(recovered, local, rtol=0, atol=2e-14)


def test_explicit_asset_front_binding_is_not_camera_probe_binding():
    # Source axes are +X right, +Y forward, +Z up. This asset was authored
    # -X right, +Y up, +Z forward; a camera-like (-Z-forward) map is different.
    basis = SignedAxisMap(("-x", "z", "y"))
    converter = QuaternionFrameMap(basis, basis)
    q_source_wxyz = (0.8, 0.0, 0.0, 0.6)
    q_target = converter.to_destination_wxyz(q_source_wxyz)
    # Converted WXYZ coefficients become the glTF XYZW field.
    np.testing.assert_allclose(
        [q_target[1], q_target[2], q_target[3], q_target[0]],
        ASSET["nodes"][1]["rotation"],
        rtol=0,
        atol=2e-14,
    )
    observations = CAPTURE["vertices"]
    world_origin = np.asarray(observations[0]["world_xyz"])
    actual_front = (np.asarray(observations[1]["world_xyz"]) - world_origin) / 2
    actual_right = (np.asarray(observations[2]["world_xyz"]) - world_origin) / 3
    actual_up = (np.asarray(observations[3]["world_xyz"]) - world_origin) / 5
    rp = Rotation.from_quat(ASSET["nodes"][0]["rotation"]).as_matrix()
    rc = Rotation.from_quat(ASSET["nodes"][1]["rotation"]).as_matrix()
    for actual, local in [
        (actual_front, [0, 0, 1]),
        (actual_right, [-1, 0, 0]),
        (actual_up, [0, 1, 0]),
    ]:
        np.testing.assert_allclose(actual, rp @ rc @ local, rtol=0, atol=2e-14)
    assert not np.allclose(actual_front, rp @ rc @ [0, 0, -1])


def test_typed_native_mesh_coverage_is_narrow():
    from scripts.evidence_schema.oriented_mesh import validate_oriented_mesh

    assert validate_oriented_mesh(ROOT, FIXTURE / "witnesses.json") == (
        "0.187.0",
        {"OrientedMesh"},
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "front",
        "world",
        "version",
        "unknown",
        "vertex_identity",
        "position_accessor",
        "index_accessor",
        "buffer_offset",
        "scene",
        "connectivity",
        "name",
        "scale",
        "matrix",
        "boolean_index",
        "accessor_offset",
        "buffer_stride",
    ],
)
def test_rehashed_semantic_or_native_mutation_is_rejected(tmp_path, mutation):
    import shutil

    from scripts.evidence_schema.oriented_mesh import validate_oriented_mesh

    asset = tmp_path / (EXPERIMENT / "oriented_mesh.gltf").relative_to(ROOT)
    witness = tmp_path / (FIXTURE / "witnesses.json").relative_to(ROOT)
    provenance = witness.parent / "provenance.json"
    for source in [
        EXPERIMENT / "oriented_mesh.gltf",
        EXPERIMENT / "capture.mjs",
        FIXTURE / "witnesses.json",
        FIXTURE / "provenance.json",
        ROOT / "experiments/transports/gltf/2.0.0/threejs/0.187.0/package-lock.json",
    ]:
        destination = tmp_path / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    manifest = json.loads(provenance.read_text(encoding="utf-8"))
    if mutation in {
        "front",
        "position_accessor",
        "index_accessor",
        "buffer_offset",
        "scene",
        "connectivity",
        "name",
        "scale",
        "matrix",
        "boolean_index",
        "accessor_offset",
        "buffer_stride",
    }:
        value = json.loads(asset.read_text(encoding="utf-8"))
        if mutation == "front":
            value["nodes"][1]["extras"]["forward"] = "-Z"
        elif mutation == "position_accessor":
            value["meshes"][0]["primitives"][0]["attributes"]["POSITION"] = 1
        elif mutation == "index_accessor":
            value["meshes"][0]["primitives"][0]["indices"] = 0
        elif mutation == "boolean_index":
            value["meshes"][0]["primitives"][0]["indices"] = True
        elif mutation == "buffer_offset":
            value["bufferViews"][0]["byteOffset"] = 12
        elif mutation == "accessor_offset":
            value["accessors"][0]["byteOffset"] = 12
        elif mutation == "buffer_stride":
            value["bufferViews"][0]["byteStride"] = 16
        elif mutation == "scene":
            value["scenes"][0]["nodes"] = [1]
        elif mutation == "connectivity":
            value["nodes"][0]["children"] = []
        elif mutation == "name":
            value["nodes"][1]["name"] = "NotOrientedMesh"
        elif mutation == "scale":
            value["nodes"][1]["scale"] = [-1, 1, 1]
        else:
            value["nodes"][1]["matrix"] = list(np.eye(4).reshape(-1))
        asset.write_text(json.dumps(value), encoding="utf-8")
        manifest["provenance"]["asset_sha256"] = hashlib.sha256(
            asset.read_bytes()
        ).hexdigest()
    elif mutation == "version":
        manifest["target"]["version"] = "0.188.0"
    else:
        value = json.loads(witness.read_text(encoding="utf-8"))
        if mutation == "world":
            value["vertices"][1]["world_xyz"][0] += 0.1
        elif mutation == "vertex_identity":
            value["vertices"][1]["index"] = 0
        else:
            value["arbitrary_mesh_front_inferred"] = True
        witness.write_text(json.dumps(value), encoding="utf-8")
        manifest["provenance"]["witnesses_sha256"] = hashlib.sha256(
            witness.read_bytes()
        ).hexdigest()
    provenance.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_oriented_mesh(tmp_path, witness)
