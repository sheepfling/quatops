"""QuatOps-to-glTF file serialization and pinned Three.js GLTFLoader proof."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from quatops.experimental.engines import QuaternionFrameMap, SignedAxisMap
from tests.support.paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
FIXTURE = ROOT / "tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter"
INPUTS_PATH = ROOT / "experiments/converters/gltf/2.0.0/converter_inputs.json"
ASSET_PATH = ROOT / "experiments/converters/gltf/2.0.0/quatops_hierarchy.gltf"
CONTRACT = json.loads((FIXTURE / "contract.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
INPUTS = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))
ASSET = json.loads(ASSET_PATH.read_text(encoding="utf-8"))
CAPTURE = json.loads((FIXTURE / "witnesses.json").read_text(encoding="utf-8"))
OBS = {row["node"]: row for row in CAPTURE["observations"]}
FRAME = SignedAxisMap(("x", "z", "-y"))
TO_GLTF = QuaternionFrameMap(FRAME, FRAME)
MAP = np.asarray(FRAME.matrix, dtype=float)
ATOL = 2e-12


def values(item):
    if isinstance(item, dict) and "values" in item and "dtype" in item:
        return item["values"]
    if isinstance(item, list):
        return [values(child) for child in item]
    if isinstance(item, dict):
        return {key: values(child) for key, child in item.items()}
    return item


def check(actual, expected, message):
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=ATOL, err_msg=message)


def rotation(q_wxyz):
    q = np.asarray(q_wxyz, dtype=float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def target_rotation(q_wxyz):
    return MAP @ rotation(q_wxyz) @ MAP.T


def trs(rotation_matrix, translation):
    result = np.eye(4)
    result[:3, :3] = rotation_matrix
    result[:3, 3] = translation
    return result


def test_gltf_converter_artifact_loader_and_provenance_are_pinned():
    assert ASSET["asset"]["version"] == "2.0"
    assert CONTRACT["destination_axes_from_source"] == ["x", "z", "-y"]
    assert CONTRACT["target"] == "glTF 2.0 loaded by Three.js GLTFLoader 0.187.0"
    assert MANIFEST["target"]["package_version"] == "0.187.0"
    assert MANIFEST["execution"]["was_run"] is True
    assert MANIFEST["selection"]["retained_observations"] == 2
    assert set(OBS) == {"QuatOpsParent", "QuatOpsChild"}
    assert CAPTURE["native_entry_points"] == [
        "GLTFLoader.parse",
        "Object3D.updateMatrixWorld",
        "Object3D.localToWorld",
        "Matrix4.invert",
    ]

    provenance_paths = {
        "input_sha256": INPUTS_PATH,
        "asset_sha256": ASSET_PATH,
        "native_observations_sha256": FIXTURE / "witnesses.json",
        "contract_sha256": FIXTURE / "contract.json",
    }
    for field, path in provenance_paths.items():
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == MANIFEST["provenance"][field]
        )
    assert MANIFEST["provenance"]["input_serialization_note"].startswith(
        "JSON formatting changed only"
    )
    assert len(MANIFEST["provenance"]["captured_input_sha256"]) == 64

    reproduction_paths = {
        "generator_sha256": ROOT / "experiments/converters/gltf/2.0.0/create_asset.py",
        "loader_script_sha256": ROOT
        / "experiments/converters/gltf/2.0.0/capture_threejs.mjs",
        "package_lock_sha256": ROOT
        / "experiments/transports/gltf/2.0.0/threejs/0.187.0/package-lock.json",
        "numeric_helper_sha256": ROOT
        / "experiments/converters/shared/numeric_evidence.mjs",
        "retention_script_sha256": ROOT / "scripts/retain_gltf_converter_witnesses.py",
        "retention_helper_sha256": ROOT / "scripts/evidence_json.py",
        "retention_formatter_sha256": ROOT / "scripts/evidence_format.py",
    }
    for field, path in reproduction_paths.items():
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == MANIFEST["reproduction"][field]
        )

    for source, asset_node in zip(INPUTS["nodes"], ASSET["nodes"]):
        assert asset_node["rotation"] == source["gltf_xyzw"]
        assert asset_node["translation"] == source["gltf_translation_xyz"]


def test_loader_local_actions_hierarchy_composition_and_inverse_preserve_physical_pose():
    probes_gltf = np.asarray(INPUTS["gltf_probe_vectors_xyz"], dtype=float)
    node_inputs = {row["node"]: row for row in INPUTS["nodes"]}
    parent = node_inputs["parent"]
    child = node_inputs["child"]

    expected_local = {}
    for node_name, row in (("QuatOpsParent", parent), ("QuatOpsChild", child)):
        q = np.asarray(row["canonical_wxyz"], dtype=float)
        q_dest = TO_GLTF.to_destination_wxyz(q)
        q_dest_xyzw = np.asarray((q_dest[1], q_dest[2], q_dest[3], q_dest[0]))
        check(
            q_dest_xyzw,
            row["gltf_xyzw"],
            f"QuatOps glTF coefficient map for {node_name}",
        )
        translation = MAP @ np.asarray(row["canonical_translation_xyz"], dtype=float)
        check(
            translation,
            row["gltf_translation_xyz"],
            f"QuatOps glTF translation map for {node_name}",
        )
        r_target = target_rotation(q)
        expected = trs(r_target, translation)
        expected_local[node_name] = expected

        output = OBS[node_name]["outputs"]
        check(
            values(output["local_quaternion_xyzw"]),
            q_dest_xyzw,
            f"GLTFLoader local quaternion for {node_name}",
        )
        native_images = (r_target @ probes_gltf[:3].T).T
        check(
            values(output["local_basis_images_xyz"]),
            native_images,
            f"GLTFLoader local basis action for {node_name}",
        )
        roundtrip = TO_GLTF.from_destination_wxyz((q_dest_xyzw[3], *q_dest_xyzw[:3]))
        check(roundtrip, q, f"glTF inverse converter for {node_name}")

    world_parent = expected_local["QuatOpsParent"]
    world_child = world_parent @ expected_local["QuatOpsChild"]
    child_output = OBS["QuatOpsChild"]["outputs"]
    expected_world_basis = (world_child[:3, :3] @ probes_gltf[:3].T).T
    check(
        values(child_output["world_basis_images_xyz"]),
        expected_world_basis,
        "Three.js loaded child world basis action",
    )
    expected_point = world_child[:3, :3] @ probes_gltf[-1] + world_child[:3, 3]
    check(
        values(child_output["world_probe_point_xyz"]),
        expected_point,
        "Three.js loaded child mixed-point action",
    )
    check(
        values(child_output["roundtrip_probe_point_xyz"]),
        probes_gltf[-1],
        "Three.js inverse world matrix point round trip",
    )

    # The parent and child turns are noncommuting; reversing the hierarchy product is exposed.
    wrong = expected_local["QuatOpsChild"] @ expected_local["QuatOpsParent"]
    assert np.max(np.abs(wrong[:3, :3] - world_child[:3, :3])) > 0.1
