"""QuatOps-to-glTF file serialization and pinned Three.js GLTFLoader proof."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from experiments.converters.gltf.inputs import build_artifacts, serialize_artifact
from quatops.experimental.engines import QuaternionFrameMap, SignedAxisMap
from tests.support.evidence_oracles import matrix_from_wxyz as rotation
from tests.support.evidence_oracles import native_values as values
from tests.support.paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
FIXTURE = ROOT / "tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter"
CONTRACT_PATH = ROOT / "experiments/converters/gltf/2.0.0/semantic_contract.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
MANIFEST = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
INPUT_MODEL, ASSET = build_artifacts(ROOT)
INPUTS = INPUT_MODEL.model_dump(mode="json")
INPUT_BYTES = serialize_artifact(INPUTS)
ASSET_BYTES = serialize_artifact(ASSET)
CAPTURE = json.loads((FIXTURE / "witnesses.json").read_text(encoding="utf-8"))
OBS = {row["node"]: row for row in CAPTURE["observations"]}
FRAME = SignedAxisMap(("x", "z", "-y"))
TO_GLTF = QuaternionFrameMap(FRAME, FRAME)
MAP = np.asarray(FRAME.matrix, dtype=float)
ATOL = 2e-12


def check(actual, expected, message):
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=ATOL, err_msg=message)


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
    assert CONTRACT["probe_frame"]["forward"] == "-Z"
    assert "asset_frame" not in CONTRACT
    assert "camera-like" in CONTRACT["frame_role_note"]
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
        "native_observations_sha256": FIXTURE / "witnesses.json",
        "contract_sha256": CONTRACT_PATH,
    }
    for field, path in provenance_paths.items():
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == MANIFEST["provenance"][field]
        )
    assert (
        hashlib.sha256(INPUT_BYTES).hexdigest()
        == MANIFEST["provenance"]["input_sha256"]
    )
    assert (
        hashlib.sha256(ASSET_BYTES).hexdigest()
        == MANIFEST["provenance"]["asset_sha256"]
    )
    assert MANIFEST["provenance"]["input_serialization_note"].startswith(
        "JSON formatting changed only"
    )
    assert (
        MANIFEST["provenance"]["captured_input_sha256"]
        == MANIFEST["provenance"]["input_sha256"]
    )

    reproduction_paths = {
        "generator_sha256": ROOT / "experiments/converters/gltf/2.0.0/create_asset.py",
        "input_builder_sha256": ROOT / "experiments/converters/gltf/inputs.py",
        "loader_script_sha256": ROOT
        / "experiments/converters/gltf/2.0.0/capture_threejs.mjs",
        "package_lock_sha256": ROOT
        / "experiments/transports/gltf/2.0.0/threejs/0.187.0/package-lock.json",
        "numeric_helper_sha256": ROOT
        / "experiments/converters/shared/numeric_evidence.mjs",
        "retention_formatter_sha256": ROOT / "scripts/evidence_format.py",
    }
    for field, path in reproduction_paths.items():
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == MANIFEST["reproduction"][field]
        )
    assert (
        MANIFEST["direct_reproduction"]["native_script_sha256"]
        == MANIFEST["reproduction"]["loader_script_sha256"]
    )
    assert (
        MANIFEST["direct_reproduction"]["reproduction_runner_sha256"]
        == hashlib.sha256(
            (ROOT / "experiments/run_converter_capture.py").read_bytes()
        ).hexdigest()
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
