"""QuatOps USDA serialization and pinned OpenUSD stage evaluation proof."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from quatops.engines import QuaternionFrameMap, SignedAxisMap
from tests.support.paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
FIXTURE = ROOT / "tests/golden/transports/openusd/26.11/converter"
CONTRACT = json.loads((FIXTURE / "contract.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
INPUTS = json.loads((FIXTURE / "inputs.json").read_text(encoding="utf-8"))
STAGE = (FIXTURE / "quatops_hierarchy.usda").read_text(encoding="utf-8")
CAPTURE = json.loads((FIXTURE / "witnesses.json").read_text(encoding="utf-8"))
OBS = {row["node"]: row for row in CAPTURE["observations"]}
FRAME = SignedAxisMap(("x", "z", "-y"))
TO_USD = QuaternionFrameMap(FRAME, FRAME)
MAP = np.asarray(FRAME.matrix, dtype=float)
ATOL = 3e-12


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


def usd_row_transform(q_wxyz, translation):
    result = np.eye(4)
    result[:3, :3] = (MAP @ rotation(q_wxyz) @ MAP.T).T
    result[3, :3] = MAP @ np.asarray(translation, dtype=float)
    return result


def test_openusd_converter_stage_provenance_and_frame_declarations():
    assert MANIFEST["target"]["distribution"] == "usd-core"
    assert MANIFEST["target"]["distribution_version"] == "26.11"
    assert MANIFEST["target"]["openusd_version_tuple"] == [0, 26, 11]
    assert MANIFEST["execution"]["was_run"] is True
    assert MANIFEST["stage_metadata"] == {
        "upAxis": "Y",
        "metersPerUnit": 1.0,
        "defaultPrim": "/Scene",
    }
    assert MANIFEST["selection"]["retained_observations"] == 3
    assert CONTRACT["destination_axes_from_source"] == ["x", "z", "-y"]
    assert CONTRACT["matrix_action"].startswith("OpenUSD Gf row vectors")
    assert 'upAxis = "Y"' in STAGE
    assert '"!resetXformStack!"' in STAGE

    provenance = MANIFEST["provenance"]
    paths = {
        "stage_sha256": FIXTURE / "quatops_hierarchy.usda",
        "converter_input_sha256": FIXTURE / "inputs.json",
        "native_observations_sha256": FIXTURE / "witnesses.json",
        "generator_sha256": ROOT
        / "experiments/converters/openusd/26.11/create_stage.py",
        "native_capture_script_sha256": ROOT
        / "experiments/converters/openusd/26.11/capture_native.py",
        "contract_sha256": FIXTURE / "contract.json",
    }
    for field, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance[field]
    assert (
        provenance["lockfile_sha256"]
        == hashlib.sha256(
            (ROOT / "experiments/converters/openusd/26.11/uv.lock").read_bytes()
        ).hexdigest()
    )


def test_openusd_native_orient_ops_local_world_stacks_and_actions_match_quatops():
    nodes = {row["node"]: row for row in INPUTS["nodes"]}
    probes = np.asarray(INPUTS["usd_probe_vectors_xyz"], dtype=float)
    expected_local = {}

    for name in ("parent", "child", "reset_child"):
        item = nodes[name]
        output = OBS[name]["outputs"]
        expected_q = np.asarray(TO_USD.to_destination_wxyz(item["canonical_wxyz"]))
        expected_local[name] = usd_row_transform(
            item["canonical_wxyz"], item["canonical_translation_xyz"]
        )
        check(
            output["orient_op_wxyz"],
            expected_q,
            f"OpenUSD serialized quatd orient op {name}",
        )
        check(
            output["local_matrix_rows"],
            expected_local[name],
            f"OpenUSD local row matrix {name}",
        )
        recovered = TO_USD.from_destination_wxyz(output["orient_op_wxyz"])
        check(
            recovered,
            item["canonical_wxyz"],
            f"OpenUSD inverse quaternion conversion {name}",
        )

        local_points = (
            np.column_stack((probes, np.ones(len(probes)))) @ expected_local[name]
        )[:, :3]
        check(
            output["local_points_xyz"],
            local_points,
            f"OpenUSD local point action {name}",
        )
        check(
            output["world_to_local_point_roundtrip_xyz"],
            probes,
            f"OpenUSD world/local inverse point {name}",
        )

    expected_world = {
        "parent": expected_local["parent"],
        "child": expected_local["child"] @ expected_local["parent"],
        "reset_child": expected_local["reset_child"],
    }
    assert OBS["parent"]["outputs"]["ordered_xform_ops"] == [
        "xformOp:translate",
        "xformOp:orient",
    ]
    assert OBS["child"]["outputs"]["ordered_xform_ops"] == [
        "xformOp:translate",
        "xformOp:orient",
    ]
    assert OBS["child"]["outputs"]["resets_xform_stack"] is False
    assert OBS["reset_child"]["outputs"]["resets_xform_stack"] is True

    for name in ("parent", "child", "reset_child"):
        output = OBS[name]["outputs"]
        check(
            output["world_matrix_rows"],
            expected_world[name],
            f"OpenUSD native world stack {name}",
        )
        world_points = (
            np.column_stack((probes, np.ones(len(probes)))) @ expected_world[name]
        )[:, :3]
        world_directions = (
            np.column_stack((probes, np.zeros(len(probes)))) @ expected_world[name]
        )[:, :3]
        check(
            output["world_points_xyz"],
            world_points,
            f"OpenUSD native world point action {name}",
        )
        check(
            output["world_directions_xyz"],
            world_directions,
            f"OpenUSD native world direction action {name}",
        )

    wrong_order = expected_local["parent"] @ expected_local["child"]
    assert np.max(np.abs(wrong_order - expected_world["child"])) > 0.1
