"""QuatOps USDA serialization and pinned OpenUSD stage evaluation proof."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from quatops.experimental.engines import QuaternionFrameMap, SignedAxisMap
from tests.support.evidence_oracles import matrix_from_wxyz as rotation
from tests.support.paths import REPOSITORY_ROOT
from tests.support.provenance_archives import read_provenance_source

ROOT = REPOSITORY_ROOT
FIXTURE = ROOT / "tests/golden/transports/openusd/26.11/converter"
INPUTS_PATH = ROOT / "experiments/converters/openusd/26.11/converter_inputs.json"
STAGE_PATH = ROOT / "experiments/converters/openusd/26.11/quatops_hierarchy.usda"
CONTRACT_PATH = ROOT / "experiments/converters/openusd/26.11/semantic_contract.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
MANIFEST = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
INPUTS = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))
STAGE = STAGE_PATH.read_text(encoding="utf-8")
CAPTURE = json.loads((FIXTURE / "witnesses.json").read_text(encoding="utf-8"))
OBS = {row["node"]: row for row in CAPTURE["observations"]}
FRAME = SignedAxisMap(("x", "z", "-y"))
TO_USD = QuaternionFrameMap(FRAME, FRAME)
MAP = np.asarray(FRAME.matrix, dtype=float)
ATOL = 3e-12


def check(actual, expected, message):
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=ATOL, err_msg=message)


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
    assert CAPTURE["ordered_xform_ops"] == [
        "xformOp:translate",
        "xformOp:orient",
    ]
    assert 'upAxis = "Y"' in STAGE
    assert '"!resetXformStack!"' in STAGE

    provenance = MANIFEST["provenance"]
    paths = {
        "stage_sha256": STAGE_PATH,
        "converter_input_sha256": INPUTS_PATH,
        "native_observations_sha256": FIXTURE / "witnesses.json",
        "contract_sha256": CONTRACT_PATH,
    }
    for field, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance[field]
    assert (
        hashlib.sha256(
            read_provenance_source(ROOT, MANIFEST["reproduction"]["generator_archive"])
        ).hexdigest()
        == provenance["generator_sha256"]
    )
    assert (
        provenance["captured_converter_input_sha256"]
        == provenance["converter_input_sha256"]
    )
    assert (
        MANIFEST["reproduction"]["lockfile_sha256"]
        == hashlib.sha256(
            (ROOT / "experiments/converters/openusd/26.11/uv.lock").read_bytes()
        ).hexdigest()
    )
    reproduction = MANIFEST["reproduction"]
    assert (
        hashlib.sha256(
            (ROOT / "experiments/converters/openusd/26.11/create_stage.py").read_bytes()
        ).hexdigest()
        == reproduction["generator_sha256"]
    )
    for field, path in {
        "capture_script_sha256": ROOT
        / "experiments/converters/openusd/26.11/capture_native.py",
        "retention_script_sha256": ROOT
        / "scripts/retain_openusd_converter_witnesses.py",
        "retention_helper_sha256": ROOT / "scripts/evidence_json.py",
        "retention_formatter_sha256": ROOT / "scripts/evidence_format.py",
    }.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reproduction[field]
    assert reproduction["capture_script_identity_note"].startswith(
        "The current capture entry point adds the semantic contract to the output; "
        "provenance.native_capture_script_sha256 remains the producer identity"
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
        recovered = TO_USD.from_destination_wxyz(output["orient_op_wxyz"])
        check(
            recovered,
            item["canonical_wxyz"],
            f"OpenUSD inverse quaternion conversion {name}",
        )

    expected_world = {
        "parent": expected_local["parent"],
        "child": expected_local["child"] @ expected_local["parent"],
        "reset_child": expected_local["reset_child"],
    }
    assert OBS["child"]["outputs"]["resets_xform_stack"] is False
    assert OBS["reset_child"]["outputs"]["resets_xform_stack"] is True

    for name in ("parent", "child", "reset_child"):
        output = OBS[name]["outputs"]
        world_directions = probes[:3] @ expected_world[name][:3, :3]
        check(
            output["world_basis_images_xyz"],
            world_directions,
            f"OpenUSD native world basis action {name}",
        )
        probe = probes[-1]
        world_probe = np.append(probe, 1.0) @ expected_world[name]
        check(
            output["world_probe_point_xyz"],
            world_probe[:3],
            f"OpenUSD world point {name}",
        )

    check(
        OBS["child"]["outputs"]["roundtrip_probe_point_xyz"],
        probes[-1],
        "OpenUSD world/local inverse point round trip",
    )

    wrong_order = expected_local["parent"] @ expected_local["child"]
    assert np.max(np.abs(wrong_order - expected_world["child"])) > 0.1
