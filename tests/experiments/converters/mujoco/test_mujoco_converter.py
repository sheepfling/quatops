"""Replay selected MuJoCo direct-quaternion boundary witnesses through QuatOps."""

from __future__ import annotations

import hashlib
import json

from tests.support.direct_quaternion_oracle import (
    canonical_cases_and_probe,
    check_composition,
    check_native_constructor,
    check_orientations,
)
from tests.support.paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
FIXTURE = ROOT / "tests/golden/oracles/mujoco/3.17.0/converter"
CONTRACT_PATH = ROOT / "experiments/converters/mujoco/3.17.0/semantic_contract.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
MANIFEST = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
INPUTS = json.loads(
    (ROOT / "experiments/converters/mujoco/3.17.0/converter_inputs.json").read_text(
        encoding="utf-8"
    )
)
WITNESSES = json.loads((FIXTURE / "witnesses.json").read_text(encoding="utf-8"))
SHARED_PATH = ROOT / "experiments/interoperability/v1/canonical_inputs.json"
SHARED = json.loads(SHARED_PATH.read_text(encoding="utf-8"))
OBS = WITNESSES["observations"]
CANONICAL, PROBE = canonical_cases_and_probe(SHARED)


def test_mujoco_boundary_contract_and_capture_provenance_are_explicit():
    assert (
        CONTRACT["contract_id"] == "quatops-mujoco-3.17.0-direct-quaternion-boundary-v1"
    )
    assert CONTRACT["canonical"]["frame"] == {
        "positive_x": "right",
        "positive_y": "forward",
        "positive_z": "up",
    }
    assert CONTRACT["target_boundary"]["source_to_target_coordinate_map"] == [
        "x",
        "y",
        "z",
    ]
    assert CONTRACT["target_boundary"]["probe_frame_assignment"]["meaning"].startswith(
        "explicit test frame assignment"
    )
    assert CONTRACT["target_boundary"]["canonical_to_native"].startswith(
        "identity on WXYZ"
    )
    assert CONTRACT["target"]["entry_points"] == [
        "mju_quat2Mat",
        "mju_rotVecQuat",
        "mju_mat2Quat",
        "mju_mulQuat",
        "mju_axisAngle2Quat",
    ]
    assert "robot link" in CONTRACT["excludes"][1]
    assert MANIFEST["execution"]["was_run"] is True
    assert MANIFEST["target"]["package_version"] == "3.17.0"
    assert MANIFEST["target"]["engine_version"] == "3.17.0"
    assert MANIFEST["target"]["native_scalar_dtype"] == "float64"
    assert MANIFEST["selection"] == {
        "retained_orientation_cases": 5,
        "native_created_orientations": 1,
        "noncommuting_product_orders": 2,
        "native_basis_images_per_orientation": 3,
        "native_mixed_probe_case_id": "mixed_a",
    }
    paths = {
        "contract_sha256": CONTRACT_PATH,
        "converter_input_spec_sha256": ROOT
        / "experiments/converters/mujoco/3.17.0/converter_inputs.json",
        "selected_inputs_sha256": SHARED_PATH,
        "witnesses_sha256": FIXTURE / "witnesses.json",
    }
    for field, path in paths.items():
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == MANIFEST["provenance"][field]
        ), field
    replay = MANIFEST["unified_lock_replay"]
    replay_paths = {
        "capture_script_sha256": ROOT
        / "experiments/converters/mujoco/3.17.0/capture_native.py",
        "shared_capture_script_sha256": ROOT / "scripts/direct_quaternion_capture.py",
        "lockfile_sha256": ROOT / "experiments/converters/python/uv.lock",
    }
    for field, path in replay_paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == replay[field], field
    assert (
        replay["native_binary_sha256"] == MANIFEST["native_binary_files"][0]["sha256"]
    )
    assert replay["witnesses_sha256"] == MANIFEST["provenance"]["witnesses_sha256"]
    for field in ("capture_script_sha256", "shared_capture_script_sha256"):
        capture_source = replay_paths[field].read_text(encoding="utf-8")
        assert "import quatops" not in capture_source
        assert "from quatops" not in capture_source
    assert MANIFEST["native_binary_files"][0]["path"] == "mujoco.dll"
    assert len(MANIFEST["native_binary_files"][0]["sha256"]) == 64


def test_mujoco_native_quaternion_action_preserves_canonical_orientation():
    check_orientations(
        OBS,
        CANONICAL,
        PROBE,
        basis_field="native_mju_rotVecQuat_basis_xyz",
        probe_field="native_mju_rotVecQuat_probe_xyz",
    )


def test_mujoco_native_constructor_converts_back_to_canonical():
    check_native_constructor(
        OBS,
        INPUTS,
        basis_field="native_mju_rotVecQuat_basis_xyz",
        quaternion_field="native_input_wxyz",
    )


def test_mujoco_noncommuting_composition_both_orders():
    check_composition(OBS, CANONICAL, basis_field="native_mju_rotVecQuat_basis_xyz")
