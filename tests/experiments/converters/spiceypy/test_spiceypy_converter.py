"""Replay selected CSPICE direct-quaternion boundary witnesses through QuatOps."""

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
from tests.support.provenance_archives import read_provenance_source

ROOT = REPOSITORY_ROOT
FIXTURE = ROOT / "tests/golden/oracles/spiceypy/8.3.0/converter"
CONTRACT_PATH = ROOT / "experiments/converters/spiceypy/8.3.0/semantic_contract.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
MANIFEST = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
INPUTS = json.loads(
    (ROOT / "experiments/converters/spiceypy/8.3.0/converter_inputs.json").read_text(
        encoding="utf-8"
    )
)
WITNESSES = json.loads((FIXTURE / "witnesses.json").read_text(encoding="utf-8"))
SHARED_PATH = ROOT / "experiments/interoperability/v1/canonical_inputs.json"
SHARED = json.loads(SHARED_PATH.read_text(encoding="utf-8"))
OBS = WITNESSES["observations"]
CANONICAL, PROBE = canonical_cases_and_probe(SHARED)


def test_spiceypy_boundary_contract_and_capture_provenance_are_explicit():
    assert (
        CONTRACT["contract_id"]
        == "quatops-spiceypy-8.3.0-direct-quaternion-boundary-v1"
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
    assert CONTRACT["target"]["entry_points"] == ["q2m", "mxv", "m2q", "qxq", "axisar"]
    assert "eul2m" in CONTRACT["excludes"][0]
    assert MANIFEST["execution"]["was_run"] is True
    assert MANIFEST["target"]["wrapper_version"] == "8.3.0"
    assert MANIFEST["target"]["toolkit_version"] == "CSPICE_N0067"
    assert MANIFEST["selection"] == {
        "retained_orientation_cases": 5,
        "native_created_orientations": 1,
        "noncommuting_product_orders": 2,
        "native_basis_images_per_orientation": 3,
        "native_mixed_probe_case_id": "mixed_a",
    }
    paths = {
        "reproduction_capture_script_sha256": ROOT
        / "experiments/converters/spiceypy/8.3.0/capture_native.py",
        "contract_sha256": CONTRACT_PATH,
        "converter_input_spec_sha256": ROOT
        / "experiments/converters/spiceypy/8.3.0/converter_inputs.json",
        "selected_inputs_sha256": SHARED_PATH,
        "witnesses_sha256": FIXTURE / "witnesses.json",
        "reproduction_lockfile_sha256": ROOT / "experiments/converters/python/uv.lock",
    }
    for field, path in paths.items():
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == MANIFEST["provenance"][field]
        ), field
    replay = MANIFEST["compact_lock_replay"]
    assert (
        replay["lockfile_sha256"]
        == MANIFEST["provenance"]["pre_shared_lockfile_sha256"]
    )
    assert (
        replay["native_binary_sha256"]
        == MANIFEST["provenance"]["loaded_cspice_binary_sha256"]
    )
    assert replay["witnesses_sha256"] == MANIFEST["provenance"]["witnesses_sha256"]
    shared = MANIFEST["shared_lock_replay"]
    assert (
        shared["lockfile_sha256"]
        == MANIFEST["provenance"]["pre_unified_lockfile_sha256"]
    )
    assert (
        shared["native_binary_sha256"]
        == MANIFEST["provenance"]["loaded_cspice_binary_sha256"]
    )
    assert shared["witnesses_sha256"] == MANIFEST["provenance"]["witnesses_sha256"]
    assert (
        hashlib.sha256(
            read_provenance_source(
                ROOT, MANIFEST["provenance"]["capture_script_archive"]
            )
        ).hexdigest()
        == MANIFEST["provenance"]["capture_script_sha256"]
    )
    assert (
        MANIFEST["unified_lock_replay"]["lockfile_sha256"]
        == MANIFEST["provenance"]["reproduction_lockfile_sha256"]
    )
    assert (
        MANIFEST["unified_lock_replay"]["witnesses_sha256"]
        == MANIFEST["provenance"]["witnesses_sha256"]
    )
    capture_source = paths["reproduction_capture_script_sha256"].read_text(
        encoding="utf-8"
    )
    assert (
        "import quatops" not in capture_source and "from quatops" not in capture_source
    )
    assert len(MANIFEST["provenance"]["loaded_cspice_binary_sha256"]) == 64
    assert (
        MANIFEST["provenance"]["loaded_cspice_binary_path"]
        .lower()
        .endswith("cspice.dll")
    )


def test_spiceypy_native_quaternion_action_preserves_canonical_orientation():
    check_orientations(
        OBS,
        CANONICAL,
        PROBE,
        basis_field="native_mxv_basis_xyz",
        probe_field="native_mxv_probe_xyz",
    )


def test_spiceypy_native_constructor_converts_back_to_canonical():
    check_native_constructor(
        OBS,
        INPUTS,
        basis_field="native_mxv_basis_xyz",
        quaternion_field="native_m2q_wxyz",
    )


def test_spiceypy_noncommuting_composition_both_orders():
    check_composition(OBS, CANONICAL, basis_field="native_mxv_basis_xyz")
