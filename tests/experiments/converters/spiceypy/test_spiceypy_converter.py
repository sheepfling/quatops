"""Replay selected CSPICE direct-quaternion boundary witnesses through QuatOps."""

from __future__ import annotations

import hashlib
import json

from tests.support.direct_quaternion_oracle import (
    canonical_cases_and_probe,
    check_composition,
    check_contract_and_provenance,
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
    check_contract_and_provenance(
        CONTRACT, MANIFEST, source=CONTRACT_PATH.parent, fixture=FIXTURE
    )
    assert CONTRACT["target"]["entry_points"] == ["q2m", "mxv", "m2q", "qxq", "axisar"]
    assert "eul2m" in CONTRACT["excludes"][0]
    assert "round trips" in CONTRACT["excludes"][-1]
    assert MANIFEST["target"]["wrapper_version"] == "8.3.0"
    assert MANIFEST["target"]["toolkit_version"] == "CSPICE_N0067"
    replay = MANIFEST["unified_lock_replay"]
    assert (
        replay["native_binary_sha256"]
        == MANIFEST["provenance"]["loaded_cspice_binary_sha256"]
    )
    assert (
        hashlib.sha256(
            read_provenance_source(
                ROOT, MANIFEST["provenance"]["capture_script_archive"]
            )
        ).hexdigest()
        == MANIFEST["provenance"]["capture_script_sha256"]
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
