"""Validate the compact, versioned engine converter witness set."""

import hashlib
import json

import pytest

from tests.support.native_engine_support import load_versioned_engine_witness
from tests.support.paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
TARGETS = (
    ("unreal", "5.7.4"),
    ("unity", "6000.5.2f1"),
    ("godot", "4.7-stable"),
    ("godot", "4.8-dev1"),
    ("blender", "5.1.2"),
)


def _load(engine, version):
    return load_versioned_engine_witness(ROOT, engine, version)


@pytest.mark.parametrize("engine,version", TARGETS)
def test_contract_and_witnesses_are_versioned_small_and_provenanced(engine, version):
    contract, witnesses, witness_path = _load(engine, version)
    assert contract["engine"]["version"] == version
    assert contract["engine"]["api"]
    assert contract["engine"]["native_quaternion_order"] in {"wxyz", "xyzw"}
    assert contract["engine"]["native_algebra"] == "Hamilton"
    conversion = contract["conversion"]
    assert conversion["contract_id"] == "quatops-canonical-hamilton-wxyz-active-rh-v1"
    assert conversion["canonical_frame"]["handedness"] == "right-handed"
    assert conversion["destination_axes_from_source"]
    assert conversion["conversion_rule"]
    assert conversion["inverse_rule"]
    assert conversion["world_and_body_map"] == "same declared signed-axis map"
    assert conversion["input_domain"].startswith("unit Hamilton quaternion")
    assert witnesses["provenance"]["native_build_record"]["version"].startswith(version)
    assert witness_path.stat().st_size < 50 * 1024

    expected_hashes = {
        "witness_sha256": hashlib.sha256(witness_path.read_bytes()).hexdigest(),
        "capture_source_sha256": hashlib.sha256(
            (ROOT / contract["provenance"]["capture_source"]).read_bytes()
        ).hexdigest(),
        "canonical_input_sha256": hashlib.sha256(
            (ROOT / witnesses["provenance"]["canonical_input_file"]).read_bytes()
        ).hexdigest(),
    }
    for key, digest in expected_hashes.items():
        assert contract["provenance"][key] == digest


def test_witnesses_keep_only_discriminating_inputs_and_native_actions():
    canonical_inputs = json.loads(
        (ROOT / "experiments/interoperability/v1/canonical_inputs.json").read_text(
            encoding="utf-8"
        )
    )
    expected_inputs = {
        row["case_id"]: row["canonical_wxyz"]
        for row in canonical_inputs["orientations"]
    }
    for engine, version in TARGETS:
        _, witnesses, _ = _load(engine, version)
        capture = witnesses["canonical_interop"]
        captures = capture.get("precision_captures", {"float32": capture})
        for profile in captures.values():
            cases = {row["case_id"]: row for row in profile["orientation_cases"]}
            assert set(cases) == set(expected_inputs)
            for case_id, expected in expected_inputs.items():
                row = cases[case_id]
                assert row["canonical_wxyz"] == expected
                assert len(row["native_quaternion"]) == 4
                assert len(row["native_rotated_positive_basis_vectors_xyz"]) == 3
                assert len(row["native_rotated_probe_vector_xyz"]) == 3
                assert set(row) == {
                    "case_id",
                    "canonical_wxyz",
                    "native_input_wxyz",
                    "native_quaternion",
                    "native_rotated_positive_basis_vectors_xyz",
                    "native_rotated_probe_vector_xyz",
                }
            for product_key in ("composition", "reverse_composition"):
                product = profile[product_key]
                assert len(product["native_product_quaternion"]) == 4
                assert len(product["native_rotated_positive_basis_vectors_xyz"]) == 3
