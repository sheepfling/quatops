"""Target Pydantic models normalize independently captured native witnesses."""

from __future__ import annotations

import shutil
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from scripts.converter_evidence import (
    load_converter_evidence,
)
from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.converter_captures import EigenInputs, EigenOrientation
from scripts.evidence_schema.crystallized import (
    FrameChainContract,
    FrameChainWitness,
    GeographicContract,
    NativeQuaternionApiSerialization,
    QuaternionBoundaryContract,
    RotationVectorBoundaryContract,
    SceneTransportContract,
)
from scripts.evidence_schema.engine_evidence import load_engine_package
from scripts.evidence_schema.normalizer import normalize_capture
from scripts.evidence_schema.shared_inputs import (
    SharedCanonicalInputCapture,
    crystallize_shared_inputs,
)
from scripts.evidence_schema.validation import (
    validate_evidence_packages,
    verify_capture_provenance,
)
from tests.support.paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
EXPECTED_WITNESS_COUNTS = MappingProxyType(
    {
        ("Unreal Engine", "5.7.4"): 14,
        ("Unity Editor", "6000.5.2f1"): 7,
        ("Godot Engine", "4.7-stable"): 7,
        ("Godot Engine", "4.8-dev1"): 7,
        ("Blender", "5.1.2"): 7,
        ("Eigen", "5.0.1"): 20,
        ("MuJoCo", "3.17.0"): 10,
        ("OpenCV", "4.13.0"): 20,
        ("ROS 2 tf2", "0.36.23"): 1,
        ("SpiceyPy / CSPICE", "8.3.0"): 10,
        ("CesiumJS", "1.124.0"): 5,
        ("Three.js GLTFLoader", "0.187.0"): 2,
        ("OpenUSD", "26.11"): 3,
    }
)


def test_converter_rejects_changed_shared_contract(tmp_path):
    fixture = "tests/golden/oracles/mujoco/3.17.0/converter"
    contract = "experiments/converters/mujoco/3.17.0/semantic_contract.json"
    shutil.copytree(ROOT / fixture, tmp_path / fixture)
    (tmp_path / contract).parent.mkdir(parents=True)
    shutil.copy2(ROOT / contract, tmp_path / contract)
    package = load_converter_evidence(tmp_path / fixture, repository_root=tmp_path)
    assert package.contract_source_path == (tmp_path / contract).resolve()
    with (tmp_path / contract).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="semantic contract digest differs"):
        load_converter_evidence(tmp_path / fixture, repository_root=tmp_path)


def test_engine_rejects_changed_shared_contract(tmp_path):
    fixture = "tests/oracles/unity/6000.5.2f1"
    contract = "experiments/unity/6000.5.2f1/semantic_contract.json"
    shutil.copytree(ROOT / fixture, tmp_path / fixture)
    (tmp_path / contract).parent.mkdir(parents=True)
    shutil.copy2(ROOT / contract, tmp_path / contract)
    package = load_engine_package(tmp_path, "unity", "6000.5.2f1")
    assert package.semantic_config_path == tmp_path / contract
    with (tmp_path / contract).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="semantic config digest differs"):
        load_engine_package(tmp_path, "unity", "6000.5.2f1")


def test_converter_rejects_modified_native_witness(tmp_path):
    fixture = "tests/golden/oracles/mujoco/3.17.0/converter"
    experiment = "experiments/converters/mujoco/3.17.0"
    shutil.copytree(ROOT / fixture, tmp_path / fixture)
    (tmp_path / experiment).mkdir(parents=True)
    for name in ("semantic_contract.json", "capture_native.py"):
        shutil.copy2(ROOT / experiment / name, tmp_path / experiment / name)
    package = load_converter_evidence(tmp_path / fixture, repository_root=tmp_path)
    verify_capture_provenance(package, tmp_path / fixture)
    with (tmp_path / fixture / "witnesses.json").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="native witness digest differs"):
        verify_capture_provenance(package, tmp_path / fixture)


def test_target_specific_captures_normalize_to_shared_witnesses():
    captures = validate_evidence_packages(ROOT)
    records = {(row.target, row.version): row for row in captures}
    assert set(records) == set(EXPECTED_WITNESS_COUNTS)
    for key, count in EXPECTED_WITNESS_COUNTS.items():
        assert len(records[key].serialization.witnesses) == count
    first_godot = records[("Godot Engine", "4.7-stable")]
    second_godot = records[("Godot Engine", "4.8-dev1")]
    assert first_godot.witness_path == second_godot.witness_path
    assert first_godot.serialization.witnesses == second_godot.serialization.witnesses

    eigen_contract = records[("Eigen", "5.0.1")].serialization.contract
    assert isinstance(eigen_contract, QuaternionBoundaryContract)
    assert eigen_contract.native_order == "wxyz"
    assert eigen_contract.native_storage_order == "xyzw"
    assert eigen_contract.destination_axes_from_source == ["x", "y", "z"]
    assert isinstance(
        records[("OpenCV", "4.13.0")].serialization.contract,
        RotationVectorBoundaryContract,
    )
    ros_contract = records[("ROS 2 tf2", "0.36.23")].serialization.contract
    assert isinstance(ros_contract, FrameChainContract)
    assert [frame.frame for frame in ros_contract.frame_mappings] == [
        "world",
        "base",
        "sensor",
    ]
    assert isinstance(
        records[("CesiumJS", "1.124.0")].serialization.contract, GeographicContract
    )
    assert isinstance(
        records[("OpenUSD", "26.11")].serialization.contract, SceneTransportContract
    )


def test_concrete_serialization_rejects_mismatched_layer_and_witness_family():
    records = {
        (row.target, row.version): row for row in validate_evidence_packages(ROOT)
    }
    eigen = records[("Eigen", "5.0.1")].serialization
    orientation = next(row for row in eigen.witnesses if row.kind == "orientation")
    frame_chain = records[("ROS 2 tf2", "0.36.23")].serialization.witnesses[0]
    assert isinstance(eigen.contract, QuaternionBoundaryContract)
    assert isinstance(frame_chain, FrameChainWitness)
    inputs = crystallize_shared_inputs(
        SharedCanonicalInputCapture.model_validate(
            read_evidence_document(
                ROOT / "experiments/interoperability/v1/canonical_inputs.json"
            ),
            strict=True,
        )
    )
    with pytest.raises(ValueError, match="precision is absent"):
        normalize_capture(eigen, inputs, native_precision=[])
    with pytest.raises(ValidationError):
        NativeQuaternionApiSerialization.model_validate(
            {
                "layer": "engine-quaternion",
                "contract": eigen.contract,
                "witnesses": [orientation],
            },
            strict=True,
        )
    with pytest.raises(ValidationError):
        NativeQuaternionApiSerialization.model_validate(
            {
                "layer": "native-quaternion-api",
                "contract": eigen.contract,
                "witnesses": [frame_chain],
            },
            strict=True,
        )


def test_eigen_storage_view_is_typed_and_matches_native_accessors():
    capture = next(
        row for row in validate_evidence_packages(ROOT) if row.target == "Eigen"
    )
    mixed = [
        witness
        for witness in capture.serialization.witnesses
        if witness.kind == "orientation" and witness.case_id == "mixed_a"
    ]
    assert len(mixed) == 2
    for witness in mixed:
        assert witness.coefficient_order == "wxyz"
        assert witness.storage_coefficients_xyzw == [
            witness.coefficients[1],
            witness.coefficients[2],
            witness.coefficients[3],
            witness.coefficients[0],
        ]


def test_target_serializers_reject_malformed_native_vectors():
    with pytest.raises(ValidationError):
        EigenOrientation.model_validate(
            {
                "case_id": "identity",
                "quaternion": {
                    "wxyz": [1.0, 0.0, 0.0, 0.0],
                    "coeffs_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "rotation_matrix": {
                    "rows": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "column_major_storage": [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                },
                "native_quaternion_basis_actions_xyz": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                "matrix_to_quaternion": {"wxyz": [1.0, 0.0, 0.0, 0.0]},
            },
            strict=True,
        )


def test_target_input_schema_requires_eigen_specific_fields():
    with pytest.raises(ValidationError):
        EigenInputs.model_validate(
            {
                "schema_version": "1.0",
                "contract_id": "fixture-contract",
                "native_constructed_orientation": {
                    "axis_xyz": [1.0, 0.0, 0.0],
                    "angle_radians": 1.0,
                },
            },
            strict=True,
        )


def test_target_serializers_reject_unknown_native_fields():
    identity_matrix = {
        "rows": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "column_major_storage": [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    }
    valid_identity = {
        "case_id": "identity",
        "quaternion": {"wxyz": [1.0, 0.0, 0.0, 0.0]},
        "rotation_matrix": identity_matrix,
        "native_quaternion_basis_actions_xyz": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "matrix_to_quaternion": {"wxyz": [1.0, 0.0, 0.0, 0.0]},
    }
    with pytest.raises(ValidationError):
        EigenOrientation.model_validate(
            {**valid_identity, "unmodeled_native_field": True},
            strict=True,
        )
