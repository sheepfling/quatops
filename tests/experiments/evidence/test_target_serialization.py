"""Target Pydantic models reduce captured packages into the common product."""

from __future__ import annotations

from types import MappingProxyType

import pytest
from pydantic import ValidationError

from scripts.converter_evidence import (
    converter_fixture_directories,
    load_converter_evidence,
)
from scripts.evidence_schema.converter_captures import EigenInputs, EigenOrientation
from scripts.evidence_schema.crystallized import (
    CrystallizedEvidence,
    EvidenceProduct,
    FrameChainContract,
    FrameChainWitness,
    GeographicContract,
    NativeQuaternionApiSerialization,
    QuaternionBoundaryContract,
    RotationVectorBoundaryContract,
    SceneTransportContract,
)
from scripts.evidence_schema.engine_evidence import ENGINE_VERSIONS, load_engine_package
from scripts.evidence_schema.product import (
    build_evidence_product,
    render_evidence_product,
)
from tests.support.paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
EXPECTED_WITNESS_COUNTS = MappingProxyType({
    ("Unreal Engine", "5.7.4"): 18,
    ("Unity Editor", "6000.5.2f1"): 9,
    ("Godot Engine", "4.7-stable"): 9,
    ("Godot Engine", "4.8-dev1"): 9,
    ("Blender", "5.1.2"): 9,
    ("Eigen", "5.0.1"): 20,
    ("MuJoCo", "3.17.0"): 10,
    ("OpenCV", "4.13.0"): 20,
    ("ROS 2 tf2", "0.36.23"): 1,
    ("SpiceyPy / CSPICE", "8.3.0"): 10,
    ("CesiumJS", "1.124.0"): 5,
    ("Three.js GLTFLoader", "0.187.0"): 2,
    ("OpenUSD", "26.11"): 3,
})


def test_target_specific_captures_reduce_to_reproducible_compact_product():
    product = build_evidence_product(ROOT)

    assert isinstance(product, EvidenceProduct)
    assert len(product.records) == len(EXPECTED_WITNESS_COUNTS)
    assert len(product.witness_sets) == 12
    assert len(product.canonical_inputs.orientations) == 8
    assert product.canonical_inputs.product_order == "lhs * rhs; rhs acts first"
    rendered = render_evidence_product(product)
    assert EvidenceProduct.model_validate_json(rendered, strict=True) == product

    records = {(row.target.product, row.target.version): row for row in product.records}
    witness_sets = {row.witness_set_id: row for row in product.witness_sets}
    assert set(records) == set(EXPECTED_WITNESS_COUNTS)

    source_capture_paths = {
        ROOT / "experiments/interoperability/v1/canonical_inputs.json"
    }
    for directory in converter_fixture_directories(ROOT):
        package = load_converter_evidence(directory, repository_root=ROOT)
        key = (package.target.name, package.target.version)
        record = records[key]
        assert isinstance(record, CrystallizedEvidence)
        assert (
            len(witness_sets[record.witness_set_id].witnesses)
            == (EXPECTED_WITNESS_COUNTS[key])
        )
        assert record.provenance.witnesses_sha256
        source_capture_paths.update(
            directory / name
            for name in ("contract.json", "manifest.json", "witnesses.json")
            if (directory / name).is_file()
        )
        if package.inputs_source_path is not None:
            source_capture_paths.add(package.inputs_source_path)

    for engine, version in ENGINE_VERSIONS:
        package = load_engine_package(ROOT, engine, version)
        key = (package.capture.target.engine, package.capture.target.version or version)
        record = records[key]
        assert (
            len(witness_sets[record.witness_set_id].witnesses)
            == (EXPECTED_WITNESS_COUNTS[key])
        )
        assert record.provenance.witnesses_sha256
        source_capture_paths.update((package.contract_path, package.witness_path))

    source_capture_bytes = sum(path.stat().st_size for path in source_capture_paths)

    assert len(rendered.encode("utf-8")) < source_capture_bytes

    assert all(not hasattr(record, "witnesses") for record in product.records)
    eigen_contract = records[("Eigen", "5.0.1")].contract
    assert isinstance(eigen_contract, QuaternionBoundaryContract)
    assert eigen_contract.native_order == "wxyz"
    assert eigen_contract.native_storage_order == "xyzw"
    assert eigen_contract.destination_axes_from_source == ["x", "y", "z"]

    assert isinstance(
        records[("OpenCV", "4.13.0")].contract,
        RotationVectorBoundaryContract,
    )
    ros_contract = records[("ROS 2 tf2", "0.36.23")].contract
    assert isinstance(ros_contract, FrameChainContract)
    assert [frame.frame for frame in ros_contract.frame_mappings] == [
        "world",
        "base",
        "sensor",
    ]
    assert isinstance(records[("CesiumJS", "1.124.0")].contract, GeographicContract)
    assert isinstance(records[("OpenUSD", "26.11")].contract, SceneTransportContract)


def test_concrete_serialization_rejects_mismatched_layer_and_witness_family():
    product = build_evidence_product(ROOT)
    records = {(row.target.product, row.target.version): row for row in product.records}
    witness_sets = {row.witness_set_id: row for row in product.witness_sets}
    eigen_record = records[("Eigen", "5.0.1")]
    eigen_witnesses = witness_sets[eigen_record.witness_set_id].witnesses
    orientation = next(row for row in eigen_witnesses if row.kind == "orientation")
    ros_record = records[("ROS 2 tf2", "0.36.23")]
    frame_chain = witness_sets[ros_record.witness_set_id].witnesses[0]

    assert isinstance(eigen_record.contract, QuaternionBoundaryContract)
    assert isinstance(frame_chain, FrameChainWitness)
    with pytest.raises(ValidationError):
        NativeQuaternionApiSerialization.model_validate(
            {
                "layer": "engine-quaternion",
                "contract": eigen_record.contract,
                "witnesses": [orientation],
            },
            strict=True,
        )

    with pytest.raises(ValidationError):
        NativeQuaternionApiSerialization.model_validate(
            {
                "layer": "native-quaternion-api",
                "contract": eigen_record.contract,
                "witnesses": [frame_chain],
            },
            strict=True,
        )


def test_eigen_storage_view_is_typed_and_matches_native_accessors():
    product = build_evidence_product(ROOT)
    record = next(row for row in product.records if row.target.product == "Eigen")
    witness_set = next(
        row
        for row in product.witness_sets
        if row.witness_set_id == record.witness_set_id
    )
    mixed = [
        witness
        for witness in witness_set.witnesses
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
