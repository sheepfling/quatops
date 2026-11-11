"""Target Pydantic models normalize independently captured native witnesses."""

from __future__ import annotations

import shutil
from types import MappingProxyType

import pytest
from pydantic import ValidationError

import experiments.converters.cesium.inputs as cesium_inputs
import experiments.converters.gltf.inputs as gltf_inputs
import experiments.converters.openusd.inputs as openusd_inputs
from scripts.converter_evidence import (
    converter_fixture_directories,
    load_converter_evidence,
)
from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.converter_captures import (
    EigenCapture,
    EigenInputs,
    EigenOrientation,
)
from scripts.evidence_schema.crystallized import (
    FrameChainContract,
    FrameChainWitness,
    GeographicContract,
    NativeQuaternionApiSerialization,
    QuaternionBoundaryContract,
    RotationVectorBoundaryContract,
    SceneTransportContract,
)
from scripts.evidence_schema.engine_captures import EngineSemanticConfig
from scripts.evidence_schema.engine_evidence import load_engine_package
from scripts.evidence_schema.normalizer import normalize_capture
from scripts.evidence_schema.semantic_serializers.eigen import serialize_eigen_witnesses
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
        ("Eigen", "5.0.1"): 16,
        ("MuJoCo", "3.17.0"): 8,
        ("OpenCV", "4.13.0"): 16,
        ("ROS 2 tf2", "0.36.23"): 1,
        ("SpiceyPy / CSPICE", "8.3.0"): 8,
        ("CesiumJS", "1.124.0"): 5,
        ("Three.js GLTFLoader", "0.187.0"): 2,
        ("OpenUSD", "26.11"): 3,
    }
)


def test_converter_inputs_are_loaded_from_the_contract_package():
    for directory in converter_fixture_directories(ROOT):
        package = load_converter_evidence(directory, repository_root=ROOT)
        if (
            package.target.name in {"CesiumJS", "Three.js GLTFLoader", "OpenUSD"}
            and package.inputs_source_path is None
        ):
            assert package.inputs is not None
            continue
        if package.inputs is not None:
            assert package.inputs_source_path == (
                package.contract_source_path.parent / "converter_inputs.json"
            )


def test_generated_cesium_inputs_must_match_the_captured_digest(monkeypatch):
    monkeypatch.setattr(cesium_inputs, "serialize_inputs", lambda _: b"{}")
    with pytest.raises(ValueError, match="CesiumJS converter input digest differs"):
        load_converter_evidence(
            ROOT / "tests/golden/transports/cesium/1.124.0", repository_root=ROOT
        )


def test_generated_gltf_asset_must_match_the_captured_digest(monkeypatch):
    monkeypatch.setattr(gltf_inputs, "serialize_artifact", lambda _: b"{}")
    with pytest.raises(ValueError, match="generated glTF asset digest differs"):
        load_converter_evidence(
            ROOT / "tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter",
            repository_root=ROOT,
        )


def test_generated_openusd_stage_must_match_the_captured_digest(monkeypatch):
    original = openusd_inputs.build_artifacts

    def changed_stage(root):
        inputs, stage = original(root)
        return inputs, stage + "# altered\n"

    monkeypatch.setattr(openusd_inputs, "build_artifacts", changed_stage)
    with pytest.raises(ValueError, match="generated OpenUSD stage digest differs"):
        load_converter_evidence(
            ROOT / "tests/golden/transports/openusd/26.11/converter",
            repository_root=ROOT,
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


def test_godot_shared_contract_names_only_captured_versions():
    stable = load_engine_package(ROOT, "godot", "4.7-stable")
    dev = load_engine_package(ROOT, "godot", "4.8-dev1")
    assert stable.semantic_config_path == dev.semantic_config_path
    config = EngineSemanticConfig.model_validate(
        read_evidence_document(stable.semantic_config_path), strict=True
    )
    assert config.engine.supports_version("4.7-stable")
    assert config.engine.supports_version("4.8-dev1")
    assert not config.engine.supports_version("4.7.1-rc1")


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


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("coefficients", [None, [0.0, 0.0, 0.0, 1.0]])
def test_eigen_rejects_missing_or_incorrect_native_storage(precision, coefficients):
    directory = ROOT / "tests/golden/oracles/eigen/5.0.1/converter"
    package = load_converter_evidence(directory, repository_root=ROOT)
    assert isinstance(package.inputs, EigenInputs)
    document = read_evidence_document(directory / "witnesses.json")
    profiles = document["observations"]["profiles"]
    changed_profiles = [
        {
            **profile,
            "orientations": [
                {
                    **row,
                    "quaternion": {**row["quaternion"], "coeffs_xyzw": coefficients},
                }
                if profile["dtype"] == precision and row["case_id"] == "mixed_a"
                else row
                for row in profile["orientations"]
            ],
        }
        for profile in profiles
    ]
    capture = EigenCapture.model_validate(
        {**document, "observations": {"profiles": changed_profiles}}, strict=True
    )
    with pytest.raises(ValueError, match="coeffs\\(\\)"):
        serialize_eigen_witnesses(capture, package.inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param(
            "native_quaternion_basis_actions_xyz",
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            id="missing-basis-vector",
        ),
        pytest.param("unmodeled_native_field", True, id="unknown-native-field"),
    ],
)
def test_eigen_orientation_rejects_invalid_native_fields(field, value):
    directory = ROOT / "tests/golden/oracles/eigen/5.0.1/converter"
    document = read_evidence_document(directory / "witnesses.json")
    valid = document["observations"]["profiles"][0]["orientations"][0]
    data = {**valid, field: value}
    with pytest.raises(ValidationError) as error:
        EigenOrientation.model_validate(data, strict=True)
    assert error.value.errors()[0]["loc"][0] == field


@pytest.mark.parametrize("field", ["scalar_profiles", "composition"])
def test_target_input_schema_requires_eigen_specific_fields(field):
    path = ROOT / "experiments/converters/eigen/5.0.1/converter_inputs.json"
    data = read_evidence_document(path)
    with pytest.raises(ValidationError) as error:
        EigenInputs.model_validate(
            {key: value for key, value in data.items() if key != field}, strict=True
        )
    assert error.value.errors()[0]["loc"] == (field,)
