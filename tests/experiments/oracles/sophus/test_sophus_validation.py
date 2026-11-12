"""Regression coverage for the pinned Sophus capture contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import struct
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.evidence_format import dump_evidence_json
from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.primitives import finite_max_abs
from scripts.evidence_schema.sophus import (
    SophusBindingManifest,
    SophusDiscoveryWitnessSet,
    SophusSemanticConfig,
)
from tests.support.paths import REPOSITORY_ROOT
from tests.support.provenance_archives import read_provenance_source

ROOT = REPOSITORY_ROOT
SOPHUS = ROOT / "experiments/oracles/sophus/1.24.6"
spec = importlib.util.spec_from_file_location(
    "sophus_retain_witnesses", SOPHUS / "retain_witnesses.py"
)
assert spec is not None and spec.loader is not None
retain = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = retain
spec.loader.exec_module(retain)


@pytest.mark.parametrize(
    "package",
    sorted(path.parent for path in SOPHUS.parent.glob("*/manifest.json")),
    ids=lambda package: package.name,
)
def test_recorded_versions_load_each_recorded_profile(package: Path) -> None:
    config = retain.load_semantic_config(package)
    for profile in config.scalar_profiles:
        witness = retain.load_shared_witness(profile.precision, package)
        assert witness.scalar_profile == profile.precision


def _witness_path(precision: str):
    manifest = json.loads((SOPHUS / "manifest.json").read_text(encoding="utf-8"))
    return (
        ROOT
        / manifest["artifacts"]["scalar_profiles"][precision]["selected_witness_file"]
    )


def _capture() -> dict:
    witness = json.loads(_witness_path("float64").read_text(encoding="utf-8"))
    fields = (
        "probe_xyz",
        "quaternion_construction",
        "composition",
    )
    return {
        "capture_schema": "quatops-sophus-direct-v2",
        "target": "Sophus SO3d",
        **{field: witness[field] for field in fields},
    }


def test_direct_capture_matches_pinned_tsv() -> None:
    capture = retain.NativeCapture.model_validate_json(
        json.dumps(_capture()), strict=True
    )
    capture.validate_pinned_inputs(
        SOPHUS / "capture_inputs.tsv",
        retain.load_recipe_manifest().artifacts.input_sha256,
    )
    selected = retain.reduce_capture(
        json.dumps(_capture()).encode(), retain.load_semantic_config()
    )
    assert tuple(row[0] for row in selected.composition) == (
        "mixed_unequal_times_nonunit",
        "nonunit_times_mixed_unequal",
    )
    frozen = json.loads(_witness_path("float64").read_text(encoding="utf-8"))
    actual = selected.model_dump(mode="json")
    for field in frozen:
        assert actual[field] == frozen[field]


def test_float32_capture_inputs_are_compared_at_exporter_precision() -> None:
    raw = _capture()
    raw["target"] = "Sophus SO3f"

    def single(value):
        return struct.unpack("=f", struct.pack("=f", value))[0]

    for row in raw["quaternion_construction"]:
        row[1] = [single(value) for value in row[1]]
    capture = retain.NativeCapture.model_validate_json(json.dumps(raw), strict=True)
    capture.validate_pinned_inputs(
        SOPHUS / "capture_inputs.tsv",
        retain.load_recipe_manifest().artifacts.input_sha256,
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing_constructor",
        "duplicate_constructor",
        "missing_composition",
        "nonfinite",
        "antipode",
        "missing_probe",
    ],
)
def test_direct_capture_rejects_invalid_witnesses(fault: str) -> None:
    raw = _capture()
    if fault == "missing_constructor":
        raw["quaternion_construction"].pop()
    elif fault == "duplicate_constructor":
        raw["quaternion_construction"][-1] = raw["quaternion_construction"][0]
    elif fault == "missing_composition":
        raw["composition"].pop()
    elif fault == "nonfinite":
        raw["probe_xyz"][0] = float("nan")
    elif fault == "antipode":
        raw["quaternion_construction"][5][3] = raw["quaternion_construction"][4][3]
    else:
        raw["quaternion_construction"][4][-1] = None
    with pytest.raises(ValidationError):
        retain.NativeCapture.model_validate_json(json.dumps(raw), strict=True)


def test_direct_capture_rejects_changed_inputs() -> None:
    raw = _capture()
    raw["quaternion_construction"][0][1] = [0.0, 0.0, 0.0, 1.0]
    capture = retain.NativeCapture.model_validate_json(json.dumps(raw), strict=True)
    with pytest.raises(ValueError, match="pinned TSV"):
        capture.validate_pinned_inputs(
            SOPHUS / "capture_inputs.tsv",
            retain.load_recipe_manifest().artifacts.input_sha256,
        )


def test_direct_capture_rejects_changed_input_zero_sign() -> None:
    raw = _capture()
    raw["quaternion_construction"][0][1][1] = -0.0
    capture = retain.parse_native_capture(json.dumps(raw).encode())
    with pytest.raises(ValueError, match="pinned TSV"):
        capture.validate_pinned_inputs(
            SOPHUS / "capture_inputs.tsv",
            retain.load_recipe_manifest().artifacts.input_sha256,
        )


def test_exact_capture_comparison_preserves_integer_spelled_negative_zero() -> None:
    raw = _capture()
    raw["quaternion_construction"][0][4][0][1] = -0.0
    raw_bytes = json.dumps(raw).encode().replace(b"-0.0", b"-0", 1)
    parsed = retain.parse_native_capture(raw_bytes)
    assert math.copysign(1.0, parsed.quaternion_construction[0][4][0][1]) == -1.0
    selected = retain.reduce_capture(raw_bytes, retain.load_semantic_config())
    selected_document = selected.model_dump(mode="json")
    witness_bytes = dump_evidence_json(selected_document).encode()
    retain.verify_exact_capture(raw_bytes, witness_bytes, retain.load_semantic_config())
    selected_document["quaternion_construction"][0][4][0][1] = 0.0
    with pytest.raises(ValueError, match="differs from the shared witness"):
        retain.verify_exact_capture(
            raw_bytes,
            dump_evidence_json(selected_document).encode(),
            retain.load_semantic_config(),
        )

    positive_raw = json.dumps(_capture()).encode()
    positive = retain.reduce_capture(positive_raw, retain.load_semantic_config())
    negative_witness = positive.model_dump(mode="json")
    negative_witness["quaternion_construction"][0][4][0][1] = -0.0
    negative_bytes = (
        dump_evidence_json(negative_witness).encode().replace(b"-0.0", b"-0", 1)
    )
    with pytest.raises(ValueError, match="differs from the shared witness"):
        retain.verify_exact_capture(
            positive_raw, negative_bytes, retain.load_semantic_config()
        )


def test_exact_compare_cli_does_not_claim_native_execution(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = tmp_path / "synthetic-capture.json"
    raw.write_text(json.dumps(_capture()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retain_witnesses.py",
            "--capture",
            str(raw),
            "--compare",
            str(_witness_path("float64")),
        ],
    )
    retain.main()
    report = json.loads(capsys.readouterr().out)
    assert report["result"] == "exact"
    assert report["declared_product_version"] == "1.24.6"
    assert report["execution_verified"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("execution", "was_run"), False),
        (("execution", "was_run"), 1),
        (("artifacts", "direct_capture", "exit_code"), False),
        (("artifacts", "direct_capture"), None),
        (("artifacts", "direct_capture", "product_version"), "9.9.9"),
        (("artifacts", "direct_capture", "dependency"), "Eigen 9.9.9"),
        (("artifacts", "scalar_profiles", "float64", "raw_capture_sha256"), "0" * 64),
        (
            ("artifacts", "scalar_profiles", "float64", "selected_witness_sha256"),
            "0" * 64,
        ),
        (("artifacts", "scalar_profiles", "float64", "comparison"), None),
        (
            ("artifacts", "scalar_profiles", "float64", "selected_witness_file"),
            "../witness.json",
        ),
    ],
)
def test_shared_witness_binding_rejects_incomplete_or_mismatched_attestations(
    path: tuple[str, ...],
    value: object,
) -> None:
    manifest = json.loads((SOPHUS / "manifest.json").read_text(encoding="utf-8"))
    selected = manifest
    for key in path[:-1]:
        selected = selected[key]
    selected[path[-1]] = value
    data = _witness_path("float64").read_bytes()
    with pytest.raises((ValidationError, ValueError)):
        binding = SophusBindingManifest.model_validate(manifest, strict=True)
        retain.validate_witness_binding(
            binding, retain.load_semantic_config(), "float64", data
        )


def test_shared_witness_binding_rejects_wrong_profile_even_with_valid_hash() -> None:
    manifest = json.loads((SOPHUS / "manifest.json").read_text(encoding="utf-8"))
    reference = manifest["artifacts"]["scalar_profiles"]["float64"]
    data = (
        _witness_path("float64")
        .read_bytes()
        .replace(b'"scalar_profile": "float64"', b'"scalar_profile": "float32"')
    )
    reference["selected_witness_sha256"] = hashlib.sha256(data).hexdigest()
    reference["selected_witness_file"] = (
        "experiments/oracles/sophus/shared/witnesses/float64-"
        + reference["selected_witness_sha256"]
        + ".json"
    )
    with pytest.raises(ValueError, match="profile or probe"):
        retain.validate_witness_binding(
            SophusBindingManifest.model_validate(manifest, strict=True),
            retain.load_semantic_config(),
            "float64",
            data,
        )


def test_synthetic_version_reuse_requires_its_own_recorded_exact_capture(
    tmp_path,
) -> None:
    """Synthetic records exercise binding only; they do not claim another native run."""
    second_data = read_evidence_document(SOPHUS / "semantic_contract.json")
    second_data["version"] = "9.9.9-rc1"
    second_data["scalar_profiles"] = [second_data["scalar_profiles"][1]]
    for profile in second_data["scalar_profiles"]:
        profile["witness_target"] = profile["witness_target"].replace(
            "1.24.6", "9.9.9-rc1"
        )
    second = SophusSemanticConfig.model_validate(second_data, strict=True)
    original = json.loads((SOPHUS / "manifest.json").read_text(encoding="utf-8"))
    synthetic = deepcopy(original)
    synthetic["target"] = f"Sophus {second.version} SO3d / {second.dependency}"
    del synthetic["artifacts"]["scalar_profiles"]["float32"]
    del synthetic["artifacts"]["direct_capture"]["raw_capture_sha256"]["float32"]
    data = _witness_path("float64").read_bytes()
    packages = tmp_path / "experiments/oracles/sophus"
    shared = packages / "shared/witnesses"
    shared.mkdir(parents=True)
    (shared / _witness_path("float64").name).write_bytes(data)
    first_package = packages / "1.24.6"
    second_package = packages / second.version
    for package in (first_package, second_package):
        package.mkdir()
        for name in (
            "source.json",
            "semantic_contract.json",
            "capture_inputs.tsv",
        ):
            shutil.copyfile(SOPHUS / name, package / name)
    shutil.copyfile(SOPHUS / "capture_native.cpp", first_package / "capture_native.cpp")
    (first_package / "manifest.json").write_text(json.dumps(original), encoding="utf-8")
    contract_bytes = json.dumps(second_data).encode()
    (second_package / "semantic_contract.json").write_bytes(contract_bytes)
    source = json.loads((second_package / "source.json").read_text(encoding="utf-8"))
    source["version"], source["tag"] = second.version, f"v{second.version}"
    source_bytes = json.dumps(source).encode()
    (second_package / "source.json").write_bytes(source_bytes)
    synthetic["semantic_contract"] = {
        "file": f"experiments/oracles/sophus/{second.version}/semantic_contract.json",
        "sha256": hashlib.sha256(contract_bytes).hexdigest(),
    }
    synthetic["artifacts"]["current_source_sha256"] = hashlib.sha256(
        source_bytes
    ).hexdigest()
    (second_package / "manifest.json").write_text(
        json.dumps(synthetic), encoding="utf-8"
    )
    assert (
        retain.load_shared_witness("float64", first_package).scalar_profile == "float64"
    )
    with pytest.raises(ValueError, match="binding differs"):
        retain.validate_witness_binding(
            SophusBindingManifest.model_validate(synthetic, strict=True),
            second,
            "float64",
            data,
        )
    run = synthetic["artifacts"]["direct_capture"]
    run["product_version"] = second.version
    reference = synthetic["artifacts"]["scalar_profiles"]["float64"]
    del reference["comparison"]
    with pytest.raises(ValidationError):
        SophusBindingManifest.model_validate(synthetic, strict=True)

    # This raw capture is synthetic; only an actual per-version run can support
    # a real version claim.  Exact comparison must precede recording its hashes.
    raw = json.dumps(_capture()).encode()
    with pytest.raises(ValueError, match="configuration differs"):
        retain.verify_exact_capture(raw, data, second)
    retain.verify_exact_capture(raw, data, second, second_package)
    raw_sha = hashlib.sha256(raw).hexdigest()
    run["raw_capture_sha256"]["float64"] = raw_sha
    reference["raw_capture_sha256"] = raw_sha
    reference["comparison"] = "verified_exact"
    (second_package / "manifest.json").write_text(
        json.dumps(synthetic), encoding="utf-8"
    )
    bound = retain.validate_witness_binding(
        SophusBindingManifest.model_validate(synthetic, strict=True),
        second,
        "float64",
        data,
    )
    assert bound == retain.load_shared_witness("float64")
    assert retain.load_shared_witness("float64", second_package) == bound

    with pytest.raises(ValueError, match="not recorded"):
        retain.validate_witness_binding(
            SophusBindingManifest.model_validate(synthetic, strict=True),
            second,
            "float32",
            data,
        )


@pytest.mark.parametrize("field", ["version", "dependency"])
def test_versioned_contract_rejects_mismatched_source_pin(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = retain.read_evidence_file

    def changed_source(*args, **kwargs):
        payload = original(*args, **kwargs)
        if kwargs["context"] == "Sophus source pin":
            document = json.loads(payload)
            if field == "version":
                document["version"] = "9.9.9"
            else:
                document["dependency"]["version"] = "9.9.9"
            return json.dumps(document).encode()
        return payload

    monkeypatch.setattr(retain, "read_evidence_file", changed_source)
    with pytest.raises(ValueError, match="source pin differs"):
        retain.load_semantic_config()


def test_historical_capture_sources_are_archived_and_current_sources_are_separate() -> (
    None
):
    manifest = json.loads((SOPHUS / "manifest.json").read_text())
    historical = manifest["artifacts"]["capture_sources_sha256"]
    archived_native = read_provenance_source(
        ROOT, manifest["artifacts"]["historical_capture_source_archive"]
    )
    assert (
        hashlib.sha256(archived_native).hexdigest() == historical["capture_native.cpp"]
    )
    assert (
        hashlib.sha256((SOPHUS / "capture_native.cpp").read_bytes()).hexdigest()
        == (manifest["artifacts"]["direct_capture"]["capture_source_sha256"])
    )
    assert manifest["pin_file"] == "source.json"
    artifacts = manifest["artifacts"]
    assert (
        hashlib.sha256((SOPHUS / "capture_inputs.tsv").read_bytes()).hexdigest()
        == artifacts["input_sha256"]
    )
    assert artifacts["historical_input_sha256"] != artifacts["input_sha256"]
    assert (
        historical["capture_native.cpp"]
        != (artifacts["direct_capture"]["capture_source_sha256"])
    )
    assert (
        hashlib.sha256((SOPHUS / "source.json").read_bytes()).hexdigest()
        == manifest["artifacts"]["current_source_sha256"]
    )
    for source_name, archive_name in (
        ("retain_witnesses.py", "sophus-1.24.6-retain_witnesses.py"),
        ("verify_scipy.py", "sophus-1.24.6-verify_scipy.py"),
    ):
        archived = read_provenance_source(
            ROOT, f"experiments/provenance/reproduction-v1.zip#{archive_name}"
        )
        assert hashlib.sha256(archived).hexdigest() == historical[source_name]
    current = manifest["artifacts"]["repair_replay_sources_sha256"]
    assert (
        hashlib.sha256(
            (ROOT / "scripts/evidence_schema/sophus.py").read_bytes()
        ).hexdigest()
        == manifest["artifacts"]["retained_schema_sha256"]
    )
    assert (
        hashlib.sha256((SOPHUS / "retain_witnesses.py").read_bytes()).hexdigest()
        == current["retain_witnesses.py"]
    )
    assert (
        hashlib.sha256((SOPHUS / "verify_scipy.py").read_bytes()).hexdigest()
        == current["verify_scipy.py"]
    )


@pytest.mark.parametrize(
    "values", [[1.0, float("nan")], [float("nan"), 1.0], [float("inf"), -float("inf")]]
)
def test_finite_max_abs_rejects_nonfinite_values_in_any_position(
    values: list[float],
) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        finite_max_abs(values, "Sophus residual")


def test_finite_max_abs_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        finite_max_abs([], "Sophus residual")


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize(
    "fault", ["input", "coefficients", "constructor_basis", "composition_basis"]
)
def test_scipy_cli_checks_retained_native_witnesses_and_rejects_corruption(
    precision: str,
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("scipy")
    verify_spec = importlib.util.spec_from_file_location(
        "sophus_verify_scipy", SOPHUS / "verify_scipy.py"
    )
    assert verify_spec is not None and verify_spec.loader is not None
    monkeypatch.setitem(sys.modules, "retain_witnesses", retain)
    verifier = importlib.util.module_from_spec(verify_spec)
    monkeypatch.setitem(sys.modules, verify_spec.name, verifier)
    verify_spec.loader.exec_module(verifier)
    witness_path = _witness_path(precision)
    monkeypatch.setattr(
        sys, "argv", ["verify_scipy.py", "--witnesses", str(witness_path)]
    )
    verifier.main()
    assert json.loads(capsys.readouterr().out)["result"] == "pass"

    native = SophusDiscoveryWitnessSet.model_validate_json(
        witness_path.read_bytes(), strict=True
    )
    forward, reverse = native.composition
    assert (
        max(abs(a - b) for x, y in zip(forward[2], reverse[2]) for a, b in zip(x, y))
        > 0.1
    )
    zero_basis = ((0.0, 0.0, 0.0),) * 3
    if fault == "composition_basis":
        row = native.composition[0]
        corrupted_row = (row[0], row[1], zero_basis, row[3])
        changed = {"composition": (corrupted_row, native.composition[1])}
        message = "basis actions exceeds"
    else:
        row = native.quaternion_construction[4]
        corrupted_row = (
            row[0],
            (1.0, 0.0, 0.0, 0.0) if fault == "input" else row[1],
            row[2],
            (0.0, 0.0, 0.0, 1.0) if fault == "coefficients" else row[3],
            zero_basis if fault == "constructor_basis" else row[4],
            *row[5:],
        )
        changed = {
            "quaternion_construction": (
                *native.quaternion_construction[:4],
                corrupted_row,
                *native.quaternion_construction[5:],
            )
        }
        message = (
            "pinned Sophus corpus"
            if fault == "input"
            else "coefficients exceeds"
            if fault == "coefficients"
            else "basis actions exceeds"
        )
    corrupted = SophusDiscoveryWitnessSet.model_validate_json(
        json.dumps({**json.loads(witness_path.read_text(encoding="utf-8")), **changed}),
        strict=True,
    )
    with pytest.raises(
        ValueError if fault == "input" else AssertionError, match=message
    ):
        verifier.verify(corrupted, 2e-6 if precision == "float32" else 2e-14)
