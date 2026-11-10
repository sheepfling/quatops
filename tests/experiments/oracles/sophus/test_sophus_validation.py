"""Regression coverage for the pinned Sophus capture contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys

import pytest
from pydantic import ValidationError

from scripts.evidence_schema.primitives import finite_max_abs
from scripts.evidence_schema.sophus import SophusDiscoveryWitnessSet
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


def _capture() -> dict:
    witness = json.loads(
        (SOPHUS / "witnesses/float64.json").read_text(encoding="utf-8")
    )
    fields = (
        "probe_xyz",
        "quaternion_construction",
        "exp_log",
        "composition",
        "inverse",
        "perturbation_side",
    )
    return {
        "capture_schema": "quatops-sophus-direct-v1",
        "target": "Sophus SO3d",
        **{field: witness[field] for field in fields},
    }


def test_direct_capture_matches_pinned_tsv() -> None:
    capture = retain.NativeCapture.model_validate_json(
        json.dumps(_capture()), strict=True
    )
    capture.validate_pinned_inputs(SOPHUS / "capture_inputs.tsv")
    selected = retain.reduce_capture(
        json.dumps(_capture()).encode(), retain.load_semantic_config()
    )
    assert tuple(row[0] for row in selected.exp_log) == retain.EXP_LOG_CASES
    frozen = json.loads((SOPHUS / "witnesses/float64.json").read_text(encoding="utf-8"))
    actual = selected.model_dump(mode="json")
    for field in frozen.keys() - {"raw_capture_sha256"}:
        assert actual[field] == frozen[field]


@pytest.mark.parametrize("precision,tolerance", [("float32", 2e-6), ("float64", 2e-14)])
def test_retained_sophus_witnesses_validate_and_match_native_basis_actions(
    precision: str, tolerance: float
) -> None:
    np = pytest.importorskip("numpy")
    rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    manifest = json.loads((SOPHUS / "manifest.json").read_text())
    profile = manifest["artifacts"]["scalar_profiles"][precision]
    payload = (SOPHUS / profile["selected_witness_file"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == profile["selected_witness_sha256"]
    witnesses = SophusDiscoveryWitnessSet.model_validate_json(payload, strict=True)
    assert witnesses.raw_capture_sha256 == profile["raw_capture_sha256"]
    rows = [(row[3], row[4]) for row in witnesses.quaternion_construction]
    rows += [(row[2], row[3]) for row in witnesses.exp_log]
    rows += [(row[1], row[2]) for row in witnesses.composition]
    rows += [(witnesses.inverse[1], witnesses.inverse[2])]
    rows += [(row[1], row[2]) for row in witnesses.perturbation_side]
    for native_xyzw, native_basis_images in rows:
        np.testing.assert_allclose(
            rotation.from_quat(native_xyzw).apply(np.eye(3)),
            native_basis_images,
            rtol=0.0,
            atol=tolerance,
        )


def test_float32_capture_inputs_are_compared_at_exporter_precision() -> None:
    raw = _capture()
    raw["target"] = "Sophus SO3f"

    def single(value):
        return struct.unpack("=f", struct.pack("=f", value))[0]

    for field in ("quaternion_construction", "exp_log"):
        for row in raw[field]:
            row[1] = [single(value) for value in row[1]]
    capture = retain.NativeCapture.model_validate_json(json.dumps(raw), strict=True)
    capture.validate_pinned_inputs(SOPHUS / "capture_inputs.tsv")


@pytest.mark.parametrize(
    "fault",
    [
        "missing_constructor",
        "duplicate_constructor",
        "missing_perturbation",
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
    elif fault == "missing_perturbation":
        raw["perturbation_side"].pop()
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
        capture.validate_pinned_inputs(SOPHUS / "capture_inputs.tsv")


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
    assert (
        hashlib.sha256((SOPHUS / "source.json").read_bytes()).hexdigest()
        == historical["source.json"]
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
    "fault", ["input", "coefficients", "constructor_basis", "perturbation_basis"]
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
    witness_path = SOPHUS / "witnesses" / f"{precision}.json"
    monkeypatch.setattr(
        sys, "argv", ["verify_scipy.py", "--witnesses", str(witness_path)]
    )
    verifier.main()
    assert json.loads(capsys.readouterr().out)["result"] == "pass"

    native = SophusDiscoveryWitnessSet.model_validate_json(
        witness_path.read_bytes(), strict=True
    )
    zero_basis = ((0.0, 0.0, 0.0),) * 3
    if fault == "perturbation_basis":
        row = native.perturbation_side[0]
        corrupted_row = (row[0], row[1], zero_basis, row[3])
        changed = {"perturbation_side": (corrupted_row, native.perturbation_side[1])}
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
