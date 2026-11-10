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


def _rotation(case_id: str, probe: list[float]) -> dict[str, object]:
    sign = -1.0 if case_id == "mixed_unequal_antipode" else 1.0
    return {
        "case_id": case_id,
        "unit_quaternion": {
            "wxyz": [sign, 0.0, 0.0, 0.0],
            "coeffs_xyzw": [0.0, 0.0, 0.0, sign],
        },
        "params_xyzw": [0.0, 0.0, 0.0, sign],
        "matrix_rows": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "native_vector_actions_xyz": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            probe,
        ],
    }


def _capture() -> dict[str, object]:
    probe = [2.0, -3.0, 5.0]
    pinned: dict[str, dict[str, list[float]]] = {"quaternion": {}, "tangent": {}}
    for line in (SOPHUS / "capture_inputs.tsv").read_text().splitlines():
        fields = line.partition("#")[0].split()
        if fields and fields[0] in pinned and fields[0] != "probe":
            pinned[fields[0]][fields[1]] = [float(value) for value in fields[2:]]
        elif fields and fields[0] == "probe":
            probe = [float(value) for value in fields[1:]]
    constructors = [
        {
            "case_id": case,
            "input_wxyz": pinned["quaternion"][case],
            "input_norm": sum(value * value for value in pinned["quaternion"][case])
            ** 0.5,
            "result": _rotation(case, probe),
            "log_xyz": [0.0, 0.0, 0.0],
        }
        for case in retain._CONSTRUCTION_CASES
    ]
    exp_log = [
        {
            "case_id": case,
            "tangent_xyz": pinned["tangent"][case],
            "exp_result": _rotation(case, probe),
            "log_of_exp_xyz": [0.0, 0.0, 0.0],
        }
        for case in retain._RAW_EXP_LOG_CASES
    ]
    return {
        "capture_schema": "quatops-sophus-discovery-v1",
        "target": "Sophus SO3d",
        "probe_xyz": probe,
        "quaternion_construction": constructors,
        "exp_log": exp_log,
        "composition": [
            _rotation("mixed_a_times_mixed_b", probe),
            _rotation("mixed_b_times_mixed_a", probe),
        ],
        "inverse": _rotation("inverse_mixed_a", probe),
        "perturbation_side": [
            _rotation("delta_times_mixed_a", probe),
            _rotation("mixed_a_times_delta", probe),
        ],
    }


def test_complete_capture_matches_pinned_tsv() -> None:
    capture = retain.NativeCapture.model_validate_json(
        json.dumps(_capture()), strict=True
    )
    capture.validate_pinned_inputs(SOPHUS / "capture_inputs.tsv")


def test_compact_witnesses_keep_the_documented_six_exp_log_rows() -> None:
    selected = retain.reduce_capture(
        json.dumps(_capture()).encode("utf-8"), retain.load_semantic_config()
    )
    assert tuple(item[0] for item in selected.exp_log) == (
        "zero",
        "near_zero",
        "mixed_a",
        "mixed_b",
        "near_pi_axis",
        "above_pi_axis",
    )
    assert len(selected.exp_log) == 6


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

    def float32(value: float) -> float:
        return struct.unpack("=f", struct.pack("=f", value))[0]

    raw["probe_xyz"] = [float32(value) for value in raw["probe_xyz"]]  # type: ignore[union-attr]
    for row in raw["quaternion_construction"]:  # type: ignore[union-attr]
        row["input_wxyz"] = [float32(value) for value in row["input_wxyz"]]
    for row in raw["exp_log"]:  # type: ignore[union-attr]
        row["tangent_xyz"] = [float32(value) for value in row["tangent_xyz"]]
    capture = retain.NativeCapture.model_validate_json(json.dumps(raw), strict=True)
    capture.validate_pinned_inputs(SOPHUS / "capture_inputs.tsv")


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_constructor",
        "duplicate_constructor",
        "missing_exp_log_perturbation",
        "nested_id",
    ],
)
def test_rejects_incomplete_duplicate_or_mismatched_cases(mutation: str) -> None:
    raw = _capture()
    if mutation == "missing_constructor":
        raw["quaternion_construction"] = raw["quaternion_construction"][:-1]  # type: ignore[index]
    elif mutation == "duplicate_constructor":
        raw["quaternion_construction"][-1] = raw["quaternion_construction"][0]  # type: ignore[index]
    elif mutation == "missing_exp_log_perturbation":
        raw["exp_log"] = raw["exp_log"][:-1]  # type: ignore[index]
    else:
        raw["quaternion_construction"][0]["result"]["case_id"] = "other"  # type: ignore[index]
    with pytest.raises(ValidationError):
        retain.NativeCapture.model_validate_json(json.dumps(raw), strict=True)


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_rejects_missing_or_unexpected_perturbation_side_rows(mutation: str) -> None:
    raw = _capture()
    if mutation == "missing":
        raw["perturbation_side"] = raw["perturbation_side"][:-1]  # type: ignore[index]
    else:
        raw["perturbation_side"].append(  # type: ignore[union-attr]
            _rotation("unexpected_case", [2.0, -3.0, 5.0])
        )
    with pytest.raises(ValidationError):
        retain.NativeCapture.model_validate_json(json.dumps(raw), strict=True)


def test_rejects_antipode_pair_with_the_same_stored_native_sign() -> None:
    raw = _capture()
    antipode = raw["quaternion_construction"][5]["result"]  # type: ignore[index]
    antipode["unit_quaternion"] = {  # type: ignore[index]
        "wxyz": [1.0, 0.0, 0.0, 0.0],
        "coeffs_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    antipode["params_xyzw"] = [0.0, 0.0, 0.0, 1.0]  # type: ignore[index]
    with pytest.raises(ValidationError, match="opposite signs"):
        retain.NativeCapture.model_validate_json(json.dumps(raw), strict=True)


def test_rejects_nan_in_native_matrix() -> None:
    encoded = json.dumps(_capture()).replace(
        '"matrix_rows": [[1.0, 0.0, 0.0]', '"matrix_rows": [[NaN, 0.0, 0.0]', 1
    )
    with pytest.raises(ValidationError):
        retain.NativeCapture.model_validate_json(encoded, strict=True)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_rejects_nonfinite_capture_numbers(token: str) -> None:
    encoded = json.dumps(_capture()).replace(
        '"probe_xyz": [2.0, -3.0, 5.0]', f'"probe_xyz": [{token}, -3.0, 5.0]'
    )
    with pytest.raises(ValidationError):
        retain.NativeCapture.model_validate_json(encoded, strict=True)


def test_rejects_all_ids_with_substituted_zero_inputs() -> None:
    raw = _capture()
    raw["probe_xyz"] = [0.0, 0.0, 0.0]
    for row in raw["quaternion_construction"]:  # type: ignore[union-attr]
        row["input_wxyz"] = [1.0, 0.0, 0.0, 0.0]
    for row in raw["exp_log"]:  # type: ignore[union-attr]
        row["tangent_xyz"] = [0.0, 0.0, 0.0]
    capture = retain.NativeCapture.model_validate_json(json.dumps(raw), strict=True)
    with pytest.raises(ValueError, match="pinned TSV"):
        capture.validate_pinned_inputs(SOPHUS / "capture_inputs.tsv")


def test_historical_capture_sources_are_archived_and_current_sources_are_separate() -> (
    None
):
    manifest = json.loads((SOPHUS / "manifest.json").read_text())
    historical = manifest["artifacts"]["capture_sources_sha256"]
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
