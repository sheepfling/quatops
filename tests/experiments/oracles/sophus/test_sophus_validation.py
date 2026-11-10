"""Regression coverage for the pinned Sophus capture contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any, cast

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


def _scientific_capture(rotation_api: Any, np: Any, target: str) -> dict[str, Any]:
    """Build a complete synthetic native-shaped capture for verifier replay."""
    is_float32 = target == "Sophus SO3f"

    def scalar(value: float) -> float:
        return float(np.float32(value)) if is_float32 else float(value)

    def vector(values: Any) -> list[float]:
        return [scalar(float(value)) for value in values]

    fixture = cast(dict[str, Any], _capture())
    probe = vector(fixture["probe_xyz"])
    basis_and_probe = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], probe],
        dtype=np.float64,
    )
    rotations: dict[str, Any] = {}
    constructors: list[dict[str, Any]] = []
    for item in fixture["quaternion_construction"]:
        input_wxyz = vector(item["input_wxyz"])
        expected = rotation_api.from_quat(
            np.asarray(input_wxyz, dtype=np.float64)[[1, 2, 3, 0]]
        )
        rotations[item["case_id"]] = expected
        stored_xyzw = vector(expected.as_quat())
        if item["case_id"] == "mixed_unequal_antipode":
            mixed_xyzw = constructors[4]["result"]["params_xyzw"]
            if float(np.dot(stored_xyzw, mixed_xyzw)) >= 0.0:
                stored_xyzw = [-value for value in stored_xyzw]
        constructors.append(
            {
                "case_id": item["case_id"],
                "input_wxyz": input_wxyz,
                "input_norm": scalar(float(np.linalg.norm(input_wxyz))),
                "result": _rotation_payload(
                    item["case_id"], expected, basis_and_probe, stored_xyzw, vector
                ),
                "log_xyz": vector(expected.as_rotvec()),
            }
        )

    exp_log: list[dict[str, Any]] = []
    for item in fixture["exp_log"]:
        tangent = vector(item["tangent_xyz"])
        expected = rotation_api.from_rotvec(tangent)
        rotations[f"exp_{item['case_id']}"] = expected
        exp_log.append(
            {
                "case_id": item["case_id"],
                "tangent_xyz": tangent,
                "exp_result": _rotation_payload(
                    item["case_id"], expected, basis_and_probe, None, vector
                ),
                "log_of_exp_xyz": vector(expected.as_rotvec()),
            }
        )

    mixed_a = rotations["exp_mixed_a"]
    mixed_b = rotations["exp_mixed_b"]
    delta = rotations["exp_perturbation"]
    return {
        "capture_schema": fixture["capture_schema"],
        "target": target,
        "probe_xyz": probe,
        "quaternion_construction": constructors,
        "exp_log": exp_log,
        "composition": [
            _rotation_payload(
                "mixed_a_times_mixed_b",
                mixed_a * mixed_b,
                basis_and_probe,
                None,
                vector,
            ),
            _rotation_payload(
                "mixed_b_times_mixed_a",
                mixed_b * mixed_a,
                basis_and_probe,
                None,
                vector,
            ),
        ],
        "inverse": _rotation_payload(
            "inverse_mixed_a", mixed_a.inv(), basis_and_probe, None, vector
        ),
        "perturbation_side": [
            _rotation_payload(
                "delta_times_mixed_a", delta * mixed_a, basis_and_probe, None, vector
            ),
            _rotation_payload(
                "mixed_a_times_delta", mixed_a * delta, basis_and_probe, None, vector
            ),
        ],
    }


def _rotation_payload(
    case_id: str, rotation: Any, probes: Any, stored_xyzw: Any, vector: Any
) -> dict[str, Any]:
    xyzw = stored_xyzw if stored_xyzw is not None else vector(rotation.as_quat())
    wxyz = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
    return {
        "case_id": case_id,
        "unit_quaternion": {"wxyz": wxyz, "coeffs_xyzw": xyzw},
        "params_xyzw": xyzw,
        "matrix_rows": [vector(row) for row in rotation.as_matrix()],
        "native_vector_actions_xyz": [vector(row) for row in rotation.apply(probes)],
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


def test_scipy_rotation_comparison_checks_every_action_and_quaternion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scipy = pytest.importorskip("scipy")
    _ = scipy
    verify_path = SOPHUS / "verify_scipy.py"
    verify_spec = importlib.util.spec_from_file_location(
        "sophus_verify_scipy", verify_path
    )
    assert verify_spec is not None and verify_spec.loader is not None
    monkeypatch.setitem(sys.modules, "retain_witnesses", retain)
    verifier = importlib.util.module_from_spec(verify_spec)
    monkeypatch.setitem(sys.modules, verify_spec.name, verifier)
    verify_spec.loader.exec_module(verifier)
    model = retain.NativeCapture.model_validate_json(
        json.dumps(_capture()), strict=True
    )
    expected = verifier.Rotation.identity()
    probes = verifier.np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], model.probe_xyz]
    )
    with pytest.raises(ValueError, match="non-finite"):
        verifier._max_difference(
            verifier.np.asarray([-1e308]),
            verifier.np.asarray([1e308]),
            "overflowed error",
        )
    corrupt = model.composition[0].model_copy(
        update={
            "params_xyzw": (0.0, 0.0, 0.0, 0.0),
            "unit_quaternion": model.composition[0].unit_quaternion.model_copy(
                update={
                    "coeffs_xyzw": (0.0, 0.0, 0.0, 0.0),
                    "wxyz": (0.0, 0.0, 0.0, 0.0),
                }
            ),
        }
    )
    matrix_error, action_error, quaternion_error = verifier._rotation_differences(
        corrupt, expected, probes, "corrupt test"
    )
    assert matrix_error == 0.0
    assert action_error == 0.0
    assert quaternion_error > 0.5
    bad_actions = model.exp_log[2].exp_result.model_copy(
        update={"native_vector_actions_xyz": ((0.0, 0.0, 0.0),) * 4}
    )
    _, action_error, _ = verifier._rotation_differences(
        bad_actions, expected, probes, "corrupt actions"
    )
    assert action_error > 0.5
    bad_perturbation = model.perturbation_side[0].model_copy(
        update={"native_vector_actions_xyz": ((0.0, 0.0, 0.0),) * 4}
    )
    _, action_error, _ = verifier._rotation_differences(
        bad_perturbation, expected, probes, "corrupt perturbation row"
    )
    assert action_error > 0.5


@pytest.mark.parametrize("target", ["Sophus SO3f", "Sophus SO3d"])
def test_scipy_cli_accepts_complete_synthetic_pinned_capture(
    target: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    np = pytest.importorskip("numpy")
    scipy_spatial = pytest.importorskip("scipy.spatial.transform")
    raw = _scientific_capture(scipy_spatial.Rotation, np, target)
    capture_path = tmp_path / f"{target.replace(' ', '_')}.json"
    capture_path.write_text(json.dumps(raw, allow_nan=False), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "retain_witnesses", retain)
    verify_spec = importlib.util.spec_from_file_location(
        "sophus_verify_scipy", SOPHUS / "verify_scipy.py"
    )
    assert verify_spec is not None and verify_spec.loader is not None
    verifier = importlib.util.module_from_spec(verify_spec)
    monkeypatch.setitem(sys.modules, verify_spec.name, verifier)
    verify_spec.loader.exec_module(verifier)
    monkeypatch.setattr(
        sys, "argv", ["verify_scipy.py", "--capture", str(capture_path)]
    )

    verifier.main()
    report = json.loads(capsys.readouterr().out)
    assert report["result"] == "pass"
    assert report["native_target"] == target


@pytest.mark.parametrize("target", ["Sophus SO3f", "Sophus SO3d"])
def test_scipy_cli_rejects_finite_perturbation_action_corruption(
    target: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    scipy_spatial = pytest.importorskip("scipy.spatial.transform")
    raw = _scientific_capture(scipy_spatial.Rotation, np, target)
    raw["perturbation_side"][0]["native_vector_actions_xyz"][0] = [0.0, 0.0, 0.0]
    capture_path = tmp_path / f"{target.replace(' ', '_')}_corrupt.json"
    capture_path.write_text(json.dumps(raw, allow_nan=False), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "retain_witnesses", retain)
    verify_spec = importlib.util.spec_from_file_location(
        "sophus_verify_scipy", SOPHUS / "verify_scipy.py"
    )
    assert verify_spec is not None and verify_spec.loader is not None
    verifier = importlib.util.module_from_spec(verify_spec)
    monkeypatch.setitem(sys.modules, verify_spec.name, verifier)
    verify_spec.loader.exec_module(verifier)
    monkeypatch.setattr(
        sys, "argv", ["verify_scipy.py", "--capture", str(capture_path)]
    )

    with pytest.raises(AssertionError, match="exceed"):
        verifier.main()
