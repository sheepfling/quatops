"""Validate packaged evidence; these are NOT QuatOps implementation tests."""
from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import struct
from typing import Any, Iterator

import jsonschema
import numpy as np
import pytest

from shared.capture_io import ROOT, numeric, sha256
from tools.verify_results import accessor, animated_quaternion, asset_header

RUNS = tuple(sorted((ROOT / "runs").glob("*/manifest.json")))
REPORT = json.loads((ROOT / "verification/comparison_report.json").read_text())
ASSET_STEMS = ("static_hierarchy", "step", "linear", "cubic", "cubic_all_negated", "cubic_middle_negated")


def observations(manifest: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in manifest.with_name("observations.jsonl").read_text().splitlines()]


def resolve_recorded_path(name: str, run: Path | None = None) -> Path:
    """Replay original absolute provenance paths after the bundle is relocated."""
    path = Path(name)
    marker = "quatops_capture_wave2"
    marker_index = next(
        (index for index, part in enumerate(path.parts) if part.casefold() == marker.casefold()),
        None,
    )
    if marker_index is not None:
        return ROOT.joinpath(*path.parts[marker_index + 1:])
    if path.is_absolute():
        return path
    return (run or ROOT) / path


def numerically_matches(actual: Any, expected: Any, atol: float = 1e-7) -> bool:
    """Allow host libm last-bit variation while keeping report structure exact."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(actual - expected) <= atol
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            numerically_matches(actual[key], expected[key], atol) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            numerically_matches(left, right, atol) for left, right in zip(actual, expected)
        )
    return actual == expected


def blobs(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if {"dtype", "shape", "values", "ieee754_hex_be"} <= value.keys():
            yield value
        else:
            for item in value.values():
                yield from blobs(item)
    elif isinstance(value, list):
        for item in value:
            yield from blobs(item)


@pytest.mark.parametrize("manifest_path", RUNS, ids=lambda p: p.parent.name)
def test_manifest_schema_and_counts(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    schema = json.loads((ROOT / "schemas/capture-manifest.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(manifest)
    records = observations(manifest_path)
    counts = Counter(item["status"] for item in records)
    assert manifest["coverage"]["status_counts"] == dict(counts)
    assert manifest["coverage"]["native_observation_count"] == counts["ok"]
    assert manifest["coverage"]["quatops_replay_count"] == 0
    assert manifest["execution"]["was_run"] is True


@pytest.mark.parametrize("manifest_path", RUNS, ids=lambda p: p.parent.name)
def test_every_observation_schema_and_bits(manifest_path: Path) -> None:
    schema = json.loads((ROOT / "schemas/observation.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)
    for record in observations(manifest_path):
        validator.validate(record)
        for blob in blobs(record):
            array = np.asarray(blob["values"])
            assert list(array.shape) == blob["shape"]
            code = ">f" if blob["dtype"] == "binary32" else ">d"
            assert len(blob["ieee754_hex_be"]) == array.size
            assert [struct.pack(code, float(x)).hex() for x in array.flat] == blob["ieee754_hex_be"]


@pytest.mark.parametrize("manifest_path", RUNS, ids=lambda p: p.parent.name)
def test_capture_artifact_hashes(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    for key in ("exporter", "corpus"):
        item = manifest[key]
        assert sha256(resolve_recorded_path(item["path"])) == item["sha256"]
    for item in manifest["artifacts"]:
        assert sha256(resolve_recorded_path(item["path"], manifest_path.parent)) == item["sha256"]


@pytest.mark.parametrize("dtype,negzero", [("binary32", "80000000"), ("binary64", "8000000000000000")])
def test_numeric_bits_and_nonfinite(dtype: str, negzero: str) -> None:
    result = numeric([0.0, -0.0, float("inf"), float("nan")], dtype)
    assert result["ieee754_hex_be"][1] == negzero
    assert result["values"][2] == {"nonfinite": "+inf"}
    assert result["values"][3] == {"nonfinite": "nan"}
    json.dumps(result, allow_nan=False)


def test_ragged_arrays_rejected() -> None:
    with pytest.raises(ValueError, match="Ragged"):
        numeric([[1.0], [1.0, 2.0]])


@pytest.mark.parametrize("stem", ASSET_STEMS)
def test_gltf_and_glb_share_authored_content(stem: str) -> None:
    gltf, buffer_a = asset_header(ROOT / f"assets/gltf/{stem}.gltf")
    glb, buffer_b = asset_header(ROOT / f"assets/gltf/{stem}.glb")
    length = gltf["buffers"][0]["byteLength"]
    assert buffer_a == buffer_b[:length]
    gltf["buffers"][0].pop("uri")
    assert gltf == glb
    assert gltf["asset"]["version"] == "2.0"
    assert gltf["meshes"] and gltf["nodes"]


def test_cubic_sign_controls() -> None:
    docs = [asset_header(ROOT / f"assets/gltf/{stem}.gltf") for stem in
            ("cubic", "cubic_all_negated", "cubic_middle_negated")]
    samples = (0.1875, 0.375, 1.1875, 1.625)
    difference_seen = False
    for t in samples:
        base, all_neg, middle = [animated_quaternion(doc, buf, t) for doc, buf in docs]
        np.testing.assert_allclose(all_neg, -base, atol=1e-14, rtol=0)
        if min(np.linalg.norm(base-middle), np.linalg.norm(base+middle)) > 1e-3:
            difference_seen = True
    assert difference_seen, "Per-key sign changes must not be silently canonicalized"
    doc, buf = docs[0]
    times = accessor(doc, buf, doc["animations"][0]["samplers"][0]["input"]).ravel()
    assert list(times) == [0.0, 0.75, 2.5]


@pytest.mark.parametrize("check", REPORT["gltf"]["checks"], ids=lambda r: f'{r["consumer"]}-{r["asset"]}-{r["operation"]}-{r["time_seconds"]}')
def test_recorded_gltf_reference_comparison(check: dict[str, Any]) -> None:
    assert check["pass"] and check["max_abs_matrix_error"] < 1e-6


@pytest.mark.parametrize("check", REPORT["proj"]["checks"], ids=lambda r: r["case_id"])
def test_recorded_proj_reference_comparison(check: dict[str, Any]) -> None:
    assert check["pass"]
    assert check["analytic_point_max_abs_error_m"] < 1e-7
    assert check["roundtrip_max_abs_error_m"] < 1e-7


def test_opencv_forward_results() -> None:
    maxima = REPORT["opencv"]["maxima"]
    assert maxima["forward_matrix_binary32"] < 1e-6
    assert maxima["forward_matrix_binary64"] < 1e-12


def test_opencv_inverse_discrepancies_remain_visible() -> None:
    discrepancies = REPORT["opencv"]["discrepancies"]
    assert len(discrepancies) == 26  # Versioned observed diagnostics, not waived accuracy tests.
    assert all(item["operation"] == "native_inverse_roundtrip_binary64" for item in discrepancies)
    assert all("classification" in item for item in discrepancies)


@pytest.mark.parametrize("check", REPORT["opencv"]["jacobian_checks"], ids=lambda r: r["case_id"])
def test_opencv_jacobian_finite_difference(check: dict[str, Any]) -> None:
    assert len(check["steps"]) == 3
    assert check["input_dimensions"] == 3 and check["output_dimensions"] == 9
    assert min(step["max_abs_error"] for step in check["steps"]) < 1e-9


@pytest.mark.parametrize("path", sorted((ROOT / "exporters").glob("*.py")), ids=lambda p: p.stem)
def test_python_exporters_have_no_comparison_oracle_import(path: Path) -> None:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else ([node.module] if isinstance(node, ast.ImportFrom) else [])
        assert not any(name and name.split(".")[0] in {"scipy", "quatops"} for name in names)


def test_capture_totals_are_not_replay_counts() -> None:
    all_records = [record for manifest in RUNS for record in observations(manifest)]
    assert Counter(record["status"] for record in all_records) == {"ok": 399, "not_run": 6}
    assert REPORT["quatops_executed"] is False
    assert REPORT["proj"]["cesium_executed"] is False
    assert len(RUNS) == 4


def test_usd_metadata_pair_is_not_a_hidden_basis_conversion() -> None:
    # Text invariant only. No pxr parser or USD runtime executed in this environment.
    a = (ROOT / "assets/usd/transform_stack_y.usda").read_text()
    b = (ROOT / "assets/usd/transform_stack_z.usda").read_text()
    assert a.replace('upAxis = "Y"', 'upAxis = "Z"') == b

@pytest.mark.parametrize("section", ["opencv", "gltf", "proj"])
def test_recompute_comparisons_from_raw_capture(section: str) -> None:
    from tools import verify_results
    evaluator = {"opencv": verify_results.verify_opencv, "gltf": verify_results.verify_gltf, "proj": verify_results.verify_proj}[section]
    assert numerically_matches(evaluator(), REPORT[section])
