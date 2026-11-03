"""glTF 2.0 input and named Three.js/Blender loader observations."""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from tests.native_engine_support import (
    from_engine_xyzw,
    quaternion_coefficient_error,
    to_hamilton_xyzw,
)
from tests.oracle_support import BACKENDS


ROOT = Path(__file__).parent.parent
ASSET_DIR = ROOT / "experiments" / "transports" / "gltf" / "2.0.0" / "assets"
GLTF_PATH = ASSET_DIR / "rotation_transport.gltf"
BINARY_PATH = ASSET_DIR / "rotation_transport.bin"
THREE_PATH = (
    Path(__file__).parent
    / "golden"
    / "transports"
    / "gltf"
    / "2.0.0"
    / "threejs"
    / "0.187.0"
    / "loader_rotation.json"
)
BLENDER_PATH = (
    Path(__file__).parent
    / "golden"
    / "transports"
    / "gltf"
    / "2.0.0"
    / "blender"
    / "5.1.2"
    / "native_rotation.json"
)
ASSET = json.loads(GLTF_PATH.read_text(encoding="utf-8"))
THREE = json.loads(THREE_PATH.read_text(encoding="utf-8"))
BLENDER = json.loads(BLENDER_PATH.read_text(encoding="utf-8"))
BINARY = BINARY_PATH.read_bytes()
TOLERANCE = 3e-7


def _accessor(index):
    accessor = ASSET["accessors"][index]
    view = ASSET["bufferViews"][accessor["bufferView"]]
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    count = accessor["count"] * width
    values = struct.unpack_from(f"<{count}f", BINARY, offset)
    return np.asarray(values, dtype=float).reshape((accessor["count"], width))


def _rotation(xyzw):
    return Rotation.from_quat(np.asarray(xyzw, dtype=float))


def _matrix_error(lhs, rhs):
    return float(np.max(np.abs(np.asarray(lhs, dtype=float) - np.asarray(rhs, dtype=float))))


def _basis_blender():
    images = BLENDER["observations"]["source_translation_axis_images_in_blender_xyz"]
    basis = np.column_stack([images[axis] for axis in "xyz"])
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), rtol=0.0, atol=1e-8)
    assert np.linalg.det(basis) == pytest.approx(1.0, abs=1e-8)
    return basis


def _map_xyzw_to_blender(xyzw, basis):
    q = np.asarray(xyzw, dtype=float)
    return np.r_[basis @ q[:3], q[3]]


def _glTF_node_matrix(node):
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=float).reshape((4, 4), order="F")
    translation = np.asarray(node.get("translation", (0, 0, 0)), dtype=float)
    rotation = _rotation(node.get("rotation", (0, 0, 0, 1))).as_matrix()
    scale = np.diag(np.asarray(node.get("scale", (1, 1, 1)), dtype=float))
    result = np.eye(4)
    result[:3, :3] = rotation @ scale
    result[:3, 3] = translation
    return result


def _basis_conjugate_matrix(matrix, basis):
    change = np.eye(4)
    change[:3, :3] = basis
    return change @ matrix @ change.T


def _cubic_value(times, values, time_seconds):
    index = int(np.searchsorted(times, time_seconds, side="right") - 1)
    index = min(max(index, 0), len(times) - 2)
    duration = times[index + 1] - times[index]
    u = (time_seconds - times[index]) / duration
    in_tangent, value, out_tangent = values[index]
    next_in_tangent, next_value, _ = values[index + 1]
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    result = h00 * value + h10 * duration * out_tangent
    result += h01 * next_value + h11 * duration * next_in_tangent
    return result / np.linalg.norm(result)


def _source_key_values():
    return _accessor(1)


def test_gltf_assets_and_loader_versions_are_frozen_and_valid():
    assert ASSET["asset"]["version"] == "2.0"
    assert THREE["evidence_kind"] == "native_gltf_loader_capture"
    assert THREE["loader"]["name"] == "Three.js GLTFLoader"
    assert THREE["loader"]["package_version"] == "0.187.0"
    assert THREE["loader"]["three_revision"] == "187"
    assert BLENDER["evidence_kind"] == "native_blender_gltf_importer_capture"
    assert BLENDER["loader"]["name"] == "Blender glTF importer"
    assert BLENDER["loader"]["version"] == "5.1.2"
    assert THREE["provenance"]["gltf_sha256"] == hashlib.sha256(GLTF_PATH.read_bytes()).hexdigest()
    assert THREE["provenance"]["binary_sha256"] == hashlib.sha256(BINARY).hexdigest()
    assert BLENDER["provenance"]["gltf_sha256"] == hashlib.sha256(GLTF_PATH.read_bytes()).hexdigest()
    assert BLENDER["provenance"]["binary_sha256"] == hashlib.sha256(BINARY).hexdigest()
    validation = THREE["validation"]
    assert validation["evidence_kind"].startswith("Khronos glTF Validator")
    assert validation["validator_version"] == "2.0.0-dev.3.10"
    assert validation["num_errors"] == validation["num_warnings"] == 0
    assert validation["messages"] == []


def test_gltf_static_trs_matrix_parent_and_axis_mapping_are_explicit():
    three_static = THREE["observations"]["static_transforms"]
    blender_static = BLENDER["observations"]["static_transforms"]
    nodes = {node["name"]: node for node in ASSET["nodes"]}
    parent_source = _glTF_node_matrix(nodes["parent_frame"])
    trs_source = _glTF_node_matrix(nodes["static_trs"])
    matrix_source = _glTF_node_matrix(nodes["static_matrix"])
    np.testing.assert_allclose(trs_source, matrix_source, rtol=0.0, atol=2e-15)

    for name, source_matrix in (("parent_frame", parent_source), ("static_trs", trs_source), ("static_matrix", matrix_source)):
        np.testing.assert_allclose(
            three_static[name]["local_matrix"]["rows"],
            source_matrix,
            rtol=0.0,
            atol=2e-14,
        )
    parent_three = np.asarray(three_static["parent_frame"]["world_matrix"]["rows"])
    child_three = np.asarray(three_static["static_trs"]["world_matrix"]["rows"])
    np.testing.assert_allclose(
        child_three,
        parent_three @ trs_source,
        rtol=0.0,
        atol=3e-14,
    )

    basis = _basis_blender()
    np.testing.assert_allclose(basis, np.array(((1, 0, 0), (0, 0, -1), (0, 1, 0))), atol=1e-8)
    for name, source_matrix in (("parent_frame", parent_source), ("static_trs", trs_source), ("static_matrix", matrix_source)):
        mapped = _basis_conjugate_matrix(source_matrix, basis)
        np.testing.assert_allclose(
            blender_static[name]["matrix_basis_rows"],
            mapped,
            rtol=0.0,
            atol=TOLERANCE,
            err_msg=f"Blender glTF {name} local transform basis mapping",
        )
    np.testing.assert_allclose(
        blender_static["static_trs"]["matrix_world_rows"],
        blender_static["static_matrix"]["matrix_world_rows"],
        rtol=0.0,
        atol=TOLERANCE,
    )
    for axis in "xyz":
        expected = basis[:, "xyz".index(axis)]
        actual = BLENDER["observations"]["source_translation_axis_images_in_blender_xyz"][axis]
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-8)


def test_threejs_gltf_loader_matches_glTF_step_slerp_and_cubic_contracts():
    times = _accessor(0).reshape(-1)
    keys = _source_key_values()
    cubic = _accessor(3).reshape((len(times), 3, 4))
    animations = {
        row["animation_name"].replace("rotation_", "").upper(): row
        for row in THREE["observations"]["animations"]
    }
    assert set(animations) == {"STEP", "LINEAR", "CUBIC"}

    for sample in animations["STEP"]["samples"]:
        index = min(max(int(np.searchsorted(times, sample["time_seconds"], side="right") - 1), 0), 2)
        assert quaternion_coefficient_error(sample["quaternion_xyzw"], keys[index]) <= TOLERANCE

    linear_slerp = Slerp(times, Rotation.from_quat(keys))
    for sample in animations["LINEAR"]["samples"]:
        expected = linear_slerp([sample["time_seconds"]]).as_quat()[0]
        assert quaternion_coefficient_error(sample["quaternion_xyzw"], expected) <= TOLERANCE

    for sample in animations["CUBIC"]["samples"]:
        expected = _cubic_value(times, cubic, sample["time_seconds"])
        assert quaternion_coefficient_error(sample["quaternion_xyzw"], expected) <= TOLERANCE
        assert sample["quaternion_length"] == pytest.approx(1.0, abs=3e-8)


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_threejs_gltf_linear_rotation_samples_match_quatops_slerp(
    backend_name, module, is_jpl, is_numpy, component_first
):
    keys = _source_key_values()
    row = next(
        item
        for item in THREE["observations"]["animations"]
        if item["animation_name"] == "rotation_linear"
    )
    for sample in row["samples"]:
        t = sample["time_seconds"]
        interval = min(max(int(np.searchsorted(_accessor(0).reshape(-1), t, side="right") - 1), 0), 1)
        times = _accessor(0).reshape(-1)
        factor = (t - times[interval]) / (times[interval + 1] - times[interval])
        start = from_engine_xyzw(module, keys[interval], is_jpl, is_numpy, component_first)
        end = from_engine_xyzw(module, keys[interval + 1], is_jpl, is_numpy, component_first)
        actual = module.slerp(start, end, factor, shortest=True)
        actual_xyzw = to_hamilton_xyzw(module, actual, is_jpl, is_numpy, component_first)
        assert quaternion_coefficient_error(actual_xyzw, sample["quaternion_xyzw"]) <= TOLERANCE, (
            f"{backend_name}: glTF LINEAR sample at {t}s"
        )


def test_blender_importer_axis_mapping_and_quaternion_animation_profiles_are_frozen():
    basis = _basis_blender()
    source_times = _accessor(0).reshape(-1)
    source_keys = _source_key_values()
    target_keys_xyzw = np.stack([_map_xyzw_to_blender(key, basis) for key in source_keys])
    target_keys = target_keys_xyzw[:, (3, 0, 1, 2)]
    rows = {row["interpolation"]: row for row in BLENDER["observations"]["animations"]}
    assert set(rows) == {"STEP", "LINEAR", "CUBICSPLINE"}
    assert np.max(np.abs(np.asarray(rows["LINEAR"]["key_times_seconds_from_imported_fcurves"]) - source_times)) < 3e-8

    for sample in rows["STEP"]["samples"]:
        if sample["sample"] in {"before-key-1-1e-06", "before-key-1"}:
            continue
        index = min(max(int(np.searchsorted(source_times, sample["time_seconds"], side="right") - 1), 0), 2)
        actual = sample["quaternion_wxyz"]
        assert quaternion_coefficient_error(actual, target_keys[index]) <= TOLERANCE
    frame_edge = next(
        sample for sample in rows["STEP"]["samples"]
        if sample["sample"] == "before-key-1-1e-06"
    )
    assert frame_edge["time_seconds"] < source_times[1]
    assert quaternion_coefficient_error(frame_edge["quaternion_wxyz"], target_keys[1]) <= TOLERANCE

    linear = rows["LINEAR"]
    assert {key["interpolation"] for curve in linear["fcurves"] for key in curve["keys"]} == {"LINEAR"}
    midpoint = next(sample for sample in linear["samples"] if sample["sample"] == "midpoint-0-1")
    alpha = (midpoint["time_seconds"] - source_times[0]) / (source_times[1] - source_times[0])
    component_linear = (1.0 - alpha) * target_keys[0] + alpha * target_keys[1]
    np.testing.assert_allclose(midpoint["quaternion_wxyz"], component_linear, rtol=0.0, atol=TOLERANCE)
    assert midpoint["quaternion_length"] == pytest.approx(0.95203203, abs=2e-8)

    cubic = rows["CUBICSPLINE"]
    assert {key["interpolation"] for curve in cubic["fcurves"] for key in curve["keys"]} == {"BEZIER"}
    assert all(key["handle_left_type"] if "handle_left_type" in key else True for curve in cubic["fcurves"] for key in curve["keys"])
    cubic_midpoint = next(sample for sample in cubic["samples"] if sample["sample"] == "midpoint-0-1")
    expected_source = _cubic_value(source_times, _accessor(3).reshape((3, 3, 4)), cubic_midpoint["time_seconds"])
    expected_target_xyzw = _map_xyzw_to_blender(expected_source, basis)
    expected_target = expected_target_xyzw[[3, 0, 1, 2]]
    assert quaternion_coefficient_error(cubic_midpoint["quaternion_wxyz"], expected_target) > 0.08
    assert cubic_midpoint["quaternion_length"] == pytest.approx(0.97526056, abs=2e-8)
    assert {key["handle_left_type"] for curve in cubic["fcurves"] for key in curve["keys"]} == {"AUTO"}
    assert {key["handle_right_type"] for curve in cubic["fcurves"] for key in curve["keys"]} == {"AUTO"}
