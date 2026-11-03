"""Independent SciPy/reference checks for local Cesium and Three.js captures."""
from __future__ import annotations

from itertools import permutations
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.spatial.transform import Rotation, Slerp

from shared.capture_io import ROOT
from tools.verify_results import expected_world


PROJECT = ROOT.parents[1]
CESIUM = PROJECT / "tests/golden/oracles/cesium/1.124.0"
THREE = PROJECT / "tests/golden/transports/gltf/2.0.0/threejs/0.187.0/wave2"
VALIDATOR = PROJECT / "tests/golden/transports/gltf/2.0.0/khronos-validator/2.0.0-dev.3.10/wave2"


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def value(blob: Any) -> np.ndarray:
    if isinstance(blob, dict) and "values" in blob:
        blob = blob["values"]
    return np.asarray(blob, dtype=np.float64)


def max_error(lhs: Any, rhs: Any) -> float:
    return float(np.max(np.abs(value(lhs) - value(rhs))))


def orientation_error(lhs_xyzw: Any, rhs_xyzw: Any) -> float:
    lhs = Rotation.from_quat(value(lhs_xyzw)).as_matrix()
    rhs = Rotation.from_quat(value(rhs_xyzw)).as_matrix()
    return float(np.max(np.abs(lhs - rhs)))


def hamilton_xyzw(lhs_xyzw: Any, rhs_xyzw: Any) -> np.ndarray:
    x1, y1, z1, w1 = value(lhs_xyzw)
    x2, y2, z2, w2 = value(rhs_xyzw)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def local_hpr_matrix(hpr: Any) -> np.ndarray:
    heading, pitch, roll = value(hpr)
    # Rz(-heading) Ry(-pitch) Rx(roll), independently evaluated by SciPy.
    return Rotation.from_euler("ZYX", [-heading, -pitch, roll]).as_matrix()


def verify_cesium() -> dict[str, Any]:
    observations = rows(CESIUM / "observations.jsonl")
    corpus = json.loads((ROOT / "corpus/orientation_inputs_v1.json").read_text())
    geographic = json.loads((ROOT / "corpus/cesium_geographic_inputs.json").read_text())
    errors: dict[str, list[float]] = {
        "local_hpr_matrix": [],
        "local_hpr_quaternion_matrix": [],
        "readback_hpr_matrix": [],
        "basis_action": [],
        "matrix_to_quaternion": [],
        "slerp": [],
        "hamilton_product": [],
        "geographic_composition": [],
        "geographic_quaternion_matrix": [],
        "geographic_origin": [],
        "geographic_enu_vs_geodetic_latitude": [],
        "geographic_enu_vs_surface_normal": [],
        "geographic_up_vs_radii_normal": [],
        "geographic_point": [],
        "geographic_direction": [],
        "geographic_point_roundtrip": [],
    }

    vectors = np.asarray(corpus["vectors_xyz"], dtype=np.float64)
    for record in observations:
        inputs, outputs = record["native_inputs"], record["native_outputs"]
        operation = record["operation_id"]
        if operation == "local-hpr":
            hpr = np.array([value(inputs[name]).item() for name in ("heading", "pitch", "roll")])
            expected = local_hpr_matrix(hpr)
            matrix = value(outputs["matrix_rows"])
            errors["local_hpr_matrix"].append(max_error(matrix, expected))
            errors["local_hpr_quaternion_matrix"].append(
                max_error(Rotation.from_quat(value(outputs["quaternion_xyzw"])).as_matrix(), expected)
            )
            back = local_hpr_matrix(outputs["readback_hpr"])
            errors["readback_hpr_matrix"].append(max_error(back, expected))
            expected_images = Rotation.from_matrix(expected).apply(vectors)
            errors["basis_action"].append(max_error(outputs["basis_and_probe_images"], expected_images))
        elif operation == "matrix-to-quaternion":
            expected = Rotation.from_matrix(value(inputs["matrix_rows"]))
            errors["matrix_to_quaternion"].append(
                orientation_error(outputs["quaternion_xyzw"], expected.as_quat())
            )
        elif operation.startswith("slerp-"):
            t = float(operation.removeprefix("slerp-"))
            a = value(inputs["a_xyzw"])
            b = value(inputs["b_xyzw"])
            expected = Slerp([0.0, 1.0], Rotation.from_quat([a, b]))([t]).as_quat()[0]
            errors["slerp"].append(orientation_error(outputs["quaternion_xyzw"], expected))
        elif operation.startswith("multiply-"):
            expected = hamilton_xyzw(inputs["lhs_xyzw"], inputs["rhs_xyzw"])
            errors["hamilton_product"].append(max_error(outputs["product_xyzw"], expected))
        elif operation.startswith("geographic-hpr-"):
            hpr = value(inputs["hpr_radians"])
            local = local_hpr_matrix(hpr)
            enu = value(outputs["enu_to_ecef_matrix_rows"])
            world = value(outputs["model_to_ecef_matrix_rows"])
            rotation = world[:3, :3]
            origin = value(outputs["origin_ecef_m"])
            location = next(item for item in geographic["locations"] if item["case_id"] == record["case_id"])
            lon = np.deg2rad(location["longitude_degrees"])
            lat = np.deg2rad(location["latitude_degrees"])
            height = location["height_m"]
            semi_major = 6_378_137.0
            flattening = 1.0 / 298.257223563
            eccentricity2 = flattening * (2.0 - flattening)
            radius = semi_major / np.sqrt(1.0 - eccentricity2 * np.sin(lat) ** 2)
            expected_origin = np.array(
                [
                    (radius + height) * np.cos(lat) * np.cos(lon),
                    (radius + height) * np.cos(lat) * np.sin(lon),
                    (radius * (1.0 - eccentricity2) + height) * np.sin(lat),
                ]
            )
            errors["geographic_composition"].append(max_error(rotation, enu[:3, :3] @ local))
            errors["geographic_quaternion_matrix"].append(
                max_error(Rotation.from_quat(value(outputs["quaternion_ecef_xyzw"])).as_matrix(), rotation)
            )
            errors["geographic_origin"].append(max_error(origin, expected_origin))
            radii = np.array([semi_major, semi_major, semi_major * (1.0 - flattening)])
            raw_normal = origin / (radii * radii)
            expected_up = raw_normal / np.linalg.norm(raw_normal)
            actual_up = enu[:3, 2]
            errors["geographic_up_vs_radii_normal"].append(max_error(actual_up, expected_up))
            # This latitude-defined frame is a separate comparison diagnostic.
            # Cesium's native normal is measured from the ECEF position.
            if abs(abs(location["latitude_degrees"]) - 90.0) > 1e-12:
                expected_enu = np.array(
                    [
                        [-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
                        [np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
                        [0.0, np.cos(lat), np.sin(lat)],
                    ]
                )
                errors["geographic_enu_vs_geodetic_latitude"].append(
                    max_error(enu[:3, :3], expected_enu)
                )
                east = np.array([-np.sin(lon), np.cos(lon), 0.0])
                north = np.cross(expected_up, east)
                north /= np.linalg.norm(north)
                surface_normal_enu = np.column_stack((east, north, expected_up))
                errors["geographic_enu_vs_surface_normal"].append(
                    max_error(enu[:3, :3], surface_normal_enu)
                )
            point = value(inputs["local_point_xyz_m"])
            direction = value(inputs["local_direction_xyz"])
            errors["geographic_point"].append(
                max_error(outputs["world_point_xyz_m"], rotation @ point + origin)
            )
            errors["geographic_direction"].append(
                max_error(outputs["world_direction_xyz"], rotation @ direction)
            )
            errors["geographic_point_roundtrip"].append(
                max_error(outputs["local_point_readback"], point)
            )

    return {
        "evidence_kind": "derived_reference",
        "native_observation_count": len(observations),
        "scipy_version": scipy.__version__,
        "quatops_executed": False,
        "max_abs_errors": {name: max(items) if items else None for name, items in errors.items()},
        "counts": {name: len(items) for name, items in errors.items()},
    }


def _unordered_world_error(expected: list[np.ndarray], actual: list[np.ndarray]) -> float:
    if len(expected) != len(actual):
        raise AssertionError(f"glTF node count mismatch: expected {len(expected)}, got {len(actual)}")
    return min(
        max(float(np.max(np.abs(found - target))) for found, target in zip(actual, pairing))
        for pairing in permutations(expected)
    )


def verify_three() -> dict[str, Any]:
    observations = rows(THREE / "observations.jsonl")
    corpus = json.loads((ROOT / "corpus/orientation_inputs_v1.json").read_text())
    euler_errors: list[float] = []
    gltf_errors: list[float] = []
    animation_values: dict[tuple[str, float], np.ndarray] = {}
    for record in observations:
        inputs, outputs = record["native_inputs"], record["native_outputs"]
        operation = record["operation_id"]
        if operation.startswith("euler-"):
            order = inputs["order"]
            xyz = value(inputs["axis_labelled_xyz_radians"])
            angles = [xyz["XYZ".index(axis)] for axis in order]
            expected = Rotation.from_euler(order, angles)
            euler_errors.append(max_error(value(outputs["matrix_rows"])[:3, :3], expected.as_matrix()))
            euler_errors.append(orientation_error(outputs["quaternion_xyzw"], expected.as_quat()))
        elif operation == "gltf-static" or operation.startswith("gltf-animation-"):
            asset = inputs["asset"]
            sample_time = value(inputs["time_seconds"]).item() if "time_seconds" in inputs else None
            expected = expected_world(ROOT / "assets/gltf" / asset, sample_time)
            actual = [
                value(item["world_matrix_rows"])
                for item in outputs["objects"]
                if item["name"] != "Parent"
            ]
            gltf_errors.append(_unordered_world_error(expected, actual))
            if sample_time is not None and asset.startswith("cubic"):
                animated = next(item for item in outputs["objects"] if item["name"] != "Parent")
                animation_values[(asset, sample_time)] = value(animated["world_matrix_rows"])

    global_sign_errors: list[float] = []
    local_sign_differences: list[float] = []
    for (asset, time), matrix in animation_values.items():
        if asset.startswith("cubic_all_negated"):
            peer = ("cubic." + asset.rsplit(".", 1)[1], time)
            global_sign_errors.append(max_error(matrix, animation_values[peer]))
        elif asset.startswith("cubic_middle_negated"):
            peer = ("cubic." + asset.rsplit(".", 1)[1], time)
            local_sign_differences.append(max_error(matrix, animation_values[peer]))

    validation = [json.loads(path.read_text()) for path in sorted(VALIDATOR.glob("*.json"))]
    issue_counts = {
        key: sum(item["report"]["issues"][key] for item in validation)
        for key in ("numErrors", "numWarnings", "numInfos", "numHints")
    }
    return {
        "evidence_kind": "derived_reference",
        "native_observation_count": len(observations),
        "scipy_version": scipy.__version__,
        "quatops_executed": False,
        "three_euler_case_count": len(euler_errors) // 2,
        "three_euler_matrix_and_orientation_max_abs_error": max(euler_errors),
        "gltf_loader_sample_count": len(gltf_errors),
        "gltf_loader_max_abs_matrix_error": max(gltf_errors),
        "global_cubic_sign_max_abs_matrix_error": max(global_sign_errors),
        "single_key_sign_change_max_abs_matrix_difference": max(local_sign_differences),
        "validator_asset_count": len(validation),
        "validator_issue_counts": issue_counts,
    }


def main() -> None:
    report = {
        "evidence_kind": "derived_reference",
        "references": {"scipy": scipy.__version__, "gltf": "2.0 independent TRS and normalized rotation Hermite evaluation"},
        "quatops_executed": False,
        "cesium": verify_cesium(),
        "threejs_and_gltf": verify_three(),
    }
    output = ROOT / "verification/local_native_comparison.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
