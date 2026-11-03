"""Probe Unreal's reflected Rotator/Quat API and preserve its raw values.

This deliberately uses Unreal's Python wrappers only. It does not import
QuatOps, SciPy, or another rotation implementation.
"""

import json
import sys
from pathlib import Path

import unreal

CASES = (
    {"case_id": "identity", "pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    {"case_id": "pitch_positive_90", "pitch": 90.0, "yaw": 0.0, "roll": 0.0},
    {"case_id": "yaw_positive_90", "pitch": 0.0, "yaw": 90.0, "roll": 0.0},
    {"case_id": "roll_positive_90", "pitch": 0.0, "yaw": 0.0, "roll": 90.0},
    {"case_id": "mixed_positive", "pitch": 31.41, "yaw": -49.16, "roll": 12.34},
    {"case_id": "mixed_negative", "pitch": -50.0, "yaw": 28.0, "roll": 39.0},
    {"case_id": "near_gimbal", "pitch": 89.9, "yaw": 27.0, "roll": -41.0},
    {"case_id": "gimbal", "pitch": 90.0, "yaw": 27.0, "roll": -41.0},
    {"case_id": "pitch_over_90", "pitch": 105.0, "yaw": 23.0, "roll": -17.0},
    {"case_id": "yaw_positive_180", "pitch": 0.0, "yaw": 180.0, "roll": 0.0},
    {"case_id": "yaw_negative_180", "pitch": 0.0, "yaw": -180.0, "roll": 0.0},
)


def components(value):
    return [float(value.x), float(value.y), float(value.z), float(value.w)]


def vector_components(value):
    return [float(value.x), float(value.y), float(value.z)]


def rotator_fields(value):
    return {
        "pitch": float(value.pitch),
        "yaw": float(value.yaw),
        "roll": float(value.roll),
    }


def capture():
    rotations = {}
    cases = []
    for spec in CASES:
        # Keep named arguments here: Unreal's positional constructor order is
        # easy to confuse with the common roll/pitch/yaw presentation order.
        rotator = unreal.Rotator(
            pitch=spec["pitch"], yaw=spec["yaw"], roll=spec["roll"]
        )
        quat = rotator.quaternion()
        rotations[spec["case_id"]] = quat
        axes = (
            unreal.Vector(1.0, 0.0, 0.0),
            unreal.Vector(0.0, 1.0, 0.0),
            unreal.Vector(0.0, 0.0, 1.0),
        )
        rotated_axes = [vector_components(quat.rotate_vector(axis)) for axis in axes]
        cases.append(
            {
                "case_id": spec["case_id"],
                "input_rotator_degrees": {
                    "pitch": spec["pitch"],
                    "yaw": spec["yaw"],
                    "roll": spec["roll"],
                },
                "python_type": type(quat).__name__,
                "rotator_quaternion_xyzw": components(quat),
                "quat_rotator_readback_degrees": rotator_fields(quat.rotator()),
                "quat_euler_readback_degrees_xyz": vector_components(quat.euler()),
                "rotated_positive_basis_vectors_xyz": rotated_axes,
            }
        )

    products = []
    for lhs_id, rhs_id in (
        ("pitch_positive_90", "yaw_positive_90"),
        ("yaw_positive_90", "roll_positive_90"),
        ("mixed_positive", "mixed_negative"),
    ):
        lhs = rotations[lhs_id]
        rhs = rotations[rhs_id]
        products.append(
            {
                "lhs_case_id": lhs_id,
                "rhs_case_id": rhs_id,
                "lhs_xyzw": components(lhs),
                "rhs_xyzw": components(rhs),
                "product_xyzw": components(lhs * rhs),
            }
        )

    return {
        "capture_method": "Unreal Editor Python commandlet",
        "engine_version": unreal.SystemLibrary.get_engine_version(),
        "platform": sys.platform,
        "python_version": sys.version,
        "python_quat_type": type(rotations["identity"]).__name__,
        "cases": cases,
        "multiplication_cases": products,
    }


output_dir = Path(unreal.Paths.project_saved_dir())
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "quatops_unreal_python_probe.json"
try:
    result = capture()
    result["status"] = "captured"
except Exception as error:  # noqa: BLE001 - record binding failures as capture evidence.
    result = {
        "status": "python_binding_unavailable_or_incomplete",
        "capture_method": "Unreal Editor Python commandlet",
        "engine_version": unreal.SystemLibrary.get_engine_version(),
        "platform": sys.platform,
        "python_version": sys.version,
        "error_type": type(error).__name__,
        "error": str(error),
    }
output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
unreal.log(f"QuatOps Unreal Python probe wrote {output_path}")
if result["status"] != "captured":
    raise RuntimeError(result["error"])
