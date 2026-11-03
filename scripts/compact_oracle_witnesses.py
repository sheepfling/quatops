"""One-time reduction of broad discovery captures to reviewed witnesses.

The original file digest is retained in provenance before the successful
discovery rows are removed. Native capture programs remain the replay source.
"""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    ROOT / "tests/golden/oracles/threejs/0.187.0/native_rotation.json",
    ROOT / "tests/golden/oracles/mujoco/3.17.0/native_rotation.json",
    ROOT / "tests/golden/oracles/spiceypy/8.3.0/native_rotation.json",
    ROOT / "tests/golden/transports/openusd/26.11/native_rotation.json",
)


def choose(rows, predicate, key=lambda row: row.get("case_id")):
    result = {}
    for row in rows:
        if predicate(row):
            result.setdefault(key(row), row)
    return list(result.values())


def compact(data, path):
    obs = data["observations"]
    if path.parts[-3] == "openusd":
        keep_algebra = {"algebra-0", "algebra-6", "algebra-7", "rotation-identity",
                        "rotation-random-000", "rotation-random-000-antipodal",
                        "invalid-rotation-zero", "invalid-rotation-nonunit", "invalid-rotation-tiny"}
        obs["quaternion_algebra"] = choose(obs["quaternion_algebra"], lambda r: r["case_id"] in keep_algebra)
        keep_angles = {"axis-angle-0-00", "axis-angle-1-05", "axis-angle-2-05", "axis-angle-3-05", "axis-angle-3-14"}
        obs["axis_angle_and_rotation"] = choose(obs["axis_angle_and_rotation"], lambda r: r["case_id"] in keep_angles)
        obs["slerp"] = choose(obs["slerp"], lambda r: r["t"] == 0.5)
        data["provenance"]["full_discovery_capture_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        data["provenance"]["retention_policy"] = "selected native witnesses; broad successful sweeps discarded"
        data["counts"] = {"quaternion_algebra": len(obs["quaternion_algebra"]),
                          "axis_angle_and_rotation": len(obs["axis_angle_and_rotation"]),
                          "slerp": len(obs["slerp"]), "stage_prims": data["counts"]["stage_prims"]}
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return
    if path.parts[-3] == "threejs":
        obs["euler_orders"] = choose(
            obs["euler_orders"], lambda r: r["case_id"] != "euler-identity",
            key=lambda r: r["order"],
        )
    elif path.parts[-3] == "mujoco":
        obs["euler_orders"] = choose(
            obs["euler_orders"], lambda r: r["case_id"] != "euler-identity",
            key=lambda r: (r["order_label"], r["sequence_case"]),
        )
    else:
        obs["euler_orders"] = choose(
            obs["euler_orders"], lambda r: r["case_id"] != "euler-identity",
            key=lambda r: r["order"],
        )

    # One discriminating boundary per order, plus the measured Three.js ZYX
    # near-lock regression. Preserve raw bits attached to each chosen input.
    boundary_rows = obs["euler_boundaries"]
    if path.parts[-3] == "threejs":
        retained = choose(boundary_rows, lambda r: r["case_id"] == "boundary-offset-ZYX-+1-+1e-04")
        for row in boundary_rows:
            if not any(x["order"] == row["order"] for x in retained):
                retained.append(row)
    else:
        field = "order_label" if path.parts[-3] == "mujoco" else "order"
        retained = choose(boundary_rows, lambda r: True, key=lambda r: r[field])
    obs["euler_boundaries"] = retained

    if "axis_angle" in obs:
        obs["axis_angle"] = choose(obs["axis_angle"], lambda r: r["case_id"] in {
            "axis-angle-0-00", "axis-angle-1-05", "axis-angle-2-05", "axis-angle-3-05", "axis-angle-3-14"
        })
    if "quaternion_normalization" in obs:
        keep = {"algebra-0", "algebra-6", "algebra-7", "rotation-identity",
                "rotation-random-000", "rotation-random-000-antipodal",
                "invalid-rotation-zero", "invalid-rotation-nonunit", "invalid-rotation-tiny"}
        obs["quaternion_normalization"] = choose(obs["quaternion_normalization"], lambda r: r["case_id"] in keep)
    if "proper_matrices" in obs:
        obs["proper_matrices"] = choose(obs["proper_matrices"], lambda r: r["case_id"] in {
            "matrix-identity", "matrix-x90", "matrix-mixed-rational"
        })
    if "independent_matrices" in obs:
        obs["independent_matrices"] = choose(obs["independent_matrices"], lambda r: r["case_id"] in {
            "matrix-identity", "matrix-x90", "matrix-mixed-rational", "matrix-shear"
        })
    if "two_vector_alignment" in obs:
        obs["two_vector_alignment"] = obs["two_vector_alignment"][:2]
    if "integration" in obs:
        rows = obs["integration"]
        obs["integration"] = [rows[0], rows[5], rows[17], rows[-1]]
    if "sub_quat" in obs:
        obs["sub_quat"] = [obs["sub_quat"][0], obs["sub_quat"][-1]]
    if "qdq2av" in obs:
        obs["qdq2av"] = [obs["qdq2av"][0], obs["qdq2av"][-1]]

    data["provenance"]["full_discovery_capture_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    data["provenance"]["retention_policy"] = "selected native witnesses; broad successful sweeps discarded"
    data["counts"] = {name: len(rows) for name, rows in obs.items() if isinstance(rows, list)}
    if "integration" in data["counts"]:
        data["counts"]["integration_and_derivatives"] = data["counts"].pop("integration")
    if "sub_quat" in data["counts"]:
        data["counts"]["quaternion_differences"] = data["counts"].pop("sub_quat")
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    for fixture_path in FILES:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        if "full_discovery_capture_sha256" in payload.get("provenance", {}):
            print(f"already compact: {fixture_path.relative_to(ROOT)}")
            continue
        compact(payload, fixture_path)
        print(f"compacted {fixture_path.relative_to(ROOT)}")
