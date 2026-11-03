"""One-time reduction of broad discovery captures to reviewed witnesses.

The original file digest is retained in provenance before the successful
discovery rows are removed. Native capture programs remain the replay source.
"""

import hashlib
import json
from pathlib import Path

from evidence_json import read_evidence_document

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    ROOT / "tests/golden/oracles/threejs/0.187.0/native_rotation.json",
    ROOT / "tests/golden/oracles/mujoco/3.17.0/native_rotation.json",
    ROOT / "tests/golden/oracles/spiceypy/8.3.0/native_rotation.json",
)


def choose(rows, predicate, key=lambda row: row.get("case_id")):
    result = {}
    for row in rows:
        if predicate(row):
            result.setdefault(key(row), row)
    return list(result.values())


def compact_threejs(data, path):
    obs = data["observations"]
    near_lock = choose(
        obs["euler_boundaries"],
        lambda row: row["case_id"] == "boundary-offset-ZYX-+1-+1e-04",
    )
    if len(near_lock) != 1:
        raise ValueError("expected the minimized Three.js near-lock regression")
    regression_fields = (
        "case_id",
        "order",
        "input_radians_xyz",
        "input_source",
        "euler_quaternion_xyzw",
        "quaternion_readback_roundtrip_xyzw",
    )
    obs["euler_regressions"] = [{key: near_lock[0][key] for key in regression_fields}]
    for field in ("euler_orders", "euler_boundaries", "euler_reorder"):
        obs.pop(field, None)
    keep_axis_angles = {
        "axis-angle-0-00",
        "axis-angle-1-05",
        "axis-angle-2-05",
        "axis-angle-3-05",
        "axis-angle-3-14",
    }
    obs["axis_angle"] = choose(
        obs["axis_angle"], lambda row: row["case_id"] in keep_axis_angles
    )
    keep_quaternions = {
        "algebra-0",
        "algebra-6",
        "algebra-7",
        "rotation-identity",
        "rotation-random-000",
        "rotation-random-000-antipodal",
        "invalid-rotation-zero",
        "invalid-rotation-nonunit",
        "invalid-rotation-tiny",
    }
    obs["quaternion_normalization"] = choose(
        obs["quaternion_normalization"],
        lambda row: row["case_id"] in keep_quaternions,
    )
    keep_matrices = {"matrix-identity", "matrix-x90", "matrix-mixed-rational"}
    obs["independent_matrices"] = choose(
        obs["independent_matrices"],
        lambda row: row["case_id"] in keep_matrices,
    )
    obs["two_vector_alignment"] = obs["two_vector_alignment"][:2]
    obs["axis_angle"] = [
        {
            key: row[key]
            for key in (
                "case_id",
                "input_axis_xyz",
                "input_angle_radians",
                "quaternion_xyzw",
            )
        }
        for row in obs["axis_angle"]
    ]
    for row in obs["quaternion_normalization"]:
        for field in ("input_xyzw", "length", "length_squared"):
            row.pop(field, None)
    for row in obs["shortest_slerp"]:
        keep_weights = (
            {-0.5, 0.5, 1.5} if row["case_id"] == "interp-quarter-turn" else {0.5}
        )
        row["samples"] = [
            sample for sample in row["samples"] if sample["t"] in keep_weights
        ]
    obs["independent_matrices"] = [
        {
            key: value
            for key, value in row.items()
            if key != "quaternion_rotated_positive_basis_vectors_xyz"
        }
        for row in obs["independent_matrices"]
        if row["domain"] == "proper_rotation"
    ]
    obs.pop("object3d_hierarchy", None)
    data["counts"] = {
        name: len(rows) for name, rows in obs.items() if isinstance(rows, list)
    }
    provenance = data["provenance"]
    provenance.setdefault(
        "full_discovery_capture_sha256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    provenance["retention_revision"] = 3
    provenance["retention_policy"] = (
        "selected API witnesses; redundant matrix views/actions and hierarchy "
        "duplicates removed; broad successful sweeps discarded"
    )
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def compact(data, path):
    obs = data["observations"]
    if path.parts[-3] == "threejs":
        compact_threejs(data, path)
        return
    if path.parts[-3] == "mujoco":
        obs["euler_orders"] = choose(
            obs["euler_orders"],
            lambda r: r["case_id"] != "euler-identity",
            key=lambda r: (r["order_label"], r["sequence_case"]),
        )
    else:
        obs["euler_orders"] = choose(
            obs["euler_orders"],
            lambda r: r["case_id"] != "euler-identity",
            key=lambda r: r["order"],
        )

    # One discriminating boundary per order, plus the measured Three.js ZYX
    # near-lock regression. Preserve raw bits attached to each chosen input.
    boundary_rows = obs["euler_boundaries"]
    if path.parts[-3] == "threejs":
        retained = choose(
            boundary_rows, lambda r: r["case_id"] == "boundary-offset-ZYX-+1-+1e-04"
        )
        for row in boundary_rows:
            if not any(x["order"] == row["order"] for x in retained):
                retained.append(row)
    else:
        field = "order_label" if path.parts[-3] == "mujoco" else "order"
        retained = choose(boundary_rows, lambda r: True, key=lambda r: r[field])
    obs["euler_boundaries"] = retained

    if "axis_angle" in obs:
        obs["axis_angle"] = choose(
            obs["axis_angle"],
            lambda r: (
                r["case_id"]
                in {
                    "axis-angle-0-00",
                    "axis-angle-1-05",
                    "axis-angle-2-05",
                    "axis-angle-3-05",
                    "axis-angle-3-14",
                }
            ),
        )
    if "quaternion_normalization" in obs:
        keep = {
            "algebra-0",
            "algebra-6",
            "algebra-7",
            "rotation-identity",
            "rotation-random-000",
            "rotation-random-000-antipodal",
            "invalid-rotation-zero",
            "invalid-rotation-nonunit",
            "invalid-rotation-tiny",
        }
        obs["quaternion_normalization"] = choose(
            obs["quaternion_normalization"], lambda r: r["case_id"] in keep
        )
    if "proper_matrices" in obs:
        obs["proper_matrices"] = choose(
            obs["proper_matrices"],
            lambda r: (
                r["case_id"]
                in {"matrix-identity", "matrix-x90", "matrix-mixed-rational"}
            ),
        )
    if "independent_matrices" in obs:
        obs["independent_matrices"] = choose(
            obs["independent_matrices"],
            lambda r: (
                r["case_id"]
                in {
                    "matrix-identity",
                    "matrix-x90",
                    "matrix-mixed-rational",
                    "matrix-shear",
                }
            ),
        )
    if "two_vector_alignment" in obs:
        obs["two_vector_alignment"] = obs["two_vector_alignment"][:2]
    if "integration" in obs:
        rows = obs["integration"]
        obs["integration"] = [rows[0], rows[5], rows[17], rows[-1]]
    if "sub_quat" in obs:
        obs["sub_quat"] = [obs["sub_quat"][0], obs["sub_quat"][-1]]
    if "qdq2av" in obs:
        obs["qdq2av"] = [obs["qdq2av"][0], obs["qdq2av"][-1]]

    data["provenance"]["full_discovery_capture_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    data["provenance"]["retention_policy"] = (
        "selected native witnesses; broad successful sweeps discarded"
    )
    data["counts"] = {
        name: len(rows) for name, rows in obs.items() if isinstance(rows, list)
    }
    if "integration" in data["counts"]:
        data["counts"]["integration_and_derivatives"] = data["counts"].pop(
            "integration"
        )
    if "sub_quat" in data["counts"]:
        data["counts"]["quaternion_differences"] = data["counts"].pop("sub_quat")
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    for fixture_path in FILES:
        payload = read_evidence_document(fixture_path)
        provenance = payload.get("provenance", {})
        if "full_discovery_capture_sha256" in provenance and (
            fixture_path.parts[-3] != "threejs"
            or provenance.get("retention_revision") == 3
        ):
            print(f"already compact: {fixture_path.relative_to(ROOT)}")
            continue
        compact(payload, fixture_path)
        print(f"compacted {fixture_path.relative_to(ROOT)}")
