"""Capture CSPICE rotation routines through the pinned SpiceyPy binding."""

# The core typecheck uses the versioned API stub; the binary module source is
# available only in this capture's pinned SpiceyPy environment.
# pyright: reportMissingModuleSource=false

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import spiceypy as spice

ORDERS = ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX")
AXIS_NUMBER = {"X": 1, "Y": 2, "Z": 3}
HERE = Path(__file__).parent


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        default=HERE.parents[1] / "shared" / "orientation_inputs_v1.json",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=HERE.parents[3]
        / "tests"
        / "golden"
        / "oracles"
        / "spiceypy"
        / "8.3.0"
        / "native_rotation.json",
        type=Path,
    )
    return parser.parse_args()


def euler_call(case_id, xyz, order, source):
    angles = {axis: float(xyz["XYZ".index(axis)]) for axis in "XYZ"}
    angle1, angle2, angle3 = (angles[axis] for axis in order)
    axis1, axis2, axis3 = (AXIS_NUMBER[axis] for axis in order)
    matrix = np.asarray(
        spice.eul2m(angle3, angle2, angle1, axis3, axis2, axis1),
        dtype=float,
    )
    read_angle3, read_angle2, read_angle1 = spice.m2eul(matrix, axis3, axis2, axis1)
    readback = {
        order[0]: float(read_angle1),
        order[1]: float(read_angle2),
        order[2]: float(read_angle3),
    }
    q = np.asarray(spice.m2q(matrix), dtype=float)
    return {
        "case_id": case_id,
        "order": order,
        "input_radians_xyz": [angles[axis] for axis in "XYZ"],
        "input_source": source,
        "axis1_angle1": {"axis": order[0], "angle_radians": angle1},
        "axis2_angle2": {"axis": order[1], "angle_radians": angle2},
        "axis3_angle3": {"axis": order[2], "angle_radians": angle3},
        "eul2m_matrix_rows": matrix.tolist(),
        "m2eul_readback_radians_xyz": [readback[axis] for axis in "XYZ"],
        "m2q_wxyz": q.tolist(),
        "q2m_from_m2q_rows": np.asarray(spice.q2m(q), dtype=float).tolist(),
    }


def explicit_hamilton_product(lhs, rhs):
    aw, ax, ay, az = lhs
    bw, bx, by, bz = rhs
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def quaternion_derivative(q, omega, multiply_side):
    pure = [0.0, *map(float, omega)]
    if multiply_side == "q_then_omega":
        product = explicit_hamilton_product(q, pure)
    else:
        product = explicit_hamilton_product(pure, q)
    return [0.5 * value for value in product]


def main():
    options = parse_args()
    corpus_bytes = options.corpus.read_bytes()
    corpus = json.loads(corpus_bytes.decode("utf-8"))

    eulers = []
    for case in corpus["euler_inputs"]:
        source = {
            "units": case["units"],
            "numeric_format": case["numeric_format"],
            "actual_bits_hex": case["actual_bits_hex"],
        }
        eulers.extend(
            euler_call(case["case_id"], case["xyz"], order, source) for order in ORDERS
        )
    euler_boundaries = []
    for case in corpus["euler_boundaries"]:
        xyz = np.asarray(case["xyz"], dtype=float)
        if case["units"] == "degrees":
            xyz = np.deg2rad(xyz)
        euler_boundaries.append(
            euler_call(
                case["case_id"],
                xyz.tolist(),
                case["order_label"],
                {
                    "units": case["units"],
                    "input_xyz": case["xyz"],
                    "numeric_format": case["numeric_format"],
                    "actual_bits_hex": case["actual_bits_hex"],
                },
            )
        )

    products = []
    for case in corpus["multiplication_pairs"]:
        a = np.asarray(case["a_wxyz"], dtype=float)
        b = np.asarray(case["b_wxyz"], dtype=float)
        products.append(
            {
                "case_id": case["case_id"],
                "a_input_wxyz": a.tolist(),
                "b_input_wxyz": b.tolist(),
                "qxq_a_b_wxyz": np.asarray(spice.qxq(a, b), dtype=float).tolist(),
                "qxq_b_a_wxyz": np.asarray(spice.qxq(b, a), dtype=float).tolist(),
                "explicit_hamilton_a_b_wxyz": explicit_hamilton_product(a, b),
                "q2m_a_rows": np.asarray(spice.q2m(a), dtype=float).tolist(),
                "q2m_b_rows": np.asarray(spice.q2m(b), dtype=float).tolist(),
                "q2m_product_rows": np.asarray(
                    spice.q2m(spice.qxq(a, b)), dtype=float
                ).tolist(),
            }
        )

    axis_angle = []
    for case in corpus["axis_angle_inputs"]:
        matrix = np.asarray(
            spice.axisar(case["axis_xyz"], case["angle_radians"]), dtype=float
        )
        q = np.asarray(spice.m2q(matrix), dtype=float)
        axis, angle = spice.raxisa(matrix)
        axis_angle.append(
            {
                "case_id": case["case_id"],
                "input_axis_xyz": case["axis_xyz"],
                "input_angle_radians": case["angle_radians"],
                "axisar_matrix_rows": matrix.tolist(),
                "axisar_wxyz": q.tolist(),
                "m2q_from_axisar_wxyz": q.tolist(),
                "raxisa_axis_xyz": np.asarray(axis, dtype=float).tolist(),
                "raxisa_angle_radians": float(angle),
            }
        )

    proper_matrices = []
    for case in corpus["matrix_inputs"]:
        if case["domain"] != "proper_rotation":
            continue
        matrix = np.asarray(case["rows"], dtype=float)
        q = np.asarray(spice.m2q(matrix), dtype=float)
        native_matrix = np.asarray(spice.q2m(q), dtype=float)
        proper_matrices.append(
            {
                "case_id": case["case_id"],
                "input_matrix_rows": matrix.tolist(),
                "m2q_wxyz": q.tolist(),
                "q2m_rows": native_matrix.tolist(),
                "q2m_positive_basis_images_xyz": [
                    (native_matrix @ axis).tolist() for axis in np.eye(3)
                ],
            }
        )

    derivatives = []
    for case in corpus["integration_inputs"]:
        q = list(map(float, case["initial_wxyz"]))
        row = {
            "case_id": case["case_id"],
            "input_q_wxyz": q,
            "input_omega_xyz": case["omega_xyz"],
            "input_omega_frame": case["omega_frame"],
            "dt_seconds": case["dt_seconds"],
            "qdq2av_probes": [],
        }
        for side in ("q_then_omega", "omega_then_q"):
            qdot = quaternion_derivative(q, case["omega_xyz"], side)
            native = np.asarray(
                spice.qdq2av(
                    np.asarray(q, dtype=np.float64),
                    np.asarray(qdot, dtype=np.float64),
                ),
                dtype=float,
            )
            row["qdq2av_probes"].append(
                {
                    "derivative_construction": side,
                    "input_qdot_wxyz_per_second": qdot,
                    "qdq2av_omega_xyz_per_second": native.tolist(),
                }
            )
        derivatives.append(row)

    libspice_path = Path(spice.__file__).parent / "utils" / "cspice.dll"
    if not libspice_path.exists():
        raise FileNotFoundError(
            f"Could not locate the loaded CSPICE binary: {libspice_path}"
        )
    lock_path = HERE / "uv.lock"
    output = {
        "schema_version": "1.0",
        "evidence_kind": "native_cspice_runtime_capture_via_spiceypy",
        "runtime": {
            "wrapper": "SpiceyPy",
            "wrapper_version": spice.__version__,
            "toolkit_version": spice.tkvrsn("TOOLKIT"),
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "loaded_library_path": str(libspice_path),
            "loaded_library_sha256": sha256_file(libspice_path),
        },
        "provenance": {
            "input_corpus_sha256": sha256(corpus_bytes),
            "lockfile_sha256": sha256_file(lock_path),
            "exporter_sha256": sha256_file(Path(__file__)),
            "exact_invocation": (
                "uv run --locked python capture_spice.py --corpus "
                "<orientation_inputs_v1.json> --output <native_rotation.json>"
            ),
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "counts": {
            "euler_orders": len(eulers),
            "euler_boundaries": len(euler_boundaries),
            "axis_angle": len(axis_angle),
            "multiplication": len(products),
            "proper_matrices": len(proper_matrices),
            "qdq2av": len(derivatives),
        },
        "observations": {
            "euler_orders": eulers,
            "euler_boundaries": euler_boundaries,
            "axis_angle": axis_angle,
            "multiplication": products,
            "proper_matrices": proper_matrices,
            "qdq2av": derivatives,
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(options.output),
                "toolkit": output["runtime"]["toolkit_version"],
                "counts": output["counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
