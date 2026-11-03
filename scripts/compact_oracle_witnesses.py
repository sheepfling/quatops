"""Select retained native witnesses from a temporary discovery capture.

The input stays untouched. Case IDs and field projections below are the
retention contract; selected values are copied into a separate output file.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable, Hashable
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from scripts.evidence_json import dump_evidence_json, read_evidence_document

Target = Literal["threejs", "mujoco", "spiceypy"]
Record = dict[str, Any]

AXIS_ANGLE_CASES = (
    "axis-angle-0-00",
    "axis-angle-1-05",
    "axis-angle-2-05",
    "axis-angle-3-05",
    "axis-angle-3-14",
)
QUATERNION_CASES = (
    "algebra-0",
    "algebra-6",
    "algebra-7",
    "rotation-identity",
    "rotation-random-000",
    "rotation-random-000-antipodal",
    "invalid-rotation-zero",
    "invalid-rotation-nonunit",
    "invalid-rotation-tiny",
)
MATRIX_CASES = ("matrix-identity", "matrix-x90", "matrix-mixed-rational")
INTEGRATION_CASES = (
    "integration-0-0-0",
    "integration-0-1-1",
    "integration-1-0-1",
    "integration-2-3-3",
)
QUATERNION_DIFFERENCE_CASES = (
    "axis-angle-0-05__minus__axis-angle-1-05",
    "axis-angle-3-05__minus__axis-angle-0-05",
)
ANGULAR_VELOCITY_CASES = ("integration-0-0-0", "integration-2-3-3")
NEAR_LOCK_CASE = "boundary-offset-ZYX-+1-+1e-04"


def _select_ids(rows: list[Record], case_ids: tuple[str, ...]) -> list[Record]:
    """Select named rows in declared order; reject duplicate or missing IDs."""
    required = set(case_ids)
    indexed: dict[str, Record] = {}
    for row in rows:
        case_id = row["case_id"]
        if case_id in required:
            if case_id in indexed:
                raise ValueError(f"duplicate retained case ID: {case_id}")
            indexed[case_id] = row
    missing = [case_id for case_id in case_ids if case_id not in indexed]
    if missing:
        raise ValueError(f"missing retained case IDs: {', '.join(missing)}")
    return [indexed[case_id] for case_id in case_ids]


def _first_per_key(
    rows: list[Record],
    key_fields: tuple[str, ...],
    *,
    include: Callable[[Record], bool] = lambda _row: True,
) -> list[Record]:
    """Keep the first matching row per declared key in capture order."""
    selected: list[Record] = []
    seen: set[tuple[Hashable, ...]] = set()
    for row in rows:
        if include(row):
            key = tuple(row[field] for field in key_fields)
            if key not in seen:
                seen.add(key)
                selected.append(row)
    return selected


def _rows(observations: Record, name: str) -> list[Record]:
    rows = observations[name]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise TypeError(f"observations.{name} must be an array of objects")
    return rows


def _counts(observations: Record) -> dict[str, int]:
    return {
        name: len(rows)
        for name, rows in observations.items()
        if isinstance(rows, list)
    }


def _provenance(document: Record, source_sha256: str, *, threejs: bool = False) -> Record:
    current = document.get("provenance", {})
    if not isinstance(current, dict):
        raise TypeError("provenance must be an object")
    policy = (
        "selected API witnesses; redundant matrix views/actions and hierarchy "
        "duplicates removed; broad successful sweeps discarded"
        if threejs
        else "selected native witnesses; broad successful sweeps discarded"
    )
    retained = {
        **current,
        "full_discovery_capture_sha256": current.get(
            "full_discovery_capture_sha256", source_sha256
        ),
        "retention_policy": policy,
    }
    if threejs:
        retained["retention_revision"] = 3
    return retained


def _compact_threejs(document: Record, source_sha256: str) -> Record:
    source = document["observations"]
    if not isinstance(source, dict):
        raise TypeError("observations must be an object")
    retained = {
        name: rows
        for name, rows in source.items()
        if name not in {"euler_orders", "euler_boundaries", "euler_reorder", "object3d_hierarchy"}
    }

    near_lock = _select_ids(_rows(source, "euler_boundaries"), (NEAR_LOCK_CASE,))[0]
    regression_fields = (
        "case_id",
        "order",
        "input_radians_xyz",
        "input_source",
        "euler_quaternion_xyzw",
        "quaternion_readback_roundtrip_xyzw",
    )
    retained["euler_regressions"] = [
        {field: near_lock[field] for field in regression_fields}
    ]

    retained["axis_angle"] = [
        {
            field: row[field]
            for field in (
                "case_id",
                "input_axis_xyz",
                "input_angle_radians",
                "quaternion_xyzw",
            )
        }
        for row in _select_ids(_rows(source, "axis_angle"), AXIS_ANGLE_CASES)
    ]
    retained["quaternion_normalization"] = [
        {
            key: value
            for key, value in row.items()
            if key not in {"input_xyzw", "length", "length_squared"}
        }
        for row in _select_ids(
            _rows(source, "quaternion_normalization"), QUATERNION_CASES
        )
    ]
    retained["independent_matrices"] = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
                "matrix4_elements_column_major",
                "quaternion_rotated_positive_basis_vectors_xyz",
            }
        }
        for row in _select_ids(
            _rows(source, "independent_matrices"), MATRIX_CASES
        )
        if row["domain"] == "proper_rotation"
    ]

    alignment = [
        row
        for row in _rows(source, "two_vector_alignment")
        if row["from_case_index"] == 0 and row["to_case_index"] in (1, 2)
    ]
    if len(alignment) != 2:
        raise ValueError("expected the two named Three.js vector-alignment cases")
    retained["two_vector_alignment"] = alignment

    slerp_rows = []
    for row in _rows(source, "shortest_slerp"):
        slerp_rows.append(
            {
                **row,
                "samples": [
                    sample for sample in row["samples"] if sample["t"] == 0.5
                ],
            }
        )
    retained["shortest_slerp"] = slerp_rows
    return {
        **document,
        "observations": retained,
        "counts": _counts(retained),
        "provenance": _provenance(document, source_sha256, threejs=True),
    }


def _compact_other(target: Literal["mujoco", "spiceypy"], document: Record, source_sha256: str) -> Record:
    source = document["observations"]
    if not isinstance(source, dict):
        raise TypeError("observations must be an object")
    retained = dict(source)

    if "euler_orders" in source:
        key = ("order_label", "sequence_case") if target == "mujoco" else ("order",)
        retained["euler_orders"] = _first_per_key(
            _rows(source, "euler_orders"),
            key,
            include=lambda row: row["case_id"] != "euler-identity",
        )
    if "euler_boundaries" in source:
        order = "order_label" if target == "mujoco" else "order"
        retained["euler_boundaries"] = _first_per_key(
            _rows(source, "euler_boundaries"), (order,)
        )
    for group, case_ids in (
        ("axis_angle", AXIS_ANGLE_CASES),
        ("quaternion_normalization", QUATERNION_CASES),
        ("proper_matrices", MATRIX_CASES),
        ("independent_matrices", (*MATRIX_CASES, "matrix-shear")),
        ("integration", INTEGRATION_CASES),
        ("sub_quat", QUATERNION_DIFFERENCE_CASES),
        ("qdq2av", ANGULAR_VELOCITY_CASES),
    ):
        if group in source:
            retained[group] = _select_ids(_rows(source, group), case_ids)
    if "two_vector_alignment" in source:
        alignment = [
            row
            for row in _rows(source, "two_vector_alignment")
            if row["from_case_index"] == 0 and row["to_case_index"] in (1, 2)
        ]
        if len(alignment) != 2:
            raise ValueError("expected the two named vector-alignment cases")
        retained["two_vector_alignment"] = alignment

    counts = _counts(retained)
    counts = {
        {"integration": "integration_and_derivatives", "sub_quat": "quaternion_differences"}.get(
            name, name
        ): count
        for name, count in counts.items()
    }
    return {
        **document,
        "observations": retained,
        "counts": counts,
        "provenance": _provenance(document, source_sha256),
    }


def compact_capture(target: Target, document: Record, source_sha256: str) -> Record:
    """Return selected witness data without modifying the source capture."""
    document = deepcopy(document)
    if target == "threejs":
        return _compact_threejs(document, source_sha256)
    return _compact_other(target, document, source_sha256)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("threejs", "mujoco", "spiceypy"))
    parser.add_argument("discovery_capture", type=Path)
    parser.add_argument("retained_output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source_path, output_path = args.discovery_capture.resolve(), args.retained_output.resolve()
    if source_path == output_path:
        raise ValueError("discovery input and retained output must be different files")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace output without --overwrite: {output_path}")

    source_bytes = source_path.read_bytes()
    capture = read_evidence_document(source_path)
    retained = compact_capture(
        args.target, capture, hashlib.sha256(source_bytes).hexdigest()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dump_evidence_json(retained), encoding="utf-8", newline="\n")
    print(f"Retained {args.target} witnesses at {output_path}")


if __name__ == "__main__":
    main()
