"""The discovery reducer selects named cases without editing its input."""

from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.compact_oracle_witnesses import compact_capture


def _case(case_id: str, **fields: object) -> dict[str, object]:
    return {"case_id": case_id, **fields}


def _threejs_discovery() -> dict[str, object]:
    orders = ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX")
    boundaries = [
        {
            "case_id": "boundary-offset-ZYX-+1-+1e-04",
            "order": "ZYX",
            "input_radians_xyz": [0.1, 1.0, 0.2],
            "input_source": "measured near-lock case",
            "euler_quaternion_xyzw": [0.1, 0.2, 0.3, 0.9],
            "quaternion_readback_roundtrip_xyzw": [0.1, 0.2, 0.31, 0.89],
        }
    ]
    boundaries.extend(
        {"case_id": f"boundary-{order}", "order": order} for order in orders[:-1]
    )
    axis_angle = [
        _case(
            case_id,
            input_axis_xyz=[1.0, 0.0, 0.0],
            input_angle_radians=0.5,
            quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
            discovery_note="removed from retained record",
        )
        for case_id in (
            "axis-angle-0-00",
            "axis-angle-1-05",
            "axis-angle-2-05",
            "axis-angle-3-05",
            "axis-angle-3-14",
            "axis-angle-extra",
        )
    ]
    quaternion_cases = (
        "algebra-0",
        "algebra-6",
        "algebra-7",
        "rotation-identity",
        "rotation-random-000",
        "rotation-random-000-antipodal",
        "invalid-rotation-zero",
        "invalid-rotation-nonunit",
        "invalid-rotation-tiny",
        "rotation-extra",
    )
    matrix_cases = ("matrix-identity", "matrix-x90", "matrix-mixed-rational")
    return {
        "schema_version": "1.0",
        "observations": {
            "euler_orders": [{"order": order} for order in orders],
            "euler_boundaries": boundaries,
            "euler_reorder": [{"case_id": "reorder-extra"}],
            "axis_angle": axis_angle,
            "quaternion_normalization": [
                _case(
                    case_id,
                    input_xyzw=[0.0, 0.0, 0.0, 1.0],
                    length=1.0,
                    length_squared=1.0,
                    native_xyzw=[0.0, 0.0, 0.0, 1.0],
                )
                for case_id in quaternion_cases
            ],
            "independent_matrices": [
                _case(
                    case_id,
                    domain="proper_rotation",
                    rows=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    quaternion_rotated_positive_basis_vectors_xyz=[[1.0, 0.0, 0.0]],
                )
                for case_id in matrix_cases
            ],
            "two_vector_alignment": [
                {"from_case_index": 0, "to_case_index": target} for target in (1, 2, 3)
            ],
            "shortest_slerp": [
                {
                    "case_id": "interp-quarter-turn",
                    "samples": [{"t": weight} for weight in (-0.5, 0.25, 0.5, 1.5)],
                },
                {
                    "case_id": "interp-antipodal",
                    "samples": [{"t": 0.25}, {"t": 0.5}],
                },
            ],
            "object3d_hierarchy": [{"case_id": "hierarchy-duplicate"}],
        },
        "provenance": {"native_capture_sha256": "capture-hash"},
    }


def test_threejs_reducer_is_non_mutating_and_keeps_declared_witnesses():
    discovery = _threejs_discovery()
    original = deepcopy(discovery)

    retained = compact_capture("threejs", discovery, "raw-discovery-hash")

    assert discovery == original
    assert retained is not discovery
    observations = retained["observations"]
    assert isinstance(observations, dict)
    assert "euler_orders" not in observations
    assert "euler_boundaries" not in observations
    assert "euler_reorder" not in observations
    assert "object3d_hierarchy" not in observations
    regression = observations["euler_regressions"]
    assert isinstance(regression, list)
    assert regression[0]["case_id"] == "boundary-offset-ZYX-+1-+1e-04"
    assert len(observations["axis_angle"]) == 5
    assert len(observations["quaternion_normalization"]) == 9
    assert len(observations["independent_matrices"]) == 3
    assert len(observations["two_vector_alignment"]) == 2
    slerp = observations["shortest_slerp"]
    assert isinstance(slerp, list)
    assert [sample["t"] for sample in slerp[0]["samples"]] == [0.5]
    assert [sample["t"] for sample in slerp[1]["samples"]] == [0.5]
    assert (
        "matrix4_elements_column_major" not in observations["independent_matrices"][0]
    )
    assert (
        "quaternion_rotated_positive_basis_vectors_xyz"
        not in observations["independent_matrices"][0]
    )
    assert (
        retained["provenance"]["full_discovery_capture_sha256"] == "raw-discovery-hash"
    )
    slerp[0]["samples"][0]["t"] = 99.0
    assert discovery == original


def test_generic_reducer_selects_integration_rows_by_case_id():
    integration_ids = (
        "integration-0-0-0",
        "integration-0-1-1",
        "integration-1-0-1",
        "integration-2-3-3",
    )
    discovery = {
        "observations": {
            "integration": [
                _case("unretained-first"),
                *[_case(case_id) for case_id in reversed(integration_ids)],
            ]
        },
        "provenance": {},
    }
    original = deepcopy(discovery)

    retained = compact_capture("mujoco", discovery, "raw-discovery-hash")

    assert discovery == original
    observations = retained["observations"]
    assert isinstance(observations, dict)
    integration = observations["integration"]
    assert isinstance(integration, list)
    assert [row["case_id"] for row in integration] == list(integration_ids)
    assert retained["counts"]["integration_and_derivatives"] == len(integration_ids)
    assert "integration" not in retained["counts"]


@pytest.mark.parametrize(
    "rows",
    [
        [{"case_id": "duplicate"}, {"case_id": "duplicate"}],
        [{"case_id": "unexpected"}],
    ],
)
def test_case_id_selection_rejects_duplicate_or_missing_witnesses(rows):
    from scripts.compact_oracle_witnesses import _select_ids

    with pytest.raises(ValueError):
        _select_ids(rows, ("duplicate",))
