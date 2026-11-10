"""Native vector-action acceptance checks for the canonical frame adapters."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quatops.experimental.engines import (
    QuaternionFrameMap,
    SignedAxisMap,
    blender,
    godot,
    unity,
    unreal,
)
from tests.support.native_engine_support import load_versioned_engine_witness
from tests.support.paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
INPUTS = json.loads(
    (
        ROOT / "experiments" / "interoperability" / "v1" / "canonical_inputs.json"
    ).read_text(encoding="utf-8")
)
CONTRACT_ID = INPUTS["contract_id"]
PROBE = np.asarray(INPUTS["probe_vector_xyz"], dtype=float)
TARGETS = (
    ("unreal", "5.7.4", 2.0e-6),
    ("unity", "6000.5.2f1", 2.0e-6),
    ("godot", "4.7-stable", 2.0e-6),
    ("godot", "4.8-dev1", 2.0e-6),
    ("blender", "5.1.2", 2.0e-6),
)
ADAPTERS = {
    "unreal": unreal,
    "unity": unity,
    "godot": godot,
    "blender": blender,
}


def _fixture(engine, version):
    return load_versioned_engine_witness(ROOT, engine, version)[1]


def _capture_rows(fixture):
    capture = fixture["canonical_interop"]
    if "precision_captures" in capture:
        return capture, capture["precision_captures"]
    if "float32" in capture and "float64" in capture:
        return capture, {key: capture[key] for key in ("float32", "float64")}
    profiles = fixture["target"]["native_scalar_profiles"]
    return capture, {profiles[0]: capture}


def _vec(value):
    if isinstance(value, dict):
        return np.asarray((value["x"], value["y"], value["z"]), dtype=float)
    return np.asarray(value, dtype=float)


def _wxyz_to_xyzw(value):
    value = np.asarray(value, dtype=float)
    return value[[1, 2, 3, 0]]


def _native_xyzw(capture, value):
    value = np.asarray(value, dtype=float)
    if capture["native_quaternion_order"] == "xyzw":
        return value
    assert capture["native_quaternion_order"] == "wxyz"
    return value[[1, 2, 3, 0]]


def _frame_map(metadata):
    axis_map = SignedAxisMap(tuple(metadata["frame_map_destination_axes_from_source"]))
    # These fixtures declare one shared map for the world and body coordinates.
    return QuaternionFrameMap(axis_map, axis_map)


def _to_native(engine, q_canonical_wxyz, frame_map):
    return getattr(ADAPTERS[engine], f"to_{engine}")(q_canonical_wxyz, frame_map)


def _from_native(engine, q_native, frame_map):
    return getattr(ADAPTERS[engine], f"from_{engine}")(q_native, frame_map)


def _same_orientation(actual_xyzw, expected_xyzw, tolerance, label):
    actual = Rotation.from_quat(actual_xyzw).as_matrix()
    expected = Rotation.from_quat(expected_xyzw).as_matrix()
    np.testing.assert_allclose(
        actual, expected, rtol=0.0, atol=tolerance, err_msg=label
    )


def _basis_matrix_images(images):
    # Captures store one row per native image of +X, +Y, and +Z.
    return np.asarray([_vec(image) for image in images], dtype=float).T


@pytest.mark.parametrize("engine,version,tolerance", TARGETS)
def test_fixture_declares_the_canonical_contract_and_explicit_frame_map(
    engine, version, tolerance
):
    fixture = _fixture(engine, version)
    metadata, _ = _capture_rows(fixture)
    assert metadata["contract_id"] == CONTRACT_ID
    assert metadata["native_quaternion_order"] in {"xyzw", "wxyz"}
    assert metadata["frame_map_destination_axes_from_source"]
    contract = fixture["contract"]
    assert fixture["target"]["version"] == version
    source_frame = contract["canonical_frame"]
    for key, value in INPUTS["source_frame"].items():
        assert source_frame[key] == value

    assert tuple(contract["destination_axes_from_source"]) == tuple(
        metadata["frame_map_destination_axes_from_source"]
    )
    assert INPUTS["contract"]["algebra"] == "Hamilton"
    assert INPUTS["contract"]["coefficient_order"] == "wxyz"
    assert INPUTS["contract"]["rotation_action"] == "active"
    assert INPUTS["contract"]["matrix_action"] == "column vectors"
    for input_case in INPUTS["orientations"]:
        assert np.linalg.norm(input_case["canonical_wxyz"]) == pytest.approx(
            1.0, abs=1e-15
        )

    assert len(PROBE) == 3


@pytest.mark.parametrize("engine,version,tolerance", TARGETS)
def test_canonical_to_native_and_native_to_canonical_preserve_physical_action(
    engine, version, tolerance
):
    fixture = _fixture(engine, version)
    metadata, captures = _capture_rows(fixture)
    axis_map = SignedAxisMap(tuple(metadata["frame_map_destination_axes_from_source"]))
    frame_map = _frame_map(metadata)
    for precision, capture in captures.items():
        precision_tolerance = 2.0e-14 if precision == "float64" else tolerance
        cases = {case["case_id"]: case for case in capture["orientation_cases"]}
        assert set(cases) == {row["case_id"] for row in INPUTS["orientations"]}

        for input_case in INPUTS["orientations"]:
            case = cases[input_case["case_id"]]
            canonical_wxyz = np.asarray(input_case["canonical_wxyz"], dtype=float)
            canonical_xyzw = _wxyz_to_xyzw(canonical_wxyz)
            expected_native = np.asarray(
                _to_native(engine, canonical_wxyz, frame_map), dtype=float
            )
            np.testing.assert_allclose(
                _native_components(case["native_quaternion"]),
                expected_native,
                rtol=0.0,
                atol=precision_tolerance,
                err_msg=f"{engine} {version} {precision} {case['case_id']} canonical-to-native coefficients",
            )

            if case["case_id"] == "mixed_a":
                # One native mixed-vector action supplements the complete
                # basis-action witness set for this linear rotation API.
                native_image = _vec(case["native_rotated_probe_vector_xyz"])
                source_image = Rotation.from_quat(canonical_xyzw).apply(PROBE)
                np.testing.assert_allclose(
                    native_image,
                    axis_map.vector(source_image),
                    rtol=0.0,
                    atol=precision_tolerance,
                    err_msg=f"{engine} {version} {precision} mixed-vector action",
                )
                np.testing.assert_allclose(
                    axis_map.inverse().vector(native_image),
                    source_image,
                    rtol=0.0,
                    atol=precision_tolerance,
                    err_msg=f"{engine} {version} {precision} native-to-canonical vector action",
                )

            expected_target_matrix = (
                np.asarray(axis_map.matrix, dtype=float)
                @ Rotation.from_quat(canonical_xyzw).as_matrix()
                @ np.asarray(axis_map.matrix, dtype=float).T
            )
            np.testing.assert_allclose(
                _basis_matrix_images(case["native_rotated_positive_basis_vectors_xyz"]),
                expected_target_matrix,
                rtol=0.0,
                atol=precision_tolerance,
                err_msg=f"{engine} {version} {precision} {case['case_id']} native basis actions",
            )

            roundtrip_wxyz = np.asarray(
                _from_native(
                    engine,
                    _native_components(case["native_quaternion"]),
                    frame_map,
                ),
                dtype=float,
            )
            np.testing.assert_allclose(
                roundtrip_wxyz,
                canonical_wxyz,
                rtol=0.0,
                atol=precision_tolerance,
                err_msg=f"{engine} {version} {precision} {case['case_id']} target-to-canonical quaternion",
            )

        mixed = cases["mixed_a"]
        antipode = cases["antipode_mixed_a"]
        np.testing.assert_allclose(
            _native_components(antipode["native_quaternion"]),
            -_native_components(mixed["native_quaternion"]),
            rtol=0.0,
            atol=precision_tolerance,
            err_msg=f"{engine} {version} {precision} antipodal coefficient preservation",
        )
        np.testing.assert_allclose(
            _basis_matrix_images(antipode["native_rotated_positive_basis_vectors_xyz"]),
            _basis_matrix_images(mixed["native_rotated_positive_basis_vectors_xyz"]),
            rtol=0.0,
            atol=precision_tolerance,
            err_msg=f"{engine} {version} {precision} antipodal physical action",
        )


@pytest.mark.parametrize("engine,version,tolerance", TARGETS)
def test_noncommuting_composition_survives_canonical_and_native_frames(
    engine, version, tolerance
):
    metadata, captures = _capture_rows(_fixture(engine, version))
    axis_map = SignedAxisMap(tuple(metadata["frame_map_destination_axes_from_source"]))
    frame_map = _frame_map(metadata)
    left = Rotation.from_quat(
        _wxyz_to_xyzw(
            next(
                row["canonical_wxyz"]
                for row in INPUTS["orientations"]
                if row["case_id"] == "mixed_a"
            )
        )
    )
    right = Rotation.from_quat(
        _wxyz_to_xyzw(
            next(
                row["canonical_wxyz"]
                for row in INPUTS["orientations"]
                if row["case_id"] == "mixed_b"
            )
        )
    )
    assert np.linalg.norm((left * right).as_matrix() - (right * left).as_matrix()) > 0.5

    for precision, capture in captures.items():
        precision_tolerance = 2.0e-14 if precision == "float64" else tolerance
        for composition, expected_product in (
            (capture["composition"], left * right),
            (capture["reverse_composition"], right * left),
        ):
            assert composition["product_order"] == "lhs * rhs; rhs acts first"
            native_product_xyzw = _native_xyzw(
                metadata, composition["native_product_quaternion"]
            )
            expected_wxyz = expected_product.as_quat()[[3, 0, 1, 2]]
            expected_native_product = _to_native(engine, expected_wxyz, frame_map)
            expected_native_product_xyzw = _native_xyzw(
                metadata, expected_native_product
            )
            _same_orientation(
                native_product_xyzw,
                expected_native_product_xyzw,
                precision_tolerance,
                f"{engine} {version} {precision} native product coefficient conversion",
            )
            np.testing.assert_allclose(
                _basis_matrix_images(
                    composition["native_rotated_positive_basis_vectors_xyz"]
                ),
                np.asarray(axis_map.matrix, dtype=float)
                @ expected_product.as_matrix()
                @ np.asarray(axis_map.matrix, dtype=float).T,
                rtol=0.0,
                atol=precision_tolerance,
                err_msg=f"{engine} {version} {precision} native product basis action",
            )
            _same_orientation(
                _wxyz_to_xyzw(
                    _from_native(
                        engine, composition["native_product_quaternion"], frame_map
                    )
                ),
                expected_product.as_quat(),
                precision_tolerance,
                f"{engine} {version} {precision} target-to-canonical product",
            )


def test_unity_to_godot_native_cross_transfer_matches_both_captured_actions():
    unity_metadata, unity_captures = _capture_rows(_fixture("unity", "6000.5.2f1"))
    godot_metadata, godot_captures = _capture_rows(_fixture("godot", "4.7-stable"))
    unity = unity_captures["float32"]
    godot = godot_captures["float32"]
    unity_to_godot = SignedAxisMap(("x", "y", "-z"))
    unity_to_source = SignedAxisMap(
        tuple(unity_metadata["frame_map_destination_axes_from_source"])
    ).inverse()
    source_to_godot = SignedAxisMap(
        tuple(godot_metadata["frame_map_destination_axes_from_source"])
    )
    unity_frame_map = _frame_map(unity_metadata)
    godot_frame_map = _frame_map(godot_metadata)
    expected_unity_to_godot = np.asarray(
        source_to_godot.matrix, dtype=float
    ) @ np.asarray(unity_to_source.matrix, dtype=float)
    np.testing.assert_array_equal(unity_to_godot.matrix, expected_unity_to_godot)
    unity_cases = {row["case_id"]: row for row in unity["orientation_cases"]}
    godot_cases = {row["case_id"]: row for row in godot["orientation_cases"]}

    case_ids = [row["case_id"] for row in INPUTS["orientations"]]
    for case_id in case_ids:
        unity_native = _native_components(unity_cases[case_id]["native_quaternion"])
        godot_native = _native_components(godot_cases[case_id]["native_quaternion"])
        canonical = _from_native("unity", unity_native, unity_frame_map)
        adapter_transfer = _to_native("godot", canonical, godot_frame_map)
        _same_orientation(
            _native_xyzw(godot_metadata, adapter_transfer),
            _native_xyzw(godot_metadata, godot_native),
            2.0e-6,
            f"Unity-to-Godot named-adapter transfer for {case_id}",
        )

        unity_xyzw = _native_xyzw(unity_metadata, unity_native)
        godot_xyzw = _native_xyzw(godot_metadata, godot_native)
        transferred = unity_to_godot.map_hamilton_xyzw(unity_xyzw)
        _same_orientation(
            transferred,
            godot_xyzw,
            2.0e-6,
            f"Unity-to-Godot native quaternion transfer for {case_id}",
        )
    case_id = "mixed_a"
    transferred_vector = unity_to_godot.vector(
        _vec(unity_cases[case_id]["native_rotated_probe_vector_xyz"])
    )
    np.testing.assert_allclose(
        transferred_vector,
        _vec(godot_cases[case_id]["native_rotated_probe_vector_xyz"]),
        rtol=0.0,
        atol=2.0e-6,
        err_msg="Unity-to-Godot native vector transfer for mixed_a",
    )


def _native_components(value):
    """Return stored target coefficients without changing their component order."""
    return np.asarray(value, dtype=float)


@pytest.mark.parametrize("engine,version,tolerance", TARGETS)
def test_named_adapter_requires_an_explicit_precompiled_frame_map(
    engine, version, tolerance
):
    del version, tolerance
    adapter = ADAPTERS[engine]
    with pytest.raises(TypeError, match="precompiled QuaternionFrameMap"):
        getattr(adapter, f"to_{engine}")((1.0, 0.0, 0.0, 0.0), None)


def test_native_witness_detects_plausible_wrong_unity_conversions():
    """The frozen native action rejects sign, axis, and product-order mutants."""
    metadata, captures = _capture_rows(_fixture("unity", "6000.5.2f1"))
    capture = captures["float32"]
    cases = {row["case_id"]: row for row in capture["orientation_cases"]}
    mixed = cases["mixed_a"]
    expected_basis = _basis_matrix_images(
        mixed["native_rotated_positive_basis_vectors_xyz"]
    )
    native_xyzw = _native_components(mixed["native_quaternion"])

    mutants = {
        "unnecessary conjugation": np.asarray(
            (-native_xyzw[0], -native_xyzw[1], -native_xyzw[2], native_xyzw[3])
        ),
        "swapped native X and Y": native_xyzw[[1, 0, 2, 3]],
    }
    canonical_wxyz = np.asarray(
        next(
            row["canonical_wxyz"]
            for row in INPUTS["orientations"]
            if row["case_id"] == "mixed_a"
        )
    )
    axis_map = SignedAxisMap(tuple(metadata["frame_map_destination_axes_from_source"]))
    assert axis_map.determinant == -1
    missing_determinant_xyzw = np.concatenate(
        (
            np.asarray(axis_map.matrix, dtype=float) @ canonical_wxyz[1:],
            canonical_wxyz[:1],
        )
    )
    mutants["missing reflection determinant sign"] = missing_determinant_xyzw
    wrong_axis_map = SignedAxisMap(("y", "x", "z"))
    wrong_frame_map = QuaternionFrameMap(wrong_axis_map, wrong_axis_map)
    mutants["swapped source X and Y axes"] = np.asarray(
        _to_native("unity", canonical_wxyz, wrong_frame_map)
    )

    reverse = capture["reverse_composition"]["native_product_quaternion"]
    product = capture["composition"]["native_product_quaternion"]
    reverse_matrix = Rotation.from_quat(_native_xyzw(metadata, reverse)).as_matrix()
    expected_product_matrix = Rotation.from_quat(
        _native_xyzw(metadata, product)
    ).as_matrix()
    assert np.linalg.norm(reverse_matrix - expected_product_matrix) > 0.5

    for mutant_name, mutant_xyzw in mutants.items():
        mutant_basis = Rotation.from_quat(mutant_xyzw).as_matrix()
        assert np.linalg.norm(mutant_basis - expected_basis) > 1e-3, mutant_name
