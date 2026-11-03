"""Native vector-action acceptance checks for the canonical frame adapters."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quatops.engines import QuaternionFrameMap, SignedAxisMap
from quatops.engines import blender, godot, unity, unreal

ROOT = Path(__file__).parent.parent
INPUTS = json.loads(
    (ROOT / "experiments" / "interoperability" / "v1" / "canonical_inputs.json")
    .read_text(encoding="utf-8")
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
    path = ROOT / "tests" / "golden" / engine / version / "native_rotation.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _capture_rows(fixture):
    capture = fixture["canonical_interop"]
    if "precision_captures" in capture:
        return capture, capture["precision_captures"]
    if "float32" in capture and "float64" in capture:
        return capture, {key: capture[key] for key in ("float32", "float64")}
    return capture, {fixture.get("engine", {}).get("precision", "float32"): capture}


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
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=tolerance, err_msg=label)


def _basis_matrix_images(images):
    # Captures store one row per native image of +X, +Y, and +Z.
    return np.asarray([_vec(image) for image in images], dtype=float).T


@pytest.mark.parametrize("engine,version,tolerance", TARGETS)
def test_fixture_declares_the_canonical_contract_and_explicit_frame_map(
    engine, version, tolerance
):
    fixture = _fixture(engine, version)
    metadata, captures = _capture_rows(fixture)
    assert metadata["contract_id"] == CONTRACT_ID
    assert metadata["native_quaternion_order"] in {"xyzw", "wxyz"}
    assert metadata["frame_map_destination_axes_from_source"]
    assert fixture["evidence_kind"].startswith("native_")
    source_frame = (
        fixture["native_semantics"]["source_frame"]
        if engine == "unreal"
        else fixture["source_frame"]
    )
    for key, value in INPUTS["source_frame"].items():
        assert source_frame[key] == value

    captured_map = {
        "unreal": fixture.get("native_semantics", {}).get("source_to_unreal_axis_map"),
        "unity": fixture.get("source_to_unity_axis_map"),
        "godot": fixture.get("source_to_godot_axis_map"),
        "blender": fixture.get("source_to_blender_axis_map"),
    }[engine]
    assert tuple(captured_map["destination_axes_from_source"]) == tuple(
        metadata["frame_map_destination_axes_from_source"]
    )
    assert INPUTS["contract"]["algebra"] == "Hamilton"
    assert INPUTS["contract"]["coefficient_order"] == "wxyz"
    assert INPUTS["contract"]["rotation_action"] == "active"
    assert INPUTS["contract"]["matrix_action"] == "column vectors"
    for input_case in INPUTS["orientations"]:
        assert np.linalg.norm(input_case["canonical_wxyz"]) == pytest.approx(1.0, abs=1e-15)

    axis_map = SignedAxisMap(tuple(metadata["frame_map_destination_axes_from_source"]))
    for precision, capture in captures.items():
        precision_tolerance = 2.0e-14 if precision == "float64" else tolerance
        np.testing.assert_allclose(
            _vec(capture["probe_vector_canonical_xyz"]),
            PROBE,
            rtol=0.0,
            atol=precision_tolerance,
        )
        assert _vec(capture["probe_vector_native_xyz"]) == pytest.approx(
            axis_map.vector(PROBE), abs=precision_tolerance
        )


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
            assert np.asarray(case["canonical_wxyz"], dtype=float) == pytest.approx(
                canonical_wxyz, abs=0.0
            )
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

            # The native right side is the vector action captured inside the pinned engine.
            native_probe = _vec(capture["probe_vector_native_xyz"])
            native_image = _vec(case["native_rotated_probe_vector_xyz"])
            source_image = Rotation.from_quat(canonical_xyzw).apply(PROBE)
            np.testing.assert_allclose(
                native_image,
                axis_map.vector(source_image),
                rtol=0.0,
                atol=precision_tolerance,
                err_msg=f"{engine} {version} {precision} {case['case_id']} native mixed-vector action",
            )
            np.testing.assert_allclose(
                axis_map.inverse().vector(native_image),
                source_image,
                rtol=0.0,
                atol=precision_tolerance,
                err_msg=f"{engine} {version} {precision} {case['case_id']} native-to-canonical vector action",
            )
            np.testing.assert_allclose(
                native_probe,
                axis_map.vector(PROBE),
                rtol=0.0,
                atol=precision_tolerance,
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


@pytest.mark.parametrize("engine,version,tolerance", TARGETS)
def test_noncommuting_composition_survives_canonical_and_native_frames(
    engine, version, tolerance
):
    metadata, captures = _capture_rows(_fixture(engine, version))
    axis_map = SignedAxisMap(tuple(metadata["frame_map_destination_axes_from_source"]))
    frame_map = _frame_map(metadata)
    left = Rotation.from_quat(
        _wxyz_to_xyzw(next(row["canonical_wxyz"] for row in INPUTS["orientations"]
                           if row["case_id"] == "mixed_a"))
    )
    right = Rotation.from_quat(
        _wxyz_to_xyzw(next(row["canonical_wxyz"] for row in INPUTS["orientations"]
                           if row["case_id"] == "mixed_b"))
    )
    expected = left * right
    assert np.linalg.norm((left * right).as_matrix() - (right * left).as_matrix()) > 0.5

    for precision, capture in captures.items():
        precision_tolerance = 2.0e-14 if precision == "float64" else tolerance
        composition = capture["composition"]
        assert composition["left_case_id"] == "mixed_a"
        assert composition["right_case_id"] == "mixed_b"
        assert composition["product_order"] == "lhs * rhs; rhs acts first"
        native_product_xyzw = _native_xyzw(
            metadata, composition["native_product_quaternion"]
        )
        expected_wxyz = expected.as_quat()[[3, 0, 1, 2]]
        expected_native_product = _to_native(engine, expected_wxyz, frame_map)
        expected_native_product_xyzw = _native_xyzw(metadata, expected_native_product)
        _same_orientation(
            native_product_xyzw,
            expected_native_product_xyzw,
            precision_tolerance,
            f"{engine} {version} {precision} native product coefficient conversion",
        )

        source_image = expected.apply(PROBE)
        native_image = _vec(composition["native_rotated_probe_vector_xyz"])
        np.testing.assert_allclose(
            native_image,
            axis_map.vector(source_image),
            rtol=0.0,
            atol=precision_tolerance,
            err_msg=f"{engine} {version} {precision} native noncommuting product vector action",
        )
        np.testing.assert_allclose(
            _basis_matrix_images(composition["native_rotated_positive_basis_vectors_xyz"]),
            np.asarray(axis_map.matrix, dtype=float)
            @ expected.as_matrix()
            @ np.asarray(axis_map.matrix, dtype=float).T,
            rtol=0.0,
            atol=precision_tolerance,
            err_msg=f"{engine} {version} {precision} native noncommuting product basis action",
        )
        _same_orientation(
            _wxyz_to_xyzw(
                _from_native(
                    engine,
                    composition["native_product_quaternion"],
                    frame_map,
                )
            ),
            expected.as_quat(),
            precision_tolerance,
            f"{engine} {version} {precision} noncommuting target-to-canonical product",
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
    expected_unity_to_godot = (
        np.asarray(source_to_godot.matrix, dtype=float)
        @ np.asarray(unity_to_source.matrix, dtype=float)
    )
    np.testing.assert_array_equal(unity_to_godot.matrix, expected_unity_to_godot)
    unity_cases = {row["case_id"]: row for row in unity["orientation_cases"]}
    godot_cases = {row["case_id"]: row for row in godot["orientation_cases"]}

    for case_id in ("mixed_a", "mixed_b"):
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
        transferred_vector = unity_to_godot.vector(
            _vec(unity_cases[case_id]["native_rotated_probe_vector_xyz"])
        )
        np.testing.assert_allclose(
            transferred_vector,
            _vec(godot_cases[case_id]["native_rotated_probe_vector_xyz"]),
            rtol=0.0,
            atol=2.0e-6,
            err_msg=f"Unity-to-Godot native vector transfer for {case_id}",
        )


def _native_components(value):
    """Return stored target coefficients without changing their component order."""
    return np.asarray(value, dtype=float)


def _native_euler_constructor_cases(engine, fixture, native_order):
    """Yield engine-created Euler quaternions with their native basis actions."""
    for case in fixture["cases"]:
        if engine == "unreal":
            for precision in ("float32", "float64"):
                capture = case[precision]
                yield (
                    case["case_id"],
                    precision,
                    capture["quat_make_from_euler_xyzw"],
                    capture["rotated_positive_basis_vectors_xyz"],
                )
        elif engine == "unity":
            raw = case["unity_engine_quaternion_xyzw"]
            quaternion = tuple(raw[axis] for axis in "xyzw")
            yield (
                case["case_id"],
                "float32",
                quaternion,
                case["unity_engine_rotated_positive_basis_vectors_xyz"],
            )
        elif engine == "godot":
            raw = case["quaternion_from_euler_xyzw"]
            quaternion = tuple(raw[axis] for axis in native_order)
            yield (
                case["case_id"],
                "float32",
                quaternion,
                case["quaternion_rotated_positive_basis_vectors_xyz"],
            )
        else:
            raw = case["euler_quaternion"]["wxyz"]
            yield (
                case["case_id"],
                "float32",
                raw,
                case["quaternion_rotated_positive_basis_vectors_xyz"],
            )


@pytest.mark.parametrize("engine,version,tolerance", TARGETS)
def test_native_to_canonical_uses_engine_constructed_orientations(
    engine, version, tolerance
):
    fixture = _fixture(engine, version)
    metadata, _ = _capture_rows(fixture)
    frame_map = _frame_map(metadata)
    axis_map = frame_map.world_map
    basis = np.asarray(axis_map.matrix, dtype=float)
    target_to_source = basis.T

    for case_id, precision, native_values, native_images in _native_euler_constructor_cases(
        engine, fixture, metadata["native_quaternion_order"]
    ):
        precision_tolerance = (
            2.0e-13 if precision == "float64" else tolerance
        )
        native_values = _native_components(native_values)
        canonical_wxyz = _from_native(engine, native_values, frame_map)
        canonical_matrix = Rotation.from_quat(_wxyz_to_xyzw(canonical_wxyz)).as_matrix()

        # The matrix on the native side is assembled from basis actions captured
        # inside the engine after its own Euler constructor made the quaternion.
        native_matrix = _basis_matrix_images(native_images)
        pulled_back_matrix = target_to_source @ native_matrix @ basis
        np.testing.assert_allclose(
            pulled_back_matrix,
            canonical_matrix,
            rtol=0.0,
            atol=precision_tolerance,
            err_msg=f"{engine} {version} {precision} {case_id} native-to-canonical physical action",
        )

        native_roundtrip = _to_native(engine, canonical_wxyz, frame_map)
        np.testing.assert_allclose(
            native_roundtrip,
            native_values,
            rtol=0.0,
            atol=precision_tolerance,
            err_msg=f"{engine} {version} {precision} {case_id} native quaternion roundtrip",
        )


@pytest.mark.parametrize("engine,version,tolerance", TARGETS)
def test_named_adapter_requires_an_explicit_precompiled_frame_map(
    engine, version, tolerance
):
    del version, tolerance
    adapter = ADAPTERS[engine]
    with pytest.raises(TypeError, match="precompiled QuaternionFrameMap"):
        getattr(adapter, f"to_{engine}")((1.0, 0.0, 0.0, 0.0), None)

