"""Tests for the public physical-orientation target gateway."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quatops.converters import (
    CONVERTERS,
    OrientationConverter,
    TargetOrientation,
    TargetOrientationBatch,
    convert_orientation_batch_from_target,
    convert_orientation_batch_to_target,
    convert_orientation_from_target,
    convert_orientation_to_target,
)
from quatops.engines import OrthogonalBasisMap, SignedAxisMap
from quatops.python import hamilton
from quatops.semantics import (
    CANONICAL_BODY_RFU,
    CANONICAL_WORLD_RFU,
    CoordinateFrame,
    PhysicalOrientation,
    PhysicalOrientationBatch,
)
from quatops.types import HamiltonQuaternion


def _source_and_target_frames():
    source_world = CANONICAL_WORLD_RFU.bind(
        "application.world.rfu", "application.world.axes"
    )
    source_body = CANONICAL_BODY_RFU.bind("asset.body.rfu", "asset.body.axes")
    target_basis = OrthogonalBasisMap(
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
    )
    target_world = CoordinateFrame(
        "application.unity-world",
        source_world.reference,
        ("right", "up", "forward"),
        target_basis,
        "left",
        role="world",
        reference_convention=source_world.reference_convention,
    )
    target_body = CoordinateFrame(
        "asset.unity-local",
        source_body.reference,
        ("right", "up", "forward"),
        target_basis,
        "left",
        role="body",
        reference_convention=source_body.reference_convention,
    )
    return source_world, source_body, target_world, target_body, target_basis


def _target_rotation_matrix(target):
    if target.representation == "rotation_vector":
        return Rotation.from_rotvec(target.values).as_matrix()
    if target.component_order == "xyzw":
        return Rotation.from_quat(target.values).as_matrix()
    return Rotation.from_quat(hamilton.to_xyzw(target.values)).as_matrix()


def test_scalar_gateway_matches_explicit_basis_action_and_roundtrips():
    source_world, source_body, target_world, target_body, basis = (
        _source_and_target_frames()
    )
    source_wxyz = (0.8, 0.2, -0.4, 0.4)
    physical = PhysicalOrientation(
        HamiltonQuaternion(*source_wxyz), source_world, source_body
    )

    target = convert_orientation_to_target(
        physical, "unity.quaternion", target_world, target_body
    )
    expected_xyzw = (-source_wxyz[1], -source_wxyz[3], -source_wxyz[2], source_wxyz[0])
    np.testing.assert_allclose(target.values, expected_xyzw, rtol=0.0, atol=1e-15)
    assert (target.from_frame, target.to_frame) == (
        target_body.name,
        target_world.name,
    )

    source_rotation = Rotation.from_quat(hamilton.to_xyzw(source_wxyz)).as_matrix()
    basis_matrix = np.asarray(basis.matrix)
    expected_target_matrix = basis_matrix @ source_rotation @ basis_matrix.T
    target_rotation = Rotation.from_quat(target.values).as_matrix()
    np.testing.assert_allclose(
        target_rotation, expected_target_matrix, rtol=0.0, atol=2e-15
    )
    vector = np.asarray((0.31, -0.72, 0.48))
    np.testing.assert_allclose(
        basis_matrix @ (source_rotation @ vector),
        target_rotation @ (basis_matrix @ vector),
        rtol=0.0,
        atol=2e-15,
    )

    recovered = convert_orientation_from_target(
        target,
        "unity.quaternion",
        source_world,
        source_body,
        target_world,
        target_body,
    )
    assert recovered.parent_frame == source_world
    assert recovered.child_frame == source_body
    assert hamilton.rotations_equivalent(recovered.quaternion_wxyz, source_wxyz)


@pytest.mark.parametrize("profile", tuple(CONVERTERS.values()), ids=CONVERTERS.keys())
def test_all_profiles_preserve_action_with_distinct_parent_and_child_maps(profile):
    source_world = CANONICAL_WORLD_RFU.bind("source.world", "world.reference")
    source_body = CANONICAL_BODY_RFU.bind("source.body", "body.reference")
    parent_map = np.diag((-1.0, 1.0, 1.0))
    child_map = np.diag((1.0, -1.0, 1.0))
    target_world = CoordinateFrame(
        "target.world",
        source_world.reference,
        ("left", "forward", "up"),
        OrthogonalBasisMap(parent_map.T),
        "left",
        role="world",
        reference_convention=source_world.reference_convention,
    )
    target_body = CoordinateFrame(
        "target.body",
        source_body.reference,
        ("right", "back", "up"),
        OrthogonalBasisMap(child_map.T),
        "left",
        role="body",
        reference_convention=source_body.reference_convention,
    )
    converter = OrientationConverter(
        profile, source_world, source_body, target_world, target_body
    )
    sample_vector = np.asarray((0.31, -0.72, 0.48))

    for quaternion in (
        (1.0, 0.0, 0.0, 0.0),
        (0.8, 0.2, -0.4, 0.4),
    ):
        physical = PhysicalOrientation.from_wxyz(quaternion, source_world, source_body)
        native = converter.to_target(physical)
        native_rotation = _target_rotation_matrix(native)
        source_rotation = Rotation.from_quat(hamilton.to_xyzw(quaternion)).as_matrix()
        expected_target_rotation = parent_map @ source_rotation @ child_map.T

        np.testing.assert_allclose(
            native_rotation, expected_target_rotation, rtol=0.0, atol=3e-14
        )
        np.testing.assert_allclose(
            native_rotation @ (child_map @ sample_vector),
            parent_map @ (source_rotation @ sample_vector),
            rtol=0.0,
            atol=3e-14,
        )


def test_different_frame_maps_reject_opposite_determinant_parity():
    source_world, source_body, _, _, _ = _source_and_target_frames()
    reflected_world = CoordinateFrame(
        "target.reflected-world",
        source_world.reference,
        ("left", "forward", "up"),
        OrthogonalBasisMap(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        "left",
        role="world",
        reference_convention=source_world.reference_convention,
    )
    unchanged_body = CoordinateFrame(
        "target.right-body",
        source_body.reference,
        ("right", "forward", "up"),
        OrthogonalBasisMap(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        "right",
        role="body",
        reference_convention=source_body.reference_convention,
    )

    with pytest.raises(ValueError, match="same determinant"):
        OrientationConverter(
            "unity.quaternion",
            source_world,
            source_body,
            reflected_world,
            unchanged_body,
        )


def test_noncommuting_composition_cancels_intermediate_frame_maps():
    identity = SignedAxisMap(("x", "y", "z"))
    reflections = (
        OrthogonalBasisMap(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        OrthogonalBasisMap(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))),
        OrthogonalBasisMap(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))),
    )
    source_frames = tuple(
        CoordinateFrame(
            f"source.{name}",
            f"{name}.reference",
            ("x", "y", "z"),
            identity,
            "right",
        )
        for name in ("a", "b", "c")
    )
    target_frames = tuple(
        CoordinateFrame(
            f"target.{name}",
            source.reference,
            ("left", "y", "z"),
            basis,
            "left",
        )
        for name, source, basis in zip(("a", "b", "c"), source_frames, reflections)
    )
    source_a, source_b, source_c = source_frames
    target_a, target_b, target_c = target_frames
    first = PhysicalOrientation.from_wxyz(
        hamilton.from_rotation_vector((0.37, -0.21, 0.53)), source_a, source_b
    )
    second = PhysicalOrientation.from_wxyz(
        hamilton.from_rotation_vector((-0.16, 0.62, 0.29)), source_b, source_c
    )
    profile = "unity.quaternion"
    a_from_b = OrientationConverter(profile, source_a, source_b, target_a, target_b)
    b_from_c = OrientationConverter(profile, source_b, source_c, target_b, target_c)
    a_from_c = OrientationConverter(profile, source_a, source_c, target_a, target_c)

    native_first = a_from_b.to_target(first)
    native_second = b_from_c.to_target(second)
    native_composed = a_from_c.to_target(first.compose(second))
    expected = _target_rotation_matrix(native_first) @ _target_rotation_matrix(
        native_second
    )
    actual = _target_rotation_matrix(native_composed)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-14)


def test_reusable_converter_binds_frames_and_serves_scalar_and_batch_values():
    source_world, source_body, target_world, target_body, _ = (
        _source_and_target_frames()
    )
    converter = OrientationConverter(
        "unity.quaternion",
        source_parent=source_world,
        source_child=source_body,
        target_parent=target_world,
        target_child=target_body,
    )
    scalar = PhysicalOrientation(
        HamiltonQuaternion(0.8, 0.2, -0.4, 0.4), source_world, source_body
    )
    native_scalar = converter.to_target(scalar)
    recovered_scalar = converter.from_target(native_scalar)
    assert converter.converter_id == "unity.quaternion"
    assert hamilton.rotations_equivalent(
        recovered_scalar.quaternion_wxyz, scalar.quaternion_wxyz
    )

    batch = PhysicalOrientationBatch(
        np.asarray([scalar.quaternion_wxyz, scalar.quaternion_wxyz]),
        source_world,
        source_body,
    )
    native_batch = converter.to_target_batch(batch)
    recovered_batch = converter.from_target_batch(native_batch)
    assert recovered_batch.values.shape == (2, 4)
    for actual in recovered_batch.values:
        assert hamilton.rotations_equivalent(actual, scalar.quaternion_wxyz)

    alternate_world = CoordinateFrame(
        source_world.name,
        source_world.reference,
        source_world.axes,
        OrthogonalBasisMap(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        source_world.handedness,
        source_world.role,
        source_world.reference_convention,
    )
    different_source = PhysicalOrientation(
        scalar.quaternion, alternate_world, source_body
    )
    with pytest.raises(ValueError, match="parent frame does not match"):
        converter.to_target(different_source)


def test_batch_gateway_preserves_component_axis_and_roundtrips():
    source_world, source_body, target_world, target_body, _ = (
        _source_and_target_frames()
    )
    source_values = np.asarray(
        [
            (0.8, 0.2, -0.4, 0.4),
            (0.7, -0.3, 0.2, 0.6164414002968976),
        ],
        dtype=np.float64,
    ).T
    physical = PhysicalOrientationBatch(source_values, source_world, source_body)

    native = convert_orientation_batch_to_target(
        physical,
        "unity.quaternion",
        target_world,
        target_body,
        component_axis=0,
    )
    assert native.values.shape == source_values.shape
    assert native.from_frame == target_body.name
    assert native.to_frame == target_world.name

    recovered = convert_orientation_batch_from_target(
        native,
        "unity.quaternion",
        source_world,
        source_body,
        target_world,
        target_body,
        component_axis=0,
    )
    assert recovered.values.shape == source_values.shape
    for actual, expected in zip(recovered.values.T, source_values.T):
        assert hamilton.rotations_equivalent(actual, expected, atol=2e-12)


def test_batch_gateway_rejects_non_unit_physical_values():
    source_world, source_body, target_world, target_body, _ = (
        _source_and_target_frames()
    )
    physical = PhysicalOrientationBatch(
        np.asarray(((2.0, 0.0, 0.0, 0.0),)), source_world, source_body
    )

    with pytest.raises(ValueError, match="unit quaternions"):
        convert_orientation_batch_to_target(
            physical, "unity.quaternion", target_world, target_body
        )


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        (np.asarray(((np.nan, 0.0, 0.0, 1.0),)), ValueError, "finite"),
        (
            np.asarray(((1.0 + 0.0j, 0.0, 0.0, 0.0),)),
            TypeError,
            "real numbers",
        ),
        (np.asarray(((1.0, 0.0, 0.0),)), ValueError, "length 4"),
    ],
)
def test_batch_gateway_rejects_nonfinite_complex_and_misshaped_values(
    values, error, message
):
    source_world, source_body, target_world, target_body, _ = (
        _source_and_target_frames()
    )
    physical = PhysicalOrientationBatch(values, source_world, source_body)
    converter = OrientationConverter(
        "unity.quaternion", source_world, source_body, target_world, target_body
    )

    with pytest.raises(error, match=message):
        converter.to_target_batch(physical)


def test_reverse_gateway_rejects_non_unit_target_quaternion_values():
    source_world, source_body, target_world, target_body, _ = (
        _source_and_target_frames()
    )
    converter = OrientationConverter(
        "unity.quaternion", source_world, source_body, target_world, target_body
    )
    scalar = converter.to_target(
        PhysicalOrientation.from_wxyz((0.8, 0.2, -0.4, 0.4), source_world, source_body)
    )
    invalid_scalar = TargetOrientation(
        converter_id=scalar.converter_id,
        values=(2.0, 0.0, 0.0, 0.0),
        representation=scalar.representation,
        component_order=scalar.component_order,
        from_frame=scalar.from_frame,
        to_frame=scalar.to_frame,
    )
    with pytest.raises(ValueError, match="unit quaternion"):
        converter.from_target(invalid_scalar)

    invalid_batch = TargetOrientationBatch(
        converter_id=scalar.converter_id,
        values=np.asarray(((2.0, 0.0, 0.0, 0.0),)),
        representation=scalar.representation,
        component_order=scalar.component_order,
        from_frame=scalar.from_frame,
        to_frame=scalar.to_frame,
    )
    with pytest.raises(ValueError, match="unit quaternions"):
        converter.from_target_batch(invalid_batch)
