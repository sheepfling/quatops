"""Tests for common vectorized target conversion."""

import numpy as np
from scipy.spatial.transform import Rotation

from quatops.experimental.converters import (
    CoordinateFrame,
    FrameCorrespondence,
    FramedOrientationBatch,
    SemanticQuaternionMap,
    get_converter,
)
from quatops.experimental.engines import OrthogonalBasisMap, SignedAxisMap
from quatops.python import hamilton


def _semantic_map(general=False):
    identity = SignedAxisMap(("x", "y", "z"))
    source_world = CoordinateFrame(
        "source.world", "world-reference", ("right", "forward", "up"), identity
    )
    source_body = CoordinateFrame(
        "source.body", "body-reference", ("right", "forward", "up"), identity
    )
    if general:
        world = Rotation.from_rotvec((0.2, -0.3, 0.4)).as_matrix()
        body = Rotation.from_rotvec((-0.1, 0.35, 0.17)).as_matrix()
        target_world = CoordinateFrame(
            "target.world",
            "world-reference",
            ("x", "y", "z"),
            OrthogonalBasisMap(world.T),
        )
        target_body = CoordinateFrame(
            "target.body",
            "body-reference",
            ("x", "y", "z"),
            OrthogonalBasisMap(body.T),
        )
    else:
        target_world = CoordinateFrame(
            "target.world", "world-reference", ("x", "y", "z"), identity
        )
        target_body = CoordinateFrame(
            "target.body", "body-reference", ("x", "y", "z"), identity
        )
    return SemanticQuaternionMap(
        FrameCorrespondence(source_world, target_world),
        FrameCorrespondence(source_body, target_body),
    )


def _batch():
    return np.asarray(
        [
            hamilton.from_rotvec((0.4, -0.1, 0.7)),
            hamilton.from_rotvec((-0.2, 0.9, 0.3)),
            hamilton.from_rotvec((0.6, 0.3, -0.5)),
        ],
        dtype=np.float64,
    )


def test_vectorized_scalar_last_and_scalar_first_profiles_roundtrip():
    mapping = _semantic_map()
    source = FramedOrientationBatch(_batch(), "source.body", "source.world")
    for converter_id in (
        "unity.quaternion",
        "blender.mathutils.quaternion",
        "opencv.rodrigues",
        "openxr.xrquaternionf",
    ):
        profile = get_converter(converter_id)
        target = profile.to_target_array(source, mapping)
        recovered = profile.from_target_array(target, mapping)
        assert recovered.values.shape == source.values.shape
        for actual, expected in zip(recovered.values, source.values):
            assert hamilton.rotations_equivalent(actual, expected, atol=2e-12)


def test_vectorized_conversion_preserves_component_axis_and_float32_fast_path():
    mapping = _semantic_map()
    source_values = _batch().astype(np.float32).T
    source = FramedOrientationBatch(source_values, "source.body", "source.world")
    profile = get_converter("unity.quaternion")
    target = profile.to_target_array(source, mapping, component_axis=0)
    assert target.values.shape == source_values.shape
    assert target.values.dtype == np.float32
    recovered = profile.from_target_array(target, mapping, component_axis=0)
    np.testing.assert_array_equal(recovered.values, source_values)


def test_vectorized_dtype_policy_for_general_maps_and_rotation_vectors():
    source_values = _batch().astype(np.float32)
    source = FramedOrientationBatch(source_values, "source.body", "source.world")

    general = get_converter("openusd.gfquatd")
    general_target = general.to_target_array(source, _semantic_map(general=True))
    assert general_target.values.dtype == np.float64

    rotation_vector = get_converter("opencv.rodrigues")
    vector_target = rotation_vector.to_target_array(source, _semantic_map())
    assert vector_target.values.dtype == np.float64
    vector_recovered = rotation_vector.from_target_array(vector_target, _semantic_map())
    assert vector_recovered.values.dtype == np.float64


def test_vectorized_antipodes_keep_sign_for_quaternion_and_collapse_for_rotvec():
    quaternion = np.asarray((0.8, 0.2, -0.4, 0.4), dtype=np.float64)
    source_values = np.stack((quaternion, -quaternion))
    source = FramedOrientationBatch(source_values, "source.body", "source.world")
    mapping = _semantic_map()

    quaternion_profile = get_converter("unity.quaternion")
    quaternion_target = quaternion_profile.to_target_array(source, mapping)
    np.testing.assert_array_equal(
        quaternion_target.values[1], -quaternion_target.values[0]
    )

    rotation_vector_profile = get_converter("opencv.rodrigues")
    rotation_vector_target = rotation_vector_profile.to_target_array(source, mapping)
    np.testing.assert_array_equal(
        rotation_vector_target.values[1], rotation_vector_target.values[0]
    )
    recovered = rotation_vector_profile.from_target_array(
        rotation_vector_target, mapping
    )
    np.testing.assert_array_equal(recovered.values[1], recovered.values[0])


def test_vectorized_two_frame_maps_apply_compiled_world_and_body_corrections():
    mapping = _semantic_map(general=True)
    values = _batch()
    source = FramedOrientationBatch(values, "source.body", "source.world")
    profile = get_converter("openusd.gfquatd")
    target = profile.to_target_array(source, mapping)
    recovered = profile.from_target_array(target, mapping)
    for actual, expected in zip(recovered.values, values):
        assert hamilton.rotations_equivalent(actual, expected, atol=5e-13)


def test_vectorized_converter_rejects_mismatched_frame_labels():
    mapping = _semantic_map()
    source = FramedOrientationBatch(_batch(), "camera", "source.world")
    try:
        get_converter("unity.quaternion").to_target_array(source, mapping)
    except ValueError as error:
        assert "child frame" in str(error)
    else:
        raise AssertionError("expected mismatched frame label to be rejected")
