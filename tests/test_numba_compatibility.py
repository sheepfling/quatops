"""Compile the optional batch kernels on each supported Python version."""

from importlib.util import find_spec
from itertools import permutations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from experiments.numba_coverage import _api_catalog, _api_catalog_complete
from quatops import _numpy
from quatops.hamilton import scalar as hamilton_scalar
from quatops.jpl import scalar as jpl_scalar
from quatops.numpy.hamilton import wxyz as hamilton_wxyz
from quatops.numpy.hamilton import xyzw as hamilton_xyzw
from quatops.numpy.jpl import wxyz as jpl_wxyz
from quatops.numpy.jpl import xyzw as jpl_xyzw

NUMBA_AVAILABLE = find_spec("numba") is not None
COUNT = 128


def test_numba_api_catalog_covers_every_public_numpy_and_engine_export():
    catalog = _api_catalog()
    categorized = [
        name
        for routes in catalog.values()
        for names in routes.values()
        for name in names
    ]

    assert _api_catalog_complete()
    assert len(categorized) == len(set(categorized))


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
def test_numba_compiles_large_batch_kernels_for_both_conventions_and_layouts():
    hamilton_lhs = hamilton_scalar.normalize((0.91, 0.11, 0.31, -0.20))
    hamilton_rhs = hamilton_scalar.normalize((0.97, -0.14, 0.08, 0.17))
    jpl_lhs = jpl_scalar.from_hamilton(hamilton_lhs)
    jpl_rhs = jpl_scalar.from_hamilton(hamilton_rhs)

    multiply_cases = (
        (
            hamilton_wxyz,
            np.tile(hamilton_lhs, (COUNT, 1)),
            np.tile(hamilton_rhs, (COUNT, 1)),
            np.tile(hamilton_scalar.multiply(hamilton_lhs, hamilton_rhs), (COUNT, 1)),
        ),
        (
            hamilton_xyzw,
            np.tile(hamilton_scalar.to_xyzw(hamilton_lhs), (COUNT, 1)),
            np.tile(hamilton_scalar.to_xyzw(hamilton_rhs), (COUNT, 1)),
            np.tile(
                hamilton_scalar.to_xyzw(
                    hamilton_scalar.multiply(hamilton_lhs, hamilton_rhs)
                ),
                (COUNT, 1),
            ),
        ),
        (
            jpl_wxyz,
            np.tile(jpl_scalar.to_wxyz(jpl_lhs), (COUNT, 1)),
            np.tile(jpl_scalar.to_wxyz(jpl_rhs), (COUNT, 1)),
            np.tile(
                jpl_scalar.to_wxyz(jpl_scalar.multiply(jpl_lhs, jpl_rhs)),
                (COUNT, 1),
            ),
        ),
        (
            jpl_xyzw,
            np.tile(jpl_scalar.to_xyzw(jpl_lhs), (COUNT, 1)),
            np.tile(jpl_scalar.to_xyzw(jpl_rhs), (COUNT, 1)),
            np.tile(
                jpl_scalar.to_xyzw(jpl_scalar.multiply(jpl_lhs, jpl_rhs)),
                (COUNT, 1),
            ),
        ),
    )
    for backend, lhs, rhs, expected in multiply_cases:
        assert_allclose(backend.multiply(lhs, rhs), expected)

    vector = (1.0, -2.0, 3.0)
    rotate_cases = (
        (
            hamilton_wxyz,
            hamilton_lhs,
            vector,
            hamilton_scalar.rotate(hamilton_lhs, vector),
        ),
        (
            hamilton_xyzw,
            hamilton_scalar.to_xyzw(hamilton_lhs),
            vector,
            hamilton_scalar.rotate(hamilton_lhs, vector),
        ),
        (
            jpl_wxyz,
            jpl_scalar.to_wxyz(jpl_lhs),
            vector,
            jpl_scalar.rotate(jpl_lhs, vector),
        ),
        (
            jpl_xyzw,
            jpl_scalar.to_xyzw(jpl_lhs),
            vector,
            jpl_scalar.rotate(jpl_lhs, vector),
        ),
    )
    for backend, quaternion, input_vector, expected in rotate_cases:
        actual = backend.rotate(
            np.tile(quaternion, (COUNT, 1)), np.tile(input_vector, (COUNT, 1))
        )
        assert_allclose(actual, np.tile(expected, (COUNT, 1)))

    assert len(_numpy._NUMBA_MULTIPLY) == 4
    assert _numpy._NUMBA_ROTATE is not None


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
def test_numba_compiles_batched_rotation_matrix_kernels_for_all_layouts():
    half_turns = np.array(
        (
            np.eye(3),
            np.diag((1.0, -1.0, -1.0)),
            np.diag((-1.0, 1.0, -1.0)),
            np.diag((-1.0, -1.0, 1.0)),
            ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        )
    )
    scipy_matrices = np.concatenate(
        (half_turns, Rotation.random(COUNT - len(half_turns), random_state=7).as_matrix())
    )
    scipy_xyzw = Rotation.from_matrix(scipy_matrices).as_quat()
    hamilton_wxyz_values = np.column_stack((scipy_xyzw[:, 3], scipy_xyzw[:, :3]))
    jpl_wxyz_values = hamilton_wxyz_values.copy()
    jpl_wxyz_values[:, 1:] *= -1.0
    cases = (
        (hamilton_wxyz, hamilton_wxyz_values),
        (hamilton_xyzw, scipy_xyzw),
        (jpl_wxyz, jpl_wxyz_values),
        (jpl_xyzw, np.column_stack((jpl_wxyz_values[:, 1:], jpl_wxyz_values[:, 0]))),
    )

    for backend, quaternions in cases:
        matrices = backend.to_rotation_matrix(quaternions)
        assert_allclose(matrices, scipy_matrices, rtol=0.0, atol=2e-14)
        recovered = backend.from_rotation_matrix(scipy_matrices)
        assert_allclose(
            backend.to_rotation_matrix(recovered), scipy_matrices,
            rtol=0.0, atol=2e-14,
        )

        tolerance = 1e-4
        boundary_matrix = np.diag((np.sqrt(1.0 + tolerance), 1.0, 1.0))
        boundary_batch = np.tile(boundary_matrix, (COUNT, 1, 1))
        boundary_result = backend.from_rotation_matrix(
            boundary_batch, atol=tolerance
        )
        assert_allclose(
            backend.to_rotation_matrix(boundary_result),
            np.tile(np.eye(3), (COUNT, 1, 1)),
            rtol=0.0, atol=2e-14,
        )

        invalid_batch = np.tile(np.eye(3), (COUNT, 1, 1))
        invalid_batch[0, 2, 2] = -1.0
        with pytest.raises(ValueError, match="proper orthonormal"):
            backend.from_rotation_matrix(invalid_batch)

        zero_batch = np.tile((1.0, 0.0, 0.0, 0.0), (COUNT, 1))
        zero_batch[0] = 0.0
        with pytest.raises(ValueError, match="zero quaternion"):
            backend.to_rotation_matrix(zero_batch)
        zero_batch[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite float64"):
            backend.to_rotation_matrix(zero_batch)

    assert set(_numpy._NUMBA_ROTATION_KERNELS) == set(
        _numpy._NUMBA_ROTATION_KERNEL_NAMES
    )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
def test_batched_euler_conversions_match_scipy_for_all_sequences():
    rng = np.random.default_rng(8)
    angles = rng.uniform(-0.8, 0.8, size=(COUNT, 3))
    cases = (
        hamilton_wxyz,
        hamilton_xyzw,
        jpl_wxyz,
        jpl_xyzw,
    )
    tait_bryan = (
        "xyz", "xzy", "yxz", "yzx", "zxy", "zyx",
        "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX",
    )
    proper = tuple(
        f"{first}{middle}{first}"
        for first, middle in permutations("xyz", 2)
    )
    sequences = (*tait_bryan, *proper, *(sequence.upper() for sequence in proper))

    for sequence in sequences:
        expected = Rotation.from_euler(sequence, angles).as_matrix()
        for backend in cases:
            quaternions = backend.from_euler(angles, sequence)
            assert_allclose(
                backend.to_rotation_matrix(quaternions), expected,
                rtol=0.0, atol=3e-14,
            )
            recovered = backend.to_euler(quaternions, sequence)
            assert_allclose(
                Rotation.from_euler(sequence, recovered).as_matrix(), expected,
                rtol=0.0, atol=3e-14,
            )

        middle_angles = (
            (0.0, np.pi)
            if sequence[0].lower() == sequence[2].lower()
            else (-np.pi / 2.0, np.pi / 2.0)
        )
        for middle_angle in middle_angles:
            singular_angles = np.tile((0.4, middle_angle, -0.7), (COUNT, 1))
            expected_singular = Rotation.from_euler(
                sequence, singular_angles
            ).as_matrix()
            for backend in cases:
                quaternions = backend.from_euler(singular_angles, sequence)
                recovered = backend.to_euler(quaternions, sequence)
                assert_allclose(recovered[:, 2], 0.0, rtol=0.0, atol=1e-14)
                assert_allclose(
                    Rotation.from_euler(sequence, recovered).as_matrix(),
                    expected_singular, rtol=0.0, atol=3e-14,
                )

    assert set(_numpy._NUMBA_ROTATION_KERNELS) == set(
        _numpy._NUMBA_ROTATION_KERNEL_NAMES
    )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
def test_numba_euler_paths_preserve_axes_degrees_and_lock_status():
    count = 8192
    angles = np.zeros((3, count), dtype=np.float64)
    angles[0] = np.linspace(-25.0, 25.0, count)
    angles[1] = 90.0
    angles[2] = np.linspace(31.0, -31.0, count)
    quaternions = hamilton_xyzw.from_euler(
        angles,
        "xyz",
        degrees=True,
        angle_axis=0,
        component_axis=0,
    )
    assert quaternions.shape == (4, count)

    result = hamilton_xyzw.to_euler_with_status(
        quaternions,
        "xyz",
        degrees=True,
        gimbal_tolerance=1e-6,
        component_axis=0,
        angle_axis=0,
    )
    assert result.angles.shape == angles.shape
    np.testing.assert_array_equal(result.gimbal_lock, np.ones(count, dtype=bool))
    expected = Rotation.from_euler("xyz", angles.T, degrees=True).as_matrix()
    recovered = Rotation.from_euler("xyz", result.angles.T, degrees=True).as_matrix()
    assert_allclose(recovered, expected, rtol=0.0, atol=3e-14)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
def test_numba_compiles_batched_rotation_vector_kernels_for_all_layouts():
    def reorder(values, source_scalar_last, target_scalar_last):
        if source_scalar_last == target_scalar_last:
            return values
        if source_scalar_last:
            return values[:, (3, 0, 1, 2)]
        return values[:, (1, 2, 3, 0)]

    rng = np.random.default_rng(9)
    vectors = rng.uniform(-2.5, 2.5, size=(COUNT, 3))
    vectors[0] = (0.0, 0.0, 0.0)
    vectors[1] = (1e-300, -2e-300, 3e-300)
    vectors[2] = (np.pi, 0.0, 0.0)
    vectors[3] = (0.0, -np.pi, 0.0)
    expected_matrices = Rotation.from_rotvec(vectors).as_matrix()
    cases = (
        (hamilton_wxyz, hamilton_scalar, False, False),
        (hamilton_xyzw, hamilton_scalar, False, True),
        (jpl_wxyz, jpl_scalar, True, False),
        (jpl_xyzw, jpl_scalar, True, True),
    )

    for backend, scalar, function_scalar_last, array_scalar_last in cases:
        scalar_quaternions = np.asarray(
            [scalar.from_rotvec(tuple(value)) for value in vectors]
        )
        expected_quaternions = reorder(
            scalar_quaternions, function_scalar_last, array_scalar_last
        )
        actual_quaternions = backend.from_rotvec(vectors)
        assert_allclose(actual_quaternions, expected_quaternions, rtol=2e-13, atol=0.0)
        assert_allclose(
            backend.to_rotation_matrix(actual_quaternions),
            expected_matrices,
            rtol=0.0,
            atol=3e-14,
        )

        quaternion_batch = actual_quaternions.copy()
        quaternion_batch[6] *= 1e308
        quaternion_batch[7] *= 1e-308
        if array_scalar_last:
            quaternion_batch[8] = (1e-300, -2e-300, 3e-300, 1.0)
            quaternion_batch[9] = (1e-308, 0.0, 0.0, 0.0)
        elif function_scalar_last:
            quaternion_batch[8] = (1.0, -1e-300, 2e-300, -3e-300)
            quaternion_batch[9] = (0.0, 0.0, 0.0, 1e-308)
        else:
            quaternion_batch[8] = (1.0, 1e-300, -2e-300, 3e-300)
            quaternion_batch[9] = (1e-308, 0.0, 0.0, 0.0)

        scalar_inputs = reorder(
            quaternion_batch, array_scalar_last, function_scalar_last
        )
        expected_vectors = np.asarray(
            [scalar.to_rotvec(tuple(value)) for value in scalar_inputs]
        )
        actual_vectors = backend.to_rotvec(quaternion_batch)
        assert_allclose(actual_vectors, expected_vectors, rtol=3e-13, atol=0.0)
        assert_allclose(
            backend.to_rotation_matrix(backend.from_rotvec(actual_vectors)),
            backend.to_rotation_matrix(quaternion_batch),
            rtol=0.0,
            atol=3e-14,
        )

        zero_batch = quaternion_batch.copy()
        zero_batch[0] = 0.0
        with pytest.raises(ValueError, match="zero quaternion"):
            backend.to_rotvec(zero_batch)

        overflow_vectors = np.zeros((COUNT, 3))
        overflow_vectors[0] = (1.1e308, 1.1e308, 1.1e308)
        with pytest.raises(OverflowError, match="non-finite result"):
            backend.from_rotvec(overflow_vectors)

    assert set(_numpy._NUMBA_ROTATION_KERNELS) == set(
        _numpy._NUMBA_ROTATION_KERNEL_NAMES
    )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
def test_numba_compiles_batched_trajectory_integration_for_all_layouts():
    def reorder(values, source_scalar_last, target_scalar_last):
        if source_scalar_last == target_scalar_last:
            return values
        if source_scalar_last:
            return values[:, (3, 0, 1, 2)]
        return values[:, (1, 2, 3, 0)]

    rng = np.random.default_rng(10)
    samples = 19
    times = np.broadcast_to(np.arange(samples) * 0.01, (COUNT, samples))
    rates = rng.normal(scale=0.3, size=(COUNT, samples, 3))
    cases = (
        (hamilton_wxyz, hamilton_scalar, False, False),
        (hamilton_xyzw, hamilton_scalar, False, True),
        (jpl_wxyz, jpl_scalar, True, False),
        (jpl_xyzw, jpl_scalar, True, True),
    )

    for backend, scalar, function_scalar_last, array_scalar_last in cases:
        initial_values = np.asarray(
            [scalar.from_rotvec(tuple(value)) for value in rng.normal(size=(COUNT, 3))]
        )
        initial = reorder(initial_values, function_scalar_last, array_scalar_last)
        for frame in ("body", "world"):
            for method in ("left", "midpoint"):
                actual = backend.integrate_angular_velocity_samples(
                    initial, rates, times, frame, method
                )
                expected = np.asarray(
                    [
                        backend.integrate_angular_velocity_samples(
                            initial[row], rates[row], times[row], frame, method
                        )
                        for row in range(COUNT)
                    ]
                )
                assert_allclose(actual, expected, rtol=3e-13, atol=3e-14)

        invalid_times = times.copy()
        invalid_times[0, 1] = invalid_times[0, 0]
        with pytest.raises(ValueError, match="strictly increasing"):
            backend.integrate_angular_velocity_samples(
                initial, rates, invalid_times
            )

        zero_attitude = initial.copy()
        zero_attitude[0] = 0.0
        with pytest.raises(ValueError, match="zero quaternion"):
            backend.integrate_angular_velocity_samples(zero_attitude, rates, times)

    assert "integrate_angular_velocity" in _numpy._NUMBA_ROTATION_KERNELS


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
def test_numba_compiles_batched_nlerp_and_mrp_for_all_layouts():
    def reorder(values, source_scalar_last, target_scalar_last):
        if source_scalar_last == target_scalar_last:
            return values
        if source_scalar_last:
            return values[:, (3, 0, 1, 2)]
        return values[:, (1, 2, 3, 0)]

    rng = np.random.default_rng(11)
    mrps = rng.normal(size=(COUNT, 3))
    mrps[0] = (1.0, 0.0, 0.0)
    mrps[1] = (1e200, -2e200, 3e200)
    mrps[2] = (0.0, 0.0, 0.0)
    left_vectors = rng.normal(size=(COUNT, 3))
    right_vectors = rng.normal(size=(COUNT, 3))
    cases = (
        (hamilton_wxyz, hamilton_scalar, False, False),
        (hamilton_xyzw, hamilton_scalar, False, True),
        (jpl_wxyz, jpl_scalar, True, False),
        (jpl_xyzw, jpl_scalar, True, True),
    )

    for backend, scalar, function_scalar_last, array_scalar_last in cases:
        expected_from_mrp = reorder(
            np.asarray([scalar.from_mrp(tuple(value)) for value in mrps]),
            function_scalar_last,
            array_scalar_last,
        )
        actual_from_mrp = backend.from_mrp(mrps)
        assert_allclose(actual_from_mrp, expected_from_mrp, rtol=4e-14, atol=3e-16)

        scaled_quaternions = actual_from_mrp.copy()
        scaled_quaternions[6] *= 1e308
        scaled_quaternions[7] *= 1e-308
        scalar_inputs = reorder(
            scaled_quaternions, array_scalar_last, function_scalar_last
        )
        expected_to_mrp = np.asarray(
            [scalar.to_mrp(tuple(value)) for value in scalar_inputs]
        )
        assert_allclose(
            backend.to_mrp(scaled_quaternions),
            expected_to_mrp,
            rtol=4e-13,
            atol=4e-14,
        )

        q0 = backend.from_rotvec(left_vectors)
        q1 = backend.from_rotvec(right_vectors)
        scalar_q0 = reorder(q0, array_scalar_last, function_scalar_last)
        scalar_q1 = reorder(q1, array_scalar_last, function_scalar_last)
        for amount, shortest in ((0.31, True), (-0.2, False)):
            expected_nlerp = np.asarray(
                [
                    scalar.nlerp(tuple(a), tuple(b), amount, shortest)
                    for a, b in zip(scalar_q0, scalar_q1)
                ]
            )
            expected_nlerp = reorder(
                expected_nlerp, function_scalar_last, array_scalar_last
            )
            assert_allclose(
                backend.nlerp(q0, q1, amount, shortest),
                expected_nlerp,
                rtol=4e-13,
                atol=4e-14,
            )

        zero_quaternions = scaled_quaternions.copy()
        zero_quaternions[0] = 0.0
        with pytest.raises(ValueError, match="zero quaternion"):
            backend.to_mrp(zero_quaternions)

    assert {"nlerp", "from_mrp", "to_mrp"}.issubset(
        _numpy._NUMBA_ROTATION_KERNELS
    )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
def test_compiled_primitives_cover_batched_advanced_operations():
    rng = np.random.default_rng(12)
    count = COUNT
    vectors = rng.normal(scale=0.4, size=(count, 3))
    deltas = rng.normal(scale=0.2, size=(count, 3))
    rates = rng.normal(scale=0.3, size=(count, 3))
    translations = rng.normal(size=(count, 3))
    cases = (hamilton_wxyz, hamilton_xyzw, jpl_wxyz, jpl_xyzw)

    for backend in cases:
        q0 = backend.from_rotvec(vectors)
        q1 = backend.from_rotvec(vectors + deltas)
        control_a = backend.from_rotvec(vectors - deltas)
        control_b = backend.from_rotvec(vectors + 2.0 * deltas)

        for operation in ("to_left_matrix", "to_right_matrix"):
            matrix_operation = getattr(backend, operation)
            actual_matrix = matrix_operation(q0)
            expected_matrix = np.asarray([matrix_operation(value) for value in q0])
            assert_allclose(actual_matrix, expected_matrix, rtol=0.0, atol=0.0)

        for frame in ("body", "world"):
            derivative = backend.quaternion_derivative(q0, rates, frame)
            expected_derivative = np.asarray(
                [backend.quaternion_derivative(q0[i], rates[i], frame) for i in range(count)]
            )
            assert_allclose(derivative, expected_derivative, rtol=2e-14, atol=2e-14)

            plus = backend.boxplus(q0, deltas, frame)
            expected_plus = np.asarray(
                [backend.boxplus(q0[i], deltas[i], frame) for i in range(count)]
            )
            assert_allclose(plus, expected_plus, rtol=3e-13, atol=3e-14)

            minus = backend.boxminus(q1, q0, frame)
            expected_minus = np.asarray(
                [backend.boxminus(q1[i], q0[i], frame) for i in range(count)]
            )
            assert_allclose(minus, expected_minus, rtol=3e-13, atol=3e-14)

        actual_squad = backend.squad(q0, q1, control_a, control_b, 0.37)
        expected_squad = np.asarray(
            [
                backend.squad(q0[i], q1[i], control_a[i], control_b[i], 0.37)
                for i in range(count)
            ]
        )
        assert_allclose(actual_squad, expected_squad, rtol=3e-13, atol=3e-14)

        for shortest in (True, False):
            actual_slerp = backend.slerp(q0, q1, 0.37, shortest)
            expected_slerp = np.asarray(
                [
                    backend.slerp(q0[i], q1[i], 0.37, shortest)
                    for i in range(count)
                ]
            )
            assert_allclose(actual_slerp, expected_slerp, rtol=3e-13, atol=3e-14)

        homogeneous = backend.to_homogeneous_matrix(q0, translations)
        expected_homogeneous = np.asarray(
            [backend.to_homogeneous_matrix(q0[i], translations[i]) for i in range(count)]
        )
        assert_allclose(homogeneous, expected_homogeneous, rtol=0.0, atol=3e-14)
        recovered = backend.from_homogeneous_matrix(homogeneous)
        assert_allclose(
            backend.to_rotation_matrix(recovered),
            backend.to_rotation_matrix(q0),
            rtol=0.0,
            atol=3e-14,
        )
        assert_allclose(backend.so3_adjoint(q0), backend.to_rotation_matrix(q0))

    assert _numpy._NUMBA_ROTATION_KERNELS["quaternion_matrix"].nopython_signatures


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
def test_numba_compiles_canonicalize_commutator_and_two_vector_batches():
    rng = np.random.default_rng(94)
    count = COUNT
    hamilton_values = rng.normal(size=(count, 4))
    hamilton_values[0] = 0.0
    hamilton_values[1] = (0.0, -1.0, 0.0, 0.0)
    hamilton_values[2] = (-1.0, 0.0, 0.0, 0.0)
    hamilton_values[3] = (-0.5, 0.2, -0.3, 0.4)
    left_vectors = rng.normal(size=(count, 3))
    right_vectors = rng.normal(size=(count, 3))
    left_vectors[0] = (1.0, 0.0, 0.0)
    right_vectors[0] = (1.0, 0.0, 0.0)
    left_vectors[1] = (1.0, 0.0, 0.0)
    right_vectors[1] = (-1.0, 0.0, 0.0)
    left_vectors[2] = (1e308, -1e308, 1e308)
    right_vectors[2] = (-1e308, 1e308, 1e308)
    left_vectors[3] = (1e-300, 2e-300, -1e-300)
    right_vectors[3] = (-2e-300, 1e-300, 3e-300)
    left_vectors[4] = (0.2, 0.5, 0.7)
    right_vectors[4] = (-0.2 + 1e-12, -0.5, -0.7)
    cases = (hamilton_wxyz, hamilton_xyzw, jpl_wxyz, jpl_xyzw)

    for backend in cases:
        values = backend.from_wxyz(hamilton_values)
        other = backend.from_rotvec(rng.normal(scale=0.3, size=(count, 3)))

        canonical = backend.canonicalize(values)
        expected_canonical = np.asarray(
            [backend.canonicalize(value) for value in values]
        )
        assert_allclose(canonical, expected_canonical, rtol=0.0, atol=0.0)

        commutator = backend.commutator(values, other)
        expected_commutator = np.asarray(
            [backend.commutator(left, right) for left, right in zip(values, other)]
        )
        assert_allclose(commutator, expected_commutator, rtol=0.0, atol=0.0)

        rotations = backend.from_two_vectors(left_vectors, right_vectors)
        expected_rotations = np.asarray(
            [
                backend.from_two_vectors(left, right)
                for left, right in zip(left_vectors, right_vectors)
            ]
        )
        assert_allclose(rotations, expected_rotations, rtol=2e-15, atol=2e-15)

        zero_vectors = left_vectors.copy()
        zero_vectors[0] = 0.0
        with pytest.raises(ValueError, match="axis must be nonzero"):
            backend.from_two_vectors(zero_vectors, right_vectors)

    for operation in ("canonicalize", "commutator", "from_two_vectors"):
        assert _numpy._NUMBA_ROTATION_KERNELS[operation].nopython_signatures


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
def test_numba_compiles_remaining_coefficient_and_sampling_helpers():
    rng = np.random.default_rng(2612)
    count = COUNT
    hamilton_values = rng.normal(size=(count, 4))
    other_values = rng.normal(size=(count, 4))
    cases = (
        (hamilton_wxyz, False, 1),
        (hamilton_xyzw, True, 1),
        (jpl_wxyz, False, -1),
        (jpl_xyzw, True, -1),
    )

    for backend, scalar_last, epsilon in cases:
        values = (
            hamilton_values[:, (1, 2, 3, 0)]
            if scalar_last
            else hamilton_values
        )
        other = (
            other_values[:, (1, 2, 3, 0)]
            if scalar_last
            else other_values
        )
        vector_start = 0 if scalar_last else 1
        expected_conjugate = values.copy()
        expected_conjugate[:, vector_start : vector_start + 3] *= -1.0
        assert_allclose(backend.add(values, other), values + other, rtol=0.0, atol=0.0)
        assert_allclose(backend.subtract(values, other), values - other, rtol=0.0, atol=0.0)
        assert_allclose(backend.conjugate(values), expected_conjugate, rtol=0.0, atol=0.0)
        assert_allclose(backend.negate(values), -values, rtol=0.0, atol=0.0)
        assert_allclose(backend.scale(values, -1.75), values * -1.75, rtol=0.0, atol=0.0)
        assert_allclose(backend.scalar_part(values), values[:, 3 if scalar_last else 0])
        assert_allclose(
            backend.vector_part(values),
            values[:, :3] if scalar_last else values[:, 1:],
            rtol=0.0,
            atol=0.0,
        )

        assert_allclose(backend.from_xyzw(backend.to_xyzw(values)), values, rtol=0.0, atol=0.0)
        assert_allclose(backend.from_wxyz(backend.to_wxyz(values)), values, rtol=0.0, atol=0.0)

        if epsilon < 0:
            expected_jpl_wxyz = hamilton_values.copy()
            expected_jpl_wxyz[:, 1:] *= -1.0
            expected_jpl_xyzw = expected_jpl_wxyz[:, (1, 2, 3, 0)]
            expected_from_hamilton = expected_jpl_xyzw if scalar_last else expected_jpl_wxyz
            assert_allclose(backend.from_hamilton(hamilton_values), expected_from_hamilton)
            assert_allclose(
                backend.to_hamilton(expected_from_hamilton),
                hamilton_values,
                rtol=0.0,
                atol=0.0,
            )

        uniforms = np.random.default_rng(428).random((count, 3))
        u1, u2, u3 = uniforms.T
        x = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
        y = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
        z = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
        scalar = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
        if epsilon < 0:
            x, y, z = -x, -y, -z
        expected_random = (
            np.column_stack((x, y, z, scalar))
            if scalar_last
            else np.column_stack((scalar, x, y, z))
        )
        random_quaternions = backend.random_quaternion(
            size=count, rng=np.random.default_rng(428)
        )
        random_rotations = backend.random_rotation(
            size=count, rng=np.random.default_rng(428)
        )
        assert_allclose(random_quaternions, expected_random, rtol=3e-15, atol=3e-15)
        assert_allclose(random_rotations, expected_random, rtol=3e-15, atol=3e-15)

    for operation in (
        "conjugate",
        "component_transform",
        "vector_part",
        "random_quaternion",
    ):
        assert _numpy._NUMBA_ROTATION_KERNELS[operation].nopython_signatures


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
def test_numba_compiles_batched_norm_inverse_and_dot_for_all_layouts():
    def reorder(values, source_scalar_last, target_scalar_last):
        if source_scalar_last == target_scalar_last:
            return values
        if source_scalar_last:
            return values[:, (3, 0, 1, 2)]
        return values[:, (1, 2, 3, 0)]

    rng = np.random.default_rng(13)
    cases = (
        (hamilton_wxyz, hamilton_scalar, False, False),
        (hamilton_xyzw, hamilton_scalar, False, True),
        (jpl_wxyz, jpl_scalar, True, False),
        (jpl_xyzw, jpl_scalar, True, True),
    )
    for backend, scalar, function_scalar_last, array_scalar_last in cases:
        first = backend.from_rotvec(rng.normal(size=(COUNT, 3)))
        second = backend.from_rotvec(rng.normal(size=(COUNT, 3)))
        first[6] *= 1e308
        first[7] *= 1e-308
        first[8] *= 1e-200
        second[8] *= 1e-200
        scalar_first = reorder(first, array_scalar_last, function_scalar_last)
        scalar_second = reorder(second, array_scalar_last, function_scalar_last)

        expected_norm = np.asarray(
            [scalar.norm(tuple(value)) for value in scalar_first]
        )
        expected_normalized = reorder(
            np.asarray([scalar.normalize(tuple(value)) for value in scalar_first]),
            function_scalar_last,
            array_scalar_last,
        )
        expected_inverse = reorder(
            np.asarray([scalar.inverse(tuple(value)) for value in scalar_first]),
            function_scalar_last,
            array_scalar_last,
        )
        expected_dot = np.asarray(
            [
                scalar.dot(tuple(a), tuple(b))
                for a, b in zip(scalar_first, scalar_second)
            ]
        )

        assert_allclose(backend.norm(first), expected_norm, rtol=3e-14, atol=0.0)
        assert_allclose(
            backend.normalize(first), expected_normalized, rtol=3e-14, atol=0.0
        )
        assert_allclose(backend.inverse(first), expected_inverse, rtol=4e-14, atol=0.0)
        assert_allclose(backend.dot(first, second), expected_dot, rtol=4e-14, atol=0.0)

        cancelling_left = np.tile((1e308, 1e308, 1e308, 1e308), (COUNT, 1))
        cancelling_right = np.tile((1e308, -1e308, 1e308, -1e308), (COUNT, 1))
        assert_allclose(backend.dot(cancelling_left, cancelling_right), np.zeros(COUNT))

        zero_batch = first.copy()
        zero_batch[0] = 0.0
        with pytest.raises(ValueError, match="zero quaternion"):
            backend.normalize(zero_batch)
        with pytest.raises(ValueError, match="zero quaternion"):
            backend.inverse(zero_batch)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
def test_numba_coverage_report_confirms_compiled_signatures():
    from experiments.numba_coverage import collect_numba_coverage

    report = collect_numba_coverage()
    uncompiled = [
        f"{kernel['name']} ({kernel['state']})"
        for kernel in report["kernels"]
        if kernel["state"] != "compiled"
    ]
    assert report["complete"], f"Numba dispatchers are not compiled: {uncompiled}"
    assert report["summary"]["compiled"] == report["summary"]["total"]
    assert report["summary"]["total"] == len(report["kernels"])
