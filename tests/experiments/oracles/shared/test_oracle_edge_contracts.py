"""Boundary and empty/batch contracts across all public representations."""

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.support.oracle_support import (
    BACKENDS,
    from_wxyz,
    orient_matrices,
    orient_vectors,
    to_wxyz,
)


def _empty_quaternions(module, is_numpy: bool, component_first: bool):
    if is_numpy:
        empty = np.empty((0, 4))
        return from_wxyz(module, empty, True, component_first)
    return ()


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_zero_and_singular_quaternion_contracts(backend_name, module, is_jpl,
                                                is_numpy, component_first):
    zero = from_wxyz(module, np.zeros(4), is_numpy, component_first)
    with pytest.raises(ValueError):
        module.normalize(zero)
    with pytest.raises(ValueError):
        module.inverse(zero)
    with pytest.raises(ValueError):
        module.log(zero)
    with pytest.raises(ValueError):
        module.to_rotation_matrix(zero)
    with pytest.raises(ValueError):
        module.to_rotation_vector(zero)
    np.testing.assert_array_equal(to_wxyz(module, module.canonicalize(zero),
                                          is_numpy, component_first), np.zeros(4))

    with pytest.raises(ValueError):
        module.from_two_vectors((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    source = np.array(((1.0, 0.0, 0.0),))
    target = np.array(((0.0, 1.0, 0.0),))
    with pytest.raises(ValueError):
        module.align_vectors(
            orient_vectors(source, component_first) if is_numpy else source,
            orient_vectors(target, component_first) if is_numpy else target,
        )

    empty = _empty_quaternions(module, is_numpy, component_first)
    with pytest.raises(ValueError):
        module.rotation_mean(empty)
    with pytest.raises(ValueError):
        module.rotation_covariance(empty)

    singleton = (module.from_wxyz((1.0, 0.0, 0.0, 0.0)) if not is_numpy
                 else from_wxyz(module, np.array(((1.0, 0.0, 0.0, 0.0),)),
                                True, component_first))
    with pytest.raises(ValueError):
        module.rotation_mean((singleton,) if not is_numpy else singleton, weights=(0.0,))


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_rotation_and_matrix_singular_cases_are_rejected(
    backend_name, module, is_jpl, is_numpy, component_first
):
    with pytest.raises(ValueError):
        module.from_wxyz((1.0, 2.0, 3.0))

    reflection = np.diag((1.0, 1.0, -1.0))
    matrix_input = orient_matrices(reflection, component_first) if is_numpy else reflection
    with pytest.raises(ValueError):
        module.from_rotation_matrix(matrix_input)

    bad_homogeneous = np.eye(4)
    bad_homogeneous[3, 0] = 0.01
    homogeneous_input = orient_matrices(bad_homogeneous, component_first) if is_numpy \
        else bad_homogeneous
    with pytest.raises(ValueError):
        module.from_homogeneous_matrix(homogeneous_input)

    identity = module.from_wxyz((1.0, 0.0, 0.0, 0.0))
    negative_identity = module.from_wxyz((-1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        module.slerp(identity, negative_identity, 0.5, shortest=False)


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_matrix_conversion_tolerances_accept_only_documented_noise(
    backend_name, module, is_jpl, is_numpy, component_first
):
    near_rotation = np.eye(3)
    near_rotation[0, 0] += 2e-5
    matrix_input = orient_matrices(near_rotation, component_first)
    homogeneous = np.eye(4)
    homogeneous[:3, :3] = near_rotation
    homogeneous_input = orient_matrices(homogeneous, component_first)

    for tolerance in (-1.0, 1.0):
        with pytest.raises(ValueError):
            module.from_rotation_matrix(matrix_input, atol=tolerance)
        with pytest.raises(ValueError):
            module.from_homogeneous_matrix(homogeneous_input, atol=tolerance)
    with pytest.raises(ValueError):
        module.from_rotation_matrix(matrix_input, atol=1e-6)
    with pytest.raises(ValueError):
        module.from_homogeneous_matrix(homogeneous_input, atol=1e-6)

    accepted = module.from_rotation_matrix(matrix_input, atol=1e-4)
    accepted_homogeneous = module.from_homogeneous_matrix(homogeneous_input, atol=1e-4)
    expected_rotation = Rotation.from_matrix(near_rotation).as_matrix()
    np.testing.assert_allclose(module.to_rotation_matrix(accepted), expected_rotation,
                               rtol=0, atol=3e-5,
                               err_msg=f"{backend_name}.from_rotation_matrix atol")
    np.testing.assert_allclose(module.to_rotation_matrix(accepted_homogeneous),
                               expected_rotation, rtol=0, atol=3e-5,
                               err_msg=f"{backend_name}.from_homogeneous_matrix atol")


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_rotation_vector_zero_small_and_half_turn_limits(
    backend_name, module, is_jpl, is_numpy, component_first
):
    vectors = np.array(((0.0, 0.0, 0.0), (1e-11, -2e-11, 3e-11),
                        (0.0, 0.0, math.pi)))
    vector_input = orient_vectors(vectors, component_first) if is_numpy else vectors
    if is_numpy:
        q = module.from_rotation_vector(vector_input)
        coefficients = to_wxyz(module, q, True, component_first)
        actual_rotvecs = np.moveaxis(module.to_rotation_vector(q), 0, -1) \
            if component_first else module.to_rotation_vector(q)
        actual_matrices = (np.moveaxis(module.to_rotation_matrix(q), 2, 0)
                           if component_first else module.to_rotation_matrix(q))
    else:
        q = [module.from_rotation_vector(tuple(vector)) for vector in vectors]
        coefficients = np.stack([module.to_wxyz(value) for value in q])
        actual_rotvecs = np.stack([tuple(module.to_rotation_vector(value)) for value in q])
        actual_matrices = np.stack([module.to_rotation_matrix(value) for value in q])

    np.testing.assert_allclose(np.linalg.norm(coefficients, axis=-1), 1.0, rtol=0, atol=3e-15)
    expected_matrices = Rotation.from_rotvec(vectors).as_matrix()
    np.testing.assert_allclose(actual_matrices, expected_matrices, rtol=4e-14, atol=4e-14,
                               err_msg=f"{backend_name} zero/small/half-turn rotation")
    np.testing.assert_allclose(actual_rotvecs[:2], vectors[:2], rtol=1e-12, atol=2e-15,
                               err_msg=f"{backend_name} zero/small rotation-vector limits")


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    [backend for backend in BACKENDS if backend[3]],
    ids=[backend[0] for backend in BACKENDS if backend[3]],
)
def test_numpy_empty_batches_keep_component_and_matrix_axes(
    backend_name, module, is_jpl, is_numpy, component_first
):
    empty_vectors = np.empty((0, 3))
    vector_input = orient_vectors(empty_vectors, component_first)
    q = module.from_rotation_vector(vector_input)
    expected_q_shape = (4, 0) if component_first else (0, 4)
    expected_vector_shape = (3, 0) if component_first else (0, 3)
    expected_matrix_shape = (3, 3, 0) if component_first else (0, 3, 3)
    assert q.shape == expected_q_shape
    assert module.to_rotation_vector(q).shape == expected_vector_shape
    assert module.to_rotation_matrix(q).shape == expected_matrix_shape
    assert module.rotate(q, vector_input).shape == expected_vector_shape

    empty_sequence = np.empty((0, 4))
    sequence_input = orient_vectors(empty_sequence, component_first)
    assert module.make_sign_continuous(sequence_input).shape == expected_q_shape
