"""Independent product and closed-form matrix oracles for every backend."""

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.oracle_support import (
    BACKENDS,
    as_hamilton_coefficients,
    from_wxyz,
    matrices_to_trailing,
    orient_matrices,
    orient_vectors,
    random_rotations,
    reference_coefficients,
    reference_quaternions,
    to_wxyz,
)


def _native_scalar_last(backend_name: str, is_jpl: bool, is_numpy: bool) -> bool:
    return (is_jpl and not is_numpy) or (is_numpy and backend_name.endswith(".xyzw"))


def _native_coefficients(values: np.ndarray, scalar_last: bool) -> np.ndarray:
    return values[:, (1, 2, 3, 0)] if scalar_last else values


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _jacobians(vector: np.ndarray) -> tuple[np.ndarray, ...]:
    theta = float(np.linalg.norm(vector))
    cross = _skew(vector)
    cross_squared = cross @ cross
    identity = np.eye(3)
    if theta < 1e-5:
        theta2 = theta * theta
        first = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
        second = 1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0
        inverse_second = (1.0 / 12.0 + theta2 / 720.0
                          + theta2 * theta2 / 30240.0)
    else:
        first = (1.0 - math.cos(theta)) / theta**2
        second = (theta - math.sin(theta)) / theta**3
        inverse_second = (1.0 - 0.5 * theta / math.tan(0.5 * theta)) / theta**2
    left = identity + first * cross + second * cross_squared
    right = identity - first * cross + second * cross_squared
    left_inverse = identity - 0.5 * cross + inverse_second * cross_squared
    right_inverse = identity + 0.5 * cross + inverse_second * cross_squared
    return left, right, left_inverse, right_inverse


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_left_and_right_product_matrices_match_numpy_quaternion_basis_products(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rng = np.random.default_rng(0x4A71 + len(backend_name))
    count = 36
    lhs_values = rng.normal(size=(count, 4))
    rhs_values = rng.normal(size=(count, 4))
    scalar_last = _native_scalar_last(backend_name, is_jpl, is_numpy)
    if is_numpy:
        lhs = from_wxyz(module, lhs_values, True, component_first)
        left_matrices = matrices_to_trailing(module.to_left_matrix(lhs), component_first)
        right_matrices = matrices_to_trailing(module.to_right_matrix(lhs), component_first)
    else:
        lhs = [module.from_wxyz(tuple(row)) for row in lhs_values]
        left_matrices = np.stack([module.to_left_matrix(value) for value in lhs])
        right_matrices = np.stack([module.to_right_matrix(value) for value in lhs])

    expected_left = np.empty((count, 4, 4))
    expected_right = np.empty((count, 4, 4))
    for basis_index in range(4):
        native_basis = np.zeros(4)
        native_basis[basis_index] = 1.0
        if scalar_last:
            public_basis = native_basis[[3, 0, 1, 2]]
        else:
            public_basis = native_basis
        reference_basis = reference_quaternions(public_basis, is_jpl)
        reference_lhs = reference_quaternions(lhs_values, is_jpl)
        left_product = reference_lhs * reference_basis
        right_product = reference_basis * reference_lhs
        left_public = reference_coefficients(left_product, is_jpl)
        right_public = reference_coefficients(right_product, is_jpl)
        expected_left[:, :, basis_index] = _native_coefficients(left_public, scalar_last)
        expected_right[:, :, basis_index] = _native_coefficients(right_public, scalar_last)

    np.testing.assert_allclose(left_matrices, expected_left, rtol=3e-14, atol=3e-14,
                               err_msg=f"{backend_name}.to_left_matrix")
    np.testing.assert_allclose(right_matrices, expected_right, rtol=3e-14, atol=3e-14,
                               err_msg=f"{backend_name}.to_right_matrix")

    if is_numpy:
        rhs = from_wxyz(module, rhs_values, True, component_first)
        native_rhs = _native_coefficients(rhs_values, scalar_last)
        right_matrices_rhs = matrices_to_trailing(module.to_right_matrix(rhs), component_first)
        actual_product = module.multiply(lhs, rhs)
        public_product = to_wxyz(module, actual_product, True, component_first)
        expected_product = reference_coefficients(
            reference_quaternions(lhs_values, is_jpl)
            * reference_quaternions(rhs_values, is_jpl),
            is_jpl,
        )
        expected_native_product = _native_coefficients(expected_product, scalar_last)
        native_actual = _native_coefficients(public_product, scalar_last)
        # The product identity is also checked directly in the backend's native
        # coefficient order, independently anchored by numpy-quaternion above.
        matrix_result = np.einsum("bij,bj->bi", left_matrices, native_rhs)
        np.testing.assert_allclose(matrix_result, expected_native_product, rtol=4e-13, atol=4e-13,
                                   err_msg=f"{backend_name} left matrix action")
        right_matrix_result = np.einsum(
            "bij,bj->bi", right_matrices_rhs, _native_coefficients(lhs_values, scalar_last)
        )
        np.testing.assert_allclose(right_matrix_result, expected_native_product,
                                   rtol=4e-13, atol=4e-13,
                                   err_msg=f"{backend_name} right matrix action")
        np.testing.assert_allclose(native_actual, expected_native_product,
                                   rtol=4e-13, atol=4e-13,
                                   err_msg=f"{backend_name} product coefficients")
    else:
        for index, (left, right) in enumerate(zip(lhs, rhs_values)):
            rhs = module.from_wxyz(tuple(right))
            native_rhs = module.to_xyzw(rhs) if is_jpl else module.to_wxyz(rhs)
            right_matrix = module.to_right_matrix(rhs)
            native_lhs = module.to_xyzw(left) if is_jpl else module.to_wxyz(left)
            product = module.multiply(left, rhs)
            product_values = module.to_xyzw(product) if is_jpl else module.to_wxyz(product)
            np.testing.assert_allclose(left_matrices[index] @ np.asarray(native_rhs),
                                       product_values, rtol=4e-13, atol=4e-13,
                                       err_msg=f"{backend_name} left product identity {index}")
            np.testing.assert_allclose(np.asarray(right_matrix) @ np.asarray(native_lhs),
                                       product_values, rtol=4e-13, atol=4e-13,
                err_msg=f"{backend_name} right product identity {index}")


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_skew_and_so3_jacobians_match_explicit_closed_forms(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rng = np.random.default_rng(0x50A3 + len(backend_name))
    vectors = rng.normal(size=(48, 3))
    vectors *= rng.uniform(0.01, 2.8, size=(48, 1)) / np.linalg.norm(
        vectors, axis=-1, keepdims=True
    )
    vectors[:2] = ((0.0, 0.0, 0.0), (1e-11, -2e-11, 3e-11))
    if is_numpy:
        vector_input = orient_vectors(vectors, component_first)
        actual = {
            "skew_matrix": matrices_to_trailing(module.skew_matrix(vector_input), component_first),
            "so3_left_jacobian": matrices_to_trailing(
                module.so3_left_jacobian(vector_input), component_first
            ),
            "so3_right_jacobian": matrices_to_trailing(
                module.so3_right_jacobian(vector_input), component_first
            ),
            "so3_left_jacobian_inverse": matrices_to_trailing(
                module.so3_left_jacobian_inverse(vector_input), component_first
            ),
            "so3_right_jacobian_inverse": matrices_to_trailing(
                module.so3_right_jacobian_inverse(vector_input), component_first
            ),
        }
    else:
        actual = {
            name: np.stack([getattr(module, name)(tuple(vector)) for vector in vectors])
            for name in (
                "skew_matrix", "so3_left_jacobian", "so3_right_jacobian",
                "so3_left_jacobian_inverse", "so3_right_jacobian_inverse",
            )
        }
    expected = {name: [] for name in actual}
    for vector in vectors:
        left, right, left_inverse, right_inverse = _jacobians(vector)
        expected["skew_matrix"].append(_skew(vector))
        expected["so3_left_jacobian"].append(left)
        expected["so3_right_jacobian"].append(right)
        expected["so3_left_jacobian_inverse"].append(left_inverse)
        expected["so3_right_jacobian_inverse"].append(right_inverse)
    for name, values in actual.items():
        np.testing.assert_allclose(values, expected[name], rtol=8e-13, atol=8e-13,
                                   err_msg=f"{backend_name}.{name}")


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_homogeneous_transforms_and_adjoint_match_scipy(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rng = np.random.default_rng(0xAD10 + len(backend_name))
    count = 40
    rotations = random_rotations(count, rng)
    public = as_hamilton_coefficients(rotations.as_quat()[..., (3, 0, 1, 2)], is_jpl)
    magnitudes = rng.uniform(0.1, 9.0, size=(count, 1))
    coefficients = public * magnitudes
    translations = rng.normal(size=(count, 3))
    expected_rotation = rotations.as_matrix()
    expected_homogeneous = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    expected_homogeneous[:, :3, :3] = expected_rotation
    expected_homogeneous[:, :3, 3] = translations

    if is_numpy:
        q = from_wxyz(module, coefficients, True, component_first)
        translation_input = orient_vectors(translations, component_first)
        homogeneous = matrices_to_trailing(
            module.to_homogeneous_matrix(q, translation_input), component_first
        )
        adjoint = matrices_to_trailing(module.so3_adjoint(q), component_first)
        recovered = module.from_homogeneous_matrix(
            orient_matrices(expected_homogeneous, component_first)
        )
        recovered_wxyz = to_wxyz(module, recovered, True, component_first)
    else:
        q = [module.from_wxyz(tuple(value)) for value in coefficients]
        homogeneous = np.stack([
            module.to_homogeneous_matrix(value, tuple(translation))
            for value, translation in zip(q, translations)
        ])
        adjoint = np.stack([module.so3_adjoint(value) for value in q])
        recovered = [module.from_homogeneous_matrix(matrix) for matrix in expected_homogeneous]
        recovered_wxyz = np.stack([module.to_wxyz(value) for value in recovered])

    np.testing.assert_allclose(homogeneous, expected_homogeneous, rtol=5e-13, atol=5e-13,
                               err_msg=f"{backend_name}.to_homogeneous_matrix")
    np.testing.assert_allclose(adjoint, expected_rotation, rtol=5e-13, atol=5e-13,
                               err_msg=f"{backend_name}.so3_adjoint")
    np.testing.assert_allclose(
        Rotation.from_quat(
            as_hamilton_coefficients(recovered_wxyz, is_jpl)[..., (1, 2, 3, 0)]
        ).as_matrix(),
        expected_rotation,
        rtol=5e-13,
        atol=5e-13,
        err_msg=f"{backend_name}.from_homogeneous_matrix",
    )
