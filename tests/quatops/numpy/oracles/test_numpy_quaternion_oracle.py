"""Seeded Monte Carlo checks against the independent numpy-quaternion package."""

import math

import numpy as np
import pytest
import quaternion

from tests.support.oracle_support import (
    BACKENDS,
)
from tests.support.oracle_support import (
    from_wxyz as _from_wxyz,
)
from tests.support.oracle_support import (
    matrices_to_trailing as _matrices_to_trailing,
)
from tests.support.oracle_support import (
    orient_matrices as _orient_matrices,
)
from tests.support.oracle_support import (
    orient_vectors as _orient_vectors,
)
from tests.support.oracle_support import (
    reference_coefficients as _reference_coefficients,
)
from tests.support.oracle_support import (
    reference_quaternions as _reference_quaternions,
)
from tests.support.oracle_support import (
    to_wxyz as _to_wxyz,
)
from tests.support.oracle_support import (
    vectors_to_trailing as _vectors_to_trailing,
)

ARRAY_SAMPLES = 1024
SCALAR_SAMPLES = 192


def _apply(module, operation: str, lhs, rhs, exponent):
    if operation in {"add", "subtract", "multiply", "divide", "left_divide",
                     "right_divide", "commutator", "power_quaternion"}:
        return getattr(module, operation)(lhs, rhs)
    if operation == "scale":
        return module.scale(lhs, 0.375)
    if operation in {"power", "pow_real"}:
        return getattr(module, operation)(lhs, exponent)
    return getattr(module, operation)(lhs)


def _assert_coefficients(actual, expected, backend_name: str, operation: str) -> None:
    np.testing.assert_allclose(
        actual, expected, rtol=4e-12, atol=4e-12,
        err_msg=f"{backend_name}.{operation} disagrees with numpy-quaternion",
    )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_random_generic_quaternion_operations_match_numpy_quaternion(
    backend_name, module, is_jpl, is_numpy, component_first
):
    """Compare generic algebra against numpy-quaternion on seeded random inputs."""
    sample_count = ARRAY_SAMPLES if is_numpy else SCALAR_SAMPLES
    rng = np.random.default_rng(0x514F + len(backend_name))
    lhs_values = rng.normal(scale=0.35, size=(sample_count, 4))
    rhs_values = rng.normal(scale=0.35, size=(sample_count, 4))
    real_exponents = rng.uniform(-1.25, 1.25, size=sample_count)
    power_exponent = (float(rng.uniform(-1.25, 1.25)) if is_numpy else real_exponents)
    small_values = rng.normal(scale=0.12, size=(sample_count, 4))

    lhs_reference = _reference_quaternions(lhs_values, is_jpl)
    rhs_reference = _reference_quaternions(rhs_values, is_jpl)
    small_reference = _reference_quaternions(small_values, is_jpl)
    expected_quaternions = {
        "add": lhs_reference + rhs_reference,
        "subtract": lhs_reference - rhs_reference,
        "negate": -lhs_reference,
        "scale": 0.375 * lhs_reference,
        "conjugate": np.conjugate(lhs_reference),
        "multiply": lhs_reference * rhs_reference,
        "divide": lhs_reference / rhs_reference,
        "right_divide": lhs_reference / rhs_reference,
        "left_divide": (1.0 / lhs_reference) * rhs_reference,
        "inverse": 1.0 / lhs_reference,
        "exp": np.exp(lhs_reference),
        "expm1": np.exp(small_reference) - 1.0,
        "log": np.log(lhs_reference),
        "log1p": np.log(1.0 + small_reference),
        "log2": np.log(lhs_reference) / math.log(2.0),
        "log10": np.log(lhs_reference) / math.log(10.0),
        "power": np.power(lhs_reference, power_exponent),
        "pow_real": np.power(lhs_reference, power_exponent),
        "power_quaternion": np.exp(rhs_reference * np.log(lhs_reference)),
        "sqrt": np.power(lhs_reference, 0.5),
        "commutator": lhs_reference * rhs_reference - rhs_reference * lhs_reference,
    }
    if is_numpy:
        lhs = _from_wxyz(module, lhs_values, True, component_first)
        rhs = _from_wxyz(module, rhs_values, True, component_first)
        small = _from_wxyz(module, small_values, True, component_first)
    else:
        lhs = rhs = small = None

    for operation, reference in expected_quaternions.items():
        if operation in {"expm1", "log1p"}:
            operation_input = small
        else:
            operation_input = lhs
        if is_numpy:
            if operation in {"add", "subtract", "multiply", "divide", "left_divide",
                             "right_divide", "commutator", "power_quaternion"}:
                actual_value = _apply(module, operation, lhs, rhs, real_exponents)
            elif operation in {"power", "pow_real"}:
                actual_value = getattr(module, operation)(lhs, power_exponent)
            elif operation in {"expm1", "log1p"}:
                actual_value = getattr(module, operation)(small)
            else:
                actual_value = _apply(module, operation, operation_input, rhs, real_exponents)
            actual_coefficients = _to_wxyz(module, actual_value, True, component_first)
            expected_coefficients = _reference_coefficients(reference, is_jpl)
            _assert_coefficients(actual_coefficients, expected_coefficients,
                                 backend_name, operation)
        else:
            actual_coefficients = []
            for index in range(sample_count):
                lhs_item = _from_wxyz(module, lhs_values[index], False, False)
                rhs_item = _from_wxyz(module, rhs_values[index], False, False)
                small_item = _from_wxyz(module, small_values[index], False, False)
                if operation in {"expm1", "log1p"}:
                    actual = _apply(module, operation, small_item, rhs_item,
                                    float(real_exponents[index]))
                elif operation in {"power", "pow_real"}:
                    actual = getattr(module, operation)(
                        lhs_item, float(real_exponents[index])
                    )
                elif operation in {"add", "subtract", "multiply", "divide", "left_divide",
                                   "right_divide", "commutator", "power_quaternion"}:
                    actual = _apply(module, operation, lhs_item, rhs_item,
                                    float(real_exponents[index]))
                else:
                    actual = _apply(module, operation, lhs_item, rhs_item,
                                    float(real_exponents[index]))
                actual_coefficients.append(_to_wxyz(module, actual, False, False))
            expected_coefficients = _reference_coefficients(reference, is_jpl)
            _assert_coefficients(np.asarray(actual_coefficients), expected_coefficients,
                                 backend_name, operation)

    # Coefficient-space reductions are also sampled independently of quaternion algebra.
    q_norms = np.linalg.norm(lhs_values, axis=-1)
    expected_reductions = {
        "norm": (lhs, q_norms),
        "norm_squared": (lhs, np.sum(lhs_values * lhs_values, axis=-1)),
        "dot": (rhs, np.sum(lhs_values * rhs_values, axis=-1)),
    }
    for operation, (other, expected) in expected_reductions.items():
        if is_numpy:
            actual = (module.dot(lhs, other) if operation == "dot"
                      else getattr(module, operation)(lhs))
            np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14,
                                       err_msg=f"{backend_name}.{operation}")
        else:
            actual = []
            for index in range(sample_count):
                q_item = _from_wxyz(module, lhs_values[index], False, False)
                if operation == "dot":
                    r_item = _from_wxyz(module, rhs_values[index], False, False)
                    actual.append(module.dot(q_item, r_item))
                else:
                    actual.append(getattr(module, operation)(q_item))
            np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14,
                                       err_msg=f"{backend_name}.{operation}")

    if is_numpy:
        normalized = module.normalize(lhs)
        actual_normalized = _to_wxyz(module, normalized, True, component_first)
        _assert_coefficients(actual_normalized, lhs_values / q_norms[:, None],
                             backend_name, "normalize")
        actual_vector = np.asarray(module.vector_part(lhs), dtype=float)
        if component_first:
            actual_vector = np.moveaxis(actual_vector, 0, -1)
        np.testing.assert_allclose(actual_vector, lhs_values[:, 1:], rtol=0, atol=0,
                                   err_msg=f"{backend_name}.vector_part")
    else:
        actual_normalized = []
        actual_vector = []
        for index in range(sample_count):
            q_item = _from_wxyz(module, lhs_values[index], False, False)
            actual_normalized.append(_to_wxyz(module, module.normalize(q_item), False, False))
            actual_vector.append(tuple(module.vector_part(q_item)))
        _assert_coefficients(np.asarray(actual_normalized), lhs_values / q_norms[:, None],
                             backend_name, "normalize")
        np.testing.assert_allclose(actual_vector, lhs_values[:, 1:], rtol=0, atol=0,
                                   err_msg=f"{backend_name}.vector_part")


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_integer_powers_match_numpy_quaternion(backend_name, module, is_jpl,
                                              is_numpy, component_first):
    """Exercise the integer-power algorithm separately from real powers."""
    rng = np.random.default_rng(0x1A7E + len(backend_name))
    count = 64
    values = rng.normal(scale=0.45, size=(count, 4))
    values += np.array((0.9, 0.0, 0.0, 0.0))
    reference = _reference_quaternions(values, is_jpl)
    if is_numpy:
        quaternions = _from_wxyz(module, values, True, component_first)
    else:
        quaternions = [_from_wxyz(module, row, False, False) for row in values]

    for exponent in (-7, -2, 0, 1, 2, 5):
        expected = _reference_coefficients(np.power(reference, exponent), is_jpl)
        if is_numpy:
            actual = _to_wxyz(
                module, module.power(quaternions, exponent), True, component_first
            )
        else:
            actual = np.asarray([
                _to_wxyz(module, module.power(value, exponent), False, False)
                for value in quaternions
            ])
        _assert_coefficients(actual, expected, backend_name, f"power integer {exponent}")

    identity = module.from_wxyz((1.0, 0.0, 0.0, 0.0))
    huge_identity = module.power(identity, 10**400)
    np.testing.assert_array_equal(
        _to_wxyz(module, huge_identity, is_numpy, component_first),
        (1.0, 0.0, 0.0, 0.0),
        err_msg=f"{backend_name}.power huge integer identity",
    )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_random_rotation_vectors_and_actions_match_numpy_quaternion(
    backend_name, module, is_jpl, is_numpy, component_first
):
    """Check rotation-vector conversions, matrices, and vector action in Monte Carlo."""
    sample_count = ARRAY_SAMPLES if is_numpy else SCALAR_SAMPLES
    rng = np.random.default_rng(0x50A5 + len(backend_name))
    axes = rng.normal(size=(sample_count, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    angles = rng.uniform(1e-7, math.pi - 1e-7, size=sample_count)
    rotation_vectors = axes * angles[:, None]
    vectors = rng.normal(size=(sample_count, 3))
    scales = np.exp(rng.uniform(-4.0, 4.0, size=sample_count))

    reference_rotors = quaternion.from_rotation_vector(rotation_vectors)
    reference_scaled = reference_rotors * scales
    reference_coefficients = _reference_coefficients(reference_rotors, is_jpl)
    scaled_coefficients = _reference_coefficients(reference_scaled, is_jpl)
    reference_matrices = quaternion.as_rotation_matrix(reference_scaled)
    reference_vectors = np.stack(
        [quaternion.rotate_vectors(reference_scaled[index], vectors[index])
         for index in range(sample_count)]
    )
    reference_rotation_vectors = quaternion.as_rotation_vector(reference_scaled)

    if is_numpy:
        made_rotors = module.from_rotation_vector(
            _orient_vectors(rotation_vectors, component_first)
        )
        actual_coefficients = _to_wxyz(module, made_rotors, True, component_first)
        _assert_coefficients(actual_coefficients, reference_coefficients,
                             backend_name, "from_rotation_vector")

        scaled_input = _from_wxyz(module, scaled_coefficients, True, component_first)
        actual_rotation_vectors = _vectors_to_trailing(
            module.to_rotation_vector(scaled_input), component_first
        )
        np.testing.assert_allclose(actual_rotation_vectors, reference_rotation_vectors,
                                   rtol=3e-12, atol=3e-12,
                                   err_msg=f"{backend_name}.to_rotation_vector")

        actual_vectors = _vectors_to_trailing(
            module.rotate(scaled_input, _orient_vectors(vectors, component_first)),
            component_first,
        )
        np.testing.assert_allclose(actual_vectors, reference_vectors, rtol=3e-12, atol=3e-12,
                                   err_msg=f"{backend_name}.rotate")

        actual_matrices = _matrices_to_trailing(
            module.to_rotation_matrix(scaled_input), component_first
        )
        np.testing.assert_allclose(actual_matrices, reference_matrices,
                                   rtol=3e-12, atol=3e-12,
                                   err_msg=f"{backend_name}.to_rotation_matrix")

        recovered = module.from_rotation_matrix(
            _orient_matrices(reference_matrices, component_first)
        )
        recovered_matrices = _matrices_to_trailing(
            module.to_rotation_matrix(recovered), component_first
        )
        np.testing.assert_allclose(recovered_matrices, reference_matrices,
                                   rtol=3e-12, atol=3e-12,
                                   err_msg=f"{backend_name}.from_rotation_matrix")
    else:
        for index in range(sample_count):
            vector = tuple(rotation_vectors[index])
            rotor = module.from_rotation_vector(vector)
            _assert_coefficients(_to_wxyz(module, rotor, False, False),
                                 reference_coefficients[index], backend_name,
                                 "from_rotation_vector")

            scaled = module.from_wxyz(tuple(scaled_coefficients[index]))
            np.testing.assert_allclose(tuple(module.to_rotation_vector(scaled)),
                                       reference_rotation_vectors[index],
                                       rtol=3e-12, atol=3e-12,
                                       err_msg=f"{backend_name}.to_rotation_vector")
            np.testing.assert_allclose(module.rotate(scaled, tuple(vectors[index])),
                                       reference_vectors[index], rtol=3e-12, atol=3e-12,
                                       err_msg=f"{backend_name}.rotate")
            actual_matrix = module.to_rotation_matrix(scaled)
            np.testing.assert_allclose(actual_matrix, reference_matrices[index],
                                       rtol=3e-12, atol=3e-12,
                                       err_msg=f"{backend_name}.to_rotation_matrix")
            recovered = module.from_rotation_matrix(reference_matrices[index])
            np.testing.assert_allclose(module.to_rotation_matrix(recovered),
                                       reference_matrices[index], rtol=3e-12, atol=3e-12,
                                       err_msg=f"{backend_name}.from_rotation_matrix")
