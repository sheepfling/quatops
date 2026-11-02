"""Float32 scalar inputs promote to the documented Python-float result types."""

import cmath
import math
from decimal import Decimal

import numpy as np
import pytest
import quaternion
from scipy.spatial.transform import Rotation, Slerp

from tests.oracle_support import BACKENDS, reference_coefficients, reference_quaternions

SCALAR_BACKENDS = tuple(backend for backend in BACKENDS if not backend[3])


def _wxyz(module, value) -> np.ndarray:
    return np.asarray(module.to_wxyz(value), dtype=np.float64)


def _assert_float_result(module, actual, expected, is_jpl: bool, name: str) -> None:
    np.testing.assert_allclose(
        _wxyz(module, actual),
        reference_coefficients(expected, is_jpl),
        rtol=6e-13,
        atol=6e-13,
        err_msg=name,
    )
    assert all(type(value) is float for value in module.to_wxyz(actual)), name


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    SCALAR_BACKENDS,
    ids=[backend[0] for backend in SCALAR_BACKENDS],
)
def test_scalar_float32_algebra_matches_numpy_quaternion(
    backend_name, module, is_jpl, is_numpy, component_first
):
    left32 = np.asarray((0.83, 0.12, -0.24, 0.31), dtype=np.float32)
    right32 = np.asarray((0.91, -0.18, 0.27, 0.14), dtype=np.float32)
    small32 = np.asarray((0.025, -0.012, 0.018, 0.009), dtype=np.float32)
    left = module.from_wxyz(left32)
    right = module.from_wxyz(right32)
    small = module.from_wxyz(small32)
    left_ref = reference_quaternions(left32.astype(np.float64), is_jpl)
    right_ref = reference_quaternions(right32.astype(np.float64), is_jpl)
    small_ref = reference_quaternions(small32.astype(np.float64), is_jpl)
    exponent = np.float32(-0.375)
    scale = np.float32(1.625)

    expected = {
        "add": left_ref + right_ref,
        "subtract": left_ref - right_ref,
        "negate": -left_ref,
        "scale": scale * left_ref,
        "conjugate": np.conjugate(left_ref),
        "multiply": left_ref * right_ref,
        "divide": left_ref / right_ref,
        "left_divide": (1.0 / left_ref) * right_ref,
        "right_divide": left_ref / right_ref,
        "inverse": 1.0 / left_ref,
        "exp": np.exp(left_ref),
        "log": np.log(left_ref),
        "expm1": np.exp(small_ref) - 1.0,
        "log1p": np.log(1.0 + small_ref),
        "log2": np.log(left_ref) / math.log(2.0),
        "log10": np.log(left_ref) / math.log(10.0),
        "power": np.power(left_ref, float(exponent)),
        "pow_real": np.power(left_ref, float(exponent)),
        "power_quaternion": np.exp(right_ref * np.log(left_ref)),
        "sqrt": np.power(left_ref, 0.5),
        "commutator": left_ref * right_ref - right_ref * left_ref,
    }
    calls = {
        "add": lambda: module.add(left, right),
        "subtract": lambda: module.subtract(left, right),
        "negate": lambda: module.negate(left),
        "scale": lambda: module.scale(left, scale),
        "conjugate": lambda: module.conjugate(left),
        "multiply": lambda: module.multiply(left, right),
        "divide": lambda: module.divide(left, right),
        "left_divide": lambda: module.left_divide(left, right),
        "right_divide": lambda: module.right_divide(left, right),
        "inverse": lambda: module.inverse(left),
        "exp": lambda: module.exp(left),
        "log": lambda: module.log(left),
        "expm1": lambda: module.expm1(small),
        "log1p": lambda: module.log1p(small),
        "log2": lambda: module.log2(left),
        "log10": lambda: module.log10(left),
        "power": lambda: module.power(left, exponent),
        "pow_real": lambda: module.pow_real(left, exponent),
        "power_quaternion": lambda: module.power_quaternion(left, right),
        "sqrt": lambda: module.sqrt(left),
        "commutator": lambda: module.commutator(left, right),
    }
    for name, function in calls.items():
        _assert_float_result(module, function(), expected[name], is_jpl,
                             f"{backend_name}.{name} accepts float32")

    assert type(module.norm(left)) is float
    assert type(module.norm_squared(left)) is float
    assert type(module.dot(left, right)) is float
    np.testing.assert_allclose(module.norm(left), np.linalg.norm(left32.astype(float)))
    np.testing.assert_allclose(module.norm_squared(left), np.dot(left32, left32))
    np.testing.assert_allclose(module.dot(left, right), np.dot(left32, right32))
    np.testing.assert_allclose(tuple(module.vector_part(left)), left32[1:], rtol=0, atol=0)
    assert module.scalar_part(left) == float(left32[0])


def _complex_slice(values: np.ndarray, function) -> np.ndarray:
    w, x, y, z = values.astype(np.float64)
    radius = math.hypot(x, y, z)
    result = function(complex(w, radius))
    direction = (1.0, 0.0, 0.0) if radius == 0.0 else (x / radius, y / radius, z / radius)
    return np.asarray((result.real, *(result.imag * value for value in direction)))


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    SCALAR_BACKENDS,
    ids=[backend[0] for backend in SCALAR_BACKENDS],
)
def test_scalar_float32_analytic_functions_match_complex_slices(
    backend_name, module, is_jpl, is_numpy, component_first
):
    values = np.asarray((0.23, -0.17, 0.08, 0.29), dtype=np.float32)
    q = module.from_wxyz(values)
    reference = reference_quaternions(values.astype(np.float64), is_jpl)
    raw = reference_coefficients(reference, is_jpl)
    functions = {
        "sin": cmath.sin,
        "cos": cmath.cos,
        "tan": cmath.tan,
        "sinh": cmath.sinh,
        "cosh": cmath.cosh,
        "tanh": cmath.tanh,
        "asin": cmath.asin,
        "acos": cmath.acos,
        "atan": cmath.atan,
        "asinh": cmath.asinh,
        "acosh": cmath.acosh,
        "atanh": cmath.atanh,
    }
    for name, function in functions.items():
        actual = getattr(module, name)(q)
        np.testing.assert_allclose(
            module.to_wxyz(actual),
            _complex_slice(raw, function),
            rtol=3e-12,
            atol=3e-12,
            err_msg=f"{backend_name}.{name} accepts float32",
        )
        assert all(type(value) is float for value in module.to_wxyz(actual))


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    SCALAR_BACKENDS,
    ids=[backend[0] for backend in SCALAR_BACKENDS],
)
def test_scalar_float32_rotation_inputs_and_float_parameters_match_scipy(
    backend_name, module, is_jpl, is_numpy, component_first
):
    vector32 = np.asarray((0.17, -0.08, 0.22), dtype=np.float32)
    omega32 = np.asarray((-0.11, 0.19, 0.07), dtype=np.float32)
    vector = tuple(vector32)
    omega = tuple(omega32)
    exponent = np.float32(0.35)
    step = np.float32(0.23)

    made = module.from_rotation_vector(vector)
    expected_rotation = Rotation.from_rotvec(vector32.astype(np.float64))
    expected_public = reference_coefficients(
        quaternion.from_rotation_vector(vector32.astype(np.float64)), is_jpl
    )
    np.testing.assert_allclose(module.to_wxyz(made), expected_public,
                               rtol=6e-13, atol=6e-13,
                               err_msg=f"{backend_name}.from_rotation_vector float32")
    np.testing.assert_allclose(module.to_rotation_matrix(made), expected_rotation.as_matrix(),
                               rtol=6e-13, atol=6e-13)

    q_values32 = np.asarray((0.91, 0.12, -0.21, 0.16), dtype=np.float32)
    q = module.from_wxyz(q_values32)
    q_reference_hamilton = quaternion.as_float_array(
        reference_quaternions(q_values32.astype(np.float64), is_jpl)
    )
    q_reference = Rotation.from_quat(q_reference_hamilton[[1, 2, 3, 0]])
    actual_rotate = module.rotate(q, vector)
    np.testing.assert_allclose(actual_rotate, q_reference.apply(vector32.astype(float)),
                               rtol=8e-13, atol=8e-13,
                               err_msg=f"{backend_name}.rotate float32 vector")

    increment = Rotation.from_rotvec(omega32.astype(float) * float(step))
    expected_integrated = q_reference * increment
    integrated = module.integrate_angular_velocity(q, omega, step, frame="body")
    np.testing.assert_allclose(module.to_rotation_matrix(integrated),
                               expected_integrated.as_matrix(), rtol=8e-13, atol=8e-13,
                               err_msg=f"{backend_name}.integrate float32 arguments")

    updated = module.boxplus(q, vector, frame="world")
    expected_updated = Rotation.from_rotvec(vector32.astype(float)) * q_reference
    np.testing.assert_allclose(module.to_rotation_matrix(updated),
                               expected_updated.as_matrix(), rtol=8e-13, atol=8e-13)

    interpolated = module.slerp(q, made, exponent)
    amount = float(exponent)
    expected_endpoints = Rotation.from_quat(np.stack(
        (q_reference.as_quat(), expected_rotation.as_quat())
    ))
    # Compare the scalar API's scalar float32 parameter against SciPy's
    # independent spherical interpolation on the same normalized endpoints.
    expected_interpolated = Slerp((0.0, 1.0), expected_endpoints)(amount)
    np.testing.assert_allclose(module.to_rotation_matrix(interpolated),
                               expected_interpolated.as_matrix(), rtol=8e-13, atol=8e-13)


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    SCALAR_BACKENDS,
    ids=[backend[0] for backend in SCALAR_BACKENDS],
)
def test_non_real_supports_float_values_use_compatibility_fallback(
    backend_name, module, is_jpl, is_numpy, component_first
):
    values = tuple(Decimal(value) for value in ("0.8", "0.1", "-0.2", "0.3"))
    q = module.from_wxyz(values)
    scaled = module.scale(q, Decimal("0.5"))

    np.testing.assert_allclose(module.to_wxyz(scaled), np.asarray(values, dtype=float) * 0.5)
    assert all(type(value) is float for value in module.to_wxyz(scaled)), backend_name
