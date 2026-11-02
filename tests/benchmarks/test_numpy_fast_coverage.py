"""Benchmark optional compiled paths for remaining NumPy batch operations.

Run with ``.venv/bin/python -m pytest tests/benchmarks/test_numpy_fast_coverage.py
-m benchmark --benchmark-only --benchmark-group-by=group --benchmark-columns=median``.
Correctness is checked before each benchmark. Scalar-loop timings expose the
benefit over the former per-row mapping paths; simple NumPy formulas provide a
baseline for the core coefficient operations.
"""

import numpy as np
import pytest
import quaternion
from numpy.testing import assert_allclose

from quatops import _numpy
from quatops.hamilton import scalar as hamilton_scalar
from quatops.numpy.hamilton import xyzw

pytestmark = pytest.mark.benchmark

BATCH_SIZE = 10_000
RNG = np.random.default_rng(202)
QUATERNIONS = xyzw.from_rotvec(RNG.uniform(-0.8, 0.8, (BATCH_SIZE, 3)))
OTHER_QUATERNIONS = xyzw.from_rotvec(
    RNG.uniform(-0.8, 0.8, (BATCH_SIZE, 3))
)
QUATERNIONS_NUMPY_QUATERNION = quaternion.as_quat_array(
    np.column_stack((QUATERNIONS[:, 3], QUATERNIONS[:, :3]))
)
OTHER_QUATERNIONS_NUMPY_QUATERNION = quaternion.as_quat_array(
    np.column_stack((OTHER_QUATERNIONS[:, 3], OTHER_QUATERNIONS[:, :3]))
)
QUATERNIONS_WXYZ = xyzw.to_wxyz(QUATERNIONS)
OTHER_QUATERNIONS_WXYZ = xyzw.to_wxyz(OTHER_QUATERNIONS)
MRPS = RNG.uniform(-0.5, 0.5, (BATCH_SIZE, 3))
INTERPOLATION_AMOUNT = 0.37

INTEGRATION_BATCH = 256
INTEGRATION_SAMPLES = 64
INITIAL_ATTITUDES = QUATERNIONS[:INTEGRATION_BATCH]
INITIAL_ATTITUDES_WXYZ = QUATERNIONS_WXYZ[:INTEGRATION_BATCH]
ANGULAR_RATES = RNG.normal(
    scale=0.3, size=(INTEGRATION_BATCH, INTEGRATION_SAMPLES, 3)
)
TIMES = np.broadcast_to(
    np.arange(INTEGRATION_SAMPLES, dtype=np.float64) * 0.01,
    (INTEGRATION_BATCH, INTEGRATION_SAMPLES),
)


def _scalar_interpolation(function):
    values = np.asarray(
        [
            function(tuple(first), tuple(second), INTERPOLATION_AMOUNT)
            for first, second in zip(QUATERNIONS_WXYZ, OTHER_QUATERNIONS_WXYZ)
        ]
    )
    return xyzw.from_wxyz(values)


def _scalar_mrp_construction():
    values = np.asarray(
        [hamilton_scalar.from_mrp(tuple(sigma)) for sigma in MRPS]
    )
    return xyzw.from_wxyz(values)


def _scalar_mrp_extraction():
    return np.asarray(
        [hamilton_scalar.to_mrp(tuple(value)) for value in QUATERNIONS_WXYZ]
    )


def _scalar_integration():
    values = np.asarray(
        [
            hamilton_scalar.integrate_angular_velocity_samples(
                tuple(attitude), rate, time
            )
            for attitude, rate, time in zip(
                INITIAL_ATTITUDES_WXYZ, ANGULAR_RATES, TIMES
            )
        ]
    )
    return xyzw.from_wxyz(values)


def _numpy_normalize():
    return QUATERNIONS / np.linalg.norm(QUATERNIONS, axis=-1, keepdims=True)


def _numpy_inverse():
    conjugate = QUATERNIONS * np.array((-1.0, -1.0, -1.0, 1.0))
    return conjugate / np.sum(QUATERNIONS * QUATERNIONS, axis=-1, keepdims=True)


def _numpy_dot():
    return np.einsum("...i,...i->...", QUATERNIONS, OTHER_QUATERNIONS)


def _numpy_quaternion_norm_checked():
    coefficients = quaternion.as_float_array(QUATERNIONS_NUMPY_QUATERNION)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("quaternion inputs must be finite")
    result = np.abs(QUATERNIONS_NUMPY_QUATERNION)
    if not np.all(np.isfinite(result)):
        raise OverflowError("quaternion norm is too large for a float")
    return result


def _numpy_quaternion_normalize():
    return quaternion.as_float_array(
        QUATERNIONS_NUMPY_QUATERNION / np.abs(QUATERNIONS_NUMPY_QUATERNION)
    )[:, (1, 2, 3, 0)]


def _numpy_quaternion_normalize_checked():
    input_coefficients = quaternion.as_float_array(QUATERNIONS_NUMPY_QUATERNION)
    if not np.all(np.isfinite(input_coefficients)):
        raise ValueError("quaternion inputs must be finite")
    normalized = QUATERNIONS_NUMPY_QUATERNION / np.abs(
        QUATERNIONS_NUMPY_QUATERNION
    )
    output_coefficients = quaternion.as_float_array(normalized)
    if not np.all(np.isfinite(output_coefficients)):
        raise OverflowError("normalization produced a non-finite result")
    return output_coefficients[:, (1, 2, 3, 0)]


def _numpy_quaternion_inverse():
    return quaternion.as_float_array(1.0 / QUATERNIONS_NUMPY_QUATERNION)[
        :, (1, 2, 3, 0)
    ]


def _numpy_quaternion_inverse_checked():
    input_coefficients = quaternion.as_float_array(QUATERNIONS_NUMPY_QUATERNION)
    if not np.all(np.isfinite(input_coefficients)):
        raise ValueError("quaternion inputs must be finite")
    inverse = 1.0 / QUATERNIONS_NUMPY_QUATERNION
    output_coefficients = quaternion.as_float_array(inverse)
    if not np.all(np.isfinite(output_coefficients)):
        raise OverflowError("inverse produced a non-finite result")
    return output_coefficients[:, (1, 2, 3, 0)]


def _numpy_quaternion_dot():
    products = np.conjugate(QUATERNIONS_NUMPY_QUATERNION) * (
        OTHER_QUATERNIONS_NUMPY_QUATERNION
    )
    return quaternion.as_float_array(products)[:, 0]


def _numpy_quaternion_dot_checked():
    left = quaternion.as_float_array(QUATERNIONS_NUMPY_QUATERNION)
    right = quaternion.as_float_array(OTHER_QUATERNIONS_NUMPY_QUATERNION)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("quaternion inputs must be finite")
    products = np.conjugate(QUATERNIONS_NUMPY_QUATERNION) * (
        OTHER_QUATERNIONS_NUMPY_QUATERNION
    )
    coefficients = quaternion.as_float_array(products)
    if not np.all(np.isfinite(coefficients)):
        raise OverflowError("dot product produced a non-finite result")
    return coefficients[:, 0]


def test_numpy_nlerp_compiled_vs_scalar(benchmark):
    benchmark.group = "NumPy NLERP (10k)"
    expected = _scalar_interpolation(hamilton_scalar.nlerp)
    actual = xyzw.nlerp(
        QUATERNIONS, OTHER_QUATERNIONS, INTERPOLATION_AMOUNT
    )
    assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    benchmark(xyzw.nlerp, QUATERNIONS, OTHER_QUATERNIONS, INTERPOLATION_AMOUNT)


def test_numpy_nlerp_scalar_loop(benchmark):
    benchmark.group = "NumPy NLERP (10k)"
    benchmark(_scalar_interpolation, hamilton_scalar.nlerp)


def test_numpy_slerp_compiled_vs_numpy_fallback(benchmark):
    benchmark.group = "NumPy SLERP (10k)"
    expected = _numpy._slerp(
        QUATERNIONS, OTHER_QUATERNIONS, INTERPOLATION_AMOUNT, True, True
    )
    actual = xyzw.slerp(
        QUATERNIONS, OTHER_QUATERNIONS, INTERPOLATION_AMOUNT
    )
    assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)
    benchmark(xyzw.slerp, QUATERNIONS, OTHER_QUATERNIONS, INTERPOLATION_AMOUNT)


def test_numpy_slerp_numpy_fallback(benchmark):
    benchmark.group = "NumPy SLERP (10k)"
    benchmark(
        _numpy._slerp,
        QUATERNIONS,
        OTHER_QUATERNIONS,
        INTERPOLATION_AMOUNT,
        True,
        True,
    )


def test_numpy_from_mrp_compiled_vs_scalar(benchmark):
    benchmark.group = "NumPy from MRP (10k)"
    expected = _scalar_mrp_construction()
    assert_allclose(xyzw.from_mrp(MRPS), expected, rtol=2e-14, atol=2e-14)
    benchmark(xyzw.from_mrp, MRPS)


def test_numpy_from_mrp_scalar_loop(benchmark):
    benchmark.group = "NumPy from MRP (10k)"
    benchmark(_scalar_mrp_construction)


def test_numpy_to_mrp_compiled_vs_scalar(benchmark):
    benchmark.group = "NumPy to MRP (10k)"
    expected = _scalar_mrp_extraction()
    assert_allclose(xyzw.to_mrp(QUATERNIONS), expected, rtol=2e-14, atol=2e-14)
    benchmark(xyzw.to_mrp, QUATERNIONS)


def test_numpy_to_mrp_scalar_loop(benchmark):
    benchmark.group = "NumPy to MRP (10k)"
    benchmark(_scalar_mrp_extraction)


def test_numpy_norm_compiled(benchmark):
    benchmark.group = "NumPy norm (10k)"
    expected = np.linalg.norm(QUATERNIONS, axis=-1)
    assert_allclose(xyzw.norm(QUATERNIONS), expected, rtol=2e-14, atol=2e-14)
    benchmark(xyzw.norm, QUATERNIONS)


def test_numpy_norm_numpy(benchmark):
    benchmark.group = "NumPy norm (10k)"
    benchmark(np.linalg.norm, QUATERNIONS, axis=-1)


def test_numpy_norm_numpy_quaternion(benchmark):
    benchmark.group = "NumPy norm (10k)"
    expected = np.linalg.norm(QUATERNIONS, axis=-1)
    assert_allclose(np.abs(QUATERNIONS_NUMPY_QUATERNION), expected)
    benchmark(np.abs, QUATERNIONS_NUMPY_QUATERNION)


def test_numpy_norm_numpy_quaternion_checked(benchmark):
    benchmark.group = "NumPy norm (10k)"
    assert_allclose(
        _numpy_quaternion_norm_checked(), np.linalg.norm(QUATERNIONS, axis=-1)
    )
    benchmark(_numpy_quaternion_norm_checked)


def test_numpy_normalize_compiled(benchmark):
    benchmark.group = "NumPy normalize (10k)"
    expected = _numpy_normalize()
    assert_allclose(xyzw.normalize(QUATERNIONS), expected, rtol=2e-14, atol=2e-14)
    benchmark(xyzw.normalize, QUATERNIONS)


def test_numpy_normalize_numpy(benchmark):
    benchmark.group = "NumPy normalize (10k)"
    benchmark(_numpy_normalize)


def test_numpy_normalize_numpy_quaternion(benchmark):
    benchmark.group = "NumPy normalize (10k)"
    assert_allclose(_numpy_quaternion_normalize(), _numpy_normalize())
    benchmark(_numpy_quaternion_normalize)


def test_numpy_normalize_numpy_quaternion_checked(benchmark):
    benchmark.group = "NumPy normalize (10k)"
    assert_allclose(_numpy_quaternion_normalize_checked(), _numpy_normalize())
    benchmark(_numpy_quaternion_normalize_checked)


def test_numpy_inverse_compiled(benchmark):
    benchmark.group = "NumPy inverse (10k)"
    expected = _numpy_inverse()
    assert_allclose(xyzw.inverse(QUATERNIONS), expected, rtol=2e-14, atol=2e-14)
    benchmark(xyzw.inverse, QUATERNIONS)


def test_numpy_inverse_numpy(benchmark):
    benchmark.group = "NumPy inverse (10k)"
    benchmark(_numpy_inverse)


def test_numpy_inverse_numpy_quaternion(benchmark):
    benchmark.group = "NumPy inverse (10k)"
    assert_allclose(_numpy_quaternion_inverse(), _numpy_inverse())
    benchmark(_numpy_quaternion_inverse)


def test_numpy_inverse_numpy_quaternion_checked(benchmark):
    benchmark.group = "NumPy inverse (10k)"
    assert_allclose(_numpy_quaternion_inverse_checked(), _numpy_inverse())
    benchmark(_numpy_quaternion_inverse_checked)


def test_numpy_dot_compiled(benchmark):
    benchmark.group = "NumPy dot (10k)"
    expected = _numpy_dot()
    assert_allclose(xyzw.dot(QUATERNIONS, OTHER_QUATERNIONS), expected)
    benchmark(xyzw.dot, QUATERNIONS, OTHER_QUATERNIONS)


def test_numpy_dot_numpy(benchmark):
    benchmark.group = "NumPy dot (10k)"
    benchmark(_numpy_dot)


def test_numpy_dot_numpy_quaternion(benchmark):
    benchmark.group = "NumPy dot (10k)"
    assert_allclose(_numpy_quaternion_dot(), _numpy_dot())
    benchmark(_numpy_quaternion_dot)


def test_numpy_dot_numpy_quaternion_checked(benchmark):
    benchmark.group = "NumPy dot (10k)"
    assert_allclose(_numpy_quaternion_dot_checked(), _numpy_dot())
    benchmark(_numpy_quaternion_dot_checked)


def test_numpy_integration_compiled_vs_scalar(benchmark):
    benchmark.group = "NumPy sampled integration (256x64)"
    expected = _scalar_integration()
    actual = xyzw.integrate_angular_velocity_samples(
        INITIAL_ATTITUDES, ANGULAR_RATES, TIMES
    )
    assert_allclose(actual, expected, rtol=4e-13, atol=4e-14)
    benchmark(
        xyzw.integrate_angular_velocity_samples,
        INITIAL_ATTITUDES,
        ANGULAR_RATES,
        TIMES,
    )


def test_numpy_integration_scalar_loop(benchmark):
    benchmark.group = "NumPy sampled integration (256x64)"
    benchmark(_scalar_integration)
