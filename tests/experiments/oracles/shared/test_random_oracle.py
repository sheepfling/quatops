"""Seeded reference checks for scalar and batched uniform random rotations."""

import importlib

import numpy as np
import pytest

from tests.support.oracle_support import BACKENDS, to_wxyz


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_random_quaternions_are_seeded_unit_and_uniform_on_s3(
    backend_name, module, is_jpl, is_numpy, component_first
):
    count = 4096
    seed = 0x5EED + len(backend_name)
    if is_numpy:
        first = module.random_quaternion(size=(count,), rng=np.random.default_rng(seed))
        second = module.random_rotation(size=(count,), rng=np.random.default_rng(seed))
        coefficients = to_wxyz(module, first, True, component_first)
        second_coefficients = to_wxyz(module, second, True, component_first)
    else:
        first_rng = np.random.default_rng(seed)
        second_rng = np.random.default_rng(seed)
        first = [module.to_wxyz(module.random_quaternion(first_rng)) for _ in range(count)]
        second = [module.to_wxyz(module.random_rotation(second_rng)) for _ in range(count)]
        coefficients = np.asarray(first)
        second_coefficients = np.asarray(second)

    np.testing.assert_allclose(coefficients, second_coefficients, rtol=0, atol=0,
                               err_msg=f"{backend_name} random aliases/reproducibility")
    np.testing.assert_allclose(np.linalg.norm(coefficients, axis=-1), 1.0, rtol=0, atol=3e-15,
                               err_msg=f"{backend_name} random unit norm")
    np.testing.assert_allclose(np.mean(coefficients, axis=0), np.zeros(4), atol=0.025, rtol=0,
                               err_msg=f"{backend_name} random antipodal sign balance")
    second_moment = coefficients.T @ coefficients / count
    np.testing.assert_allclose(second_moment, np.eye(4) / 4.0, atol=0.025, rtol=0,
                               err_msg=f"{backend_name} random S3 second moment")

    if is_numpy:
        empty = module.random_quaternion(size=(0, 2), rng=np.random.default_rng(seed))
        expected_shape = (4, 0, 2) if component_first else (0, 2, 4)
        assert empty.shape == expected_shape


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    [backend for backend in BACKENDS if not backend[3]],
    ids=[backend[0] for backend in BACKENDS if not backend[3]],
)
def test_scalar_random_draws_match_numpy_batch_stream(
    backend_name, module, is_jpl, is_numpy, component_first
):
    numpy_module = importlib.import_module(
        "quatops.numpy.jpl.wxyz" if is_jpl else "quatops.numpy.hamilton.wxyz"
    )
    count = 128
    seed = 0xB17E + len(backend_name)
    scalar_rng = np.random.default_rng(seed)
    scalar_values = np.asarray([
        module.to_wxyz(module.random_quaternion(scalar_rng)) for _ in range(count)
    ])
    array_values = to_wxyz(
        numpy_module,
        numpy_module.random_quaternion(size=count, rng=np.random.default_rng(seed)),
        True,
        False,
    )
    np.testing.assert_allclose(scalar_values, array_values, rtol=3e-15, atol=3e-15,
                               err_msg=f"{backend_name} scalar/NumPy seeded stream parity")
