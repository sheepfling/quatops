"""Algebraic quaternion functions and physical rotation-vector conversions."""

import math

import numpy as np
import pytest

from quatops.hamilton import scalar as hamilton
from quatops.jpl import scalar as jpl
from quatops.numpy.hamilton import wxyz as hamilton_numpy
from quatops.numpy.jpl import wxyz as jpl_numpy

try:
    from scipy.linalg import cosm, expm, logm, sinm
except ImportError:
    MATRIX_FUNCTIONS = None
else:
    MATRIX_FUNCTIONS = {"exp": expm, "log": logm, "sin": sinm, "cos": cosm}


def hamilton_matrix(coefficients):
    """Represent w + xi + yj + zk as a complex matrix algebra homomorphism."""
    w, x, y, z = coefficients
    return np.array(
        ((w + 1j * x, y + 1j * z), (-y + 1j * z, w - 1j * x)),
        dtype=np.complex128,
    )


class TestAnalyticFunctions:
    @pytest.mark.parametrize(
        "backend",
        (hamilton, jpl, hamilton_numpy, jpl_numpy),
        ids=("hamilton", "jpl", "numpy-hamilton", "numpy-jpl"),
    )
    def test_algebraic_functions_use_the_same_component_formulas(self, backend):
        coefficients = (0.25, 0.3, -0.4, 0.2)
        w, x, y, z = coefficients
        radius = math.hypot(x, y, z)
        sinc = math.sin(radius) / radius
        sinhc = math.sinh(radius) / radius
        expected = {
            "exp": (math.exp(w) * math.cos(radius), *(math.exp(w) * c * sinc for c in (x, y, z))),
            "log": (
                math.log(math.hypot(w, x, y, z)),
                *(c * math.atan2(radius, w) / radius for c in (x, y, z)),
            ),
            "sin": (math.sin(w) * math.cosh(radius), *(c * math.cos(w) * sinhc for c in (x, y, z))),
            "cos": (math.cos(w) * math.cosh(radius), *(-c * math.sin(w) * sinhc for c in (x, y, z))),
        }
        q = backend.from_wxyz(coefficients)
        for name, result in expected.items():
            actual = backend.to_wxyz(getattr(backend, name)(q))
            np.testing.assert_allclose(actual, result, rtol=1e-14, atol=1e-14)

    @pytest.mark.skipif(MATRIX_FUNCTIONS is None, reason="SciPy is not installed")
    def test_analytic_functions_match_scipy_matrix_functions(self):
        coefficients = [
            (0.4, 0.2, -0.3, 0.5),
            (1.7, -0.6, 0.1, 0.8),
            (-0.8, 0.15, 0.05, -0.1),
        ]
        coefficients.extend(np.random.default_rng(713).normal(size=(8, 4)))
        for backend in (hamilton, jpl):
            for values in coefficients:
                q = backend.from_wxyz(values)
                source_matrix = hamilton_matrix(values)
                for name, matrix_function in MATRIX_FUNCTIONS.items():
                    actual = backend.to_wxyz(getattr(backend, name)(q))
                    np.testing.assert_allclose(
                        hamilton_matrix(actual),
                        matrix_function(source_matrix),
                        rtol=2e-14,
                        atol=2e-14,
                    )

    def test_pure_quaternion_sine_and_cosine_use_hyperbolic_functions(self):
        q = (0, 1, 0, 0)
        np.testing.assert_allclose(hamilton.sin(q), (0, math.sinh(1), 0, 0), rtol=0, atol=1e-15)
        np.testing.assert_allclose(hamilton.cos(q), (math.cosh(1), 0, 0, 0), rtol=0, atol=1e-15)

    @pytest.mark.parametrize("backend", (hamilton, jpl, hamilton_numpy, jpl_numpy))
    def test_zero_vector_part_uses_continuous_limits(self, backend):
        values = (0.75, 0, 0, 0)
        q = backend.from_wxyz(values)
        expected = {
            "exp": (math.exp(0.75), 0, 0, 0),
            "log": (math.log(0.75), 0, 0, 0),
            "sin": (math.sin(0.75), 0, 0, 0),
            "cos": (math.cos(0.75), 0, 0, 0),
        }
        for name, result in expected.items():
            np.testing.assert_allclose(
                backend.to_wxyz(getattr(backend, name)(q)), result, rtol=0, atol=1e-15
            )

    @pytest.mark.parametrize("backend", (hamilton_numpy, jpl_numpy))
    def test_numpy_analytic_functions_preserve_empty_batch_shapes(self, backend):
        q = np.empty((0, 2, 4))
        for operation in (backend.exp, backend.log, backend.sin, backend.cos):
            assert operation(q).shape == q.shape

    @pytest.mark.parametrize("backend", (hamilton, jpl, hamilton_numpy, jpl_numpy))
    def test_log_zero_rejects_and_negative_real_uses_the_documented_axis(self, backend):
        with pytest.raises(ValueError):
            backend.log(backend.from_wxyz((0, 0, 0, 0)))
        result = backend.log(backend.from_wxyz((-2, 0, 0, 0)))
        np.testing.assert_allclose(backend.to_wxyz(result), (math.log(2), math.pi, 0, 0))

    def test_rotation_vector_aliases_keep_the_jpl_physical_sign(self):
        phi = (0, 0, math.pi / 2)
        hq = hamilton.from_rotation_vector(phi)
        jq = jpl.from_rotation_vector(phi)
        root_half = math.sqrt(0.5)

        np.testing.assert_allclose(hamilton.to_wxyz(hq), (root_half, 0, 0, root_half))
        np.testing.assert_allclose(jpl.to_wxyz(jq), (root_half, 0, 0, -root_half))
        np.testing.assert_allclose(hamilton.rotate(hq, (1, 0, 0)), (0, 1, 0), atol=1e-14)
        np.testing.assert_allclose(jpl.rotate(jq, (1, 0, 0)), (0, 1, 0), atol=1e-14)
        np.testing.assert_allclose(hamilton.to_rotation_vector(hq), phi, atol=1e-14)
        np.testing.assert_allclose(jpl.to_rotation_vector(jq), phi, atol=1e-14)
