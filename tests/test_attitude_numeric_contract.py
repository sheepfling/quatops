"""Documented singularity and convergence behavior for attitude operations."""

import math

import numpy as np
import pytest

from tests.oracle_support import BACKENDS, from_wxyz


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend[0])
def test_inverse_jacobian_and_nonunique_interpolation_reject_singular_inputs(backend):
    _, module, _, is_numpy, component_first = backend
    with pytest.raises(ValueError):
        module.so3_left_jacobian_inverse((2.0 * math.pi, 0.0, 0.0))
    identity = from_wxyz(module, (1.0, 0.0, 0.0, 0.0), is_numpy, component_first)
    antipode = from_wxyz(module, (-1.0, 0.0, 0.0, 0.0), is_numpy, component_first)
    with pytest.raises(ValueError):
        module.slerp(identity, antipode, 0.5, shortest=False)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend[0])
def test_intrinsic_mean_reports_nonconvergence(backend):
    _, module, _, is_numpy, component_first = backend
    vectors = ((1.2, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -0.8))
    samples = [module.from_rotation_vector(vector) for vector in vectors]
    if is_numpy:
        samples = np.stack(samples, axis=1 if component_first else 0)
    with pytest.raises(RuntimeError, match="converge"):
        module.rotation_mean(samples, method="intrinsic", tolerance=1e-30, max_iterations=1)
