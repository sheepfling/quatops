"""One set of independent mathematical examples for every representation."""

import pytest
import math

from quatops import dataclass, namedtuple, scalar
from typing import Callable

import numpy as np

from quatops.numpy.hamilton import wxyz as qnumpy

# Each adapter constructs the representation explicitly. Scalar-last input is
# supplied as a plain sequence so named scalar-first fields are never mislabeled.
BACKENDS: list[tuple[str, type, Callable]] = [
    ('scalar', scalar, lambda *a: tuple(*a)),
    ('namedtuple', namedtuple, lambda values: namedtuple.Quaternion(*values)),
    ('dataclass', dataclass, lambda values: dataclass.Quaternion(*values)),
]
BACKENDS.append(('numpy', qnumpy, lambda values: np.asarray(values, dtype=float)))

class SharedConformanceTests:
    def assert_components_close(self, actual, expected):
        actual, expected = tuple(actual), tuple(expected)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-14)

    def test_shared_core_operations(self):
        q_values, r_values = (1, 2, 3, 4), (5, 6, 7, 8)
        magnitude = math.sqrt(30)
        for name, backend, make in BACKENDS:
            q, r = make(q_values), make(r_values)
            # All expected values here are independent of other backends.
            self.assert_components_close(backend.from_xyzw((2, 3, 4, 1)), q_values)
            self.assert_components_close(backend.to_xyzw(q), (2, 3, 4, 1))
            self.assert_components_close(backend.conjugate(q), (1, -2, -3, -4))
            self.assert_components_close(backend.multiply(q, r), (-60, 12, 30, 24))
            np.testing.assert_allclose(float(backend.norm(q)), magnitude, rtol=1e-14, atol=0)
            self.assert_components_close(backend.normalize(q), (1 / magnitude, 2 / magnitude,
                                                              3 / magnitude, 4 / magnitude))
            self.assert_components_close(backend.inverse(q), (1 / 30, -2 / 30, -3 / 30, -4 / 30))
            # sqrt(2)*(1, 0, 0, 1) is nonunit; conjugate alone
            # would scale the rotated vector by its squared norm.
            h = math.sqrt(2)
            quarter_turn = make((h, 0, 0, h))
            self.assert_components_close(backend.rotate(quarter_turn, (1, 0, 0)), (0, 1, 0))
            np.testing.assert_allclose(float(backend.norm(quarter_turn)), 2.0, rtol=1e-9, atol=0)
            self.assert_components_close(backend.multiply(q, make((1, 0, 0, 0))), q_values)

    def test_shared_keyword_arguments(self):
        for name, backend, make in BACKENDS:
            q = make((2, 0, 0, 0))
            identity = make((1, 0, 0, 0))
            self.assert_components_close(backend.from_xyzw(q=(0, 0, 0, 2)), (2, 0, 0, 0))
            self.assert_components_close(backend.to_xyzw(q=q), (0, 0, 0, 2))
            self.assert_components_close(backend.conjugate(q=q), (2, 0, 0, 0))
            self.assert_components_close(backend.multiply(lhs=q, rhs=identity), (2, 0, 0, 0))
            assert float(backend.norm(q=q)) == 2.0
            self.assert_components_close(backend.normalize(q=q), (1, 0, 0, 0))
            self.assert_components_close(backend.inverse(q=q), (.5, 0, 0, 0))
            self.assert_components_close(backend.rotate(q=q, v=(1, 2, 3)), (1, 2, 3))

    def test_shared_input_and_numerical_errors(self):
        for name, backend, make in BACKENDS:
            zero = make((0, 0, 0, 0))
            assert float(backend.norm(zero)) == 0.0
            for operation in (backend.normalize, backend.inverse):
                with pytest.raises(ValueError):
                    operation(zero)
            with pytest.raises(ValueError):
                backend.rotate(zero, (1, 0, 0))
            for bad in ((1, 2, float('nan'), 4), (1, 2, float('inf'), 4), (1, 2, 3)):
                for operation in (backend.from_xyzw, backend.to_xyzw, backend.conjugate,
                                  backend.norm, backend.normalize, backend.inverse):
                    with pytest.raises(ValueError):
                        operation(bad)
            with pytest.raises(ValueError):
                backend.multiply(make((1, 0, 0, 0)), (1, 2, 3))
            with pytest.raises(ValueError):
                backend.rotate(make((1, 0, 0, 0)), (1, 2))
            with pytest.raises(OverflowError):
                backend.multiply(make((1e308, 0, 0, 0)), make((1e308, 0, 0, 0)))
