"""One set of independent mathematical examples for every representation."""

import math
import unittest

from quatops import dataclass, namedtuple, scalar

try:
    import numpy as np

    from quatops import numpy as qnumpy
except ImportError:
    np = None
    qnumpy = None


# Each adapter constructs the representation explicitly. Scalar-last input is
# supplied as a plain sequence so named scalar-first fields are never mislabeled.
BACKENDS = [
    ('scalar', scalar, tuple),
    ('namedtuple', namedtuple, lambda values: namedtuple.Quaternion(*values)),
    ('dataclass', dataclass, lambda values: dataclass.Quaternion(*values)),
]
if np is not None:
    BACKENDS.append(('numpy', qnumpy, lambda values: np.asarray(values, dtype=float)))


class SharedConformanceTests(unittest.TestCase):
    def assertComponentsClose(self, actual, expected):
        actual, expected = tuple(actual), tuple(expected)
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertTrue(math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-14), (actual, expected))

    def test_shared_core_operations(self):
        q_values, r_values = (1, 2, 3, 4), (5, 6, 7, 8)
        magnitude = math.sqrt(30)
        for name, backend, make in BACKENDS:
            with self.subTest(backend=name):
                q, r = make(q_values), make(r_values)
                # All expected values here are independent of other backends.
                self.assertComponentsClose(backend.from_xyzw((2, 3, 4, 1)), q_values)
                self.assertComponentsClose(backend.to_xyzw(q), (2, 3, 4, 1))
                self.assertComponentsClose(backend.conjugate(q), (1, -2, -3, -4))
                self.assertComponentsClose(backend.multiply(q, r), (-60, 12, 30, 24))
                self.assertTrue(math.isclose(float(backend.norm(q)), magnitude, rel_tol=1e-14))
                self.assertComponentsClose(backend.normalize(q), (1/magnitude, 2/magnitude,
                                                                 3/magnitude, 4/magnitude))
                self.assertComponentsClose(backend.inverse(q), (1/30, -2/30, -3/30, -4/30))
                # sqrt(2)*(1, 0, 0, 1) is nonunit; conjugate alone
                # would scale the rotated vector by its squared norm.
                h = math.sqrt(2)
                quarter_turn = make((h, 0, 0, h))
                self.assertComponentsClose(backend.rotate(quarter_turn, (1, 0, 0)), (0, 1, 0))
                self.assertTrue(math.isclose(float(backend.norm(quarter_turn)), 2.0))
                self.assertComponentsClose(backend.multiply(q, make((1, 0, 0, 0))), q_values)

    def test_shared_keyword_arguments(self):
        for name, backend, make in BACKENDS:
            with self.subTest(backend=name):
                q = make((2, 0, 0, 0))
                identity = make((1, 0, 0, 0))
                self.assertComponentsClose(backend.from_xyzw(q=(0, 0, 0, 2)), (2, 0, 0, 0))
                self.assertComponentsClose(backend.to_xyzw(q=q), (0, 0, 0, 2))
                self.assertComponentsClose(backend.conjugate(q=q), (2, 0, 0, 0))
                self.assertComponentsClose(backend.multiply(q=q, r=identity), (2, 0, 0, 0))
                self.assertEqual(float(backend.norm(q=q)), 2.0)
                self.assertComponentsClose(backend.normalize(q=q), (1, 0, 0, 0))
                self.assertComponentsClose(backend.inverse(q=q), (.5, 0, 0, 0))
                self.assertComponentsClose(backend.rotate(q=q, v=(1, 2, 3)), (1, 2, 3))

    def test_shared_input_and_numerical_errors(self):
        for name, backend, make in BACKENDS:
            with self.subTest(backend=name):
                zero = make((0, 0, 0, 0))
                self.assertEqual(float(backend.norm(zero)), 0.0)
                for operation in (backend.normalize, backend.inverse):
                    with self.assertRaises(ValueError):
                        operation(zero)
                with self.assertRaises(ValueError):
                    backend.rotate(zero, (1, 0, 0))
                for bad in ((1, 2, float('nan'), 4), (1, 2, float('inf'), 4), (1, 2, 3)):
                    for operation in (backend.from_xyzw, backend.to_xyzw, backend.conjugate,
                                      backend.norm, backend.normalize, backend.inverse):
                        with self.assertRaises(ValueError):
                            operation(bad)
                with self.assertRaises(ValueError):
                    backend.multiply(make((1, 0, 0, 0)), (1, 2, 3))
                with self.assertRaises(ValueError):
                    backend.rotate(make((1, 0, 0, 0)), (1, 2))
                with self.assertRaises(OverflowError):
                    backend.multiply(make((1e308, 0, 0, 0)), make((1e308, 0, 0, 0)))


if __name__ == '__main__':
    unittest.main()
