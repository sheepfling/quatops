"""Extreme vector magnitudes in all pure-Python rotation backends."""

import dataclasses
import math
import sys
import unittest

from quatops import dataclass, namedtuple, scalar


class DataclassCompatibilityTests(unittest.TestCase):
    def test_frozen_manual_slots(self):
        q = dataclass.Quaternion(1, 2, 3, 4)
        self.assertFalse(hasattr(q, '__dict__'))
        self.assertEqual(tuple(q), (1, 2, 3, 4))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            q.x = 8


class ScalarExtremeRotationTests(unittest.TestCase):
    def test_tiny_and_large_vectors(self):
        # A right-handed quarter-turn around z maps (x, 0, 0) to (0, x, 0).
        # Subnormal intermediates must not disappear before the final result.
        tiny = math.nextafter(0.0, 1.0)
        h = math.sqrt(0.5)
        for backend in (scalar, namedtuple, dataclass):
            for qs in (1e-300, 1.0, 1e300):
                for magnitude in (tiny, 1e-300, 1e300, sys.float_info.max):
                    with self.subTest(backend=backend.__name__, qs=qs, magnitude=magnitude):
                        x, y, z = backend.rotate((qs*h, 0, 0, qs*h), (magnitude, 0, 0))
                        self.assertEqual(x, 0.0)
                        self.assertEqual(y, magnitude)
                        self.assertEqual(z, 0.0)

    def test_zero_and_unrepresentable_result(self):
        q = (math.cos(math.pi/8), 0, 0, math.sin(math.pi/8))
        for backend in (scalar, namedtuple, dataclass):
            with self.subTest(backend=backend.__name__):
                self.assertEqual(backend.rotate(q, (0, 0, 0)), (0, 0, 0))
                with self.assertRaises(ValueError):
                    backend.rotate((0, 0, 0, 0), (0, 0, 0))
                with self.assertRaises(OverflowError):
                    backend.rotate(q, (1.4e308, 1.4e308, 0))


if __name__ == '__main__':
    unittest.main()
