"""Explicit convention imports, backwards compatibility, and layouts."""

import inspect
import unittest
from contextlib import ExitStack
from typing import get_type_hints
from unittest.mock import patch

from quatops import dataclass as legacy_dataclass
from quatops import namedtuple as legacy_namedtuple
from quatops import scalar as legacy_scalar
from quatops.hamilton import dataclass as h_dataclass
from quatops.hamilton import namedtuple as h_namedtuple
from quatops.hamilton import scalar as h_scalar
from quatops.jpl import dataclass as j_dataclass
from quatops.jpl import namedtuple as j_namedtuple
from quatops.jpl import scalar as j_scalar

try:
    import numpy as np

    from quatops.numpy.hamilton import wxyz as h_numpy
    from quatops.numpy.jpl import xyzw as j_numpy
except ImportError:
    np = None
    h_numpy = None
    j_numpy = None


SCALAR_PAIRS = ((legacy_scalar, h_scalar), (legacy_namedtuple, h_namedtuple),
                (legacy_dataclass, h_dataclass))
ALL_PAIRS = SCALAR_PAIRS
JPL_MODULES = (j_scalar, j_namedtuple, j_dataclass) + ((j_numpy,) if np is not None else ())
OPERATIONS = (
    'from_xyzw', 'to_xyzw', 'from_wxyz', 'to_wxyz', 'conjugate', 'multiply',
    'norm', 'normalize', 'inverse', 'rotate', 'from_axis_angle', 'to_axis_angle',
    'from_rotvec', 'to_rotvec',
)


class HamiltonNamespaceTests(unittest.TestCase):
    def test_direct_reexports_preserve_function_and_type_identity(self):
        for legacy, explicit in ALL_PAIRS:
            with self.subTest(backend=legacy.__name__):
                for name in OPERATIONS:
                    with self.subTest(name=name):
                        self.assertIs(getattr(explicit, name), getattr(legacy, name))
                if explicit in (h_namedtuple, h_dataclass):
                    for name in ('Quaternion', 'AxisAngle', 'RotVec'):
                        self.assertIs(getattr(explicit, name), getattr(legacy, name))

    def test_convention_is_chosen_by_module_without_a_parameter(self):
        for legacy, explicit in ALL_PAIRS:
            with self.subTest(backend=explicit.__name__):
                self.assertEqual(tuple(inspect.signature(explicit.multiply).parameters), ('q', 'r'))
                self.assertEqual(tuple(inspect.signature(explicit.rotate).parameters), ('q', 'v'))
                self.assertEqual(tuple(explicit.from_wxyz((2, 3, 4, 5))), (2, 3, 4, 5))
                self.assertEqual(tuple(explicit.to_wxyz((2, 3, 4, 5))), (2, 3, 4, 5))
                for fn in (explicit.from_wxyz, explicit.to_wxyz):
                    with self.assertRaises(ValueError):
                        fn((1, 2, 3))
                    with self.assertRaises(ValueError):
                        fn((1, 2, float('nan'), 4))
                self.assertEqual(tuple(legacy.multiply((0, 1, 0, 0), (0, 0, 1, 0))),
                                 (0, 0, 0, 1))

    def test_explicit_jpl_conversions_change_convention_not_only_storage(self):
        for backend in JPL_MODULES:
            with self.subTest(backend=backend.__name__):
                self.assertEqual(tuple(inspect.signature(backend.multiply).parameters), ('q', 'r'))
                self.assertEqual(tuple(inspect.signature(backend.rotate).parameters), ('q', 'v'))
                for fn in (backend.from_hamilton, backend.to_hamilton):
                    hints = get_type_hints(fn)
                    self.assertIn('q', hints)
                    self.assertIn('return', hints)
                hamilton = (4, 1, 2, 3)
                jpl = (-1, -2, -3, 4)
                self.assertEqual(tuple(backend.from_hamilton(hamilton)), jpl)
                self.assertEqual(tuple(backend.to_hamilton(jpl)), hamilton)
                self.assertEqual(tuple(backend.from_wxyz(hamilton)), (1, 2, 3, 4))
                if backend in (j_namedtuple, j_dataclass):
                    self.assertIs(type(backend.from_hamilton(hamilton)), backend.Quaternion)
                    self.assertIs(type(backend.to_hamilton(jpl)), tuple)

    def test_jpl_hot_paths_do_not_call_conversion_or_hamilton_kernels(self):
        # Any accidental cross-convention conversion or dispatch fails here.
        with ExitStack() as stack:
            for legacy, explicit in ALL_PAIRS:
                for module in (legacy, explicit):
                    for name in ('multiply', 'rotate'):
                        stack.enter_context(patch.object(
                            module, name, side_effect=AssertionError('Hamilton dispatch')
                        ))
            for backend in JPL_MODULES:
                for name in ('from_hamilton', 'to_hamilton'):
                    stack.enter_context(patch.object(
                        backend, name, side_effect=AssertionError('convention conversion')
                    ))
            for backend in JPL_MODULES:
                with self.subTest(backend=backend.__name__):
                    q = (0, 0, -1, 1)
                    self.assertEqual(tuple(backend.multiply(q, (0, 0, 0, 1))), q)
                    actual = backend.rotate(q, (1, 0, 0))
                    for value, expected in zip(actual, (0, 1, 0)):
                        self.assertAlmostEqual(float(value), expected)

    @unittest.skipIf(np is None, 'NumPy is not installed')
    def test_numpy_layouts_keep_shape_and_inputs_unchanged(self):
        q = np.array([[[1, 2, 3, 4]], [[5, 6, 7, 8]]], dtype=np.int64)
        old = q.copy()
        q.flags.writeable = False
        for fn in (h_numpy.from_wxyz, h_numpy.to_wxyz):
            out = fn(q)
            self.assertEqual(out.shape, (2, 1, 4))
            self.assertEqual(out.dtype, np.float64)
            np.testing.assert_array_equal(out, old)
        empty = np.empty((0, 2, 4))
        self.assertEqual(h_numpy.from_wxyz(empty).shape, (0, 2, 4))
        self.assertEqual(h_numpy.to_wxyz(empty).shape, (0, 2, 4))
        np.testing.assert_array_equal(q, old)

    @unittest.skipIf(np is None, 'NumPy is not installed')
    def test_numpy_namespace_requires_convention_and_layout_modules(self):
        import quatops.numpy as namespace
        import quatops.numpy.hamilton as hamilton_namespace
        import quatops.numpy.jpl as jpl_namespace

        for module in (namespace, hamilton_namespace, jpl_namespace):
            self.assertFalse(hasattr(module, 'multiply'))
        self.assertIs(h_numpy.multiply, __import__(
            'quatops.hamilton.numpy_wxyz', fromlist=['multiply']
        ).multiply)


if __name__ == '__main__':
    unittest.main()
