"""Explicit convention imports, backwards compatibility, and layouts."""

import inspect
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

from quatops import dataclass as legacy_dataclass
from quatops import namedtuple as legacy_namedtuple
from quatops import scalar as legacy_scalar
from quatops.hamilton import dataclass as h_dataclass
from quatops.hamilton import namedtuple as h_namedtuple
from quatops.hamilton import scalar as h_scalar
from quatops.jpl import dataclass as j_dataclass
from quatops.jpl import namedtuple as j_namedtuple
from quatops.jpl import scalar as j_scalar
from quatops.numpy.hamilton import wxyz as h_numpy
from quatops.numpy.jpl import xyzw as j_numpy
from tests.typing_tests import get_type_hints

SCALAR_PAIRS = ((legacy_scalar, h_scalar), (legacy_namedtuple, h_namedtuple),
                (legacy_dataclass, h_dataclass))
ALL_PAIRS = SCALAR_PAIRS
JPL_MODULES = (j_scalar, j_namedtuple, j_dataclass, j_numpy)


class HamiltonNamespaceTests:
    def test_hamilton_functions_are_direct_exports_without_forwarders(self):
        for legacy, explicit in ALL_PAIRS:
            assert explicit.__all__ == legacy.__all__
            for name in legacy.__all__:
                function = getattr(legacy, name)
                if inspect.isfunction(function):
                    assert getattr(explicit, name) is function
            if explicit in (h_namedtuple, h_dataclass):
                for name in ('Quaternion', 'AxisAngle', 'RotVec'):
                    assert getattr(explicit, name) is getattr(legacy, name)

    def test_convention_is_chosen_by_module_without_a_parameter(self):
        for legacy, explicit in ALL_PAIRS:
            assert tuple(inspect.signature(explicit.multiply).parameters) == ('lhs', 'rhs')
            assert tuple(inspect.signature(explicit.rotate).parameters) == ('q', 'v')
            assert tuple(explicit.from_wxyz((2, 3, 4, 5))) == (2, 3, 4, 5)
            assert tuple(explicit.to_wxyz((2, 3, 4, 5))) == (2, 3, 4, 5)
            for fn in (explicit.from_wxyz, explicit.to_wxyz):
                with pytest.raises(ValueError):
                    fn((1, 2, 3))
                with pytest.raises(ValueError):
                    fn((1, 2, float('nan'), 4))
            assert tuple(legacy.multiply((0, 1, 0, 0), (0, 0, 1, 0))) == (0, 0, 0, 1)

    def test_explicit_jpl_conversions_change_convention_not_only_storage(self):
        for backend in JPL_MODULES:
            assert tuple(inspect.signature(backend.multiply).parameters) == ('lhs', 'rhs')
            assert tuple(inspect.signature(backend.rotate).parameters) == ('q', 'v')
            for fn in (backend.from_hamilton, backend.to_hamilton):
                hints = get_type_hints(fn)
                assert 'q' in hints
                assert 'return' in hints
            hamilton = (4, 1, 2, 3)
            jpl = (-1, -2, -3, 4)
            assert tuple(backend.from_hamilton(hamilton)) == jpl
            assert tuple(backend.to_hamilton(jpl)) == hamilton
            assert tuple(backend.from_wxyz(hamilton)) == (1, 2, 3, 4)
            if backend in (j_namedtuple, j_dataclass):
                assert type(backend.from_hamilton(hamilton)) is backend.Quaternion
                assert type(backend.to_hamilton(jpl)) is tuple

    def test_jpl_hot_paths_do_not_call_conversion_or_hamilton_kernels(self):
        # Convention-neutral formulas share function objects. Convention-
        # dependent hot kernels must not convert or dispatch through Hamilton.
        hamilton_kernels = (
            'exp', 'log', 'sin', 'cos', 'sinh', 'cosh', 'tanh', 'tan', 'power',
            'slerp', 'integrate_angular_velocity', 'rotation_distance',
            'rotations_equivalent', 'canonicalize', 'from_rotation_matrix',
            'from_two_vectors', 'asin', 'acos', 'atan', 'asinh', 'acosh', 'atanh',
            'expm1', 'log1p', 'log2', 'log10', 'power_quaternion', 'commutator',
            'quaternion_derivative', 'integrate_angular_velocity_samples',
            'boxplus', 'boxminus', 'nlerp', 'squad', 'from_euler', 'to_euler',
            'from_mrp', 'to_mrp', 'swing_twist', 'rotation_mean',
            'rotation_covariance', 'align_vectors', 'random_quaternion',
            'to_homogeneous_matrix', 'from_homogeneous_matrix', 'so3_adjoint',
        )
        with ExitStack() as stack:
            for legacy, explicit in ALL_PAIRS:
                for module in (legacy, explicit):
                    for name in ('multiply', 'rotate'):
                        stack.enter_context(patch.object(
                            module, name, side_effect=AssertionError('Hamilton dispatch')
                        ))
            for name in hamilton_kernels:
                stack.enter_context(patch.object(
                    legacy_scalar, name,
                    side_effect=AssertionError('Hamilton dispatch'),
                ))
            for backend in JPL_MODULES:
                for name in ('from_hamilton', 'to_hamilton'):
                    stack.enter_context(patch.object(
                        backend, name, side_effect=AssertionError('convention conversion')
                    ))
            for backend in JPL_MODULES:
                q, r = (0.1, -0.2, 0.3, 0.9), (-0.2, 0.1, 0.4, 0.8)
                assert tuple(backend.multiply(q, (0, 0, 0, 1))) == q
                actual = backend.rotate((0, 0, -1, 1), (1, 0, 0))
                for value, expected in zip(actual, (0, 1, 0)):
                    np.testing.assert_allclose(float(value), expected, rtol=0, atol=5e-8)
                for name, args in (
                    ('exp', (q,)), ('log', (q,)), ('sin', (q,)), ('cos', (q,)),
                    ('sinh', (q,)), ('cosh', (q,)), ('tanh', (q,)), ('tan', (q,)),
                    ('power', (q, 0.5)), ('slerp', (q, r, 0.4)),
                    ('rotation_distance', (q, r)),
                    ('integrate_angular_velocity', (q, (0.2, -0.1, 0.3), 0.01)),
                    ('canonicalize', (q,)),
                    ('from_two_vectors', ((1, 0, 0), (0, 1, 0))),
                ):
                    getattr(backend, name)(*args)
                if backend is not j_numpy:
                    backend.from_rotation_matrix(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
                    for name, args in (
                        ('asin', (q,)), ('acos', (q,)), ('atan', (q,)),
                        ('asinh', (q,)), ('acosh', (q,)), ('atanh', (q,)),
                        ('expm1', (q,)), ('log1p', (q,)), ('log2', (q,)),
                        ('log10', (q,)), ('power_quaternion', (q, r)),
                        ('commutator', (q, r)),
                        ('quaternion_derivative', (q, (0.2, -0.1, 0.3))),
                        ('integrate_angular_velocity_samples',
                         (q, ((0.2, -0.1, 0.3), (0.1, 0.2, -0.1)), (0.0, 0.05))),
                        ('boxplus', (q, (0.01, -0.02, 0.03))),
                        ('boxminus', (q, r)), ('nlerp', (q, r, 0.4)),
                        ('squad', (q, r, q, r, 0.4)),
                        ('align_sign', (q, r)),
                        ('make_sign_continuous', ((q, r),)),
                        ('from_euler', ((0.2, -0.3, 0.1),)), ('to_euler', (q,)),
                        ('from_mrp', ((0.1, -0.2, 0.05),)), ('to_mrp', (q,)),
                        ('swing_twist', (q, (0.0, 0.0, 1.0))),
                        ('rotation_mean', ((q, r),)),
                        ('rotation_covariance', ((q, r),)),
                        ('align_vectors', (((1, 0, 0), (0, 1, 0)),
                                           ((0, 1, 0), (-1, 0, 0)))),
                        ('random_quaternion', ()), ('random_rotation', ()),
                        ('to_homogeneous_matrix', (q, (1.0, 2.0, 3.0))),
                        ('from_homogeneous_matrix', (((1, 0, 0, 1), (0, 1, 0, 2),
                                                       (0, 0, 1, 3), (0, 0, 0, 1)),)),
                        ('so3_adjoint', (q,)),
                    ):
                        getattr(backend, name)(*args)

    def test_jpl_convention_neutral_scalar_kernels_are_direct_aliases(self):
        for name in (
            'dot', 'norm_squared', 'skew_matrix', 'so3_left_jacobian',
            'so3_right_jacobian', 'so3_left_jacobian_inverse',
            'so3_right_jacobian_inverse', 'align_sign', 'make_sign_continuous',
        ):
            assert getattr(j_scalar, name) is getattr(legacy_scalar, name)
        assert j_scalar.random_rotation is j_scalar.random_quaternion
        assert j_scalar.so3_adjoint is j_scalar.to_rotation_matrix

    def test_numpy_layouts_keep_shape_and_inputs_unchanged(self):
        q = np.array([[[1, 2, 3, 4]], [[5, 6, 7, 8]]], dtype=np.int64)
        old = q.copy()
        q.flags.writeable = False
        for fn in (h_numpy.from_wxyz, h_numpy.to_wxyz):
            out = fn(q)
            assert out.shape == (2, 1, 4)
            assert out.dtype == np.float64
            np.testing.assert_array_equal(out, old)
        empty = np.empty((0, 2, 4))
        assert h_numpy.from_wxyz(empty).shape == (0, 2, 4)
        assert h_numpy.to_wxyz(empty).shape == (0, 2, 4)
        np.testing.assert_array_equal(q, old)

    def test_numpy_namespace_requires_convention_and_layout_modules(self):
        import quatops.numpy as namespace
        import quatops.numpy.hamilton as hamilton_namespace
        import quatops.numpy.jpl as jpl_namespace

        for module in (namespace, hamilton_namespace, jpl_namespace):
            assert not hasattr(module, 'multiply')
        assert h_numpy.multiply is __import__(
            'quatops.hamilton.numpy_wxyz', fromlist=['multiply']
        ).multiply
