"""Component-first NumPy adapters preserve the trailing-axis kernel behavior."""

import importlib
import inspect
import math

import numpy as np
import pytest

BACKENDS = (
    ("hamilton", "wxyz"),
    ("hamilton", "xyzw"),
    ("jpl", "wxyz"),
    ("jpl", "xyzw"),
)


def _to_component_first(array):
    return np.moveaxis(array, -1, 0)


def _to_component_last(array):
    return np.moveaxis(array, 0, -1)


def _module(convention, order, leading):
    suffix = ".components_first" if leading else ""
    return importlib.import_module(f"quatops.numpy.{convention}{suffix}.{order}")


def _physical_coefficients(values, convention, order):
    array = np.asarray(values, dtype=np.float64).copy()
    if convention == "jpl":
        array[..., 1:] *= -1.0
    if order == "xyzw":
        array = array[..., (1, 2, 3, 0)]
    return array


@pytest.mark.parametrize(("convention", "order"), BACKENDS)
def test_component_first_unary_operations_match_trailing_modules(convention, order):
    trailing = _module(convention, order, False)
    leading = _module(convention, order, True)
    hamilton = np.array((((1.0, 0.2, -0.3, 0.4),),
                         ((0.7, -0.1, 0.5, 0.2),)))
    q_last = _physical_coefficients(hamilton, convention, order)
    q_first = _to_component_first(q_last)
    original = q_first.copy()
    q_first.flags.writeable = False

    quaternion_operations = (
        "conjugate", "negate", "normalize", "inverse", "exp", "log", "sin", "cos",
        "sinh", "cosh", "tanh", "power", "pow_real", "sqrt", "to_wxyz", "to_xyzw",
    )
    for name in quaternion_operations:
        args = (q_first, 1.25) if name in ("power", "pow_real") else (q_first,)
        expected_args = (q_last, 1.25) if name in ("power", "pow_real") else (q_last,)
        actual = getattr(leading, name)(*args)
        expected = getattr(trailing, name)(*expected_args)
        np.testing.assert_array_equal(_to_component_last(actual), expected)

    for name, width in (("to_axis_angle", 4), ("to_rotation_vector", 3), ("to_rotvec", 3)):
        actual = getattr(leading, name)(q_first)
        expected = getattr(trailing, name)(q_last)
        assert actual.shape[0] == width
        np.testing.assert_array_equal(_to_component_last(actual), expected)

    assert leading.norm(q_first).shape == (2, 1)
    assert leading.to_rotation_matrix(q_first).shape == (3, 3, 2, 1)
    assert leading.to_left_matrix(q_first).shape == (4, 4, 2, 1)
    np.testing.assert_array_equal(leading.to_rotation_matrix(q_first),
                                  np.moveaxis(trailing.to_rotation_matrix(q_last), (-2, -1), (0, 1)))
    np.testing.assert_array_equal(leading.to_left_matrix(q_first),
                                  np.moveaxis(trailing.to_left_matrix(q_last), (-2, -1), (0, 1)))
    np.testing.assert_array_equal(q_first, original)


@pytest.mark.parametrize(("convention", "order"), BACKENDS)
def test_component_first_broadcasting_and_vector_outputs(convention, order):
    trailing = _module(convention, order, False)
    leading = _module(convention, order, True)
    left_last = _physical_coefficients(
        np.array((((1., 0.1, 0.2, -0.1),), ((.8, -.2, .1, .3),))), convention, order
    )
    right_last = _physical_coefficients(
        np.array((((.9, .2, 0.1, .1),), ((.7, .1, -.1, .2),), ((1., 0., .2, .1),))),
        convention, order,
    ).transpose(1, 0, 2)
    left, right = _to_component_first(left_last), _to_component_first(right_last)
    assert left.shape == (4, 2, 1)
    assert right.shape == (4, 1, 3)

    for name in ("add", "subtract", "multiply", "divide"):
        actual = getattr(leading, name)(left, right)
        expected = getattr(trailing, name)(left_last, right_last)
        assert actual.shape == (4, 2, 3)
        np.testing.assert_array_equal(_to_component_last(actual), expected)

    vectors_last = np.array((((1., 0., 0.), (.2, -.5, .7), (-1., 2., 1.)),))
    vectors_first = _to_component_first(vectors_last)
    rotated = leading.rotate(left, vectors_first)
    expected_rotated = trailing.rotate(left_last, vectors_last)
    assert rotated.shape == (3, 2, 3)
    np.testing.assert_array_equal(_to_component_last(rotated), expected_rotated)

    assert leading.norm(left).shape == (2, 1)
    assert leading.to_rotvec(left).shape == (3, 2, 1)


@pytest.mark.parametrize(("convention", "order"), BACKENDS)
def test_component_first_constructors_jacobians_and_integration(convention, order):
    trailing = _module(convention, order, False)
    leading = _module(convention, order, True)
    rotation_vectors_last = np.array((((0., 0., 0.),), ((0., 0., math.pi / 2),)))
    rotation_vectors_first = _to_component_first(rotation_vectors_last)
    q = leading.from_rotation_vector(rotation_vectors_first)
    assert q.shape == (4, 2, 1)
    np.testing.assert_array_equal(_to_component_last(q),
                                  trailing.from_rotation_vector(rotation_vectors_last))

    axis_angle_last = np.array((((1., 0., 0., .2),), ((0., 1., 0., -.3),)))
    axis_angle_first = _to_component_first(axis_angle_last)
    built = leading.from_axis_angle(axis_angle_first)
    np.testing.assert_array_equal(_to_component_last(built),
                                  trailing.from_axis_angle(axis_angle_last))

    wxyz_last = np.array((((.8, .1, -.2, .3),), ((.9, -.2, .1, .1),)))
    xyzw_last = wxyz_last[..., (1, 2, 3, 0)]
    for name, values in (("from_wxyz", wxyz_last), ("from_xyzw", xyzw_last)):
        actual = getattr(leading, name)(_to_component_first(values))
        expected = getattr(trailing, name)(values)
        np.testing.assert_array_equal(_to_component_last(actual), expected)

    jacobian = leading.so3_left_jacobian(rotation_vectors_first)
    assert jacobian.shape == (3, 3, 2, 1)
    np.testing.assert_array_equal(
        jacobian,
        np.moveaxis(trailing.so3_left_jacobian(rotation_vectors_last), (-2, -1), (0, 1)),
    )

    angular_velocity_last = np.array((((0., 0., 0.),), ((0., 0., -0.5),)))
    angular_velocity_first = _to_component_first(angular_velocity_last)
    integrated = leading.integrate_angular_velocity(q, angular_velocity_first, .5)
    expected = trailing.integrate_angular_velocity(q=_to_component_last(q),
                                                   omega=angular_velocity_last, dt=.5)
    assert integrated.shape == (4, 2, 1)
    np.testing.assert_array_equal(_to_component_last(integrated), expected)


@pytest.mark.parametrize(("convention", "order"), BACKENDS)
def test_component_first_interpolation_matrices_and_vector_kernels(convention, order):
    trailing = _module(convention, order, False)
    leading = _module(convention, order, True)
    q0_last = _physical_coefficients(
        np.array((((1., .1, .2, -.1),), ((.8, -.2, .1, .3),))), convention, order
    )
    q1_last = _physical_coefficients(
        np.array((((.9, .2, .1, .1),), ((.7, .1, -.1, .2),), ((1., 0., .2, .1),))),
        convention,
        order,
    ).transpose(1, 0, 2)
    q0_first, q1_first = _to_component_first(q0_last), _to_component_first(q1_last)

    for name, args in (
        ("scale", (q0_first, -2.0)),
        ("slerp", (q0_first, q1_first, 0.35)),
    ):
        expected_args = tuple(
            _to_component_last(value)
            if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == 4
            else value
            for value in args
        )
        actual = getattr(leading, name)(*args)
        expected = getattr(trailing, name)(*expected_args)
        expected_shape = (4, 2, 3) if name == "slerp" else (4, 2, 1)
        assert actual.shape == expected_shape
        np.testing.assert_array_equal(_to_component_last(actual), expected)

    vector_last = np.array((((.2, -.1, .3), (.7, .2, -.4), (-.2, .5, .1)),))
    vector_first = _to_component_first(vector_last)
    for name in (
        "skew_matrix", "so3_right_jacobian", "so3_left_jacobian_inverse",
        "so3_right_jacobian_inverse",
    ):
        actual = getattr(leading, name)(vector_first)
        expected = getattr(trailing, name)(vector_last)
        assert actual.shape == (3, 3, 1, 3)
        np.testing.assert_array_equal(
            actual, np.moveaxis(expected, (-2, -1), (0, 1))
        )

    actual_matrix = leading.to_right_matrix(q0_first)
    expected_matrix = trailing.to_right_matrix(q0_last)
    assert actual_matrix.shape == (4, 4, 2, 1)
    np.testing.assert_array_equal(
        actual_matrix, np.moveaxis(expected_matrix, (-2, -1), (0, 1))
    )

    assert inspect.signature(leading.integrate_angular_velocity) == inspect.signature(
        trailing.integrate_angular_velocity
    )
    assert leading.integrate_angular_velocity.__annotations__ == trailing.integrate_angular_velocity.__annotations__


@pytest.mark.parametrize("order", ("wxyz", "xyzw"))
def test_component_first_jpl_hamilton_conversion(order):
    jpl = _module("jpl", order, False)
    leading_jpl = _module("jpl", order, True)
    q_hamilton_last = np.array((((1., .2, -.3, .4),), ((.7, -.1, .5, .2),)))
    q_hamilton_first = _to_component_first(q_hamilton_last)
    expected_jpl = jpl.from_hamilton(q_hamilton_last)
    actual_jpl = leading_jpl.from_hamilton(q_hamilton_first)
    np.testing.assert_array_equal(_to_component_last(actual_jpl), expected_jpl)
    np.testing.assert_array_equal(
        _to_component_last(leading_jpl.to_hamilton(_to_component_first(expected_jpl))),
        jpl.to_hamilton(expected_jpl),
    )


@pytest.mark.parametrize(("convention", "order"), BACKENDS)
def test_component_first_empty_batches_validation_and_signatures(convention, order):
    trailing = _module(convention, order, False)
    leading = _module(convention, order, True)
    empty_q = np.empty((4, 0, 2))
    empty_v = np.empty((3, 0, 2))
    assert leading.add(empty_q, empty_q).shape == (4, 0, 2)
    assert leading.rotate(empty_q, empty_v).shape == (3, 0, 2)
    assert leading.to_rotation_matrix(empty_q).shape == (3, 3, 0, 2)
    assert leading.from_rotvec(empty_v).shape == (4, 0, 2)
    assert leading.multiply is not trailing.multiply
    assert leading.multiply.__doc__
    assert "(4, ...)" in leading.from_axis_angle.__doc__
    assert "(..., 4)" not in leading.from_axis_angle.__doc__
    for name in leading.__all__:
        assert hasattr(trailing, name)
        assert inspect.signature(getattr(leading, name)) == inspect.signature(
            getattr(trailing, name)
        )
        assert getattr(leading, name).__annotations__ == getattr(
            trailing, name
        ).__annotations__
    assert np.isscalar(leading.norm(np.array((1., 0., 0., 0.))))

    with pytest.raises(ValueError):
        leading.normalize(np.ones((3, 5)))
    with pytest.raises(ValueError):
        leading.normalize(np.zeros(4))


def test_component_first_aliases_are_direct_exports():
    for convention, order in BACKENDS:
        alias = importlib.import_module(f"quatops.{convention}.numpy_{order}_components_first")
        backend = _module(convention, order, True)
        assert alias.multiply is backend.multiply
        assert alias.rotate is backend.rotate
        assert set(alias.__all__) == set(backend.__all__)


@pytest.mark.parametrize(("convention", "order"), BACKENDS)
def test_square_component_first_input_uses_axis_zero(convention, order):
    trailing = _module(convention, order, False)
    leading = _module(convention, order, True)
    # The same 4x4 storage is a batch of four component-first quaternions in
    # the leading API and one component-last batch with four quaternions in
    # the trailing API. Make the interpretations measurably distinct.
    square = np.array(
        ((1.0, .2, .4, .1), (.3, 1.0, -.1, .5),
         (.2, -.4, 1.0, .7), (-.2, .1, .3, 1.0))
    )
    actual = leading.normalize(square)
    expected = trailing.normalize(square.T).T
    assert actual.shape == (4, 4)
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(actual, trailing.normalize(square))
