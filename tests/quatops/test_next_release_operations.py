"""Cross-representation tests for the foundational quaternion API additions."""

import importlib
import math

import numpy as np
import pytest

SCALAR_APIS = (
    "quatops.scalar",
    "quatops.jpl.scalar",
    "quatops.python.hamilton",
    "quatops.python.jpl",
    "quatops.hamilton.scalar",
    "quatops.hamilton.namedtuple",
    "quatops.hamilton.dataclass",
    "quatops.types.hamilton_namedtuple",
    "quatops.types.hamilton_dataclass",
    "quatops.namedtuple",
    "quatops.dataclass",
    "quatops.jpl.namedtuple",
    "quatops.jpl.dataclass",
    "quatops.types.jpl_namedtuple",
    "quatops.types.jpl_dataclass",
)
NUMPY_LAYOUTS = tuple(
    (convention, order, component_first)
    for convention in ("hamilton", "jpl")
    for order in ("wxyz", "xyzw")
    for component_first in (False, True)
)


def _last(module, component_first, value):
    value = np.asarray(value)
    return np.moveaxis(value, 0, -1) if component_first else value


def _first(module, component_first, value):
    value = np.asarray(value)
    return np.moveaxis(value, -1, 0) if component_first else value


def _scalar_q(api, rotvec):
    return api.from_rotation_vector(rotvec)


def test_scalar_and_typed_foundational_operations():
    for path in SCALAR_APIS:
        api = importlib.import_module(path)
        q = api.from_rotation_vector((0.0, 0.0, math.pi / 2))
        q2 = api.from_rotation_vector((0.0, 0.0, math.pi))
        raw = (1.0, 2.0, 3.0, 4.0)
        other = (-2.0, 3.0, 1.0, 0.5)

        assert api.dot(raw, other) == pytest.approx(9.0)
        assert api.norm_squared(raw) == pytest.approx(30.0)
        assert api.norm_squared(iter(raw)) == pytest.approx(30.0)
        assert api.dot(raw, raw) == api.norm_squared(raw)

        q_divisor = api.from_rotation_vector((math.pi, 0.0, 0.0))
        left = api.left_divide(q, q_divisor)
        right = api.right_divide(q, q_divisor)
        assert tuple(api.multiply(q, left)) == pytest.approx(tuple(q_divisor))
        assert tuple(api.multiply(right, q_divisor)) == pytest.approx(tuple(q))
        assert tuple(api.divide(q, q_divisor)) == pytest.approx(tuple(right))
        assert not np.allclose(tuple(left), tuple(right))

        assert api.rotation_distance(q, tuple(-v for v in q)) == pytest.approx(0.0, abs=1e-14)
        assert api.rotation_distance(q, q2) == pytest.approx(math.pi / 2)
        assert api.rotations_equivalent(q, tuple(-v for v in q))
        assert not api.rotations_equivalent(q, q2)
        assert api.rotations_equivalent(q, q2, atol=math.pi)

        canonical = api.canonicalize(tuple(-v for v in q))
        assert api.rotation_distance(q, canonical) == pytest.approx(0.0, abs=1e-14)
        assert api.norm(canonical) == pytest.approx(api.norm(q))

        is_jpl = path.endswith(".jpl") or ".jpl." in path or ".jpl_" in path
        real_q = (0.0, 0.0, 0.0, 0.1) if is_jpl else (0.1, 0.0, 0.0, 0.0)
        expected_tan = ((0.0, 0.0, 0.0, math.tan(0.1)) if is_jpl
                        else (math.tan(0.1), 0.0, 0.0, 0.0))
        assert tuple(api.tan(real_q)) == pytest.approx(
            expected_tan
        )
        sample = api.from_rotation_vector((0.2, -0.1, 0.3))
        assert tuple(api.tan(sample)) == pytest.approx(
            tuple(api.divide(api.sin(sample), api.cos(sample)))
        )

        for rv in ((math.pi, 0.0, 0.0), (0.0, math.pi, 0.0), (0.0, 0.0, math.pi)):
            rotation = api.from_rotation_vector(rv)
            matrix = api.to_rotation_matrix(rotation)
            recovered = api.from_rotation_matrix(matrix)
            assert api.rotation_distance(rotation, recovered) == pytest.approx(0.0, abs=1e-14)

        align = api.from_two_vectors((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        assert api.rotate(align, (1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0))
        anti = api.from_two_vectors((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
        assert api.rotate(anti, (1.0, 0.0, 0.0)) == pytest.approx((-1.0, 0.0, 0.0))
        with pytest.raises(ValueError):
            api.from_two_vectors((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

        with pytest.raises(ValueError):
            api.from_rotation_matrix(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)))
        with pytest.raises(ValueError):
            api.from_rotation_matrix(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), atol=1.0)
        with pytest.raises(ValueError):
            api.rotations_equivalent(q, q2, atol=-1.0)
        assert tuple(api.canonicalize((0.0, 0.0, 0.0, 0.0))) == (0.0, 0.0, 0.0, 0.0)

        assert api.dot((1e308, 1e308, 0.0, 0.0), (1e308, -1e308, 0.0, 0.0)) == 0.0
        assert api.dot((1e308, 1e-308, 0.0, 0.0), (1e-308, 1e308, 0.0, 0.0)) == pytest.approx(2.0)
        with pytest.raises(OverflowError):
            api.dot((1e308, 0.0, 0.0, 0.0), (1e308, 0.0, 0.0, 0.0))


def test_one_shot_iterators_work_across_scalar_surfaces():
    raw = (1.0, 2.0, 3.0, 4.0)
    other = (2.0, -1.0, 0.5, 3.0)
    vector = (0.25, -0.5, 1.0)
    matrix = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    def once(values):
        return iter(values)

    for path in SCALAR_APIS:
        api = importlib.import_module(path)
        assert api.norm_squared(once(raw)) == pytest.approx(30.0)
        assert api.dot(once(raw), once(other)) == api.dot(raw, other)
        np.testing.assert_allclose(
            tuple(api.multiply(once(raw), once(other))),
            tuple(api.multiply(raw, other)),
        )
        np.testing.assert_allclose(
            tuple(api.normalize(once(raw))), tuple(api.normalize(raw))
        )
        np.testing.assert_allclose(
            tuple(api.rotate(once(raw), once(vector))),
            tuple(api.rotate(raw, vector)),
        )
        np.testing.assert_allclose(
            tuple(api.from_rotation_vector(once(vector))),
            tuple(api.from_rotation_vector(vector)),
        )
        np.testing.assert_allclose(
            tuple(api.from_two_vectors(once(vector), once((0.0, 1.0, 0.0)))),
            tuple(api.from_two_vectors(vector, (0.0, 1.0, 0.0))),
        )
        np.testing.assert_allclose(
            tuple(api.from_rotation_matrix(once(once(row) for row in matrix))),
            tuple(api.from_rotation_matrix(matrix)),
        )


@pytest.mark.parametrize(("convention", "order", "component_first"), NUMPY_LAYOUTS)
def test_numpy_foundational_operations_broadcast_and_preserve_layout(
    convention, order, component_first
):
    suffix = ".components_first" if component_first else ""
    api = importlib.import_module(f"quatops.numpy.{convention}{suffix}.{order}")
    trailing = importlib.import_module(f"quatops.numpy.{convention}.{order}")
    q0_last = trailing.from_rotation_vector(
        np.array((((0.0, 0.0, 0.0),), ((0.0, 0.0, math.pi / 2),)))
    )
    q1_last = trailing.from_rotation_vector(
        np.array((((0.0, 0.0, math.pi),), ((0.0, 0.0, -math.pi / 2),), ((0.0, math.pi, 0.0),)))
    ).transpose(1, 0, 2)
    q0 = _first(api, component_first, q0_last)
    q1 = _first(api, component_first, q1_last)

    assert api.dot(q0, q0).shape == (2, 1)
    np.testing.assert_allclose(api.norm_squared(q0), api.dot(q0, q0), rtol=0, atol=0)
    product_shape = (4, 2, 3) if component_first else (2, 3, 4)
    assert api.left_divide(q0, q1).shape == product_shape
    assert api.right_divide(q0, q1).shape == product_shape

    distances = api.rotation_distance(q0, q1)
    assert distances.shape == (2, 3)
    assert api.rotations_equivalent(q0, q0).shape == (2, 1)
    assert np.all(api.rotations_equivalent(q0, -q0))

    canonical = api.canonicalize(-q0)
    assert canonical.shape == q0.shape
    np.testing.assert_allclose(api.rotation_distance(q0, canonical), 0.0, atol=1e-14)
    assert api.tan(q0).shape == q0.shape

    extreme_lhs = np.array(((1e308, 1e308, 0.0, 0.0), (1e308, 1e-308, 0.0, 0.0)))
    extreme_rhs = np.array(((1e308, -1e308, 0.0, 0.0), (1e-308, 1e308, 0.0, 0.0)))
    np.testing.assert_allclose(
        api.dot(_first(api, component_first, extreme_lhs),
                _first(api, component_first, extreme_rhs)),
        (0.0, 2.0),
        rtol=0.0,
        atol=1e-15,
    )
    assert api.dot(_first(api, component_first, extreme_lhs[0]),
                   _first(api, component_first, extreme_rhs[0])) == 0.0
    with pytest.raises(OverflowError):
        api.dot(_first(api, component_first, extreme_lhs[0]),
                _first(api, component_first, extreme_lhs[0]))

    matrices = api.to_rotation_matrix(q0)
    if component_first:
        assert matrices.shape == (3, 3, 2, 1)
    else:
        assert matrices.shape == (2, 1, 3, 3)
    recovered = api.from_rotation_matrix(matrices)
    np.testing.assert_allclose(api.rotation_distance(q0, recovered), 0.0, atol=1e-14)

    source = np.array((1.0, 0.0, 0.0))
    target = np.array((0.0, 1.0, 0.0))
    if component_first:
        source = source[:, None]
        target = target[:, None]
    aligned = api.from_two_vectors(source, target)
    result = api.rotate(aligned, source)
    expected = ((0.0, 1.0, 0.0),) if component_first else (0.0, 1.0, 0.0)
    np.testing.assert_allclose(_last(api, component_first, result), expected, atol=1e-14)


def test_numpy_rotation_matrix_batches_reject_invalid_matrices():
    from quatops.numpy.hamilton import wxyz

    identity = np.eye(3)
    reflection = identity.copy()
    reflection[2, 2] = -1
    with pytest.raises(ValueError):
        wxyz.from_rotation_matrix(reflection)
    with pytest.raises(ValueError):
        wxyz.from_rotation_matrix(np.ones((2, 3, 4)))
    with pytest.raises(ValueError):
        wxyz.from_rotation_matrix(identity, atol=-1)

    empty = np.empty((0, 3, 3))
    assert wxyz.from_rotation_matrix(empty).shape == (0, 4)
