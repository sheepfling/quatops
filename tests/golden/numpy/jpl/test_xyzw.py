import math

import numpy as np

from quatops.numpy.jpl import xyzw


def test_jpl_xyzw_reference_values():
    q = xyzw.from_hamilton((1, 2, 3, 4))
    r = xyzw.from_hamilton((5, 6, 7, 8))

    np.testing.assert_allclose(xyzw.to_hamilton(xyzw.add(q, r)), (6, 8, 10, 12))
    np.testing.assert_allclose(xyzw.to_hamilton(xyzw.subtract(q, r)), (-4, -4, -4, -4))
    np.testing.assert_allclose(xyzw.to_hamilton(xyzw.negate(q)), (-1, -2, -3, -4))
    np.testing.assert_allclose(xyzw.to_hamilton(xyzw.scale(q, 2)), (2, 4, 6, 8))
    np.testing.assert_allclose(xyzw.to_hamilton(xyzw.multiply(q, r)), (-60, 12, 30, 24))
    np.testing.assert_allclose(xyzw.to_hamilton(xyzw.divide(q, r)), (70 / 174, 8 / 174, 0, 16 / 174))
    np.testing.assert_allclose(xyzw.to_hamilton(xyzw.conjugate(q)), (1, -2, -3, -4))
    np.testing.assert_allclose(float(xyzw.norm(q)), math.sqrt(30), rtol=1e-9, atol=0)
    np.testing.assert_allclose(xyzw.to_hamilton(xyzw.inverse(q)), (1 / 30, -2 / 30, -3 / 30, -4 / 30))

    turn = xyzw.from_hamilton((math.sqrt(0.5), 0, 0, math.sqrt(0.5)))
    np.testing.assert_allclose(xyzw.rotate(turn, (1, 0, 0)), (0, 1, 0), atol=1e-14)
    np.testing.assert_allclose(xyzw.to_axis_angle(xyzw.from_axis_angle((1, 0, 0, math.pi / 2))), (1, 0, 0, math.pi / 2))

    # Algebraic functions use the same coefficient formulas in both conventions.
    pure_i = xyzw.from_wxyz((0, 1, 0, 0))
    np.testing.assert_allclose(
    xyzw.to_wxyz(xyzw.exp(pure_i)),
    (math.cos(1), math.sin(1), 0, 0), rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
    xyzw.to_wxyz(xyzw.log(pure_i)),
    (0, math.pi / 2, 0, 0), rtol=0, atol=1e-15,
    )

    np.testing.assert_allclose(
    xyzw.to_wxyz(xyzw.log(xyzw.from_wxyz((-1, 0, 0, 0)))),
    (0, math.pi, 0, 0), rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
    xyzw.to_wxyz(xyzw.sin(pure_i)),
    (0, math.sinh(1), 0, 0), rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
    xyzw.to_wxyz(xyzw.cos(pure_i)),
    (math.cosh(1), 0, 0, 0), rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
    tuple(xyzw.to_rotation_vector(xyzw.from_rotation_vector((0, 0, math.pi / 2)))),
    (0, 0, math.pi / 2), rtol=0, atol=1e-15,
    )
