import math

import numpy as np

from quatops.numpy.jpl import wxyz


def test_jpl_wxyz_reference_values():
    q = wxyz.from_hamilton((1, 2, 3, 4))
    r = wxyz.from_hamilton((5, 6, 7, 8))

    np.testing.assert_allclose(wxyz.to_hamilton(wxyz.add(q, r)), (6, 8, 10, 12))
    np.testing.assert_allclose(wxyz.to_hamilton(wxyz.subtract(q, r)), (-4, -4, -4, -4))
    np.testing.assert_allclose(wxyz.to_hamilton(wxyz.negate(q)), (-1, -2, -3, -4))
    np.testing.assert_allclose(wxyz.to_hamilton(wxyz.scale(q, 2)), (2, 4, 6, 8))
    np.testing.assert_allclose(wxyz.to_hamilton(wxyz.multiply(q, r)), (-60, 12, 30, 24))
    np.testing.assert_allclose(wxyz.to_hamilton(wxyz.divide(q, r)), (70 / 174, 8 / 174, 0, 16 / 174))
    np.testing.assert_allclose(wxyz.to_hamilton(wxyz.conjugate(q)), (1, -2, -3, -4))
    np.testing.assert_allclose(float(wxyz.norm(q)), math.sqrt(30), rtol=1e-9, atol=0)
    np.testing.assert_allclose(wxyz.to_hamilton(wxyz.inverse(q)), (1 / 30, -2 / 30, -3 / 30, -4 / 30))

    turn = wxyz.from_hamilton((math.sqrt(0.5), 0, 0, math.sqrt(0.5)))
    np.testing.assert_allclose(wxyz.rotate(turn, (1, 0, 0)), (0, 1, 0), atol=1e-14)
    np.testing.assert_allclose(wxyz.to_axis_angle(wxyz.from_axis_angle((1, 0, 0, math.pi / 2))), (1, 0, 0, math.pi / 2))

    # Algebraic functions use the same coefficient formulas in both conventions.
    pure_i = wxyz.from_wxyz((0, 1, 0, 0))
    np.testing.assert_allclose(
        wxyz.to_wxyz(wxyz.exp(pure_i)),
        (math.cos(1), math.sin(1), 0, 0), rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
        wxyz.to_wxyz(wxyz.log(pure_i)),
        (0, math.pi / 2, 0, 0), rtol=0, atol=1e-15,
    )

    np.testing.assert_allclose(
        wxyz.to_wxyz(wxyz.log(wxyz.from_wxyz((-1, 0, 0, 0)))),
        (0, math.pi, 0, 0), rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
        wxyz.to_wxyz(wxyz.sin(pure_i)),
        (0, math.sinh(1), 0, 0), rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
        wxyz.to_wxyz(wxyz.cos(pure_i)),
        (math.cosh(1), 0, 0, 0), rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
        tuple(wxyz.to_rotation_vector(wxyz.from_rotation_vector((0, 0, math.pi / 2)))),
        (0, 0, math.pi / 2), rtol=0, atol=1e-15,
    )
