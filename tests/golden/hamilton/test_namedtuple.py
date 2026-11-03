import math
import numpy as np

from quatops.hamilton import namedtuple

def test_namedtuple_reference_values():
    q = namedtuple.from_wxyz((1, 2, 3, 4))
    r = namedtuple.from_wxyz((5, 6, 7, 8))

    assert tuple(namedtuple.to_xyzw(q)) == (2, 3, 4, 1)
    assert tuple(namedtuple.add(q, r)) == (6, 8, 10, 12)
    assert tuple(namedtuple.subtract(q, r)) == (-4, -4, -4, -4)
    assert tuple(namedtuple.negate(q)) == (-1, -2, -3, -4)
    assert tuple(namedtuple.scale(q, 2)) == (2, 4, 6, 8)
    assert tuple(namedtuple.multiply(q, r)) == (-60, 12, 30, 24)
    np.testing.assert_allclose(namedtuple.divide(q, r), (70 / 174, 8 / 174, 0, 16 / 174))
    assert tuple(namedtuple.conjugate(q)) == (1, -2, -3, -4)
    np.testing.assert_allclose(namedtuple.norm(q), math.sqrt(30), rtol=1e-9, atol=0)
    assert tuple(namedtuple.inverse(q)) == (1 / 30, -2 / 30, -3 / 30, -4 / 30)
    rotated = namedtuple.rotate((math.sqrt(0.5), 0, 0, math.sqrt(0.5)), (1, 0, 0))
    np.testing.assert_allclose(rotated, (0, 1, 0), rtol=1e-9, atol=1e-14)

    turn = namedtuple.from_axis_angle((1, 0, 0, math.pi / 2))
    np.testing.assert_allclose(namedtuple.to_axis_angle(turn), (1, 0, 0, math.pi / 2), rtol=1e-9, atol=1e-14)
    rotvec = namedtuple.to_rotvec(namedtuple.from_rotvec((math.pi / 2, 0, 0)))
    np.testing.assert_allclose(rotvec, (math.pi / 2, 0, 0), rtol=1e-9, atol=1e-14)

    # Algebraic functions use the same coefficient formulas in both conventions.
    pure_i = namedtuple.from_wxyz((0, 1, 0, 0))
    np.testing.assert_allclose(
        namedtuple.to_wxyz(namedtuple.exp(pure_i)),
        (math.cos(1), math.sin(1), 0, 0), rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
        namedtuple.to_wxyz(namedtuple.log(pure_i)),
        (0, math.pi / 2, 0, 0), rtol=0, atol=1e-15,
    )

    np.testing.assert_allclose(
        namedtuple.to_wxyz(namedtuple.log(namedtuple.from_wxyz((-1, 0, 0, 0)))),
        (0, math.pi, 0, 0), rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
        namedtuple.to_wxyz(namedtuple.sin(pure_i)),
        (0, math.sinh(1), 0, 0), rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
        namedtuple.to_wxyz(namedtuple.cos(pure_i)),
        (math.cosh(1), 0, 0, 0), rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
        tuple(namedtuple.to_rotation_vector(namedtuple.from_rotation_vector((0, 0, math.pi / 2)))),
        (0, 0, math.pi / 2), rtol=0, atol=1e-15,
    )
