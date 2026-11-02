import math
import numpy as np

from quatops.hamilton import scalar

def test_scalar_reference_values() -> None:
    q = scalar.from_wxyz((1, 2, 3, 4))
    r = scalar.from_wxyz((5, 6, 7, 8))

    assert scalar.to_xyzw(q) == (2, 3, 4, 1)
    assert scalar.add(q, r) == (6, 8, 10, 12)
    assert scalar.subtract(q, r) == (-4, -4, -4, -4)
    assert scalar.negate(q) == (-1, -2, -3, -4)
    assert scalar.scale(q, 2) == (2, 4, 6, 8)
    assert scalar.multiply(q, r) == (-60, 12, 30, 24)
    np.testing.assert_allclose(scalar.divide(q, r), (70 / 174, 8 / 174, 0, 16 / 174))
    assert scalar.conjugate(q) == (1, -2, -3, -4)
    np.testing.assert_allclose(scalar.inverse(q), (1 / 30, -2 / 30, -3 / 30, -4 / 30))
    np.testing.assert_allclose(scalar.norm(q), math.sqrt(30), rtol=1e-9, atol=0)
    rotated = scalar.rotate((math.sqrt(0.5), 0, 0, math.sqrt(0.5)), (1, 0, 0))
    np.testing.assert_allclose(rotated, (0, 1, 0), atol=1e-14)

    turn = scalar.from_axis_angle((1, 0, 0, math.pi / 2))
    np.testing.assert_allclose(scalar.to_axis_angle(turn), (1, 0, 0, math.pi / 2), atol=1e-14)
    rotvec = scalar.to_rotvec(scalar.from_rotvec((math.pi / 2, 0, 0)))
    np.testing.assert_allclose(rotvec, (math.pi / 2, 0, 0), atol=1e-14)

    # Algebraic functions use the same coefficient formulas in both conventions.
    pure_i = scalar.from_wxyz((0, 1, 0, 0))
    np.testing.assert_allclose(
        scalar.to_wxyz(scalar.exp(pure_i)),
        (math.cos(1), math.sin(1), 0, 0), rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
        scalar.to_wxyz(scalar.log(pure_i)),
        (0, math.pi / 2, 0, 0), rtol=0, atol=1e-15,
    )

    np.testing.assert_allclose(
        scalar.to_wxyz(scalar.log(scalar.from_wxyz((-1, 0, 0, 0)))),
        (0, math.pi, 0, 0), rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
        scalar.to_wxyz(scalar.sin(pure_i)),
        (0, math.sinh(1), 0, 0), rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
        scalar.to_wxyz(scalar.cos(pure_i)),
        (math.cosh(1), 0, 0, 0), rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
        tuple(scalar.to_rotation_vector(scalar.from_rotation_vector((0, 0, math.pi / 2)))),
        (0, 0, math.pi / 2), rtol=0, atol=1e-15,
    )
