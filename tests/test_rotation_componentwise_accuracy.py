"""Exact componentwise references for rotation edge cases."""

import math
import random
import struct
from fractions import Fraction
from importlib import import_module

import numpy as np
import pytest


def exact_rotate(q, v):
    """Return correctly rounded components for the exact binary64 inputs."""
    w, x, y, z = (Fraction.from_float(float(component)) for component in q)
    vx, vy, vz = (Fraction.from_float(float(component)) for component in v)
    denominator = w * w + x * x + y * y + z * z

    def multiply(lhs, rhs):
        a, b, c, d = lhs
        e, f, g, h = rhs
        return (
            a*e - b*f - c*g - d*h,
            a*f + b*e + c*h - d*g,
            a*g - b*h + c*e + d*f,
            a*h + b*g - c*f + d*e,
        )

    product = multiply(multiply((w, x, y, z), (0, vx, vy, vz)), (w, -x, -y, -z))
    return tuple(float(component / denominator) for component in product[1:])


BACKENDS = (
    ("Hamilton scalar", "quatops.scalar", False, False, False, "wxyz"),
    ("JPL scalar", "quatops.jpl.scalar", True, False, False, "xyzw"),
    *(
        (
            f"{convention} NumPy {'leading' if axis else 'trailing'} {order}",
            f"quatops.numpy.{convention}.{axis}{order}",
            convention == "jpl",
            True,
            axis == "components_first.",
            order,
        )
        for convention in ("hamilton", "jpl")
        for axis in ("", "components_first.")
        for order in ("wxyz", "xyzw")
    ),
)


def backend_quaternion(q_hamilton, *, is_jpl, order):
    w, x, y, z = q_hamilton
    if is_jpl:
        return (-x, -y, -z, w) if order == "xyzw" else (w, -x, -y, -z)
    return (w, x, y, z) if order == "wxyz" else (x, y, z, w)


def rotate_on_backend(backend, q_hamilton, vector):
    _, module_name, is_jpl, is_numpy, component_first, order = backend
    module = import_module(module_name)
    quaternion = backend_quaternion(q_hamilton, is_jpl=is_jpl, order=order)
    if not is_numpy:
        return tuple(module.rotate(quaternion, vector, accurate=True))

    if component_first:
        quaternion = np.asarray(quaternion)[:, None]
        vector = np.asarray(vector)[:, None]
        result = module.rotate(quaternion, vector, accurate=True)
        return tuple(np.moveaxis(result, 0, -1).reshape(3))

    quaternion = np.asarray(quaternion)[None, :]
    vector = np.asarray(vector)[None, :]
    return tuple(module.rotate(quaternion, vector, accurate=True).reshape(3))


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend[0])
def test_pythagorean_quaternion_cancellation_matches_exact_components(backend):
    """The first component is exactly 17/13, not a large cancellation remnant."""
    quaternion = (5.0, 12.0, 13.0, 0.0)
    vector = (1e308, 1.0, 1.0)

    actual = rotate_on_backend(backend, quaternion, vector)
    expected = exact_rotate(quaternion, vector)

    assert expected[0] == 17.0 / 13.0
    assert actual == expected


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend[0])
def test_near_identity_subnormal_quaternion_component_survives(backend):
    """A subnormal quaternion component still contributes after a large vector."""
    smallest_subnormal = np.nextafter(0.0, 1.0)
    quaternion = (1.0, smallest_subnormal, 0.0, 0.0)
    vector = (0.0, 1e308, 0.0)

    actual = rotate_on_backend(backend, quaternion, vector)
    expected = exact_rotate(quaternion, vector)

    assert expected[2] == 9.881312916824931e-16
    assert actual == expected


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend[0])
def test_near_diagonal_subnormal_output_survives(backend):
    quaternion = (1.0, 1.0, 1e-8, 0.0)
    vector = (0.0, 1e-300, 0.0)

    actual = rotate_on_backend(backend, quaternion, vector)
    expected = exact_rotate(quaternion, vector)

    assert expected[1] == 5e-317
    assert actual == expected


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend[0])
def test_tiny_quadratic_term_survives_when_final_component_is_representable(backend):
    """A q-component square can underflow before its vector contribution does."""
    quaternion = (1.0, 1.0, 1e-200, 0.0)
    vector = (0.0, 1e308, 0.0)

    actual = rotate_on_backend(backend, quaternion, vector)
    expected = exact_rotate(quaternion, vector)

    assert expected[1] == 5e-93
    assert actual == expected


@pytest.mark.parametrize(
    ("module_name", "is_jpl"),
    (("quatops.namedtuple", False), ("quatops.dataclass", False),
     ("quatops.jpl.namedtuple", True), ("quatops.jpl.dataclass", True)),
)
def test_accurate_rotation_is_available_on_typed_backends(module_name, is_jpl):
    quaternion = (5.0, 12.0, 13.0, 0.0)
    vector = (1e308, 1.0, 1.0)
    native = backend_quaternion(quaternion, is_jpl=is_jpl, order="xyzw" if is_jpl else "wxyz")
    assert import_module(module_name).rotate(native, vector, accurate=True) == exact_rotate(
        quaternion, vector
    )


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend[0])
@pytest.mark.parametrize(
    ("quaternion", "vector"),
    (
        ((1.0, 0.0, 0.0, 0.0), (1e308, 1e-308, 5e-324)),
        ((1.0, 0.0, 0.0, 1.0), (1e308, 1e-308, 5e-324)),
        ((1e308, 0.0, 0.0, 1e-308), (0.0, 1e308, 0.0)),
        ((5e-324, 5e-324, 0.0, 0.0), (1e308, 1e-308, 0.0)),
        ((1e-300, 2e-300, -1e-300, 3e-300), (5e-324, 0.0, 0.0)),
        ((-2.0, -1.0, 0.0, 1.0), (1e308, -1e308, 3e-308)),
        (
            (4576996.0, 641332161.0, 462136015.0, 0.0),
            (0.0, math.ldexp(4576996.0, 950), -math.ldexp(641332161.0, 950)),
        ),
        ((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 3.0)),
    ),
    ids=("identity", "permutation", "tiny-ratio", "tiny-quaternion", "subnormal-vector",
         "cancellation", "dot-cancellation", "ordinary"),
)
def test_accurate_rotation_matches_fraction_reference(backend, quaternion, vector):
    actual = rotate_on_backend(backend, quaternion, vector)
    expected = exact_rotate(quaternion, vector)
    assert actual == expected
    for observed, reference in zip(actual, expected):
        assert math.copysign(1.0, observed) == math.copysign(1.0, reference)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend[0])
def test_accurate_rotation_uses_converted_float32_inputs(backend):
    quaternion = tuple(np.float32(value) for value in (0.1, -0.2, 0.3, 0.4))
    vector = tuple(np.float32(value) for value in (0.5, -1.0, 0.25))
    assert rotate_on_backend(backend, quaternion, vector) == exact_rotate(quaternion, vector)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend[0])
def test_accurate_rotation_reports_true_overflow(backend):
    quaternion = (-2.0, -1.0, 0.0, 1.0)
    vector = (1.7e308, -1.7e308, -1.7e308)
    with pytest.raises(OverflowError):
        exact_rotate(quaternion, vector)
    with pytest.raises(OverflowError):
        rotate_on_backend(backend, quaternion, vector)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend[0])
def test_accurate_rotation_rejects_zero_quaternion(backend):
    with pytest.raises(ValueError):
        rotate_on_backend(backend, (0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0))


@pytest.mark.parametrize("backend", BACKENDS[2:], ids=lambda backend: backend[0])
def test_accurate_numpy_broadcasts_without_mutating_inputs(backend, monkeypatch):
    from quatops import _numpy

    def unexpected_numba():
        raise AssertionError("accurate rotation must not dispatch to Numba")

    monkeypatch.setattr(_numpy, "_numba_rotate_kernel", unexpected_numba)
    _, module_name, is_jpl, _, component_first, order = backend
    module = import_module(module_name)
    physical = ((5.0, 12.0, 13.0, 0.0), (1.0, 1.0, 1e-200, 0.0))
    quaternions = np.array([
        backend_quaternion(q, is_jpl=is_jpl, order=order) for q in physical
    ])[:, None, :]
    vectors = np.array([[(1e308, 1.0, 1.0), (0.0, 1e308, 0.0)]])
    # Exceed the usual Numba threshold while retaining nontrivial broadcast axes.
    vectors = np.tile(vectors, (1, 64, 1))
    if component_first:
        quaternions = np.moveaxis(quaternions, -1, 0)
        vectors = np.moveaxis(vectors, -1, 0)
    original_q, original_v = quaternions.copy(), vectors.copy()
    quaternions.flags.writeable = False
    vectors.flags.writeable = False
    previous = np.seterr(all="raise")
    try:
        result = module.rotate(quaternions, vectors, accurate=True)
        assert all(value == "raise" for value in np.geterr().values())
    finally:
        np.seterr(**previous)
    np.testing.assert_array_equal(quaternions, original_q)
    np.testing.assert_array_equal(vectors, original_v)
    if component_first:
        result = np.moveaxis(result, 0, -1)
    assert result.shape == (2, 128, 3)
    inputs = ((1e308, 1.0, 1.0), (0.0, 1e308, 0.0))
    for i, quaternion in enumerate(physical):
        expected = [exact_rotate(quaternion, vector) for vector in inputs]
        for j in range(128):
            assert tuple(result[i, j]) == expected[j % 2]


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend[0])
def test_accurate_flag_and_input_validation(backend):
    _, module_name, _, _, _, _ = backend
    module = import_module(module_name)
    quaternion = (1.0, 0.0, 0.0, 0.0)
    vector = (1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="accurate must be a bool"):
        module.rotate(quaternion, vector, accurate=1)
    with pytest.raises(ValueError):
        module.rotate((float("nan"), 0.0, 0.0, 0.0), vector, accurate=True)
    with pytest.raises(ValueError):
        module.rotate(quaternion, (1.0, 2.0), accurate=True)


@pytest.mark.parametrize("backend", BACKENDS[2:], ids=lambda backend: backend[0])
def test_accurate_numpy_empty_and_invalid_broadcasts(backend):
    _, module_name, _, _, component_first, _ = backend
    module = import_module(module_name)
    if component_first:
        empty_q, vectors = np.empty((4, 0, 1)), np.zeros((3, 1, 3))
        bad_q, bad_v = np.ones((4, 2)), np.ones((3, 3))
        expected_shape = (3, 0, 3)
    else:
        empty_q, vectors = np.empty((0, 1, 4)), np.zeros((1, 3, 3))
        bad_q, bad_v = np.ones((2, 4)), np.ones((3, 3))
        expected_shape = (0, 3, 3)
    assert module.rotate(empty_q, vectors, accurate=True).shape == expected_shape
    with pytest.raises(ValueError):
        module.rotate(bad_q, bad_v, accurate=True)


def test_accurate_rotation_matches_fraction_for_random_binary64_inputs():
    rng = random.Random(622761)

    def finite_float():
        while True:
            value = struct.unpack(">d", rng.getrandbits(64).to_bytes(8, "big"))[0]
            if math.isfinite(value):
                return value

    # Both scalar conventions and two array layouts cover independent dispatch paths.
    backends = (BACKENDS[0], BACKENDS[1], BACKENDS[2], BACKENDS[-1])
    for _ in range(200):
        quaternion = tuple(finite_float() for _ in range(4))
        vector = tuple(finite_float() for _ in range(3))
        try:
            expected = exact_rotate(quaternion, vector)
        except OverflowError:
            for backend in backends:
                with pytest.raises(OverflowError):
                    rotate_on_backend(backend, quaternion, vector)
        else:
            for backend in backends:
                actual = rotate_on_backend(backend, quaternion, vector)
                assert actual == expected
                for observed, reference in zip(actual, expected):
                    assert math.copysign(1.0, observed) == math.copysign(1.0, reference)


def test_exact_ratio_rounding_matches_fraction_at_binary64_boundaries():
    from quatops._rotation_accurate import _round_binary64

    cases = (
        (1, 1 << 1075), (3, 1 << 1075), (-1, 1 << 1075),
        ((1 << 53) + 1, 1 << 53), ((1 << 53) + 3, 1 << 53),
        (((1 << 53) - 1) << 971, 1),
        ((((1 << 53) - 1) << 971) + (1 << 970) - 1, 1),
    )
    for numerator, denominator in cases:
        expected = float(Fraction(numerator, denominator))
        actual = _round_binary64(numerator, denominator)
        assert actual == expected
        assert math.copysign(1.0, actual) == math.copysign(1.0, expected)
    with pytest.raises(OverflowError):
        _round_binary64((((1 << 53) - 1) << 971) + (1 << 970), 1)


def test_exact_ratio_rounding_matches_random_fractions():
    from quatops._rotation_accurate import _round_binary64

    rng = random.Random(243906)
    for _ in range(1000):
        numerator = rng.getrandbits(rng.randrange(1, 2200))
        denominator = rng.getrandbits(rng.randrange(1, 2200)) | 1
        if rng.randrange(2):
            numerator = -numerator
        try:
            expected = float(Fraction(numerator, denominator))
        except OverflowError:
            with pytest.raises(OverflowError):
                _round_binary64(numerator, denominator)
        else:
            actual = _round_binary64(numerator, denominator)
            assert actual == expected
            assert math.copysign(1.0, actual) == math.copysign(1.0, expected)
