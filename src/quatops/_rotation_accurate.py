"""Exact binary64 quaternion rotation for explicitly accurate calls."""

from struct import pack, unpack
from typing import Iterable


def _integer_components(values: Iterable[float]) -> tuple[tuple[int, ...], int]:
    """Express binary64 values over a shared power-of-two denominator."""
    ratios = tuple(value.as_integer_ratio() for value in values)
    exponent = max(denominator.bit_length() - 1 for _, denominator in ratios)
    integers = tuple(
        numerator << (exponent - (denominator.bit_length() - 1))
        for numerator, denominator in ratios
    )
    return integers, exponent


def _rounded_quotient(numerator: int, denominator: int) -> int:
    """Round a nonnegative integer ratio to nearest, ties to even."""
    quotient, remainder = divmod(numerator, denominator)
    twice_remainder = 2 * remainder
    if twice_remainder > denominator or (
        twice_remainder == denominator and quotient & 1
    ):
        quotient += 1
    return quotient


def _round_binary64(numerator: int, denominator: int) -> float:
    """Round an exact ratio to binary64, including subnormals and overflow."""
    if numerator == 0:
        return 0.0
    sign = 1 if numerator < 0 else 0
    numerator = abs(numerator)
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1

    if exponent > 1023:
        raise OverflowError("quaternion rotation produced a non-finite result")
    if exponent < -1022:
        # For subnormals each unit of the significand is 2**-1074.
        bits = _rounded_quotient(numerator << 1074, denominator)
    else:
        shift = 52 - exponent
        if shift >= 0:
            significand = _rounded_quotient(numerator << shift, denominator)
        else:
            significand = _rounded_quotient(numerator, denominator << -shift)
        if significand == 1 << 53:
            significand >>= 1
            exponent += 1
        if exponent > 1023:
            raise OverflowError("quaternion rotation produced a non-finite result")
        bits = ((exponent + 1023) << 52) | (significand - (1 << 52))
    return unpack(">d", pack(">Q", (sign << 63) | bits))[0]


def rotate_accurate(
    quaternion: tuple[float, ...], vector: tuple[float, ...]
) -> tuple[float, float, float]:
    """Rotate using exact Hamilton wxyz inputs and one binary64 rounding."""
    # Each q component has the same power-of-two denominator; it cancels
    # between the quadratic rotation numerator and the squared quaternion norm.
    # The vector's shared denominator remains in the final integer ratio.
    (w, x, y, z), _ = _integer_components(quaternion)
    (vx, vy, vz), vector_exponent = _integer_components(vector)
    denominator = w*w + x*x + y*y + z*z
    if denominator == 0:
        raise ValueError("zero quaternion has no normalization or inverse")
    denominator <<= vector_exponent
    numerators = (
        (w*w + x*x - y*y - z*z)*vx + 2*(x*y - w*z)*vy + 2*(x*z + w*y)*vz,
        2*(x*y + w*z)*vx + (w*w - x*x + y*y - z*z)*vy + 2*(y*z - w*x)*vz,
        2*(x*z - w*y)*vx + 2*(y*z + w*x)*vy + (w*w - x*x - y*y + z*z)*vz,
    )
    return tuple(_round_binary64(value, denominator) for value in numerators)  # type: ignore[return-value]
