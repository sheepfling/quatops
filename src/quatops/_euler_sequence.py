"""Precomputed metadata for supported Euler axis sequences."""

from __future__ import annotations

from itertools import permutations


def _make_metadata() -> dict[str, tuple[bool, int, int, int, int, int, bool]]:
    sequences = (*("".join(axes) for axes in permutations("xyz")),
                 *(a + b + a for a in "xyz" for b in "xyz" if a != b))
    metadata = {}
    for axes in sequences:
        for intrinsic in (False, True):
            sequence = axes.upper() if intrinsic else axes
            kernel_axes = axes if intrinsic else axes[::-1]
            i, j, k = ("xyz".index(axis) + 1 for axis in kernel_axes)
            proper = i == k
            ell = 6 - i - j if proper else k
            epsilon = ((i - j) * (j - ell) * (ell - i)) // 2
            metadata[sequence] = intrinsic, i, j, k, ell, epsilon, proper
    return metadata


_METADATA = _make_metadata()


def euler_metadata(sequence: str) -> tuple[bool, int, int, int, int, int, bool]:
    """Look up a validated intrinsic or extrinsic Euler sequence."""
    if not isinstance(sequence, str):
        raise ValueError("sequence must be Tait-Bryan or proper Euler")  # noqa: TRY004
    try:
        return _METADATA[sequence]
    except KeyError:
        raise ValueError("sequence must be Tait-Bryan or proper Euler") from None
