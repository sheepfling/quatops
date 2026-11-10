"""Pure quaternion coefficient-layout operations without frame conversion."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Literal

ComponentOrder = Literal["wxyz", "xyzw"]


def reorder_quaternion_components(
    values: Sequence[float],
    *,
    source_order: ComponentOrder,
    target_order: ComponentOrder,
) -> tuple[float, float, float, float]:
    """Reorder four coefficients without changing algebra or frame meaning.

    This operation is purely representational. It does not normalize, change
    Hamilton/JPL algebra, invert an orientation, or map coordinate frames.
    """
    if source_order not in ("wxyz", "xyzw") or target_order not in (
        "wxyz",
        "xyzw",
    ):
        raise ValueError("component orders must be 'wxyz' or 'xyzw'")
    quaternion = tuple(float(value) for value in values)
    if len(quaternion) != 4:
        raise ValueError("a quaternion must contain exactly four coefficients")
    if any(not isfinite(value) for value in quaternion):
        raise ValueError("quaternion coefficients must be finite")
    if source_order == target_order:
        return quaternion  # type: ignore[return-value]
    if source_order == "wxyz":
        w, x, y, z = quaternion
        return x, y, z, w
    x, y, z, w = quaternion
    return w, x, y, z
