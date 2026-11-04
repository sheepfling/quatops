"""Explicit frame maps for quaternion and rotation-matrix interoperability.

Use ``SignedAxisMap`` for caller-declared signed permutations,
``QuaternionFrameMap`` for separate world/body orientation maps, and
``change_rotation_basis`` for matrix-valued maps. NumPy array adapters live in
``quatops.engines.numpy`` and are optional.
"""

from .basis import (
    OrthogonalBasisMap,
    QuaternionFrameMap,
    SignedAxisMap,
    change_rotation_basis,
)

__all__ = [
    "OrthogonalBasisMap",
    "QuaternionFrameMap",
    "SignedAxisMap",
    "change_rotation_basis",
]
