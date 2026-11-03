"""Explicit immutable dataclass quaternion types for each convention.

For named tuples and axis-angle or rotation-vector types, import the relevant
convention-specific representation submodule. No NumPy import is required.
"""

from ..hamilton.dataclass import Quaternion as HamiltonQuaternion
from ..jpl.dataclass import Quaternion as JPLQuaternion

__all__ = ["HamiltonQuaternion", "JPLQuaternion"]
