"""Explicit, low-overhead coordinate adapters for game-engine quaternion APIs.

Import ``SignedAxisMap`` to compile a source-to-engine signed-axis mapping.
NumPy array adapters live in ``quatops.engines.numpy`` and are optional.
"""

from .basis import SignedAxisMap

__all__ = ["SignedAxisMap"]
