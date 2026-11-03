"""Zero-adapter exports for Hamilton operations on scalar tuples."""

from .. import scalar as _implementation
from ..scalar import *  # noqa: F403

__all__ = list(_implementation.__all__)
