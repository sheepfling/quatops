"""Zero-adapter exports for Hamilton dataclass operations."""

from .. import dataclass as _implementation
from ..dataclass import *  # noqa: F403

__all__ = list(_implementation.__all__)
