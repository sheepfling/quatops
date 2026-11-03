"""Zero-adapter exports for Hamilton NamedTuple operations."""

from .. import namedtuple as _implementation
from ..namedtuple import *  # noqa: F403

__all__ = list(_implementation.__all__)
