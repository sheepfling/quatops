"""Alias to the Hamilton NumPy component-first ``xyzw`` API."""

from ..numpy.hamilton.components_first import xyzw as _backend
from ..numpy.hamilton.components_first.xyzw import *  # noqa: F403

__all__ = _backend.__all__
