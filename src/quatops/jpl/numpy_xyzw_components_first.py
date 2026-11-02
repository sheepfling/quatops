"""Alias to the JPL NumPy component-first ``xyzw`` API."""

from ..numpy.jpl.components_first import xyzw as _backend
from ..numpy.jpl.components_first.xyzw import *  # noqa: F403

__all__ = _backend.__all__
