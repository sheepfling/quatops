"""Alias to the JPL NumPy component-first ``wxyz`` API."""

from ..numpy.jpl.components_first import wxyz as _backend
from ..numpy.jpl.components_first.wxyz import *  # noqa: F403

__all__ = _backend.__all__
