"""Alias to the Hamilton NumPy component-first ``wxyz`` API."""

from ..numpy.hamilton.components_first import wxyz as _backend
from ..numpy.hamilton.components_first.wxyz import *  # noqa: F403

__all__ = _backend.__all__
