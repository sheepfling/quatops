"""JPL NumPy API with a leading component axis, shaped ``(4, ...)``.

Each quaternion stores its scalar coefficient last, in ``xyzw`` order.
"""

from ..numpy.jpl.components_first import xyzw as _backend
from ..numpy.jpl.components_first.xyzw import *  # noqa: F403

__all__ = _backend.__all__
