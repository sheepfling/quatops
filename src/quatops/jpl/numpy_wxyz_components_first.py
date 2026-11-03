"""JPL NumPy API with a leading component axis, shaped ``(4, ...)``.

Each quaternion stores its scalar coefficient first, in ``wxyz`` order.
"""

from ..numpy.jpl.components_first import wxyz as _backend
from ..numpy.jpl.components_first.wxyz import *  # noqa: F403

__all__ = _backend.__all__
