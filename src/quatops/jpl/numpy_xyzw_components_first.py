"""JPL NumPy API with a leading component axis, shaped ``(4, ...)``.

Each quaternion stores its scalar coefficient last, in ``xyzw`` order.
"""

from ..numpy.jpl.components_first import xyzw as _backend
from ..numpy.jpl.components_first.xyzw import *  # noqa: F403

# The component-first backend builds this export tuple from its adapter table.
__all__ = [*vars(_backend)["__all__"]]  # noqa: PLE0604
