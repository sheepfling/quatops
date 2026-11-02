"""JPL NumPy API with leading component axis ``(4, ...)``.

Each quaternion stores its scalar coefficient last, in ``xyzw`` order.
"""

from ..._component_first import bind_component_first
from .. import xyzw as _backend

_exports = bind_component_first(_backend, __name__)
globals().update(_exports)
__all__ = tuple(_exports)
