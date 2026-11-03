"""Hamilton NumPy API with leading component axis ``(4, ...)``.

Each quaternion stores its scalar coefficient first, in ``wxyz`` order.
"""

from ..._component_first import bind_component_first
from .. import wxyz as _backend

_exports = bind_component_first(_backend, __name__)
globals().update(_exports)
__all__ = tuple(_exports)
