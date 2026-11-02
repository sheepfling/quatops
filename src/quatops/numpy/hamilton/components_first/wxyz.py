"""Hamilton NumPy API with leading-axis ``wxyz`` coefficient arrays."""

from ..._component_first import bind_component_first
from .. import wxyz as _backend

_exports = bind_component_first(_backend, __name__)
globals().update(_exports)
__all__ = tuple(_exports)
