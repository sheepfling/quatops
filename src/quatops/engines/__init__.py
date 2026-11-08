"""Core signed-axis maps; general frame maps live in quatops.experimental."""

from functools import partial as _partial
from typing import TYPE_CHECKING

from .._experimental_compat import (
    list_engine_attributes as _list_engine_attributes,
)
from .._experimental_compat import (
    resolve_engine_attribute as _resolve_engine_attribute,
)
from .basis import SignedAxisMap

__all__ = ["SignedAxisMap"]

if TYPE_CHECKING:
    from ..experimental.engines.basis import (
        BasisMap as BasisMap,
    )
    from ..experimental.engines.basis import (
        OrthogonalBasisMap as OrthogonalBasisMap,
    )
    from ..experimental.engines.basis import (
        QuaternionFrameMap as QuaternionFrameMap,
    )
    from ..experimental.engines.basis import (
        change_rotation_basis as change_rotation_basis,
    )


__getattr__ = _partial(_resolve_engine_attribute, __name__)
__dir__ = _partial(_list_engine_attributes, globals())
