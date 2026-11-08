"""Core signed-axis maps; general frame maps live in quatops.experimental."""

from typing import TYPE_CHECKING

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


def __getattr__(name: str):
    if name not in (
        "BasisMap",
        "OrthogonalBasisMap",
        "QuaternionFrameMap",
        "change_rotation_basis",
    ):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import warnings
    from importlib import import_module

    warnings.warn(
        f"{__name__}.{name} is experimental; use quatops.experimental.engines.{name}. "
        "This compatibility path is deprecated and may be removed before 0.1.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(import_module("quatops.experimental.engines.basis"), name)


def __dir__() -> list[str]:
    return sorted(
        set(globals())
        | {
            "OrthogonalBasisMap",
            "QuaternionFrameMap",
            "change_rotation_basis",
            "BasisMap",
        }
    )
