"""Deprecated compatibility path; use ``quatops.experimental.converters.targets`` instead."""

import warnings as _warnings
from importlib import import_module as _import_module

from ..experimental.converters.targets import *  # noqa: F403

_implementation = _import_module("quatops.experimental.converters.targets")
__all__ = list(
    getattr(
        _implementation,
        "__all__",
        [name for name in vars(_implementation) if not name.startswith("_")],
    )
)
_warnings.warn(
    "converters.targets is experimental; use quatops.experimental.converters.targets. "
    "This compatibility path is deprecated and may be removed before 0.1.0.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_implementation)))
