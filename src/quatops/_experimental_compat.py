"""Compatibility hooks for experimental APIs formerly exposed as core APIs."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from importlib import import_module
from typing import Any

_ENGINE_EXPORTS = frozenset(
    {"BasisMap", "OrthogonalBasisMap", "QuaternionFrameMap", "change_rotation_basis"}
)


def warn_compatibility(old_module: str, target_module: str) -> None:
    """Warn that a former import path has an experimental replacement."""
    warnings.warn(
        f"{old_module} is experimental; use {target_module}. "
        "This compatibility path is deprecated and may be removed before 0.1.0.",
        DeprecationWarning,
        stacklevel=3,
    )


def resolve_attribute(target_module: str, name: str) -> Any:
    """Forward attribute lookup without creating duplicate API objects."""
    return getattr(import_module(target_module), name)


def list_attributes(namespace: Mapping[str, object], target_module: str) -> list[str]:
    """Include forwarded attributes in compatibility-module discovery."""
    return sorted(set(namespace) | set(dir(import_module(target_module))))


def resolve_engine_attribute(old_module: str, name: str) -> Any:
    """Load experimental frame maps only when a caller explicitly asks for one."""
    if name not in _ENGINE_EXPORTS:
        raise AttributeError(f"module {old_module!r} has no attribute {name!r}")
    warn_compatibility(f"{old_module}.{name}", f"quatops.experimental.engines.{name}")
    return resolve_attribute("quatops.experimental.engines.basis", name)


def list_engine_attributes(namespace: Mapping[str, object]) -> list[str]:
    """Advertise compatibility names without importing the experimental package."""
    return sorted(set(namespace) | _ENGINE_EXPORTS)
