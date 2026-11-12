"""Explicit Godot ``Quaternion`` boundary conversion helpers.

These functions encode Hamilton XYZW component storage only. Callers must pass
a precompiled :class:`QuaternionFrameMap` whose world and body mappings describe
the particular frames being converted; Godot axes are not inferred here.
"""

from __future__ import annotations

from collections.abc import Iterable

from ._adapter import from_xyzw as _from_xyzw
from ._adapter import to_xyzw as _to_xyzw
from .basis import QuaternionFrameMap


def to_godot(
    q_source_wxyz: Iterable[float], frame_map: QuaternionFrameMap
) -> tuple[float, float, float, float]:
    """Map canonical Hamilton WXYZ into explicitly framed Godot XYZW."""
    return _to_xyzw(q_source_wxyz, frame_map)


def from_godot(
    q_godot_xyzw: Iterable[float], frame_map: QuaternionFrameMap
) -> tuple[float, float, float, float]:
    """Map explicitly framed Godot Hamilton XYZW into canonical WXYZ."""
    return _from_xyzw(q_godot_xyzw, frame_map)


__all__ = ["from_godot", "to_godot"]
