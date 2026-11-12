"""Explicit Blender ``mathutils.Quaternion`` boundary conversion helpers.

Blender exposes Hamilton coefficients in WXYZ order. These functions encode
that storage only. Callers must pass a precompiled :class:`QuaternionFrameMap`
for the specific world and body frames; Blender axes and mesh-forward direction
are not inferred here.
"""

from __future__ import annotations

from collections.abc import Iterable

from ._adapter import from_wxyz as _from_wxyz
from ._adapter import to_wxyz as _to_wxyz
from .basis import QuaternionFrameMap


def to_blender(
    q_source_wxyz: Iterable[float], frame_map: QuaternionFrameMap
) -> tuple[float, float, float, float]:
    """Map canonical Hamilton WXYZ into explicitly framed Blender WXYZ."""
    return _to_wxyz(q_source_wxyz, frame_map)


def from_blender(
    q_blender_wxyz: Iterable[float], frame_map: QuaternionFrameMap
) -> tuple[float, float, float, float]:
    """Map explicitly framed Blender Hamilton WXYZ into canonical WXYZ."""
    return _from_wxyz(q_blender_wxyz, frame_map)


__all__ = ["from_blender", "to_blender"]
