"""Shared storage/frame plumbing for explicitly configured engine adapters."""

from __future__ import annotations

from collections.abc import Iterable

from ...python import hamilton
from .basis import QuaternionFrameMap


def to_xyzw(
    q_source_wxyz: Iterable[float], frame_map: QuaternionFrameMap
) -> tuple[float, float, float, float]:
    """Convert canonical Hamilton WXYZ to a mapped Hamilton XYZW boundary."""
    if not isinstance(frame_map, QuaternionFrameMap):
        raise TypeError("frame_map must be a precompiled QuaternionFrameMap")
    destination_wxyz = frame_map.to_destination_wxyz(hamilton.from_wxyz(q_source_wxyz))
    return hamilton.to_xyzw(destination_wxyz)


def from_xyzw(
    q_destination_xyzw: Iterable[float], frame_map: QuaternionFrameMap
) -> tuple[float, float, float, float]:
    """Convert mapped Hamilton XYZW boundary values to canonical WXYZ."""
    if not isinstance(frame_map, QuaternionFrameMap):
        raise TypeError("frame_map must be a precompiled QuaternionFrameMap")
    destination_wxyz = hamilton.from_xyzw(q_destination_xyzw)
    return frame_map.from_destination_wxyz(destination_wxyz)


def to_wxyz(
    q_source_wxyz: Iterable[float], frame_map: QuaternionFrameMap
) -> tuple[float, float, float, float]:
    """Convert canonical Hamilton WXYZ to a mapped Hamilton WXYZ boundary."""
    if not isinstance(frame_map, QuaternionFrameMap):
        raise TypeError("frame_map must be a precompiled QuaternionFrameMap")
    return frame_map.to_destination_wxyz(hamilton.from_wxyz(q_source_wxyz))


def from_wxyz(
    q_destination_wxyz: Iterable[float], frame_map: QuaternionFrameMap
) -> tuple[float, float, float, float]:
    """Convert mapped Hamilton WXYZ boundary values to canonical WXYZ."""
    if not isinstance(frame_map, QuaternionFrameMap):
        raise TypeError("frame_map must be a precompiled QuaternionFrameMap")
    return frame_map.from_destination_wxyz(hamilton.from_wxyz(q_destination_wxyz))
