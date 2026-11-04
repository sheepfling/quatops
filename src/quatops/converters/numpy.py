"""Optional NumPy batch conversion through declared target profiles."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from ..engines import numpy as _engine_numpy
from ..engines.basis import SignedAxisMap
from ..numpy.hamilton import wxyz as _hamilton
from ..python import hamilton as _scalar_hamilton
from .frames import FramedOrientationBatch, SemanticQuaternionMap
from .targets import (
    ConverterProfile,
    TargetOrientationBatch,
)


def _component_array(
    values: Any, axis: int, length: int, name: str
) -> tuple[np.ndarray, int]:
    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError(f"{name} must have a quaternion or rotation-vector axis")
    try:
        normalized_axis = operator.index(axis)
    except TypeError as exc:
        raise TypeError("component_axis must be an integer") from exc
    if normalized_axis < 0:
        normalized_axis += array.ndim
    if not 0 <= normalized_axis < array.ndim:
        raise ValueError("component_axis is out of range for the input")
    moved = np.moveaxis(array, normalized_axis, -1)
    if moved.shape[-1] != length:
        raise ValueError(f"component axis must have length {length}")
    return moved, normalized_axis


def _to_destination_wxyz(
    values_wxyz: np.ndarray, frame_map: SemanticQuaternionMap
) -> np.ndarray:
    compiled = frame_map._compiled
    shared_map = compiled._shared_map
    if isinstance(shared_map, SignedAxisMap):
        xyzw = _engine_numpy.map_hamilton_wxyz(values_wxyz, shared_map)
        return xyzw[..., (3, 0, 1, 2)]
    after_world = _hamilton.multiply(
        np.asarray(compiled.world_correction_wxyz), values_wxyz
    )
    return _hamilton.multiply(
        after_world,
        np.asarray(_scalar_hamilton.inverse(compiled.body_correction_wxyz)),
    )


def _from_destination_wxyz(
    values_wxyz: np.ndarray, frame_map: SemanticQuaternionMap
) -> np.ndarray:
    compiled = frame_map._compiled
    shared_map = compiled._shared_map
    if isinstance(shared_map, SignedAxisMap):
        xyzw = _engine_numpy.map_hamilton_wxyz(values_wxyz, shared_map.inverse())
        return xyzw[..., (3, 0, 1, 2)]
    before_body = _hamilton.multiply(
        np.asarray(_scalar_hamilton.inverse(compiled.world_correction_wxyz)),
        values_wxyz,
    )
    return _hamilton.multiply(before_body, np.asarray(compiled.body_correction_wxyz))


def to_target_array(
    profile: ConverterProfile,
    orientations: FramedOrientationBatch,
    frame_map: SemanticQuaternionMap,
    component_axis: int = -1,
) -> TargetOrientationBatch:
    """Convert a canonical WXYZ batch to a target's declared representation.

    The source batch carries its parent/child frame IDs. Signed-axis maps keep
    the input dtype; general world/body corrections and rotation-vector
    conversion use the float64 Hamilton NumPy kernels.
    """
    if not isinstance(orientations, FramedOrientationBatch):
        raise TypeError("orientations must be a FramedOrientationBatch")
    frame_map.validate_source_labels(orientations.from_frame, orientations.to_frame)
    moved, axis = _component_array(
        orientations.values, component_axis, 4, "orientations"
    )
    mapped = _to_destination_wxyz(moved, frame_map)
    if profile.representation == "rotation_vector":
        output = _hamilton.to_rotation_vector(mapped)
    elif profile.component_order == "xyzw":
        output = mapped[..., (1, 2, 3, 0)]
    else:
        output = mapped
    values = np.moveaxis(output, -1, axis)
    return TargetOrientationBatch(
        converter_id=profile.converter_id,
        values=values,
        representation=profile.representation,
        component_order=profile.component_order,
        from_frame=frame_map.body.destination.name,
        to_frame=frame_map.world.destination.name,
    )


def from_target_array(
    profile: ConverterProfile,
    orientations: TargetOrientationBatch,
    frame_map: SemanticQuaternionMap,
    component_axis: int = -1,
) -> FramedOrientationBatch:
    """Convert a target batch to canonical active Hamilton WXYZ values."""
    if not isinstance(orientations, TargetOrientationBatch):
        raise TypeError("orientations must be a TargetOrientationBatch")
    if orientations.converter_id != profile.converter_id:
        raise ValueError("target batch belongs to a different converter")
    if (
        orientations.representation != profile.representation
        or orientations.component_order != profile.component_order
    ):
        raise ValueError("target batch layout does not match this converter")
    frame_map.validate_destination_labels(
        orientations.from_frame, orientations.to_frame
    )

    component_count = 3 if profile.representation == "rotation_vector" else 4
    moved, axis = _component_array(
        orientations.values, component_axis, component_count, "orientations"
    )
    if profile.representation == "rotation_vector":
        destination_wxyz = _hamilton.from_rotation_vector(moved)
    elif profile.component_order == "xyzw":
        destination_wxyz = moved[..., (3, 0, 1, 2)]
    else:
        destination_wxyz = moved
    source_wxyz = _from_destination_wxyz(destination_wxyz, frame_map)
    return FramedOrientationBatch(
        values=np.moveaxis(source_wxyz, -1, axis),
        from_frame=frame_map.body.source.name,
        to_frame=frame_map.world.source.name,
    )
