"""Frame-labelled active Hamilton orientations and their semantic maps.

An orientation is written ``q_parent_from_child`` and acts as
``v_parent = R(q) @ v_child``. Coefficients use canonical Hamilton WXYZ order.
Frame axes and their bases in named references are explicit data;
source-to-destination maps are derived from those bases, and target names
never supply axis defaults.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

from ..engines.basis import (
    BasisMap,
    OrthogonalBasisMap,
    QuaternionFrameMap,
    SignedAxisMap,
)
from ..python import hamilton as _hamilton
from ..types import HamiltonQuaternion

Handedness = Literal["right", "left"]


def _checked_wxyz(values: Iterable[float]) -> tuple[float, float, float, float]:
    components = tuple(float(value) for value in values)
    if len(components) != 4:
        raise ValueError("quaternion_wxyz must contain exactly four components")
    w, x, y, z = components
    if any(not isfinite(value) for value in (w, x, y, z)):
        raise ValueError("quaternion components must be finite")
    return w, x, y, z


@dataclass(frozen=True)
class CoordinateFrame:
    """A named frame whose basis vectors are declared in a named reference.

    The columns of ``basis_to_reference`` are this frame's positive X/Y/Z axes
    expressed in the right-handed reference frame. Axis labels are descriptive;
    the numeric basis is the conversion source of truth.
    """

    name: str
    reference: str
    axes: tuple[str, str, str]
    basis_to_reference: BasisMap
    handedness: Handedness = "right"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("frame name must be a non-empty string")
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise ValueError("reference must be a non-empty frame name")
        if len(self.axes) != 3 or any(
            not isinstance(axis, str) or not axis.strip() for axis in self.axes
        ):
            raise ValueError("axes must name the positive X, Y, and Z directions")
        if len({axis.strip().casefold() for axis in self.axes}) != 3:
            raise ValueError("axis meanings must be distinct")
        if self.handedness not in ("right", "left"):
            raise ValueError("handedness must be 'right' or 'left'")
        if not isinstance(self.basis_to_reference, (SignedAxisMap, OrthogonalBasisMap)):
            raise TypeError(
                "basis_to_reference must be a SignedAxisMap or OrthogonalBasisMap"
            )
        expected_parity = 1 if self.handedness == "right" else -1
        if self.basis_to_reference.determinant != expected_parity:
            raise ValueError("frame handedness does not match basis_to_reference")


_IDENTITY_BASIS = SignedAxisMap(("x", "y", "z"))

ECEF_FRAME = CoordinateFrame(
    name="quatops.ecef",
    reference="quatops.ecef",
    axes=(
        "+X equator at prime meridian",
        "+Y equator at 90 degrees east",
        "+Z north pole",
    ),
    basis_to_reference=_IDENTITY_BASIS,
    handedness="right",
)
"""QuatOps' right-handed Earth-Centered, Earth-Fixed orientation frame."""

CANONICAL_BODY_FRAME = CoordinateFrame(
    name="quatops.body.right-forward-up",
    reference="quatops.body.right-forward-up",
    axes=("right", "forward", "up"),
    basis_to_reference=_IDENTITY_BASIS,
    handedness="right",
)
"""QuatOps' explicit right-handed body frame: +X right, +Y forward, +Z up."""


@dataclass(frozen=True)
class FrameCorrespondence:
    """A coordinate map derived from two frames in the same reference."""

    source: CoordinateFrame
    destination: CoordinateFrame
    coordinates: BasisMap = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, CoordinateFrame):
            raise TypeError("source must be a CoordinateFrame")
        if not isinstance(self.destination, CoordinateFrame):
            raise TypeError("destination must be a CoordinateFrame")
        if self.source.reference != self.destination.reference:
            raise ValueError("mapped frames must share the same declared reference")
        source = self.source.basis_to_reference.matrix
        destination = self.destination.basis_to_reference.matrix
        matrix = tuple(
            tuple(
                sum(destination[k][row] * source[k][column] for k in range(3))
                for column in range(3)
            )
            for row in range(3)
        )
        coordinate_map = _basis_map_from_matrix(matrix)
        expected_parity = (1 if self.source.handedness == "right" else -1) * (
            1 if self.destination.handedness == "right" else -1
        )
        if coordinate_map.determinant != expected_parity:
            raise ValueError("derived coordinate map conflicts with frame handedness")
        object.__setattr__(self, "coordinates", coordinate_map)


def _basis_map_from_matrix(matrix: tuple[tuple[float, ...], ...]) -> BasisMap:
    """Use the signed-permutation fast path when the derived map permits it."""
    tolerance = 1e-12
    axes: list[str] = []
    for row in matrix:
        nonzero = [index for index, value in enumerate(row) if abs(value) > tolerance]
        if len(nonzero) != 1 or abs(abs(row[nonzero[0]]) - 1.0) > tolerance:
            return OrthogonalBasisMap(matrix)
        index = nonzero[0]
        axes.append(("-" if row[index] < 0.0 else "") + "xyz"[index])
    return SignedAxisMap(axes)


@dataclass(frozen=True)
class FramedOrientation:
    """An active Hamilton WXYZ orientation labelled by its frame pair.

    ``to_frame`` is the parent/reference frame and ``from_frame`` is the
    child/local frame. Construction preserves the supplied magnitude and
    quaternion sign; it does not normalize or canonicalize values.
    """

    quaternion_wxyz: tuple[float, float, float, float]
    from_frame: str
    to_frame: str

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.quaternion_wxyz)
        if len(values) != 4:
            raise ValueError("quaternion_wxyz must contain exactly four components")
        if any(not isfinite(value) for value in values):
            raise ValueError("quaternion components must be finite")
        if not isinstance(self.from_frame, str) or not self.from_frame.strip():
            raise ValueError("from_frame must be a non-empty frame name")
        if not isinstance(self.to_frame, str) or not self.to_frame.strip():
            raise ValueError("to_frame must be a non-empty frame name")
        object.__setattr__(self, "quaternion_wxyz", values)

    def compose(self, right: FramedOrientation) -> FramedOrientation:
        """Compose ``self * right``; the right-hand orientation acts first."""
        if not isinstance(right, FramedOrientation):
            raise TypeError("right must be a FramedOrientation")
        if right.to_frame != self.from_frame:
            raise ValueError(
                f"cannot compose {self.to_frame}<-{self.from_frame} with "
                f"{right.to_frame}<-{right.from_frame}"
            )
        return FramedOrientation(
            _hamilton.multiply(self.quaternion_wxyz, right.quaternion_wxyz),
            from_frame=right.from_frame,
            to_frame=self.to_frame,
        )

    def inverse(self) -> FramedOrientation:
        """Reverse the frame direction and invert the quaternion value."""
        return FramedOrientation(
            _hamilton.inverse(self.quaternion_wxyz),
            from_frame=self.to_frame,
            to_frame=self.from_frame,
        )


@dataclass(frozen=True)
class ECEFBodyOrientation:
    """A standard Hamilton orientation ``q_ECEF_from_body``.

    The child/body axes use :data:`CANONICAL_BODY_FRAME` (+X right, +Y
    forward, +Z up); the parent axes use :data:`ECEF_FRAME`. This semantic
    value stores the ordinary Hamilton WXYZ dataclass quaternion and can be
    passed to a target converter after converting it to a labelled
    :class:`FramedOrientation`.
    """

    quaternion: HamiltonQuaternion

    def __post_init__(self) -> None:
        if not isinstance(self.quaternion, HamiltonQuaternion):
            raise TypeError("quaternion must be a HamiltonQuaternion")
        values = _checked_wxyz(
            (self.quaternion.w, self.quaternion.x, self.quaternion.y, self.quaternion.z)
        )
        object.__setattr__(self, "quaternion", HamiltonQuaternion(*values))

    @classmethod
    def from_wxyz(cls, values: Iterable[float]) -> ECEFBodyOrientation:
        """Construct ``q_ECEF_from_body`` from Hamilton WXYZ coefficients."""
        checked = _checked_wxyz(values)
        return cls(HamiltonQuaternion(*checked))

    @property
    def quaternion_wxyz(self) -> tuple[float, float, float, float]:
        """Return the stored Hamilton coefficients in canonical WXYZ order."""
        w, x, y, z = self.quaternion
        return w, x, y, z

    def to_framed_hamilton(self) -> FramedOrientation:
        """Expose the quaternion with its explicit ECEF/body frame labels."""
        return FramedOrientation(
            self.quaternion_wxyz,
            from_frame=CANONICAL_BODY_FRAME.name,
            to_frame=ECEF_FRAME.name,
        )

    @classmethod
    def from_framed_hamilton(
        cls, orientation: FramedOrientation
    ) -> ECEFBodyOrientation:
        """Accept a framed Hamilton value only when its labels match this type."""
        if not isinstance(orientation, FramedOrientation):
            raise TypeError("orientation must be a FramedOrientation")
        if orientation.from_frame != CANONICAL_BODY_FRAME.name:
            raise ValueError(
                "child frame must be "
                f"{CANONICAL_BODY_FRAME.name!r}, got {orientation.from_frame!r}"
            )
        if orientation.to_frame != ECEF_FRAME.name:
            raise ValueError(
                f"parent frame must be {ECEF_FRAME.name!r}, "
                f"got {orientation.to_frame!r}"
            )
        return cls(HamiltonQuaternion(*orientation.quaternion_wxyz))


@dataclass(frozen=True)
class FramedOrientationBatch:
    """A batch of Hamilton WXYZ orientations sharing one named frame pair."""

    values: object
    from_frame: str
    to_frame: str

    def __post_init__(self) -> None:
        if not isinstance(self.from_frame, str) or not self.from_frame.strip():
            raise ValueError("from_frame must be a non-empty frame name")
        if not isinstance(self.to_frame, str) or not self.to_frame.strip():
            raise ValueError("to_frame must be a non-empty frame name")


@dataclass(frozen=True)
class SemanticQuaternionMap:
    """World/body frame correspondences for an orientation conversion.

    ``world`` maps the orientation's parent frame. ``body`` maps its
    child/local frame. Each coordinate map maps vector components from the
    correspondence's source frame into its destination frame.
    """

    world: FrameCorrespondence
    body: FrameCorrespondence
    _compiled: QuaternionFrameMap = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.world, FrameCorrespondence):
            raise TypeError("world must be a FrameCorrespondence")
        if not isinstance(self.body, FrameCorrespondence):
            raise TypeError("body must be a FrameCorrespondence")
        # Compile here so invalid parity and non-rotational output are rejected
        # when the semantic map is declared, rather than on a later keyframe.
        object.__setattr__(
            self,
            "_compiled",
            QuaternionFrameMap(self.world.coordinates, self.body.coordinates),
        )

    def to_destination(self, orientation: FramedOrientation) -> FramedOrientation:
        """Map one source orientation and its frame labels to the destination."""
        self._check_frames(orientation, self.world.source, self.body.source)
        mapped = self._compiled.to_destination_wxyz(orientation.quaternion_wxyz)
        return FramedOrientation(
            mapped,
            from_frame=self.body.destination.name,
            to_frame=self.world.destination.name,
        )

    def from_destination(self, orientation: FramedOrientation) -> FramedOrientation:
        """Map one destination orientation and its labels back to the source."""
        self._check_frames(orientation, self.world.destination, self.body.destination)
        mapped = self._compiled.from_destination_wxyz(orientation.quaternion_wxyz)
        return FramedOrientation(
            mapped,
            from_frame=self.body.source.name,
            to_frame=self.world.source.name,
        )

    def validate_source_labels(self, from_frame: str, to_frame: str) -> None:
        """Reject labels that do not match this map's source frame pair."""
        self._check_labels(from_frame, to_frame, self.world.source, self.body.source)

    def validate_destination_labels(self, from_frame: str, to_frame: str) -> None:
        """Reject labels that do not match this map's destination frame pair."""
        self._check_labels(
            from_frame, to_frame, self.world.destination, self.body.destination
        )

    def _check_frames(
        self,
        orientation: FramedOrientation,
        expected_world: CoordinateFrame,
        expected_body: CoordinateFrame,
    ) -> None:
        if not isinstance(orientation, FramedOrientation):
            raise TypeError("orientation must be a FramedOrientation")
        self._check_labels(
            orientation.from_frame,
            orientation.to_frame,
            expected_world,
            expected_body,
        )

    @staticmethod
    def _check_labels(
        from_frame: str,
        to_frame: str,
        expected_world: CoordinateFrame,
        expected_body: CoordinateFrame,
    ) -> None:
        if to_frame != expected_world.name:
            raise ValueError(
                f"orientation parent frame {to_frame!r} does not match "
                f"{expected_world.name!r}"
            )
        if from_frame != expected_body.name:
            raise ValueError(
                f"orientation child frame {from_frame!r} does not match "
                f"{expected_body.name!r}"
            )
