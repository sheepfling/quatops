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
from math import hypot, isfinite
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
FrameRole = Literal["world", "body", "sensor", "generic"]


def _checked_wxyz(values: Iterable[float]) -> tuple[float, float, float, float]:
    components = tuple(float(value) for value in values)
    if len(components) != 4:
        raise ValueError("quaternion_wxyz must contain exactly four components")
    w, x, y, z = components
    if any(not isfinite(value) for value in (w, x, y, z)):
        raise ValueError("quaternion components must be finite")
    return w, x, y, z


def _checked_unit_wxyz(values: Iterable[float]) -> tuple[float, float, float, float]:
    quaternion = _checked_wxyz(values)
    norm = hypot(
        hypot(quaternion[0], quaternion[1]), hypot(quaternion[2], quaternion[3])
    )
    if abs(norm - 1.0) > 1e-6:
        raise ValueError(
            "physical orientation requires a unit quaternion within 1e-6; "
            "normalize explicitly before construction"
        )
    return quaternion


@dataclass(frozen=True)
class CoordinateFrame:
    """A named frame whose basis vectors are declared in a named reference.

    The columns of ``basis_to_reference`` are this frame's positive X/Y/Z axes
    expressed in the right-handed reference frame. ``reference`` identifies
    the common physical basis; ``reference_convention`` optionally describes
    its axis semantics. Axis labels are descriptive; the numeric basis is the
    conversion source of truth.
    """

    name: str
    reference: str
    axes: tuple[str, str, str]
    basis_to_reference: BasisMap
    handedness: Handedness = "right"
    role: FrameRole = "generic"
    reference_convention: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("frame name must be a non-empty string")
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise ValueError("reference must be a non-empty frame name")
        if self.reference_convention is not None and (
            not isinstance(self.reference_convention, str)
            or not self.reference_convention.strip()
        ):
            raise ValueError("reference_convention must be a non-empty string when set")
        if len(self.axes) != 3 or any(
            not isinstance(axis, str) or not axis.strip() for axis in self.axes
        ):
            raise ValueError("axes must name the positive X, Y, and Z directions")
        if len({axis.strip().casefold() for axis in self.axes}) != 3:
            raise ValueError("axis meanings must be distinct")
        if self.handedness not in ("right", "left"):
            raise ValueError("handedness must be 'right' or 'left'")
        if self.role not in ("world", "body", "sensor", "generic"):
            raise ValueError("unsupported coordinate-frame role")
        if not isinstance(self.basis_to_reference, (SignedAxisMap, OrthogonalBasisMap)):
            raise TypeError(
                "basis_to_reference must be a SignedAxisMap or OrthogonalBasisMap"
            )
        expected_parity = 1 if self.handedness == "right" else -1
        if self.basis_to_reference.determinant != expected_parity:
            raise ValueError("frame handedness does not match basis_to_reference")

    def has_same_definition(self, other: object) -> bool:
        """Return whether ``other`` declares this same named physical frame.

        Basis matrices are compared within ``1e-12`` so equivalent signed-axis
        and general orthogonal maps can be used interchangeably.
        """
        return isinstance(other, CoordinateFrame) and _same_frame_definition(
            self, other
        )


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
        if (
            self.source.reference_convention is not None
            and self.destination.reference_convention is not None
            and self.source.reference_convention
            != self.destination.reference_convention
        ):
            raise ValueError("mapped frames must use the same reference convention")
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


def _same_frame_definition(left: CoordinateFrame, right: CoordinateFrame) -> bool:
    if left is right:
        return True
    if (
        left.name != right.name
        or left.reference != right.reference
        or left.axes != right.axes
        or left.handedness != right.handedness
        or left.role != right.role
        or (
            left.reference_convention is not None
            and right.reference_convention is not None
            and left.reference_convention != right.reference_convention
        )
    ):
        return False
    left_basis = left.basis_to_reference.matrix
    right_basis = right.basis_to_reference.matrix
    return all(
        abs(left_basis[row][column] - right_basis[row][column]) <= 1e-12
        for row in range(3)
        for column in range(3)
    )


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
class PhysicalOrientation:
    """A Hamilton quaternion with explicit physical parent and child frames.

    The stored value is ``q_parent_from_child`` in active Hamilton WXYZ form.
    Frame bases carry its physical axis semantics. Construction requires a
    unit quaternion within 1e-6 and preserves its sign. Call ``normalized``
    explicitly when normalization is intended.
    """

    quaternion: HamiltonQuaternion
    parent_frame: CoordinateFrame
    child_frame: CoordinateFrame

    def __post_init__(self) -> None:
        if not isinstance(self.quaternion, HamiltonQuaternion):
            raise TypeError("quaternion must be a HamiltonQuaternion")
        if not isinstance(self.parent_frame, CoordinateFrame):
            raise TypeError("parent_frame must be a CoordinateFrame")
        if not isinstance(self.child_frame, CoordinateFrame):
            raise TypeError("child_frame must be a CoordinateFrame")
        if self.parent_frame.handedness != self.child_frame.handedness:
            raise ValueError(
                "a physical orientation requires parent and child frames "
                "with matching handedness"
            )
        values = _checked_unit_wxyz(
            (self.quaternion.w, self.quaternion.x, self.quaternion.y, self.quaternion.z)
        )
        object.__setattr__(self, "quaternion", HamiltonQuaternion(*values))

    @classmethod
    def from_wxyz(
        cls,
        values: Iterable[float],
        parent_frame: CoordinateFrame,
        child_frame: CoordinateFrame,
    ) -> PhysicalOrientation:
        """Construct a physical orientation from active Hamilton WXYZ values."""
        return cls(
            HamiltonQuaternion(*_checked_wxyz(values)), parent_frame, child_frame
        )

    @property
    def quaternion_wxyz(self) -> tuple[float, float, float, float]:
        """Return the standard Hamilton coefficients in WXYZ order."""
        w, x, y, z = self.quaternion
        return w, x, y, z

    def to_framed_hamilton(self) -> FramedOrientation:
        """Expose the canonical value with its directed frame labels."""
        return FramedOrientation(
            self.quaternion_wxyz,
            from_frame=self.child_frame.name,
            to_frame=self.parent_frame.name,
        )

    def normalized(self) -> PhysicalOrientation:
        """Return a unit orientation after explicit Hamilton normalization."""
        values = _hamilton.normalize(self.quaternion_wxyz)
        return PhysicalOrientation(
            HamiltonQuaternion(*values), self.parent_frame, self.child_frame
        )

    def compose(self, right: PhysicalOrientation) -> PhysicalOrientation:
        """Compose ``self * right`` after matching the intermediate frame."""
        if not isinstance(right, PhysicalOrientation):
            raise TypeError("right must be a PhysicalOrientation")
        if not self.child_frame.has_same_definition(right.parent_frame):
            raise ValueError(
                "composition requires the same intermediate frame definition"
            )
        product = _hamilton.multiply(self.quaternion_wxyz, right.quaternion_wxyz)
        return PhysicalOrientation.from_wxyz(
            product, parent_frame=self.parent_frame, child_frame=right.child_frame
        )

    def inverse(self) -> PhysicalOrientation:
        """Reverse the parent/child frames and invert the orientation."""
        return PhysicalOrientation.from_wxyz(
            _hamilton.inverse(self.quaternion_wxyz),
            parent_frame=self.child_frame,
            child_frame=self.parent_frame,
        )

    @classmethod
    def from_framed_hamilton(
        cls,
        orientation: FramedOrientation,
        parent_frame: CoordinateFrame,
        child_frame: CoordinateFrame,
    ) -> PhysicalOrientation:
        """Attach frame definitions after checking the directed frame labels."""
        if not isinstance(orientation, FramedOrientation):
            raise TypeError("orientation must be a FramedOrientation")
        if orientation.to_frame != parent_frame.name:
            raise ValueError(
                f"parent frame must be {parent_frame.name!r}, "
                f"got {orientation.to_frame!r}"
            )
        if orientation.from_frame != child_frame.name:
            raise ValueError(
                f"child frame must be {child_frame.name!r}, "
                f"got {orientation.from_frame!r}"
            )
        return cls(
            HamiltonQuaternion(*orientation.quaternion_wxyz), parent_frame, child_frame
        )


@dataclass(frozen=True)
class PhysicalOrientationBatch:
    """A unit WXYZ orientation batch sharing parent and child frames.

    The common NumPy conversion gateway checks that every quaternion is unit
    length within 1e-6. Construction itself does not import NumPy.
    """

    values: object
    parent_frame: CoordinateFrame
    child_frame: CoordinateFrame

    def __post_init__(self) -> None:
        if not isinstance(self.parent_frame, CoordinateFrame):
            raise TypeError("parent_frame must be a CoordinateFrame")
        if not isinstance(self.child_frame, CoordinateFrame):
            raise TypeError("child_frame must be a CoordinateFrame")
        if self.parent_frame.handedness != self.child_frame.handedness:
            raise ValueError(
                "a physical orientation batch requires parent and child frames "
                "with matching handedness"
            )

    def to_framed_batch(self) -> FramedOrientationBatch:
        """Expose batch values with their directed frame labels."""
        return FramedOrientationBatch(
            values=self.values,
            from_frame=self.child_frame.name,
            to_frame=self.parent_frame.name,
        )


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
