"""Reusable physical frame conventions, separate from target encodings."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from ...types import HamiltonQuaternion
from ..engines.basis import BasisMap, OrthogonalBasisMap, SignedAxisMap
from .frames import (
    CoordinateFrame,
    FramedOrientation,
    PhysicalOrientation,
    _checked_unit_wxyz,
)

ConventionRole = Literal["world", "body", "sensor"]


@dataclass(frozen=True)
class FrameConvention:
    """Axis semantics reusable across frame instances and target libraries.

    ``basis_to_reference`` maps this convention's component axes into the
    named ``reference_convention`` axes. Bind it to an application frame with
    :meth:`bind`; the binding's ``reference`` must identify that same shared
    basis for every frame being compared.
    """

    convention_id: str
    role: ConventionRole
    axes: tuple[str, str, str]
    reference_convention: str
    basis_to_reference: BasisMap
    handedness: Literal["right", "left"]

    def __post_init__(self) -> None:
        if not self.convention_id or not self.reference_convention:
            raise ValueError("convention and reference IDs must not be empty")
        if self.role not in ("world", "body", "sensor"):
            raise ValueError("role must be 'world', 'body', or 'sensor'")
        if self.handedness not in ("right", "left"):
            raise ValueError("handedness must be 'right' or 'left'")
        if not isinstance(self.basis_to_reference, (SignedAxisMap, OrthogonalBasisMap)):
            raise TypeError("basis_to_reference must be an orthogonal basis map")
        CoordinateFrame(
            name=self.convention_id,
            reference=self.reference_convention,
            axes=self.axes,
            basis_to_reference=self.basis_to_reference,
            handedness=self.handedness,
            role=self.role,
            reference_convention=self.reference_convention,
        )

    def bind(
        self,
        name: str,
        reference: str,
        *,
        basis_to_reference: BasisMap | None = None,
        reference_convention: str | None = None,
    ) -> CoordinateFrame:
        """Bind this convention to a named frame and reference.

        Supply both optional arguments when a provider has computed a dynamic
        basis, such as local ENU axes expressed in ECEF at a specific origin.
        """
        if (basis_to_reference is None) != (reference_convention is None):
            raise ValueError(
                "custom basis_to_reference and reference_convention must be "
                "provided together"
            )
        basis = (
            self.basis_to_reference
            if basis_to_reference is None
            else basis_to_reference
        )
        basis_convention = (
            self.reference_convention
            if reference_convention is None
            else reference_convention
        )
        return CoordinateFrame(
            name=name,
            reference=reference,
            axes=self.axes,
            basis_to_reference=basis,
            handedness=self.handedness,
            role=self.role,
            reference_convention=basis_convention,
        )


_IDENTITY = SignedAxisMap(("x", "y", "z"))
_BODY_RFU_REFERENCE = "quatops.body.right-forward-up"
_WORLD_RFU_REFERENCE = "quatops.world.right-forward-up"
_WORLD_ENU_REFERENCE = "quatops.world.east-north-up"

CANONICAL_WORLD_RFU = FrameConvention(
    convention_id="quatops.world.right-forward-up",
    role="world",
    axes=("right", "forward", "up"),
    reference_convention=_WORLD_RFU_REFERENCE,
    basis_to_reference=_IDENTITY,
    handedness="right",
)
CANONICAL_BODY_RFU = FrameConvention(
    convention_id="quatops.body.right-forward-up",
    role="body",
    axes=("right", "forward", "up"),
    reference_convention=_BODY_RFU_REFERENCE,
    basis_to_reference=_IDENTITY,
    handedness="right",
)
ECEF_WORLD = FrameConvention(
    convention_id="quatops.ecef",
    role="world",
    axes=(
        "+X equator at prime meridian",
        "+Y equator at 90 degrees east",
        "+Z north pole",
    ),
    reference_convention="quatops.ecef",
    basis_to_reference=_IDENTITY,
    handedness="right",
)
WORLD_ENU = FrameConvention(
    convention_id="quatops.world.enu",
    role="world",
    axes=("east", "north", "up"),
    reference_convention=_WORLD_ENU_REFERENCE,
    basis_to_reference=_IDENTITY,
    handedness="right",
)
WORLD_NED = FrameConvention(
    convention_id="navigation.world.ned",
    role="world",
    axes=("north", "east", "down"),
    reference_convention=_WORLD_ENU_REFERENCE,
    basis_to_reference=OrthogonalBasisMap(
        ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
    ),
    handedness="right",
)
ROS_BODY_FLU = FrameConvention(
    convention_id="ros.body.flu",
    role="body",
    axes=("forward", "left", "up"),
    reference_convention=_BODY_RFU_REFERENCE,
    basis_to_reference=OrthogonalBasisMap(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    ),
    handedness="right",
)
ROS_CAMERA_OPTICAL = FrameConvention(
    convention_id="ros.camera.optical",
    role="sensor",
    axes=("right", "down", "forward"),
    reference_convention=_BODY_RFU_REFERENCE,
    basis_to_reference=OrthogonalBasisMap(
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
    ),
    handedness="right",
)
AEROSPACE_BODY_FRD = FrameConvention(
    convention_id="aerospace.body.frd",
    role="body",
    axes=("forward", "right", "down"),
    reference_convention=_BODY_RFU_REFERENCE,
    basis_to_reference=OrthogonalBasisMap(
        ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
    ),
    handedness="right",
)
OPENXR_SPACE_RUB = FrameConvention(
    convention_id="openxr.space.right-up-back",
    role="world",
    axes=("right", "up", "back"),
    reference_convention=_WORLD_RFU_REFERENCE,
    basis_to_reference=OrthogonalBasisMap(
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
    ),
    handedness="right",
)

ECEF_FRAME = ECEF_WORLD.bind("quatops.ecef", "quatops.ecef")
CANONICAL_WORLD_FRAME = CANONICAL_WORLD_RFU.bind(
    "quatops.world.right-forward-up", _WORLD_RFU_REFERENCE
)
CANONICAL_BODY_FRAME = CANONICAL_BODY_RFU.bind(
    "quatops.body.right-forward-up", _BODY_RFU_REFERENCE
)


@dataclass(frozen=True)
class ECEFBodyOrientation:
    """Convenience semantic value for ``q_ECEF_from_body`` in WXYZ order."""

    quaternion: HamiltonQuaternion

    def __post_init__(self) -> None:
        if not isinstance(self.quaternion, HamiltonQuaternion):
            raise TypeError("quaternion must be a HamiltonQuaternion")
        values = _checked_unit_wxyz(
            (self.quaternion.w, self.quaternion.x, self.quaternion.y, self.quaternion.z)
        )
        object.__setattr__(self, "quaternion", HamiltonQuaternion(*values))

    @classmethod
    def from_wxyz(cls, values: Iterable[float]) -> ECEFBodyOrientation:
        """Construct ``q_ECEF_from_body`` from Hamilton WXYZ coefficients."""
        return cls(HamiltonQuaternion(*_checked_unit_wxyz(values)))

    @property
    def quaternion_wxyz(self) -> tuple[float, float, float, float]:
        """Return the standard Hamilton coefficients in WXYZ order."""
        w, x, y, z = self.quaternion
        return w, x, y, z

    def to_physical_orientation(self) -> PhysicalOrientation:
        """Return the generic physical orientation with its standard frames."""
        return PhysicalOrientation(self.quaternion, ECEF_FRAME, CANONICAL_BODY_FRAME)

    def to_framed_hamilton(self) -> FramedOrientation:
        """Expose the canonical Hamilton value with explicit frame labels."""
        return self.to_physical_orientation().to_framed_hamilton()

    @classmethod
    def from_framed_hamilton(
        cls, orientation: FramedOrientation
    ) -> ECEFBodyOrientation:
        """Accept a framed Hamilton value only when its labels match ECEF/body."""
        physical = PhysicalOrientation.from_framed_hamilton(
            orientation,
            parent_frame=ECEF_FRAME,
            child_frame=CANONICAL_BODY_FRAME,
        )
        return cls(physical.quaternion)
