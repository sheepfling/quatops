"""Experimental general orthogonal and two-frame orientation maps."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import isfinite
from typing import Union

from ...engines.basis import SignedAxisMap
from ...python import hamilton as _hamilton


class OrthogonalBasisMap:
    """Map vector coordinates with a general orthogonal 3x3 basis matrix.

    The row-major matrix maps source components to destination components.
    Use :class:`SignedAxisMap` for the faster signed-permutation case. This
    class also supports continuously varying bases such as a local tangent
    frame at a geographic origin.
    """

    __slots__ = ("_matrix", "determinant")

    _matrix: tuple[tuple[float, float, float], ...]
    determinant: int

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("OrthogonalBasisMap instances are immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self, matrix: Sequence[Sequence[float]], tolerance: float = 1e-10
    ) -> None:
        rows = tuple(tuple(float(value) for value in row) for row in matrix)
        if len(rows) != 3 or any(len(row) != 3 for row in rows):
            raise ValueError("matrix must be 3x3")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        if any(not isfinite(value) for row in rows for value in row):
            raise ValueError("matrix values must be finite")

        gram = tuple(
            tuple(sum(rows[k][i] * rows[k][j] for k in range(3)) for j in range(3))
            for i in range(3)
        )
        if any(
            abs(gram[i][j] - (1.0 if i == j else 0.0)) > tolerance
            for i in range(3)
            for j in range(3)
        ):
            raise ValueError("matrix must be orthogonal within tolerance")

        determinant_value = (
            rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
            - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
            + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
        )
        if abs(abs(determinant_value) - 1.0) > tolerance:
            raise ValueError("matrix determinant must be +1 or -1")
        self._matrix = rows  # type: ignore[assignment]
        self.determinant = 1 if determinant_value > 0.0 else -1

    @property
    def matrix(self) -> tuple[tuple[float, float, float], ...]:
        """Return the row-major source-to-destination component map."""
        return self._matrix

    def vector(self, value: Iterable[float]) -> tuple[float, float, float]:
        """Map one 3-vector into destination coordinates."""
        vector = tuple(float(component) for component in value)
        if len(vector) != 3:
            raise ValueError("a vector must contain exactly three components")
        return tuple(
            sum(row[column] * vector[column] for column in range(3))
            for row in self._matrix
        )  # type: ignore[return-value]

    def inverse(self) -> OrthogonalBasisMap:
        """Return the inverse map, which is the transpose of this matrix."""
        return OrthogonalBasisMap(
            tuple(
                tuple(self._matrix[column][row] for column in range(3))
                for row in range(3)
            )
        )

    def __repr__(self) -> str:
        return f"OrthogonalBasisMap({self._matrix!r})"


BasisMap = Union[SignedAxisMap, OrthogonalBasisMap]


def _is_basis_map(value: object) -> bool:
    return isinstance(value, (SignedAxisMap, OrthogonalBasisMap))


class QuaternionFrameMap:
    """Convert active Hamilton WXYZ orientations between explicit frames.

    ``world_map`` and ``body_map`` each map source-frame vector coordinates to
    destination-frame coordinates. Their determinants must match so the
    resulting orientation remains proper. The two fixed frame corrections are
    compiled once; calls preserve quaternion magnitude and the input sign.
    Component storage conversion is intentionally left to the caller.
    """

    __slots__ = (
        "_body_correction",
        "_shared_inverse_map",
        "_shared_map",
        "_world_correction",
        "body_map",
        "world_map",
    )

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("QuaternionFrameMap instances are immutable")
        object.__setattr__(self, name, value)

    def __init__(self, world_map: BasisMap, body_map: BasisMap | None = None) -> None:
        if not _is_basis_map(world_map):
            raise TypeError("world_map must be a SignedAxisMap or OrthogonalBasisMap")
        if body_map is None:
            body_map = world_map
        elif not _is_basis_map(body_map):
            raise TypeError("body_map must be a SignedAxisMap or OrthogonalBasisMap")
        if world_map.determinant != body_map.determinant:
            raise ValueError(
                "world_map and body_map must have the same determinant "
                "to produce a proper quaternion orientation"
            )

        world_correction = tuple(
            tuple(world_map.determinant * value for value in row)
            for row in world_map.matrix
        )
        body_correction = tuple(
            tuple(body_map.determinant * value for value in row)
            for row in body_map.matrix
        )
        self.world_map = world_map
        self.body_map = body_map
        self._shared_map = (
            world_map
            if (
                isinstance(world_map, SignedAxisMap)
                and isinstance(body_map, SignedAxisMap)
                and world_map.destination_axes == body_map.destination_axes
            )
            else None
        )
        self._shared_inverse_map = (
            self._shared_map.inverse() if self._shared_map is not None else None
        )
        self._world_correction = _hamilton.from_rotation_matrix(world_correction)
        self._body_correction = _hamilton.from_rotation_matrix(body_correction)

    @staticmethod
    def _four(q: Iterable[float]) -> tuple[float, float, float, float]:
        values = tuple(q)
        if len(values) != 4:
            raise ValueError("a quaternion must contain exactly four components")
        return values  # type: ignore[return-value]

    def to_destination_wxyz(
        self, q_source_wxyz: Iterable[float]
    ) -> tuple[float, float, float, float]:
        """Map a source-frame Hamilton WXYZ orientation into destination frames."""
        source = self._four(q_source_wxyz)
        if self._shared_map is not None:
            target_xyzw = self._shared_map.map_hamilton_wxyz(source)
            return (target_xyzw[3], target_xyzw[0], target_xyzw[1], target_xyzw[2])
        return _hamilton.multiply(
            _hamilton.multiply(self._world_correction, source),
            _hamilton.inverse(self._body_correction),
        )

    def from_destination_wxyz(
        self, q_destination_wxyz: Iterable[float]
    ) -> tuple[float, float, float, float]:
        """Map a destination-frame Hamilton WXYZ orientation back to source."""
        destination = self._four(q_destination_wxyz)
        if self._shared_map is not None:
            target_xyzw = (
                destination[1],
                destination[2],
                destination[3],
                destination[0],
            )
            # Compile the inverse alongside the forward map; callers can reuse
            # this object for every keyframe without per-sample map creation.
            assert self._shared_inverse_map is not None
            source_xyzw = self._shared_inverse_map.map_hamilton_xyzw(target_xyzw)
            return (source_xyzw[3], source_xyzw[0], source_xyzw[1], source_xyzw[2])
        return _hamilton.multiply(
            _hamilton.multiply(_hamilton.inverse(self._world_correction), destination),
            self._body_correction,
        )

    @property
    def world_correction_wxyz(self) -> tuple[float, float, float, float]:
        """Return the compiled proper-rotation correction for the world map."""
        return self._world_correction

    @property
    def body_correction_wxyz(self) -> tuple[float, float, float, float]:
        """Return the compiled proper-rotation correction for the body map."""
        return self._body_correction


def change_rotation_basis(
    rotation_matrix: Iterable[Iterable[float]],
    world_map: BasisMap,
    body_map: BasisMap | None = None,
) -> tuple[tuple[float, float, float], ...]:
    """Express a rotation matrix in separately mapped world and body frames.

    The input is a row-major matrix acting on column vectors. ``world_map``
    maps source-world vector coordinates into destination-world coordinates;
    ``body_map`` does the same for body coordinates and defaults to
    ``world_map``. The result is ``B_world @ R_source @ B_body.T``.

    This returns a matrix even when the result is improper. It represents an
    orientation quaternion only when the world and body maps have the same
    determinant, making the result's determinant +1.
    """
    if not _is_basis_map(world_map):
        raise TypeError("world_map must be a SignedAxisMap or OrthogonalBasisMap")
    if body_map is None:
        body_map = world_map
    elif not _is_basis_map(body_map):
        raise TypeError("body_map must be a SignedAxisMap or OrthogonalBasisMap")

    rows = tuple(tuple(float(value) for value in row) for row in rotation_matrix)
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("rotation_matrix must be a 3x3 matrix")

    world = world_map.matrix
    body = body_map.matrix

    def entry(row: int, column: int) -> float:
        return float(
            sum(
                world[row][i] * rows[i][j] * body[column][j]
                for i in range(3)
                for j in range(3)
            )
        )

    return (
        (entry(0, 0), entry(0, 1), entry(0, 2)),
        (entry(1, 0), entry(1, 1), entry(1, 2)),
        (entry(2, 0), entry(2, 1), entry(2, 2)),
    )
