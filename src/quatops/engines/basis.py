"""Precomputed signed-axis maps for Hamilton and JPL quaternion values."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..python import hamilton as _hamilton

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class SignedAxisMap:
    """Map source vector axes into destination axes using signed permutations.

    ``destination_axes`` gives each destination component as a signed source
    axis. For example, ``("x", "y", "-z")`` means
    ``v_destination = (v_source.x, v_source.y, -v_source.z)``. Source and
    destination frames may be right- or left-handed; their relative
    parity is recorded by ``determinant``. Frame meanings must be supplied by
    the caller and are never inferred from an engine name.

    The constructor validates and precomputes the component selectors once.
    Quaternion methods return Hamilton XYZW coefficients ready for game APIs;
    they preserve input magnitude and do not normalize or alter keyframe signs.
    """

    __slots__ = ("destination_axes", "_indices", "_vector_signs",
                 "_hamilton_signs", "_jpl_signs", "determinant")

    destination_axes: tuple[str, str, str]
    _indices: tuple[int, int, int]
    _vector_signs: tuple[int, int, int]
    _hamilton_signs: tuple[int, int, int]
    _jpl_signs: tuple[int, int, int]
    determinant: int

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("SignedAxisMap instances are immutable")
        object.__setattr__(self, name, value)

    def __init__(self, destination_axes: Sequence[str]) -> None:
        """Compile a map such as ``SignedAxisMap(("x", "y", "-z"))``."""
        axes = tuple(destination_axes)
        if len(axes) != 3:
            raise ValueError("destination_axes must contain exactly three axes")

        indices: list[int] = []
        signs: list[int] = []
        normalized: list[str] = []
        for axis in axes:
            if not isinstance(axis, str):
                raise TypeError("each destination axis must be a string")
            token = axis.strip().lower()
            sign = -1 if token.startswith("-") else 1
            name = token[1:] if token.startswith(("+", "-")) else token
            if name not in _AXIS_INDEX or token not in (name, "+" + name, "-" + name):
                raise ValueError(
                    "axes must be x, y, z, -x, -y, or -z"
                )
            indices.append(_AXIS_INDEX[name])
            signs.append(sign)
            normalized.append(("-" if sign < 0 else "") + name)
        if len(set(indices)) != 3:
            raise ValueError("destination_axes must use x, y, and z exactly once")

        inversions = sum(
            indices[left] > indices[right]
            for left in range(3)
            for right in range(left + 1, 3)
        )
        permutation_sign = -1 if inversions % 2 else 1
        determinant = permutation_sign * signs[0] * signs[1] * signs[2]
        hamilton_signs = tuple(determinant * sign for sign in signs)

        self.destination_axes = tuple(normalized)  # type: ignore[assignment]
        self._indices = tuple(indices)  # type: ignore[assignment]
        self._vector_signs = tuple(signs)  # type: ignore[assignment]
        self._hamilton_signs = hamilton_signs  # type: ignore[assignment]
        self._jpl_signs = tuple(
            -sign for sign in hamilton_signs
        )  # type: ignore[assignment]
        self.determinant = determinant

    @staticmethod
    def _four(q: Iterable[float]) -> tuple[float, float, float, float]:
        values = tuple(q)
        if len(values) != 4:
            raise ValueError("a quaternion must contain exactly four components")
        return values  # type: ignore[return-value]

    def vector(self, v: Iterable[float]) -> tuple[float, float, float]:
        """Map a three-vector from source to destination component axes."""
        x, y, z = v
        source = (x, y, z)
        indices = self._indices
        signs = self._vector_signs
        return (
            source[indices[0]] if signs[0] > 0 else -source[indices[0]],
            source[indices[1]] if signs[1] > 0 else -source[indices[1]],
            source[indices[2]] if signs[2] > 0 else -source[indices[2]],
        )

    @property
    def matrix(self) -> tuple[tuple[int, int, int], ...]:
        """Return the row-major matrix ``B`` such that ``v_destination = B v_source``."""
        rows = [[0, 0, 0] for _ in range(3)]
        for destination_index, (source_index, sign) in enumerate(
            zip(self._indices, self._vector_signs)
        ):
            rows[destination_index][source_index] = sign
        return tuple(tuple(row) for row in rows)  # type: ignore[return-value]

    def inverse(self) -> "SignedAxisMap":
        """Return the destination-to-source signed-axis map ``B.T``."""
        inverse_axes = [""] * 3
        for destination_index, (source_index, sign) in enumerate(
            zip(self._indices, self._vector_signs)
        ):
            axis = "xyz"[destination_index]
            inverse_axes[source_index] = ("-" if sign < 0 else "") + axis
        return SignedAxisMap(inverse_axes)

    def map_hamilton_xyzw(
        self, q: Iterable[float]
    ) -> tuple[float, float, float, float]:
        """Map active Hamilton ``xyzw`` coefficients to engine ``xyzw``."""
        x, y, z, w = self._four(q)
        source = (x, y, z)
        indices = self._indices
        signs = self._hamilton_signs
        return (
            source[indices[0]] if signs[0] > 0 else -source[indices[0]],
            source[indices[1]] if signs[1] > 0 else -source[indices[1]],
            source[indices[2]] if signs[2] > 0 else -source[indices[2]],
            w,
        )

    def map_hamilton_wxyz(
        self, q: Iterable[float]
    ) -> tuple[float, float, float, float]:
        """Map active Hamilton ``wxyz`` coefficients to engine ``xyzw``."""
        w, x, y, z = self._four(q)
        source = (x, y, z)
        indices = self._indices
        signs = self._hamilton_signs
        return (
            source[indices[0]] if signs[0] > 0 else -source[indices[0]],
            source[indices[1]] if signs[1] > 0 else -source[indices[1]],
            source[indices[2]] if signs[2] > 0 else -source[indices[2]],
            w,
        )

    def map_jpl_xyzw(
        self, q: Iterable[float]
    ) -> tuple[float, float, float, float]:
        """Map active JPL ``xyzw`` values directly to engine Hamilton ``xyzw``."""
        x, y, z, w = self._four(q)
        source = (x, y, z)
        indices = self._indices
        signs = self._jpl_signs
        return (
            source[indices[0]] if signs[0] > 0 else -source[indices[0]],
            source[indices[1]] if signs[1] > 0 else -source[indices[1]],
            source[indices[2]] if signs[2] > 0 else -source[indices[2]],
            w,
        )

    def map_jpl_wxyz(
        self, q: Iterable[float]
    ) -> tuple[float, float, float, float]:
        """Map active JPL ``wxyz`` values directly to engine Hamilton ``xyzw``."""
        w, x, y, z = self._four(q)
        source = (x, y, z)
        indices = self._indices
        signs = self._jpl_signs
        return (
            source[indices[0]] if signs[0] > 0 else -source[indices[0]],
            source[indices[1]] if signs[1] > 0 else -source[indices[1]],
            source[indices[2]] if signs[2] > 0 else -source[indices[2]],
            w,
        )

    def __repr__(self) -> str:
        return f"SignedAxisMap({self.destination_axes!r})"


class QuaternionFrameMap:
    """Convert active Hamilton WXYZ orientations between explicit frames.

    ``world_map`` and ``body_map`` each map source-frame vector coordinates to
    destination-frame coordinates. Their determinants must match so the
    resulting orientation remains proper. The two fixed frame corrections are
    compiled once; calls preserve quaternion magnitude and the input sign.
    Component storage conversion is intentionally left to the caller.
    """

    __slots__ = (
        "world_map",
        "body_map",
        "_world_correction",
        "_body_correction",
        "_shared_map",
        "_shared_inverse_map",
    )

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("QuaternionFrameMap instances are immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self, world_map: SignedAxisMap, body_map: SignedAxisMap | None = None
    ) -> None:
        if not isinstance(world_map, SignedAxisMap):
            raise TypeError("world_map must be a SignedAxisMap")
        if body_map is None:
            body_map = world_map
        elif not isinstance(body_map, SignedAxisMap):
            raise TypeError("body_map must be a SignedAxisMap")
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
            if world_map.destination_axes == body_map.destination_axes
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
            _hamilton.multiply(
                _hamilton.inverse(self._world_correction), destination
            ),
            self._body_correction,
        )


def change_rotation_basis(
    rotation_matrix: Iterable[Iterable[float]],
    world_map: SignedAxisMap,
    body_map: SignedAxisMap | None = None,
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
    if not isinstance(world_map, SignedAxisMap):
        raise TypeError("world_map must be a SignedAxisMap")
    if body_map is None:
        body_map = world_map
    elif not isinstance(body_map, SignedAxisMap):
        raise TypeError("body_map must be a SignedAxisMap")

    rows = tuple(tuple(float(value) for value in row) for row in rotation_matrix)
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("rotation_matrix must be a 3x3 matrix")

    world = world_map.matrix
    body = body_map.matrix
    return tuple(
        tuple(
            sum(
                world[row][i] * rows[i][j] * body[column][j]
                for i in range(3)
                for j in range(3)
            )
            for column in range(3)
        )
        for row in range(3)
    )
