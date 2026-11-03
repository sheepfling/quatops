"""Precomputed signed-axis maps for Hamilton and JPL quaternion values."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class SignedAxisMap:
    """Map source vector axes into destination axes using signed permutations.

    ``destination_axes`` gives each destination component as a signed source
    axis. For example, ``("x", "y", "-z")`` means
    ``v_destination = (v_source.x, v_source.y, -v_source.z)``. The source frame
    must be right-handed. Destination frames may be right- or left-handed.

    The constructor validates and precomputes the component selectors once.
    Quaternion methods return Hamilton XYZW coefficients ready for game APIs;
    they preserve input magnitude and do not normalize or alter keyframe signs.
    """

    __slots__ = (
        "_hamilton_signs",
        "_indices",
        "_jpl_signs",
        "_vector_signs",
        "destination_axes",
        "determinant",
    )

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
