"""Shared result record for Euler extraction APIs."""

from typing import Any, NamedTuple


class EulerResult(NamedTuple):
    """Euler angles and a flag identifying singular or tolerance-snapped input."""

    angles: Any
    gimbal_lock: Any
