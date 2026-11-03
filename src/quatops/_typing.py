"""Shared scalar input types for the dependency-free Python APIs."""

from typing import SupportsFloat, TypeVar

FloatLike = SupportsFloat
FloatT = TypeVar("FloatT", bound=SupportsFloat)
