"""Golden contract tests for quatops.hamilton.scalar."""

from quatops.hamilton import scalar as backend

from .._vectors import QuaternionGoldens


class TestGolden(QuaternionGoldens):
    backend = backend
    convention = "hamilton"
    order = "wxyz"
    representation = "tuple"
