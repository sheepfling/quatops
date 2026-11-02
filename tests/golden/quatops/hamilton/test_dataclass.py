"""Golden contract tests for quatops.hamilton.dataclass."""

from quatops.hamilton import dataclass as backend

from .._vectors import QuaternionGoldens


class TestGolden(QuaternionGoldens):
    backend = backend
    convention = "hamilton"
    order = "wxyz"
    representation = "dataclass"
