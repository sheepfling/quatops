"""Golden contract tests for quatops.jpl.dataclass."""

from quatops.jpl import dataclass as backend

from .._vectors import QuaternionGoldens


class TestGolden(QuaternionGoldens):
    backend = backend
    convention = "jpl"
    order = "xyzw"
    representation = "dataclass"
