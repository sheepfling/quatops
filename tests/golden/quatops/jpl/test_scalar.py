"""Golden contract tests for quatops.jpl.scalar."""

from quatops.jpl import scalar as backend

from .._vectors import QuaternionGoldens


class TestGolden(QuaternionGoldens):
    backend = backend
    convention = "jpl"
    order = "xyzw"
    representation = "tuple"
