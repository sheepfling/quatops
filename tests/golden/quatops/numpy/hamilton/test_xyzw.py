"""Golden contract tests for quatops.numpy.hamilton.xyzw."""

from quatops.numpy.hamilton import xyzw as backend

from ..._vectors import QuaternionGoldens


class TestGolden(QuaternionGoldens):
    backend = backend
    convention = "hamilton"
    order = "xyzw"
    representation = "numpy"
