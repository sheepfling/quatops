"""Golden contract tests for quatops.numpy.jpl.xyzw."""

from quatops.numpy.jpl import xyzw as backend

from ..._vectors import QuaternionGoldens


class TestGolden(QuaternionGoldens):
    backend = backend
    convention = "jpl"
    order = "xyzw"
    representation = "numpy"
