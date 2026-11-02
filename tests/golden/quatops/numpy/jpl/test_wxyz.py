"""Golden contract tests for quatops.numpy.jpl.wxyz."""

from quatops.numpy.jpl import wxyz as backend

from ..._vectors import QuaternionGoldens


class TestGolden(QuaternionGoldens):
    backend = backend
    convention = "jpl"
    order = "wxyz"
    representation = "numpy"
