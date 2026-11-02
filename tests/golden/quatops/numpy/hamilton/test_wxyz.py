"""Golden contract tests for quatops.numpy.hamilton.wxyz."""

from quatops.numpy.hamilton import wxyz as backend

from ..._vectors import QuaternionGoldens


class TestGolden(QuaternionGoldens):
    backend = backend
    convention = "hamilton"
    order = "wxyz"
    representation = "numpy"
