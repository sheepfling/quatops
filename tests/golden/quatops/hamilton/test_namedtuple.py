"""Golden contract tests for quatops.hamilton.namedtuple."""

from quatops.hamilton import namedtuple as backend

from .._vectors import QuaternionGoldens


class TestGolden(QuaternionGoldens):
    backend = backend
    convention = "hamilton"
    order = "wxyz"
    representation = "namedtuple"
