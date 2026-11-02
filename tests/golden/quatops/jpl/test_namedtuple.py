"""Golden contract tests for quatops.jpl.namedtuple."""

from quatops.jpl import namedtuple as backend

from .._vectors import QuaternionGoldens


class TestGolden(QuaternionGoldens):
    backend = backend
    convention = "jpl"
    order = "xyzw"
    representation = "namedtuple"
