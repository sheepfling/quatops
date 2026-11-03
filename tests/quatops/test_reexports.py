"""Golden checks that every convenience path matches its canonical export."""

import importlib

import numpy as np
import pytest

from quatops import dataclass, namedtuple, scalar
from quatops.types import HamiltonQuaternion, JPLQuaternion


def assert_operation_aliases(alias, canonical):
    for name in (
        "conjugate", "from_axis_angle", "from_rotvec", "from_wxyz",
        "from_xyzw", "inverse", "multiply", "norm", "normalize", "rotate",
        "to_axis_angle", "to_rotvec", "to_wxyz", "to_xyzw",
    ):
        assert getattr(alias, name) is getattr(canonical, name), name


@pytest.mark.parametrize(
    ("alias", "target"),
    (
        (scalar, "quatops.hamilton.scalar"),
        (dataclass, "quatops.hamilton.dataclass"),
        (namedtuple, "quatops.hamilton.namedtuple"),
        (importlib.import_module("quatops.python.hamilton"), "quatops.hamilton.scalar"),
        (importlib.import_module("quatops.python.jpl"), "quatops.jpl.scalar"),
        (importlib.import_module("quatops.types.hamilton_dataclass"), "quatops.hamilton.dataclass"),
        (importlib.import_module("quatops.types.hamilton_namedtuple"), "quatops.hamilton.namedtuple"),
        (importlib.import_module("quatops.types.jpl_dataclass"), "quatops.jpl.dataclass"),
        (importlib.import_module("quatops.types.jpl_namedtuple"), "quatops.jpl.namedtuple"),
    ),
)
def test_python_and_root_aliases(alias, target):
    assert_operation_aliases(alias, importlib.import_module(target))


def test_type_exports():
    assert HamiltonQuaternion is importlib.import_module("quatops.hamilton.dataclass").Quaternion
    assert JPLQuaternion is importlib.import_module("quatops.jpl.dataclass").Quaternion


@pytest.mark.parametrize("convention", ("hamilton", "jpl"))
@pytest.mark.parametrize("order", ("wxyz", "xyzw"))
def test_convention_first_numpy_aliases(convention, order):
    canonical = importlib.import_module(f"quatops.numpy.{convention}.{order}")
    alias = importlib.import_module(f"quatops.{convention}.numpy_{order}")
    assert_operation_aliases(alias, canonical)
    assert isinstance(canonical.from_wxyz((1, 0, 0, 0)), np.ndarray)
