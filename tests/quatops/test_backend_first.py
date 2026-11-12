"""Backend-first imports re-export the selected native kernels directly."""

import importlib
import subprocess
import sys

COMMON = (
    "from_xyzw",
    "to_xyzw",
    "from_wxyz",
    "to_wxyz",
    "conjugate",
    "add",
    "subtract",
    "negate",
    "scale",
    "multiply",
    "divide",
    "norm",
    "normalize",
    "inverse",
    "rotate",
    "from_axis_angle",
    "to_axis_angle",
    "from_rotvec",
    "to_rotvec",
    "from_rotation_vector",
    "to_rotation_vector",
    "exp",
    "log",
    "sin",
    "cos",
    "to_rotation_matrix",
    "to_left_matrix",
    "to_right_matrix",
    "sinh",
    "cosh",
    "tanh",
    "power",
    "pow_real",
    "sqrt",
    "slerp",
    "skew_matrix",
    "so3_left_jacobian",
    "so3_right_jacobian",
    "so3_left_jacobian_inverse",
    "so3_right_jacobian_inverse",
    "integrate_angular_velocity",
)
JPL_EXTRA = ("from_hamilton", "to_hamilton")

class BackendFirstTests:
    def assert_direct_exports(self, alias_name, target_name, operations):
        alias = importlib.import_module(alias_name)
        target = importlib.import_module(target_name)
        for name in operations:
            assert getattr(alias, name) is getattr(target, name)

    def test_python_aliases_are_the_native_scalar_functions(self):
        self.assert_direct_exports(
            "quatops.python.hamilton", "quatops.hamilton.scalar", COMMON
        )
        self.assert_direct_exports(
            "quatops.python.jpl", "quatops.jpl.scalar", COMMON + JPL_EXTRA
        )

    def test_types_explicitly_select_convention_and_representation(self):
        types = importlib.import_module("quatops.types")
        assert not hasattr(types, "Quaternion")
        assert types.HamiltonQuaternion is importlib.import_module("quatops.hamilton.dataclass").Quaternion
        assert types.JPLQuaternion is importlib.import_module("quatops.jpl.dataclass").Quaternion
        for convention in ("hamilton", "jpl"):
            for representation in ("namedtuple", "dataclass"):
                alias = importlib.import_module(
                    f"quatops.types.{convention}_{representation}"
                )
                target = importlib.import_module(
                    f"quatops.{convention}.{representation}"
                )
                for name in ("Quaternion", "AxisAngle", "RotVec"):
                    assert getattr(alias, name) is getattr(target, name)

    def test_scalar_backend_first_imports_do_not_load_numpy_or_plainquat(self):
        script = """\
import sys
from quatops import python, types
from quatops.python import hamilton, jpl
import quatops.numpy as numpy_namespace
from quatops.types import hamilton_namedtuple, hamilton_dataclass, jpl_namedtuple, jpl_dataclass
assert not hasattr(python, 'multiply')
assert not hasattr(numpy_namespace, 'multiply')
assert jpl.rotate((0, 0, 0, 1), (1, 2, 3)) == (1, 2, 3)
assert 'numpy' not in sys.modules
assert not any(name == 'plainquat' or name.startswith('plainquat.') for name in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    def test_numpy_aliases_are_the_four_native_kernels(self):
        numpy_namespace = importlib.import_module("quatops.numpy")
        assert not hasattr(numpy_namespace, "multiply")
        for convention in ("hamilton", "jpl"):
            for order in ("wxyz", "xyzw"):
                operations = COMMON + (JPL_EXTRA if convention == "jpl" else ())
                self.assert_direct_exports(
                    f"quatops.numpy.{convention}.{order}",
                    f"quatops.{convention}.numpy_{order}",
                    operations,
                )
