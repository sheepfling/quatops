"""Backend-first imports re-export the selected native kernels directly."""

import importlib
import subprocess
import sys
import unittest

COMMON = (
    "from_xyzw",
    "to_xyzw",
    "from_wxyz",
    "to_wxyz",
    "conjugate",
    "multiply",
    "norm",
    "normalize",
    "inverse",
    "rotate",
    "from_axis_angle",
    "to_axis_angle",
    "from_rotvec",
    "to_rotvec",
)
JPL_EXTRA = ("from_hamilton", "to_hamilton")

try:
    import numpy as np
except ImportError:
    np = None


class BackendFirstTests(unittest.TestCase):
    def assertDirectExports(self, alias_name, target_name, operations):
        alias = importlib.import_module(alias_name)
        target = importlib.import_module(target_name)
        for name in operations:
            with self.subTest(alias=alias_name, operation=name):
                self.assertIs(getattr(alias, name), getattr(target, name))

    def test_python_aliases_are_the_native_scalar_functions(self):
        self.assertDirectExports("quatops.python", "quatops.hamilton.scalar", COMMON)
        self.assertDirectExports(
            "quatops.python.hamilton", "quatops.hamilton.scalar", COMMON
        )
        self.assertDirectExports(
            "quatops.python.jpl", "quatops.jpl.scalar", COMMON + JPL_EXTRA
        )

    def test_types_explicitly_select_convention_and_representation(self):
        types = importlib.import_module("quatops.types")
        self.assertFalse(hasattr(types, "Quaternion"))
        self.assertIs(
            types.HamiltonQuaternion,
            importlib.import_module("quatops.hamilton.dataclass").Quaternion,
        )
        self.assertIs(
            types.JPLQuaternion,
            importlib.import_module("quatops.jpl.dataclass").Quaternion,
        )
        for convention in ("hamilton", "jpl"):
            for representation in ("namedtuple", "dataclass"):
                alias = importlib.import_module(
                    f"quatops.types.{convention}_{representation}"
                )
                target = importlib.import_module(
                    f"quatops.{convention}.{representation}"
                )
                for name in ("Quaternion", "AxisAngle", "RotVec"):
                    with self.subTest(
                        convention=convention, representation=representation, name=name
                    ):
                        self.assertIs(getattr(alias, name), getattr(target, name))

    def test_scalar_backend_first_imports_do_not_load_numpy_or_plainquat(self):
        script = """\
import sys
from quatops import python, types
from quatops.python import hamilton, jpl
from quatops.types import hamilton_namedtuple, hamilton_dataclass, jpl_namedtuple, jpl_dataclass
assert python.multiply is hamilton.multiply
assert jpl.rotate((0, 0, 0, 1), (1, 2, 3)) == (1, 2, 3)
assert 'numpy' not in sys.modules
assert not any(name == 'plainquat' or name.startswith('plainquat.') for name in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipIf(np is None, "NumPy is not installed")
    def test_numpy_aliases_are_the_four_native_kernels(self):
        self.assertDirectExports("quatops.numpy", "quatops.hamilton.numpy_wxyz", COMMON)
        for convention in ("hamilton", "jpl"):
            for order in ("wxyz", "xyzw"):
                operations = COMMON + (JPL_EXTRA if convention == "jpl" else ())
                self.assertDirectExports(
                    f"quatops.numpy.{convention}_{order}",
                    f"quatops.{convention}.numpy_{order}",
                    operations,
                )


if __name__ == "__main__":
    unittest.main()
