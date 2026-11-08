"""The experimental API is opt-in, compatible, and removable from the core."""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "src"


def _run(code: str, source: Path = SOURCE, *, no_site: bool = False) -> None:
    environment = dict(os.environ, PYTHONPATH=str(source))
    command = [sys.executable]
    if no_site:
        command.append("-S")
    result = subprocess.run(
        [*command, "-c", textwrap.dedent(code)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_import_does_not_load_experimental_or_optional_dependencies():
    _run(
        """
        import sys
        import quatops
        from quatops.engines import SignedAxisMap
        from quatops.engines import *
        from quatops.python import hamilton
        assert SignedAxisMap(("x", "y", "z")).map_hamilton_wxyz((1, 0, 0, 0)) == (0, 0, 0, 1)
        assert hamilton.multiply((1, 0, 0, 0), (1, 0, 0, 0)) == (1, 0, 0, 0)
        assert not any(name.startswith("quatops.experimental") for name in sys.modules)
        assert "numpy" not in sys.modules
        assert "pydantic" not in sys.modules
    """,
        no_site=True,
    )


def test_experimental_scalar_is_dependency_free_and_does_not_use_aliases():
    _run(
        """
        import sys
        from quatops.experimental.converters import OrientationConverter
        from quatops.experimental.semantics import PhysicalOrientation, CANONICAL_WORLD_RFU, CANONICAL_BODY_RFU
        world = CANONICAL_WORLD_RFU.bind("world", "world.reference")
        body = CANONICAL_BODY_RFU.bind("body", "body.reference")
        converter = OrientationConverter("unity.quaternion", world, body, world, body)
        orientation = PhysicalOrientation.from_wxyz((1, 0, 0, 0), world, body)
        assert converter.from_target(converter.to_target(orientation)) == orientation
        assert "numpy" not in sys.modules
        assert "pydantic" not in sys.modules
        assert "quatops.converters" not in sys.modules
        assert "quatops.semantics" not in sys.modules
    """,
        no_site=True,
    )


@pytest.mark.parametrize(
    "module",
    [
        "converters",
        "converters.frames",
        "converters.gateway",
        "converters.targets",
        "converters.numpy",
        "semantics",
        "semantics.frames",
        "semantics.conventions",
        "engines.blender",
        "engines.godot",
        "engines.unity",
        "engines.unreal",
    ],
)
def test_compatibility_exports_preserve_identity_and_warn(module):
    _run(f"""
        import importlib
        import warnings
        canonical = importlib.import_module("quatops.experimental.{module}")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", DeprecationWarning)
            old = importlib.import_module("quatops.{module}")
        assert any(item.category is DeprecationWarning for item in captured)
        for name in old.__all__:
            assert getattr(old, name) is getattr(canonical, name), name
    """)


def test_lazy_engine_exports_preserve_identity_and_pickle():
    _run("""
        import importlib
        import pickle
        import warnings
        from quatops.experimental import engines as canonical
        for module in ("quatops.engines", "quatops.engines.basis"):
            old = importlib.import_module(module)
            for name in ("OrthogonalBasisMap", "QuaternionFrameMap", "change_rotation_basis"):
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always", DeprecationWarning)
                    assert getattr(old, name) is getattr(canonical, name)
                assert any(item.category is DeprecationWarning for item in captured)
        from quatops.experimental.semantics import CANONICAL_WORLD_RFU
        frame = CANONICAL_WORLD_RFU.bind("world", "reference")
        assert type(pickle.loads(pickle.dumps(frame))) is type(frame)
        # Old module-qualified pickle class references still resolve.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert pickle.loads(b"cquatops.semantics.frames\\nCoordinateFrame\\n.") is type(frame)
    """)


def test_core_operates_when_experimental_package_is_physically_absent(tmp_path):
    shutil.copytree(
        SOURCE / "quatops",
        tmp_path / "quatops",
        ignore=shutil.ignore_patterns("experimental", "__pycache__", "*.nbc", "*.nbi"),
    )
    _run(
        """
        import importlib.util
        import sys
        import numpy as np
        from quatops.engines import SignedAxisMap
        from quatops.engines import numpy as engine_numpy
        from quatops.python import hamilton
        assert importlib.util.find_spec("quatops.experimental") is None
        axis_map = SignedAxisMap(("x", "y", "z"))
        np.testing.assert_array_equal(engine_numpy.map_hamilton_wxyz(np.array([[1., 0., 0., 0.]]), axis_map), [[0., 0., 0., 1.]])
        assert hamilton.multiply((1, 0, 0, 0), (1, 0, 0, 0)) == (1, 0, 0, 0)
        assert not any(name.startswith("quatops.experimental") for name in sys.modules)
    """,
        source=tmp_path,
    )
