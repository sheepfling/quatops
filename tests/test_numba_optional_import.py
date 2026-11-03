"""Keep optional Numba import and compilation failures on the NumPy path."""

import builtins
import sys
from types import ModuleType

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from quatops import _numpy
from quatops.engines import SignedAxisMap
from quatops.engines import numpy as engine_numpy
from quatops.numpy.hamilton import wxyz, xyzw

COUNT = 128


@pytest.fixture
def reset_numba_state(monkeypatch):
    monkeypatch.setattr(_numpy, "_NUMBA_MULTIPLY_CHECKED", False)
    monkeypatch.setattr(_numpy, "_NUMBA_MULTIPLY", {})
    monkeypatch.setattr(_numpy, "_NUMBA_CHECKED", False)
    monkeypatch.setattr(_numpy, "_NUMBA_ROTATE", None)
    monkeypatch.setattr(_numpy, "_NUMBA_ROTATION_CHECKED", False)
    monkeypatch.setattr(_numpy, "_NUMBA_ROTATION_KERNELS", {})


def _identity_inputs():
    scalar_first = np.zeros((COUNT, 4))
    scalar_first[:, 0] = 1.0
    scalar_last = np.zeros((COUNT, 4))
    scalar_last[:, 3] = 1.0
    vectors = np.tile((1.0, -2.0, 3.0), (COUNT, 1))
    return scalar_first, scalar_last, vectors


def _assert_numpy_operations():
    from quatops.jpl import numpy_wxyz, numpy_xyzw
    from quatops.jpl import scalar as jpl_scalar

    scalar_first, scalar_last, vectors = _identity_inputs()
    axis_map = SignedAxisMap(("y", "-z", "x"))
    assert_array_equal(
        engine_numpy.map_hamilton_xyzw(scalar_last, axis_map), scalar_last
    )
    assert_array_equal(wxyz.multiply(scalar_first, scalar_first), scalar_first)
    assert_array_equal(xyzw.multiply(scalar_last, scalar_last), scalar_last)
    assert_array_equal(wxyz.normalize(scalar_first), scalar_first)
    assert_array_equal(xyzw.normalize(scalar_last), scalar_last)
    assert_array_equal(numpy_wxyz.normalize(scalar_first), scalar_first)
    assert_array_equal(numpy_xyzw.normalize(scalar_last), scalar_last)
    assert_array_equal(wxyz.inverse(scalar_first), scalar_first)
    assert_array_equal(xyzw.inverse(scalar_last), scalar_last)
    assert_array_equal(numpy_wxyz.inverse(scalar_first), scalar_first)
    assert_array_equal(numpy_xyzw.inverse(scalar_last), scalar_last)
    assert_array_equal(wxyz.norm(scalar_first), np.ones(COUNT))
    assert_array_equal(xyzw.norm(scalar_last), np.ones(COUNT))
    assert_array_equal(numpy_wxyz.norm(scalar_first), np.ones(COUNT))
    assert_array_equal(numpy_xyzw.norm(scalar_last), np.ones(COUNT))
    assert_array_equal(wxyz.dot(scalar_first, scalar_first), np.ones(COUNT))
    assert_array_equal(xyzw.dot(scalar_last, scalar_last), np.ones(COUNT))
    assert_array_equal(numpy_wxyz.dot(scalar_first, scalar_first), np.ones(COUNT))
    assert_array_equal(numpy_xyzw.dot(scalar_last, scalar_last), np.ones(COUNT))
    for backend, identity in (
        (wxyz, scalar_first),
        (xyzw, scalar_last),
        (numpy_wxyz, scalar_first),
        (numpy_xyzw, scalar_last),
    ):
        assert_array_equal(backend.add(identity, identity), 2.0 * identity)
        assert_array_equal(backend.subtract(identity, identity), np.zeros_like(identity))
        assert_array_equal(backend.conjugate(identity), identity)
        assert_array_equal(backend.negate(identity), -identity)
        assert_array_equal(backend.scale(identity, 1.0), identity)
        assert_array_equal(backend.scalar_part(identity), np.ones(COUNT))
        assert_array_equal(backend.vector_part(identity), np.zeros((COUNT, 3)))
        assert_array_equal(backend.canonicalize(identity), identity)
        assert_array_equal(backend.commutator(identity, identity), np.zeros_like(identity))
        assert_array_equal(backend.from_two_vectors(vectors, vectors), identity)
        assert_array_equal(backend.left_divide(identity, identity), identity)
        assert_array_equal(backend.right_divide(identity, identity), identity)
        assert_array_equal(backend.rotation_distance(identity, identity), np.zeros(COUNT))
        assert_array_equal(
            backend.rotations_equivalent(identity, identity), np.ones(COUNT, dtype=bool)
        )
        multiplication_identity = np.tile(np.eye(4), (COUNT, 1, 1))
        assert_array_equal(backend.to_left_matrix(identity), multiplication_identity)
        assert_array_equal(backend.to_right_matrix(identity), multiplication_identity)
    assert_array_equal(wxyz.rotate(scalar_first, vectors), vectors)
    assert_array_equal(xyzw.rotate(scalar_last, vectors), vectors)
    matrices = np.tile(np.eye(3), (COUNT, 1, 1))
    assert_array_equal(wxyz.to_rotation_matrix(scalar_first), matrices)
    assert_array_equal(xyzw.to_rotation_matrix(scalar_last), matrices)
    assert_array_equal(wxyz.from_rotation_matrix(matrices), scalar_first)
    assert_array_equal(xyzw.from_rotation_matrix(matrices), scalar_last)
    angles = np.zeros((COUNT, 3))
    assert_array_equal(wxyz.from_euler(angles), scalar_first)
    assert_array_equal(xyzw.from_euler(angles), scalar_last)
    assert_array_equal(wxyz.to_euler(scalar_first), angles)
    assert_array_equal(xyzw.to_euler(scalar_last), angles)
    rotation_vectors = np.zeros((COUNT, 3))
    assert_array_equal(wxyz.from_rotvec(rotation_vectors), scalar_first)
    assert_array_equal(xyzw.from_rotvec(rotation_vectors), scalar_last)
    assert_array_equal(wxyz.to_rotvec(scalar_first), rotation_vectors)
    assert_array_equal(xyzw.to_rotvec(scalar_last), rotation_vectors)
    jpl_identity_xyzw = jpl_scalar.from_euler((0.0, 0.0, 0.0))
    jpl_identity_wxyz = (jpl_identity_xyzw[3], *jpl_identity_xyzw[:3])
    assert_array_equal(
        numpy_wxyz.from_euler(angles), np.tile(jpl_identity_wxyz, (COUNT, 1))
    )
    assert_array_equal(
        numpy_xyzw.from_euler(angles), np.tile(jpl_identity_xyzw, (COUNT, 1))
    )
    assert_array_equal(numpy_wxyz.to_euler(scalar_first), angles)
    assert_array_equal(numpy_xyzw.to_euler(scalar_last), angles)
    assert_array_equal(numpy_wxyz.from_rotvec(rotation_vectors), scalar_first)
    assert_array_equal(numpy_xyzw.from_rotvec(rotation_vectors), scalar_last)
    assert_array_equal(numpy_wxyz.to_rotvec(scalar_first), rotation_vectors)
    assert_array_equal(numpy_xyzw.to_rotvec(scalar_last), rotation_vectors)
    assert_array_equal(wxyz.from_xyzw(scalar_last), scalar_first)
    assert_array_equal(wxyz.to_xyzw(scalar_first), scalar_last)
    assert_array_equal(xyzw.from_wxyz(scalar_first), scalar_last)
    assert_array_equal(xyzw.to_wxyz(scalar_last), scalar_first)
    assert_array_equal(numpy_wxyz.from_xyzw(scalar_last), scalar_first)
    assert_array_equal(numpy_wxyz.to_xyzw(scalar_first), scalar_last)
    assert_array_equal(numpy_xyzw.from_wxyz(scalar_first), scalar_last)
    assert_array_equal(numpy_xyzw.to_wxyz(scalar_last), scalar_first)
    assert_array_equal(numpy_wxyz.from_hamilton(scalar_first), scalar_first * (1, -1, -1, -1))
    assert_array_equal(numpy_wxyz.to_hamilton(scalar_first), scalar_first * (1, -1, -1, -1))
    assert_array_equal(numpy_xyzw.from_hamilton(scalar_first), scalar_last * (-1, -1, -1, 1))
    assert_array_equal(numpy_xyzw.to_hamilton(scalar_last), scalar_first * (1, -1, -1, -1))
    random_h = wxyz.random_quaternion(size=COUNT, rng=np.random.default_rng(83))
    random_h_again = wxyz.random_rotation(size=COUNT, rng=np.random.default_rng(83))
    assert_array_equal(random_h, random_h_again)
    np.testing.assert_allclose(wxyz.norm(random_h), np.ones(COUNT), atol=3e-15, rtol=0)
    assert_array_equal(wxyz.nlerp(scalar_first, scalar_first, 0.5), scalar_first)
    assert_array_equal(xyzw.nlerp(scalar_last, scalar_last, 0.5), scalar_last)
    assert_array_equal(numpy_wxyz.nlerp(scalar_first, scalar_first, 0.5), scalar_first)
    assert_array_equal(numpy_xyzw.nlerp(scalar_last, scalar_last, 0.5), scalar_last)
    assert_array_equal(wxyz.slerp(scalar_first, scalar_first, 0.5), scalar_first)
    assert_array_equal(xyzw.slerp(scalar_last, scalar_last, 0.5), scalar_last)
    assert_array_equal(numpy_wxyz.slerp(scalar_first, scalar_first, 0.5), scalar_first)
    assert_array_equal(numpy_xyzw.slerp(scalar_last, scalar_last, 0.5), scalar_last)
    mrps = np.zeros((COUNT, 3))
    assert_array_equal(wxyz.from_mrp(mrps), scalar_first)
    assert_array_equal(xyzw.from_mrp(mrps), scalar_last)
    assert_array_equal(numpy_wxyz.from_mrp(mrps), scalar_first)
    assert_array_equal(numpy_xyzw.from_mrp(mrps), scalar_last)
    assert_array_equal(wxyz.to_mrp(scalar_first), mrps)
    assert_array_equal(xyzw.to_mrp(scalar_last), mrps)
    assert_array_equal(numpy_wxyz.to_mrp(scalar_first), mrps)
    assert_array_equal(numpy_xyzw.to_mrp(scalar_last), mrps)
    rates = np.zeros((COUNT, 2, 3))
    times = np.broadcast_to((0.0, 1.0), (COUNT, 2))
    assert_array_equal(
        wxyz.integrate_angular_velocity_samples(scalar_first, rates, times),
        scalar_first,
    )
    assert_array_equal(
        xyzw.integrate_angular_velocity_samples(scalar_last, rates, times),
        scalar_last,
    )
    assert_array_equal(
        numpy_wxyz.integrate_angular_velocity_samples(scalar_first, rates, times),
        scalar_first,
    )
    assert_array_equal(
        numpy_xyzw.integrate_angular_velocity_samples(scalar_last, rates, times),
        scalar_last,
    )
    zeros = np.zeros((COUNT, 3))
    assert_array_equal(wxyz.quaternion_derivative(scalar_first, zeros), np.zeros_like(scalar_first))
    assert_array_equal(xyzw.quaternion_derivative(scalar_last, zeros), np.zeros_like(scalar_last))
    assert_array_equal(numpy_wxyz.quaternion_derivative(scalar_first, zeros), np.zeros_like(scalar_first))
    assert_array_equal(numpy_xyzw.quaternion_derivative(scalar_last, zeros), np.zeros_like(scalar_last))
    assert_array_equal(wxyz.boxplus(scalar_first, zeros), scalar_first)
    assert_array_equal(xyzw.boxplus(scalar_last, zeros), scalar_last)
    assert_array_equal(numpy_wxyz.boxplus(scalar_first, zeros), scalar_first)
    assert_array_equal(numpy_xyzw.boxplus(scalar_last, zeros), scalar_last)
    assert_array_equal(wxyz.boxminus(scalar_first, scalar_first), zeros)
    assert_array_equal(xyzw.boxminus(scalar_last, scalar_last), zeros)
    assert_array_equal(numpy_wxyz.boxminus(scalar_first, scalar_first), zeros)
    assert_array_equal(numpy_xyzw.boxminus(scalar_last, scalar_last), zeros)
    assert_array_equal(
        wxyz.squad(scalar_first, scalar_first, scalar_first, scalar_first, 0.5),
        scalar_first,
    )
    assert_array_equal(
        xyzw.squad(scalar_last, scalar_last, scalar_last, scalar_last, 0.5),
        scalar_last,
    )
    assert_array_equal(
        numpy_wxyz.squad(scalar_first, scalar_first, scalar_first, scalar_first, 0.5),
        scalar_first,
    )
    assert_array_equal(
        numpy_xyzw.squad(scalar_last, scalar_last, scalar_last, scalar_last, 0.5),
        scalar_last,
    )
    homogeneous = np.tile(np.eye(4), (COUNT, 1, 1))
    assert_array_equal(wxyz.to_homogeneous_matrix(scalar_first), homogeneous)
    assert_array_equal(xyzw.to_homogeneous_matrix(scalar_last), homogeneous)
    assert_array_equal(numpy_wxyz.to_homogeneous_matrix(scalar_first), homogeneous)
    assert_array_equal(numpy_xyzw.to_homogeneous_matrix(scalar_last), homogeneous)
    assert_array_equal(wxyz.from_homogeneous_matrix(homogeneous), scalar_first)
    assert_array_equal(xyzw.from_homogeneous_matrix(homogeneous), scalar_last)
    assert_array_equal(numpy_wxyz.from_homogeneous_matrix(homogeneous), scalar_first)
    assert_array_equal(numpy_xyzw.from_homogeneous_matrix(homogeneous), scalar_last)


def test_missing_numba_is_a_quiet_numpy_fallback(
    monkeypatch, caplog, reset_numba_state
):
    original_import = builtins.__import__

    def missing_numba(name, *args, **kwargs):
        if name == "numba":
            error = ModuleNotFoundError("No module named 'numba'")
            error.name = "numba"
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_numba)
    angles = np.zeros((8192, 3))
    identity = np.tile((0.0, 0.0, 0.0, 1.0), (8192, 1))
    assert_array_equal(xyzw.from_euler(angles), identity)
    assert_array_equal(xyzw.to_euler(identity), angles)
    _assert_numpy_operations()

    assert not [record for record in caplog.records if record.name == "quatops._numpy"]


def test_acceleration_import_failures_fall_back_once(
    monkeypatch, caplog, reset_numba_state
):
    original_import = builtins.__import__
    attempts = []

    def broken_import(name, *args, **kwargs):
        if name == "numba":
            attempts.append(name)
            raise OSError("simulated missing Numba runtime library")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    _assert_numpy_operations()
    _assert_numpy_operations()

    assert len(attempts) == 3
    assert _numpy._NUMBA_MULTIPLY_CHECKED
    assert not _numpy._NUMBA_MULTIPLY
    assert _numpy._NUMBA_CHECKED
    assert _numpy._NUMBA_ROTATE is None
    assert _numpy._NUMBA_ROTATION_CHECKED
    assert not _numpy._NUMBA_ROTATION_KERNELS
    assert sum(record.name == "quatops._numpy" for record in caplog.records) == 3


def test_acceleration_registration_failures_fall_back_once(
    monkeypatch, caplog, reset_numba_state
):
    numba = ModuleType("numba")
    attempts = {"guvectorize": 0, "njit": 0}

    def fail_registration(name):
        def decorator(*args, **kwargs):
            attempts[name] += 1

            def fail(_implementation):
                raise RuntimeError("simulated incompatible Numba API")

            return fail

        return decorator

    numba.guvectorize = fail_registration("guvectorize")
    numba.njit = fail_registration("njit")
    monkeypatch.setitem(sys.modules, "numba", numba)

    _assert_numpy_operations()
    _assert_numpy_operations()

    assert attempts == {"guvectorize": 1, "njit": 2}
    assert _numpy._NUMBA_MULTIPLY_CHECKED
    assert not _numpy._NUMBA_MULTIPLY
    assert _numpy._NUMBA_CHECKED
    assert _numpy._NUMBA_ROTATE is None
    assert _numpy._NUMBA_ROTATION_CHECKED
    assert not _numpy._NUMBA_ROTATION_KERNELS
    assert sum(record.name == "quatops._numpy" for record in caplog.records) == 3


def test_acceleration_execution_failures_disable_the_dispatchers(
    monkeypatch, caplog, reset_numba_state
):
    attempts = {
        "multiply": 0,
        "rotate": 0,
        "from_matrix": 0,
        "to_matrix": 0,
        "engine_map": 0,
        "from_rotvec": 0,
        "to_rotvec": 0,
        "integrate": 0,
        "nlerp": 0,
        "slerp": 0,
        "from_mrp": 0,
        "to_mrp": 0,
        "normalize": 0,
        "inverse": 0,
        "norm": 0,
        "dot": 0,
        "quaternion_matrix": 0,
        "canonicalize": 0,
        "commutator": 0,
        "from_two_vectors": 0,
        "conjugate": 0,
        "component_transform": 0,
        "vector_part": 0,
        "random_quaternion": 0,
    }

    def fail(operation):
        def dispatch(*args, **kwargs):
            attempts[operation] += 1
            raise RuntimeError("simulated JIT compilation failure")

        return dispatch

    monkeypatch.setattr(_numpy, "_NUMBA_MULTIPLY_CHECKED", True)
    monkeypatch.setattr(_numpy, "_NUMBA_MULTIPLY", {(False, 1): fail("multiply")})
    monkeypatch.setattr(_numpy, "_NUMBA_CHECKED", True)
    monkeypatch.setattr(_numpy, "_NUMBA_ROTATE", fail("rotate"))
    monkeypatch.setattr(_numpy, "_NUMBA_ROTATION_CHECKED", True)
    monkeypatch.setattr(
        _numpy,
        "_NUMBA_ROTATION_KERNELS",
        {
            "from_rotation_matrix": fail("from_matrix"),
            "to_rotation_matrix": fail("to_matrix"),
            "quaternion_matrix": fail("quaternion_matrix"),
            "engine_axis_map": fail("engine_map"),
            "from_rotvec": fail("from_rotvec"),
            "to_rotvec": fail("to_rotvec"),
            "integrate_angular_velocity": fail("integrate"),
            "nlerp": fail("nlerp"),
            "slerp": fail("slerp"),
            "from_mrp": fail("from_mrp"),
            "to_mrp": fail("to_mrp"),
            "normalize": fail("normalize"),
            "inverse": fail("inverse"),
            "norm": fail("norm"),
            "dot": fail("dot"),
            "canonicalize": fail("canonicalize"),
            "commutator": fail("commutator"),
            "from_two_vectors": fail("from_two_vectors"),
            "conjugate": fail("conjugate"),
            "component_transform": fail("component_transform"),
            "vector_part": fail("vector_part"),
            "random_quaternion": fail("random_quaternion"),
        },
    )

    _assert_numpy_operations()
    _assert_numpy_operations()

    assert attempts == {
        "multiply": 1,
        "rotate": 1,
        "from_matrix": 1,
        "to_matrix": 1,
        "engine_map": 1,
        "from_rotvec": 1,
        "to_rotvec": 1,
        "integrate": 1,
        "nlerp": 1,
        "slerp": 1,
        "from_mrp": 1,
        "to_mrp": 1,
        "normalize": 1,
        "inverse": 1,
        "norm": 1,
        "dot": 1,
        "quaternion_matrix": 1,
        "canonicalize": 1,
        "commutator": 1,
        "from_two_vectors": 1,
        "conjugate": 1,
        "component_transform": 1,
        "vector_part": 1,
        "random_quaternion": 1,
    }
    assert not _numpy._NUMBA_MULTIPLY
    assert _numpy._NUMBA_ROTATE is None
    assert not _numpy._NUMBA_ROTATION_KERNELS
    assert sum(record.name == "quatops._numpy" for record in caplog.records) == 24


def test_euler_kernel_execution_failures_disable_dispatchers(
    monkeypatch, caplog, reset_numba_state
):
    attempts = {"from_euler": 0, "to_euler": 0}

    def fail(operation):
        def dispatch(*args, **kwargs):
            attempts[operation] += 1
            raise RuntimeError("simulated Euler kernel compilation failure")

        return dispatch

    monkeypatch.setattr(_numpy, "_NUMBA_ROTATION_CHECKED", True)
    monkeypatch.setattr(
        _numpy,
        "_NUMBA_ROTATION_KERNELS",
        {
            "from_euler": fail("from_euler"),
            "to_euler": fail("to_euler"),
        },
    )
    count = 8192
    angles = np.zeros((count, 3))
    identity = np.tile((0.0, 0.0, 0.0, 1.0), (count, 1))

    assert_array_equal(xyzw.from_euler(angles), identity)
    assert_array_equal(xyzw.to_euler(identity), angles)
    assert attempts == {"from_euler": 1, "to_euler": 1}
    assert not _numpy._NUMBA_ROTATION_KERNELS
    assert sum(record.name == "quatops._numpy" for record in caplog.records) == 2
