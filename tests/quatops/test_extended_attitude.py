"""Cross-representation coverage for the extended quaternion and attitude APIs."""

import importlib
import itertools
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

SCALAR_APIS = (
    "quatops.scalar",
    "quatops.python.hamilton",
    "quatops.hamilton.scalar",
    "quatops.namedtuple",
    "quatops.hamilton.namedtuple",
    "quatops.types.hamilton_namedtuple",
    "quatops.dataclass",
    "quatops.hamilton.dataclass",
    "quatops.types.hamilton_dataclass",
    "quatops.jpl.scalar",
    "quatops.python.jpl",
    "quatops.jpl.namedtuple",
    "quatops.types.jpl_namedtuple",
    "quatops.jpl.dataclass",
    "quatops.types.jpl_dataclass",
)
NUMPY_APIS = tuple(
    f"quatops.numpy.{convention}{'.components_first' if component_first else ''}.{order}"
    for convention in ("hamilton", "jpl")
    for order in ("wxyz", "xyzw")
    for component_first in (False, True)
)
EULER_SEQUENCES = (
    tuple("".join(axes) for axes in itertools.permutations("xyz"))
    + tuple("".join(axes).upper() for axes in itertools.permutations("xyz"))
    + ("xyx", "xzx", "yxy", "yzy", "zxz", "zyz")
    + ("XYX", "XZX", "YXY", "YZY", "ZXZ", "ZYZ")
)


def _tuple(value):
    return tuple(value)


def _array(value):
    """Convert immutable NamedTuple/dataclass API results to numeric arrays."""
    if isinstance(value, np.ndarray):
        return value
    if np.isscalar(value):
        return np.asarray(value)
    return np.asarray(tuple(value), dtype=np.float64)


def _module(path):
    return importlib.import_module(path)


def _component_first(path):
    return ".components_first." in path


def _to_first(value):
    return np.moveaxis(np.asarray(value), -1, 0)


def _to_last(value):
    return np.moveaxis(np.asarray(value), 0, -1)


@pytest.mark.parametrize("path", SCALAR_APIS)
def test_extended_algebra_functions_and_extraction(path):
    api = _module(path)
    q = api.from_wxyz((0.35, 0.12, -0.08, 0.17))
    exponent = api.from_wxyz((0.07, -0.2, 0.11, 0.03))

    np.testing.assert_allclose(_array(api.sin(api.asin(q))), _array(q), rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(_array(api.cos(api.acos(q))), _array(q), rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(_array(api.tan(api.atan(q))), _array(q), rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(_array(api.sinh(api.asinh(q))), _array(q), rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(_array(api.cosh(api.acosh(q))), _array(q), rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(_array(api.tanh(api.atanh(q))), _array(q), rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(_array(api.exp(api.log1p(q))),
                               _array(api.add(q, api.from_wxyz((1, 0, 0, 0)))),
                               rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(_array(api.log2(q)), _array(api.scale(api.log(q), 1 / math.log(2))),
                               rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(_array(api.log10(q)), _array(api.scale(api.log(q), 1 / math.log(10))),
                               rtol=2e-14, atol=2e-14)

    # The compensated small-argument functions retain values lost by direct
    # subtraction of exp(q)-1 or log(1+q).
    tiny = api.from_wxyz((1e-12, 2e-13, 0, 0))
    expm1 = api.expm1(tiny)
    log1p = api.log1p(tiny)
    tiny_wxyz = api.to_wxyz(tiny)
    expm1_wxyz = api.to_wxyz(expm1)
    log1p_wxyz = api.to_wxyz(log1p)
    assert expm1_wxyz[0] == pytest.approx(math.expm1(tiny_wxyz[0]), rel=2e-15)
    assert log1p_wxyz[0] == pytest.approx(math.log1p(tiny_wxyz[0]), rel=2e-15)
    assert expm1_wxyz[1] != 0.0
    assert log1p_wxyz[1] != 0.0

    expected_power = api.exp(api.multiply(exponent, api.log(q)))
    np.testing.assert_allclose(_array(api.power_quaternion(q, exponent)), _array(expected_power),
                               rtol=2e-14, atol=2e-14)
    lhs = api.from_wxyz((0.8, 0.1, 0.2, -0.3))
    rhs = api.from_wxyz((0.7, -0.4, 0.2, 0.1))
    expected_commutator = api.subtract(api.multiply(lhs, rhs), api.multiply(rhs, lhs))
    np.testing.assert_allclose(_array(api.commutator(lhs, rhs)), _array(expected_commutator), atol=0)

    native = api.to_wxyz(q)
    assert api.scalar_part(q) == pytest.approx(native[0])
    np.testing.assert_allclose(_array(api.vector_part(q)), native[1:])


@pytest.mark.parametrize("path", SCALAR_APIS)
@pytest.mark.parametrize("sequence", EULER_SEQUENCES)
def test_euler_conversions_match_scipy(path, sequence):
    api = _module(path)
    angles = np.array((0.43, -0.61, 0.27))
    expected = Rotation.from_euler(sequence, angles)
    q = api.from_euler(angles, sequence)
    np.testing.assert_allclose(api.to_rotation_matrix(q), expected.as_matrix(),
                               rtol=2e-14, atol=2e-14)
    recovered_angles = api.to_euler(q, sequence)
    recovered = Rotation.from_euler(sequence, _tuple(recovered_angles))
    np.testing.assert_allclose(recovered.as_matrix(), expected.as_matrix(),
                               rtol=3e-14, atol=3e-14)

    middle = 0.0 if sequence[0].lower() == sequence[2].lower() else math.pi / 2
    gimbal_angles = (0.3, middle, -0.2)
    gimbal = Rotation.from_euler(sequence, gimbal_angles)
    gimbal_q = api.from_euler(gimbal_angles, sequence)
    extracted = _tuple(api.to_euler(gimbal_q, sequence))
    assert extracted[2] == 0.0
    np.testing.assert_allclose(Rotation.from_euler(sequence, extracted).as_matrix(),
                               gimbal.as_matrix(), atol=3e-14)


@pytest.mark.parametrize("path", SCALAR_APIS)
def test_rotation_manifold_operations_and_factorizations(path):
    api = _module(path)
    is_jpl = ".jpl" in path
    q = api.from_rotation_vector((0.31, -0.22, 0.47))
    q2 = api.from_rotation_vector((-0.17, 0.29, 0.08))

    for frame in ("body", "world"):
        delta = (0.04, -0.03, 0.02)
        updated = api.boxplus(q, delta, frame=frame)
        np.testing.assert_allclose(_array(api.boxminus(updated, q, frame=frame)), delta,
                                   rtol=2e-13, atol=2e-13)
        omega = (0.3, -0.1, 0.2)
        derivative = api.quaternion_derivative(q, omega, frame=frame)
        coefficient_rate = tuple(-item for item in omega) if is_jpl else omega
        pure_rate = api.from_wxyz((0.0, *coefficient_rate))
        product = (api.multiply(q, pure_rate) if frame == "body"
                   else api.multiply(pure_rate, q))
        np.testing.assert_allclose(_array(derivative), _array(api.scale(product, 0.5)), atol=0)
        step = 1e-7
        next_attitude = api.integrate_angular_velocity(q, omega, step, frame=frame)
        finite_difference = (_array(next_attitude) - _array(q)) / step
        np.testing.assert_allclose(finite_difference, _array(derivative),
                                   rtol=2e-7, atol=2e-9)
        integrated = api.integrate_angular_velocity_samples(
            q, (omega, omega, omega), (0.0, 0.1, 0.3), frame=frame, method="midpoint"
        )
        constant = api.integrate_angular_velocity(q, omega, 0.3, frame=frame)
        assert api.rotations_equivalent(integrated, constant, atol=2e-12)
        ramp = api.integrate_angular_velocity_samples(
            q, ((0.0, 0.0, 0.0), (0.0, 0.0, 2.0)), (0.0, 1.0),
            frame=frame, method="midpoint",
        )
        expected_ramp = api.integrate_angular_velocity(q, (0.0, 0.0, 1.0), 1.0,
                                                       frame=frame)
        assert api.rotations_equivalent(ramp, expected_ramp, atol=2e-12)

    a = api.from_rotation_vector((0.1, 0.0, 0.0))
    b = api.from_rotation_vector((0.0, 0.2, 0.0))
    assert api.rotations_equivalent(api.nlerp(a, b, 0.0), a, atol=1e-14)
    assert api.rotations_equivalent(api.nlerp(a, b, 1.0), b, atol=1e-14)
    assert api.norm(api.nlerp(a, b, 0.4)) == pytest.approx(1.0)
    assert api.rotations_equivalent(api.squad(a, b, a, b, 0.0), a, atol=1e-14)
    assert api.rotations_equivalent(api.squad(a, b, a, b, 1.0), b, atol=1e-14)

    sequence = (a, tuple(-x for x in b), tuple(-x for x in q2))
    continuous = api.make_sign_continuous(sequence)
    assert all(sum(x * y for x, y in zip(left, right)) >= -1e-15
               for left, right in zip(continuous, continuous[1:]))
    assert _tuple(api.align_sign(tuple(-x for x in a), a)) == pytest.approx(_tuple(a))

    swing, twist = api.swing_twist(tuple(-x for x in q), (0.0, 0.0, 1.0))
    recomposed = api.multiply(swing, twist)
    np.testing.assert_allclose(_array(recomposed), _array(api.normalize(tuple(-x for x in q))),
                               rtol=3e-14, atol=3e-14)
    degenerate_swing, degenerate_twist = api.swing_twist(
        api.from_rotation_vector((math.pi, 0, 0)), (0, 0, 1)
    )
    assert api.rotations_equivalent(degenerate_twist, api.from_wxyz((1, 0, 0, 0)), atol=1e-14)
    assert api.rotations_equivalent(degenerate_swing,
                                    api.from_rotation_vector((math.pi, 0, 0)), atol=1e-14)


@pytest.mark.parametrize("path", SCALAR_APIS)
def test_euler_mrp_homogeneous_and_adjoint_conversions(path):
    api = _module(path)
    rotation = Rotation.from_rotvec((0.35, -0.21, 0.16))
    q = api.from_rotation_vector(rotation.as_rotvec())

    for sigma in ((0.12, -0.07, 0.19), (1.3, -0.4, 0.2)):
        from_mrp = api.from_mrp(sigma)
        back = api.to_mrp(from_mrp)
        assert np.linalg.norm(_array(back)) <= 1.0 + 1e-14
        assert api.rotations_equivalent(from_mrp, api.from_mrp(back), atol=2e-13)

    translation = (1.25, -2.0, 0.5)
    matrix = np.asarray(api.to_homogeneous_matrix(q, translation))
    np.testing.assert_allclose(matrix[:3, :3], rotation.as_matrix(), atol=2e-14)
    np.testing.assert_array_equal(matrix[:3, 3], translation)
    np.testing.assert_array_equal(matrix[3], (0, 0, 0, 1))
    recovered = api.from_homogeneous_matrix(matrix)
    assert api.rotations_equivalent(q, recovered, atol=2e-14)
    np.testing.assert_allclose(api.so3_adjoint(q), rotation.as_matrix(), atol=2e-14)

    invalid = matrix.copy()
    invalid[3, 0] = 1e-3
    with pytest.raises(ValueError):
        api.from_homogeneous_matrix(invalid)


@pytest.mark.parametrize("path", SCALAR_APIS)
def test_markley_intrinsic_covariance_and_wahba_reference(path):
    api = _module(path)
    rng = np.random.default_rng(1117)
    center = Rotation.from_rotvec((0.3, -0.2, 0.1))
    tangent = rng.normal(scale=0.035, size=(7, 3))
    rotations = Rotation.from_rotvec(tangent) * center
    samples = tuple(api.from_rotation_vector(item.as_rotvec()) for item in rotations)
    weights = np.array((1.0, 2.0, 0.5, 3.0, 0.7, 1.2, 0.9))

    mean = api.rotation_mean(samples, weights)
    reference_mean = rotations.mean(weights).as_matrix()
    np.testing.assert_allclose(api.to_rotation_matrix(mean), reference_mean,
                               rtol=2e-13, atol=2e-13)
    mean_small_weights = api.rotation_mean(samples, weights * 1e-100)
    mean_large_weights = api.rotation_mean(samples, weights * 1e100)
    assert api.rotations_equivalent(mean, mean_small_weights, atol=1e-13)
    assert api.rotations_equivalent(mean, mean_large_weights, atol=1e-13)

    intrinsic = api.rotation_mean(samples, weights, method="intrinsic")
    residuals = np.array([_array(api.boxminus(sample, intrinsic)) for sample in samples])
    weighted_residual = np.average(residuals, axis=0, weights=weights)
    np.testing.assert_allclose(weighted_residual, 0.0, atol=2e-12)
    covariance = np.asarray(api.rotation_covariance(samples, intrinsic, weights))
    covariance_scaled = np.asarray(api.rotation_covariance(
        samples, intrinsic, weights * 1e-100
    ))
    np.testing.assert_allclose(covariance_scaled, covariance, rtol=2e-14, atol=2e-15)
    default_covariance = np.asarray(api.rotation_covariance(samples, weights=weights))
    explicit_covariance = np.asarray(api.rotation_covariance(samples, mean, weights))
    np.testing.assert_allclose(default_covariance, explicit_covariance,
                               rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(covariance, covariance.T, atol=2e-15)
    assert np.linalg.eigvalsh(covariance).min() >= -2e-15
    mean_matrix = np.asarray(api.to_rotation_matrix(intrinsic))
    world_covariance = np.asarray(api.rotation_covariance(
        samples, intrinsic, weights, frame="world"
    ))
    np.testing.assert_allclose(world_covariance,
                               mean_matrix @ covariance @ mean_matrix.T,
                               rtol=2e-12, atol=2e-14)

    source = rng.normal(size=(8, 3))
    truth = Rotation.from_rotvec((0.7, -0.3, 0.4))
    target = truth.apply(source) + rng.normal(scale=0.008, size=source.shape)
    vector_weights = np.linspace(0.2, 2.0, len(source))
    fit = api.align_vectors(source, target, weights=vector_weights)
    scipy_fit, _ = Rotation.align_vectors(target, source, weights=vector_weights)
    np.testing.assert_allclose(api.to_rotation_matrix(fit), scipy_fit.as_matrix(),
                               rtol=3e-13, atol=3e-13)
    tiny_weight_fit = api.align_vectors(source, target, weights=vector_weights * 1e-200)
    assert api.rotations_equivalent(fit, tiny_weight_fit, atol=2e-13)
    rescaled_fit = api.align_vectors(source * 1e200, target * 1e-200,
                                     weights=vector_weights * 1e100)
    assert api.rotations_equivalent(fit, rescaled_fit, atol=2e-13)

    with pytest.raises(ValueError):
        api.align_vectors(((1, 0, 0),), ((0, 1, 0),))
    with pytest.raises(ValueError):
        api.rotation_mean(samples, np.zeros(len(samples)))


@pytest.mark.parametrize("path", SCALAR_APIS)
def test_random_attitudes_are_deterministic_and_uniform_on_s3(path):
    api = _module(path)
    first_rng, second_rng = np.random.default_rng(42), np.random.default_rng(42)
    first = api.random_quaternion(first_rng)
    second = api.random_rotation(second_rng)
    np.testing.assert_array_equal(_array(first), _array(second))
    assert api.norm(first) == pytest.approx(1.0, abs=2e-15)

    samples = np.array([_array(api.random_quaternion(np.random.default_rng(seed)))
                        for seed in range(2048)])
    native = np.array([api.to_wxyz(sample) for sample in samples])
    second_moment = native.T @ native / len(native)
    np.testing.assert_allclose(second_moment, np.eye(4) / 4, atol=0.035, rtol=0)


@pytest.mark.parametrize("path", NUMPY_APIS)
def test_numpy_extended_batch_apis_across_all_layouts(path):
    api = _module(path)
    first = _component_first(path)

    angles = np.array(((0.3, -0.1, 0.4), (-0.2, 0.25, 0.1)))
    if first:
        angles = _to_first(angles)
    q = api.from_euler(angles, "zyx")
    extracted = api.to_euler(q, "zyx")
    assert q.shape == ((4, 2) if first else (2, 4))
    assert extracted.shape == ((3, 2) if first else (2, 3))

    sigma = np.array(((0.1, -0.2, 0.05), (0.3, 0.04, -0.1)))
    if first:
        sigma = _to_first(sigma)
    q_mrp = api.from_mrp(sigma)
    assert api.to_mrp(q_mrp).shape == ((3, 2) if first else (2, 3))

    raw = np.stack((q, api.from_rotation_vector(
        _to_first(np.array(((0.05, 0.02, -0.03), (0.03, -0.01, 0.04)))) if first
        else np.array(((0.05, 0.02, -0.03), (0.03, -0.01, 0.04)))
    )), axis=-1 if first else 1)
    # Above produces (4, batch, samples) for component-first and
    # (batch, samples, 4) for the standard layout.
    assert raw.shape == ((4, 2, 2) if first else (2, 2, 4))
    weights = np.array((1.0, 2.0))
    mean = api.rotation_mean(raw, weights)
    covariance = api.rotation_covariance(raw, mean, weights)
    assert mean.shape == ((4, 2) if first else (2, 4))
    assert covariance.shape == ((3, 3, 2) if first else (2, 3, 3))

    source = np.array((((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                       ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))))
    target = np.array((((0., 1., 0.), (-1., 0., 0.), (0., 0., 1.)),
                       ((0., 0., 1.), (0., 1., 0.), (-1., 0., 0.))))
    if first:
        source, target = _to_first(source), _to_first(target)
    aligned = api.align_vectors(source, target)
    assert aligned.shape == ((4, 2) if first else (2, 4))

    rate_samples = np.broadcast_to(np.array((0.0, 0.0, 0.2)), (2, 3, 3)).copy()
    times = np.array((0.0, 0.1, 0.25))
    if first:
        rate_samples = _to_first(rate_samples)
    integrated = api.integrate_angular_velocity_samples(q, rate_samples, times)
    assert integrated.shape == q.shape

    continuous = api.make_sign_continuous(raw)
    assert continuous.shape == raw.shape
    swing, twist = api.swing_twist(q, (0.0, 0.0, 1.0))
    assert swing.shape == q.shape and twist.shape == q.shape

    random = api.random_quaternion(size=(2, 3), rng=np.random.default_rng(82))
    assert random.shape == ((4, 2, 3) if first else (2, 3, 4))
    np.testing.assert_allclose(api.norm(random), 1.0, atol=2e-15)
    with pytest.raises(ValueError):
        api.random_quaternion(size=(2.5,), rng=np.random.default_rng(1))

    translation = np.array((1.0, -2.0, 0.25))
    homogeneous = api.to_homogeneous_matrix(q, translation)
    assert homogeneous.shape == ((4, 4, 2) if first else (2, 4, 4))
    recovered = api.from_homogeneous_matrix(homogeneous)
    assert recovered.shape == q.shape
    np.testing.assert_allclose(api.rotation_distance(q, recovered), 0.0, atol=2e-14)

    assert api.power_quaternion(q, api.from_wxyz((0.2, -0.1, 0.05, 0.1))).shape == q.shape
    assert api.from_wxyz((0.5, 0.1, 0.0, -0.2)).shape == (4,)


@pytest.mark.parametrize("path", NUMPY_APIS)
def test_numpy_mixed_component_width_operations_broadcast(path):
    api = _module(path)
    first = _component_first(path)
    trailing = _module(path.replace(".components_first", ""))
    q_last = trailing.from_rotation_vector(np.array((((0.1, 0.0, 0.0),),
                                                     ((0.0, 0.2, 0.0),))))
    delta_last = np.array((((0.0, 0.0, 0.03), (0.01, 0.02, 0.0), (-0.02, 0.0, 0.01)),))
    rate_last = np.array((((0.1, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.3)),))
    if first:
        q, delta, rate = _to_first(q_last), _to_first(delta_last), _to_first(rate_last)
    else:
        q, delta, rate = q_last, delta_last, rate_last

    updated = api.boxplus(q, delta)
    derivative = api.quaternion_derivative(q, rate)
    swing, twist = api.swing_twist(q, rate)
    homogeneous = api.to_homogeneous_matrix(q, rate)
    assert updated.shape == ((4, 2, 3) if first else (2, 3, 4))
    assert derivative.shape == updated.shape
    assert swing.shape == updated.shape and twist.shape == updated.shape
    assert homogeneous.shape == ((4, 4, 2, 3) if first else (2, 3, 4, 4))

    recovered = api.boxminus(updated, q)
    expected_delta = np.broadcast_to(delta_last, (2, 3, 3))
    if first:
        recovered = _to_last(recovered)
    np.testing.assert_allclose(recovered, expected_delta, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("trailing_path", tuple(
    f"quatops.numpy.{convention}.{order}"
    for convention in ("hamilton", "jpl") for order in ("wxyz", "xyzw")
))
def test_component_first_advanced_outputs_match_trailing_layout(trailing_path):
    trailing = _module(trailing_path)
    leading = _module(trailing_path.replace(
        ".wxyz", ".components_first.wxyz").replace(".xyzw", ".components_first.xyzw"
    ))
    q_last = trailing.from_euler(np.array(((0.1, 0.2, -0.3), (0.2, -0.1, 0.25))), "XYZ")
    rates_last = np.broadcast_to(np.array((0.1, -0.05, 0.2)), (2, 3, 3)).copy()
    times = np.array((0.0, 0.1, 0.2))

    pairs = (
        ("to_euler", (q_last,), {"sequence": "zyx"}),
        ("to_mrp", (q_last,), {}),
        ("to_homogeneous_matrix", (q_last,), {}),
        ("so3_adjoint", (q_last,), {}),
        ("rotation_mean", (np.stack((q_last, q_last), axis=1),), {}),
        ("rotation_covariance", (np.stack((q_last, q_last), axis=1),), {}),
        ("align_vectors", (np.broadcast_to(np.eye(3), (2, 3, 3)),
                           np.broadcast_to(np.array(((0., -1., 0.), (1., 0., 0.),
                                                     (0., 0., 1.))), (2, 3, 3))), {}),
        ("integrate_angular_velocity_samples", (q_last, rates_last, times), {}),
    )
    for name, args, kwargs in pairs:
        expected = getattr(trailing, name)(*args, **kwargs)
        converted = tuple(_to_first(item) if isinstance(item, np.ndarray) and item.ndim else item
                          for item in args)
        actual = getattr(leading, name)(*converted, **kwargs)
        if name in ("to_homogeneous_matrix", "so3_adjoint", "rotation_covariance"):
            actual_last = np.moveaxis(actual, (0, 1), (-2, -1))
        elif name in ("to_euler", "to_mrp") or name in (
            "rotation_mean",
            "align_vectors",
            "integrate_angular_velocity_samples",
        ):
            actual_last = _to_last(actual)
        else:
            actual_last = actual
        np.testing.assert_allclose(actual_last, expected, rtol=2e-14, atol=2e-14)
