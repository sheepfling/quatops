"""Independent SciPy and numpy-quaternion oracles for manifold operations."""

import numpy as np
import pytest
import quaternion
from scipy.spatial.transform import Rotation, Slerp

from tests.oracle_support import (
    BACKENDS,
    as_hamilton_coefficients,
    from_wxyz,
    orient_vectors,
    reference_coefficients,
    reference_quaternions,
    to_wxyz,
    vectors_to_trailing,
)


def _scipy_rotations(public_wxyz: np.ndarray, is_jpl: bool) -> Rotation:
    hamilton_wxyz = as_hamilton_coefficients(public_wxyz, is_jpl)
    return Rotation.from_quat(hamilton_wxyz[..., (1, 2, 3, 0)])


def _wxyz_values(module, q, is_numpy: bool, component_first: bool) -> np.ndarray:
    if is_numpy:
        return to_wxyz(module, q, True, component_first)
    return np.stack([to_wxyz(module, item, False, False) for item in q])


def _make_quaternions(module, values: np.ndarray, is_numpy: bool, component_first: bool):
    if is_numpy:
        return from_wxyz(module, values, True, component_first)
    return [module.from_wxyz(tuple(row)) for row in values]


def _slerp_pair(first: Rotation, second: Rotation, amount: float) -> Rotation:
    endpoints = Rotation.from_quat(np.stack((first.as_quat(), second.as_quat())))
    return Slerp((0.0, 1.0), endpoints)(amount)


def _squad_reference(values: np.ndarray, amount: np.ndarray, is_jpl: bool) -> np.ndarray:
    rotations = [_scipy_rotations(values[:, index], is_jpl) for index in range(4)]
    output = []
    for index, t in enumerate(amount):
        first = _slerp_pair(rotations[0][index], rotations[1][index], float(t))
        second = _slerp_pair(rotations[2][index], rotations[3][index], float(t))
        output.append(_slerp_pair(first, second, float(2.0 * t * (1.0 - t))).as_matrix())
    return np.asarray(output)


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_slerp_nlerp_and_squad_match_independent_interpolation_formulas(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rng = np.random.default_rng(0x51E2 + len(backend_name))
    count = 48
    values = rng.normal(scale=0.65, size=(count, 4, 4))
    values /= np.linalg.norm(values, axis=-1, keepdims=True)
    amounts = rng.uniform(0.0, 1.0, size=count)
    q0_values, q1_values, a_values, b_values = np.moveaxis(values, 1, 0)
    slerped, nlerped, squadded = [], [], []
    for index, amount in enumerate(amounts):
        args = [from_wxyz(module, row[index], is_numpy, component_first)
                for row in (q0_values, q1_values, a_values, b_values)]
        if is_numpy:
            slerped.append(to_wxyz(module, module.slerp(args[0], args[1], float(amount)),
                                   True, component_first))
            nlerped.append(to_wxyz(module, module.nlerp(args[0], args[1], float(amount)),
                                   True, component_first))
            squadded.append(to_wxyz(module, module.squad(*args, float(amount)),
                                    True, component_first))
        else:
            slerped.append(module.to_wxyz(module.slerp(args[0], args[1], float(amount))))
            nlerped.append(module.to_wxyz(module.nlerp(args[0], args[1], float(amount))))
            squadded.append(module.to_wxyz(module.squad(*args, float(amount))))
    slerped, nlerped, squadded = map(np.asarray, (slerped, nlerped, squadded))

    expected_slerp = np.stack([
        _slerp_pair(left, right, float(amount)).as_matrix()
        for left, right, amount in zip(
            _scipy_rotations(q0_values, is_jpl),
            _scipy_rotations(q1_values, is_jpl),
            amounts,
        )
    ])
    np.testing.assert_allclose(_scipy_rotations(slerped, is_jpl).as_matrix(), expected_slerp,
                               rtol=4e-13, atol=4e-13,
                               err_msg=f"{backend_name}.slerp SciPy Slerp")

    first = q0_values / np.linalg.norm(q0_values, axis=-1, keepdims=True)
    second = q1_values / np.linalg.norm(q1_values, axis=-1, keepdims=True)
    flip = np.sum(first * second, axis=-1) < 0.0
    second[flip] *= -1.0
    blend = (1.0 - amounts[:, None]) * first + amounts[:, None] * second
    expected_nlerp = blend / np.linalg.norm(blend, axis=-1, keepdims=True)
    np.testing.assert_allclose(nlerped, expected_nlerp, rtol=4e-13, atol=4e-13,
                               err_msg=f"{backend_name}.nlerp normalized coefficient blend")

    expected_squad = _squad_reference(values, amounts, is_jpl)
    np.testing.assert_allclose(_scipy_rotations(squadded, is_jpl).as_matrix(), expected_squad,
                               rtol=7e-13, atol=7e-13,
                               err_msg=f"{backend_name}.squad nested SciPy Slerp")


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_long_arc_interpolation_flags_and_extrapolation_match_coefficient_formulas(
    backend_name, module, is_jpl, is_numpy, component_first
):
    """Check sign-preserving interpolation and extrapolation away from singular pairs."""
    rng = np.random.default_rng(0x10A6 + len(backend_name))
    count = 36
    bases = Rotation.random(count, rng=rng)
    axes = rng.normal(size=(count, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    angles = rng.uniform(0.45, 2.35, size=count)
    targets = bases * Rotation.from_rotvec(axes * angles[:, None])
    q0_values = as_hamilton_coefficients(
        bases.as_quat()[..., (3, 0, 1, 2)], is_jpl
    )
    q1_values = -as_hamilton_coefficients(
        targets.as_quat()[..., (3, 0, 1, 2)], is_jpl
    )
    dot = np.sum(q0_values * q1_values, axis=-1)
    assert np.all(dot < 0.0)
    amounts = (-0.35, 0.4, 1.35)

    if is_numpy:
        q0 = from_wxyz(module, q0_values, True, component_first)
        q1 = from_wxyz(module, q1_values, True, component_first)
        actual_slerp = np.stack([
            to_wxyz(module, module.slerp(q0, q1, amount, shortest=False),
                    True, component_first)
            for amount in amounts
        ])
        actual_nlerp = np.stack([
            to_wxyz(module, module.nlerp(q0, q1, amount, shortest=False),
                    True, component_first)
            for amount in amounts
        ])
    else:
        q0 = [module.from_wxyz(tuple(value)) for value in q0_values]
        q1 = [module.from_wxyz(tuple(value)) for value in q1_values]
        actual_slerp = np.stack([
            [module.to_wxyz(module.slerp(left, right, amount, shortest=False))
             for left, right in zip(q0, q1)]
            for amount in amounts
        ])
        actual_nlerp = np.stack([
            [module.to_wxyz(module.nlerp(left, right, amount, shortest=False))
             for left, right in zip(q0, q1)]
            for amount in amounts
        ])

    # Direct S^3 geodesic weights retain the negative endpoint sign; no
    # QuatOps operation supplies these expected coefficients.
    amount_array = np.asarray(amounts)[:, None]
    half_angle = np.arccos(np.clip(dot, -1.0, 1.0))[None, :]
    sine = np.sin(half_angle)
    first_weight = np.sin((1.0 - amount_array) * half_angle) / sine
    second_weight = np.sin(amount_array * half_angle) / sine
    expected_slerp = (first_weight[..., None] * q0_values[None, ...]
                      + second_weight[..., None] * q1_values[None, ...])
    expected_slerp /= np.linalg.norm(expected_slerp, axis=-1, keepdims=True)
    blend = ((1.0 - amount_array[..., None]) * q0_values[None, ...]
             + amount_array[..., None] * q1_values[None, ...])
    expected_nlerp = blend / np.linalg.norm(blend, axis=-1, keepdims=True)

    np.testing.assert_allclose(actual_slerp, expected_slerp, rtol=8e-13, atol=8e-13,
                               err_msg=f"{backend_name}.slerp shortest=False")
    np.testing.assert_allclose(actual_nlerp, expected_nlerp, rtol=8e-13, atol=8e-13,
                               err_msg=f"{backend_name}.nlerp shortest=False")


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_boxplus_and_boxminus_match_scipy_rotation_composition(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rng = np.random.default_rng(0xB0A + len(backend_name))
    count = 40
    bases = Rotation.random(count, rng=rng)
    increments = rng.normal(scale=0.3, size=(count, 3))
    public = as_hamilton_coefficients(
        bases.as_quat()[..., (3, 0, 1, 2)], is_jpl
    )
    q = _make_quaternions(module, public, is_numpy, component_first)
    delta_input = orient_vectors(increments, component_first) if is_numpy else increments
    for frame in ("body", "world"):
        if is_numpy:
            updated = module.boxplus(q, delta_input, frame=frame)
            actual_updated = _wxyz_values(module, updated, True, component_first)
            actual_delta = vectors_to_trailing(module.boxminus(updated, q, frame=frame),
                                               component_first)
        else:
            updated = [module.boxplus(item, tuple(delta), frame=frame)
                       for item, delta in zip(q, increments)]
            actual_updated = _wxyz_values(module, updated, False, False)
            actual_delta = np.stack([
                tuple(module.boxminus(value, base, frame=frame))
                for value, base in zip(updated, q)
            ])
        base_rotations = _scipy_rotations(public, is_jpl)
        increment_rotations = Rotation.from_rotvec(increments)
        expected = (base_rotations * increment_rotations if frame == "body"
                    else increment_rotations * base_rotations)
        np.testing.assert_allclose(_scipy_rotations(actual_updated, is_jpl).as_matrix(),
                                   expected.as_matrix(), rtol=4e-13, atol=4e-13,
                                   err_msg=f"{backend_name}.boxplus {frame}")
        expected_delta = ((base_rotations.inv() * expected) if frame == "body"
                          else (expected * base_rotations.inv())).as_rotvec()
        np.testing.assert_allclose(actual_delta, expected_delta, rtol=5e-13, atol=5e-13,
                                   err_msg=f"{backend_name}.boxminus {frame}")


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_quaternion_derivative_matches_numpy_quaternion_product(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rng = np.random.default_rng(0xD3A + len(backend_name))
    count = 64
    values = rng.normal(size=(count, 4))
    rates = rng.normal(size=(count, 3))
    reference_q = reference_quaternions(values, is_jpl)
    pure_rates = quaternion.as_quat_array(np.column_stack((np.zeros(count), rates)))
    if is_numpy:
        q = from_wxyz(module, values, True, component_first)
        omega = orient_vectors(rates, component_first)
        for frame in ("body", "world"):
            actual = to_wxyz(
                module,
                module.quaternion_derivative(q, omega, frame=frame),
                True,
                component_first,
            )
            expected = (reference_q * pure_rates if frame == "body"
                        else pure_rates * reference_q) * 0.5
            expected_values = reference_coefficients(expected, is_jpl)
            np.testing.assert_allclose(actual, expected_values, rtol=2e-14, atol=2e-14,
                                       err_msg=f"{backend_name}.quaternion_derivative {frame}")
    else:
        for frame in ("body", "world"):
            actual = np.stack([
                module.to_wxyz(module.quaternion_derivative(
                    module.from_wxyz(tuple(value)), tuple(rate), frame=frame
                ))
                for value, rate in zip(values, rates)
            ])
            expected = (reference_q * pure_rates if frame == "body"
                        else pure_rates * reference_q) * 0.5
            expected_values = reference_coefficients(expected, is_jpl)
            np.testing.assert_allclose(actual, expected_values, rtol=2e-14, atol=2e-14,
                                       err_msg=f"{backend_name}.quaternion_derivative {frame}")


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_constant_and_sampled_angular_rate_integrators_match_scipy_products(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rng = np.random.default_rng(0x1A73 + len(backend_name))
    count, sample_count = 20, 6
    initial = Rotation.random(count, rng=rng)
    public = as_hamilton_coefficients(initial.as_quat()[..., (3, 0, 1, 2)], is_jpl)
    rates = rng.normal(scale=0.5, size=(count, sample_count, 3))
    times = np.linspace(0.0, 0.7, sample_count)
    q = _make_quaternions(module, public, is_numpy, component_first)
    if is_numpy:
        rate_input = orient_vectors(rates, component_first)
        constant_rates = rates[:, 0]
        constant_input = orient_vectors(constant_rates, component_first)
        for frame in ("body", "world"):
            integrated = module.integrate_angular_velocity(q, constant_input, 0.37, frame=frame)
            actual_constant = _wxyz_values(module, integrated, True, component_first)
            increment = Rotation.from_rotvec(constant_rates * 0.37)
            expected_constant = (initial * increment if frame == "body"
                                 else increment * initial)
            np.testing.assert_allclose(
                _scipy_rotations(actual_constant, is_jpl).as_matrix(),
                expected_constant.as_matrix(), rtol=4e-13, atol=4e-13,
                err_msg=f"{backend_name}.integrate_angular_velocity {frame}",
            )
            for method in ("left", "midpoint"):
                actual = _wxyz_values(
                    module,
                    module.integrate_angular_velocity_samples(
                        q, rate_input, times, frame=frame, method=method
                    ),
                    True,
                    component_first,
                )
                expected_rotations = []
                for base, sample_rates in zip(initial, rates):
                    result = base
                    for index, dt in enumerate(np.diff(times)):
                        rate = (sample_rates[index] if method == "left"
                                else 0.5 * (sample_rates[index] + sample_rates[index + 1]))
                        increment = Rotation.from_rotvec(rate * dt)
                        result = result * increment if frame == "body" else increment * result
                    expected_rotations.append(result.as_matrix())
                np.testing.assert_allclose(
                    _scipy_rotations(actual, is_jpl).as_matrix(), expected_rotations,
                    rtol=7e-13, atol=7e-13,
                    err_msg=f"{backend_name}.integrate_angular_velocity_samples {frame}/{method}",
                )
    else:
        for frame in ("body", "world"):
            for index, (base, value, sample_rates) in enumerate(zip(initial, q, rates)):
                constant = module.integrate_angular_velocity(
                    value, tuple(sample_rates[0]), 0.37, frame=frame
                )
                expected_constant = Rotation.from_rotvec(sample_rates[0] * 0.37)
                expected_constant = (base * expected_constant if frame == "body"
                                     else expected_constant * base)
                np.testing.assert_allclose(
                    module.to_rotation_matrix(constant), expected_constant.as_matrix(),
                    rtol=4e-13, atol=4e-13,
                    err_msg=f"{backend_name}.integrate_angular_velocity {frame}",
                )
                for method in ("left", "midpoint"):
                    actual = module.integrate_angular_velocity_samples(
                        value, sample_rates, times, frame=frame, method=method
                    )
                    expected = base
                    for sample_index, dt in enumerate(np.diff(times)):
                        rate = (sample_rates[sample_index] if method == "left"
                                else 0.5 * (sample_rates[sample_index]
                                            + sample_rates[sample_index + 1]))
                        increment = Rotation.from_rotvec(rate * dt)
                        expected = (expected * increment if frame == "body"
                                    else increment * expected)
                    np.testing.assert_allclose(
                        module.to_rotation_matrix(actual), expected.as_matrix(),
                        rtol=7e-13, atol=7e-13,
                        err_msg=f"{backend_name}.integrate_angular_velocity_samples "
                                f"{frame}/{method} sample {index}",
                    )
