"""SciPy and direct tangent-space references for alignment and statistics."""

import numpy as np
import pytest
from scipy.optimize import root
from scipy.spatial.transform import Rotation

from tests.support.oracle_support import (
    BACKENDS,
    as_hamilton_coefficients,
    from_wxyz,
    matrices_to_trailing,
    orient_vectors,
    random_rotations,
    to_wxyz,
)


def _public_coefficients(rotations: Rotation, is_jpl: bool) -> np.ndarray:
    wxyz = rotations.as_quat()[..., (3, 0, 1, 2)]
    return as_hamilton_coefficients(wxyz, is_jpl)


def _rotations_from_public(values: np.ndarray, is_jpl: bool) -> Rotation:
    wxyz = as_hamilton_coefficients(values, is_jpl)
    return Rotation.from_quat(wxyz[..., (1, 2, 3, 0)])


def _sample_cloud(rng: np.random.Generator, batch_count: int, sample_count: int):
    centers = random_rotations(batch_count, rng)
    tangent = rng.normal(scale=0.19, size=(batch_count, sample_count, 3))
    rotations = [Rotation.from_rotvec(tangent[index]) * centers[index]
                 for index in range(batch_count)]
    return centers, rotations


def _sample_coefficients(rotations: list[Rotation], is_jpl: bool) -> np.ndarray:
    return np.stack([_public_coefficients(values, is_jpl) for values in rotations])


def _make_batches(module, values: np.ndarray, is_numpy: bool, component_first: bool):
    if is_numpy:
        return from_wxyz(module, values, True, component_first)
    return [[module.from_wxyz(tuple(row)) for row in batch] for batch in values]


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_markley_intrinsic_means_and_tangent_covariance_match_scipy(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rng = np.random.default_rng(0x57A7 + len(backend_name))
    batch_count, sample_count = 6, 9
    _, sample_rotations = _sample_cloud(rng, batch_count, sample_count)
    samples = _sample_coefficients(sample_rotations, is_jpl)
    sample_argument = _make_batches(module, samples, is_numpy, component_first)
    weights = np.linspace(0.3, 1.7, sample_count)
    normalized_weights = weights / weights.sum()
    reference_means = [rotations.mean(weights=weights) for rotations in sample_rotations]
    unweighted_means = [rotations.mean() for rotations in sample_rotations]

    if is_numpy:
        mean_result = module.rotation_mean(sample_argument, weights)
        actual_mean = to_wxyz(module, mean_result, True, component_first)
        intrinsic_result = module.rotation_mean(sample_argument, weights, method="intrinsic")
        actual_intrinsic = to_wxyz(module, intrinsic_result, True, component_first)
        default_mean_result = module.rotation_mean(sample_argument)
        actual_default_mean = to_wxyz(module, default_mean_result, True, component_first)
        default_intrinsic_result = module.rotation_mean(
            sample_argument, method="intrinsic", tolerance=2e-12, max_iterations=200
        )
        actual_default_intrinsic = to_wxyz(
            module, default_intrinsic_result, True, component_first
        )
    else:
        mean_result = [module.rotation_mean(batch, weights) for batch in sample_argument]
        actual_mean = np.stack([module.to_wxyz(value) for value in mean_result])
        intrinsic_result = [module.rotation_mean(batch, weights, method="intrinsic")
                            for batch in sample_argument]
        actual_intrinsic = np.stack([module.to_wxyz(value) for value in intrinsic_result])
        default_mean_result = [module.rotation_mean(batch) for batch in sample_argument]
        actual_default_mean = np.stack([
            module.to_wxyz(value) for value in default_mean_result
        ])
        default_intrinsic_result = [
            module.rotation_mean(batch, method="intrinsic", tolerance=2e-12,
                                 max_iterations=200)
            for batch in sample_argument
        ]
        actual_default_intrinsic = np.stack([
            module.to_wxyz(value) for value in default_intrinsic_result
        ])

    expected_mean_matrices = np.stack([value.as_matrix() for value in reference_means])
    np.testing.assert_allclose(_rotations_from_public(actual_mean, is_jpl).as_matrix(),
                               expected_mean_matrices, rtol=8e-13, atol=8e-13,
                               err_msg=f"{backend_name}.rotation_mean Markley")

    reference_intrinsic = []
    for rotations, initial in zip(sample_rotations, reference_means):
        def residual(candidate_vector, rotations=rotations):
            candidate = Rotation.from_rotvec(candidate_vector)
            errors = (candidate.inv() * rotations).as_rotvec()
            return np.average(errors, axis=0, weights=normalized_weights)

        solution = root(residual, initial.as_rotvec(), tol=1e-12)
        assert solution.success, f"SciPy intrinsic-mean reference failed: {solution.message}"
        reference_intrinsic.append(Rotation.from_rotvec(solution.x))
    expected_intrinsic_matrices = np.stack([value.as_matrix() for value in reference_intrinsic])
    np.testing.assert_allclose(_rotations_from_public(actual_intrinsic, is_jpl).as_matrix(),
                               expected_intrinsic_matrices, rtol=2e-11, atol=2e-11,
                               err_msg=f"{backend_name}.rotation_mean intrinsic")

    expected_unweighted_matrices = np.stack([value.as_matrix() for value in unweighted_means])
    np.testing.assert_allclose(_rotations_from_public(actual_default_mean, is_jpl).as_matrix(),
                               expected_unweighted_matrices, rtol=8e-13, atol=8e-13,
                               err_msg=f"{backend_name}.rotation_mean default weights")
    reference_default_intrinsic = []
    for rotations, initial in zip(sample_rotations, unweighted_means):
        def unweighted_residual(candidate_vector, rotations=rotations):
            candidate = Rotation.from_rotvec(candidate_vector)
            return np.mean((candidate.inv() * rotations).as_rotvec(), axis=0)

        solution = root(unweighted_residual, initial.as_rotvec(), tol=1e-12)
        assert solution.success, f"SciPy unweighted intrinsic reference failed: {solution.message}"
        reference_default_intrinsic.append(Rotation.from_rotvec(solution.x))
    expected_default_intrinsic = np.stack([
        value.as_matrix() for value in reference_default_intrinsic
    ])
    np.testing.assert_allclose(
        _rotations_from_public(actual_default_intrinsic, is_jpl).as_matrix(),
        expected_default_intrinsic,
        rtol=2e-11,
        atol=2e-11,
        err_msg=f"{backend_name}.rotation_mean unweighted intrinsic",
    )

    # Covariance uses an independently selected SciPy mean and SciPy tangent
    # logarithms, not QuatOps.boxminus or QuatOps.rotation_mean.
    mean_values = _public_coefficients(Rotation.from_quat(np.stack(
        [value.as_quat() for value in reference_means]
    )), is_jpl)
    means_argument = (from_wxyz(module, mean_values, True, component_first) if is_numpy
                      else [module.from_wxyz(tuple(value)) for value in mean_values])
    if is_numpy:
        body_covariance = matrices_to_trailing(
            module.rotation_covariance(sample_argument, means_argument, weights, frame="body"),
            component_first,
        )
        world_covariance = matrices_to_trailing(
            module.rotation_covariance(sample_argument, means_argument, weights, frame="world"),
            component_first,
        )
        default_covariance = matrices_to_trailing(
            module.rotation_covariance(sample_argument, weights=weights), component_first
        )
        unweighted_covariance = matrices_to_trailing(
            module.rotation_covariance(sample_argument), component_first
        )
    else:
        body_covariance, world_covariance, default_covariance = [], [], []
        unweighted_covariance = []
        for batch, mean in zip(sample_argument, means_argument):
            body_covariance.append(module.rotation_covariance(batch, mean, weights, frame="body"))
            world_covariance.append(module.rotation_covariance(batch, mean, weights, frame="world"))
            default_covariance.append(module.rotation_covariance(batch, weights=weights))
            unweighted_covariance.append(module.rotation_covariance(batch))
        body_covariance = np.asarray(body_covariance)
        world_covariance = np.asarray(world_covariance)
        default_covariance = np.asarray(default_covariance)
        unweighted_covariance = np.asarray(unweighted_covariance)

    expected_body, expected_world = [], []
    for rotations, mean in zip(sample_rotations, reference_means):
        body_errors = (mean.inv() * rotations).as_rotvec()
        world_errors = (rotations * mean.inv()).as_rotvec()
        expected_body.append(np.einsum("n,ni,nj->ij", normalized_weights,
                                       body_errors, body_errors))
        expected_world.append(np.einsum("n,ni,nj->ij", normalized_weights,
                                        world_errors, world_errors))
    expected_body = np.asarray(expected_body)
    expected_world = np.asarray(expected_world)
    np.testing.assert_allclose(body_covariance, expected_body, rtol=1e-12, atol=1e-14,
                               err_msg=f"{backend_name}.rotation_covariance body")
    np.testing.assert_allclose(world_covariance, expected_world, rtol=1e-12, atol=1e-14,
                               err_msg=f"{backend_name}.rotation_covariance world")
    np.testing.assert_allclose(default_covariance, expected_body, rtol=1e-12, atol=1e-14,
                               err_msg=f"{backend_name}.rotation_covariance default mean")
    expected_unweighted_covariance = []
    for rotations, mean in zip(sample_rotations, unweighted_means):
        errors = (mean.inv() * rotations).as_rotvec()
        expected_unweighted_covariance.append(errors.T @ errors / sample_count)
    np.testing.assert_allclose(
        unweighted_covariance,
        expected_unweighted_covariance,
        rtol=1e-12,
        atol=1e-14,
        err_msg=f"{backend_name}.rotation_covariance default weights and mean",
    )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_weighted_wahba_alignment_matches_scipy_for_batched_inputs(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rng = np.random.default_rng(0xA119 + len(backend_name))
    batch_count, vector_count = 8, 10
    truths = random_rotations(batch_count, rng)
    source = rng.normal(size=(batch_count, vector_count, 3))
    target = np.stack([
        truth.apply(vectors) + rng.normal(scale=0.012, size=vectors.shape)
        for truth, vectors in zip(truths, source)
    ])
    weights = np.linspace(0.2, 2.0, vector_count)
    expected = [Rotation.align_vectors(target[index], source[index], weights=weights)[0]
                for index in range(batch_count)]
    if is_numpy:
        actual = to_wxyz(
            module,
            module.align_vectors(
                orient_vectors(source, component_first),
                orient_vectors(target, component_first),
                weights,
            ),
            True,
            component_first,
        )
    else:
        actual = np.stack([
            module.to_wxyz(module.align_vectors(src, dst, weights))
            for src, dst in zip(source, target)
        ])
    np.testing.assert_allclose(
        _rotations_from_public(actual, is_jpl).as_matrix(),
        np.stack([value.as_matrix() for value in expected]),
        rtol=8e-13,
        atol=8e-13,
        err_msg=f"{backend_name}.align_vectors weighted SciPy Wahba",
    )

    expected_unweighted = [Rotation.align_vectors(target[index], source[index])[0]
                           for index in range(batch_count)]
    if is_numpy:
        actual_unweighted = to_wxyz(
            module,
            module.align_vectors(
                orient_vectors(source, component_first),
                orient_vectors(target, component_first),
            ),
            True,
            component_first,
        )
    else:
        actual_unweighted = np.stack([
            module.to_wxyz(module.align_vectors(src, dst))
            for src, dst in zip(source, target)
        ])
    np.testing.assert_allclose(
        _rotations_from_public(actual_unweighted, is_jpl).as_matrix(),
        np.stack([value.as_matrix() for value in expected_unweighted]),
        rtol=8e-13,
        atol=8e-13,
        err_msg=f"{backend_name}.align_vectors default weights",
    )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_swing_twist_factors_match_independent_scipy_factorization(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rng = np.random.default_rng(0x5A17 + len(backend_name))
    count = 40
    rotations = random_rotations(count, rng)
    public = _public_coefficients(rotations, is_jpl)
    axes = rng.normal(size=(count, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    q = (from_wxyz(module, public, True, component_first) if is_numpy
         else [module.from_wxyz(tuple(value)) for value in public])
    axis_input = orient_vectors(axes, component_first) if is_numpy else axes
    if is_numpy:
        swing, twist = module.swing_twist(q, axis_input)
        swing_values = to_wxyz(module, swing, True, component_first)
        twist_values = to_wxyz(module, twist, True, component_first)
    else:
        factors = [module.swing_twist(value, tuple(axis)) for value, axis in zip(q, axes)]
        swing_values = np.stack([module.to_wxyz(pair[0]) for pair in factors])
        twist_values = np.stack([module.to_wxyz(pair[1]) for pair in factors])

    hamilton = as_hamilton_coefficients(public, is_jpl)
    expected_twist = []
    expected_swing = []
    for index, axis in enumerate(axes):
        coefficients = hamilton[index] / np.linalg.norm(hamilton[index])
        projection = np.dot(coefficients[1:], axis)
        twist_coefficients = np.array((coefficients[0], *(projection * axis)))
        twist_norm = np.linalg.norm(twist_coefficients)
        if twist_norm == 0.0:
            twist_coefficients = np.array((1.0, 0.0, 0.0, 0.0))
        else:
            twist_coefficients /= twist_norm
        twist_rotation = Rotation.from_quat(twist_coefficients[[1, 2, 3, 0]])
        expected_twist.append(twist_rotation.as_matrix())
        expected_swing.append((rotations[index] * twist_rotation.inv()).as_matrix())
    np.testing.assert_allclose(_rotations_from_public(twist_values, is_jpl).as_matrix(),
                               expected_twist, rtol=5e-13, atol=5e-13,
                               err_msg=f"{backend_name}.swing_twist twist")
    np.testing.assert_allclose(_rotations_from_public(swing_values, is_jpl).as_matrix(),
                               expected_swing, rtol=5e-13, atol=5e-13,
                               err_msg=f"{backend_name}.swing_twist swing")
