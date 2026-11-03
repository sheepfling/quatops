"""Independent SciPy oracles for coefficient and attitude conversions."""

import itertools
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.oracle_support import (
    BACKENDS,
    as_hamilton_coefficients,
    from_wxyz,
    orient_vectors,
    to_wxyz,
    vectors_to_trailing,
)

TAIT_BRYAN_SEQUENCES = tuple(
    "".join(axes) for axes in itertools.permutations("xyz")
)
PROPER_EULER_SEQUENCES = ("xyx", "xzx", "yxy", "yzy", "zxz", "zyz")
EULER_SEQUENCES = (
    TAIT_BRYAN_SEQUENCES
    + PROPER_EULER_SEQUENCES
    + tuple(sequence.upper() for sequence in TAIT_BRYAN_SEQUENCES)
    + tuple(sequence.upper() for sequence in PROPER_EULER_SEQUENCES)
)


def _scipy_rotations(public_wxyz: np.ndarray, is_jpl: bool) -> Rotation:
    hamilton_wxyz = as_hamilton_coefficients(public_wxyz, is_jpl)
    return Rotation.from_quat(hamilton_wxyz[..., (1, 2, 3, 0)])


def _canonical_coefficients(public_wxyz: np.ndarray, is_jpl: bool) -> np.ndarray:
    hamilton = as_hamilton_coefficients(public_wxyz, is_jpl).copy()
    flip = hamilton[:, 0] < 0.0
    tie = hamilton[:, 0] == 0.0
    for index in np.flatnonzero(tie):
        first_nonzero = next((value for value in hamilton[index, 1:] if value != 0.0), 0.0)
        flip[index] = first_nonzero < 0.0
    hamilton[flip] *= -1.0
    hamilton[hamilton == 0.0] = 0.0
    return as_hamilton_coefficients(hamilton, is_jpl)


def _scipy_rotations_from_public_array(
    values: np.ndarray, is_jpl: bool, scalar_last: bool, component_axis: int
) -> Rotation:
    q = np.moveaxis(values, component_axis, -1)
    if scalar_last:
        x, y, z, w = np.moveaxis(q, -1, 0)
        q = np.stack((w, x, y, z), axis=-1)
    hamilton = as_hamilton_coefficients(q, is_jpl)
    return Rotation.from_quat(hamilton.reshape((-1, 4))[..., (1, 2, 3, 0)])


def _make_quaternions(module, values: np.ndarray, is_numpy: bool, component_first: bool):
    if is_numpy:
        return from_wxyz(module, values, True, component_first)
    return [module.from_wxyz(tuple(float(value) for value in row)) for row in values]


def _wxyz_values(module, quaternions, is_numpy: bool, component_first: bool) -> np.ndarray:
    if is_numpy:
        return to_wxyz(module, quaternions, True, component_first)
    return np.stack([to_wxyz(module, value, False, False) for value in quaternions])


def _converted_values(module, operation: str, quaternions, is_numpy: bool,
                      component_first: bool) -> np.ndarray:
    if is_numpy:
        return vectors_to_trailing(getattr(module, operation)(quaternions), component_first)
    return np.stack([tuple(getattr(module, operation)(value)) for value in quaternions])


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_coefficient_orders_and_hamilton_conversion_preserve_values(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rng = np.random.default_rng(0xC0EF + len(backend_name))
    values = rng.normal(size=(80, 4))
    q = _make_quaternions(module, values, is_numpy, component_first)
    np.testing.assert_allclose(_wxyz_values(module, q, is_numpy, component_first), values,
                               rtol=0, atol=0)

    expected_xyzw = values[:, (1, 2, 3, 0)]
    actual_xyzw = _converted_values(module, "to_xyzw", q, is_numpy, component_first)
    np.testing.assert_allclose(actual_xyzw, expected_xyzw, rtol=0, atol=0,
                               err_msg=f"{backend_name}.to_xyzw")
    if is_numpy:
        restored = module.from_xyzw(orient_vectors(expected_xyzw, component_first))
    else:
        restored = [module.from_xyzw(row) for row in expected_xyzw]
    np.testing.assert_allclose(_wxyz_values(module, restored, is_numpy, component_first), values,
                               rtol=0, atol=0, err_msg=f"{backend_name}.from_xyzw")

    if is_jpl:
        expected_hamilton = as_hamilton_coefficients(values, True)
        actual_hamilton = _converted_values(module, "to_hamilton", q, is_numpy, component_first)
        np.testing.assert_allclose(actual_hamilton, expected_hamilton, rtol=0, atol=0,
                                   err_msg=f"{backend_name}.to_hamilton")
        if is_numpy:
            converted = module.from_hamilton(orient_vectors(expected_hamilton, component_first))
        else:
            converted = [module.from_hamilton(row) for row in expected_hamilton]
        np.testing.assert_allclose(_wxyz_values(module, converted, is_numpy, component_first),
                                   values, rtol=0, atol=0,
                                   err_msg=f"{backend_name}.from_hamilton")


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_canonicalize_uses_hamilton_tie_policy_and_preserves_magnitude(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rng = np.random.default_rng(0xCA110 + len(backend_name))
    values = rng.normal(size=(64, 4))
    values[:4] = ((-1.0, 2.0, -3.0, 4.0), (0.0, -1.0, 0.0, 0.0),
                  (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, -1.0))
    q = _make_quaternions(module, values, is_numpy, component_first)
    canonical = (module.canonicalize(q) if is_numpy
                 else [module.canonicalize(value) for value in q])
    actual = _wxyz_values(module, canonical, is_numpy, component_first)
    expected = _canonical_coefficients(values, is_jpl)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0,
                               err_msg=f"{backend_name}.canonicalize")
    np.testing.assert_allclose(np.linalg.norm(actual, axis=-1),
                               np.linalg.norm(values, axis=-1), rtol=2e-15, atol=0)


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_sign_alignment_and_rotation_distance_match_coefficient_and_scipy_references(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rng = np.random.default_rng(0x51A9 + len(backend_name))
    q_values = rng.normal(size=(64, 4))
    ref_values = rng.normal(size=(64, 4))
    if is_numpy:
        q = from_wxyz(module, q_values, True, component_first)
        reference = from_wxyz(module, ref_values, True, component_first)
        aligned = to_wxyz(module, module.align_sign(q, reference), True, component_first)
        dots = np.sum(q_values * ref_values, axis=-1)
        expected_aligned = q_values * np.where(dots < 0.0, -1.0, 1.0)[:, None]
        np.testing.assert_allclose(aligned, expected_aligned, rtol=0, atol=0,
                                   err_msg=f"{backend_name}.align_sign")
        distance = module.rotation_distance(q, reference)
        equivalent = module.rotations_equivalent(q, reference, atol=0.7)
    else:
        aligned = []
        distance = []
        equivalent = []
        for left, right in zip(q_values, ref_values):
            q = from_wxyz(module, left, False, False)
            reference = from_wxyz(module, right, False, False)
            aligned.append(to_wxyz(module, module.align_sign(q, reference), False, False))
            distance.append(module.rotation_distance(q, reference))
            equivalent.append(module.rotations_equivalent(q, reference, atol=0.7))
        aligned = np.asarray(aligned)
        dots = np.sum(q_values * ref_values, axis=-1)
        expected_aligned = q_values * np.where(dots < 0.0, -1.0, 1.0)[:, None]
    np.testing.assert_allclose(aligned, expected_aligned, rtol=0, atol=0,
                               err_msg=f"{backend_name}.align_sign")

    expected_distance = (_scipy_rotations(q_values, is_jpl).inv()
                         * _scipy_rotations(ref_values, is_jpl)).magnitude()
    np.testing.assert_allclose(distance, expected_distance, rtol=5e-13, atol=5e-13,
                               err_msg=f"{backend_name}.rotation_distance")
    np.testing.assert_array_equal(equivalent, expected_distance <= 0.7,
                                  err_msg=f"{backend_name}.rotations_equivalent")

    # Sequential sign policy is checked in coefficient space, including
    # batches of trajectories for NumPy layouts.
    sequence = rng.normal(size=(5, 12, 4))
    if is_numpy:
        sequence_input = orient_vectors(sequence, component_first)
        continuous = vectors_to_trailing(
            module.make_sign_continuous(sequence_input), component_first
        )
        expected = sequence.copy()
        for sample_index in range(1, expected.shape[-2]):
            flip = np.sum(expected[:, sample_index - 1] * expected[:, sample_index], axis=-1) < 0.0
            expected[flip, sample_index] *= -1.0
    else:
        continuous = np.stack([
            [module.to_wxyz(value) for value in module.make_sign_continuous(
                [module.from_wxyz(tuple(q)) for q in row]
            )]
            for row in sequence
        ])
        expected = sequence.copy()
        for sample_index in range(1, expected.shape[-2]):
            flip = np.sum(expected[:, sample_index - 1] * expected[:, sample_index], axis=-1) < 0.0
            expected[flip, sample_index] *= -1.0
    np.testing.assert_allclose(continuous, expected, rtol=0, atol=0,
                               err_msg=f"{backend_name}.make_sign_continuous")


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_axis_angle_and_two_vector_conversions_match_scipy(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rng = np.random.default_rng(0xA81A + len(backend_name))
    count = 72
    axes = rng.normal(size=(count, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    angles = rng.uniform(-2.6 * math.pi, 2.6 * math.pi, size=count)
    axis_angles = np.column_stack((axes * rng.uniform(0.1, 10.0, size=(count, 1)), angles))
    expected_axis_rotations = Rotation.from_rotvec(axes * angles[:, None])
    axis_input = orient_vectors(axis_angles, component_first) if is_numpy else axis_angles
    if is_numpy:
        made = module.from_axis_angle(axis_input)
        made_values = _wxyz_values(module, made, True, component_first)
    else:
        made_values = np.stack(
            [module.to_wxyz(module.from_axis_angle(row)) for row in axis_angles]
        )
    np.testing.assert_allclose(_scipy_rotations(made_values, is_jpl).as_matrix(),
                               expected_axis_rotations.as_matrix(), rtol=3e-13, atol=3e-13,
                               err_msg=f"{backend_name}.from_axis_angle")

    # Stay away from the pi sign ambiguity for exact axis-angle coefficient checks.
    reference_rotations = Rotation.from_rotvec(axes * rng.uniform(0.02, math.pi - 0.02,
                                                                  size=(count, 1)))
    reference_hamilton = np.asarray(reference_rotations.as_quat())[..., (3, 0, 1, 2)]
    public = as_hamilton_coefficients(reference_hamilton, is_jpl)
    public *= rng.uniform(0.1, 20.0, size=(count, 1))
    q = _make_quaternions(module, public, is_numpy, component_first)
    if is_numpy:
        axis_angle = vectors_to_trailing(module.to_axis_angle(q), component_first)
    else:
        axis_angle = np.stack([tuple(module.to_axis_angle(item)) for item in q])
    expected_rotvec = reference_rotations.as_rotvec()
    expected_angle = np.linalg.norm(expected_rotvec, axis=-1)
    expected_axis = expected_rotvec / expected_angle[:, None]
    np.testing.assert_allclose(axis_angle[:, :3], expected_axis, rtol=3e-13, atol=3e-13,
                               err_msg=f"{backend_name}.to_axis_angle axis")
    np.testing.assert_allclose(axis_angle[:, 3], expected_angle, rtol=3e-13, atol=3e-13,
                               err_msg=f"{backend_name}.to_axis_angle angle")

    near_antiparallel_source = np.array(((0.2, 0.5, 0.7),))
    near_antiparallel_target = np.array(((-0.2 + 1e-12, -0.5, -0.7),))
    if is_numpy:
        near_antiparallel = module.from_two_vectors(
            orient_vectors(near_antiparallel_source, component_first),
            orient_vectors(near_antiparallel_target, component_first),
        )
        near_antiparallel_values = _wxyz_values(
            module, near_antiparallel, True, component_first
        )
    else:
        near_antiparallel = module.from_two_vectors(
            tuple(near_antiparallel_source[0]), tuple(near_antiparallel_target[0])
        )
        near_antiparallel_values = np.asarray(
            (module.to_wxyz(near_antiparallel),)
        )
    near_antiparallel_rotation = _scipy_rotations(
        near_antiparallel_values, is_jpl
    )
    expected_direction = near_antiparallel_target / np.linalg.norm(
        near_antiparallel_target, axis=-1, keepdims=True
    )
    actual_direction = near_antiparallel_rotation.apply(
        near_antiparallel_source
        / np.linalg.norm(near_antiparallel_source, axis=-1, keepdims=True)
    )
    np.testing.assert_allclose(
        actual_direction, expected_direction, rtol=0.0, atol=3e-15,
        err_msg=f"{backend_name}.from_two_vectors near-antiparallel mapping",
    )

    sources = rng.normal(size=(count, 3))
    targets = rng.normal(size=(count, 3))
    source_input = orient_vectors(sources, component_first) if is_numpy else sources
    target_input = orient_vectors(targets, component_first) if is_numpy else targets
    if is_numpy:
        align_values = _wxyz_values(
            module, module.from_two_vectors(source_input, target_input), True, component_first
        )
    else:
        align_values = np.stack(
            [module.to_wxyz(module.from_two_vectors(a, b))
             for a, b in zip(sources, targets)]
        )
    fit = _scipy_rotations(align_values, is_jpl)
    source_unit = sources / np.linalg.norm(sources, axis=-1, keepdims=True)
    target_unit = targets / np.linalg.norm(targets, axis=-1, keepdims=True)
    np.testing.assert_allclose(fit.apply(source_unit), target_unit, rtol=4e-13, atol=4e-13,
                               err_msg=f"{backend_name}.from_two_vectors mapping")
    expected_angle = np.arccos(np.clip(np.sum(source_unit * target_unit, axis=-1), -1.0, 1.0))
    np.testing.assert_allclose(fit.magnitude(), expected_angle, rtol=4e-13, atol=4e-13,
                               err_msg=f"{backend_name}.from_two_vectors shortest angle")

    antipodal_source = np.array(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    if is_numpy:
        halfturn = _wxyz_values(
            module,
            module.from_two_vectors(
                orient_vectors(antipodal_source, component_first),
                orient_vectors(-antipodal_source, component_first),
            ),
            True,
            component_first,
        )
    else:
        halfturn = np.stack(
            [module.to_wxyz(module.from_two_vectors(a, -a))
             for a in antipodal_source]
        )
    halfturn_rotations = _scipy_rotations(halfturn, is_jpl)
    np.testing.assert_allclose(halfturn_rotations.apply(antipodal_source), -antipodal_source,
                               rtol=0, atol=3e-15)
    np.testing.assert_allclose(halfturn_rotations.magnitude(), math.pi, rtol=0, atol=3e-15)


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_euler_conversions_match_scipy_for_all_sequences(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rng = np.random.default_rng(0xE013 + len(backend_name))
    angles = rng.uniform(-1.0, 1.0, size=(24, 3))
    if is_numpy:
        angle_input = orient_vectors(angles, component_first)
        for sequence in EULER_SEQUENCES:
            made = module.from_euler(angle_input, sequence)
            made_values = _wxyz_values(module, made, True, component_first)
            expected = Rotation.from_euler(sequence, angles)
            np.testing.assert_allclose(_scipy_rotations(made_values, is_jpl).as_matrix(),
                                       expected.as_matrix(), rtol=4e-13, atol=4e-13,
                                       err_msg=f"{backend_name}.from_euler({sequence})")
            extracted = vectors_to_trailing(module.to_euler(made, sequence), component_first)
            np.testing.assert_allclose(
                Rotation.from_euler(sequence, extracted).as_matrix(), expected.as_matrix(),
                rtol=5e-13, atol=5e-13,
                err_msg=f"{backend_name}.to_euler({sequence})",
            )
            degree_input = orient_vectors(np.rad2deg(angles), component_first)
            degree_made = module.from_euler(degree_input, sequence, degrees=True)
            np.testing.assert_allclose(
                _scipy_rotations(_wxyz_values(module, degree_made, True, component_first),
                                 is_jpl).as_matrix(),
                Rotation.from_euler(sequence, np.rad2deg(angles), degrees=True).as_matrix(),
                rtol=4e-13, atol=4e-13,
                err_msg=f"{backend_name}.from_euler({sequence}, degrees=True)",
            )
            degree_extracted = vectors_to_trailing(
                module.to_euler(made, sequence, degrees=True), component_first
            )
            np.testing.assert_allclose(
                Rotation.from_euler(sequence, degree_extracted, degrees=True).as_matrix(),
                expected.as_matrix(), rtol=5e-13, atol=5e-13,
                err_msg=f"{backend_name}.to_euler({sequence}, degrees=True)",
            )
    else:
        for sequence in EULER_SEQUENCES:
            for row in angles:
                made = module.from_euler(row, sequence)
                expected = Rotation.from_euler(sequence, row)
                np.testing.assert_allclose(module.to_rotation_matrix(made), expected.as_matrix(),
                                           rtol=4e-13, atol=4e-13,
                                           err_msg=f"{backend_name}.from_euler({sequence})")
                extracted = module.to_euler(made, sequence)
                np.testing.assert_allclose(
                    Rotation.from_euler(sequence, tuple(extracted)).as_matrix(),
                    expected.as_matrix(),
                    rtol=5e-13, atol=5e-13,
                    err_msg=f"{backend_name}.to_euler({sequence})",
                )
                degree_row = np.rad2deg(row)
                degree_made = module.from_euler(degree_row, sequence, degrees=True)
                np.testing.assert_allclose(
                    _scipy_rotations(
                        _wxyz_values(module, [degree_made], False, False), is_jpl
                    ).as_matrix()[0],
                    Rotation.from_euler(sequence, degree_row, degrees=True).as_matrix(),
                    rtol=4e-13, atol=4e-13,
                    err_msg=f"{backend_name}.from_euler({sequence}, degrees=True)",
                )
                degree_extracted = module.to_euler(made, sequence, degrees=True)
                np.testing.assert_allclose(
                    Rotation.from_euler(sequence, tuple(degree_extracted), degrees=True).as_matrix(),
                    expected.as_matrix(), rtol=5e-13, atol=5e-13,
                    err_msg=f"{backend_name}.to_euler({sequence}, degrees=True)",
                )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_euler_gimbal_status_matches_scipy_rotations(
    backend_name, module, is_jpl, is_numpy, component_first
):
    for sequence in EULER_SEQUENCES:
        proper = sequence[0].lower() == sequence[2].lower()
        singular_middle = 0.0 if proper else math.pi / 2.0
        opposite_middle = math.pi if proper else -math.pi / 2.0
        angles = np.array(((0.3, singular_middle, -0.2),
                           (-0.4, opposite_middle, 0.25),
                           (0.1, 0.37 if proper else -0.29, 0.2)))
        expected = Rotation.from_euler(sequence, angles)
        if is_numpy:
            made = module.from_euler(orient_vectors(angles, component_first), sequence)
            result = module.to_euler_with_status(made, sequence)
            extracted = vectors_to_trailing(result.angles, component_first)
            actual_status = np.asarray(result.gimbal_lock)
            actual = _scipy_rotations(
                _wxyz_values(module, made, True, component_first), is_jpl
            )
        else:
            made = [module.from_euler(tuple(row), sequence) for row in angles]
            results = [module.to_euler_with_status(value, sequence) for value in made]
            extracted = np.stack([tuple(result.angles) for result in results])
            actual_status = np.asarray([result.gimbal_lock for result in results])
            actual = _scipy_rotations(
                _wxyz_values(module, made, False, False), is_jpl
            )
            degree_results = [
                module.to_euler_with_status(value, sequence, degrees=True)
                for value in made
            ]
            degree_extracted = np.stack([tuple(result.angles) for result in degree_results])
            np.testing.assert_array_equal(
                [result.gimbal_lock for result in degree_results], actual_status
            )
            np.testing.assert_allclose(
                Rotation.from_euler(sequence, degree_extracted, degrees=True).as_matrix(),
                expected.as_matrix(), rtol=7e-13, atol=7e-13,
                err_msg=f"{backend_name}.to_euler_with_status({sequence}, degrees=True)",
            )
        np.testing.assert_array_equal(actual_status, (True, True, False),
                                      err_msg=f"{backend_name}.to_euler_with_status({sequence})")
        np.testing.assert_allclose(actual.as_matrix(), expected.as_matrix(),
                                   rtol=4e-13, atol=4e-13)
        np.testing.assert_allclose(
            Rotation.from_euler(sequence, extracted).as_matrix(), expected.as_matrix(),
            rtol=7e-13, atol=7e-13,
            err_msg=f"{backend_name}.singular representatives for {sequence}",
        )
        np.testing.assert_allclose(extracted[:2, 2], 0.0, atol=0.0,
                                   err_msg=f"{backend_name}.third angle at {sequence} lock")


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    [backend for backend in BACKENDS if backend[3]],
    ids=[backend[0] for backend in BACKENDS if backend[3]],
)
@pytest.mark.parametrize("component_axis", (0, -1))
@pytest.mark.parametrize("sequence", EULER_SEQUENCES)
def test_numpy_euler_independent_axes_and_read_only_batch(
    backend_name, module, is_jpl, is_numpy, component_first, component_axis, sequence
):
    rng = np.random.default_rng(0xA815 + len(backend_name))
    storage = rng.uniform(-45.0, 45.0, size=(2, 3, 8))
    angles = storage[:, :, ::2]
    proper = sequence[0].lower() == sequence[2].lower()
    if proper:
        angles[:, 1, :] = rng.uniform(30.0, 150.0, size=(2, 4))
    angles[0, :, 0] = (23.0, 0.0 if proper else 90.0, -37.0)
    angles.setflags(write=False)
    angle_values = np.moveaxis(angles, 1, -1)
    expected = Rotation.from_euler(sequence, angle_values.reshape((-1, 3)), degrees=True)

    made = module.from_euler(
        angles, sequence, degrees=True, angle_axis=1, component_axis=component_axis
    )
    actual = _scipy_rotations_from_public_array(
        made, is_jpl, scalar_last=backend_name.endswith("xyzw"),
        component_axis=component_axis,
    )
    expected_matrices = expected.as_matrix().reshape((2, 4, 3, 3))
    np.testing.assert_allclose(actual.as_matrix().reshape((2, 4, 3, 3)), expected_matrices,
                               rtol=5e-13, atol=5e-13,
                               err_msg=f"{backend_name}.from_euler arbitrary axes")

    scaled = made * -3.75
    result = module.to_euler_with_status(
        scaled, sequence, degrees=True, component_axis=component_axis, angle_axis=2
    )
    extracted = np.moveaxis(result.angles, 2, -1)
    assert result.angles.shape == (2, 4, 3)
    assert result.gimbal_lock.shape == (2, 4)
    expected_status = np.zeros((2, 4), dtype=bool)
    expected_status[0, 0] = True
    np.testing.assert_array_equal(result.gimbal_lock, expected_status)
    np.testing.assert_allclose(
        Rotation.from_euler(sequence, extracted.reshape((-1, 3)), degrees=True)
        .as_matrix().reshape((2, 4, 3, 3)),
        expected_matrices, rtol=8e-13, atol=8e-13,
        err_msg=f"{backend_name}.to_euler arbitrary axes",
    )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_rotation_vector_aliases_match_scipy(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rng = np.random.default_rng(0xA115 + len(backend_name))
    vectors = rng.normal(size=(48, 3))
    vectors *= rng.uniform(0.0, math.pi, size=(48, 1)) / np.linalg.norm(
        vectors, axis=-1, keepdims=True
    )
    vector_input = orient_vectors(vectors, component_first) if is_numpy else vectors
    if is_numpy:
        made = module.from_rotvec(vector_input)
        made_values = _wxyz_values(module, made, True, component_first)
        actual_vectors = vectors_to_trailing(module.to_rotvec(made), component_first)
    else:
        made = [module.from_rotvec(tuple(row)) for row in vectors]
        made_values = np.stack([module.to_wxyz(value) for value in made])
        actual_vectors = np.stack([tuple(module.to_rotvec(value)) for value in made])
    expected = Rotation.from_rotvec(vectors)
    np.testing.assert_allclose(_scipy_rotations(made_values, is_jpl).as_matrix(),
                               expected.as_matrix(), rtol=4e-13, atol=4e-13,
                               err_msg=f"{backend_name}.from_rotvec")
    np.testing.assert_allclose(actual_vectors, expected.as_rotvec(), rtol=5e-13, atol=5e-13,
                               err_msg=f"{backend_name}.to_rotvec")


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_modified_rodrigues_parameters_match_closed_form_and_scipy(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rng = np.random.default_rng(0x0A2 + len(backend_name))
    sigma = rng.normal(scale=0.23, size=(64, 3))
    sigma_norm = np.linalg.norm(sigma, axis=-1)
    expected_rotations = Rotation.from_rotvec(
        sigma * (4.0 * np.arctan(sigma_norm) / sigma_norm)[:, None]
    )
    sigma_input = orient_vectors(sigma, component_first) if is_numpy else sigma
    if is_numpy:
        made = module.from_mrp(sigma_input)
        made_values = to_wxyz(module, made, True, component_first)
        extracted = vectors_to_trailing(module.to_mrp(made), component_first)
    else:
        made_values = np.stack(
            [to_wxyz(module, module.from_mrp(row), False, False) for row in sigma]
        )
        extracted = np.stack(
            [tuple(module.to_mrp(from_wxyz(module, row, False, False))) for row in made_values]
        )
    np.testing.assert_allclose(_scipy_rotations(made_values, is_jpl).as_matrix(),
                               expected_rotations.as_matrix(), rtol=4e-13, atol=4e-13,
                               err_msg=f"{backend_name}.from_mrp")
    np.testing.assert_allclose(extracted, sigma, rtol=3e-13, atol=3e-13,
                               err_msg=f"{backend_name}.to_mrp round trip")

    # Independent inverse formula from SciPy's canonical scalar-first values.
    rotvec = rng.normal(size=(64, 3))
    rotvec *= rng.uniform(0.03, math.pi - 0.03, size=(64, 1)) / np.linalg.norm(
        rotvec, axis=-1, keepdims=True
    )
    rotations = Rotation.from_rotvec(rotvec)
    scipy_xyzw = rotations.as_quat()
    scipy_wxyz = scipy_xyzw[:, (3, 0, 1, 2)]
    public = as_hamilton_coefficients(scipy_wxyz, is_jpl)
    q = _make_quaternions(module, public, is_numpy, component_first)
    if is_numpy:
        actual = vectors_to_trailing(module.to_mrp(q), component_first)
    else:
        actual = np.stack([tuple(module.to_mrp(from_wxyz(module, row, False, False)))
                           for row in public])
    expected = scipy_wxyz[:, 1:] / (1.0 + scipy_wxyz[:, :1])
    np.testing.assert_allclose(actual, expected, rtol=4e-13, atol=4e-13,
                               err_msg=f"{backend_name}.to_mrp SciPy oracle")
