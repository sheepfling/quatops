"""Numba kernels and dispatch for batched quaternion operations."""

from __future__ import annotations

import cmath
from math import (
    acos,
    atan2,
    copysign,
    cos,
    exp,
    frexp,
    hypot,
    isfinite,
    ldexp,
    log,
    pi,
    sin,
    sqrt,
)
from typing import Any

import numpy as np

from ._numpy_support import _warn_numba_failure

_NUMBA_PRANGE: Any = range


def _jacobi_largest_numba(
    matrix: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Find the dominant eigenvector of a 4x4 symmetric matrix."""
    values = matrix.copy()
    vectors = np.eye(4, dtype=np.float64)
    for _ in range(1600):
        p, q = 0, 1
        off = abs(values[p, q])
        for row in range(4):
            for column in range(row + 1, 4):
                candidate = abs(values[row, column])
                if candidate > off:
                    p, q, off = row, column, candidate
        diagonal_scale = 1.0
        for index in range(4):
            diagonal_scale = max(diagonal_scale, abs(values[index, index]))
        if off <= 2e-16 * diagonal_scale:
            break
        tau = (values[q, q] - values[p, p]) / (2.0 * values[p, q])
        t = copysign(1.0, tau) / (abs(tau) + hypot(1.0, tau))
        if tau == 0.0:
            t = 1.0
        c = 1.0 / hypot(1.0, t)
        s = t * c
        app, aqq = values[p, p], values[q, q]
        values[p, p] = app - t * values[p, q]
        values[q, q] = aqq + t * values[p, q]
        values[p, q] = values[q, p] = 0.0
        for index in range(4):
            if index != p and index != q:
                akp, akq = values[index, p], values[index, q]
                values[index, p] = values[p, index] = c * akp - s * akq
                values[index, q] = values[q, index] = s * akp + c * akq
            vkp, vkq = vectors[index, p], vectors[index, q]
            vectors[index, p] = c * vkp - s * vkq
            vectors[index, q] = s * vkp + c * vkq
    largest_index = 0
    runner_up = -np.inf
    largest = -np.inf
    for index in range(4):
        value = values[index, index]
        if value > largest:
            runner_up = largest
            largest = value
            largest_index = index
        elif value > runner_up:
            runner_up = value
    eigenvector = np.empty(4, dtype=np.float64)
    for index in range(4):
        eigenvector[index] = vectors[index, largest_index]
    return eigenvector, largest, runner_up


def _normalize_quaternion_row_numba(
    value: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Normalize a quaternion row using scaled coefficients."""
    scale = max(abs(value[0]), abs(value[1]), abs(value[2]), abs(value[3]))
    result = np.empty(4, dtype=np.float64)
    if scale == 0.0:
        return result, False
    total = 0.0
    for component in range(4):
        scaled = value[component] / scale
        total += scaled * scaled
    length = sqrt(total)
    for component in range(4):
        result[component] = (value[component] / scale) / length
    return result, True


def _canonical_quaternion_row_numba(
    value: np.ndarray, epsilon: int, scalar_last: bool
) -> np.ndarray:
    """Choose the public deterministic sign for one quaternion row."""
    result = value.copy()
    if scalar_last:
        scalar = result[3]
        vector_start = 0
    else:
        scalar = result[0]
        vector_start = 1
    first = 0.0
    for offset in range(3):
        component = result[vector_start + offset]
        if component != 0.0:
            first = component
            break
    flip = scalar < 0.0 or (
        scalar == 0.0 and ((first < 0.0) if epsilon > 0 else (first > 0.0))
    )
    for component in range(4):
        if flip:
            result[component] = -result[component]
        if result[component] == 0.0:
            result[component] = 0.0
    return result


def _align_vectors_numba_impl(
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    epsilon: int,
    scalar_last: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve batched Wahba problems through Davenport's eigenproblem."""
    result = np.empty((sources.shape[0], 4), dtype=np.float64)
    statuses = np.zeros(sources.shape[0], dtype=np.int64)
    for group in range(sources.shape[0]):
        source_scale = 0.0
        target_scale = 0.0
        for sample in range(sources.shape[1]):
            for component in range(3):
                source_scale = max(source_scale, abs(sources[group, sample, component]))
                target_scale = max(target_scale, abs(targets[group, sample, component]))
        if source_scale == 0.0 or target_scale == 0.0:
            statuses[group] = 1
            continue
        b = np.zeros((3, 3), dtype=np.float64)
        for sample in range(sources.shape[1]):
            for row in range(3):
                source = sources[group, sample, row] / source_scale
                for column in range(3):
                    target = targets[group, sample, column] / target_scale
                    b[row, column] += weights[group, sample] * source * target
        sigma = b[0, 0] + b[1, 1] + b[2, 2]
        k = np.zeros((4, 4), dtype=np.float64)
        for row in range(3):
            for column in range(3):
                k[row, column] = b[row, column] + b[column, row]
                if row == column:
                    k[row, column] -= sigma
        k[0, 3] = k[3, 0] = b[1, 2] - b[2, 1]
        k[1, 3] = k[3, 1] = b[2, 0] - b[0, 2]
        k[2, 3] = k[3, 2] = b[0, 1] - b[1, 0]
        k[3, 3] = sigma
        matrix_scale = 0.0
        for row in range(4):
            for column in range(4):
                matrix_scale = max(matrix_scale, abs(k[row, column]))
        if matrix_scale == 0.0:
            statuses[group] = 2
            continue
        eigenvector, largest, runner_up = _jacobi_largest_numba(k / matrix_scale)
        if largest - runner_up <= 1e-12 * max(1.0, abs(largest)):
            statuses[group] = 2
            continue
        quaternion = np.zeros(4, dtype=np.float64)
        if scalar_last:
            quaternion[0] = epsilon * eigenvector[0]
            quaternion[1] = epsilon * eigenvector[1]
            quaternion[2] = epsilon * eigenvector[2]
            quaternion[3] = eigenvector[3]
        else:
            quaternion[0] = eigenvector[3]
            quaternion[1] = epsilon * eigenvector[0]
            quaternion[2] = epsilon * eigenvector[1]
            quaternion[3] = epsilon * eigenvector[2]
        quaternion, valid = _normalize_quaternion_row_numba(quaternion)
        if not valid:
            statuses[group] = 2
            continue
        result[group] = _canonical_quaternion_row_numba(
            quaternion, epsilon, scalar_last
        )
    return result, statuses


def _from_two_vectors_numba_impl(
    source: np.ndarray, target: np.ndarray, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, bool]:
    """Build shortest rotations between vector rows without batch temporaries."""
    result = np.empty((source.shape[0], 4), dtype=np.float64)
    valid = True
    for row in range(source.shape[0]):
        source_scale = max(
            abs(source[row, 0]), abs(source[row, 1]), abs(source[row, 2])
        )
        target_scale = max(
            abs(target[row, 0]), abs(target[row, 1]), abs(target[row, 2])
        )
        if source_scale == 0.0 or target_scale == 0.0:
            valid = False
            continue

        sx, sy, sz = (
            source[row, 0] / source_scale,
            source[row, 1] / source_scale,
            source[row, 2] / source_scale,
        )
        tx, ty, tz = (
            target[row, 0] / target_scale,
            target[row, 1] / target_scale,
            target[row, 2] / target_scale,
        )
        source_norm = sqrt(sx * sx + sy * sy + sz * sz)
        target_norm = sqrt(tx * tx + ty * ty + tz * tz)
        ux, uy, uz = sx / source_norm, sy / source_norm, sz / source_norm
        vx, vy, vz = tx / target_norm, ty / target_norm, tz / target_norm

        cross_x = uy * vz - uz * vy
        cross_y = uz * vx - ux * vz
        cross_z = ux * vy - uy * vx
        sum_norm = hypot(
            hypot(ux + vx, uy + vy),
            uz + vz,
        )
        antiparallel = sum_norm == 0.0
        if 0.0 < sum_norm < 1.0:
            sum_x, sum_y, sum_z = ux + vx, uy + vy, uz + vz
            cross_x = uy * sum_z - uz * sum_y
            cross_y = uz * sum_x - ux * sum_z
            cross_z = ux * sum_y - uy * sum_x
        if antiparallel:
            least_aligned = 0
            smallest = abs(ux)
            if abs(uy) < smallest:
                least_aligned, smallest = 1, abs(uy)
            if abs(uz) < smallest:
                least_aligned = 2
            basis_x = 1.0 if least_aligned == 0 else 0.0
            basis_y = 1.0 if least_aligned == 1 else 0.0
            basis_z = 1.0 if least_aligned == 2 else 0.0
            cross_x = uy * basis_z - uz * basis_y
            cross_y = uz * basis_x - ux * basis_z
            cross_z = ux * basis_y - uy * basis_x
            cross_scale = max(abs(cross_x), abs(cross_y), abs(cross_z))
            cross_x /= cross_scale
            cross_y /= cross_scale
            cross_z /= cross_scale
            cross_norm = sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
            vector_x, vector_y, vector_z = (
                cross_x / cross_norm,
                cross_y / cross_norm,
                cross_z / cross_norm,
            )
            scalar = 0.0
        else:
            vector_x = cross_x / sum_norm
            vector_y = cross_y / sum_norm
            vector_z = cross_z / sum_norm
            scalar = 0.5 * sum_norm

        quaternion_scale = max(abs(scalar), abs(vector_x), abs(vector_y), abs(vector_z))
        scalar /= quaternion_scale
        vector_x /= quaternion_scale
        vector_y /= quaternion_scale
        vector_z /= quaternion_scale
        quaternion_norm = sqrt(
            scalar * scalar
            + vector_x * vector_x
            + vector_y * vector_y
            + vector_z * vector_z
        )
        scalar /= quaternion_norm
        vector_x /= quaternion_norm
        vector_y /= quaternion_norm
        vector_z /= quaternion_norm

        # Match the Hamilton canonical sign before converting conventions.
        first = (
            vector_x if vector_x != 0.0 else (vector_y if vector_y != 0.0 else vector_z)
        )
        if scalar < 0.0 or (scalar == 0.0 and first < 0.0):
            scalar, vector_x, vector_y, vector_z = (
                -scalar,
                -vector_x,
                -vector_y,
                -vector_z,
            )
        if epsilon < 0:
            vector_x, vector_y, vector_z = -vector_x, -vector_y, -vector_z
        if scalar_last:
            result[row, 0], result[row, 1], result[row, 2], result[row, 3] = (
                vector_x,
                vector_y,
                vector_z,
                scalar,
            )
            scalar_index, vector_start = 3, 0
        else:
            result[row, 0], result[row, 1], result[row, 2], result[row, 3] = (
                scalar,
                vector_x,
                vector_y,
                vector_z,
            )
            scalar_index, vector_start = 0, 1

        # Apply the final layout/convention-specific tie rule used by _canonicalize.
        scalar = result[row, scalar_index]
        first = 0.0
        for offset in range(3):
            value = result[row, vector_start + offset]
            if value != 0.0:
                first = value
                break
        flip = scalar < 0.0 or (
            scalar == 0.0 and ((first < 0.0) if epsilon > 0 else (first > 0.0))
        )
        for component in range(4):
            value = -result[row, component] if flip else result[row, component]
            result[row, component] = 0.0 if value == 0.0 else value
    return result, valid


def _from_euler_numba_impl(
    angles: np.ndarray,
    first_axis: int,
    middle_axis: int,
    last_axis: int,
    repeated_axis: int,
    sequence_epsilon: int,
    proper: bool,
    algebra_epsilon: int,
    scalar_last: bool,
) -> np.ndarray:
    """Construct canonical quaternion rows from prepared Euler angles."""
    result = np.empty((angles.shape[0], 4), dtype=np.float64)
    for sample in _NUMBA_PRANGE(angles.shape[0]):
        a1, a2, a3 = angles[sample, 0], angles[sample, 1], angles[sample, 2]
        h = np.zeros(4, dtype=np.float64)
        if proper:
            u = 0.5 * a1 + 0.5 * a3
            v = 0.5 * a1 - 0.5 * a3
            c2 = cos(0.5 * a2)
            s2 = sin(0.5 * a2)
            h[0] = c2 * cos(u)
            h[first_axis] = c2 * sin(u)
            h[middle_axis] = s2 * cos(v)
            h[repeated_axis] = sequence_epsilon * s2 * sin(v)
        else:
            c1, c2, c3 = cos(0.5 * a1), cos(0.5 * a2), cos(0.5 * a3)
            s1, s2, s3 = sin(0.5 * a1), sin(0.5 * a2), sin(0.5 * a3)
            h[0] = c1 * c2 * c3 - sequence_epsilon * s1 * s2 * s3
            h[first_axis] = s1 * c2 * c3 + sequence_epsilon * c1 * s2 * s3
            h[middle_axis] = c1 * s2 * c3 - sequence_epsilon * s1 * c2 * s3
            h[last_axis] = c1 * c2 * s3 + sequence_epsilon * s1 * s2 * c3

        first_is_negative = (
            h[1] < 0.0
            or (h[1] == 0.0 and h[2] < 0.0)
            or (h[1] == 0.0 and h[2] == 0.0 and h[3] < 0.0)
        )
        flip = h[0] < 0.0 or (h[0] == 0.0 and first_is_negative)
        for component in range(4):
            if scalar_last:
                value = h[component + 1] * algebra_epsilon if component < 3 else h[0]
            else:
                value = h[0] if component == 0 else h[component] * algebra_epsilon
            if flip:
                value = -value
            result[sample, component] = 0.0 if value == 0.0 else value
    return result


def _to_euler_numba_impl(
    quaternions: np.ndarray,
    first_axis: int,
    middle_axis: int,
    last_axis: int,
    repeated_axis: int,
    sequence_epsilon: int,
    proper: bool,
    intrinsic: bool,
    tolerance: float,
    algebra_epsilon: int,
    scalar_last: bool,
    degrees: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Extract principal Euler angles and gimbal-lock flags from rows."""
    result = np.empty((quaternions.shape[0], 3), dtype=np.float64)
    locked = np.empty(quaternions.shape[0], dtype=np.bool_)
    valid_rows = np.ones(quaternions.shape[0], dtype=np.bool_)
    for sample in _NUMBA_PRANGE(quaternions.shape[0]):
        if scalar_last:
            x, y, z, w = (
                quaternions[sample, 0],
                quaternions[sample, 1],
                quaternions[sample, 2],
                quaternions[sample, 3],
            )
        else:
            w, x, y, z = (
                quaternions[sample, 0],
                quaternions[sample, 1],
                quaternions[sample, 2],
                quaternions[sample, 3],
            )
        scale = max(abs(w), abs(x), abs(y), abs(z))
        if scale == 0.0:
            valid_rows[sample] = False
            result[sample, :] = np.nan
            locked[sample] = False
            continue
        w, x, y, z = w / scale, x / scale, y / scale, z / scale
        x, y, z = algebra_epsilon * x, algebra_epsilon * y, algebra_epsilon * z
        coefficients = (w, x, y, z)
        if proper:
            a, b = w, coefficients[first_axis]
            c = coefficients[middle_axis]
            d = sequence_epsilon * coefficients[repeated_axis]
            kappa = 1
        else:
            a = w + coefficients[middle_axis]
            b = coefficients[first_axis] + sequence_epsilon * coefficients[last_axis]
            c = w - coefficients[middle_axis]
            d = coefficients[first_axis] - sequence_epsilon * coefficients[last_axis]
            kappa = sequence_epsilon

        chi = 2.0 * atan2(hypot(c, d), hypot(a, b))
        near_zero = chi <= tolerance
        near_pi = pi - chi <= tolerance
        is_locked = near_zero or near_pi
        u = atan2(b, a)
        v = atan2(d, c)
        if intrinsic:
            alpha = 2.0 * u if near_zero else (2.0 * v if near_pi else u + v)
            gamma = 0.0 if is_locked else kappa * (u - v)
        else:
            alpha = (
                2.0 * kappa * u
                if near_zero
                else (-2.0 * kappa * v if near_pi else kappa * (u - v))
            )
            gamma = 0.0 if is_locked else u + v
        chi = 0.0 if near_zero else (pi if near_pi else chi)
        beta = chi if proper else pi / 2.0 - chi
        if alpha >= pi:
            alpha -= 2.0 * pi
        elif alpha < -pi:
            alpha += 2.0 * pi
        if gamma >= pi:
            gamma -= 2.0 * pi
        elif gamma < -pi:
            gamma += 2.0 * pi
        if degrees:
            factor = 180.0 / pi
            alpha, beta, gamma = alpha * factor, beta * factor, gamma * factor
        result[sample, 0] = alpha
        result[sample, 1] = beta
        result[sample, 2] = gamma
        locked[sample] = is_locked
    return result, locked, bool(np.all(valid_rows))


def _matrix_to_quaternion_numba_impl(
    matrices: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert contiguous active rotation matrices in one loop."""
    count = matrices.shape[0]
    result = np.empty((count, 4), dtype=np.float64)
    status = np.zeros(count, dtype=np.uint8)
    margin = 128.0 * 2.220446049250313e-16 * (1.0 + tolerance) ** 3
    for index in range(count):
        matrix = matrices[index]
        maximum = 0.0
        finite = True
        for row in range(3):
            for column in range(3):
                component = matrix[row, column]
                if not np.isfinite(component):
                    finite = False
                    continue
                value = abs(component)
                maximum = max(maximum, value)
        if not finite:
            status[index] = 3
            continue
        if maximum > 1.0 + tolerance:
            status[index] = 1
            continue

        invalid = False
        uncertain = False
        for column_a in range(3):
            for column_b in range(3):
                total = (
                    matrix[0, column_a] * matrix[0, column_b]
                    + matrix[1, column_a] * matrix[1, column_b]
                    + matrix[2, column_a] * matrix[2, column_b]
                )
                target = 1.0 if column_a == column_b else 0.0
                error = abs(total - target)
                if error > tolerance + margin:
                    invalid = True
                elif error >= tolerance - margin:
                    uncertain = True
        determinant = (
            matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
            - matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
            + matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
        )
        determinant_error = abs(determinant - 1.0)
        if determinant_error > tolerance + margin:
            invalid = True
        elif determinant_error >= tolerance - margin:
            uncertain = True
        if invalid:
            status[index] = 1
            continue
        if uncertain:
            status[index] = 2

        trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * scale
            x = (matrix[2, 1] - matrix[1, 2]) / scale
            y = (matrix[0, 2] - matrix[2, 0]) / scale
            z = (matrix[1, 0] - matrix[0, 1]) / scale
        elif matrix[0, 0] >= matrix[1, 1] and matrix[0, 0] >= matrix[2, 2]:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif matrix[1, 1] >= matrix[2, 2]:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale

        magnitude = max(abs(w), abs(x), abs(y), abs(z))
        w /= magnitude
        x /= magnitude
        y /= magnitude
        z /= magnitude
        norm = np.sqrt(w * w + x * x + y * y + z * z)
        w /= norm
        x /= norm
        y /= norm
        z /= norm
        first = x if x != 0.0 else (y if y != 0.0 else z)
        if w < 0.0 or (w == 0.0 and first < 0.0):
            w = -w
            x = -x
            y = -y
            z = -z
        result[index, 0] = w
        result[index, 1] = x
        result[index, 2] = y
        result[index, 3] = z
    return result, status


def _rotation_matrix_numba_impl(
    quaternions: np.ndarray, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, bool, bool]:
    """Build active rotation matrices without NumPy's temporary arrays."""
    count = quaternions.shape[0]
    result = np.empty((count, 3, 3), dtype=np.float64)
    finite = True
    nonzero = True
    for index in range(count):
        if scalar_last:
            x, y, z, w = (
                quaternions[index, 0],
                quaternions[index, 1],
                quaternions[index, 2],
                quaternions[index, 3],
            )
        else:
            w, x, y, z = (
                quaternions[index, 0],
                quaternions[index, 1],
                quaternions[index, 2],
                quaternions[index, 3],
            )
        if not (
            np.isfinite(w) and np.isfinite(x) and np.isfinite(y) and np.isfinite(z)
        ):
            finite = False
            result[index, :, :] = np.nan
            continue
        scale = max(abs(w), abs(x), abs(y), abs(z))
        if scale == 0.0:
            nonzero = False
            result[index, :, :] = np.nan
            continue
        w /= scale
        x /= scale
        y /= scale
        z /= scale
        denominator = w * w + x * x + y * y + z * z
        result[index, 0, 0] = (w * w + x * x - y * y - z * z) / denominator
        result[index, 0, 1] = 2.0 * (x * y - epsilon * w * z) / denominator
        result[index, 0, 2] = 2.0 * (x * z + epsilon * w * y) / denominator
        result[index, 1, 0] = 2.0 * (x * y + epsilon * w * z) / denominator
        result[index, 1, 1] = (w * w - x * x + y * y - z * z) / denominator
        result[index, 1, 2] = 2.0 * (y * z - epsilon * w * x) / denominator
        result[index, 2, 0] = 2.0 * (x * z - epsilon * w * y) / denominator
        result[index, 2, 1] = 2.0 * (y * z + epsilon * w * x) / denominator
        result[index, 2, 2] = (w * w - x * x - y * y + z * z) / denominator
    return result, finite, nonzero


def _quaternion_matrix_numba_impl(
    quaternions: np.ndarray,
    epsilon: int,
    scalar_last: bool,
    right: bool,
) -> np.ndarray:
    """Build left or right 4x4 multiplication matrices in one batch pass."""
    result = np.empty((quaternions.shape[0], 4, 4), dtype=np.float64)
    cross = -epsilon if right else epsilon
    for index in range(quaternions.shape[0]):
        if scalar_last:
            x, y, z, w = (
                quaternions[index, 0],
                quaternions[index, 1],
                quaternions[index, 2],
                quaternions[index, 3],
            )
            result[index, 0, 0] = w
            result[index, 0, 1] = -cross * z
            result[index, 0, 2] = cross * y
            result[index, 0, 3] = x
            result[index, 1, 0] = cross * z
            result[index, 1, 1] = w
            result[index, 1, 2] = -cross * x
            result[index, 1, 3] = y
            result[index, 2, 0] = -cross * y
            result[index, 2, 1] = cross * x
            result[index, 2, 2] = w
            result[index, 2, 3] = z
            result[index, 3, 0] = -x
            result[index, 3, 1] = -y
            result[index, 3, 2] = -z
            result[index, 3, 3] = w
        else:
            w, x, y, z = (
                quaternions[index, 0],
                quaternions[index, 1],
                quaternions[index, 2],
                quaternions[index, 3],
            )
            result[index, 0, 0] = w
            result[index, 0, 1] = -x
            result[index, 0, 2] = -y
            result[index, 0, 3] = -z
            result[index, 1, 0] = x
            result[index, 1, 1] = w
            result[index, 1, 2] = -cross * z
            result[index, 1, 3] = cross * y
            result[index, 2, 0] = y
            result[index, 2, 1] = cross * z
            result[index, 2, 2] = w
            result[index, 2, 3] = -cross * x
            result[index, 3, 0] = z
            result[index, 3, 1] = -cross * y
            result[index, 3, 2] = cross * x
            result[index, 3, 3] = w
    return result


def _from_rotvec_numba_impl(
    vectors: np.ndarray, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, bool]:
    """Construct canonical quaternions directly in the requested layout."""
    count = vectors.shape[0]
    result = np.empty((count, 4), dtype=np.float64)
    overflow = False
    for index in range(count):
        rx, ry, rz = vectors[index, 0], vectors[index, 1], vectors[index, 2]
        scale = max(abs(rx), abs(ry), abs(rz))
        if scale == 0.0:
            if scalar_last:
                result[index, 0] = 0.0
                result[index, 1] = 0.0
                result[index, 2] = 0.0
                result[index, 3] = 1.0
            else:
                result[index, 0] = 1.0
                result[index, 1] = 0.0
                result[index, 2] = 0.0
                result[index, 3] = 0.0
            continue

        if 1e-150 < scale < 1e150:
            # Typical finite rotations can use the direct norm and factor.
            # The bounds keep the sum of squares away from underflow/overflow;
            # extreme magnitudes use the scaled calculation below.
            angle = sqrt(rx * rx + ry * ry + rz * rz)
            if not isfinite(angle):
                overflow = True
                result[index, :] = np.nan
                continue
            half_angle = 0.5 * angle
            w = cos(half_angle)
            if abs(angle) == np.pi:
                w = 0.0
            factor = sin(half_angle) / angle
            x, y, z = rx * factor, ry * factor, rz * factor
        else:
            sx, sy, sz = rx / scale, ry / scale, rz / scale
            direction_norm = sqrt(sx * sx + sy * sy + sz * sz)
            angle = scale * direction_norm
            if not isfinite(angle):
                overflow = True
                result[index, :] = np.nan
                continue
            half_angle = 0.5 * angle
            w = cos(half_angle)
            if abs(angle) == np.pi:
                w = 0.0
            sine = sin(half_angle)
            x = (sx / direction_norm) * sine
            y = (sy / direction_norm) * sine
            z = (sz / direction_norm) * sine
        first = x if x != 0.0 else (y if y != 0.0 else z)
        if w < 0.0 or (w == 0.0 and first < 0.0):
            w, x, y, z = -w, -x, -y, -z
        x *= epsilon
        y *= epsilon
        z *= epsilon
        if scalar_last:
            result[index, 0] = 0.0 if x == 0.0 else x
            result[index, 1] = 0.0 if y == 0.0 else y
            result[index, 2] = 0.0 if z == 0.0 else z
            result[index, 3] = 0.0 if w == 0.0 else w
        else:
            result[index, 0] = 0.0 if w == 0.0 else w
            result[index, 1] = 0.0 if x == 0.0 else x
            result[index, 2] = 0.0 if y == 0.0 else y
            result[index, 3] = 0.0 if z == 0.0 else z
    return result, overflow


def _to_rotvec_numba_impl(
    quaternions: np.ndarray, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, bool]:
    """Extract canonical physical rotation vectors with stable tiny limits."""
    count = quaternions.shape[0]
    result = np.empty((count, 3), dtype=np.float64)
    valid = True
    for index in range(count):
        if scalar_last:
            x, y, z, w = (
                quaternions[index, 0],
                quaternions[index, 1],
                quaternions[index, 2],
                quaternions[index, 3],
            )
        else:
            w, x, y, z = (
                quaternions[index, 0],
                quaternions[index, 1],
                quaternions[index, 2],
                quaternions[index, 3],
            )
        if w == 0.0 and x == 0.0 and y == 0.0 and z == 0.0:
            valid = False
            result[index, :] = np.nan
            continue

        first = x if x != 0.0 else (y if y != 0.0 else z)
        flip = w < 0.0 or (
            w == 0.0
            and ((epsilon > 0 and first < 0.0) or (epsilon < 0 and first > 0.0))
        )
        if flip:
            w, x, y, z = -w, -x, -y, -z
        if w == 0.0:
            w = 0.0
        if x == 0.0:
            x = 0.0
        if y == 0.0:
            y = 0.0
        if z == 0.0:
            z = 0.0

        vx, vy, vz = epsilon * x, epsilon * y, epsilon * z
        vector_scale = max(abs(vx), abs(vy), abs(vz))
        if 1e-150 < vector_scale < 1e150:
            # For ordinary vector magnitudes, the direct norm avoids the
            # scaling divisions needed by the extreme-value fallback below.
            vector_norm = sqrt(vx * vx + vy * vy + vz * vz)
            angle = 2.0 * atan2(vector_norm, w)
            tiny = w > 0.0 and vector_scale <= w * 1e-8
            if tiny:
                out_x, out_y, out_z = (
                    (2.0 * vx) / w,
                    (2.0 * vy) / w,
                    (2.0 * vz) / w,
                )
            else:
                factor = angle / vector_norm
                out_x, out_y, out_z = vx * factor, vy * factor, vz * factor
        else:
            if vector_scale == 0.0:
                length = 0.0
                axis_x, axis_y, axis_z = 1.0, 0.0, 0.0
            else:
                sx, sy, sz = vx / vector_scale, vy / vector_scale, vz / vector_scale
                length = sqrt(sx * sx + sy * sy + sz * sz)
                axis_x, axis_y, axis_z = sx / length, sy / length, sz / length

            scale = max(w, vector_scale)
            angle = 2.0 * atan2((vector_scale / scale) * length, w / scale)
            tiny = w > 0.0 and vector_scale <= w * 1e-8
            if tiny:
                vector_mantissa, vector_exponent = frexp(vector_scale)
                scalar_mantissa, scalar_exponent = frexp(w)
                angle = ldexp(
                    (2.0 * vector_mantissa / scalar_mantissa) * length,
                    vector_exponent - scalar_exponent,
                )

            if angle == 0.0:
                angle = 0.0
            if tiny:
                out_x, out_y, out_z = (
                    (2.0 * vx) / w,
                    (2.0 * vy) / w,
                    (2.0 * vz) / w,
                )
            else:
                out_x = axis_x * angle
                out_y = axis_y * angle
                out_z = axis_z * angle
        result[index, 0] = 0.0 if out_x == 0.0 else out_x
        result[index, 1] = 0.0 if out_y == 0.0 else out_y
        result[index, 2] = 0.0 if out_z == 0.0 else out_z
    return result, valid


def _integrate_angular_velocity_numba_impl(
    attitudes: np.ndarray,
    rates: np.ndarray,
    times: np.ndarray,
    world_frame: bool,
    midpoint: bool,
    epsilon: int,
    scalar_last: bool,
) -> tuple[np.ndarray, int]:
    """Integrate each sampled trajectory using physical Hamilton components."""
    count, samples = rates.shape[:2]
    result = np.empty((count, 4), dtype=np.float64)
    status = 0
    for row in range(count):
        if scalar_last:
            x, y, z, w = (
                attitudes[row, 0],
                attitudes[row, 1],
                attitudes[row, 2],
                attitudes[row, 3],
            )
        else:
            w, x, y, z = (
                attitudes[row, 0],
                attitudes[row, 1],
                attitudes[row, 2],
                attitudes[row, 3],
            )
        if epsilon < 0:
            x, y, z = -x, -y, -z

        scale = max(abs(w), abs(x), abs(y), abs(z))
        if scale == 0.0:
            status = 1
            break
        norm = sqrt(
            (w / scale) * (w / scale)
            + (x / scale) * (x / scale)
            + (y / scale) * (y / scale)
            + (z / scale) * (z / scale)
        )
        w, x, y, z = (
            (w / scale) / norm,
            (x / scale) / norm,
            (y / scale) / norm,
            (z / scale) / norm,
        )

        for sample in range(samples - 1):
            dt = times[row, sample + 1] - times[row, sample]
            if not isfinite(dt):
                status = 3
                break
            if dt <= 0.0:
                status = 2
                break

            if midpoint:
                rx = (rates[row, sample, 0] + rates[row, sample + 1, 0]) * 0.5
                ry = (rates[row, sample, 1] + rates[row, sample + 1, 1]) * 0.5
                rz = (rates[row, sample, 2] + rates[row, sample + 1, 2]) * 0.5
                if not isfinite(rx) or not isfinite(ry) or not isfinite(rz):
                    status = 4
                    break
            else:
                rx, ry, rz = (
                    rates[row, sample, 0],
                    rates[row, sample, 1],
                    rates[row, sample, 2],
                )
            rx, ry, rz = rx * dt, ry * dt, rz * dt
            if not isfinite(rx) or not isfinite(ry) or not isfinite(rz):
                status = 5
                break

            vector_scale = max(abs(rx), abs(ry), abs(rz))
            if vector_scale == 0.0:
                dw, dx, dy, dz = 1.0, 0.0, 0.0, 0.0
            else:
                sx, sy, sz = rx / vector_scale, ry / vector_scale, rz / vector_scale
                vector_norm = sqrt(sx * sx + sy * sy + sz * sz)
                angle = vector_scale * vector_norm
                if not isfinite(angle):
                    status = 6
                    break
                half_angle = 0.5 * angle
                dw = cos(half_angle)
                if angle == np.pi:
                    dw = 0.0
                sine = sin(half_angle)
                dx = (sx / vector_norm) * sine
                dy = (sy / vector_norm) * sine
                dz = (sz / vector_norm) * sine
                first = dx if dx != 0.0 else (dy if dy != 0.0 else dz)
                if dw < 0.0 or (dw == 0.0 and first < 0.0):
                    dw, dx, dy, dz = -dw, -dx, -dy, -dz

            if world_frame:
                nw = dw * w - dx * x - dy * y - dz * z
                nx = dw * x + dx * w + dy * z - dz * y
                ny = dw * y - dx * z + dy * w + dz * x
                nz = dw * z + dx * y - dy * x + dz * w
            else:
                nw = w * dw - x * dx - y * dy - z * dz
                nx = w * dx + dw * x + y * dz - z * dy
                ny = w * dy + dw * y + z * dx - x * dz
                nz = w * dz + dw * z + x * dy - y * dx

            product_scale = max(abs(nw), abs(nx), abs(ny), abs(nz))
            product_norm = sqrt(
                (nw / product_scale) * (nw / product_scale)
                + (nx / product_scale) * (nx / product_scale)
                + (ny / product_scale) * (ny / product_scale)
                + (nz / product_scale) * (nz / product_scale)
            )
            w, x, y, z = (
                (nw / product_scale) / product_norm,
                (nx / product_scale) / product_norm,
                (ny / product_scale) / product_norm,
                (nz / product_scale) / product_norm,
            )

        if status != 0:
            break
        if epsilon < 0:
            x, y, z = -x, -y, -z
        if scalar_last:
            result[row, 0], result[row, 1], result[row, 2], result[row, 3] = x, y, z, w
        else:
            result[row, 0], result[row, 1], result[row, 2], result[row, 3] = w, x, y, z
    return result, status


def _nlerp_numba_impl(
    first_values: np.ndarray,
    second_values: np.ndarray,
    amount: float,
    shortest: bool,
) -> tuple[np.ndarray, bool]:
    """Normalize linear quaternion blends without per-sample Python calls."""
    count = first_values.shape[0]
    result = np.empty((count, 4), dtype=np.float64)
    valid = True
    for index in range(count):
        a0, a1, a2, a3 = (
            first_values[index, 0],
            first_values[index, 1],
            first_values[index, 2],
            first_values[index, 3],
        )
        b0, b1, b2, b3 = (
            second_values[index, 0],
            second_values[index, 1],
            second_values[index, 2],
            second_values[index, 3],
        )
        scale_a = max(abs(a0), abs(a1), abs(a2), abs(a3))
        scale_b = max(abs(b0), abs(b1), abs(b2), abs(b3))
        if scale_a == 0.0 or scale_b == 0.0:
            valid = False
            break
        norm_a = sqrt(
            (a0 / scale_a) * (a0 / scale_a)
            + (a1 / scale_a) * (a1 / scale_a)
            + (a2 / scale_a) * (a2 / scale_a)
            + (a3 / scale_a) * (a3 / scale_a)
        )
        norm_b = sqrt(
            (b0 / scale_b) * (b0 / scale_b)
            + (b1 / scale_b) * (b1 / scale_b)
            + (b2 / scale_b) * (b2 / scale_b)
            + (b3 / scale_b) * (b3 / scale_b)
        )
        a0, a1, a2, a3 = (
            (a0 / scale_a) / norm_a,
            (a1 / scale_a) / norm_a,
            (a2 / scale_a) / norm_a,
            (a3 / scale_a) / norm_a,
        )
        b0, b1, b2, b3 = (
            (b0 / scale_b) / norm_b,
            (b1 / scale_b) / norm_b,
            (b2 / scale_b) / norm_b,
            (b3 / scale_b) / norm_b,
        )
        dot = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3
        if shortest and dot < 0.0:
            b0, b1, b2, b3 = -b0, -b1, -b2, -b3
        c0 = (1.0 - amount) * a0 + amount * b0
        c1 = (1.0 - amount) * a1 + amount * b1
        c2 = (1.0 - amount) * a2 + amount * b2
        c3 = (1.0 - amount) * a3 + amount * b3
        scale_c = max(abs(c0), abs(c1), abs(c2), abs(c3))
        if scale_c == 0.0 or not isfinite(scale_c):
            valid = False
            break
        norm_c = sqrt(
            (c0 / scale_c) * (c0 / scale_c)
            + (c1 / scale_c) * (c1 / scale_c)
            + (c2 / scale_c) * (c2 / scale_c)
            + (c3 / scale_c) * (c3 / scale_c)
        )
        result[index, 0] = (c0 / scale_c) / norm_c
        result[index, 1] = (c1 / scale_c) / norm_c
        result[index, 2] = (c2 / scale_c) / norm_c
        result[index, 3] = (c3 / scale_c) / norm_c
        for component in range(4):
            if not isfinite(result[index, component]):
                valid = False
    return result, valid


def _slerp_numba_impl(
    first_values: np.ndarray,
    second_values: np.ndarray,
    amount: float,
    shortest: bool,
) -> tuple[np.ndarray, bool]:
    """Spherically interpolate rows using a fused, stable compiled loop."""
    count = first_values.shape[0]
    result = np.empty((count, 4), dtype=np.float64)
    valid = True
    for index in range(count):
        a0, a1, a2, a3 = (
            first_values[index, 0],
            first_values[index, 1],
            first_values[index, 2],
            first_values[index, 3],
        )
        b0, b1, b2, b3 = (
            second_values[index, 0],
            second_values[index, 1],
            second_values[index, 2],
            second_values[index, 3],
        )
        scale_a = max(abs(a0), abs(a1), abs(a2), abs(a3))
        scale_b = max(abs(b0), abs(b1), abs(b2), abs(b3))
        if scale_a == 0.0 or scale_b == 0.0:
            valid = False
            break
        norm_a = sqrt(
            (a0 / scale_a) * (a0 / scale_a)
            + (a1 / scale_a) * (a1 / scale_a)
            + (a2 / scale_a) * (a2 / scale_a)
            + (a3 / scale_a) * (a3 / scale_a)
        )
        norm_b = sqrt(
            (b0 / scale_b) * (b0 / scale_b)
            + (b1 / scale_b) * (b1 / scale_b)
            + (b2 / scale_b) * (b2 / scale_b)
            + (b3 / scale_b) * (b3 / scale_b)
        )
        a0, a1, a2, a3 = (
            (a0 / scale_a) / norm_a,
            (a1 / scale_a) / norm_a,
            (a2 / scale_a) / norm_a,
            (a3 / scale_a) / norm_a,
        )
        b0, b1, b2, b3 = (
            (b0 / scale_b) / norm_b,
            (b1 / scale_b) / norm_b,
            (b2 / scale_b) / norm_b,
            (b3 / scale_b) / norm_b,
        )
        dot = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3
        if shortest and dot < 0.0:
            b0, b1, b2, b3 = -b0, -b1, -b2, -b3
            dot = -dot
        dot = min(1.0, max(-1.0, dot))
        theta = acos(dot)
        sine = sin(theta)
        if abs(sine) <= 1e-15 and dot < 0.0:
            valid = False
            break
        denominator = 1.0 if abs(sine) <= 1e-15 else sine
        weight_a = sin((1.0 - amount) * theta) / denominator
        weight_b = sin(amount * theta) / denominator
        s0 = weight_a * a0 + weight_b * b0
        s1 = weight_a * a1 + weight_b * b1
        s2 = weight_a * a2 + weight_b * b2
        s3 = weight_a * a3 + weight_b * b3
        if dot > 1.0 - 1e-12:
            s0 = (1.0 - amount) * a0 + amount * b0
            s1 = (1.0 - amount) * a1 + amount * b1
            s2 = (1.0 - amount) * a2 + amount * b2
            s3 = (1.0 - amount) * a3 + amount * b3
        scale_s = max(abs(s0), abs(s1), abs(s2), abs(s3))
        if scale_s == 0.0 or not isfinite(scale_s):
            valid = False
            break
        norm_s = sqrt(
            (s0 / scale_s) * (s0 / scale_s)
            + (s1 / scale_s) * (s1 / scale_s)
            + (s2 / scale_s) * (s2 / scale_s)
            + (s3 / scale_s) * (s3 / scale_s)
        )
        result[index, 0] = (s0 / scale_s) / norm_s
        result[index, 1] = (s1 / scale_s) / norm_s
        result[index, 2] = (s2 / scale_s) / norm_s
        result[index, 3] = (s3 / scale_s) / norm_s
        for component in range(4):
            if not isfinite(result[index, component]):
                valid = False
    return result, valid


def _from_mrp_numba_impl(
    mrps: np.ndarray, epsilon: int, scalar_last: bool
) -> np.ndarray:
    """Convert physical MRP vectors to canonical convention-specific q values."""
    count = mrps.shape[0]
    result = np.empty((count, 4), dtype=np.float64)
    for index in range(count):
        x, y, z = mrps[index, 0], mrps[index, 1], mrps[index, 2]
        scale = max(abs(x), abs(y), abs(z))
        if scale == 0.0:
            w, x, y, z = 1.0, 0.0, 0.0, 0.0
        elif scale <= 1.0:
            norm2 = x * x + y * y + z * z
            denominator = 1.0 + norm2
            w = (1.0 - norm2) / denominator
            x, y, z = (
                2.0 * x / denominator,
                2.0 * y / denominator,
                2.0 * z / denominator,
            )
        else:
            sx, sy, sz = x / scale, y / scale, z / scale
            inverse_scale = 1.0 / scale
            norm2 = sx * sx + sy * sy + sz * sz
            denominator = inverse_scale * inverse_scale + norm2
            w = (inverse_scale * inverse_scale - norm2) / denominator
            x = (2.0 * inverse_scale * sx) / denominator
            y = (2.0 * inverse_scale * sy) / denominator
            z = (2.0 * inverse_scale * sz) / denominator

        scale_q = max(abs(w), abs(x), abs(y), abs(z))
        norm_q = sqrt(
            (w / scale_q) * (w / scale_q)
            + (x / scale_q) * (x / scale_q)
            + (y / scale_q) * (y / scale_q)
            + (z / scale_q) * (z / scale_q)
        )
        w, x, y, z = (
            (w / scale_q) / norm_q,
            (x / scale_q) / norm_q,
            (y / scale_q) / norm_q,
            (z / scale_q) / norm_q,
        )
        first = x if x != 0.0 else (y if y != 0.0 else z)
        if w < 0.0 or (w == 0.0 and first < 0.0):
            w, x, y, z = -w, -x, -y, -z
        x, y, z = epsilon * x, epsilon * y, epsilon * z
        if scalar_last:
            result[index, 0], result[index, 1], result[index, 2], result[index, 3] = (
                x,
                y,
                z,
                w,
            )
        else:
            result[index, 0], result[index, 1], result[index, 2], result[index, 3] = (
                w,
                x,
                y,
                z,
            )
    return result


def _to_mrp_numba_impl(
    quaternions: np.ndarray, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, bool]:
    """Convert arbitrary nonzero quaternion magnitudes to principal MRPs."""
    count = quaternions.shape[0]
    result = np.empty((count, 3), dtype=np.float64)
    valid = True
    for index in range(count):
        if scalar_last:
            x, y, z, w = (
                quaternions[index, 0],
                quaternions[index, 1],
                quaternions[index, 2],
                quaternions[index, 3],
            )
        else:
            w, x, y, z = (
                quaternions[index, 0],
                quaternions[index, 1],
                quaternions[index, 2],
                quaternions[index, 3],
            )
        scale = max(abs(w), abs(x), abs(y), abs(z))
        if scale == 0.0:
            valid = False
            result[index, :] = np.nan
            continue
        norm = sqrt(
            (w / scale) * (w / scale)
            + (x / scale) * (x / scale)
            + (y / scale) * (y / scale)
            + (z / scale) * (z / scale)
        )
        w, x, y, z = (
            (w / scale) / norm,
            (x / scale) / norm,
            (y / scale) / norm,
            (z / scale) / norm,
        )
        x, y, z = epsilon * x, epsilon * y, epsilon * z
        first = x if x != 0.0 else (y if y != 0.0 else z)
        if w < 0.0 or (w == 0.0 and first < 0.0):
            w, x, y, z = -w, -x, -y, -z
        denominator = 1.0 + w
        if denominator == 0.0:
            valid = False
            result[index, :] = np.nan
            continue
        result[index, 0] = 0.0 if x == 0.0 else x / denominator
        result[index, 1] = 0.0 if y == 0.0 else y / denominator
        result[index, 2] = 0.0 if z == 0.0 else z / denominator
    return result, valid


def _normalize_numba_impl(
    quaternions: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Normalize rows with scaled arithmetic and no intermediate arrays."""
    count = quaternions.shape[0]
    result = np.empty((count, 4), dtype=np.float64)
    valid = True
    for index in range(count):
        w, x, y, z = (
            quaternions[index, 0],
            quaternions[index, 1],
            quaternions[index, 2],
            quaternions[index, 3],
        )
        scale = max(abs(w), abs(x), abs(y), abs(z))
        if scale == 0.0:
            valid = False
            result[index, :] = np.nan
            continue
        norm = sqrt(
            (w / scale) * (w / scale)
            + (x / scale) * (x / scale)
            + (y / scale) * (y / scale)
            + (z / scale) * (z / scale)
        )
        result[index, 0] = (w / scale) / norm
        result[index, 1] = (x / scale) / norm
        result[index, 2] = (y / scale) / norm
        result[index, 3] = (z / scale) / norm
    return result, valid


def _inverse_numba_impl(
    quaternions: np.ndarray, scalar_last: bool
) -> tuple[np.ndarray, bool]:
    """Invert rows while preserving the vectorized API's overflow behavior."""
    count = quaternions.shape[0]
    result = np.empty((count, 4), dtype=np.float64)
    valid = True
    for index in range(count):
        values = (
            quaternions[index, 0],
            quaternions[index, 1],
            quaternions[index, 2],
            quaternions[index, 3],
        )
        scale = max(abs(values[0]), abs(values[1]), abs(values[2]), abs(values[3]))
        if scale == 0.0:
            valid = False
            result[index, :] = np.nan
            continue
        w, x, y, z = values
        parts = (w / scale, x / scale, y / scale, z / scale)
        denominator = (
            parts[0] * parts[0]
            + parts[1] * parts[1]
            + parts[2] * parts[2]
            + parts[3] * parts[3]
        )
        if scalar_last:
            inverse_values = (
                (-parts[0] / denominator) / scale,
                (-parts[1] / denominator) / scale,
                (-parts[2] / denominator) / scale,
                (parts[3] / denominator) / scale,
            )
        else:
            inverse_values = (
                (parts[0] / denominator) / scale,
                (-parts[1] / denominator) / scale,
                (-parts[2] / denominator) / scale,
                (-parts[3] / denominator) / scale,
            )
        for component in range(4):
            result[index, component] = inverse_values[component]
            if not isfinite(inverse_values[component]):
                valid = False
    return result, valid


def _norm_numba_impl(quaternions: np.ndarray) -> tuple[np.ndarray, bool]:
    """Compute scaled quaternion norms in one compiled batch loop."""
    count = quaternions.shape[0]
    result = np.empty(count, dtype=np.float64)
    valid = True
    for index in range(count):
        w, x, y, z = (
            quaternions[index, 0],
            quaternions[index, 1],
            quaternions[index, 2],
            quaternions[index, 3],
        )
        scale = max(abs(w), abs(x), abs(y), abs(z))
        length = (
            sqrt(
                (w / scale) * (w / scale)
                + (x / scale) * (x / scale)
                + (y / scale) * (y / scale)
                + (z / scale) * (z / scale)
            )
            if scale != 0.0
            else 0.0
        )
        result[index] = scale * length
        if not isfinite(result[index]):
            valid = False
    return result, valid


def _dot_numba_impl(
    left: np.ndarray,
    right: np.ndarray,
    result: np.ndarray,
    exact: np.ndarray,
) -> None:
    """Compute a fast dot and flag products that need exact scalar arithmetic."""
    tiny = 2.2250738585072014e-308
    p0 = left[0] * right[0]
    p1 = left[1] * right[1]
    p2 = left[2] * right[2]
    p3 = left[3] * right[3]
    result[0] = (p0 + p1) + (p2 + p3)
    exact[0] = (
        not isfinite(result[0])
        or result[0] == 0.0
        or (p0 != 0.0 and abs(p0) < tiny)
        or (p1 != 0.0 and abs(p1) < tiny)
        or (p2 != 0.0 and abs(p2) < tiny)
        or (p3 != 0.0 and abs(p3) < tiny)
    )


def _analytic_numba_impl(
    quaternions: np.ndarray, operation: int, scalar_last: bool
) -> tuple[np.ndarray, int]:
    """Apply scalar-slice quaternion analytic functions in one batch loop."""
    result = np.empty_like(quaternions)
    for index in range(quaternions.shape[0]):
        if scalar_last:
            x = quaternions[index, 0]
            y = quaternions[index, 1]
            z = quaternions[index, 2]
            w = quaternions[index, 3]
        else:
            w = quaternions[index, 0]
            x = quaternions[index, 1]
            y = quaternions[index, 2]
            z = quaternions[index, 3]
        if operation == 0:
            radius_squared = x * x + y * y + z * z
            if 1e-300 < radius_squared < 1e300:
                radius = sqrt(radius_squared)
            elif x == 0.0 and y == 0.0 and z == 0.0:
                radius = 0.0
            else:
                radius = hypot(hypot(x, y), z)
            magnitude = exp(w)
            scalar = magnitude * cos(radius)
            if radius == 0.0:
                vector_x = vector_y = vector_z = 0.0
            else:
                factor = magnitude * sin(radius) / radius
                vector_x, vector_y, vector_z = factor * x, factor * y, factor * z
            if scalar_last:
                result[index, 0] = vector_x
                result[index, 1] = vector_y
                result[index, 2] = vector_z
                result[index, 3] = scalar
            else:
                result[index, 0] = scalar
                result[index, 1] = vector_x
                result[index, 2] = vector_y
                result[index, 3] = vector_z
            continue
        if operation == 1:
            scale = max(abs(w), abs(x), abs(y), abs(z))
            if scale == 0.0:
                return result, 1
            if 1e-150 < scale < 1e150:
                radius = sqrt(x * x + y * y + z * z)
                norm_squared = w * w + x * x + y * y + z * z
                scalar = 0.5 * log(norm_squared)
            else:
                radius = hypot(hypot(x, y), z)
                norm_scale = max(abs(w), radius)
                scalar = log(norm_scale) + log(
                    hypot(w / norm_scale, radius / norm_scale)
                )
            if radius == 0.0:
                vector_x = pi if w < 0.0 else 0.0
                vector_y = vector_z = 0.0
            elif radius < 1e-150:
                angle = atan2(radius, w)
                vector_x = (x / radius) * angle
                vector_y = (y / radius) * angle
                vector_z = (z / radius) * angle
            else:
                factor = atan2(radius, w) / radius
                vector_x, vector_y, vector_z = factor * x, factor * y, factor * z
            if scalar_last:
                result[index, 0] = vector_x
                result[index, 1] = vector_y
                result[index, 2] = vector_z
                result[index, 3] = scalar
            else:
                result[index, 0] = scalar
                result[index, 1] = vector_x
                result[index, 2] = vector_y
                result[index, 3] = vector_z
            continue
        radius = hypot(hypot(x, y), z)
        value = complex(w, radius)
        if operation == 2:
            transformed = cmath.sin(value)
        elif operation == 3:
            transformed = cmath.cos(value)
        elif operation == 4:
            transformed = cmath.sinh(value)
        elif operation == 5:
            transformed = cmath.cosh(value)
        elif operation == 6:
            transformed = cmath.tan(value)
        elif operation == 7:
            transformed = cmath.tanh(value)
        elif operation == 8:
            transformed = cmath.asin(value)
        elif operation == 9:
            transformed = cmath.acos(value)
        elif operation == 10:
            transformed = cmath.atan(value)
        elif operation == 11:
            transformed = cmath.asinh(value)
        elif operation == 12:
            transformed = cmath.acosh(value)
        elif operation == 13:
            transformed = cmath.atanh(value)
        elif operation == 14:
            if abs(value) >= 0.5:
                transformed = cmath.exp(value) - 1.0
            else:
                term = value
                transformed = value
                for power in range(2, 48):
                    term = term * value / power
                    transformed += term
                    if abs(term) <= 2e-17 * max(1.0, abs(transformed)):
                        break
        elif operation == 15:
            if abs(value) >= 0.25:
                shifted = complex(1.0 + w, radius)
                if shifted.real == 0.0 and shifted.imag == 0.0:
                    return result, 1
                transformed = cmath.log(shifted)
                if radius == 0.0 and shifted.real < 0.0:
                    transformed = complex(log(-shifted.real), pi)
            else:
                term = value
                transformed = value
                for power in range(2, 96):
                    term = term * value
                    factor = 1.0 / power if power % 2 == 1 else -1.0 / power
                    transformed += factor * term
                    if abs(term) <= 2e-17 * max(1.0, abs(transformed)):
                        break
        elif operation == 16:
            if w == 0.0 and radius == 0.0:
                return result, 1
            transformed = cmath.log(value) / log(2.0)
        elif operation == 17:
            if w == 0.0 and radius == 0.0:
                return result, 1
            transformed = cmath.log(value) / log(10.0)
        else:
            return result, 2

        if radius == 0.0:
            vector_x = transformed.imag
            vector_y = 0.0
            vector_z = 0.0
        else:
            factor = transformed.imag / radius
            vector_x = factor * x
            vector_y = factor * y
            vector_z = factor * z
        if scalar_last:
            result[index, 0] = vector_x
            result[index, 1] = vector_y
            result[index, 2] = vector_z
            result[index, 3] = transformed.real
        else:
            result[index, 0] = transformed.real
            result[index, 1] = vector_x
            result[index, 2] = vector_y
            result[index, 3] = vector_z
    return result, 0


def _quaternion_product_row_numba(
    lhs: np.ndarray, rhs: np.ndarray, epsilon: int, scalar_last: bool
) -> np.ndarray:
    """Multiply two coefficient rows in either convention and storage order."""
    result = np.empty(4, dtype=np.float64)
    if scalar_last:
        ax, ay, az, aw = lhs[0], lhs[1], lhs[2], lhs[3]
        bx, by, bz, bw = rhs[0], rhs[1], rhs[2], rhs[3]
    else:
        aw, ax, ay, az = lhs[0], lhs[1], lhs[2], lhs[3]
        bw, bx, by, bz = rhs[0], rhs[1], rhs[2], rhs[3]
    scalar = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + bw * ax + epsilon * (ay * bz - az * by)
    y = aw * by + bw * ay + epsilon * (az * bx - ax * bz)
    z = aw * bz + bw * az + epsilon * (ax * by - ay * bx)
    if scalar_last:
        result[0], result[1], result[2], result[3] = x, y, z, scalar
    else:
        result[0], result[1], result[2], result[3] = scalar, x, y, z
    return result


def _inverse_row_numba(value: np.ndarray, scalar_last: bool) -> tuple[np.ndarray, bool]:
    """Invert a quaternion row using scaled coefficients."""
    scale = max(abs(value[0]), abs(value[1]), abs(value[2]), abs(value[3]))
    result = np.empty(4, dtype=np.float64)
    if scale == 0.0:
        return result, False
    total = 0.0
    for component in range(4):
        normalized = value[component] / scale
        total += normalized * normalized
    for component in range(4):
        sign = -1.0 if (component < 3 if scalar_last else component > 0) else 1.0
        result[component] = ((value[component] / scale) * sign / total) / scale
    return result, True


def _divide_numba_impl(
    lhs: np.ndarray,
    rhs: np.ndarray,
    epsilon: int,
    scalar_last: bool,
    left: bool,
) -> tuple[np.ndarray, int]:
    """Fuse ordinary-range quaternion division over a batch."""
    result = np.empty_like(lhs)
    denominator_values = lhs if left else rhs
    numerator_values = rhs if left else lhs
    cross_sign = -epsilon if left else epsilon
    for index in range(lhs.shape[0]):
        denominator = denominator_values[index]
        scale = max(
            abs(denominator[0]),
            abs(denominator[1]),
            abs(denominator[2]),
            abs(denominator[3]),
        )
        if scale == 0.0:
            return result, 1
        if scale < 1e-150 or scale > 1e150:
            return result, 2
    for index in range(lhs.shape[0]):
        denominator = denominator_values[index]
        numerator = numerator_values[index]
        if scalar_last:
            denominator_x, denominator_y, denominator_z, denominator_w = (
                denominator[0],
                denominator[1],
                denominator[2],
                denominator[3],
            )
            numerator_x, numerator_y, numerator_z, numerator_w = (
                numerator[0],
                numerator[1],
                numerator[2],
                numerator[3],
            )
        else:
            denominator_w, denominator_x, denominator_y, denominator_z = (
                denominator[0],
                denominator[1],
                denominator[2],
                denominator[3],
            )
            numerator_w, numerator_x, numerator_y, numerator_z = (
                numerator[0],
                numerator[1],
                numerator[2],
                numerator[3],
            )
        denominator_norm_squared = (
            (denominator_w * denominator_w + denominator_x * denominator_x)
            + denominator_y * denominator_y
        ) + denominator_z * denominator_z
        inverse_norm_squared = 1.0 / denominator_norm_squared
        scalar = (
            (denominator_w * numerator_w + denominator_x * numerator_x)
            + denominator_y * numerator_y
        ) + denominator_z * numerator_z
        result_w = scalar * inverse_norm_squared
        result_x = (
            denominator_w * numerator_x
            - numerator_w * denominator_x
            + cross_sign * (denominator_y * numerator_z - denominator_z * numerator_y)
        ) * inverse_norm_squared
        result_y = (
            denominator_w * numerator_y
            - numerator_w * denominator_y
            + cross_sign * (denominator_z * numerator_x - denominator_x * numerator_z)
        ) * inverse_norm_squared
        result_z = (
            denominator_w * numerator_z
            - numerator_w * denominator_z
            + cross_sign * (denominator_x * numerator_y - denominator_y * numerator_x)
        ) * inverse_norm_squared
        if scalar_last:
            result[index, 0], result[index, 1], result[index, 2], result[index, 3] = (
                result_x,
                result_y,
                result_z,
                result_w,
            )
        else:
            result[index, 0], result[index, 1], result[index, 2], result[index, 3] = (
                result_w,
                result_x,
                result_y,
                result_z,
            )
    return result, 0


def _log_row_numba(value: np.ndarray, scalar_last: bool) -> tuple[np.ndarray, bool]:
    """Compute the principal quaternion logarithm for one validated row."""
    if scalar_last:
        x, y, z, w = value[0], value[1], value[2], value[3]
    else:
        w, x, y, z = value[0], value[1], value[2], value[3]
    scale = max(abs(w), abs(x), abs(y), abs(z))
    result = np.zeros(4, dtype=np.float64)
    if scale == 0.0:
        return result, False
    sw, sx, sy, sz = w / scale, x / scale, y / scale, z / scale
    vector_norm = sqrt(sx * sx + sy * sy + sz * sz)
    norm_scaled = sqrt(sw * sw + vector_norm * vector_norm)
    real = log(scale) + log(norm_scaled)
    if vector_norm == 0.0:
        vx = pi if sw < 0.0 else 0.0
        vy, vz = 0.0, 0.0
    else:
        angle = atan2(vector_norm, sw)
        factor = angle / vector_norm
        vx, vy, vz = sx * factor, sy * factor, sz * factor
    if scalar_last:
        result[0], result[1], result[2], result[3] = vx, vy, vz, real
    else:
        result[0], result[1], result[2], result[3] = real, vx, vy, vz
    return result, True


def _exp_row_numba(value: np.ndarray, scalar_last: bool) -> np.ndarray:
    """Compute the quaternion exponential for one coefficient row."""
    if scalar_last:
        x, y, z, w = value[0], value[1], value[2], value[3]
    else:
        w, x, y, z = value[0], value[1], value[2], value[3]
    radius = hypot(hypot(x, y), z)
    magnitude = exp(w)
    scalar = magnitude * cos(radius)
    if radius == 0.0:
        vx, vy, vz = 0.0, 0.0, 0.0
    else:
        factor = magnitude * sin(radius) / radius
        vx, vy, vz = factor * x, factor * y, factor * z
    result = np.empty(4, dtype=np.float64)
    if scalar_last:
        result[0], result[1], result[2], result[3] = vx, vy, vz, scalar
    else:
        result[0], result[1], result[2], result[3] = scalar, vx, vy, vz
    return result


def _power_integer_numba_impl(
    quaternions: np.ndarray, exponent: int, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, int]:
    """Apply integer powers by squaring in one compiled batch loop."""
    result = np.empty_like(quaternions)
    for index in range(quaternions.shape[0]):
        n = exponent
        base = quaternions[index].copy()
        if n < 0:
            base, valid = _inverse_row_numba(base, scalar_last)
            if not valid:
                return result, 1
            n = -n
        accumulated = np.zeros(4, dtype=np.float64)
        accumulated[3 if scalar_last else 0] = 1.0
        while n:
            if n & 1:
                accumulated = _quaternion_product_row_numba(
                    accumulated, base, epsilon, scalar_last
                )
            n >>= 1
            if n:
                base = _quaternion_product_row_numba(base, base, epsilon, scalar_last)
        result[index] = accumulated
    return result, 0


def _power_real_numba_impl(
    quaternions: np.ndarray, exponent: float, scalar_last: bool
) -> tuple[np.ndarray, int]:
    """Apply principal real powers through the stable complex slice formula."""
    result = np.empty_like(quaternions)
    for index in range(quaternions.shape[0]):
        if scalar_last:
            x, y, z, w = (
                quaternions[index, 0],
                quaternions[index, 1],
                quaternions[index, 2],
                quaternions[index, 3],
            )
        else:
            w, x, y, z = (
                quaternions[index, 0],
                quaternions[index, 1],
                quaternions[index, 2],
                quaternions[index, 3],
            )
        radius = hypot(hypot(x, y), z)
        if w == 0.0 and radius == 0.0:
            if exponent < 0.0:
                return result, 1
            if exponent == 0.0:
                scalar, vx, vy, vz = 1.0, 0.0, 0.0, 0.0
            else:
                scalar, vx, vy, vz = 0.0, 0.0, 0.0, 0.0
        elif exponent == 0.0:
            scalar, vx, vy, vz = 1.0, 0.0, 0.0, 0.0
        else:
            transformed = cmath.exp(exponent * cmath.log(complex(w, radius)))
            if radius == 0.0:
                vx, vy, vz = transformed.imag, 0.0, 0.0
            else:
                factor = transformed.imag / radius
                vx, vy, vz = factor * x, factor * y, factor * z
            scalar = transformed.real
        if scalar_last:
            result[index, 0], result[index, 1], result[index, 2], result[index, 3] = (
                vx,
                vy,
                vz,
                scalar,
            )
        else:
            result[index, 0], result[index, 1], result[index, 2], result[index, 3] = (
                scalar,
                vx,
                vy,
                vz,
            )
    return result, 0


def _power_quaternion_numba_impl(
    quaternions: np.ndarray,
    exponents: np.ndarray,
    epsilon: int,
    scalar_last: bool,
) -> tuple[np.ndarray, int]:
    """Apply quaternion-valued powers with one fused batch pass."""
    result = np.empty_like(quaternions)
    for index in range(quaternions.shape[0]):
        zero_exponent = True
        for component in range(4):
            if exponents[index, component] != 0.0:
                zero_exponent = False
                break
        if zero_exponent:
            result[index] = 0.0
            result[index, 3 if scalar_last else 0] = 1.0
            continue
        logarithm, valid = _log_row_numba(quaternions[index], scalar_last)
        if not valid:
            return result, 1
        product = _quaternion_product_row_numba(
            exponents[index], logarithm, epsilon, scalar_last
        )
        result[index] = _exp_row_numba(product, scalar_last)
    return result, 0


def _from_axis_angle_numba_impl(
    axis_angles: np.ndarray, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, int]:
    """Convert physical axis-angle rows to convention-native quaternions."""
    result = np.empty((axis_angles.shape[0], 4), dtype=np.float64)
    for index in range(axis_angles.shape[0]):
        x, y, z, angle = (
            axis_angles[index, 0],
            axis_angles[index, 1],
            axis_angles[index, 2],
            axis_angles[index, 3],
        )
        scale = max(abs(x), abs(y), abs(z))
        if scale == 0.0:
            return result, 1
        x, y, z = x / scale, y / scale, z / scale
        length = sqrt(x * x + y * y + z * z)
        x, y, z = x / length, y / length, z / length
        half = angle * 0.5
        scalar = 0.0 if abs(angle) == pi else cos(half)
        vector_scale = sin(half) * epsilon
        x, y, z = x * vector_scale, y * vector_scale, z * vector_scale
        first = x if x != 0.0 else y if y != 0.0 else z
        flip = scalar < 0.0 or (
            scalar == 0.0 and (first < 0.0 if epsilon > 0 else first > 0.0)
        )
        if flip:
            scalar, x, y, z = -scalar, -x, -y, -z
        if scalar_last:
            result[index, 0] = 0.0 if x == 0.0 else x
            result[index, 1] = 0.0 if y == 0.0 else y
            result[index, 2] = 0.0 if z == 0.0 else z
            result[index, 3] = 0.0 if scalar == 0.0 else scalar
        else:
            result[index, 0] = 0.0 if scalar == 0.0 else scalar
            result[index, 1] = 0.0 if x == 0.0 else x
            result[index, 2] = 0.0 if y == 0.0 else y
            result[index, 3] = 0.0 if z == 0.0 else z
    return result, 0


def _to_axis_angle_numba_impl(
    quaternions: np.ndarray, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, int]:
    """Extract canonical physical axis-angle rows from quaternion data."""
    result = np.empty((quaternions.shape[0], 4), dtype=np.float64)
    for index in range(quaternions.shape[0]):
        if scalar_last:
            x, y, z, w = (
                quaternions[index, 0],
                quaternions[index, 1],
                quaternions[index, 2],
                quaternions[index, 3],
            )
        else:
            w, x, y, z = (
                quaternions[index, 0],
                quaternions[index, 1],
                quaternions[index, 2],
                quaternions[index, 3],
            )
        if w == 0.0 and x == 0.0 and y == 0.0 and z == 0.0:
            return result, 1
        first = x if x != 0.0 else y if y != 0.0 else z
        flip = w < 0.0 or (w == 0.0 and (first < 0.0 if epsilon > 0 else first > 0.0))
        if flip:
            w, x, y, z = -w, -x, -y, -z
        x, y, z = epsilon * x, epsilon * y, epsilon * z
        scale = max(abs(x), abs(y), abs(z))
        if scale == 0.0:
            axis_x, axis_y, axis_z, angle = 1.0, 0.0, 0.0, 0.0
        else:
            sx, sy, sz = x / scale, y / scale, z / scale
            length = sqrt(sx * sx + sy * sy + sz * sz)
            axis_x, axis_y, axis_z = sx / length, sy / length, sz / length
            if w > 0.0 and scale <= w * 1e-8:
                vm, ve = frexp(scale)
                wm, we = frexp(w)
                angle = ldexp((2.0 * (vm / wm)) * length, ve - we)
            else:
                base = max(w, scale)
                angle = 2.0 * atan2((scale / base) * length, w / base)
            if angle == 0.0:
                axis_x, axis_y, axis_z = 1.0, 0.0, 0.0
        result[index, 0] = 0.0 if axis_x == 0.0 else axis_x
        result[index, 1] = 0.0 if axis_y == 0.0 else axis_y
        result[index, 2] = 0.0 if axis_z == 0.0 else axis_z
        result[index, 3] = 0.0 if angle == 0.0 else angle
    return result, 0


def _so3_jacobian_numba_impl(
    vectors: np.ndarray, side: int, inverse_matrix: bool
) -> tuple[np.ndarray, int]:
    """Build SO(3) Jacobians in a fused compiled batch loop."""
    result = np.empty((vectors.shape[0], 3, 3), dtype=np.float64)
    for index in range(vectors.shape[0]):
        x, y, z = vectors[index, 0], vectors[index, 1], vectors[index, 2]
        theta = hypot(hypot(x, y), z)
        if not isfinite(theta):
            return result, 2
        if theta == 0.0:
            axis_x, axis_y, axis_z = 0.0, 0.0, 0.0
        else:
            axis_x, axis_y, axis_z = x / theta, y / theta, z / theta
        theta2 = theta * theta
        small = theta < 1e-4
        if inverse_matrix:
            if small:
                theta4 = theta2 * theta2
                a = -0.5 * side * theta
                b = theta2 * (1.0 / 12.0 + theta2 / 720.0 + theta4 / 30240.0)
            else:
                half = theta * 0.5
                sine_half = sin(half)
                if abs(sine_half) <= 1e-15:
                    return result, 1
                a = -0.5 * side * theta
                b = 1.0 - half * cos(half) / sine_half
        elif small:
            theta4 = theta2 * theta2
            a = side * theta * (0.5 - theta2 / 24.0 + theta4 / 720.0)
            b = theta2 * (1.0 / 6.0 - theta2 / 120.0 + theta4 / 5040.0)
        else:
            a = side * (1.0 - cos(theta)) / theta
            b = 1.0 - sin(theta) / theta

        k = np.empty((3, 3), dtype=np.float64)
        k[0, 0], k[0, 1], k[0, 2] = 0.0, -axis_z, axis_y
        k[1, 0], k[1, 1], k[1, 2] = axis_z, 0.0, -axis_x
        k[2, 0], k[2, 1], k[2, 2] = -axis_y, axis_x, 0.0
        for row in range(3):
            for column in range(3):
                squared = 0.0
                for inner in range(3):
                    squared += k[row, inner] * k[inner, column]
                result[index, row, column] = (
                    (1.0 if row == column else 0.0) + a * k[row, column] + b * squared
                )
    return result, 0


def _skew_numba_impl(vectors: np.ndarray) -> np.ndarray:
    """Build skew matrices in a fused compiled batch loop."""
    result = np.zeros((vectors.shape[0], 3, 3), dtype=np.float64)
    for index in range(vectors.shape[0]):
        x, y, z = vectors[index, 0], vectors[index, 1], vectors[index, 2]
        result[index, 0, 1] = -z
        result[index, 0, 2] = y
        result[index, 1, 0] = z
        result[index, 1, 2] = -x
        result[index, 2, 0] = -y
        result[index, 2, 1] = x
    return result


def _make_sign_continuous_numba_impl(
    sequences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Make quaternion sequences sign-continuous and flag uncertain dots."""
    result = np.empty_like(sequences)
    ambiguous = np.zeros(sequences.shape[0], dtype=np.bool_)
    for sequence in range(sequences.shape[0]):
        count = sequences.shape[1]
        if count == 0:
            continue
        for component in range(4):
            result[sequence, 0, component] = sequences[sequence, 0, component]
        for index in range(1, count):
            left_scale = 0.0
            right_scale = 0.0
            for component in range(4):
                left_scale = max(
                    left_scale, abs(result[sequence, index - 1, component])
                )
                right_scale = max(
                    right_scale, abs(sequences[sequence, index, component])
                )
            dot = 0.0
            magnitude = 0.0
            if left_scale != 0.0 and right_scale != 0.0:
                for component in range(4):
                    term = (result[sequence, index - 1, component] / left_scale) * (
                        sequences[sequence, index, component] / right_scale
                    )
                    dot += term
                    magnitude += abs(term)
            if abs(dot) <= 8.881784197001252e-16 * magnitude:
                ambiguous[sequence] = True
            flip = dot < 0.0
            for component in range(4):
                value = sequences[sequence, index, component]
                result[sequence, index, component] = -value if flip else value
    return result, ambiguous


def _rotation_distance_row_numba(
    q0: np.ndarray,
    q1: np.ndarray,
    epsilon: int,
    scalar_last: bool,
) -> tuple[float, bool]:
    """Return one shortest angle from scaled conjugate-product coefficients."""
    scale0 = max(abs(q0[0]), abs(q0[1]), abs(q0[2]), abs(q0[3]))
    scale1 = max(abs(q1[0]), abs(q1[1]), abs(q1[2]), abs(q1[3]))
    if scale0 == 0.0 or scale1 == 0.0:
        return 0.0, False
    if scalar_last:
        x0, y0, z0, w0 = q0[0] / scale0, q0[1] / scale0, q0[2] / scale0, q0[3] / scale0
        x1, y1, z1, w1 = q1[0] / scale1, q1[1] / scale1, q1[2] / scale1, q1[3] / scale1
    else:
        w0, x0, y0, z0 = q0[0] / scale0, q0[1] / scale0, q0[2] / scale0, q0[3] / scale0
        w1, x1, y1, z1 = q1[0] / scale1, q1[1] / scale1, q1[2] / scale1, q1[3] / scale1
        w1, x1, y1, z1 = q1[0] / scale1, q1[1] / scale1, q1[2] / scale1, q1[3] / scale1
    scalar = abs(((w0 * w1 + x0 * x1) + y0 * y1) + z0 * z1)
    vector_x = w0 * x1 - w1 * x0 - epsilon * (y0 * z1 - z0 * y1)
    vector_y = w0 * y1 - w1 * y0 - epsilon * (z0 * x1 - x0 * z1)
    vector_z = w0 * z1 - w1 * z0 - epsilon * (x0 * y1 - y0 * x1)
    vector_norm = hypot(hypot(vector_x, vector_y), vector_z)
    return 2.0 * atan2(vector_norm, scalar), True


def _rotation_distance_numba_impl(
    lhs: np.ndarray,
    rhs: np.ndarray,
    epsilon: int,
    scalar_last: bool,
) -> tuple[np.ndarray, bool]:
    """Return all distances from a batch of quaternion row pairs."""
    result = np.empty(lhs.shape[0], dtype=np.float64)
    for index in range(lhs.shape[0]):
        result[index], valid = _rotation_distance_row_numba(
            lhs[index], rhs[index], epsilon, scalar_last
        )
        if not valid:
            return result, False
    return result, True


def _rotation_distance_numba_parallel_impl(
    lhs: np.ndarray,
    rhs: np.ndarray,
    epsilon: int,
    scalar_last: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute shortest angles in parallel for large quaternion batches."""
    result = np.empty(lhs.shape[0], dtype=np.float64)
    valid = np.ones(lhs.shape[0], dtype=np.bool_)
    for index in _NUMBA_PRANGE(lhs.shape[0]):
        scale0 = max(
            abs(lhs[index, 0]),
            abs(lhs[index, 1]),
            abs(lhs[index, 2]),
            abs(lhs[index, 3]),
        )
        scale1 = max(
            abs(rhs[index, 0]),
            abs(rhs[index, 1]),
            abs(rhs[index, 2]),
            abs(rhs[index, 3]),
        )
        if scale0 == 0.0 or scale1 == 0.0:
            result[index] = 0.0
            valid[index] = False
            continue
        if scalar_last:
            x0, y0, z0, w0 = (
                lhs[index, 0] / scale0,
                lhs[index, 1] / scale0,
                lhs[index, 2] / scale0,
                lhs[index, 3] / scale0,
            )
            x1, y1, z1, w1 = (
                rhs[index, 0] / scale1,
                rhs[index, 1] / scale1,
                rhs[index, 2] / scale1,
                rhs[index, 3] / scale1,
            )
        else:
            w0, x0, y0, z0 = (
                lhs[index, 0] / scale0,
                lhs[index, 1] / scale0,
                lhs[index, 2] / scale0,
                lhs[index, 3] / scale0,
            )
            w1, x1, y1, z1 = (
                rhs[index, 0] / scale1,
                rhs[index, 1] / scale1,
                rhs[index, 2] / scale1,
                rhs[index, 3] / scale1,
            )
        scalar = abs(((w0 * w1 + x0 * x1) + y0 * y1) + z0 * z1)
        vector_x = w0 * x1 - w1 * x0 - epsilon * (y0 * z1 - z0 * y1)
        vector_y = w0 * y1 - w1 * y0 - epsilon * (z0 * x1 - x0 * z1)
        vector_z = w0 * z1 - w1 * z0 - epsilon * (x0 * y1 - y0 * x1)
        vector_norm = hypot(hypot(vector_x, vector_y), vector_z)
        result[index] = 2.0 * atan2(vector_norm, scalar)
    return result, valid


def _rotations_equivalent_numba_impl(
    lhs: np.ndarray,
    rhs: np.ndarray,
    tolerance: float,
    epsilon: int,
    scalar_last: bool,
) -> tuple[np.ndarray, bool]:
    """Compare a batch of shortest distances against a tolerance."""
    result = np.empty(lhs.shape[0], dtype=np.bool_)
    for index in range(lhs.shape[0]):
        distance, valid = _rotation_distance_row_numba(
            lhs[index], rhs[index], epsilon, scalar_last
        )
        if not valid:
            return result, False
        result[index] = distance <= tolerance
    return result, True


def _to_rotvec_row_numba(
    value: np.ndarray, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, bool]:
    """Convert a nonzero quaternion row to a canonical physical rotation vector."""
    canonical = _canonical_quaternion_row_numba(value, epsilon, scalar_last)
    if scalar_last:
        x, y, z, w = canonical[0], canonical[1], canonical[2], canonical[3]
    else:
        w, x, y, z = canonical[0], canonical[1], canonical[2], canonical[3]
    x, y, z = epsilon * x, epsilon * y, epsilon * z
    scale = max(abs(x), abs(y), abs(z))
    result = np.zeros(3, dtype=np.float64)
    if scale == 0.0:
        return result, w != 0.0
    sx, sy, sz = x / scale, y / scale, z / scale
    vector_norm = sqrt(sx * sx + sy * sy + sz * sz)
    if w > 0.0 and scale < w * 1e-8:
        wm, we = frexp(w)
        for index, component in enumerate((x, y, z)):
            if component != 0.0:
                mantissa, exponent = frexp(component)
                result[index] = ldexp(2.0 * (mantissa / wm), exponent - we)
    else:
        base = max(w, scale)
        angle = 2.0 * atan2((scale / base) * vector_norm, w / base)
        result[0] = (sx / vector_norm) * angle
        result[1] = (sy / vector_norm) * angle
        result[2] = (sz / vector_norm) * angle
    for component in range(3):
        if result[component] == 0.0:
            result[component] = 0.0
    return result, True


def _from_rotvec_row_numba(
    value: np.ndarray, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, bool]:
    """Convert one physical rotation vector to a native quaternion row."""
    result = np.zeros(4, dtype=np.float64)
    scale = max(abs(value[0]), abs(value[1]), abs(value[2]))
    if scale == 0.0:
        result[3 if scalar_last else 0] = 1.0
        return result, True
    sx, sy, sz = value[0] / scale, value[1] / scale, value[2] / scale
    scaled_norm = sqrt(sx * sx + sy * sy + sz * sz)
    theta = scale * scaled_norm
    if not isfinite(theta):
        return result, False
    half = theta * 0.5
    vector_scale = sin(half) / scaled_norm * epsilon
    scalar = cos(half)
    if scalar_last:
        result[0] = sx * vector_scale
        result[1] = sy * vector_scale
        result[2] = sz * vector_scale
        result[3] = scalar
    else:
        result[0] = scalar
        result[1] = sx * vector_scale
        result[2] = sy * vector_scale
        result[3] = sz * vector_scale
    return _canonical_quaternion_row_numba(result, epsilon, scalar_last), True


def _rotation_mean_one_numba(
    samples: np.ndarray,
    weights: np.ndarray,
    epsilon: int,
    scalar_last: bool,
    intrinsic: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int]:
    """Compute one Markley or intrinsic weighted rotation mean."""
    count = samples.shape[0]
    normalized = np.empty_like(samples)
    scatter = np.zeros((4, 4), dtype=np.float64)
    for sample_index in range(count):
        value, valid = _normalize_quaternion_row_numba(samples[sample_index])
        if not valid:
            return np.zeros(4, dtype=np.float64), 1
        normalized[sample_index] = value
        for row in range(4):
            for column in range(4):
                scatter[row, column] += (
                    weights[sample_index] * value[row] * value[column]
                )
    eigenvector, _, _ = _jacobi_largest_numba(scatter)
    mean, valid = _normalize_quaternion_row_numba(eigenvector)
    if not valid:
        return np.zeros(4, dtype=np.float64), 1
    mean = _canonical_quaternion_row_numba(mean, epsilon, scalar_last)
    if not intrinsic:
        return mean, 0
    for _ in range(max_iterations):
        inverse, valid = _inverse_row_numba(mean, scalar_last)
        if not valid:
            return mean, 1
        delta = np.zeros(3, dtype=np.float64)
        for sample_index in range(count):
            relative = _quaternion_product_row_numba(
                inverse, normalized[sample_index], epsilon, scalar_last
            )
            error, valid = _to_rotvec_row_numba(relative, epsilon, scalar_last)
            if not valid:
                return mean, 1
            for component in range(3):
                delta[component] += weights[sample_index] * error[component]
        magnitude = hypot(hypot(delta[0], delta[1]), delta[2])
        if magnitude <= tolerance:
            return mean, 0
        increment, valid = _from_rotvec_row_numba(delta, epsilon, scalar_last)
        if not valid:
            return mean, 3
        mean = _quaternion_product_row_numba(mean, increment, epsilon, scalar_last)
        mean, valid = _normalize_quaternion_row_numba(mean)
        if not valid:
            return mean, 1
        mean = _canonical_quaternion_row_numba(mean, epsilon, scalar_last)
    return mean, 2


def _rotation_mean_numba_impl(
    samples: np.ndarray,
    weights: np.ndarray,
    epsilon: int,
    scalar_last: bool,
    intrinsic: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run weighted rotation means for a batch of sample groups."""
    result = np.empty((samples.shape[0], 4), dtype=np.float64)
    statuses = np.zeros(samples.shape[0], dtype=np.int64)
    for group in range(samples.shape[0]):
        result[group], statuses[group] = _rotation_mean_one_numba(
            samples[group],
            weights[group],
            epsilon,
            scalar_last,
            intrinsic,
            tolerance,
            max_iterations,
        )
    return result, statuses


def _rotation_covariance_numba_impl(
    samples: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    has_mean: bool,
    world_frame: bool,
    epsilon: int,
    scalar_last: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute weighted tangent covariance matrices for batched samples."""
    result = np.zeros((samples.shape[0], 3, 3), dtype=np.float64)
    statuses = np.zeros(samples.shape[0], dtype=np.int64)
    for group in range(samples.shape[0]):
        normalized = np.empty_like(samples[group])
        for sample_index in range(samples.shape[1]):
            normalized[sample_index], valid = _normalize_quaternion_row_numba(
                samples[group, sample_index]
            )
            if not valid:
                statuses[group] = 1
                break
        if statuses[group] != 0:
            continue
        if has_mean:
            center, valid = _normalize_quaternion_row_numba(means[group])
            if not valid:
                statuses[group] = 1
                continue
        else:
            center, status = _rotation_mean_one_numba(
                normalized, weights[group], epsilon, scalar_last, False, 1e-12, 100
            )
            if status != 0:
                statuses[group] = status
                continue
        inverse, valid = _inverse_row_numba(center, scalar_last)
        if not valid:
            statuses[group] = 1
            continue
        for sample_index in range(samples.shape[1]):
            if world_frame:
                relative = _quaternion_product_row_numba(
                    normalized[sample_index], inverse, epsilon, scalar_last
                )
            else:
                relative = _quaternion_product_row_numba(
                    inverse, normalized[sample_index], epsilon, scalar_last
                )
            error, valid = _to_rotvec_row_numba(relative, epsilon, scalar_last)
            if not valid:
                statuses[group] = 1
                break
            for row in range(3):
                for column in range(3):
                    result[group, row, column] += (
                        weights[group, sample_index] * error[row] * error[column]
                    )
    return result, statuses


def _engine_axis_map_numba_impl(
    values: np.ndarray,
    indices: np.ndarray,
    signs: np.ndarray,
    scalar_first: bool,
) -> np.ndarray:
    """Reorder and sign-flip batched quaternion vectors into engine XYZW."""
    result = np.empty_like(values)
    offset = 1 if scalar_first else 0
    scalar_index = 0 if scalar_first else 3
    for sample in range(values.shape[0]):
        for destination in range(3):
            source = values[sample, offset + indices[destination]]
            result[sample, destination] = source if signs[destination] > 0 else -source
        result[sample, 3] = values[sample, scalar_index]
    return result


def _canonicalize_numba_impl(
    quaternions: np.ndarray, epsilon: int, scalar_last: bool
) -> np.ndarray:
    """Choose canonical signs and clear signed zeros in one batch pass."""
    result = np.empty_like(quaternions)
    scalar_index = 3 if scalar_last else 0
    vector_start = 0 if scalar_last else 1
    for row in range(quaternions.shape[0]):
        scalar = quaternions[row, scalar_index]
        first = 0.0
        for offset in range(3):
            value = quaternions[row, vector_start + offset]
            if value != 0.0:
                first = value
                break
        flip = scalar < 0.0 or (
            scalar == 0.0 and ((first < 0.0) if epsilon > 0 else (first > 0.0))
        )
        for component in range(4):
            value = (
                -quaternions[row, component] if flip else quaternions[row, component]
            )
            result[row, component] = 0.0 if value == 0.0 else value
    return result


def _commutator_numba_impl(
    lhs: np.ndarray, rhs: np.ndarray, epsilon: int, scalar_last: bool
) -> np.ndarray:
    """Compute a quaternion commutator from its vector cross product."""
    result = np.zeros_like(lhs)
    vector_start = 0 if scalar_last else 1
    factor = 2.0 * epsilon
    for row in range(lhs.shape[0]):
        ax, ay, az = (
            lhs[row, vector_start],
            lhs[row, vector_start + 1],
            lhs[row, vector_start + 2],
        )
        bx, by, bz = (
            rhs[row, vector_start],
            rhs[row, vector_start + 1],
            rhs[row, vector_start + 2],
        )
        result[row, vector_start] = factor * (ay * bz - az * by)
        result[row, vector_start + 1] = factor * (az * bx - ax * bz)
        result[row, vector_start + 2] = factor * (ax * by - ay * bx)
    return result


def _conjugate_numba_impl(quaternions: np.ndarray) -> np.ndarray:
    """Conjugate scalar-first quaternion rows in one contiguous pass."""
    result = np.empty_like(quaternions)
    for row in range(quaternions.shape[0]):
        result[row, 0] = quaternions[row, 0]
        result[row, 1] = -quaternions[row, 1]
        result[row, 2] = -quaternions[row, 2]
        result[row, 3] = -quaternions[row, 3]
    return result


def _component_transform_numba_impl(
    values: np.ndarray,
    source_scalar_last: bool,
    target_scalar_last: bool,
    negate_vector: bool,
) -> np.ndarray:
    """Reorder quaternion rows and optionally convert Hamilton/JPL signs."""
    result = np.empty_like(values)
    for row in range(values.shape[0]):
        if source_scalar_last:
            scalar = values[row, 3]
            x, y, z = values[row, 0], values[row, 1], values[row, 2]
        else:
            scalar = values[row, 0]
            x, y, z = values[row, 1], values[row, 2], values[row, 3]
        if negate_vector:
            x, y, z = -x, -y, -z
        if target_scalar_last:
            result[row, 0], result[row, 1], result[row, 2], result[row, 3] = (
                x,
                y,
                z,
                scalar,
            )
        else:
            result[row, 0], result[row, 1], result[row, 2], result[row, 3] = (
                scalar,
                x,
                y,
                z,
            )
    return result


def _vector_part_numba_impl(quaternions: np.ndarray, scalar_last: bool) -> np.ndarray:
    """Copy vector components from each quaternion row."""
    result = np.empty((quaternions.shape[0], 3), dtype=np.float64)
    vector_start = 0 if scalar_last else 1
    for row in range(quaternions.shape[0]):
        for component in range(3):
            result[row, component] = quaternions[row, vector_start + component]
    return result


def _random_quaternion_numba_impl(
    samples: np.ndarray, epsilon: int, scalar_last: bool
) -> np.ndarray:
    """Map Shoemake uniform samples to quaternion rows."""
    result = np.empty((samples.shape[0], 4), dtype=np.float64)
    two_pi = 2.0 * pi
    for row in range(samples.shape[0]):
        u1, u2, u3 = samples[row, 0], samples[row, 1], samples[row, 2]
        root_complement = sqrt(1.0 - u1)
        root = sqrt(u1)
        x = root_complement * sin(two_pi * u2)
        y = root_complement * cos(two_pi * u2)
        z = root * sin(two_pi * u3)
        scalar = root * cos(two_pi * u3)
        if epsilon < 0:
            x, y, z = -x, -y, -z
        if scalar_last:
            result[row, 0], result[row, 1], result[row, 2], result[row, 3] = (
                x,
                y,
                z,
                scalar,
            )
        else:
            result[row, 0], result[row, 1], result[row, 2], result[row, 3] = (
                scalar,
                x,
                y,
                z,
            )
    return result


_NUMBA_ROTATION_KERNEL_NAMES = frozenset(
    {
        "algebraic",
        "from_axis_angle",
        "from_rotation_matrix",
        "to_rotation_matrix",
        "quaternion_matrix",
        "canonicalize",
        "commutator",
        "from_two_vectors",
        "conjugate",
        "component_transform",
        "vector_part",
        "random_quaternion",
        "from_euler",
        "to_euler",
        "from_rotvec",
        "to_rotvec",
        "integrate_angular_velocity",
        "nlerp",
        "slerp",
        "from_mrp",
        "to_mrp",
        "left_divide",
        "right_divide",
        "rotation_distance",
        "rotation_distance_parallel",
        "rotations_equivalent",
        "engine_axis_map",
        "normalize",
        "inverse",
        "align_vectors",
        "power_integer",
        "power_quaternion",
        "power_real",
        "norm",
        "make_sign_continuous",
        "rotation_covariance",
        "rotation_mean",
        "dot",
        "skew_matrix",
        "so3_jacobian",
        "to_axis_angle",
    }
)

_NUMBA_ROTATION_CHECKED = False

_NUMBA_ROTATION_KERNELS: dict[str, Any] = {}


def _numba_rotation_kernel(operation: str) -> Any:
    """Lazily load optional rotation kernels and isolate kernel failures."""
    global _NUMBA_PRANGE, _NUMBA_ROTATION_CHECKED
    if not _NUMBA_ROTATION_CHECKED:
        _NUMBA_ROTATION_CHECKED = True
        try:
            import numba as numba_module
            from numba import guvectorize, njit

            _NUMBA_PRANGE = getattr(numba_module, "prange", range)

            for helper_name in (
                "_quaternion_product_row_numba",
                "_inverse_row_numba",
                "_rotation_distance_row_numba",
                "_log_row_numba",
                "_exp_row_numba",
                "_jacobi_largest_numba",
                "_normalize_quaternion_row_numba",
                "_canonical_quaternion_row_numba",
                "_to_rotvec_row_numba",
                "_from_rotvec_row_numba",
                "_rotation_mean_one_numba",
            ):
                helper = globals()[helper_name]
                if not hasattr(helper, "nopython_signatures"):
                    if helper_name == "_divide_row_numba":
                        globals()[helper_name] = njit(
                            cache=True,
                            nogil=True,
                            inline="always",
                            fastmath={"reassoc"},
                        )(helper)
                    elif helper_name == "_rotation_distance_row_numba":
                        globals()[helper_name] = njit(
                            cache=True, nogil=True, inline="always"
                        )(helper)
                    else:
                        globals()[helper_name] = njit(cache=True, nogil=True)(helper)

            _NUMBA_ROTATION_KERNELS.update(
                {
                    "from_rotation_matrix": njit(cache=True, nogil=True)(
                        _matrix_to_quaternion_numba_impl
                    ),
                    "to_rotation_matrix": njit(cache=True, nogil=True)(
                        _rotation_matrix_numba_impl
                    ),
                    "quaternion_matrix": njit(cache=True, nogil=True)(
                        _quaternion_matrix_numba_impl
                    ),
                    "canonicalize": njit(cache=True, nogil=True)(
                        _canonicalize_numba_impl
                    ),
                    "commutator": njit(cache=True, nogil=True)(_commutator_numba_impl),
                    "from_two_vectors": njit(cache=True, nogil=True)(
                        _from_two_vectors_numba_impl
                    ),
                    "conjugate": njit(cache=True, nogil=True)(_conjugate_numba_impl),
                    "component_transform": njit(cache=True, nogil=True)(
                        _component_transform_numba_impl
                    ),
                    "vector_part": njit(cache=True, nogil=True)(
                        _vector_part_numba_impl
                    ),
                    "random_quaternion": njit(cache=True, nogil=True)(
                        _random_quaternion_numba_impl
                    ),
                    "from_euler": njit(cache=True, nogil=True, parallel=True)(
                        _from_euler_numba_impl
                    ),
                    "to_euler": njit(cache=True, nogil=True, parallel=True)(
                        _to_euler_numba_impl
                    ),
                    "from_rotvec": njit(cache=True, nogil=True)(
                        _from_rotvec_numba_impl
                    ),
                    "to_rotvec": njit(cache=True, nogil=True)(_to_rotvec_numba_impl),
                    "integrate_angular_velocity": njit(cache=True, nogil=True)(
                        _integrate_angular_velocity_numba_impl
                    ),
                    "nlerp": njit(cache=True, nogil=True)(_nlerp_numba_impl),
                    "slerp": njit(cache=True, nogil=True)(_slerp_numba_impl),
                    "from_mrp": njit(cache=True, nogil=True)(_from_mrp_numba_impl),
                    "to_mrp": njit(cache=True, nogil=True)(_to_mrp_numba_impl),
                    "normalize": njit(cache=True, nogil=True, fastmath=True)(
                        _normalize_numba_impl
                    ),
                    "inverse": njit(cache=True, nogil=True, fastmath=True)(
                        _inverse_numba_impl
                    ),
                    "left_divide": njit(cache=True, nogil=True, fastmath=True)(
                        _divide_numba_impl
                    ),
                    "right_divide": njit(cache=True, nogil=True, fastmath=True)(
                        _divide_numba_impl
                    ),
                    "rotation_distance": njit(cache=True, nogil=True)(
                        _rotation_distance_numba_impl
                    ),
                    "rotation_distance_parallel": njit(
                        cache=True, nogil=True, fastmath=True, parallel=True
                    )(_rotation_distance_numba_parallel_impl),
                    "rotations_equivalent": njit(cache=True, nogil=True)(
                        _rotations_equivalent_numba_impl
                    ),
                    "engine_axis_map": njit(cache=True, nogil=True)(
                        _engine_axis_map_numba_impl
                    ),
                    "norm": njit(cache=True, nogil=True, fastmath=True)(
                        _norm_numba_impl
                    ),
                    "dot": guvectorize(
                        ["void(float64[:], float64[:], float64[:], boolean[:])"],
                        "(n),(n)->(),()",
                        target="cpu",
                        nopython=True,
                        cache=True,
                    )(_dot_numba_impl),
                    "algebraic": njit(cache=True, nogil=True)(_analytic_numba_impl),
                    "from_axis_angle": njit(cache=True, nogil=True)(
                        _from_axis_angle_numba_impl
                    ),
                    "to_axis_angle": njit(cache=True, nogil=True)(
                        _to_axis_angle_numba_impl
                    ),
                    "so3_jacobian": njit(cache=True, nogil=True)(
                        _so3_jacobian_numba_impl
                    ),
                    "skew_matrix": njit(cache=True, nogil=True)(_skew_numba_impl),
                    "power_integer": njit(cache=True, nogil=True)(
                        _power_integer_numba_impl
                    ),
                    "power_real": njit(cache=True, nogil=True)(_power_real_numba_impl),
                    "power_quaternion": njit(cache=True, nogil=True)(
                        _power_quaternion_numba_impl
                    ),
                    "make_sign_continuous": njit(cache=True, nogil=True)(
                        _make_sign_continuous_numba_impl
                    ),
                    "rotation_mean": njit(cache=True, nogil=True)(
                        _rotation_mean_numba_impl
                    ),
                    "rotation_covariance": njit(cache=True, nogil=True)(
                        _rotation_covariance_numba_impl
                    ),
                    "align_vectors": njit(cache=True, nogil=True)(
                        _align_vectors_numba_impl
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001
            _NUMBA_ROTATION_KERNELS.clear()
            _warn_numba_failure("rotation conversions", exc)
    return _NUMBA_ROTATION_KERNELS.get(operation)
