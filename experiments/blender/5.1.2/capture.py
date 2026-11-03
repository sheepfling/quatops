"""Capture Blender mathutils rotations from the pinned native runtime."""

import argparse
import json
import math
import tempfile
import random
import sys
from pathlib import Path

import bpy
from mathutils import Euler, Quaternion, Vector


ORDERS = ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX")
IDENTITY_AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
TOLERANCE = 2.0e-5


def as_float_list(values):
    return [float(value) for value in values]


def xyz(values):
    return {axis: float(values[index]) for index, axis in enumerate("xyz")}


def quaternion_record(value):
    wxyz = as_float_list(value)
    return {
        "wxyz": wxyz,
        "xyzw": [wxyz[1], wxyz[2], wxyz[3], wxyz[0]],
    }


def matrix_rows(value):
    return [as_float_list(row) for row in value]


def vector_images(rotation):
    return [as_float_list(rotation @ Vector(axis)) for axis in IDENTITY_AXES]


def matrix_images(rotation):
    return [as_float_list(rotation @ Vector(axis)) for axis in IDENTITY_AXES]


def quaternion_from_axis_angle(axis, angle):
    return Quaternion(Vector(axis), float(angle))


def quaternion_from_rotvec(values):
    return Quaternion(tuple(float(value) for value in values))


def same_rotation(lhs, rhs, tolerance=TOLERANCE):
    lhs_matrix = lhs.to_matrix()
    rhs_matrix = rhs.to_matrix()
    return max(
        abs(float(lhs_matrix[row][column] - rhs_matrix[row][column]))
        for row in range(3)
        for column in range(3)
    ) <= tolerance


def base_euler_cases():
    return [
        ("identity", (0.0, 0.0, 0.0)),
        ("x_positive_90", (90.0, 0.0, 0.0)),
        ("y_positive_90", (0.0, 90.0, 0.0)),
        ("z_positive_90", (0.0, 0.0, 90.0)),
        ("mixed_positive", (31.41, -49.16, 12.34)),
        ("mixed_negative", (-121.5, 67.25, -203.75)),
        ("outside_canonical_range", (405.0, -275.0, 725.0)),
        ("signed_180", (-180.0, 180.0, -180.0)),
        ("x_gimbal_center", (90.0, 23.0, -41.0)),
        ("x_gimbal_below", (89.999, 23.0, -41.0)),
        ("x_gimbal_above", (90.001, 23.0, -41.0)),
        ("y_gimbal_center", (31.0, 90.0, -47.0)),
        ("y_gimbal_below", (31.0, 89.999, -47.0)),
        ("y_gimbal_above", (31.0, 90.001, -47.0)),
        ("z_gimbal_center", (-29.0, 58.0, -90.0)),
    ]


def euler_order_inputs(order):
    rows = list(base_euler_cases())
    seed = 0xB1E000 + ORDERS.index(order)
    rng = random.Random(seed)
    for index in range(64):
        rows.append(
            (
                "seeded_%02d" % index,
                tuple(rng.uniform(-720.0, 720.0) for _ in range(3)),
            )
        )

    middle_axis = order[1].lower()
    for sign_name, sign in (("positive", 1.0), ("negative", -1.0)):
        for neighbor_name, delta in (("below", -1.0e-4), ("above", 1.0e-4)):
            angles = {"x": 17.0, "y": -29.0, "z": 43.0}
            angles[middle_axis] = sign * 90.0 + delta
            case_id = "%s_middle_gimbal_%s_%s" % (
                sign_name,
                neighbor_name,
                middle_axis,
            )
            rows.append((case_id, tuple(angles[axis] for axis in "xyz")))
        for neighbor_name, direction in (("below", -math.inf), ("above", math.inf)):
            angles = {"x": 17.0, "y": -29.0, "z": 43.0}
            angles[middle_axis] = nextafter_float32(sign * 90.0, direction)
            case_id = "%s_middle_gimbal_%s_float32_ulp_%s" % (
                sign_name,
                neighbor_name,
                middle_axis,
            )
            rows.append((case_id, tuple(angles[axis] for axis in "xyz")))
    return rows


def nextafter_float32(value, toward):
    import struct

    packed = struct.pack("f", float(value))
    bits = struct.unpack("I", packed)[0]
    if (toward > value) == (value >= 0.0):
        bits += 1
    else:
        bits -= 1
    return float(struct.unpack("f", struct.pack("I", bits))[0])


def capture_euler_row(order, case_id, angles_degrees):
    angles_radians = tuple(math.radians(value) for value in angles_degrees)
    euler = Euler(angles_radians, order)
    quaternion = euler.to_quaternion()
    matrix = euler.to_matrix()
    quaternion_readback = quaternion.to_euler(order)
    matrix_readback = matrix.to_euler(order)
    quaternion_readback_roundtrip = Euler(tuple(quaternion_readback), order).to_quaternion()
    matrix_readback_roundtrip = Euler(tuple(matrix_readback), order).to_quaternion()
    matrix_quaternion = matrix.to_quaternion()
    quaternion_matrix = quaternion.to_matrix()

    return {
        "case_id": case_id,
        "order": order,
        "input_degrees_xyz": xyz(angles_degrees),
        "input_radians_xyz": xyz(angles_radians),
        "euler_quaternion": quaternion_record(quaternion),
        "euler_matrix_rows": matrix_rows(matrix),
        "quaternion_matrix_rows": matrix_rows(quaternion_matrix),
        "matrix_quaternion": quaternion_record(matrix_quaternion),
        "quaternion_rotated_positive_basis_vectors_xyz": vector_images(quaternion),
        "matrix_rotated_positive_basis_vectors_xyz": matrix_images(matrix),
        "quaternion_to_euler_radians_xyz": xyz(quaternion_readback),
        "matrix_to_euler_radians_xyz": xyz(matrix_readback),
        "quaternion_readback_roundtrip": quaternion_record(quaternion_readback_roundtrip),
        "matrix_readback_roundtrip": quaternion_record(matrix_readback_roundtrip),
    }


def capture_euler_cases():
    cases = []
    for order in ORDERS:
        for case_id, angles in euler_order_inputs(order):
            cases.append(capture_euler_row(order, case_id, angles))
    return cases


def capture_base_cases():
    rows = []
    for case_id, angles_degrees in base_euler_cases():
        euler = Euler(tuple(math.radians(value) for value in angles_degrees), "XYZ")
        quaternion = euler.to_quaternion()
        matrix = euler.to_matrix()
        readback = quaternion.to_euler("XYZ")
        readback_roundtrip = Euler(tuple(readback), "XYZ").to_quaternion()
        rows.append(
            {
                "case_id": case_id,
                "order": "XYZ",
                "input_degrees_xyz": xyz(angles_degrees),
                "input_radians_xyz": xyz(tuple(math.radians(value) for value in angles_degrees)),
                "euler_quaternion": quaternion_record(quaternion),
                "euler_matrix_rows": matrix_rows(matrix),
                "matrix_quaternion": quaternion_record(matrix.to_quaternion()),
                "quaternion_to_euler_radians_xyz": xyz(readback),
                "quaternion_readback_roundtrip": quaternion_record(readback_roundtrip),
                "quaternion_rotated_positive_basis_vectors_xyz": vector_images(quaternion),
            }
        )
    return rows


def capture_matrix_to_quaternion():
    cases = []
    inputs = (
        ("identity", (1.0, 0.0, 0.0), 0.0),
        ("near_180_x", (1.0, 0.0, 0.0), math.pi - 1.0e-5),
        ("near_180_y", (0.0, 1.0, 0.0), math.pi - 1.0e-5),
        ("near_180_z", (0.0, 0.0, 1.0), math.pi - 1.0e-5),
        ("negative_trace_mixed", (1.0, -2.0, 3.0), math.pi - 1.0e-4),
        ("half_turn_mixed", (1.0, 2.0, -1.0), math.pi),
        ("near_zero", (1.0, 2.0, -1.0), 1.0e-6),
        ("mixed", (-2.0, 1.0, 3.0), 1.7),
    )
    for case_id, axis, angle in inputs:
        quaternion = quaternion_from_axis_angle(axis, angle)
        matrix = quaternion.to_matrix()
        reconstructed = matrix.to_quaternion()
        cases.append(
            {
                "case_id": case_id,
                "input_axis_xyz": as_float_list(Vector(axis).normalized()),
                "input_angle_radians": float(angle),
                "matrix_rows": matrix_rows(matrix),
                "matrix_quaternion": quaternion_record(reconstructed),
                "expected_quaternion": quaternion_record(quaternion),
            }
        )
    return cases


def capture_axis_angle():
    rows = []
    inputs = (
        ("identity", (1.0, 0.0, 0.0), 0.0),
        ("tiny", (0.0, 1.0, 0.0), 1.0e-7),
        ("positive_90", (1.0, 2.0, -3.0), math.pi / 2.0),
        ("near_half_turn", (-2.0, 1.0, 4.0), math.pi - 1.0e-5),
        ("half_turn", (0.0, -3.0, 2.0), math.pi),
        ("beyond_half_turn", (0.0, 0.0, 1.0), math.radians(220.0)),
    )
    for case_id, axis, angle in inputs:
        quaternion = quaternion_from_axis_angle(axis, angle)
        negative = Quaternion(tuple(-value for value in quaternion))
        for sign_name, value in (("q", quaternion), ("negative_q", negative)):
            read_axis, read_angle = value.to_axis_angle()
            rows.append(
                {
                    "case_id": case_id,
                    "input_sign": sign_name,
                    "input_axis_xyz": as_float_list(Vector(axis).normalized()),
                    "input_angle_radians": float(angle),
                    "quaternion": quaternion_record(value),
                    "readback_axis_xyz": as_float_list(read_axis),
                    "readback_angle_radians": float(read_angle),
                }
            )
    return rows


def capture_rotation_vectors():
    rows = []
    vectors = (
        ("zero", (0.0, 0.0, 0.0)),
        ("tiny", (1.0e-8, -2.0e-8, 3.0e-8)),
        ("quarter_turn_x", (math.pi / 2.0, 0.0, 0.0)),
        ("near_half_turn_y", (0.0, math.pi - 1.0e-5, 0.0)),
        ("half_turn_mixed", tuple(value * math.pi / math.sqrt(14.0) for value in (1.0, 2.0, -3.0))),
        ("beyond_half_turn", (0.0, 0.0, math.radians(220.0))),
    )
    for case_id, vector in vectors:
        quaternion = quaternion_from_rotvec(vector)
        readback = quaternion.to_exponential_map()
        rows.append(
            {
                "case_id": case_id,
                "input_rotation_vector_xyz": as_float_list(vector),
                "constructed_quaternion": quaternion_record(quaternion),
                "readback_rotation_vector_xyz": as_float_list(readback),
                "readback_quaternion": quaternion_record(quaternion_from_rotvec(readback)),
            }
        )
    return rows


def capture_slerp():
    axis_x = (1.0, 0.0, 0.0)
    inputs = (
        (
            "ordinary",
            quaternion_from_axis_angle(axis_x, 0.63),
            quaternion_from_axis_angle((0.0, 1.0, 1.0), 1.42),
        ),
        (
            "near_identical",
            quaternion_from_axis_angle((0.0, 0.0, 1.0), 0.5),
            quaternion_from_axis_angle((0.0, 0.0, 1.0), 0.50001),
        ),
        (
            "antipodal_sign_pair",
            quaternion_from_axis_angle((1.0, -2.0, 3.0), 1.3),
            Quaternion(tuple(-value for value in quaternion_from_axis_angle((1.0, -2.0, 3.0), 1.3))),
        ),
        (
            "near_half_turn",
            Quaternion(),
            quaternion_from_axis_angle((0.0, 1.0, 0.0), math.pi - 1.0e-5),
        ),
        (
            "greater_than_180_endpoint",
            Quaternion(),
            quaternion_from_axis_angle((0.0, 0.0, 1.0), math.radians(220.0)),
        ),
    )
    rows = []
    for case_id, start, end in inputs:
        for factor in (0.0, 0.25, 0.5, 0.75, 1.0):
            value = start.slerp(end, factor)
            rows.append(
                {
                    "case_id": case_id,
                    "start_quaternion": quaternion_record(start),
                    "end_quaternion": quaternion_record(end),
                    "factor": factor,
                    "interpolated_quaternion": quaternion_record(value),
                    "interpolated_matrix_rows": matrix_rows(value.to_matrix()),
                }
            )
    return rows


def capture_normalization():
    rows = []
    values = (
        ("zero", (0.0, 0.0, 0.0, 0.0)),
        ("tiny", (1.0e-20, -2.0e-20, 3.0e-20, 4.0e-20)),
        ("nonunit", (2.0, -3.0, 4.0, 5.0)),
        ("unit", (0.5, -0.5, 0.5, 0.5)),
        ("sign_reversed", (-0.5, 0.5, -0.5, -0.5)),
    )
    for case_id, value in values:
        quaternion = Quaternion(value)
        try:
            normalized = quaternion.normalized()
            result = {
                "normalized_quaternion": quaternion_record(normalized),
                "normalized_magnitude": float(normalized.magnitude),
                "error": None,
            }
        except Exception as error:  # Capture native API failure as fixture data.
            result = {
                "normalized_quaternion": None,
                "normalized_magnitude": None,
                "error": "%s: %s" % (type(error).__name__, error),
            }
        rows.append(
            {
                "case_id": case_id,
                "input_wxyz": as_float_list(value),
                "input_magnitude": float(quaternion.magnitude),
                **result,
            }
        )
    return rows


def capture_rotation_difference():
    cases = (
        (
            "noncommuting",
            quaternion_from_axis_angle((1.0, 0.0, 0.0), 0.7),
            quaternion_from_axis_angle((0.0, 1.0, 1.0), -1.1),
        ),
        (
            "same_orientation_sign_pair",
            quaternion_from_axis_angle((1.0, 2.0, -1.0), 1.7),
            Quaternion(tuple(-value for value in quaternion_from_axis_angle((1.0, 2.0, -1.0), 1.7))),
        ),
    )
    rows = []
    for case_id, start, end in cases:
        difference = start.rotation_difference(end)
        start_inverse_then_end = start.inverted() @ end
        end_then_start_inverse = end @ start.inverted()
        rows.append(
            {
                "case_id": case_id,
                "start_quaternion": quaternion_record(start),
                "end_quaternion": quaternion_record(end),
                "difference_quaternion": quaternion_record(difference),
                "start_inverse_then_end_quaternion": quaternion_record(start_inverse_then_end),
                "end_then_start_inverse_quaternion": quaternion_record(end_then_start_inverse),
                "matches_start_inverse_then_end": same_rotation(difference, start_inverse_then_end),
                "matches_end_then_start_inverse": same_rotation(difference, end_then_start_inverse),
            }
        )
    return rows


def capture_swing_twist():
    rows = []
    inputs = (
        ("mixed_rotation", quaternion_from_axis_angle((1.0, -2.0, 3.0), 2.2)),
        ("pure_twist_x", quaternion_from_axis_angle((1.0, 0.0, 0.0), 1.1)),
        ("orthogonal_half_turn", quaternion_from_axis_angle((0.0, 1.0, 0.0), math.pi)),
    )
    axis_vectors = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
    for case_id, quaternion in inputs:
        for axis_name, axis in axis_vectors.items():
            swing, twist_angle = quaternion.to_swing_twist(axis_name)
            twist = quaternion_from_axis_angle(axis, twist_angle)
            swing_then_twist = swing @ twist
            twist_then_swing = twist @ swing
            rows.append(
                {
                    "case_id": case_id,
                    "twist_axis": axis_name,
                    "quaternion": quaternion_record(quaternion),
                    "swing_quaternion": quaternion_record(swing),
                    "twist_angle_radians": float(twist_angle),
                    "reconstructed_twist_quaternion": quaternion_record(twist),
                    "swing_then_twist_quaternion": quaternion_record(swing_then_twist),
                    "twist_then_swing_quaternion": quaternion_record(twist_then_swing),
                    "swing_then_twist_matches": same_rotation(quaternion, swing_then_twist),
                    "twist_then_swing_matches": same_rotation(quaternion, twist_then_swing),
                }
            )
    return rows


def capture_compat_trajectory(order, trajectory_id, inputs):
    rows = []
    previous = None
    for frame, (case_id, angles_degrees) in enumerate(inputs):
        euler = Euler(tuple(math.radians(value) for value in angles_degrees), order)
        quaternion = euler.to_quaternion()
        ordinary = quaternion.to_euler(order)
        compatible = quaternion.to_euler(order, previous) if previous is not None else ordinary.copy()
        rows.append(
            {
                "frame": frame,
                "case_id": case_id,
                "input_degrees_xyz": xyz(angles_degrees),
                "quaternion": quaternion_record(quaternion),
                "ordinary_to_euler_radians_xyz": xyz(ordinary),
                "compatible_to_euler_radians_xyz": xyz(compatible),
            }
        )
        previous = compatible.copy()
    return {"order": order, "trajectory_id": trajectory_id, "frames": rows}


def capture_euler_compat_trajectories():
    trajectories = []
    crossing_positive = tuple(
        ("x_%d" % angle, (float(angle), 24.0, -37.0))
        for angle in (170, 175, 179, 181, 185, 190)
    )
    crossing_negative = tuple(
        ("x_%d" % angle, (float(angle), 24.0, -37.0))
        for angle in (-170, -175, -179, -181, -185, -190)
    )
    for order in ORDERS:
        trajectories.append(
            capture_compat_trajectory(order, "x_crosses_positive_180", crossing_positive)
        )
        trajectories.append(
            capture_compat_trajectory(order, "x_crosses_negative_180", crossing_negative)
        )

        middle_axis = order[1].lower()
        gimbal = []
        for angle in (88.0, 89.5, 89.9, 90.0, 90.1, 90.5, 92.0):
            values = {"x": 31.0, "y": -22.0, "z": 47.0}
            values[middle_axis] = angle
            gimbal.append(("%s_%s" % (middle_axis, str(angle).replace(".", "_")), tuple(values[a] for a in "xyz")))
        trajectories.append(capture_compat_trajectory(order, "middle_axis_gimbal", tuple(gimbal)))
    return trajectories


def capture_scene_object_routes():
    scene = bpy.context.scene
    obj = bpy.data.objects.new("QuatOpsRotationProof", None)
    scene.collection.objects.link(obj)
    angles = (math.radians(31.0), math.radians(-49.0), math.radians(12.0))
    euler = Euler(angles, "XYZ")
    euler_quaternion = euler.to_quaternion()

    obj.rotation_mode = "XYZ"
    obj.rotation_euler = euler
    bpy.context.view_layer.update()
    euler_mode = {
        "rotation_mode": obj.rotation_mode,
        "rotation_euler_radians_xyz": xyz(obj.rotation_euler),
        "matrix_world_3x3_rows": matrix_rows(obj.matrix_world.to_3x3()),
        "matrix_world_quaternion": quaternion_record(obj.matrix_world.to_quaternion()),
    }

    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = euler_quaternion
    bpy.context.view_layer.update()
    quaternion_mode = {
        "rotation_mode": obj.rotation_mode,
        "rotation_quaternion_wxyz": as_float_list(obj.rotation_quaternion),
        "matrix_world_3x3_rows": matrix_rows(obj.matrix_world.to_3x3()),
        "matrix_world_quaternion": quaternion_record(obj.matrix_world.to_quaternion()),
    }
    bpy.data.objects.remove(obj, do_unlink=True)
    return {
        "case_id": "scene_object_local_rotation",
        "input_euler_order": "XYZ",
        "input_euler_radians_xyz": xyz(angles),
        "euler_mode": euler_mode,
        "quaternion_mode": quaternion_mode,
    }


def capture_cross_products():
    axes = {"x": Vector((1.0, 0.0, 0.0)), "y": Vector((0.0, 1.0, 0.0)), "z": Vector((0.0, 0.0, 1.0))}
    return [
        {
            "case_id": "%s_cross_%s" % (lhs, rhs),
            "lhs_xyz": as_float_list(axes[lhs]),
            "rhs_xyz": as_float_list(axes[rhs]),
            "result_xyz": as_float_list(axes[lhs].cross(axes[rhs])),
            "reversed_result_xyz": as_float_list(axes[rhs].cross(axes[lhs])),
        }
        for lhs, rhs in (("x", "y"), ("y", "z"), ("z", "x"))
    ]


def capture_explicit_source_map():
    axes = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
    }
    probes = []
    for axis_name, axis in axes.items():
        source_quaternion = quaternion_from_axis_angle(axis, math.radians(67.0))
        blender_quaternion = quaternion_from_axis_angle(axis, math.radians(67.0))
        probes.append(
            {
                "source_rotation_axis": axis_name,
                "source_quaternion": quaternion_record(source_quaternion),
                "blender_quaternion": quaternion_record(blender_quaternion),
                "blender_rotated_positive_basis_vectors_xyz": vector_images(blender_quaternion),
            }
        )
    return {
        "source_frame": "fixture-declared right-handed +X right, +Y forward, +Z up",
        "blender_probe_frame": "the same explicitly assigned numeric axes for this conversion test only",
        "destination_axes_from_source": ["x", "y", "z"],
        "determinant": 1,
        "principal_rotation_probes": probes,
    }


def capture_canonical_interop():
    orientations = (
        ("identity", (1.0, 0.0, 0.0, 0.0)),
        ("quarter_x", (0.7071067811865476, 0.7071067811865476, 0.0, 0.0)),
        ("quarter_y", (0.7071067811865476, 0.0, 0.7071067811865476, 0.0)),
        ("quarter_z", (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)),
        ("mixed_a", (0.8, 0.2, -0.4, 0.4)),
        ("mixed_b", (0.7, -0.3, 0.2, 0.6164414002968976)),
        ("antipode_mixed_a", (-0.8, -0.2, 0.4, -0.4)),
        ("nonprincipal_halfturn", (0.0, 0.5773502691896258, 0.5773502691896258, 0.5773502691896258)),
    )
    canonical_probe = Vector((0.25, -1.5, 2.75))
    native_probe = Vector(canonical_probe)

    def orientation(case_id, canonical_wxyz, native):
        return {
            "case_id": case_id,
            "canonical_wxyz": list(canonical_wxyz),
            "native_quaternion": as_float_list(native),
            "native_rotated_positive_basis_vectors_xyz": vector_images(native),
            "native_rotated_probe_vector_xyz": as_float_list(native @ native_probe),
        }

    native = {case_id: Quaternion(wxyz) for case_id, wxyz in orientations}
    lhs_rhs = native["mixed_a"] @ native["mixed_b"]
    rhs_lhs = native["mixed_b"] @ native["mixed_a"]

    def composition(left_id, right_id, product):
        return {
            "left_case_id": left_id,
            "right_case_id": right_id,
            "product_order": "lhs * rhs; rhs acts first",
            "native_product_quaternion": as_float_list(product),
            "native_rotated_positive_basis_vectors_xyz": vector_images(product),
            "native_rotated_probe_vector_xyz": as_float_list(product @ native_probe),
        }

    return {
        "contract_id": "quatops-canonical-hamilton-wxyz-active-rh-v1",
        "native_quaternion_order": "wxyz",
        "frame_map_destination_axes_from_source": ["x", "y", "z"],
        "probe_vector_canonical_xyz": as_float_list(canonical_probe),
        "probe_vector_native_xyz": as_float_list(native_probe),
        "orientation_cases": [orientation(case_id, wxyz, native[case_id]) for case_id, wxyz in orientations],
        "composition": composition("mixed_a", "mixed_b", lhs_rhs),
        "reverse_composition": composition("mixed_b", "mixed_a", rhs_lhs),
    }


def capture_quaternion_composition():
    lhs = quaternion_from_axis_angle((1.0, 0.0, 0.0), 0.83)
    rhs = quaternion_from_axis_angle((0.0, 1.0, 0.0), -1.17)
    probe = Vector((0.31, -0.72, 1.23))
    product = lhs @ rhs
    product_image = product @ probe
    rhs_then_lhs = lhs @ (rhs @ probe)
    lhs_then_rhs = rhs @ (lhs @ probe)
    return {
        "lhs_quaternion": quaternion_record(lhs),
        "rhs_quaternion": quaternion_record(rhs),
        "product_quaternion": quaternion_record(product),
        "probe_vector_xyz": as_float_list(probe),
        "product_rotated_vector_xyz": as_float_list(product_image),
        "rhs_then_lhs_rotated_vector_xyz": as_float_list(rhs_then_lhs),
        "lhs_then_rhs_rotated_vector_xyz": as_float_list(lhs_then_rhs),
        "product_matches_rhs_then_lhs": (product_image - rhs_then_lhs).length < TOLERANCE,
        "product_matches_lhs_then_rhs": (product_image - lhs_then_rhs).length < TOLERANCE,
    }


def capture_precision_probe():
    input_value = 1.234567890123
    euler_value = float(Euler((input_value, 0.0, 0.0), "XYZ").x)
    quaternion_value = float(Quaternion((input_value, 0.0, 0.0, 0.0)).w)
    rounded_float32 = struct_float32(input_value)
    return {
        "input_decimal": input_value,
        "float32_rounded_input": rounded_float32,
        "mathutils_euler_observed": euler_value,
        "mathutils_quaternion_observed": quaternion_value,
        "euler_matches_float32_rounding": euler_value == rounded_float32,
        "quaternion_matches_float32_rounding": quaternion_value == rounded_float32,
    }


def struct_float32(value):
    import struct

    return float(struct.unpack("f", struct.pack("f", float(value)))[0])


def build_fixture():
    version = bpy.app.version_string
    build_hash = bpy.app.build_hash
    if isinstance(build_hash, bytes):
        build_hash = build_hash.decode("ascii")
    return {
        "schema_version": 1,
        "evidence_kind": "native_blender_runtime_capture",
        "capture_method": "Blender background mode with factory startup and embedded Python script",
        "quaternion_algebra": "Hamilton",
        "quaternion_order": "mathutils constructor/readback is WXYZ; fixture also writes XYZW for QuatOps comparison",
        "rotation_action": "active rotation applied to vectors with quaternion @ Vector",
        "matrix_layout": "3x3 matrix rows as returned by mathutils.Matrix indexing; matrices act on column vectors",
        "proof_profile": {
            "euler_to_quaternion": "native mathutils.Euler.to_quaternion for all six Tait-Bryan orders; 64 seeded cases per order plus fixed and middle-axis gimbal probes",
            "euler_readback_and_threshold_neighbors": "native Quaternion.to_euler and Matrix.to_euler readbacks, with and without Euler compatibility, across all six orders",
            "matrix_to_quaternion": "native mathutils.Matrix.to_quaternion, including near-half-turn and negative-trace matrices",
            "axis_angle": "native mathutils.Quaternion axis-angle construction and to_axis_angle for q and -q",
            "rotation_vector": "native mathutils.Quaternion exponential-map constructor and to_exponential_map readback",
            "slerp_shortest": "native mathutils.Quaternion.slerp including antipodal, near-identical, and greater-than-half-turn endpoints",
            "slerp_full_path": "not exposed by the captured mathutils.Quaternion.slerp API",
            "normalization_edges": "native mathutils.Quaternion.normalized for zero, tiny, unit, sign-reversed, and nonunit values",
            "basis_vector_action": "native quaternion @ Vector and mathutils.Matrix action on positive basis vectors",
            "composition_and_frame_map": "native quaternion composition and cross-product probes; source basis is declared explicitly per fixture",
            "rotation_difference": "native mathutils.Quaternion.rotation_difference and both relative-product candidates",
            "swing_twist": "native Quaternion.to_swing_twist for X, Y, and Z twist axes, with reconstruction products",
            "euler_continuity": "native Quaternion.to_euler(order, euler_compat) trajectories across 180 degrees and middle-axis gimbal neighborhoods",
            "scene_object_transform": "native Object.rotation_mode, rotation_euler, rotation_quaternion, and matrix_world rotation readbacks",
        },
        "engine": {
            "name": "Blender",
            "version": version,
            "major": int(bpy.app.version[0]),
            "minor": int(bpy.app.version[1]),
            "patch": int(bpy.app.version[2]),
            "build_hash": build_hash,
            "build_branch": str(bpy.app.build_branch.decode("ascii") if isinstance(bpy.app.build_branch, bytes) else bpy.app.build_branch),
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
            "precision_probe": capture_precision_probe(),
        },
        "source_frame": {
            "handedness": "right-handed",
            "positive_x": "right",
            "positive_y": "forward",
            "positive_z": "up",
            "declaration": "explicit probe frame; not inferred from Blender or an imported object's local axes",
        },
        "blender_world_frame_probe": {
            "positive_x_image": [1.0, 0.0, 0.0],
            "positive_y_image": [0.0, 1.0, 0.0],
            "positive_z_image": [0.0, 0.0, 1.0],
            "positive_x_cross_positive_y": as_float_list(Vector((1.0, 0.0, 0.0)).cross(Vector((0.0, 1.0, 0.0)))),
            "declared_probe_semantics": "+X right, +Y forward, +Z up for this fixture",
            "semantic_imported_object_forward_axis": "not assumed",
        },
        "source_to_blender_axis_map": capture_explicit_source_map(),
        "canonical_interop": capture_canonical_interop(),
        "cases": capture_base_cases(),
        "cross_products": capture_cross_products(),
        "quaternion_composition": capture_quaternion_composition(),
        "scene_object_rotation": capture_scene_object_routes(),
        "extended_conversions": {
            "capabilities": {
                "euler_orders": "native mathutils.Euler and Quaternion readback for all six Tait-Bryan orders",
                "slerp_full_path": "not exposed by the public mathutils.Quaternion.slerp method",
            },
            "euler_orders": capture_euler_cases(),
            "matrix_to_quaternion": capture_matrix_to_quaternion(),
            "axis_angle": capture_axis_angle(),
            "rotation_vector": capture_rotation_vectors(),
            "slerp": capture_slerp(),
            "normalization": capture_normalization(),
            "rotation_difference": capture_rotation_difference(),
            "swing_twist": capture_swing_twist(),
            "euler_compat_trajectories": capture_euler_compat_trajectories(),
        },
    }


def parse_output_path():
    script_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(script_args)
    version = bpy.app.version_string
    if Path(__file__).resolve().parent.name != version:
        raise RuntimeError(
            "capture directory %r does not match Blender runtime version %r"
            % (Path(__file__).resolve().parent.name, version)
        )
    return args.output or Path(tempfile.gettempdir()) / "quatops-blender-full.json"


def main():
    output = parse_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_fixture()
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("Wrote native Blender rotation capture:", output)
    print("Blender:", payload["engine"]["version"], payload["engine"]["build_hash"])
    print("Euler rows:", len(payload["extended_conversions"]["euler_orders"]))


if __name__ == "__main__":
    main()
