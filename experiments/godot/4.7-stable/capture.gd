extends SceneTree

const CASES := [
	{"case_id": "identity", "angles_degrees_xyz": Vector3(0.0, 0.0, 0.0)},
	{"case_id": "x_positive_90", "angles_degrees_xyz": Vector3(90.0, 0.0, 0.0)},
	{"case_id": "y_positive_90", "angles_degrees_xyz": Vector3(0.0, 90.0, 0.0)},
	{"case_id": "z_positive_90", "angles_degrees_xyz": Vector3(0.0, 0.0, 90.0)},
	{"case_id": "mixed_positive", "angles_degrees_xyz": Vector3(31.41, -49.16, 12.34)},
	{"case_id": "mixed_negative", "angles_degrees_xyz": Vector3(-50.0, 28.0, 39.0)},
	{"case_id": "near_gimbal", "angles_degrees_xyz": Vector3(89.9, 27.0, -41.0)},
	{"case_id": "gimbal", "angles_degrees_xyz": Vector3(90.0, 27.0, -41.0)},
	{"case_id": "x_gimbal_threshold_below_lower", "angles_degrees_xyz": Vector3(89.91, 27.0, -41.0)},
	{"case_id": "x_gimbal_threshold_above_lower", "angles_degrees_xyz": Vector3(89.93, 27.0, -41.0)},
	{"case_id": "x_gimbal_threshold_below_upper", "angles_degrees_xyz": Vector3(90.07, 27.0, -41.0)},
	{"case_id": "x_gimbal_threshold_above_upper", "angles_degrees_xyz": Vector3(90.09, 27.0, -41.0)},
	{"case_id": "x_over_90", "angles_degrees_xyz": Vector3(105.0, 23.0, -17.0)},
	{"case_id": "y_positive_180", "angles_degrees_xyz": Vector3(0.0, 180.0, 0.0)},
	{"case_id": "y_negative_180", "angles_degrees_xyz": Vector3(0.0, -180.0, 0.0)},
]
const EULER_ORDER_NAMES := ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]
const EULER_RANDOM_SEED := 0x514F5053


func _initialize() -> void:
	var fixture := capture_fixture()
	var user_args := OS.get_cmdline_user_args()
	var temp_dir := OS.get_environment("TEMP")
	var output_path := temp_dir.path_join("quatops-godot-full.json") if not temp_dir.is_empty() else "user://quatops-godot-full.json"
	if not user_args.is_empty():
		output_path = user_args[0]
	var file := FileAccess.open(output_path, FileAccess.WRITE)
	assert(file != null, "Could not open fixture output: %s" % output_path)
	file.store_string(JSON.stringify(fixture, "\t") + "\n")
	file.close()
	print("Captured native Godot fixtures to %s (Godot %s)." % [output_path, Engine.get_version_info().string])
	quit()


func capture_fixture() -> Dictionary:
	var version: Dictionary = Engine.get_version_info()
	return {
		"schema_version": 1,
		"evidence_kind": "native_godot_runtime_capture",
		"capture_method": "Godot headless GDScript SceneTree script",
		"quaternion_algebra": "Hamilton",
		"quaternion_order": "xyzw constructor and named x/y/z/w properties",
		"rotation_action": "active rotation applied to vectors",
		"matrix_layout": "Basis x/y/z are rotated positive basis columns",
		"proof_profile": {
			"euler_to_quaternion": "native Quaternion.from_euler YXZ and all six Basis.from_euler orders",
			"euler_readback_and_threshold_neighbors": "native Basis.get_euler and Quaternion.get_euler for all orders; probes around positive and negative 90 degrees",
			"matrix_to_quaternion": "native",
			"axis_angle": "native",
			"rotation_vector": "native Quaternion.exp/log recorded as rotation-vector APIs and compared with QuatOps from_rotvec/to_rotvec; q and -q branches captured",
			"slerp_shortest": "native Quaternion.slerp",
			"slerp_full_path": "native Quaternion.slerpni",
			"normalization_edges": "native",
			"basis_vector_action": "native",
			"composition_and_frame_map": "native",
		},
		"engine": {
			"name": "Godot Engine",
			"version": version.string,
			"major": version.major,
			"minor": version.minor,
			"patch": version.patch,
			"status": version.status,
			"build": version.build,
			"hash": version.hash,
			"platform": OS.get_name(),
			"precision": "float32",
		},
		"source_frame": {
			"handedness": "right-handed",
			"positive_x": "right",
			"positive_y": "forward",
			"positive_z": "up",
		},
		"godot_world_frame": {
			"handedness": "right-handed",
			"positive_x": "right",
			"positive_y": "up",
			"positive_z": "backward",
			"forward_direction": "negative_z",
		},
		"source_to_godot_axis_map": capture_axis_map(),
		"canonical_interop": capture_canonical_interop(),
		"cases": capture_euler_cases(),
		"extended_conversions": capture_extended_conversions(),
		"cross_products": capture_cross_products(),
		"quaternion_composition": capture_quaternion_composition(),
		"transform_composition": capture_transform_composition(),
	}


func capture_euler_cases() -> Array[Dictionary]:
	var result: Array[Dictionary] = []
	for input_case in CASES:
		var angles_degrees: Vector3 = input_case.angles_degrees_xyz
		var angles_radians := angles_degrees * (PI / 180.0)
		var quaternion := Quaternion.from_euler(angles_radians)
		var quaternion_euler_readback := quaternion.get_euler(EULER_ORDER_YXZ)
		var quaternion_euler_roundtrip := Quaternion.from_euler(quaternion_euler_readback)
		var basis := Basis.from_euler(angles_radians, EULER_ORDER_YXZ)
		var quaternion_basis := Basis(quaternion)
		var quaternion_axes := rotated_axes(quaternion)
		var basis_axes := [vector(basis.x), vector(basis.y), vector(basis.z)]
		var quaternion_basis_axes := [vector(quaternion_basis.x), vector(quaternion_basis.y), vector(quaternion_basis.z)]
		var basis_quaternion := basis.get_rotation_quaternion()
		var row_records := basis_rows(basis)
		var axes_from_basis := [basis * Vector3.RIGHT, basis * Vector3.UP, basis * Vector3.BACK]
		var row_axes := [vector(axes_from_basis[0]), vector(axes_from_basis[1]), vector(axes_from_basis[2])]
		assert(vectors_near(quaternion_axes, basis_axes, 2.0e-5), "Quaternion.from_euler and Basis.from_euler differ: %s" % input_case.case_id)
		result.append({
			"case_id": input_case.case_id,
			"input_degrees_xyz": vector(angles_degrees),
			"input_radians_xyz": vector(angles_radians),
			"quaternion_from_euler_xyzw": quaternion_xyzw(quaternion),
			"quaternion_get_euler_radians_xyz": vector(quaternion_euler_readback),
			"quaternion_euler_readback_roundtrip_xyzw": quaternion_xyzw(quaternion_euler_roundtrip),
			"quaternion_euler_readback_roundtrip_angle_error_degrees": orientation_error_degrees(quaternion, quaternion_euler_roundtrip),
			"quaternion_rotated_positive_basis_vectors_xyz": quaternion_axes,
			"basis_from_euler_columns_xyz": basis_axes,
			"basis_from_euler_rows": row_records,
			"basis_from_euler_rotated_positive_basis_vectors_xyz": row_axes,
			"basis_quaternion_xyzw": quaternion_xyzw(basis_quaternion),
			"quaternion_basis_columns_xyz": quaternion_basis_axes,
		})
	return result


func euler_order_constant(order_name: String) -> int:
	match order_name:
		"XYZ": return EULER_ORDER_XYZ
		"XZY": return EULER_ORDER_XZY
		"YXZ": return EULER_ORDER_YXZ
		"YZX": return EULER_ORDER_YZX
		"ZXY": return EULER_ORDER_ZXY
		"ZYX": return EULER_ORDER_ZYX
		_: assert(false, "Unknown Godot Euler order: %s" % order_name)
	return EULER_ORDER_YXZ


func euler_order_input_cases() -> Array[Dictionary]:
	var result: Array[Dictionary] = []
	for input_case in CASES:
		result.append({"case_id": "baseline_%s" % input_case.case_id, "angles_degrees_xyz": input_case.angles_degrees_xyz})
	result.append({"case_id": "diagnostic_10_20_30", "angles_degrees_xyz": Vector3(10.0, 20.0, 30.0)})
	result.append({"case_id": "outside_canonical_ranges", "angles_degrees_xyz": Vector3(420.0, -390.0, 765.0)})

	var positive_probes := [
		{"suffix": "near_gimbal", "angle": 89.9},
		{"suffix": "threshold_89_91", "angle": 89.91},
		{"suffix": "threshold_89_93", "angle": 89.93},
		{"suffix": "gimbal", "angle": 90.0},
		{"suffix": "threshold_90_07", "angle": 90.07},
		{"suffix": "threshold_90_09", "angle": 90.09},
	]
	for axis_name in ["y", "z"]:
		for probe in positive_probes:
			var angles := Vector3(27.0, -41.0, -41.0)
			if axis_name == "y":
				angles.y = probe.angle
			else:
				angles.z = probe.angle
			result.append({"case_id": "%s_%s" % [axis_name, probe.suffix], "angles_degrees_xyz": angles})

	for axis_name in ["x", "y", "z"]:
		for magnitude in [89.9, 89.91, 89.93, 90.0, 90.07, 90.09]:
			var angles := Vector3(27.0, -41.0, -41.0)
			if axis_name == "x":
				angles.x = -magnitude
			elif axis_name == "y":
				angles.y = -magnitude
			else:
				angles.z = -magnitude
			result.append({"case_id": "%s_negative_%s" % [axis_name, str(magnitude).replace(".", "_")], "angles_degrees_xyz": angles})

	var float32_neighbors := [
		{"suffix": "positive_90_below", "angle": 89.99999237060547},
		{"suffix": "positive_90_above", "angle": 90.00000762939453},
		{"suffix": "negative_90_below", "angle": -90.00000762939453},
		{"suffix": "negative_90_above", "angle": -89.99999237060547},
	]
	for axis_name in ["x", "y", "z"]:
		for neighbor in float32_neighbors:
			var angles := Vector3(27.0, -41.0, -41.0)
			if axis_name == "x":
				angles.x = neighbor.angle
			elif axis_name == "y":
				angles.y = neighbor.angle
			else:
				angles.z = neighbor.angle
			result.append({"case_id": "%s_%s_ulp" % [axis_name, neighbor.suffix], "angles_degrees_xyz": angles})

	var rng := RandomNumberGenerator.new()
	rng.seed = EULER_RANDOM_SEED
	for index in range(8):
		var angles := Vector3(
			rng.randf_range(-720.0, 720.0),
			rng.randf_range(-720.0, 720.0),
			rng.randf_range(-720.0, 720.0))
		result.append({"case_id": "seeded_random_%02d" % index, "angles_degrees_xyz": angles})
	return result


func capture_euler_orders() -> Array[Dictionary]:
	var result: Array[Dictionary] = []
	for input_case in euler_order_input_cases():
		var angles_degrees: Vector3 = input_case.angles_degrees_xyz
		var angles_radians := angles_degrees * (PI / 180.0)
		for order_name in EULER_ORDER_NAMES:
			var order_value := euler_order_constant(order_name)
			var basis := Basis.from_euler(angles_radians, order_value)
			var basis_quaternion := basis.get_rotation_quaternion()
			var constructor_quaternion := Quaternion(basis)
			var basis_readback := basis.get_euler(order_value)
			var quaternion_readback := basis_quaternion.get_euler(order_value)
			var basis_roundtrip_quaternion := Basis.from_euler(basis_readback, order_value).get_rotation_quaternion()
			var quaternion_roundtrip_quaternion := Basis.from_euler(quaternion_readback, order_value).get_rotation_quaternion()
			var basis_columns := [vector(basis.x), vector(basis.y), vector(basis.z)]
			var basis_axes := [basis * Vector3.RIGHT, basis * Vector3.UP, basis * Vector3.BACK]
			var constructor_matches := vectors_near(rotated_axes(constructor_quaternion), rotated_axes(basis_quaternion), 2.0e-5)
			assert(constructor_matches, "Quaternion(Basis) disagreed for Euler%s %s" % [order_name, input_case.case_id])
			result.append({
				"case_id": input_case.case_id,
				"order": order_name,
				"input_degrees_xyz": vector(angles_degrees),
				"input_radians_xyz": vector(angles_radians),
				"basis_from_euler_rows": basis_rows(basis),
				"basis_from_euler_columns_xyz": basis_columns,
				"basis_from_euler_rotated_positive_basis_vectors_xyz": [vector(basis_axes[0]), vector(basis_axes[1]), vector(basis_axes[2])],
				"basis_get_rotation_quaternion_xyzw": quaternion_xyzw(basis_quaternion),
				"quaternion_basis_constructor_xyzw": quaternion_xyzw(constructor_quaternion),
				"basis_get_euler_radians_xyz": vector(basis_readback),
				"quaternion_get_euler_radians_xyz": vector(quaternion_readback),
				"basis_readback_roundtrip_quaternion_xyzw": quaternion_xyzw(basis_roundtrip_quaternion),
				"quaternion_readback_roundtrip_quaternion_xyzw": quaternion_xyzw(quaternion_roundtrip_quaternion),
				"basis_readback_roundtrip_angle_error_degrees": orientation_error_degrees(basis_quaternion, basis_roundtrip_quaternion),
				"quaternion_readback_roundtrip_angle_error_degrees": orientation_error_degrees(basis_quaternion, quaternion_roundtrip_quaternion),
				"constructor_matches_basis_rotation": constructor_matches,
			})
	return result


func capture_extended_conversions() -> Dictionary:
	return {
		"capabilities": {
			"euler_orders": "Basis.from_euler, Basis.get_euler, Quaternion.get_euler for XYZ/XZY/YXZ/YZX/ZXY/ZYX",
			"matrix_to_quaternion": "native Basis.get_rotation_quaternion and Quaternion(Basis)",
		"matrix_conversion_diagnostics": "native readback from an almost singular non-orthonormal Basis; diagnostic only",
			"axis_angle": "native Quaternion(axis, angle), get_axis, get_angle",
			"shortest_arc": "native Quaternion(from_vector, to_vector), including parallel, antiparallel, near-antiparallel, and nonunit vectors",
			"rotation_vector": "native Quaternion.exp/log with identity, 90-degree, 180-degree, and antipodal quaternion samples",
			"slerp_shortest": "native Quaternion.slerp",
			"slerp_full_path": "native Quaternion.slerpni",
			"normalization": "native Quaternion.normalized",
		},
		"euler_orders": capture_euler_orders(),
		"matrix_to_quaternion": capture_matrix_to_quaternion(),
		"matrix_conversion_diagnostics": capture_matrix_conversion_diagnostics(),
		"axis_angle": capture_axis_angle(),
		"rotation_vector": capture_rotation_vectors(),
		"rotation_exp_log": capture_rotation_exp_log(),
		"from_to": capture_from_to(),
		"slerp": capture_slerp(),
		"normalization": capture_normalization(),
	}


func capture_matrix_to_quaternion() -> Array[Dictionary]:
	var inv_sqrt_14 := 1.0 / sqrt(14.0)
	var cases := [
		{"case_id": "identity", "quaternion": Quaternion.IDENTITY},
		{"case_id": "x_90", "quaternion": Quaternion(Vector3.RIGHT, PI / 2.0)},
		{"case_id": "y_90", "quaternion": Quaternion(Vector3.UP, PI / 2.0)},
		{"case_id": "z_90", "quaternion": Quaternion(Vector3.BACK, PI / 2.0)},
		{"case_id": "x_180", "quaternion": Quaternion(Vector3.RIGHT, PI)},
		{"case_id": "y_180", "quaternion": Quaternion(Vector3.UP, PI)},
		{"case_id": "z_180", "quaternion": Quaternion(Vector3.BACK, PI)},
		{"case_id": "mixed_axis", "quaternion": Quaternion(Vector3(1, 2, 3).normalized(), 1.234)},
		{"case_id": "negative_trace", "quaternion": Quaternion(Vector3(inv_sqrt_14, 2.0 * inv_sqrt_14, 3.0 * inv_sqrt_14), deg_to_rad(170.0))},
		{"case_id": "near_negative_trace", "quaternion": Quaternion(Vector3(inv_sqrt_14, 2.0 * inv_sqrt_14, 3.0 * inv_sqrt_14), deg_to_rad(179.999))},
	]
	var result: Array[Dictionary] = []
	for item in cases:
		var source: Quaternion = item.quaternion
		var basis := Basis(source)
		var from_basis := basis.get_rotation_quaternion()
		var from_constructor := Quaternion(basis)
		var axes := [vector(basis.x), vector(basis.y), vector(basis.z)]
		var same_orientation := vectors_near(rotated_axes(source), rotated_axes(from_basis), 2.0e-5)
		assert(same_orientation, "Basis.get_rotation_quaternion changed %s" % item.case_id)
		assert(vectors_near(rotated_axes(from_basis), rotated_axes(from_constructor), 2.0e-5), "Quaternion(Basis) disagreed for %s" % item.case_id)
		result.append({
			"case_id": item.case_id,
			"matrix_rows_native": basis_rows(basis),
			"matrix_columns_native": axes,
			"source_quaternion_xyzw": quaternion_xyzw(source),
			"basis_get_rotation_quaternion_xyzw": quaternion_xyzw(from_basis),
			"quaternion_basis_constructor_xyzw": quaternion_xyzw(from_constructor),
			"same_rotation": same_orientation,
		})
	return result


func capture_matrix_conversion_diagnostics() -> Array[Dictionary]:
	# A valid rotation Basis must be orthonormal. These almost-collinear axes
	# intentionally violate that contract and are recorded as native-only data.
	var result: Array[Dictionary] = []
	for epsilon in [1.0e-2, 1.0e-4, 1.0e-6]:
		var basis := Basis(Vector3.RIGHT, Vector3(1.0, epsilon, 0.0), Vector3.BACK)
		var basis_quaternion := basis.get_rotation_quaternion()
		var constructor_quaternion := Quaternion(basis)
		result.append({
			"case_id": "near_collinear_basis_det_%s" % str(epsilon),
			"diagnostic_only": true,
			"matrix_rows_native": basis_rows(basis),
			"matrix_columns_native": [vector(basis.x), vector(basis.y), vector(basis.z)],
			"matrix_determinant": basis.determinant(),
			"basis_get_rotation_quaternion_xyzw": quaternion_xyzw(basis_quaternion),
			"basis_get_rotation_quaternion_is_finite": basis_quaternion.is_finite(),
			"basis_get_rotation_quaternion_norm": basis_quaternion.length(),
			"quaternion_basis_constructor_xyzw": quaternion_xyzw(constructor_quaternion),
			"quaternion_basis_constructor_is_finite": constructor_quaternion.is_finite(),
			"quaternion_basis_constructor_norm": constructor_quaternion.length(),
		})
	return result


func capture_axis_angle() -> Array[Dictionary]:
	var inv_sqrt_14 := 1.0 / sqrt(14.0)
	var pi := PI
	var cases := [
		{"case_id": "zero_angle", "axis": Vector3.RIGHT, "angle": 0.0},
		{"case_id": "tiny_angle", "axis": Vector3(1, 2, 3).normalized(), "angle": 1.0e-7},
		{"case_id": "nonprincipal_angle", "axis": Vector3(1, 2, 3).normalized(), "angle": 2.4},
		{"case_id": "exact_pi", "axis": Vector3.BACK, "angle": pi},
		{"case_id": "just_above_pi", "axis": Vector3(-2.0 * inv_sqrt_14, inv_sqrt_14, 3.0 * inv_sqrt_14), "angle": pi + 0.2},
	]
	var result: Array[Dictionary] = []
	for item in cases:
		var value := Quaternion(item.axis, item.angle)
		var negative := Quaternion(-value.x, -value.y, -value.z, -value.w)
		result.append({
			"case_id": item.case_id,
			"input_axis_xyz": vector(item.axis),
			"input_angle_radians": item.angle,
			"constructed_quaternion_xyzw": quaternion_xyzw(value),
			"negative_quaternion_xyzw": quaternion_xyzw(negative),
			"axis_angle_readback_axis_xyz": vector(value.get_axis()),
			"axis_angle_readback_angle_radians": value.get_angle(),
			"negative_axis_angle_readback_axis_xyz": vector(negative.get_axis()),
			"negative_axis_angle_readback_angle_radians": negative.get_angle(),
			"rotation_vector_readback_xyz": vector(quaternion_to_rotvec(value)),
			"negative_rotation_vector_readback_xyz": vector(quaternion_to_rotvec(negative)),
			"constructed_norm": value.length(),
		})
	return result


func capture_rotation_vectors() -> Array[Dictionary]:
	var pi := PI
	var inv_sqrt_14 := 1.0 / sqrt(14.0)
	var inputs := [
		{"case_id": "zero", "rotation_vector": Vector3.ZERO},
		{"case_id": "tiny", "rotation_vector": Vector3(1.0e-7, -2.0e-7, 3.0e-7)},
		{"case_id": "mixed", "rotation_vector": Vector3(0.3, -0.7, 1.2)},
		{"case_id": "exact_pi", "rotation_vector": Vector3(0, 0, pi)},
		{"case_id": "just_above_pi", "rotation_vector": Vector3(-2.0 * inv_sqrt_14, inv_sqrt_14, 3.0 * inv_sqrt_14) * (pi + 0.2)},
	]
	var result: Array[Dictionary] = []
	for item in inputs:
		var input_vector: Vector3 = item.rotation_vector
		var value := rotvec_to_quaternion(input_vector)
		result.append({
			"case_id": item.case_id,
			"input_rotation_vector_xyz": vector(input_vector),
			"constructed_quaternion_xyzw": quaternion_xyzw(value),
			"axis_angle_readback_axis_xyz": vector(value.get_axis()),
			"axis_angle_readback_angle_radians": value.get_angle(),
			"rotation_vector_readback_xyz": vector(quaternion_to_rotvec(value)),
			"constructed_norm": value.length(),
		})
	return result


func capture_rotation_exp_log() -> Dictionary:
	var z90 := Quaternion(Vector3.BACK, PI / 2.0)
	var z180 := Quaternion(Vector3.BACK, PI)
	var exp_inputs := [
		{"case_id": "identity", "rotation_vector": Vector3.ZERO},
		{"case_id": "tiny", "rotation_vector": Vector3(1.0e-7, -2.0e-7, 3.0e-7)},
		{"case_id": "z_90_degrees", "rotation_vector": Vector3(0.0, 0.0, PI / 2.0)},
		{"case_id": "z_180_degrees", "rotation_vector": Vector3(0.0, 0.0, PI)},
		{"case_id": "mixed", "rotation_vector": Vector3(0.3, -0.7, 1.2)},
		{"case_id": "antipodal_z_90_branch", "rotation_vector": Vector3(0.0, 0.0, -3.0 * PI / 2.0)},
	]
	var exp_result: Array[Dictionary] = []
	for item in exp_inputs:
		var rotation_vector: Vector3 = item.rotation_vector
		var pure := Quaternion(rotation_vector.x, rotation_vector.y, rotation_vector.z, 0.0)
		var value := pure.exp()
		var logarithm := value.log()
		exp_result.append({
			"case_id": item.case_id,
			"input_rotation_vector_xyz": vector(rotation_vector),
			"input_pure_quaternion_xyzw": quaternion_xyzw(pure),
			"exp_result_quaternion_xyzw": quaternion_xyzw(value),
			"exp_result_norm": value.length(),
			"exp_result_is_finite": value.is_finite(),
			"exp_then_log_quaternion_xyzw": quaternion_xyzw(logarithm),
			"exp_then_log_rotation_vector_xyz": vector(Vector3(logarithm.x, logarithm.y, logarithm.z)),
		})

	var log_inputs := [
		{"case_id": "identity", "quaternion": Quaternion.IDENTITY},
		{"case_id": "z_90_degrees", "quaternion": z90},
		{"case_id": "z_180_degrees", "quaternion": z180},
		{"case_id": "antipodal_z_90", "quaternion": Quaternion(-z90.x, -z90.y, -z90.z, -z90.w)},
		{"case_id": "antipodal_z_180", "quaternion": Quaternion(-z180.x, -z180.y, -z180.z, -z180.w)},
	]
	var log_result: Array[Dictionary] = []
	for item in log_inputs:
		var value: Quaternion = item.quaternion
		var logarithm := value.log()
		log_result.append({
			"case_id": item.case_id,
			"input_quaternion_xyzw": quaternion_xyzw(value),
			"log_result_quaternion_xyzw": quaternion_xyzw(logarithm),
			"rotation_vector_xyz": vector(Vector3(logarithm.x, logarithm.y, logarithm.z)),
		})
	return {"exp": exp_result, "log": log_result}


func capture_from_to() -> Array[Dictionary]:
	var cases := [
		{"case_id": "parallel", "from": Vector3.RIGHT, "to": Vector3.RIGHT, "unique": true},
		{"case_id": "orthogonal", "from": Vector3.RIGHT, "to": Vector3.UP, "unique": true},
		{"case_id": "nearly_parallel", "from": Vector3.RIGHT, "to": Vector3(1.0, 1.0e-5, 0.0), "unique": true},
		{"case_id": "resolved_nearly_parallel", "from": Vector3.RIGHT, "to": Vector3(1.0, 1.0e-3, 0.0), "unique": true},
		{"case_id": "antiparallel_x", "from": Vector3.RIGHT, "to": -Vector3.RIGHT, "unique": false},
		{"case_id": "antiparallel_diagonal", "from": Vector3(1.0, 2.0, -3.0), "to": Vector3(-1.0, -2.0, 3.0), "unique": false},
		{"case_id": "nearly_antiparallel", "from": Vector3.RIGHT, "to": Vector3(-1.0, 1.0e-4, 0.0), "unique": true},
		{"case_id": "resolved_nearly_antiparallel", "from": Vector3.RIGHT, "to": Vector3(-1.0, 1.0e-2, 0.0), "unique": true},
		{"case_id": "nonunit_inputs", "from": Vector3(2.0, 0.0, 0.0), "to": Vector3(0.0, 3.0, 0.0), "unique": true},
	]
	var result: Array[Dictionary] = []
	for item in cases:
		var source: Vector3 = item.from
		var target: Vector3 = item.to
		var source_unit := source.normalized()
		var target_unit := target.normalized()
		var value := Quaternion(source, target)
		var rotated_source := value * source_unit
		var value_is_finite := value.is_finite()
		var target_matches := value_is_finite and near_vector(rotated_source, target_unit, 2.0e-5)
		var target_alignment_error := rotated_source.distance_to(target_unit) if value_is_finite else -1.0
		result.append({
			"case_id": item.case_id,
			"input_from_xyz": vector(source),
			"input_to_xyz": vector(target),
			"source_unit_xyz": vector(source_unit),
			"target_unit_xyz": vector(target_unit),
			"quaternion_xyzw": quaternion_xyzw(value),
			"quaternion_norm": value.length(),
			"quaternion_is_finite": value_is_finite,
			"rotation_axis_xyz": vector(value.get_axis()),
			"rotation_angle_radians": value.get_angle(),
			"rotated_source_xyz": vector(rotated_source),
			"target_alignment_error": target_alignment_error,
			"target_matches": target_matches,
			"shortest_rotation_orientation_is_unique": item.unique,
		})
	return result


func capture_slerp() -> Array[Dictionary]:
	var inv_sqrt_14 := 1.0 / sqrt(14.0)
	var identity := Quaternion.IDENTITY
	var x90 := Quaternion(Vector3.RIGHT, PI / 2.0)
	var z270 := Quaternion(Vector3.BACK, 1.5 * PI)
	var arbitrary_a := Quaternion(Vector3(inv_sqrt_14, 2.0 * inv_sqrt_14, 3.0 * inv_sqrt_14), 1.1)
	var arbitrary_b := Quaternion(Vector3(-2.0 * inv_sqrt_14, inv_sqrt_14, 3.0 * inv_sqrt_14), 2.2)
	var near_a := Quaternion(Vector3.BACK, 0.8)
	var near_b := Quaternion(Vector3.BACK, 0.801)
	var near_pi := Quaternion(Vector3.BACK, PI - 1.0e-4)
	var inputs: Array[Dictionary] = []
	for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
		inputs.append({"case_id": "identity_to_x90", "lhs": identity, "rhs": x90, "alpha": weight})
		inputs.append({"case_id": "long_path", "lhs": identity, "rhs": z270, "alpha": weight})
		inputs.append({"case_id": "antipodal", "lhs": arbitrary_a, "rhs": Quaternion(-arbitrary_a.x, -arbitrary_a.y, -arbitrary_a.z, -arbitrary_a.w), "alpha": weight})
	for weight in [0.0, 0.5, 1.0]:
		inputs.append({"case_id": "arbitrary_pair", "lhs": arbitrary_a, "rhs": arbitrary_b, "alpha": weight})
	inputs.append({"case_id": "nearly_identical", "lhs": near_a, "rhs": near_b, "alpha": 0.5})
	inputs.append({"case_id": "near_180_separation", "lhs": identity, "rhs": near_pi, "alpha": 0.5})
	inputs.append({"case_id": "extrapolation", "lhs": identity, "rhs": x90, "alpha": -0.5})
	inputs.append({"case_id": "extrapolation", "lhs": identity, "rhs": x90, "alpha": 1.5})
	var result: Array[Dictionary] = []
	for item in inputs:
		var shortest: Quaternion = item.lhs.slerp(item.rhs, item.alpha)
		var full_path: Quaternion = item.lhs.slerpni(item.rhs, item.alpha)
		result.append({
			"case_id": item.case_id,
			"alpha": item.alpha,
			"lhs_xyzw": quaternion_xyzw(item.lhs),
			"rhs_xyzw": quaternion_xyzw(item.rhs),
			"slerp_xyzw": quaternion_xyzw(shortest),
			"slerp_norm": shortest.length(),
			"slerp_is_finite": shortest.is_finite(),
			"slerpni_xyzw": quaternion_xyzw(full_path),
			"slerpni_norm": full_path.length(),
			"slerpni_is_finite": full_path.is_finite(),
		})
	return result


func capture_normalization() -> Array[Dictionary]:
	var inputs := [
		{"case_id": "identity", "quaternion": Quaternion.IDENTITY},
		{"case_id": "zero", "quaternion": Quaternion(0, 0, 0, 0)},
		{"case_id": "below_threshold", "quaternion": Quaternion(5.0e-5, 0, 0, 0)},
		{"case_id": "above_threshold", "quaternion": Quaternion(2.0e-4, 0, 0, 0)},
		{"case_id": "nonunit", "quaternion": Quaternion(0.2, -0.3, 0.4, 0.5)},
	]
	var result: Array[Dictionary] = []
	for item in inputs:
		var input: Quaternion = item.quaternion
		var normalized := input.normalized()
		result.append({
			"case_id": item.case_id,
			"input_xyzw": quaternion_xyzw(input),
			"input_norm_squared": input.length_squared(),
			"normalized_xyzw": quaternion_xyzw(normalized),
			"normalized_norm": normalized.length(),
			"normalized_is_finite": normalized.is_finite(),
		})
	return result


func rotvec_to_quaternion(value: Vector3) -> Quaternion:
	return Quaternion(value.x, value.y, value.z, 0.0).exp()


func quaternion_to_rotvec(value: Quaternion) -> Vector3:
	var logarithm := value.log()
	return Vector3(logarithm.x, logarithm.y, logarithm.z)


func capture_cross_products() -> Array[Dictionary]:
	var axes := [
		{"case_id": "x_cross_y", "lhs": Vector3.RIGHT, "rhs": Vector3.UP},
		{"case_id": "y_cross_z", "lhs": Vector3.UP, "rhs": Vector3.BACK},
		{"case_id": "z_cross_x", "lhs": Vector3.BACK, "rhs": Vector3.RIGHT},
	]
	var result: Array[Dictionary] = []
	for item in axes:
		var cross: Vector3 = item.lhs.cross(item.rhs)
		var reversed: Vector3 = item.rhs.cross(item.lhs)
		assert(near_vector(reversed, -cross, 1.0e-6), "Godot cross product is not anti-commutative: %s" % item.case_id)
		result.append({
			"case_id": item.case_id,
			"lhs_xyz": vector(item.lhs),
			"rhs_xyz": vector(item.rhs),
			"result_xyz": vector(cross),
			"reversed_result_xyz": vector(reversed),
		})
	return result


func capture_quaternion_composition() -> Dictionary:
	var lhs := Quaternion(Vector3.RIGHT, PI / 2.0)
	var rhs := Quaternion(Vector3.UP, PI / 2.0)
	var probe := Vector3(1.25, -2.5, 3.75)
	var product := lhs * rhs
	var product_image := product * probe
	var rhs_then_lhs := lhs * (rhs * probe)
	var lhs_then_rhs := rhs * (lhs * probe)
	var matches_rhs_then_lhs := near_vector(product_image, rhs_then_lhs, 1.0e-5)
	assert(matches_rhs_then_lhs, "Godot quaternion product order did not match right operand first.")
	return {
		"lhs_xyzw": quaternion_xyzw(lhs),
		"rhs_xyzw": quaternion_xyzw(rhs),
		"lhs_times_rhs_xyzw": quaternion_xyzw(product),
		"probe_vector_xyz": vector(probe),
		"product_rotated_vector_xyz": vector(product_image),
		"rhs_then_lhs_rotated_vector_xyz": vector(rhs_then_lhs),
		"lhs_then_rhs_rotated_vector_xyz": vector(lhs_then_rhs),
		"product_matches_rhs_then_lhs": matches_rhs_then_lhs,
	}


func capture_transform_composition() -> Dictionary:
	var lhs_rotation := Quaternion(Vector3.RIGHT, PI / 2.0)
	var rhs_rotation := Quaternion(Vector3.UP, PI / 2.0)
	var lhs_translation := Vector3(3.0, -2.0, 5.0)
	var rhs_translation := Vector3(-7.0, 11.0, 13.0)
	var lhs := Transform3D(Basis(lhs_rotation), lhs_translation)
	var rhs := Transform3D(Basis(rhs_rotation), rhs_translation)
	var probe := Vector3(2.0, 3.0, 5.0)
	var product := lhs * rhs
	var product_image := product * probe
	var rhs_then_lhs := lhs * (rhs * probe)
	var lhs_then_rhs := rhs * (lhs * probe)
	var product_rotation := product.basis.get_rotation_quaternion()
	var rhs_then_lhs_rotation := lhs.basis.get_rotation_quaternion() * rhs.basis.get_rotation_quaternion()
	var matches_rhs_then_lhs := near_vector(product_image, rhs_then_lhs, 1.0e-5)
	var rotation_matches_rhs_then_lhs := vectors_near(rotated_axes(product_rotation), rotated_axes(rhs_then_lhs_rotation), 2.0e-5)
	assert(matches_rhs_then_lhs, "Godot Transform3D product order did not match right transform first.")
	assert(rotation_matches_rhs_then_lhs, "Godot Transform3D rotation did not match right operand first.")
	return {
		"lhs_rotation_xyzw": quaternion_xyzw(lhs_rotation),
		"rhs_rotation_xyzw": quaternion_xyzw(rhs_rotation),
		"lhs_translation_xyz": vector(lhs_translation),
		"rhs_translation_xyz": vector(rhs_translation),
		"scale_xyz": vector(Vector3.ONE),
		"probe_position_xyz": vector(probe),
		"lhs_times_rhs_rotation_xyzw": quaternion_xyzw(product.basis.get_rotation_quaternion()),
		"product_rotation_matches_rhs_then_lhs": rotation_matches_rhs_then_lhs,
		"lhs_times_rhs_translation_xyz": vector(product.origin),
		"lhs_times_rhs_position_xyz": vector(product_image),
		"rhs_then_lhs_position_xyz": vector(rhs_then_lhs),
		"lhs_then_rhs_position_xyz": vector(lhs_then_rhs),
		"product_matches_rhs_then_lhs": matches_rhs_then_lhs,
	}


func capture_axis_map() -> Dictionary:
	var half_sqrt := sqrt(0.5)
	var source_quaternions := [
		{"axis": "+X right", "quat": Quaternion(half_sqrt, 0.0, 0.0, half_sqrt)},
		{"axis": "+Y forward", "quat": Quaternion(0.0, half_sqrt, 0.0, half_sqrt)},
		{"axis": "+Z up", "quat": Quaternion(0.0, 0.0, half_sqrt, half_sqrt)},
	]
	var source_basis := [Vector3.RIGHT, Vector3.UP, Vector3.BACK]
	var probes: Array[Dictionary] = []
	for item in source_quaternions:
		var source: Quaternion = item.quat
		var mapped := Quaternion(source.x, source.z, -source.y, source.w)
		var preserves := true
		for axis in source_basis:
			var expected := map_source_vector(source * axis)
			var actual := mapped * map_source_vector(axis)
			preserves = preserves and near_vector(expected, actual, 1.0e-6)
		assert(preserves, "Godot source-axis mapping failed for %s." % item.axis)
		probes.append({
			"source_rotation_axis": item.axis,
			"source_quaternion_xyzw": quaternion_xyzw(source),
			"godot_quaternion_xyzw": quaternion_xyzw(mapped),
			"godot_rotated_positive_basis_vectors_xyz": rotated_axes(mapped),
			"godot_quaternion_get_euler_radians_xyz": vector(mapped.get_euler(EULER_ORDER_YXZ)),
			"preserves_mapped_rotation_action": preserves,
		})
	return {
		"destination_axes_from_source": ["x", "z", "-y"],
		"determinant": 1,
		"principal_rotation_probes": probes,
	}


func capture_canonical_interop() -> Dictionary:
	var orientations := [
		["identity", [1.0, 0.0, 0.0, 0.0]],
		["quarter_x", [0.7071067811865476, 0.7071067811865476, 0.0, 0.0]],
		["quarter_y", [0.7071067811865476, 0.0, 0.7071067811865476, 0.0]],
		["quarter_z", [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]],
		["mixed_a", [0.8, 0.2, -0.4, 0.4]],
		["mixed_b", [0.7, -0.3, 0.2, 0.6164414002968976]],
		["antipode_mixed_a", [-0.8, -0.2, 0.4, -0.4]],
		["nonprincipal_halfturn", [0.0, 0.5773502691896258, 0.5773502691896258, 0.5773502691896258]],
	]
	var canonical_probe := Vector3(0.25, -1.5, 2.75)
	var native_probe := map_source_vector(canonical_probe)
	var native_by_id := {}
	var cases := []
	for item in orientations:
		var native := map_canonical_quaternion(item[1])
		native_by_id[item[0]] = native
		cases.append(capture_canonical_interop_orientation(item[0], item[1], native, native_probe))
	var lhs_rhs: Quaternion = native_by_id["mixed_a"] * native_by_id["mixed_b"]
	var rhs_lhs: Quaternion = native_by_id["mixed_b"] * native_by_id["mixed_a"]
	return {
		"contract_id": "quatops-canonical-hamilton-wxyz-active-rh-v1",
		"native_quaternion_order": "xyzw",
		"frame_map_destination_axes_from_source": ["x", "z", "-y"],
		"probe_vector_canonical_xyz": vector(canonical_probe),
		"probe_vector_native_xyz": vector(native_probe),
		"orientation_cases": cases,
		"composition": capture_canonical_interop_composition("mixed_a", "mixed_b", lhs_rhs, native_probe),
		"reverse_composition": capture_canonical_interop_composition("mixed_b", "mixed_a", rhs_lhs, native_probe),
	}


func capture_canonical_interop_composition(
	left_id: String, right_id: String, product: Quaternion, native_probe: Vector3
) -> Dictionary:
	return {
		"left_case_id": left_id,
		"right_case_id": right_id,
		"product_order": "lhs * rhs; rhs acts first",
		"native_product_quaternion": quaternion_xyzw_array(product),
		"native_rotated_positive_basis_vectors_xyz": rotated_axes(product),
		"native_rotated_probe_vector_xyz": vector(product * native_probe),
	}


func map_canonical_quaternion(wxyz: Array) -> Quaternion:
	var source := Quaternion(wxyz[1], wxyz[2], wxyz[3], wxyz[0])
	# The fixture's explicit Godot frame map is (x, z, -y), with det(B)=+1.
	return Quaternion(source.x, source.z, -source.y, source.w)


func capture_canonical_interop_orientation(
	case_id: String, canonical_wxyz: Array, native: Quaternion, native_probe: Vector3
) -> Dictionary:
	return {
		"case_id": case_id,
		"canonical_wxyz": canonical_wxyz,
		"native_quaternion": quaternion_xyzw_array(native),
		"native_rotated_positive_basis_vectors_xyz": rotated_axes(native),
		"native_rotated_probe_vector_xyz": vector(native * native_probe),
	}


func quaternion_xyzw_array(value: Quaternion) -> Array:
	return [value.x, value.y, value.z, value.w]


func map_source_vector(value: Vector3) -> Vector3:
	return Vector3(value.x, value.z, -value.y)


func rotated_axes(quaternion: Quaternion) -> Array[Dictionary]:
	return [
		vector(quaternion * Vector3.RIGHT),
		vector(quaternion * Vector3.UP),
		vector(quaternion * Vector3.BACK),
	]


func basis_rows(basis: Basis) -> Array[Dictionary]:
	return [
		vector(Vector3(basis.x.x, basis.y.x, basis.z.x)),
		vector(Vector3(basis.x.y, basis.y.y, basis.z.y)),
		vector(Vector3(basis.x.z, basis.y.z, basis.z.z)),
	]


func vector(value: Vector3) -> Dictionary:
	return {"x": value.x, "y": value.y, "z": value.z}


func quaternion_xyzw(value: Quaternion) -> Dictionary:
	return {"x": value.x, "y": value.y, "z": value.z, "w": value.w}


func near_vector(lhs: Vector3, rhs: Vector3, tolerance: float) -> bool:
	return lhs.distance_squared_to(rhs) <= tolerance * tolerance


func orientation_error_degrees(lhs: Quaternion, rhs: Quaternion) -> float:
	var relative: Quaternion = lhs.inverse() * rhs
	var relative_vector_length: float = Vector3(relative.x, relative.y, relative.z).length()
	return rad_to_deg(2.0 * atan2(relative_vector_length, abs(relative.w)))


func vectors_near(lhs: Array, rhs: Array, tolerance: float) -> bool:
	if lhs.size() != rhs.size():
		return false
	for index in range(lhs.size()):
		var left: Dictionary = lhs[index]
		var right: Dictionary = rhs[index]
		if not near_vector(Vector3(left.x, left.y, left.z), Vector3(right.x, right.y, right.z), tolerance):
			return false
	return true
