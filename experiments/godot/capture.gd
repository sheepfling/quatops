extends SceneTree

const ORIENTATIONS := [
    ["identity", [1.0, 0.0, 0.0, 0.0]],
    ["quarter_x", [0.7071067811865476, 0.7071067811865476, 0.0, 0.0]],
    ["quarter_y", [0.7071067811865476, 0.0, 0.7071067811865476, 0.0]],
    ["quarter_z", [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]],
    ["mixed_a", [0.8, 0.2, -0.4, 0.4]],
    ["mixed_b", [0.7, -0.3, 0.2, 0.6164414002968976]],
    ["antipode_mixed_a", [-0.8, -0.2, 0.4, -0.4]],
    ["nonprincipal_halfturn", [0.0, 0.5773502691896258, 0.5773502691896258, 0.5773502691896258]],
]
const CANONICAL_PROBE := Vector3(0.25, -1.5, 2.75)
const DESTINATION_AXES := ["x", "z", "-y"]


func _initialize() -> void:
    var version: Dictionary = Engine.get_version_info()
    var fixture := {
        "schema_version": 1,
        "evidence_kind": "native_godot_runtime_capture",
        "capture_method": "Godot headless GDScript SceneTree script",
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
        "source_to_godot_axis_map": {
            "destination_axes_from_source": DESTINATION_AXES,
            "determinant": 1,
        },
        "canonical_interop": capture_canonical_interop(),
    }
    var output_path := OS.get_environment("TEMP").path_join("quatops-godot-witnesses.json")
    var user_args := OS.get_cmdline_user_args()
    if not user_args.is_empty():
        output_path = user_args[0]
    var file := FileAccess.open(output_path, FileAccess.WRITE)
    assert(file != null, "Could not open fixture output: %s" % output_path)
    file.store_string(JSON.stringify(fixture, "\t") + "\n")
    file.close()
    print("Captured Godot quaternion witnesses to %s (%s)." % [output_path, version.string])
    quit()


func capture_canonical_interop() -> Dictionary:
    var native_probe := map_source_vector(CANONICAL_PROBE)
    var native_by_id: Dictionary = {}
    var cases: Array[Dictionary] = []
    for item in ORIENTATIONS:
        var case_id: String = item[0]
        var canonical_wxyz: Array = item[1]
        var native := map_canonical_quaternion(canonical_wxyz)
        native_by_id[case_id] = native
        cases.append(capture_orientation(case_id, canonical_wxyz, native, native_probe))

    var lhs_rhs: Quaternion = native_by_id["mixed_a"] * native_by_id["mixed_b"]
    var rhs_lhs: Quaternion = native_by_id["mixed_b"] * native_by_id["mixed_a"]
    return {
        "contract_id": "quatops-canonical-hamilton-wxyz-active-rh-v1",
        "native_quaternion_order": "xyzw",
        "frame_map_destination_axes_from_source": DESTINATION_AXES,
        "probe_vector_canonical_xyz": vector(CANONICAL_PROBE),
        "probe_vector_native_xyz": vector(native_probe),
        "orientation_cases": cases,
        "composition": capture_composition("mixed_a", "mixed_b", lhs_rhs, native_probe),
        "reverse_composition": capture_composition("mixed_b", "mixed_a", rhs_lhs, native_probe),
    }


func capture_orientation(
    case_id: String, canonical_wxyz: Array, native: Quaternion, native_probe: Vector3
) -> Dictionary:
    return {
        "case_id": case_id,
        "canonical_wxyz": canonical_wxyz,
        "native_quaternion": quaternion_xyzw(native),
        "native_rotated_positive_basis_vectors_xyz": rotated_axes(native),
        "native_rotated_probe_vector_xyz": vector(native * native_probe),
    }


func capture_composition(
    left_id: String, right_id: String, product: Quaternion, native_probe: Vector3
) -> Dictionary:
    return {
        "left_case_id": left_id,
        "right_case_id": right_id,
        "product_order": "lhs * rhs; rhs acts first",
        "native_product_quaternion": quaternion_xyzw(product),
        "native_rotated_positive_basis_vectors_xyz": rotated_axes(product),
        "native_rotated_probe_vector_xyz": vector(product * native_probe),
    }


func map_canonical_quaternion(wxyz: Array) -> Quaternion:
    var source := Quaternion(wxyz[1], wxyz[2], wxyz[3], wxyz[0])
    return Quaternion(source.x, source.z, -source.y, source.w)


func map_source_vector(value: Vector3) -> Vector3:
    return Vector3(value.x, value.z, -value.y)


func quaternion_xyzw(value: Quaternion) -> Array:
    return [value.x, value.y, value.z, value.w]


func rotated_axes(quaternion: Quaternion) -> Array[Dictionary]:
    return [
        vector(quaternion * Vector3.RIGHT),
        vector(quaternion * Vector3.UP),
        vector(quaternion * Vector3.BACK),
    ]


func vector(value: Vector3) -> Dictionary:
    return {"x": value.x, "y": value.y, "z": value.z}
