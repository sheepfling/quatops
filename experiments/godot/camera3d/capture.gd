extends SceneTree

func _initialize() -> void:
    call_deferred("capture")

func vector(v: Vector3) -> Array:
    return [v.x, v.y, v.z]

func axes(b: Basis) -> Dictionary:
    return {"right": vector(b * Vector3.RIGHT),
        "up": vector(b * Vector3.UP), "forward": vector(b * Vector3.FORWARD)}

func capture() -> void:
    var cases: Array = []
    var rotations := [Quaternion.IDENTITY,
        Quaternion(0.2, 0.4, 0.4, 0.8),
        Quaternion(-0.3, 0.6164414002968976, -0.2, 0.7)]
    for pair in [[0, 0], [1, 2], [2, 1]]:
        var parent := Node3D.new()
        var camera := Camera3D.new()
        root.add_child(parent)
        parent.add_child(camera)
        camera.current = true
        parent.quaternion = rotations[pair[0]]
        parent.position = Vector3(1.25, -2.5, 4.0)
        camera.quaternion = rotations[pair[1]]
        camera.position = Vector3(-0.5, 1.75, 2.25)
        var pq: Quaternion = rotations[pair[0]]
        var cq: Quaternion = rotations[pair[1]]
        cases.append({"id": "%d_%d" % pair,
            "parent_xyzw": [pq.x, pq.y, pq.z, pq.w],
            "camera_xyzw": [cq.x, cq.y, cq.z, cq.w],
            "local_axes": axes(camera.transform.basis),
            "global_axes": axes(camera.global_transform.basis),
            "camera_axes": axes(camera.get_camera_transform().basis),
            "center_ray": vector(camera.project_ray_normal(Vector2(root.size) * 0.5)),
            "global_origin": vector(camera.global_position),
            "camera_origin": vector(camera.get_camera_transform().origin)})
        parent.free()
    var version := Engine.get_version_info()
    var result := {"schema_version": 1, "target": "Godot Camera3D",
        "version": version.string, "build_hash": version.hash,
        "platform": OS.get_name(), "cases": cases}
    var args := OS.get_cmdline_user_args()
    assert(args.size() == 1, "Pass an output filename after --")
    var file := FileAccess.open(args[0], FileAccess.WRITE)
    assert(file != null, "Cannot open capture output")
    file.store_string(JSON.stringify(result, "  ") + "\n")
    file.close()
    print("Captured Camera3D transforms: ", version.string)
    quit()
