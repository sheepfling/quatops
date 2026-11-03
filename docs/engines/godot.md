# Godot

This page summarizes the Godot 4.5 `Quaternion`, `Basis`, and `Node3D` APIs.
Keep the class and method in view: their Euler-order controls are not
interchangeable.

## Quaternion and basis APIs

The `Quaternion(x, y, z, w)` constructor is XYZW, while the type's identity is
documented as `Quaternion(0, 0, 0, 1)` and its named scalar property is `w`.
Use named fields or the constructor explicitly; do not assume the constructor
order is a binary serialization layout.

`Quaternion.from_euler(Vector3(x, y, z))` takes radians and always uses the
YXZ order. `Basis.from_euler` also takes radians; it can select among the six
Tait-Bryan orders. Its default order is YXZ, applying Y, then X, then Z.

`Node3D.rotation_order` controls the local rotation order and defaults to
YXZ. `Node3D.global_rotation` always uses YXZ. Do not assume changing the local
order changes the global rotation API.

`Quaternion.slerp` requires normalized endpoints and applies Godot's default
path check. `slerpni` omits that path check; use it only when the input signs
intentionally encode a different path. For ordinary keyframed motion, convert
the keys once and sample adjacent keys with `slerp`.

```gdscript
var rotation: Quaternion = engine_keys[i].slerp(engine_keys[i + 1], alpha)
```

Godot documents its coordinate system as right-handed, with the camera
convention looking along -Z. Other imported models can use different forward
axes, so account for a model/import basis separately from the world basis.

## Primary references

- [Godot 4.5 `Quaternion`](https://docs.godotengine.org/en/4.5/classes/class_quaternion.html)
- [Godot 4.5 `Basis`](https://docs.godotengine.org/en/4.5/classes/class_basis.html)
- [Godot 4.5 `Node3D`](https://docs.godotengine.org/en/4.5/classes/class_node3d.html)

For the research bundle's component-basis mapping, see
[coordinate mappings](coordinate-mapping.md). For keyframe conversion and
sampling, see the [trajectory guide](trajectory.md).
