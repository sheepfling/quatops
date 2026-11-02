# Unity

Unity exposes more than one Euler API. Keep `UnityEngine.Quaternion` separate
from `Unity.Mathematics.quaternion`; their units and order controls differ.

## `UnityEngine.Quaternion`

The constructor and named components use XYZW order.

`Quaternion.Euler(x, y, z)` accepts degrees. Unity documents the application
order as Z, then X, then Y. The attached research bundle represents the result
in a right-handed active matrix model as
`R = Ry(y) Rx(x) Rz(z)`. This is the bundle's conversion model and should be
checked against the target Unity runtime before becoming an adapter guarantee.

`Quaternion.Slerp(a, b, t)` expects unit endpoints and clamps `t` to `[0, 1]`;
`SlerpUnclamped` leaves `t` unclamped. Keep trajectory sampling weights in
`[0, 1]` when using `Slerp`, and use the unclamped method only when extrapolation
is intended.

```csharp
Quaternion rotation = Quaternion.Slerp(engineKeys[i], engineKeys[i + 1], alpha);
```

## `Unity.Mathematics.quaternion`

The Mathematics package provides sequence-specific constructors such as
`quaternion.EulerXYZ`. These accept radians and name the order explicitly;
`EulerXYZ` applies X, then Y, then Z. This API is distinct from
`UnityEngine.Quaternion.Euler`, which uses degrees and its documented ZXY
order.

When writing an adapter, name the API and sequence in the function rather
than accepting an ambiguous `euler` triple. The same three numbers sent to
the two Unity APIs generally represent different rotations.

## Primary references

- [Unity 6.0 `Quaternion.Euler`](https://docs.unity.com/en-us/engine/6000.0/script-reference/unityengine/quaternion/euler)
- [Unity 6.0 `Quaternion.Slerp`](https://docs.unity.com/en-us/engine/6000.0/script-reference/unityengine/quaternion/slerp)
- [Unity 6.0 `Quaternion.SlerpUnclamped`](https://docs.unity.com/en-us/engine/6000.0/script-reference/unityengine/quaternion/slerpunclamped)
- [Unity.Mathematics 1.3.2 `quaternion.EulerXYZ`](https://docs.unity3d.com/Packages/com.unity.mathematics@1.3/api/Unity.Mathematics.quaternion.EulerXYZ.html)

For the research bundle's component-basis mapping, see
[coordinate mappings](coordinate-mapping.md). For keyframe conversion and
sampling, see the [trajectory guide](trajectory.md).
