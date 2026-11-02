# Unreal Engine

This page covers the public `FRotator` and `FQuat` conventions described by
Epic's documentation. It does not assert runtime parity for a particular
Unreal release; pin and test the version used by an integration.

## Euler API

`FRotator` stores `Pitch`, `Yaw`, and `Roll` in degrees. Epic describes its
intrinsic rotation sequence as Yaw, then Pitch, then Roll. The Blueprint
`Make Rotator` node exposes pins as X = Roll, Y = Pitch, Z = Yaw, so the pin
order differs from the C++ field listing. Do not pass an XYZ vector to either
surface without explicitly reordering it.

The attached research bundle models these values against a right-handed,
active matrix convention as
`R = Rz(yaw) Ry(-pitch) Rx(-roll)`. Treat that as the bundle's numerical
conversion model. It needs a check against the exact Unreal runtime and target
API before it is used as an engine guarantee.

## Coordinate and composition conventions

`FQuat` exposes XYZW components and takes X, Y, Z, W in its component
constructor. Treat that as API value order; do not infer a serialization or
binary memory layout from it.

Unreal documents its native coordinate system as left-handed, with X forward,
Y right, and Z up. That describes the world basis; it does not make the
quaternion algebra JPL. Keep basis conversion separate from the choice of
Hamilton or JPL representation in QuatOps.

Epic documents `FQuat` multiplication so that `A * B` applies `B` first and
then `A`. `FTransform` multiplication has a different documented ordering;
do not transfer a quaternion composition rule to transform multiplication.

`FQuat::Slerp` corrects endpoint alignment and returns a normalized result.
`SlerpFullPath` does not check for the shortest distance and is intended for
paths whose signs have been chosen deliberately. For a sampled trajectory,
convert and normalize keyframes once, then sample between adjacent keys with
the intended interpolation method.

```cpp
const FQuat rotation = FQuat::Slerp(engineKeys[i], engineKeys[i + 1], alpha);
```

## Primary references

- [TRotator API](https://dev.epicgames.com/documentation/unreal-engine/API/Runtime/Core/TRotator?lang=en-US)
- [Make Rotator Blueprint API](https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/Math/Rotator/MakeRotator)
- [Unreal coordinate system and spaces](https://dev.epicgames.com/documentation/unreal-engine/coordinate-system-and-spaces-in-unreal-engine)
- [TQuat API](https://dev.epicgames.com/documentation/unreal-engine/API/Runtime/Core/TQuat)
- [TQuat::SlerpFullPath](https://dev.epicgames.com/documentation/unreal-engine/API/Runtime/Core/TQuat/SlerpFullPath?lang=en-US)

For the research bundle's component-basis mapping, see
[coordinate mappings](coordinate-mapping.md). For keyframe conversion and
sampling, see the [trajectory guide](trajectory.md).
