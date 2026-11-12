# Three.js object consumer

The runnable [consumer example](../../examples/converters/threejs_consumer.py)
uses Three.js **0.187.0**, from the existing pinned glTF experiment dependency.
It declares its source as right-handed right/forward/up and independently
declares this application's scene/model as right/up/back, with model-forward
along -Z. The `threejs.quaternion` profile supplies the coefficient boundary;
these axes and authored marker directions come from the example.

## Native boundary

The probe assigns XYZW coefficients through `Object3D.quaternion.fromArray`,
adds distinct forward/up child markers, updates world matrices, and reads their
world positions. Subtracting the object's world origin gives observable
directions. The object has unit scale and no transformed ancestor. The script
compares these native results with the independently declared source-to-scene
vector map.

The ordinary JavaScript Number assignment and an explicit `Float32Array` cast
are checked separately. The cast illustrates a consumer storage boundary; the
profile's Python floats do not perform that cast implicitly. The example uses
`2e-14` direction tolerance for ordinary assignment and `3e-7` for the explicit
float32 route. It does not normalize either native assignment silently.

Primary API references:

- [Object3D](https://threejs.org/docs/pages/Object3D.html): local quaternion,
  world-matrix updates, and world-position operations.
- [Quaternion](https://threejs.org/docs/pages/Quaternion.html): XYZW components
  and unit-quaternion expectation.
- [Pinned r187 Object3D source](https://github.com/mrdoob/three.js/blob/r187/src/core/Object3D.js).
- [Pinned r187 Quaternion source](https://github.com/mrdoob/three.js/blob/r187/src/math/Quaternion.js).

## Verification scope

Run `python examples/converters/threejs_consumer.py` after installing the
documented development and pinned npm dependencies. It checks an actual native
object action, explicit float32 assignment, and a deliberately incorrect scene
frame rejected by the reverse gateway. It requires Node.js and runs without a
browser or renderer. This live probe is separate from retained capture replay
and from SciPy formula tests. It does not prove arbitrary scene decomposition,
nonuniform or mirrored parent handling, animated assets, imported geometry's
front, or physical sensor mounting.
