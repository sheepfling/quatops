# Three.js 0.187.0 native capture

This is a pinned Node.js math-library runtime capture. It does not launch a
browser or renderer and does not exercise GPU arithmetic. The exporter does not
import QuatOps, SciPy, or Python packages. Its raw observations are frozen in
[`native_rotation.json`](../../../../tests/golden/oracles/threejs/0.187.0/native_rotation.json).

## Reproduce

From this directory, install exactly the lockfile and run:

```powershell
npm ci
node capture_three.mjs --corpus ..\..\shared\orientation_inputs_v1.json --output ..\..\..\..\tests\golden\oracles\threejs\0.187.0\native_rotation.json
```

The checked-in manifest records Node/V8/platform, `THREE.REVISION`, corpus,
lockfile and exporter SHA-256 values, plus the observed operation counts.
Install `three@0.187.0` from the lockfile; do not regenerate it from a floating
version range.

## Captured API surface

The exporter records all six Euler orders over the shared 89 input cases and
all 540 unit-boundary probes, Euler and matrix readback, raw column-major
`Matrix4.elements`, explicit matrix rows, vector action, axis-angle, all shared
quaternion normalization inputs, both multiplication operand orders, shortest
SLERP at the shared weights, independent proper matrices, `setFromUnitVectors`,
Euler `reorder`, and a clean unit-scale `Object3D` parent/child hierarchy.

Three.js Euler inputs are stored as axis-labelled XYZ. The capture preserves
the order string separately; comparing that string with a QuatOps/SciPy order
requires measured order semantics, not spelling by itself. Three.js
quaternions are stored as XYZW. `Matrix4.elements` is retained in its native
array order alongside row-labelled matrix values and transformed basis vectors.
The capture does not establish an asset's model-forward axis or a universal
world-frame conversion.
