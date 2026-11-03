# Three.js rotation oracle

## Captured release

The auxiliary native capture uses the lockfile-pinned `three@0.187.0` package
with Node.js `v24.21.0` on Windows x64. It records the Node/V8 versions,
`THREE.REVISION`, corpus hash, lockfile hash, and exporter hash in
[`native_rotation.json`](../../tests/golden/oracles/threejs/0.187.0/native_rotation.json).
The exporter and reproduction command are in
[`experiments/oracles/threejs/0.187.0`](../../experiments/oracles/threejs/0.187.0/README.md).
This is a JavaScript `Number` capture, not float32 engine evidence.

## Observed behavior

The discovery sweep compared the six uppercase Euler orders with SciPy across
89 general inputs and 540 boundary probes. The retained file keeps the
minimized near-lock readback discrepancy as a regression; it does not retain
the broader Euler-order sweep. Quaternions are recorded as `[x, y, z, w]`.

The retained quaternion witnesses cover `Quaternion.setFromRotationMatrix`,
`Quaternion.setFromAxisAngle`,
`Quaternion.multiply`/`premultiply` semantics, `Quaternion.slerp`,
`Quaternion.setFromUnitVectors`, and vector action. One
`Euler.setFromQuaternion` readback regression is retained separately. The public API documents those quaternion
operations and states its rotation-matrix precondition at the
[Quaternion API page](https://threejs.org/docs/pages/Quaternion.html).

Two native edges are retained as observations. First, `normalize()` maps a
zero quaternion to identity. Second, for selected inputs just beyond gimbal
lock, `Euler.setFromQuaternion`'s readback route has a matrix max-absolute
residual around `1.95e-4`; general cases in this corpus remain below
`2e-13`. The readback diagnostic remains separate from the accurate forward
Euler conversions and does not widen the QuatOps comparison tolerance.

## Limits

This proves the pinned Node math implementation only. It does not exercise a
browser, WebGL/WebGPU renderer, GPU arithmetic, imported asset-axis defaults,
or an application-specific world/body frame. The separate glTF capture retains
the native loader hierarchy and selected animation samples.
