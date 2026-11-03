# Three.js 0.187.0 native capture

This is a pinned Node.js math-library runtime capture. It does not launch a
browser or renderer and does not exercise GPU arithmetic. The exporter does not
import QuatOps, SciPy, or Python packages. Its raw observations are frozen in
[`native_rotation.json`](../../../../tests/golden/oracles/threejs/0.187.0/native_rotation.json).

## Reproduce

From this directory, install exactly the lockfile and run:

```powershell
npm ci
```

Keep the full discovery result in a temporary file, then select retained
witnesses into the repository. The selector reads the raw capture and writes a
separate output; it never edits its input:

```powershell
$repositoryRoot = git rev-parse --show-toplevel
$discovery = Join-Path $env:TEMP 'quatops-threejs-0.187.0-discovery.json'
node capture_three.mjs --corpus ..\..\shared\orientation_inputs_v1.json --output $discovery
Push-Location $repositoryRoot
$selectorArgs = @(
  'threejs',
  $discovery,
  'tests\golden\oracles\threejs\0.187.0\native_rotation.json',
  '--overwrite'
)
.venv\Scripts\python.exe scripts\compact_oracle_witnesses.py @selectorArgs
Pop-Location
```

The selector matches case IDs and checks that required witnesses are present
exactly once. It stores the raw capture hash in the retained fixture provenance.

The checked-in manifest records Node/V8/platform, `THREE.REVISION`, corpus,
lockfile and exporter SHA-256 values, plus the observed operation counts.
Install `three@0.187.0` from the lockfile; do not regenerate it from a floating
version range.

## Captured API surface

The discovery exporter records all six Euler orders over 89 inputs and 540
unit-boundary probes, matrix readback, vector action, axis-angle, normalization,
multiplication, SLERP, matrix constructors, `setFromUnitVectors`, Euler
`reorder`, and an `Object3D` hierarchy. Retained witnesses keep selected
quaternion API cases plus one minimized Euler readback regression. They omit
the broader Euler sweep and duplicate hierarchy, which is covered by the glTF
loader proof.

Three.js Euler inputs are stored as axis-labelled XYZ. The capture preserves
the order string separately; comparing that string with a QuatOps/SciPy order
requires measured order semantics, not spelling by itself. Three.js
quaternions are stored as XYZW. Retained matrices use row-labelled values. The
capture does not establish an asset's model-forward axis or a universal
world-frame conversion.
