# CesiumJS 1.124.0 local ENU-to-ECEF converter

## Accepted scope

The retained proof validates a QuatOps converter for explicit local right-handed
frames assigned to Cesium's east-north-up frame: canonical `+X right/east`,
`+Y forward/north`, and `+Z up` map to Cesium `+X east`, `+Y north`, and `+Z`
ellipsoid normal. Quaternion coefficients are Hamilton WXYZ in QuatOps and
active XYZW at the Cesium API boundary. The local quaternion is composed with
Cesium's native WGS84 ENU-to-ECEF transform; frame placement is not folded into
the local quaternion conversion.

The native capture ran CesiumJS `1.124.0` with Node.js `v24.21.0` on Windows
x64. It retains five selected cases covering identity, mixed rotations, an
antipode, noncommuting composition, and four distinct geographic origins.
Native outputs include the ENU-to-ECEF and
model-to-ECEF matrices, local basis/probe actions, point and direction actions,
inverse point transforms, ECEF quaternion, and recovered local quaternion.

The selected values, frame contract, and version provenance are in the
[Cesium converter fixture](../../tests/golden/transports/cesium/1.124.0/manifest.json).
The versioned [capture project](../../experiments/converters/cesium/1.124.0/README.md)
contains the pinned package, input generator, and native exporter. Replay is
covered by [test_cesium_converter.py](../../tests/experiments/converters/cesium/test_cesium_converter.py).
SciPy matrix checks are performed during replay and are reported separately
from execution in the pinned Cesium runtime.

The captured ECEF origin agrees with the independent WGS84 calculation within
`4.66e-10 m`; local point round trips are within `9.32e-10 m`. The exact-pole
case records Cesium's observed fallback for the selected input; it does not
establish a universal east-axis convention at a pole.

## Limits

This proof covers quaternion conversion and static point/direction transforms
for the declared local ENU frame and selected WGS84 origins. It does not cover
Cesium Model/glTF import, asset model-forward axes, rendering, CZML, ion
services, ICRF, or time-dependent Earth orientation.
