# CesiumJS 1.124.0 rotation oracle

## Runtime capture

The capture uses `cesium@1.124.0` with Node.js `v24.21.0` on Windows x64. The
native calls ran without a browser, renderer, WebGL context, or GPU. The pinned
package lock and resolved dependency graph are retained with the capture
project. The exporter, raw 171-observation output, and capture manifest are
linked from [the versioned fixture](../../tests/golden/oracles/cesium/1.124.0/manifest.json)
and [capture project](../../experiments/quatops_capture_wave2/LOCAL_REPLAY.md).

The manifest records the runtime string `1.124`; npm's lock records the exact
package version `1.124.0`. It also records the Cesium engine/widgets and
`@zip.js/zip.js` versions required by that release. These dependency pins are
part of this capture's identity.

## Local heading-pitch-roll

Cesium documents heading about negative Z, pitch about negative Y, and roll
about positive X, in radians ([Cesium 1.124 HeadingPitchRoll](https://cesium.com/downloads/cesiumjs/releases/1.124/Build/Documentation/HeadingPitchRoll.html)).
Across 89 cases, `Quaternion.fromHeadingPitchRoll`, matrix construction,
quaternion readback, and vector action agree with an independent SciPy
evaluation of `Rz(-heading) Ry(-pitch) Rx(roll)`. The largest matrix residual
was `9.77e-15`; the direct forward matrix residual was below `5e-16`.

The same run also covers eight proper-matrix-to-quaternion cases, six native
Hamilton products, and 35 shortest-path SLERP samples. The product coefficients
matched an explicit Hamilton formula exactly for this corpus; SLERP's maximum
orientation-matrix residual was `3.34e-16`.

## Geographic placement

For 11 WGS84 locations and three HPR inputs per location, the capture keeps the
local HPR quaternion separate from the ENU-to-ECEF frame and the final
model-to-ECEF transform. It records origins, matrices, quaternions, points, and
directions independently. Cesium documents `eastNorthUpToFixedFrame` as local
east/north plus the ellipsoid surface normal, and uses that frame by default
for `headingPitchRollToFixedFrame` and
`headingPitchRollQuaternion` ([Cesium 1.124 Transforms](https://cesium.com/downloads/cesiumjs/releases/1.124/Build/Documentation/Transforms.html)).

The ECEF origins agree with the separate WGS84 formula within `4.66e-10 m`.
The captured ENU up axis matches the radius-scaled Cartesian normal within
`1.12e-16`. At the 400 km altitude case, the resulting ENU matrix differs from
one computed only from the input geodetic latitude by `1.40e-4` in its largest
element. That difference is retained as a Cesium frame observation; the tests
do not substitute a latitude-only basis. Cesium exposes both Cartesian and
cartographic geodetic-normal routes on its [Ellipsoid API](https://cesium.com/downloads/cesiumjs/releases/1.124/Build/Documentation/Ellipsoid.html).

The model-to-ECEF matrix and quaternion agree within `4.45e-16`; the point
round-trip residual is at most `9.32e-10 m`. The exact-pole probes record
Cesium's native fallback at the supplied zero-longitude inputs. They do not
establish a universal east-axis convention at a pole.

## Limits

This does not capture Cesium Model/glTF import, asset model-forward axes,
rendering, CZML, ion services, ICRF, or time-dependent Earth-orientation data.
The captured model frame is the explicit local ENU frame in the input corpus.
SciPy checks are separately replayed from the pinned Cesium runtime output; no
QuatOps conversion adapter was exercised in this capture.
