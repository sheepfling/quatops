# PROJ 9.5.1 geographic frame reference

PROJ is retained as supporting evidence for geodetic, Earth-centered Earth-fixed
(ECEF), and local east-north-up (ENU) coordinate transforms. It is not a
quaternion implementation or a Cesium runtime.

The selected native baseline contains 11 WGS84 locations covering ordinary,
oblique, high-altitude, antimeridian, near-pole, and exact-pole branches. The
original capture ran PROJ `9.5.1` through pyproj `3.7.2` on Linux x64 with
Python `3.13.5`. A separate Windows 11 execution used the same pyproj and PROJ
versions with Python `3.13.14`; all case IDs matched and the largest absolute
difference across origins, ECEF points, and local readbacks was
`9.32e-10 m`. The frozen observations and native-run identity are in the
[versioned fixture](../../tests/golden/oracles/proj/9.5.1/manifest.json); the
cross-platform summary is in
[`audit.json`](../../tests/golden/oracles/proj/9.5.1/audit.json).

For the 11 locations, native ECEF images of the ENU origin and three unit
directions were compared with the analytic local ENU basis. The maximum
component residual was `3.97e-10`; the analytic matrix orthogonality and
determinant residuals were `2.22e-16` and `4.44e-16`. A separate SciPy
`Rotation` matrix check reproduced the analytic basis within `2.22e-16`.
Subtracting Earth-scale coordinates limits this basis comparison, so these
are frame-consistency results, not a geodetic accuracy claim or a QuatOps
execution.

The versioned [capture package](../../experiments/oracles/proj/9.5.1/README.md)
retains the geographic inputs, pyproj requirement, and exporter for rerunning
the target capture. The native observation values remain frozen even though
the original broad discovery output is not shipped. Each retained location
stores the native ECEF origin, three ENU axis endpoints, one transformed mixed
point, and that point's local inverse readback. Location inputs resolve by
case ID from the shared geographic corpus. Redundant inputs and readbacks were
removed after checking their values; retained decimal values round-trip to the
original binary64 patterns, so duplicate per-value hex arrays are omitted.
PROJ's topocentric
operation is described in the
[PROJ topocentric conversion reference](https://proj.org/en/stable/operations/conversions/topocentric.html).
The exact-pole longitude convention remains distinct from Cesium's own
observed pole behavior.
