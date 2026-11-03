# PROJ 9.5.1 geographic reference

The Wave 2 run used PROJ 9.5.1 through pyproj 3.7.2 on Python 3.13.5, Linux
x64. It records 11 WGS84 geodetic/ECEF and topocentric probes, including
oblique, high-altitude, antimeridian, near-pole, and exact-pole locations.
The raw run and manifest are in the [capture packet](../../experiments/quatops_capture_wave2/README.md)
and [versioned golden folder](../../tests/golden/oracles/proj/9.5.1/manifest.json).

Compared with an independent WGS84/ENU calculation, the maximum ECEF point
residual is `9.32e-10 m`; the largest native local-coordinate round trip is
`6.35e-10 m`. These are numerical consistency results, not geodetic accuracy
claims. PROJ's topocentric operation is documented
[here](https://proj.org/en/stable/operations/conversions/topocentric.html).

PROJ is supporting coordinate-frame evidence, not a quaternion implementation
or a Cesium runtime. The exact-pole longitude convention is kept separate
from Cesium's own pole behavior.
