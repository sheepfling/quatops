# OpenUSD converter evidence

The OpenUSD 26.11 converter proof serializes explicit canonical orientations
and translations into a USDA hierarchy, reopens it with the pinned
`usd-core==26.11` runtime, and records native `UsdGeom`/`Gf` evaluation. The
capture runbook, lock, and scripts are in
[`experiments/converters/openusd/26.11`](../../experiments/converters/openusd/26.11/README.md).

The contract specifies a right-handed source frame (+X right, +Y forward, +Z
up) and a right-handed Y-up stage frame (+X right, +Y up, -Z forward). The
converter uses the explicit destination-axis map `(+X, +Z, -Y)`, Hamilton
active quaternions, and real-first `quatd` WXYZ values. It tests a parent, a
composed child, and a child with `resetXformStack`, including native local and
world row-vector matrices, point/direction actions, and inverse point
recovery. The stage up-axis metadata does not infer or change the source frame.

The selected native values and exact runtime provenance are in the
[converter fixture](../../tests/golden/transports/openusd/26.11/converter/manifest.json);
the replay assertions are in
[`test_openusd_converter_roundtrip.py`](../../tests/experiments/converters/openusd/test_openusd_converter_roundtrip.py).
This evidence covers the authored orient/translate stack only. It does not
claim time-sampled animation, scale/shear decomposition, renderer execution,
or application-specific import behavior. Primary API references:
[`UsdGeomXformable`](https://openusd.org/release/api/class_usd_geom_xformable.html)
and [`GfQuatd`](https://openusd.org/release/api/class_gf_quatd.html).
