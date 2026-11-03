# OpenUSD rotation and transform-stack transport

## Runtime and evidence

The pinned run used [`usd-core==26.11`](https://pypi.org/project/usd-core/26.11/)
with OpenUSD 26.11 and CPython 3.14.6 on Windows x86-64. The saved USDA input
stage, lock, exporter, binary and corpus hashes are in the
[frozen runtime capture](../../tests/golden/transports/openusd/26.11/native_rotation.json).
Reproduction instructions and the versioned stage are under
[`experiments/transports/openusd/26.11`](../../experiments/transports/openusd/26.11/README.md).

The exporter executed `Gf` and `UsdGeom` APIs in the pinned runtime. Fixture
tests then check the recorded results with SciPy rotations and explicit
row-vector matrix products; those tests read the fixture and do not start
OpenUSD. Primary API references: [`GfQuatf`](https://openusd.org/release/api/class_gf_quatf.html),
[`GfQuatd`](https://openusd.org/release/api/class_gf_quatd.html),
[`GfRotation`](https://openusd.org/release/api/class_gf_rotation.html),
[`GfSlerp`](https://openusd.org/release/api/quatd_8h.html), and
[`UsdGeomXformable`](https://openusd.org/release/api/class_usd_geom_xformable.html).

## Measured behavior

`Gf.Quatf` and `Gf.Quatd` store real then three imaginary coefficients. The
capture exercises each precision separately for construction, product,
normalization, conjugate, inverse and vector action. A zero quaternion's
normalization falls back to identity; its inverse/action outputs are nonfinite
in this runtime and are serialized as the explicit string `"NaN"`, keeping the
fixture valid JSON. In the `Quatf` tiny-input probe, normalization also falls
back to identity because its length underflows to zero, while inverse and
vector-action outputs become signed infinities or `NaN`. The corresponding
double-precision `Quatd` input remains finite. These are captured precision
specific boundaries, not shared `Quatf`/`Quatd` math claims.

`Gf.Rotation` takes and returns angle values in degrees. The captured
axis-angle route preserves the sign and full-turn magnitude for the tested
values. `Gf.Rotation.TransformDir` and the stage's `Gf.Matrix4d` actions are
row-vector operations, as the API documents.

The stage declares `upAxis = "Y"` but its native world translations for the
authored +X, +Y, and +Z probes remain those same coordinate vectors. This
shows that the stage metadata did not remap the authored values in this test;
it does not select the source frame or an asset's model-forward axis. In the
captured stack, `xformOpOrder` is preserved, a child local matrix followed by
its parent matrix gives the recorded world matrix, and `resetXformStack`
leaves the reset child world matrix equal to its local matrix.

`Gf.Slerp` was checked on the frozen interpolation pairs and weights against a
separate shortest-path SLERP formula. The stage stores and evaluates authored
orient, rotate, and translate ops; it does not represent a file importer or a
general asset pipeline.

## Limits

This establishes the tested Python distribution and build on one platform. It
does not cover renderer behavior, other importers, time-sampled interpolation
of xform ops, nonuniform-scale decomposition, shear, or a model-forward
convention. The native OpenUSD capture, its stage asset, and SciPy comparisons
remain separate evidence types.

## QuatOps stage converter round trip

A separate proof writes a compact USDA stage from explicit canonical
`+X right, +Y forward, +Z up` inputs using the declared stage map `(+X, +Z,
-Y)`. The file is then reopened with the pinned `usd-core==26.11` runtime and
evaluated through `UsdGeom`/`Gf` APIs. The stage has a parent, a composed
child, and a child with `resetXformStack`; its metadata explicitly says Y-up
and meters-per-unit 1.

The three selected native observations retain orient-op values, ordered xform
ops, local and world row-vector matrices, point/direction actions, inverse
point transforms, and the reset-stack result. The USDA, native outputs, exact
runtime identity, and script/input/lock hashes are in the versioned
[converter fixture](../../tests/golden/transports/openusd/26.11/converter/manifest.json).
Replay is [test_openusd_converter_roundtrip.py](../../tests/test_openusd_converter_roundtrip.py).
This establishes only the authored orient/translate stack behavior exercised
by this stage, not scale/shear decomposition, time samples, or renderer/importer
behavior.
