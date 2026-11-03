# CSPICE rotation oracle through SpiceyPy

## Pinned runtime

The native run used SpiceyPy `8.3.0`, CSPICE toolkit `CSPICE_N0067`, Python
`3.10.11`, and the bundled Windows `cspice.dll` whose path and SHA-256 are in
[`native_rotation.json`](../../tests/golden/oracles/spiceypy/8.3.0/native_rotation.json).
The exact environment and invocation are in
[`experiments/oracles/spiceypy/8.3.0`](../../experiments/oracles/spiceypy/8.3.0/README.md).
These rotation routines required no mission kernels.

## Measured behavior

`qxq` produced scalar-first WXYZ coefficients matching the Hamilton product
and matrix composition order `M(a) @ M(b)`. NAIF documents the SPICE product
as scalar times scalar minus the vector dot product, with the positive vector
cross term, and states that CSPICE interfaces use SPICE-style quaternions
([`qxq_c`](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/qxq_c.html)).
The raw nonunit product pair is retained independently from rotation-matrix
claims.

For the captured positional mapping of `eul2m(angle3, angle2, angle1,
axis3, axis2, axis1)`, all six observed matrix results equal the transpose of
SciPy's uppercase intrinsic active matrix on these inputs. `m2eul` reconstructs
those matrices. `m2q` followed by `q2m` returns the same matrix; to compare its
quaternion to QuatOps' active Euler quaternion, the comparison explicitly
conjugates the captured value. The argument mapping and measured result are
kept separate from a general frame-direction claim. See NAIF's
[`eul2m_c`](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/eul2m_c.html)
and [rotation required reading](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/rotation.html).

`axisar` is a different native route: its matrix matched SciPy's active
axis-angle matrix in this corpus, and `m2q` returned a scalar-first quaternion
for that matrix. The fixture records the matrix output from `axisar` directly;
it does not mislabel that call as a quaternion constructor. NAIF describes
`axisar_c` as producing a matrix that rotates a vector by the requested angle
about the axis ([`axisar_c`](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/axisar_c.html)).

For `qdq2av`, the captured inputs distinguish derivative multiplication side.
With `qdot = 0.5 * q * [0, omega]`, the native output is `-omega`. With
`qdot = 0.5 * [0, omega] * q`, the output is `-R(q)^T * omega` for the
captured active Hamilton matrix `R(q)`. NAIF defines the routine as the vector
part of `-2 * conjugate(q) * dq` for the angular velocity of the object whose
attitude is represented by `q` ([`qdq2av_c`](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/qdq2av_c.html)).
The frame expression is tested directly; it is not inferred from an aerospace
label.

## Limits

This is evidence for the pinned SpiceyPy/CSPICE binary and the tested
operation profiles. It does not select a universal aerospace convention,
frame direction, units policy for all consumers, kernel setup, or spacecraft
attitude model. Keep wrapper and CSPICE version lineage together when adding a
new release.
