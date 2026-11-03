# MuJoCo quaternion oracle

## Runtime and evidence

The native capture used `mujoco==3.17.0` and MuJoCo 3.17.0 through the pinned
Windows x86-64 wheel, CPython 3.10.11, and NumPy 2.2.6. The binding accepts
`float64` arrays. Binary hashes, lock hash, capture script hash and corpus hash
are in the [frozen capture](../../tests/golden/oracles/mujoco/3.17.0/native_rotation.json);
the exporter and exact invocation are under
[`experiments/oracles/mujoco/3.17.0`](../../experiments/oracles/mujoco/3.17.0/README.md).
The exporter calls the native routines directly and imports no QuatOps code.

The fixture tests separately compare those frozen outputs with SciPy matrices,
explicit Hamilton products, and finite differences. They do not launch the
MuJoCo runtime. The official [MuJoCo 3.13 function reference](https://mujoco.readthedocs.io/en/3.13.0/APIreference/APIfunctions.html)
describes the relevant routine contracts; pinned 3.17 source is linked for
[`engine_util_spatial.c`](https://github.com/google-deepmind/mujoco/blob/3.17.0/src/engine/engine_util_spatial.c)
and [`engine_derivative.c`](https://github.com/google-deepmind/mujoco/blob/3.17.0/src/engine/engine_derivative.c).

## Measured conventions

The source used for formula comparisons is an explicit right-handed XYZ frame.
MuJoCo stores WXYZ coefficients. The measured Hamilton product has the
positive vector cross term; `mju_negQuat` keeps W and negates XYZ, so it is the
conjugate. `mju_quat2Mat` writes nine row-major coefficients and the resulting
matrix acts on column vectors like SciPy's active matrix for unit quaternions.
The native matrix routine does not normalize arbitrary algebra inputs, so its
nonunit matrix outputs are compared with the raw polynomial rather than a
rotation-only SciPy conversion.

For the pinned `mju_euler2Quat`, lowercase sequence labels matched SciPy's
uppercase intrinsic orders, and uppercase labels matched SciPy's lowercase
extrinsic orders. The six noncommuting orders and lower-case boundary sweeps
are captured independently. The result is specific to the tested API and
argument mapping; do not copy a sequence label between libraries without a
mapping.

`mju_quatIntegrate(q, omega_local, h)` normalizes `q` and right-multiplies the
increment `Exp(h * omega_local)`. Its velocity is local/body expressed.
`mju_derivQuat(q, omega_world)` evaluates
`0.5 * [0, omega_world] * q`, for a world-expressed rate. Mapping the local
integration rate through the captured active matrix `R(q)` makes this
derivative agree with finite differences of integration. `mju_subQuat(qa,
qb)` returns the rotation vector for `conjugate(qb) * qa`; `mju_quat2Vel`
returns the captured relative rotation vector divided by its time scale.

The `mjd_subQuat` 3x3 outputs match right-local tangent perturbations of
`qa` and `qb`. The `mjd_quatIntegrate` `Dquat` output matches a right-local
perturbation of the starting quaternion. In the pinned binary, `Dvel` matches
the derivative with respect to the scaled increment `h * omega`; `Dscale`
matches `Dvel @ omega`. The frozen outputs and finite-difference checks keep
these tangent dimensions separate from a four-coefficient quaternion
derivative.

## Limits

This is one double-precision Python binding and binary on one platform. It does
not cover single-precision MuJoCo builds, viewer or simulator behavior,
robot-specific joint frames, or all robotics wrapper conventions. The native
capture and its SciPy/formula replay are separate evidence types.

## QuatOps direct-quaternion boundary proof

The compact converter capture at
[`tests/golden/oracles/mujoco/3.17.0/converter`](../../tests/golden/oracles/mujoco/3.17.0/converter/manifest.json)
assigns MuJoCo's numeric vectors the same explicit right-handed probe frame as
the canonical QuatOps inputs (`+X right, +Y forward, +Z up`). This assignment
does not infer a model or simulator world frame. In MuJoCo 3.17.0, native
`mju_quat2Mat` and `mju_rotVecQuat` matched the active Hamilton WXYZ action on
eight selected unit inputs; `mju_mat2Quat` reversed the matrix route up to
quaternion sign. A native `mju_axisAngle2Quat` orientation and both operand
orders of a noncommuting `mju_mulQuat` pair were also checked. The selected
native values are replayed by
[`tests/experiments/converters/mujoco/test_mujoco_converter.py`](../../tests/experiments/converters/mujoco/test_mujoco_converter.py),
separately from the pinned native capture.

This does not establish model body, joint, camera, or robot-link frames, nor
simulator, viewer, or rendering behavior.
