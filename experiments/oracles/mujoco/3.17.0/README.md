# MuJoCo 3.17.0 capture

## Reproduction

The capture used the official `mujoco` Python package 3.17.0, its bundled
MuJoCo 3.17.0 native library, CPython 3.10.11, and NumPy 2.2.6 on Windows
x86-64. It calls only low-level `mju_`/`mjd_` routines; no model, simulator,
renderer, or QuatOps code is loaded.

```powershell
uv sync --locked --python 3.10.11
uv run --locked python capture_mujoco.py
```

The frozen output is
[`native_rotation.json`](../../../../tests/golden/oracles/mujoco/3.17.0/native_rotation.json).
It includes hashes for the lockfile, exporter, loaded package binaries, and
input corpus.

## Captured surfaces

- `mju_euler2Quat`: six mixed-axis orders and the frozen Euler boundary inputs.
  Lowercase sequences matched SciPy uppercase intrinsic orders; uppercase
  sequences matched SciPy lowercase extrinsic orders in this corpus.
- `mju_mulQuat`, `mju_negQuat`, axis-angle, vector action, quaternion/matrix
  conversion, and both multiplication operand orders.
- `mju_subQuat`, `mju_quat2Vel`, `mju_derivQuat`, and
  `mju_quatIntegrate` on nonidentity orientation inputs. The relative-rotation
  velocity from `mju_quat2Vel` recovers the rate used by the local integrator.
- `mjd_subQuat` and `mjd_quatIntegrate` tangent derivative outputs, checked
  against finite differences in the explicitly recorded perturbation chart.

MuJoCo stores quaternion coefficients as WXYZ. `mju_quat2Mat` writes nine
row-major values and its action agrees with the matrix acting on column vectors
in the capture. `mju_negQuat` conjugates the vector part; it does not negate all
four coefficients. The two angular-velocity routines carry different frames:
`mju_quatIntegrate` right-multiplies a local/body increment, while
`mju_derivQuat` computes the derivative for a world-expressed angular
velocity. Mapping a local rate through the captured quaternion matrix makes
the differential and integrated actions agree.

`mjd_quatIntegrate`'s three returned arrays are kept with their raw storage and
tested independently. In this runtime, `Dvel` differentiates with respect to
the scaled increment (`scale * vel`); `Dscale` is `Dvel @ vel`. This observed
contract is labelled explicitly and is not generalized to another MuJoCo
version.

## Limits

This is one CPython/Windows/x86-64 double-precision binding and one MuJoCo
library build. The Euler, integration, and Jacobian profiles do not prove
simulation timestep accuracy, a universal world/body frame convention, a
single-precision build, or behavior of higher-level robotics wrappers.
