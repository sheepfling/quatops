# MuJoCo quaternion converter evidence

The accepted proof covers MuJoCo 3.17.0's direct quaternion boundary through
the pinned Python binding: WXYZ construction, `mju_quat2Mat`,
`mju_rotVecQuat`, `mju_mat2Quat`, `mju_mulQuat`, and `mju_axisAngle2Quat`.
The explicit probe frame is right-handed `+X right, +Y forward, +Z up`; this
does not assign axes to a MuJoCo model or simulator.

The [native witness set](../../tests/golden/oracles/mujoco/3.17.0/converter/witnesses.json)
stores selected quaternion values and native basis actions. The
[versioned contract](../../experiments/converters/mujoco/3.17.0/semantic_contract.json)
and [manifest](../../tests/golden/oracles/mujoco/3.17.0/converter/manifest.json)
record scope, input identity, native binary, and capture provenance. Replay is
implemented in
[`test_mujoco_converter.py`](../../tests/experiments/converters/mujoco/test_mujoco_converter.py);
it does not launch MuJoCo.

The capture and pinned dependencies are under
[`experiments/converters/mujoco/3.17.0`](../../experiments/converters/mujoco/3.17.0/README.md).
Euler order, integration, derivatives, simulation models, and rendering are
outside this converter claim. See the official
[MuJoCo 3.17 API reference](https://mujoco.readthedocs.io/en/3.17.0/APIreference/APIfunctions.html)
for the native routine definitions.
