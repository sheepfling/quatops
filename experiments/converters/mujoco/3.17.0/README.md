# MuJoCo 3.17.0 converter capture

This auxiliary capture exercises only the direct quaternion boundary in the
pinned MuJoCo Python binding. It calls `mju_quat2Mat`, `mju_rotVecQuat`,
`mju_mat2Quat`, `mju_mulQuat`, and `mju_axisAngle2Quat` without importing
QuatOps or assigning frames to a simulation model.

From the repository root, install the pinned environment and capture the
selected witnesses:

```powershell
uv sync --locked --project experiments/converters/mujoco/3.17.0 --python 3.10.11
uv run --locked --project experiments/converters/mujoco/3.17.0 --python 3.10.11 `
  python experiments/converters/mujoco/3.17.0/capture_native.py
```

The script writes only the retained converter witnesses and run provenance to
`tests/golden/oracles/mujoco/3.17.0/converter/`. Replay those frozen native
values with `tests/experiments/converters/mujoco/test_mujoco_converter.py`.
Broader Euler, integration, and derivative discovery is outside this evidence
package.
