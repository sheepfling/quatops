# MuJoCo 3.17.0 converter capture

This auxiliary capture exercises only the direct quaternion boundary in the
pinned MuJoCo Python binding. It calls `mju_quat2Mat`, `mju_rotVecQuat`,
`mju_mat2Quat`, `mju_mulQuat`, and `mju_axisAngle2Quat` without importing
QuatOps or assigning frames to a simulation model.

From the repository root, install the pinned environment and capture the
selected witnesses:

```shell
uv sync --locked --project experiments/converters/python --extra mujoco --python 3.13.14
uv run --locked --project experiments/converters/python --extra mujoco --python 3.13.14 \
  python experiments/converters/mujoco/3.17.0/capture_native.py --repository-root .
```

The [`mujoco` optional dependency set](../../../pyproject.toml) pins
`mujoco==3.17.0` and `numpy==2.3.5`; the root QuatOps development install does
not need either native capture dependency. The Python 3.13 lock is shared
with the other Python captures while the two capture dependencies remain separate extras.

The script writes only retained converter witnesses and provenance to
[`tests/golden/oracles/mujoco/3.17.0/converter`](../../../../tests/golden/oracles/mujoco/3.17.0/converter/manifest.json).
Replay those native values with
[`test_mujoco_converter.py`](../../../../tests/experiments/converters/mujoco/test_mujoco_converter.py).
Broader Euler, integration, and derivative discovery is outside this evidence
package.
