# MuJoCo 3.17.0 converter-boundary capture

This auxiliary proof captures direct WXYZ quaternion operations from the
pinned MuJoCo 3.17.0 Python binding. It is independent of `src/quatops` and is
limited to `mju_quat2Mat`, `mju_rotVecQuat`, `mju_mat2Quat`, `mju_mulQuat`, and
`mju_axisAngle2Quat`. It does not assign frames to a simulation model.

Run from the repository root with:

```powershell
experiments\oracles\mujoco\3.17.0\.venv\Scripts\python.exe experiments\converters\mujoco\3.17.0\capture_native.py
```

The capture writes only the selected witnesses and their provenance to
`tests/golden/oracles/mujoco/3.17.0/converter/`. The existing pinned
environment and lockfile are under `experiments/oracles/mujoco/3.17.0/`.
Replay the frozen native values with `tests/test_mujoco_converter.py`.
