# SpiceyPy / CSPICE 8.3.0 converter-boundary capture

This auxiliary proof captures direct WXYZ quaternion operations from the
pinned SpiceyPy 8.3.0 / CSPICE_N0067 environment. It is independent of
`src/quatops` and is limited to `q2m`, `mxv`, `m2q`, `qxq`, and the native
`axisar` matrix/`m2q` route. It does not assert an `eul2m` convention or named
SPICE frame mapping.

Run from the repository root with:

```powershell
experiments\oracles\spiceypy\8.3.0\.venv\Scripts\python.exe experiments\converters\spiceypy\8.3.0\capture_native.py
```

The capture writes only the selected witnesses and their provenance to
`tests/golden/oracles/spiceypy/8.3.0/converter/`. The existing pinned
environment and lockfile are under `experiments/oracles/spiceypy/8.3.0/`.
Replay the frozen native values with `tests/experiments/converters/spiceypy/test_spiceypy_converter.py`.
