# SpiceyPy / CSPICE 8.3.0 converter capture

This auxiliary capture exercises only the direct WXYZ quaternion boundary in
SpiceyPy 8.3.0 / CSPICE_N0067: `q2m`, `mxv`, `m2q`, `qxq`, and the native
`axisar` matrix followed by `m2q`. It does not assert an `eul2m` convention or
named SPICE frame mapping, and it imports no QuatOps code.

From the repository root, install the pinned environment and capture the
selected witnesses:

```powershell
uv sync --locked --project experiments/converters/spiceypy/8.3.0 --python 3.10.11
uv run --locked --project experiments/converters/spiceypy/8.3.0 --python 3.10.11 `
  python experiments/converters/spiceypy/8.3.0/capture_native.py --repository-root .
```

This version directory is SpiceyPy's optional native-capture dependency set.
Its locked project installs `spiceypy==8.3.0`; the root QuatOps environment
does not need this native capture dependency.

The script writes only the retained converter witnesses and run provenance to
[`tests/golden/oracles/spiceypy/8.3.0/converter`](../../../../tests/golden/oracles/spiceypy/8.3.0/converter/manifest.json).
Replay those frozen native values with
[`test_spiceypy_converter.py`](../../../../tests/experiments/converters/spiceypy/test_spiceypy_converter.py).
Euler, derivative, named-frame, and mission-kernel behavior is outside this
evidence package.
