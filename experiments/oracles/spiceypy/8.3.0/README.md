# SpiceyPy 8.3.0 / CSPICE_N0067 capture

This lane pins the Python wrapper and the CSPICE DLL separately. It uses only
pure rotation routines and loads no mission kernels.

## Reproduce

From this directory:

```powershell
uv sync --locked --python 3.10.11
uv run --locked python capture_spice.py --corpus ..\..\shared\orientation_inputs_v1.json --output ..\..\..\..\tests\golden\oracles\spiceypy\8.3.0\native_rotation.json
```

The frozen manifest records Python, SpiceyPy, `tkvrsn('TOOLKIT')`, the loaded
`cspice.dll` path/hash, the lockfile and exporter hashes, and the input corpus
hash. It captures 534 six-order Euler cases, 540 boundary cases, 60 axis-angle
cases, 3 products, 8 proper matrices and 48 quaternion-derivative probes.

The exporter calls CSPICE only through SpiceyPy. Its small Hamilton product
formula is included solely to assemble derivative test inputs and check
`qxq`; that formula does not produce any native fixture output.
