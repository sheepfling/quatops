# SpiceyPy / CSPICE quaternion converter evidence

The accepted proof covers the direct quaternion boundary in SpiceyPy 8.3.0 /
CSPICE_N0067: WXYZ storage, `q2m`, `mxv`, `m2q`, `qxq`, and the native
`axisar` matrix followed by `m2q`. It assigns an explicit right-handed probe
frame (`+X right, +Y forward, +Z up`) to these numeric APIs. It makes no claim
about named SPICE frames, aerospace defaults, or spacecraft attitude models.

The [native witness set](../../tests/golden/oracles/spiceypy/8.3.0/converter/witnesses.json)
stores selected quaternion values and native basis actions. The
[versioned contract](../../experiments/converters/spiceypy/8.3.0/semantic_contract.json)
and [manifest](../../tests/golden/oracles/spiceypy/8.3.0/converter/manifest.json)
record scope, input identity, CSPICE binary, and capture provenance. Replay is
implemented in
[`test_spiceypy_converter.py`](../../tests/experiments/converters/spiceypy/test_spiceypy_converter.py);
it does not launch CSPICE.

The capture and pinned dependencies are under
[`experiments/converters/spiceypy/8.3.0`](../../experiments/converters/spiceypy/8.3.0/README.md).
A standalone quaternion-to-matrix-to-quaternion round trip, Euler/frame
direction, derivatives, mission kernels, and named reference frames are outside
this converter claim. See NAIF's references for
[`q2m_c`](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/q2m_c.html),
[`qxq_c`](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/qxq_c.html),
and [`axisar_c`](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/axisar_c.html).
