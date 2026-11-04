# Sophus 1.24.6 SO(3) discovery

This first native discovery probe exercises the fixed release tag with Eigen
5.0.1 and a C++17 compiler. It captures quaternion construction and storage,
native vector action, matrix output, `SO3::exp`/`log`, inverse, both orders of
noncommuting composition, and left/right perturbation products. The fixed input
set is in `capture_inputs.tsv`. For these probes, the assigned right-handed
frame is +X right, +Y forward, +Z up; this does not claim Sophus assigns a
universal world frame. The harness does not include QuatOps.

This is discovery tooling, not an accepted converter or complete Sophus
contract. In particular, the exact tangent perturbation side, near-π
logarithm branch, and behavior across supported scalar profiles remain to be
reviewed. The capture currently covers `SO3d` only. Raw output and SciPy
review JSON are written to temporary paths and are not retained as product
goldens.

## Reproduce the native capture

Fetch the exact tags recorded in `source.json`, then compile from this
directory, substituting the local source paths:

```powershell
g++ -std=c++17 -O2 `
  -I C:/path/to/eigen `
  -I C:/path/to/Sophus `
  capture_native.cpp -o capture_sophus.exe
$capture = Join-Path $env:TEMP 'quatops-sophus-1.24.6-discovery.json'
./capture_sophus.exe capture_inputs.tsv > $capture
python verify_scipy.py --capture $capture
```

The initial capture ran with GCC 15.2.0 on Windows x86-64; its build and input
identity are recorded in `manifest.json`. Re-run the native executable before
using the SciPy comparison for a new build. The SciPy check is separate
derived-reference evidence and does not replace native execution.

The official [Sophus release](https://github.com/strasdat/Sophus/releases/tag/1.24.6)
and versioned [`SO3` implementation](https://github.com/strasdat/Sophus/blob/1.24.6/sophus/so3.hpp)
are the API sources for this pin.
