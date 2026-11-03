# Sophus 1.24.6 SO(3) discovery

This first native discovery probe exercises the fixed release tag with Eigen
5.0.1 and a C++17 compiler. It captures quaternion construction and storage,
native vector action, matrix output, `SO3::exp`/`log`, inverse, both orders of
noncommuting composition, and left/right perturbation products. The fixed input
set is in `capture_inputs.tsv`. For these probes, the assigned right-handed
frame is +X right, +Y forward, +Z up; this does not claim Sophus assigns a
universal world frame. The harness does not include QuatOps.

The captured result is discovery evidence, not yet an accepted converter or
complete Sophus contract. In particular, the exact tangent perturbation side,
near-π logarithm branch, and behavior across supported scalar profiles remain
to be reviewed. The capture currently covers `SO3d` only.

## Reproduce the native capture

Fetch the exact tags recorded in `source.json`, then compile from this
directory, substituting the local source paths:

```powershell
g++ -std=c++17 -O2 `
  -I C:/path/to/eigen `
  -I C:/path/to/Sophus `
  capture_native.cpp -o capture_sophus.exe
./capture_sophus.exe capture_inputs.tsv > discovery.json
```

The current capture ran with GCC 15.2.0 on Windows x86-64. The output records
native `SO3d` values; SciPy comparisons are separate derived-reference
evidence and do not replace this native execution.

The official [Sophus release](https://github.com/strasdat/Sophus/releases/tag/1.24.6)
and versioned [`SO3` implementation](https://github.com/strasdat/Sophus/blob/1.24.6/sophus/so3.hpp)
are the API sources for this pin.
