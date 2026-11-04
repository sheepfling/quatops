# Sophus 1.24.6 SO(3) discovery

The reviewed native capture covers `SO3f` and `SO3d` from Sophus 1.24.6,
compiled against Eigen 5.0.1 with GCC 15.2.0 on Windows x86-64. The assigned
right-handed probe frame is +X right, +Y forward, +Z up; Sophus does not assign
that frame universally.

[`witnesses/float32.json`](witnesses/float32.json) and
[`witnesses/float64.json`](witnesses/float64.json) each retain eight
quaternion-construction cases, six `exp`/`log` cases, both orders of one
noncommuting product, the inverse, and both perturbation sides. They keep
native quaternion outputs and basis-vector images. Table columns are declared
once; duplicate matrices and quaternion views are omitted. Full capture output
is regenerated temporarily and is not checked in.

Both scalar profiles normalize non-unit `Eigen::Quaternion` construction,
take constructor arguments in WXYZ order, expose XYZW through `SO3::params()`,
preserve opposite signs for antipodes with equivalent rotation actions, and
use a principal rotation-vector branch for the selected exact-π and
above-π cases. The selected noncommuting products show the native
composition order; left and right perturbation results are distinct.

Separate SciPy 1.18.1 `Rotation` checks passed: float64 at `2e-14` tolerance
with maximum error `1.78e-15`, and float32 at `2e-6` tolerance with maximum
error `9.5e-7`. These are formula checks, not QuatOps execution.

## Reproduce

Fetch the Sophus and Eigen commits in `source.json`, then compile with a C++17
compiler:

```powershell
g++ -std=c++17 -O2 `
  -I C:/path/to/eigen `
  -I C:/path/to/Sophus `
  capture_native.cpp -o capture_sophus.exe
$env:PYTHONPATH = (Resolve-Path ..\..\..\..).Path
foreach ($profile in @('float32', 'float64')) {
  $capture = Join-Path $env:TEMP "quatops-sophus-$profile.json"
  $captureCopy = "import pathlib, subprocess, sys; " + `
    "pathlib.Path(sys.argv[1]).write_bytes(subprocess.check_output(sys.argv[2:]))"
  python -c $captureCopy `
    $capture ./capture_sophus.exe capture_inputs.tsv $profile
  uv run --no-project --python 3.12 --with-requirements requirements-reference.txt `
    python verify_scipy.py --capture $capture
  uv run --no-project --python 3.12 --with-requirements requirements-reference.txt `
    python retain_witnesses.py --capture $capture --output "witnesses/$profile.json"
}
```

The manifest records profile-specific native capture hashes, selected witness
hashes, source and dependency identities, and SciPy tolerances. Jacobians and
tangent derivatives are outside this supporting discovery; no named QuatOps
converter is claimed.

Primary API references are the [Sophus 1.24.6 release](https://github.com/strasdat/Sophus/releases/tag/1.24.6),
its pinned [`SO3` implementation](https://github.com/strasdat/Sophus/blob/1.24.6/sophus/so3.hpp),
and [Eigen 5.0.1's Quaternion API](https://libeigen.gitlab.io/eigen/docs-5.0/classEigen_1_1Quaternion.html).
