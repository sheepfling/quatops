# Sophus 1.24.6 SO(3) discovery

The reviewed native capture covers `SO3f` and `SO3d` from Sophus 1.24.6,
compiled against Eigen 5.0.1 with GCC 15.2.0 on Windows x86-64. The assigned
right-handed probe frame is +X right, +Y forward, +Z up; Sophus does not assign
that frame universally.

The experiment-owned [`semantic_contract.json`](semantic_contract.json)
declares the representation facts, that fixture-only probe frame, both scalar
profiles, and their SciPy formula tolerances. The reducer and reference checker
load it through strict Pydantic models and verify its manifest SHA-256. This
remains supporting SO(3) evidence, not an accepted application-frame converter.
The native exporter now emits selected rows directly, and the Python validator
checks them against the typed witness schema and pinned inputs. The historical
native source and capture hashes remain separate from the direct replay run.
The direct runs reproduced the retained constructor values, including signed
zero, for both scalar profiles. Product observations were newly captured using
two of those constructors; they are not the old product witnesses. The older
raw run also emitted temporary matrices and duplicate coefficient views.

The content-addressed float32 and float64 witnesses referenced by the
[version manifest](manifest.json) each retain eight
quaternion-construction cases and both orders of a noncommuting product using
two of those rotations. They keep native quaternion
outputs and basis-vector images. Table columns are declared once. Direct
capture output is generated temporarily and is not checked in.
Witness schema `4.0` names the retained table layout, not a Sophus version.
Another version needs its own contract, native capture and provenance before
claiming coverage. It may reference the same numeric file only after its own
native output matches exactly, including signed zero.
The validators accept an explicit `--package-dir` for another version's recipe;
they check that version's source, contract and input pin rather than falling
back to this directory.

Both scalar profiles normalize non-unit `Eigen::Quaternion` construction,
take constructor arguments in WXYZ order, expose XYZW through `SO3::params()`,
preserve opposite signs for antipodes with equivalent rotation actions.
The selected noncommuting products show the native composition order.
Logarithms, inverse, perturbations, and tangent-branch behavior are outside
the retained quaternion-boundary evidence.

Separate SciPy 1.18.1 `Rotation` checks passed: float64 at `2e-14` tolerance
with maximum error `1.78e-15`, and float32 at `2e-6` tolerance with maximum
error `9.5e-7`. Those historical results cover the full original capture.
The current reference checker reduces a fresh capture to the selected native
witnesses first, then checks their constructor, actions, and product cases.
These are formula checks, not QuatOps execution.

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
  $run = Start-Process -FilePath "$PWD/capture_sophus.exe" `
    -ArgumentList @('capture_inputs.tsv', $profile) -RedirectStandardOutput $capture `
    -Wait -PassThru -WindowStyle Hidden
  if ($run.ExitCode -ne 0) { throw "Sophus native capture failed: $($run.ExitCode)" }
  uv run --no-project --python 3.12 --with-requirements requirements-reference.txt `
    python verify_scipy.py --capture $capture
  uv run --no-project --python 3.12 --with-requirements requirements-reference.txt `
    python retain_witnesses.py --capture $capture `
      --output (Join-Path $env:TEMP "quatops-sophus-$profile-selected.json")
}
```

The selected constructor values match the historical fixture; the product
inputs and outputs were newly captured. Keep a regenerated capture separate
from shared witnesses unless deliberately recording a new version's run.

The manifest records profile-specific native capture hashes, selected witness
hashes, source and dependency identities, and SciPy tolerances. Run attestations
are recorded per version and are not independent proof of execution when raw
captures are unavailable. To check a fresh native capture before reusing an
existing numeric payload, run `retain_witnesses.py` with `--capture` and
`--compare` paths. It requires exact equality, including signed zero; the
comparison does not itself establish the source version or prove execution.
Hashes and revision for the original full capture are retained separately. Jacobians and
tangent derivatives are outside this supporting discovery. The experimental
`sophus.so3-params` profile remains supporting-only, not an accepted
application-frame converter.

Primary API references are the [Sophus 1.24.6 release](https://github.com/strasdat/Sophus/releases/tag/1.24.6),
its pinned [`SO3` implementation](https://github.com/strasdat/Sophus/blob/1.24.6/sophus/so3.hpp),
and [Eigen 5.0.1's Quaternion API](https://libeigen.gitlab.io/eigen/docs-5.0/classEigen_1_1Quaternion.html).
